#!/usr/bin/env python3
"""Tests of versioned page publishing (lib/serve/versions.py, lib/serve/coalesce.py, the sub-LST switch in
lib/serve/sublst_build.py): data-free, plain checks, runnable with python, under a minute.

    NUMPY_MADVISE_HUGEPAGE=0 /opt/devel/pipeline/envs/py38_orca_nkosogor/bin/python3 tests/test_publish_pointer.py

1. The pointer switch: while a new version is built, readers of U see the old version (every read a whole file of the old
   version, never a file of the new one); after the switch every read is the new version; readers that read throughout the
   switch never see a missing or partial file. The served run's realpath keeps the run's name.
2. Adoption: the first switch of a legacy directory (and of a legacy link to a directory elsewhere) moves it into the versions
   root as an adopted version, its content intact; the previous version is that one.
3. Completion: no switch for a version that is not sealed, or whose run changed after the seal (a file appended, added or
   removed): Refusal, U unchanged. The seal refuses a missing or empty required path, a failed check, an empty run and a
   second seal. Review state (the judgement journal) appended to a served version leaves it verifiable.
4. place / unplace / an abandoned placed build found again.
5. Holds and retention: the dry run deletes nothing; never the pointer's version, a held one, one being built or one of the
   newest kept; a plan that went stale refuses with nothing deleted; the applied plan deletes exactly its versions; the
   retire_plan and run_version content tests.
6. Coalescing: five requests while a build runs give exactly one follow-up build (two builds in all); a failed build still
   gets its follow-up; a request with no running builder builds at once.
7. The coalesce command line: exit 0 for a build, 3 when coalesced, the build's own exit code when it fails.
8. The sub-LST switch stage: an incomplete version (a page's queue missing) is not switched and goes back to the build
   directory; a complete one is sealed and switched, the journal lines written on the old version between the carry and the
   switch are appended; an abandoned placed build is moved back to the build directory by the next serve; a seal that
   passed and a switch that then refused moves the build back (and so does the next serve after a kill between them).
9. No judgement lost after a switch (PR 155 review, finding 1): a page process that resolved --run to the old version's
   realpath writes after the switch, and its line is in the served version; the two-port server answers a POST made after
   the switch with a page built on the served version (its version in the line), never the page built before; a POST in
   flight across the switch (holding the run's journal lock) is written before the switch and caught up into the new
   version; the two-port lock file and versions.journal_locks name the same file for every version and for the legacy run.
10. The review's other findings: the page index rebuilt in place leaves a sealed version verifiable; adoption refuses an empty
   run, a failed rename and no free name with U intact; an adopted version gets its manifest and can be switched back to,
   and one without a manifest is not counted among the kept newest; leases (another host's holder alive until its lease
   expires; holds and builders renewed); one walk per publish and none in the content test of a unit's own seal; an empty
   stage selection refuses; the coalescer's status probe never takes the lock.
11. The coalescing builder wired in: serve --tier sublst with build or start coalesces into a running build (a no-input
   unit, nothing built), builds with one follow-up for a request made during its build (the plan made again), re-raises a
   build's refusal once the slot is free, and runs stages without build or start directly; the sub-LST placement requests a
   build of the cumulative run beside the runs it placed.
12. The second review round: no pointer log means never served (the move back works at the first switch) and a failed move
   back never masks the refusal; the seed carry/copy reads the journals under the journal locks and the design-B maps are
   loaded before them (seed and catch-up); a catch-up that fails after the switch is owed (builds and switches refuse, the
   old version kept) until the next serve completes it; a label form rendered by another version, or without its token, is
   answered 409 and nothing is recorded; the legacy LOCK_DIR lock is taken too when it exists; lock files group-writable,
   a lock that cannot be opened a refusal naming its owner; another host's record without a lease is alive, a lease dead
   only after twice its length, renewals fsynced outside the renewer lock; adoption rolled back on a failure before U
   points at the new version; the content test walks the run against MANIFEST.tsv.
13. Content: MANIFEST.tsv holds the sha256 of every file not a binary and of the binaries' stable sample (2%, at least 20,
   by the path's hash); the content test re-hashes those and every file changed in size or time - a changed content with
   size and time kept fails, a touched file passes, a changed binary without a hash fails; 'all' hashes every binary; a
   file written during the seal refuses it.
"""
import contextlib
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import types

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
TMP = '/tmp/exopipe3_tests_publish_pointer'

from exopipe.phase3.config import CFG                      # noqa: E402
from exopipe.phase3.core.record import Refusal             # noqa: E402
from exopipe.phase3.lib.serve import coalesce, versions    # noqa: E402

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


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(text)


def read(path):
    with open(path) as f:
        return f.read()


def placer_gone(v):
    """v's building process is gone: it no longer renews its lease (renew_stop first, so the renewal of this process never
    rewrites the record after it), and its record names a process start that is not this process's."""
    versions.renew_stop(v.path('BUILDING.json'))
    versions._atomic_json(v.path('BUILDING.json'), dict(v.builder(), start=-1))


def new_version(U, tag, files=('a.txt', 'sub/b.txt')):
    v = versions.begin(U, note=f'test {tag}')
    for fn in files:
        write(os.path.join(v.run, fn), tag * 1000)
    return v


# ---- 1. the pointer switch ----------------------------------------------------------------------------------------------
def test_switch():
    d = f'{TMP}/switch'
    U = f'{d}/Run_X'
    os.makedirs(d)
    v1 = new_version(U, 'A')
    sw = versions.publish(U, v1)
    check('first publish: U is a pointer to version 1, nothing adopted', os.path.islink(U) and versions.served(U).vid == v1.vid
          and sw['previous'] is None and sw['adopted'] is None, (os.readlink(U) if os.path.islink(U) else None, sw))
    check('the pointer is relative and the served realpath keeps the run name',
          not os.path.isabs(os.readlink(U)) and os.path.basename(os.path.realpath(U)) == 'Run_X', os.readlink(U))
    v2 = new_version(U, 'B', files=('a.txt',))
    check('a reader during the build sees the old version (its files, not the new version\'s)',
          read(f'{U}/a.txt') == 'A' * 1000 and read(f'{U}/sub/b.txt') == 'A' * 1000, read(f'{U}/a.txt')[:3])
    write(os.path.join(v2.run, 'sub/b.txt'), 'B' * 1000)
    seen, bad, stop = set(), [], threading.Event()

    def reader():
        while not stop.is_set():
            try:
                t = read(f'{U}/a.txt')
            except OSError as e:
                bad.append(repr(e))
                continue
            if t not in ('A' * 1000, 'B' * 1000):
                bad.append(f'partial {len(t)} bytes')
            seen.add(t[:1])
    th = [threading.Thread(target=reader) for _ in range(4)]
    for t in th:
        t.start()
    time.sleep(0.05)
    before = read(f'{U}/a.txt')
    versions.publish(U, v2)
    time.sleep(0.05)
    stop.set()
    for t in th:
        t.join()
    check('readers throughout the switch: every read a whole file of one version, none missing', not bad, bad[:3])
    check('before the switch the old version, after it the new one', before == 'A' * 1000 and read(f'{U}/a.txt') == 'B' * 1000
          and read(f'{U}/sub/b.txt') == 'B' * 1000 and seen == {'A', 'B'}, (before[:1], seen))
    log = [json.loads(x) for x in open(f'{versions.root_of(U)}/POINTER_LOG.jsonl')]
    check('the pointer log records both switches', [(r['from'], r['to']) for r in log] == [(None, v1.vid), (v1.vid, v2.vid)], log)
    check('no temporary pointer is left', not [n for n in os.listdir(d) if '.pointer.' in n], os.listdir(d))
    check('a switch to the version already served refuses', 'already names' in (refuses(versions.switch, U, v2) or ''))
    other = versions.begin(f'{d}/Run_Y')
    write(f'{other.run}/a.txt', 'x')
    versions.seal(other)
    check('a version of another run refuses', 'not a version of' in (refuses(versions.switch, U, other) or ''))
    with versions.hold(U) as R:
        check('hold: resolves the pointer once to the served version\'s directory', R == v2.run, R)
        check('hold: the held version lists its holder', len(v2.holds()) == 1 and v2.holds()[0][1]['pid'] == os.getpid())
    check('hold: released at the end of the block', v2.holds() == [], v2.holds())


# ---- 2. adoption ----------------------------------------------------------------------------------------------------------
def test_adopt():
    d = f'{TMP}/adopt'
    U = f'{d}/Run_SLSTB'
    write(f'{U}/candidates/V/x.txt', 'legacy')
    write(f'{U}/judgements_V/judgements.jsonl', '{"a": 1}\n')
    v = new_version(U, 'N')
    check('a legacy run: its versions root lies beside it', versions.root_of(U) == os.path.realpath(f'{d}/Run_SLSTB.versions'),
          versions.root_of(U))
    sw = versions.publish(U, v)
    ad = sw['adopted']
    check('the first switch adopts the legacy directory as the previous version, then gives it its manifest',
          ad is not None and sw['previous'] is ad and ad.vid.endswith('-adopted')
          and read(f'{ad.run}/candidates/V/x.txt') == 'legacy' and (ad.record() or {}).get('state') == 'sealed'
          and (ad.record() or {}).get('adopted') is True and versions.verify(ad)[0], (sw, ad and ad.record()))
    check('after adoption U is the pointer to the new version', os.path.islink(U) and read(f'{U}/a.txt') == 'N' * 1000)
    from exopipe.phase3.lib.serve import tests_versions as TV
    r = TV.adopted_sealed(ad.path('VERSION.json'))
    check('run_version_adopted content test on the adopted version', r[0], r)
    sb = versions.switch(U, ad)
    check('the adopted version (with its manifest) can be switched back to', read(f'{U}/candidates/V/x.txt') == 'legacy'
          and sb['previous'].vid == v.vid, sb)
    versions.switch(U, v)
    # a legacy LINK to a directory elsewhere (Run_P3S5 -> /fast/...): the versions root lies beside the real directory
    real = f'{TMP}/adopt_fast/Run_P3S5'
    write(f'{real}/x.txt', 'deep')
    U2 = f'{d}/Run_P3S5'
    os.symlink(real, U2)
    check('a legacy link: its versions root lies beside the real directory',
          versions.root_of(U2) == os.path.realpath(f'{TMP}/adopt_fast/Run_P3S5.versions'), versions.root_of(U2))
    v2 = new_version(U2, 'D')
    sw2 = versions.publish(U2, v2)
    check('a legacy link is adopted (the real directory moved into the versions root) and replaced by an absolute pointer',
          sw2['adopted'] is not None and (sw2['adopted'].record() or {}).get('pointer_was_link') == real
          and not os.path.exists(real) and os.path.isabs(os.readlink(U2)) and read(f'{U2}/a.txt') == 'D' * 1000
          and read(f'{sw2["adopted"].run}/x.txt') == 'deep', (sw2, os.readlink(U2)))


# ---- 3. completion -------------------------------------------------------------------------------------------------------
def test_completion():
    d = f'{TMP}/complete'
    U = f'{d}/Run_X'
    os.makedirs(d)
    v1 = new_version(U, 'A')
    versions.publish(U, v1)
    v2 = new_version(U, 'B')
    why = refuses(versions.switch, U, v2)
    check('no switch for a version not sealed (a build not finished)', why and 'not sealed' in why and
          read(f'{U}/a.txt') == 'A' * 1000, why)
    check('seal refuses a missing required path', 'missing' in (refuses(versions.seal, v2, required=['queue.txt']) or ''))
    write(f'{v2.run}/queue.txt', '')
    check('seal refuses an empty required file', 'empty' in (refuses(versions.seal, v2, required=['queue.txt']) or ''))
    check('seal refuses a required path outside the run', 'outside' in (refuses(versions.seal, v2, required=['../x']) or ''))
    why = refuses(versions.seal, v2, checks=[('index', lambda r: (False, 'stale'))])
    check('seal refuses a failed check', why and 'index failed (stale)' in why, why)
    check('a refused seal writes no record', v2.record() is None and not os.path.exists(v2.path('MANIFEST.tsv')))
    write(f'{v2.run}/queue.txt', 'J1\n')
    rec = versions.seal(v2, required=['queue.txt'], checks=[('index', lambda r: (True, 'current'))])
    check('seal: the record counts the files and lists the check', rec['n_files'] == 3 and rec['checks'][0]['passed']
          and rec['required'] == ['queue.txt'], rec)
    check('a second seal refuses', 'already sealed' in (refuses(versions.seal, v2) or ''))
    for what, act, undo in (
            ('a file appended', lambda: open(f'{v2.run}/a.txt', 'a').write('x'), lambda: write(f'{v2.run}/a.txt', 'B' * 1000)),
            ('a file added', lambda: write(f'{v2.run}/new.txt', 'x'), lambda: os.remove(f'{v2.run}/new.txt')),
            ('a file removed', lambda: os.rename(f'{v2.run}/queue.txt', f'{d}/q'), lambda: os.rename(f'{d}/q', f'{v2.run}/queue.txt'))):
        st = os.stat(f'{v2.run}/a.txt')
        act()
        why = refuses(versions.switch, U, v2)
        check(f'no switch for a sealed version whose run changed ({what})', why and 'differs from its manifest' in why
              and read(f'{U}/a.txt') == 'A' * 1000, why)
        undo()
        os.utime(f'{v2.run}/a.txt', ns=(st.st_atime_ns, st.st_mtime_ns))
    write(v2.path('MANIFEST.tsv'), read(v2.path('MANIFEST.tsv')).replace('queue.txt', 'queue.tx_'))
    why = refuses(versions.switch, U, v2)
    check('no switch when the manifest does not match its record (sha256)', why and 'sha256' in why, why)
    v3 = new_version(U, 'C')
    empty = versions.begin(U)
    check('seal refuses an empty run', 'holds no file' in (refuses(versions.seal, empty) or ''))
    write(f'{v3.run}/judgements_V/judgements.jsonl', '{"a": 1}\n')
    write(f'{v3.run}/incremental.log', 'x\n')
    versions.publish(U, v3)
    open(f'{U}/judgements_V/judgements.jsonl', 'a').write('{"b": 2}\n')
    open(f'{U}/incremental.log', 'a').write('y\n')
    ok, detail = versions.verify(v3)
    check('review state appended to the served version (journal, incremental.log) leaves it verifiable', ok, detail)
    check('the record lists the mutable paths left out of the manifest',
          v3.record()['mutable'] == list(CFG.serve.RUN_VERSIONS_MUTABLE), v3.record()['mutable'])


# ---- 4. place / unplace ----------------------------------------------------------------------------------------------------
def test_place():
    d = f'{TMP}/place'
    U, B = f'{d}/Run_SLSTB', f'{d}/Run_SLSTB.build'
    write(f'{B}/candidates/x.txt', 'b')
    v = versions.place(U, B)
    check('place: the build directory becomes the run of a new unsealed version (one rename)', not os.path.exists(B)
          and read(f'{v.run}/candidates/x.txt') == 'b' and v.record() is None
          and v.builder()['placed_from'] == os.path.abspath(B), v.builder())
    check('a live placer is not an abandoned build', versions.placed_abandoned(U, B) is None)
    versions.unplace(v, B)
    check('unplace: the run back at the build directory, the version gone', read(f'{B}/candidates/x.txt') == 'b'
          and not os.path.exists(v.dir), os.listdir(versions.root_of(U)))
    v = versions.place(U, B)
    placer_gone(v)
    check('a placed build whose placer is gone is found', versions.placed_abandoned(U, B).vid == v.vid)
    versions.seal(v)
    check('a sealed version is not moved back without sealed=True; one never switched to (no pointer log) whose placer is '
          'gone is found as abandoned', 'is sealed' in (refuses(versions.unplace, v, B) or '')
          and (versions.placed_abandoned(U, B) or types.SimpleNamespace(vid=None)).vid == v.vid)
    check('place refuses a missing build directory', 'no build directory' in (refuses(versions.place, U, B) or ''))


# ---- 5. holds and retention ---------------------------------------------------------------------------------------------------
def test_retention():
    from exopipe.phase3.lib.serve import tests_versions as TV
    d = f'{TMP}/retain'
    U = f'{d}/Run_X'
    os.makedirs(d)
    vs = []
    for i in range(6):
        v = new_version(U, 'ABCDEF'[i])
        versions.publish(U, v)
        vs.append(v)
        time.sleep(1.05)                       # one version name per second
    old_served = vs[1]
    building = new_version(U, 'G')             # being built by this process (live)
    dead = new_version(U, 'H')                 # an abandoned build
    placer_gone(dead)
    held = vs[0]
    os.makedirs(held.path('holds'))
    hold_file = held.path('holds/x.json')
    versions._atomic_json(hold_file, versions.me('a build reading it'))
    other = f'{d}/Run_X_old_link'
    os.symlink(os.path.relpath(old_served.run, d), other)          # another link naming a version
    rows = {v.vid: (act, why) for v, act, why in versions.retention(U, keep=2)}
    want = {vs[0].vid: 'keep', vs[1].vid: 'keep', vs[2].vid: 'retire', vs[3].vid: 'retire', vs[4].vid: 'keep',
            vs[5].vid: 'keep', building.vid: 'keep', dead.vid: 'retire'}
    check('retention (keep 2): the pointer\'s, the newest 2, a held one, a linked one and one being built kept; older ones and '
          'an abandoned build retired', {k: a for k, (a, _) in rows.items()} == want, rows)
    n0 = sorted(os.listdir(versions.root_of(U)))
    res = versions.retire(U, keep=2)
    check('the dry run deletes nothing and writes the plan', set(os.listdir(versions.root_of(U))) == set(n0) | {'retire_plans'}
          and os.path.isfile(res['plan']), os.listdir(versions.root_of(U)))
    check('the plan lists the versions to retire with their sizes', sorted(r['version'] for r in res['retire'])
          == sorted([vs[2].vid, vs[3].vid, dead.vid]) and all(r['bytes'] > 0 for r in res['retire']), res['retire'])
    r = TV.plan_consistent(res['plan'])
    check('retire_plan content test (dry run)', r[0], r)
    check('a plan for another keep refuses', 'not for' in (refuses(versions.retire, U, keep=3, plan=res['plan']) or ''))
    os.makedirs(vs[2].path('holds'))
    versions._atomic_json(vs[2].path('holds/late.json'), versions.me('a build that started after the dry run'))
    why = refuses(versions.retire, U, keep=2, plan=res['plan'])
    check('a plan that went stale (a planned version is now held) refuses with nothing deleted', why and 'no longer' in why
          and all(os.path.isdir(v.dir) for v in vs + [dead]), why)
    os.remove(vs[2].path('holds/late.json'))
    out = versions.retire(U, keep=2, plan=res['plan'])
    left = {v.vid for v in versions.all_versions(U)}
    check('the applied plan deletes exactly its versions', sorted(out['retired']) == sorted([vs[2].vid, vs[3].vid, dead.vid])
          and left == {vs[0].vid, vs[1].vid, vs[4].vid, vs[5].vid, building.vid} and out['bytes'] > 0, (out, left))
    check('the pointer still serves its version', read(f'{U}/a.txt') == 'F' * 1000)
    r = TV.plan_consistent(res['plan'], applied=True)
    check('retire_plan content test (applied: gone and logged)', r[0], r)
    r = TV.sealed_complete(vs[5].path('VERSION.json'))
    r2 = TV.switched_to(vs[5].path('VERSION.json'), pointer=U)
    check('run_version content tests on the served version', r[0] and r2[0], (r, r2))
    write(f'{vs[4].run}/a.txt', 'changed')
    r = TV.sealed_complete(vs[4].path('VERSION.json'))
    check('run_version content test fails on a version changed after its seal', not r[0], r)
    os.remove(hold_file)
    os.remove(other)
    check('retention refuses keep 0 (the newest version is always kept)', 'at least' in (refuses(versions.retention, U, keep=0) or ''))


# ---- 6. coalescing -----------------------------------------------------------------------------------------------------------
def test_coalesce():
    slot = coalesce.Slot(f'{TMP}/coalesce/slot')
    started, release, builds = threading.Event(), threading.Event(), []

    def build():
        builds.append(time.time())
        if len(builds) == 1:
            started.set()
            release.wait(10)
        return 0
    res1 = {}
    t1 = threading.Thread(target=lambda: res1.update(slot.submit(build, 'first')))
    t1.start()
    started.wait(10)
    check('while a build runs the slot names its builder', (slot.builder() or {}).get('pid') == os.getpid(), slot.builder())
    outs = []
    th = [threading.Thread(target=lambda i=i: outs.append(slot.submit(build, f'request {i}'))) for i in range(5)]
    for t in th:
        t.start()
    for t in th:
        t.join(10)
    check('five requests during a build are coalesced (none builds itself)',
          len(outs) == 5 and all(o['status'] == 'coalesced' for o in outs) and len(builds) == 1, (outs, len(builds)))
    check('the five requests leave one flag', slot.pending())
    release.set()
    t1.join(10)
    check('five requests during a build give exactly one follow-up build (two builds in all)',
          len(builds) == 2 and res1['status'] == 'built' and len(res1['builds']) == 2 and not slot.pending(), (res1, len(builds)))
    j = [json.loads(x) for x in open(slot.journal)]
    check('the journal records 6 requests and 2 builds', sum(r['event'] == 'request' for r in j) == 6
          and sum(r['event'] == 'build-end' for r in j) == 2, [r['event'] for r in j])
    check('the builder record is removed when the builder ends', slot.builder() is None
          and not os.path.exists(slot.builder_file))
    res = slot.submit(lambda: 0, 'idle')
    check('a request with no running builder builds at once', res['status'] == 'built' and len(res['builds']) == 1, res)
    # a failed build still gets its follow-up
    slot2 = coalesce.Slot(f'{TMP}/coalesce/slot2')
    started2, release2, n2 = threading.Event(), threading.Event(), []

    def failing():
        n2.append(1)
        if len(n2) == 1:
            started2.set()
            release2.wait(10)
            raise RuntimeError('stand-in failure')
        return 0
    r2 = {}
    t2 = threading.Thread(target=lambda: r2.update(slot2.submit(failing)))
    t2.start()
    started2.wait(10)
    slot2.request('during the failing build')
    release2.set()
    t2.join(10)
    check('a failed build is recorded and a request made during it still gets its follow-up',
          [b['rc'] for b in r2['builds']] == [1, 0] and 'stand-in failure' in (r2['builds'][0]['error'] or '')
          and coalesce.exit_code(r2) == 1, r2)


# ---- 7. the command line ---------------------------------------------------------------------------------------------------
def test_cli():
    py = sys.executable
    env = dict(os.environ, PYTHONPATH=REPO)
    sd = f'{TMP}/coalesce/cli'
    p = subprocess.run([py, '-m', 'exopipe.phase3.lib.serve.coalesce', 'run', '--slot', sd, '--who', 't', '--', py, '-c',
                        'print("built")'], env=env, capture_output=True, text=True, timeout=120)
    out = json.loads(p.stdout.strip().splitlines()[-1])
    check('CLI run: exit 0 and one build', p.returncode == 0 and out['status'] == 'built' and len(out['builds']) == 1
          and 'built' in p.stdout, (p.returncode, p.stdout, p.stderr[-300:]))
    p = subprocess.run([py, '-m', 'exopipe.phase3.lib.serve.coalesce', 'run', '--slot', sd, '--', py, '-c',
                        'import sys; sys.exit(5)'], env=env, capture_output=True, text=True, timeout=120)
    check('CLI run: a failed build exits with its own code', p.returncode == 5, (p.returncode, p.stdout))
    slot = coalesce.Slot(sd)
    fd = slot._try_lock()                       # a builder holding the slot
    try:
        p = subprocess.run([py, '-m', 'exopipe.phase3.lib.serve.coalesce', 'run', '--slot', sd, '--', py, '-c', 'pass'],
                           env=env, capture_output=True, text=True, timeout=120)
        check('CLI run while a builder holds the slot: exit 3 (coalesced), the request left for it',
              p.returncode == coalesce.EXIT_COALESCED and json.loads(p.stdout)['status'] == 'coalesced' and slot.pending(),
              (p.returncode, p.stdout))
    finally:
        os.close(fd)
    p = subprocess.run([py, '-m', 'exopipe.phase3.lib.serve.coalesce', 'status', '--slot', sd], env=env, capture_output=True,
                       text=True, timeout=120)
    st = json.loads(p.stdout)
    check('CLI status: pending and no builder', p.returncode == 0 and st['pending'] and st['builder'] is None, st)


# ---- 8. the sub-LST switch stage ---------------------------------------------------------------------------------------------
def test_sublst_switch():
    from exopipe.phase3.core.record import Run
    from exopipe.phase3.lib.serve import flow, sublst_build
    d = f'{TMP}/sublst'
    U = f'{d}/Run_SLSTB'
    B = U + CFG.serve.SUBLST_BUILD_SUFFIX
    out = f'{d}/out'
    rels = {'queue': 'judgements/queue_V.txt', 'catalog': 'source_catalogs_V'}
    page = types.SimpleNamespace(key='V', at=lambda f, root=None: os.path.join(root, rels[f]) if f in rels else None)
    plan = types.SimpleNamespace(U=U, U_src=U, run_name='Run_SLSTB', parity=False, pages=[page], tier='sublst')
    J = CFG.serve.SUBLST_JOURNALS[0]
    write(f'{U}/{J}/judgements.jsonl', '{"a": 1}\n')                       # the served legacy run with a journal
    write(f'{U}/candidates/V/x.txt', 'old')
    write(f'{B}/candidates/V/x.txt', 'new')                                 # the build, missing its V queue
    write(f'{B}/source_catalogs_V/23MHz_sources_survivors.csv', 'source_id\nJ1\n')
    write(f'{B}/{J}/judgements.jsonl', '{"a": 1}\n')                       # the journal copied at the carry
    write(f'{U}/{J}/judgements.jsonl', '{"a": 1}\n{"b": 2}\n')             # a verdict between the carry and the switch
    orig = (flow.build_page_index, flow._index_current)
    flow.build_page_index = lambda u, R, publisher=False: None
    flow._index_current = lambda R: ''
    try:
        run = Run('serve', ['test'], out, CFG, config_sections=None)
        state = flow.new_state()
        state.update(version=versions.place(U, B), carry_src=U, design_b=False, marks={})
        u = sublst_build.switch_stage(run, plan, state)
        check('an incomplete version (the V queue missing) is not switched: the switch unit refused, the build back at B, U '
              'still the legacy run', u.status == 'refused' and 'missing' in (u.reason or '') and os.path.isdir(B)
              and not os.path.islink(U) and read(f'{U}/candidates/V/x.txt') == 'old'
              and not os.path.exists(state['version'].dir), (u.status, u.reason))
        write(f'{B}/judgements/queue_V.txt', 'J1\n')
        flow._index_current = lambda R: 'records_current: stale'
        state['version'] = versions.place(U, B)
        u = sublst_build.switch_stage(run, plan, state)
        check('a stale page index refuses the seal: not switched, the build back at B',
              u.status == 'refused' and 'page_index_current' in (u.reason or '') and os.path.isdir(B) and not os.path.islink(U),
              (u.status, u.reason))
        flow._index_current = lambda R: ''
        state['version'] = v = versions.place(U, B)
        u = sublst_build.switch_stage(run, plan, state)
        check('a complete version is sealed and switched to; the legacy run adopted as the previous version',
              u.status == 'done' and os.path.islink(U) and read(f'{U}/candidates/V/x.txt') == 'new'
              and u.meta.get('version') == v.vid and (u.meta.get('adopted') or '').endswith('-adopted'),
              (u.status, u.reason, u.meta))
        check('the journal line written on the old version between the carry and the switch is appended',
              read(f'{U}/{J}/judgements.jsonl') == '{"a": 1}\n{"b": 2}\n', read(f'{U}/{J}/judgements.jsonl'))
        check('the switch unit records the version (run_version) and the journal, their content tests passing',
              {o['class'] for o in u.outputs} >= {'run_version', 'judgement_journal'} and u.tests
              and all(t['passed'] for t in u.tests), [(t['name'], t['passed'], t['detail']) for t in u.tests if not t['passed']])
        # a swap killed between its place and its switch: the next serve moves the build back
        write(f'{B}/candidates/V/x.txt', 'newer')
        w = versions.place(U, B)
        placer_gone(w)
        got = sublst_build.recover_placed(plan)
        check('an abandoned placed build is moved back to the build directory by the next serve',
              got is not None and got.vid == w.vid and read(f'{B}/candidates/V/x.txt') == 'newer'
              and not os.path.exists(w.dir), got)
        # finding 3: the seal passes, then the switch refuses - the build goes back to B (it was stranded as a sealed version)
        served_before = os.readlink(U)
        write(f'{B}/judgements/queue_V.txt', 'J1\n')
        write(f'{B}/source_catalogs_V/23MHz_sources_survivors.csv', 'source_id\nJ1\n')
        real_switch = versions.switch

        def refusing_switch(*a, **k):
            raise Refusal('stand-in: the switch refused after the seal')
        versions.switch = refusing_switch
        try:
            state['version'] = w2 = versions.place(U, B)
            u = sublst_build.switch_stage(run, plan, state)
        finally:
            versions.switch = real_switch
        check('a seal that passed and a switch that refused: the unit refused, the sealed build moved back to B (its record '
              'removed), U unchanged', u.status == 'refused' and 'stand-in' in (u.reason or '') and os.path.isdir(B)
              and read(f'{B}/candidates/V/x.txt') == 'newer' and not os.path.exists(w2.dir)
              and not os.path.exists(f'{B}/VERSION.json') and os.readlink(U) == served_before, (u.status, u.reason))
        # ... and a swap killed between its seal and its switch: the next serve moves the sealed, never served build back
        w3 = versions.place(U, B)
        versions.seal(w3)
        placer_gone(w3)
        got = sublst_build.recover_placed(plan)
        check('a sealed build never switched to, its swap gone, is moved back to the build directory by the next serve',
              got is not None and got.vid == w3.vid and os.path.isdir(B) and not os.path.exists(w3.dir), got)
        check('a served version is never moved back (unplace refuses it even with sealed=True)',
              'was served' in (refuses(versions.unplace, versions.served(U), f'{d}/elsewhere', sealed=True) or ''))
    finally:
        flow.build_page_index, flow._index_current = orig


# ---- 9. no judgement lost after a switch (finding 1) -------------------------------------------------------------------------
# the page code's path handling, as classify_transients.py does it: run_dir = os.path.realpath(--run) once at start (:5310),
# labels_dir = --labels-dir as given (:5313; the plan passes <U>/<journal dir>, plan.py:190-191), the journal opened at every
# write (append_journal, :2503-2507)
PAGE_PROC = r'''
import json, os, sys, time
run_dir = os.path.realpath(sys.argv[1])
labels_dir = sys.argv[2]
print(run_dir, flush=True)
while not os.path.exists(sys.argv[3]):
    time.sleep(0.02)
with open(os.path.join(labels_dir, 'judgements.jsonl'), 'a') as f:
    f.write(json.dumps({'source_id': 'J1', 'action': 'classify', 'page_run': run_dir}, sort_keys=True) + '\n')
    f.flush()
    os.fsync(f.fileno())
print('written', flush=True)
'''

# a review page for the two-port store whose label records the version its build resolved (--run realpath'd once)
PAGE_V = r'''
import argparse, json, os, urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
INSPECT = False
HUB_URL = 'http://localhost:8765/'

class App:
    def __init__(self, run_dir, labels):
        self.run_dir, self.labels = run_dir, labels
    def version(self):
        return os.path.basename(os.path.dirname(os.path.dirname(self.run_dir)))
    def handle_label(self, form):
        with open(os.path.join(self.labels, 'judgements.jsonl'), 'a') as f:
            f.write(json.dumps(dict({k: v[0] for k, v in form.items()}, page_version=os.path.basename(
                os.path.dirname(self.run_dir)))) + '\n')
        return '/source/' + urllib.parse.quote(form.get('source_id', [''])[0])

class Handler(BaseHTTPRequestHandler):
    app = None
    def do_GET(self):
        body = ('version=' + os.path.basename(os.path.dirname(self.app.run_dir)) + '<form method="POST" action="/label">'
                '<input name="source_id" value="J7"><button>fp</button></form>').encode()
        self.send_response(200); self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_POST(self):
        n = int(self.headers.get('Content-Length', 0))
        dest = self.app.handle_label(urllib.parse.parse_qs(self.rfile.read(n).decode()))
        self.send_response(303); self.send_header('Location', dest); self.end_headers()
    def log_message(self, *a):
        pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True); ap.add_argument('--labels-dir'); ap.add_argument('--title', default='')
    ap.add_argument('--port', type=int, default=1); ap.add_argument('--inspect', action='store_true')
    a = ap.parse_args()
    run_dir = os.path.realpath(a.run)
    Handler.app = App(run_dir, a.labels_dir or os.path.join(run_dir, 'judgements'))
    ThreadingHTTPServer(('127.0.0.1', a.port), Handler).serve_forever()
'''


def lines(path):
    with open(path) as f:
        return [json.loads(x) for x in f if x.strip()]


def next_version(U, tag, J):
    """A new sealed version of U whose journal J is a copy of the served one (the carry), with one data file."""
    v = new_version(U, tag)
    cur = os.path.join(U, J, 'judgements.jsonl')
    os.makedirs(os.path.join(v.run, J))
    if os.path.exists(cur):
        shutil.copy2(cur, os.path.join(v.run, J, 'judgements.jsonl'))
    return v, versions.seal(v)


def test_judgements_after_switch():
    from exopipe.phase3.lib.serve import journals
    from exopipe.phase3.lib.serve import two_port as TP
    J = CFG.serve.SUBLST_JOURNALS[0]
    d = f'{TMP}/judge'
    U = f'{d}/Run_SLSTB'
    os.makedirs(d)
    v1 = new_version(U, 'A')
    write(f'{v1.run}/{J}/judgements.jsonl', '{"source_id": "J0", "action": "classify"}\n')
    versions.publish(U, v1)
    # 9a. a page process holding the old realpath writes after the switch
    go = f'{d}/go'
    p = subprocess.Popen([sys.executable, '-c', PAGE_PROC, U, os.path.join(U, J), go], stdout=subprocess.PIPE, text=True)
    held_run = p.stdout.readline().strip()
    v2, rec2 = next_version(U, 'B', J)
    versions.switch(U, v2, record=rec2)
    write(go, 'go')
    done = p.stdout.readline().strip()
    p.wait(30)
    served = lines(os.path.join(U, J, 'judgements.jsonl'))
    check('a page process that resolved --run to the old version writes after the switch: the line is in the served '
          'version\'s journal, not the old one',
          held_run == v1.run and done == 'written' and os.path.realpath(U) == v2.run
          and served[-1].get('page_run') == v1.run and not any(r.get('page_run') for r in lines(f'{v1.run}/{J}/judgements.jsonl')),
          (held_run, done, served[-1:]))
    # 9b. the two-port store: a POST made after a switch is answered by a page built on the served version
    code = f'{d}/page_v.py'
    write(code, PAGE_V)
    reg = f'{d}/registry.json'
    with open(reg, 'w') as f:
        json.dump({'pages': [{'id': 'pv', 'kind': 'review', 'code': code, 'legacy_port': 47991,
                              'argv': ['--run', U, '--labels-dir', os.path.join(U, J), '--title', 'V']}]}, f)
    st = TP.Store(reg)
    r = TP.serve_request(st, False, 'GET', '/p/pv/', {})
    check('two-port: the page is built on the served version', r[0] == 200 and f'version={v2.vid}'.encode() in r[2], r[:1])
    tok_re = re.compile(TP.PAGE_TOKEN_FIELD.encode() + rb'" value="([0-9a-f]+)"')
    t2 = (tok_re.findall(r[2]) or [b''])[0].decode()
    check('two-port: the label form of a page in a version carries that version\'s token',
          t2 and t2 == TP.page_token(os.path.realpath(os.path.join(v2.run, J))) and r[2].count(b'name="_twoport_page"') == 1,
          r[2][:300])
    v3, rec3 = next_version(U, 'C', J)
    with versions.journal_locks(U, [J]):
        versions.switch(U, v3, record=rec3)
    r = TP.serve_request(st, False, 'GET', '/p/pv/', {})
    check('two-port: a GET right after the switch (no registry change) may still be answered by the page built before',
          r[0] == 200 and f'version={v2.vid}'.encode() in r[2] and t2.encode() in r[2], r[:1])
    n_before = (len(lines(os.path.join(U, J, 'judgements.jsonl'))), len(lines(f'{v2.run}/{J}/judgements.jsonl')))

    def post(body):
        return TP.serve_request(st, False, 'POST', '/p/pv/label', {'Content-Length': str(len(body))}, body)
    r_old = post(f'source_id=J7&confidence=fp&{TP.PAGE_TOKEN_FIELD}={t2}'.encode())
    r_none = post(b'source_id=J7&confidence=fp')
    n_after = (len(lines(os.path.join(U, J, 'judgements.jsonl'))), len(lines(f'{v2.run}/{J}/judgements.jsonl')))
    check('two-port: a label form rendered by the version before the switch (its token) and one without a token are '
          'answered 409 "the page changed since you loaded it; reload and judge again; nothing was recorded", nothing '
          'written to either journal', r_old[0] == 409 and r_none[0] == 409 and b'reload and judge again' in r_old[2]
          and b'nothing was recorded' in r_old[2] and n_after == n_before, (r_old[:1], r_none[:1], n_before, n_after))
    r = TP.serve_request(st, False, 'GET', '/p/pv/', {})
    t3 = (tok_re.findall(r[2]) or [b''])[0].decode()
    check('two-port: the reload is the page built on the served version, its form the new token',
          f'version={v3.vid}'.encode() in r[2] and t3 and t3 != t2, r[2][:200])
    r = post(f'source_id=J7&{TP.PAGE_TOKEN_FIELD}={t3}&confidence=fp'.encode())
    got = lines(os.path.join(U, J, 'judgements.jsonl'))
    check('two-port: the form of the served version is recorded by a page built on it, in the served journal, without the '
          'token field (the page reads the form it always read)',
          r[0] == 303 and got[-1].get('source_id') == 'J7' and got[-1].get('confidence') == 'fp'
          and got[-1].get('page_version') == v3.vid and TP.PAGE_TOKEN_FIELD not in got[-1]
          and os.path.realpath(U) == v3.run and not any(x.get('source_id') == 'J7' for x in lines(f'{v2.run}/{J}/judgements.jsonl')),
          (r[:2], got[-1:]))
    pg = st.pages.get(('pv', False))
    check('two-port: the page now answering is current (built on the version U names)', pg is not None and not st.moved(pg)
          and pg.labels_real == os.path.realpath(os.path.join(U, J)), pg and pg.labels_real)
    # 9c. a POST in flight across the switch: it holds the run's journal lock, so the switch waits; its line goes to the old
    # version and the catch-up (under the same locks) carries it into the new one
    lk = TP.journal_lock_path(os.path.realpath(os.path.join(U, J)))
    check('the two-port POST lock of a labels directory in a version is the run\'s journal lock (one file for every version)',
          lk == TP.journal_lock_file(versions.root_of(U), J) == TP.journal_lock_path(os.path.join(v1.run, J)), lk)
    v4, rec4 = next_version(U, 'D', J)
    holding, order = threading.Event(), []

    def post_in_flight():
        with TP.post_lock(threading.Lock(), os.path.realpath(os.path.join(U, J))):
            holding.set()
            time.sleep(0.5)
            with open(os.path.join(U, J, 'judgements.jsonl'), 'a') as f:     # through the pointer, as the page writes
                f.write(json.dumps({'source_id': 'J9', 'action': 'classify'}) + '\n')
            order.append('post')
    t = threading.Thread(target=post_in_flight)
    t.start()
    holding.wait(10)
    with versions.journal_locks(U, [J]):
        order.append('switch')
        sw = versions.switch(U, v4, record=rec4)
        app = journals.append_between(sw['previous'].run, U)
    t.join(10)
    served = lines(os.path.join(U, J, 'judgements.jsonl'))
    check('a POST in flight across the switch is written before it (the switch waits for the journal lock) and caught up '
          'into the served version', order == ['post', 'switch'] and served[-1].get('source_id') == 'J9'
          and app.get(J) == 1 and sum(x.get('source_id') == 'J9' for x in served) == 1, (order, app, served[-2:]))
    # the legacy run (before its first switch, once its versions root exists): the same lock file as after it
    L = f'{d}/Run_LEG'
    write(f'{L}/{J}/judgements.jsonl', '')
    os.makedirs(versions.root_of(L))
    check('the legacy run\'s POST lock is the journal lock its first switch takes',
          TP.journal_lock_path(os.path.join(L, J)) == TP.journal_lock_file(versions.root_of(L), J), TP.journal_lock_path(L))
    check('a run without versions keeps the two-port lock by directory', TP.journal_lock_path(f'{TMP}/judge') is None)
    check('the two-port copies of the versions suffix and version-name pattern match the config and versions.py',
          TP.VERSIONS_SUFFIX == CFG.serve.RUN_VERSIONS_SUFFIX and TP.VERSION_ID_RE.pattern == versions._VID_RE.pattern)
    # 9d. the sub-LST pages pass --labels-dir through the pointer (the page resolves only --run)
    src = read(os.path.join(REPO, 'exopipe/phase3/lib/serve/plan.py'))
    check('the plan gives the sub-LST pages --labels-dir under U (page.at(\'labels\', U)), never a version path',
          "argv += ['--labels-dir', page.at('labels', U)]" in src and 'U = root or page.root' in src)


# ---- 10. the review's other findings -----------------------------------------------------------------------------------------
def test_review_findings():
    from exopipe.phase3.lib.serve import tests_versions as TV
    d = f'{TMP}/findings'
    os.makedirs(d)
    # finding 4: the page index rebuilt in place on a sealed version
    U = f'{d}/Run_PI'
    v = new_version(U, 'P', files=('a.txt', 'page_index/manifest.json', 'page_index/DIRTY', 'page_index_build.log'))
    versions.publish(U, v)
    write(f'{U}/page_index/manifest.json', 'rebuilt')
    write(f'{U}/page_index/V/new_folder.json', '{}')
    write(f'{U}/page_index_build.log', 'repair')
    ok, det = versions.verify(v)
    check('the page index rebuilt in place (serve --stages page-index) leaves the served version verifiable', ok, det)
    write(f'{U}/a.txt', 'changed')
    check('... and a change to any other file still fails it', not versions.verify(v)[0])
    # finding 5: adoption
    E = f'{d}/Run_EMPTY'
    os.makedirs(f'{E}/sub')
    ve = new_version(E, 'E')
    why = refuses(versions.publish, E, ve)
    check('an empty legacy run is not adopted: the switch refuses, U still its directory, no adopted version',
          why and 'holds no file' in why and os.path.isdir(E) and not os.path.islink(E)
          and not any(x.vid.endswith('adopted') for x in versions.all_versions(E)), why)
    F = f'{d}/Run_FAIL'
    write(f'{F}/x.txt', 'legacy')
    vf, recf = new_version(F, 'F'), None
    recf = versions.seal(vf)
    real_rename = os.rename

    def failing_rename(a, b):
        if os.path.realpath(a) == os.path.realpath(F):
            raise OSError(5, 'stand-in rename failure')
        return real_rename(a, b)
    os.rename = failing_rename
    try:
        why = refuses(versions.switch, F, vf, record=recf)
    finally:
        os.rename = real_rename
    check('a failed rename at adoption refuses with U intact, no version directory or record left',
          why and 'could not be moved' in why and read(f'{F}/x.txt') == 'legacy' and not os.path.islink(F)
          and not any(x.vid.endswith('adopted') for x in versions.all_versions(F)), why)
    stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime(os.stat(F).st_mtime))
    R = versions.root_of(F)
    for n in range(1, 1000):
        os.mkdir(os.path.join(R, f'{stamp}-adopted' if n == 1 else f'{stamp}-{n}-adopted'))
    why = refuses(versions.switch, F, vf, record=recf)
    check('no free adopted-version name (999 taken) refuses: no name reused, U intact', why and 'no free version name' in why
          and read(f'{F}/x.txt') == 'legacy' and not os.path.islink(F), why)
    # a legacy run holding only review state (a journal) is adopted with an empty manifest; a seal of the adopted run that
    # fails after the switch is returned, not raised (U is switched by then)
    Jr = CFG.serve.SUBLST_JOURNALS[0]
    M = f'{d}/Run_MUT'
    write(f'{M}/{Jr}/judgements.jsonl', '{"a": 1}\n')
    sw = versions.publish(M, new_version(M, 'M'))
    check('a legacy run holding only a journal: adopted, given an empty manifest, verifiable',
          sw['adopted'] is not None and (sw.get('adopted_record') or {}).get('n_files') == 0
          and versions.verify(sw['adopted'])[0], sw.get('adopted_record'))
    M2 = f'{d}/Run_MUT2'
    write(f'{M2}/x.txt', 'legacy')
    real_sa = versions.seal_adopted

    def failing_seal_adopted(v, note=''):
        raise Refusal('stand-in: the adopted run could not be walked')
    versions.seal_adopted = failing_seal_adopted
    try:
        m2 = new_version(M2, 'N')
        sw = versions.publish(M2, m2)
    finally:
        versions.seal_adopted = real_sa
    check('a failed seal of the adopted run after the switch is returned (adopted_error), U switched, the version kept '
          'as adopted', os.path.realpath(M2) == os.path.realpath(m2.run) and 'stand-in' in (sw.get('adopted_error') or '')
          and (sw['adopted'].record() or {}).get('state') == 'adopted', sw.get('adopted_error'))
    # finding 6: an adopted version without a manifest is not one of the kept newest (and is kept itself)
    K = f'{d}/Run_KEEP'
    k1 = new_version(K, '1')
    versions.publish(K, k1)
    time.sleep(1.05)
    k2 = new_version(K, '2')
    versions.publish(K, k2)
    fake = versions.Version(versions.root_of(K), '20991231T000000Z-adopted', 'Run_KEEP')
    write(f'{fake.run}/x.txt', 'legacy')
    versions._atomic_json(fake.path('VERSION.json'), {'format': versions.FORMAT, 'name': 'Run_KEEP', 'version': fake.vid,
                                                      'state': 'adopted', 'manifest': None})
    rows = {x.vid: (act, why) for x, act, why in versions.retention(K, keep=2)}
    check('retention (keep 2): an adopted version without a manifest is kept but not counted, so the two newest sealed '
          'versions are kept', rows[k1.vid][0] == 'keep' and 'newest' in rows[k1.vid][1] and rows[fake.vid][0] == 'keep'
          and 'without a manifest' in rows[fake.vid][1], rows)
    # finding 7: leases
    now = time.time()
    Ls = versions._lease_s()
    check('alive: a holder on another host is alive until twice its lease has passed, not for ever; a record without a '
          'lease (before leases) counts as alive',
          versions.alive({'host': 'another-node', 'pid': 1, 'lease_until': now + 60, 'lease_s': Ls})
          and versions.alive({'host': 'another-node', 'pid': 1, 'lease_until': now - 1, 'lease_s': Ls})
          and not versions.alive({'host': 'another-node', 'pid': 1, 'lease_until': now - Ls - 1, 'lease_s': Ls})
          and versions.alive({'host': 'another-node', 'pid': 1}))
    check('alive: on this host the pid decides (a dead pid with a lease is not alive)',
          not versions.alive(dict(versions.me(), start=-1)) and versions.alive(versions.me()))
    real_lease = versions._lease_s
    versions._lease_s = lambda: 0.4
    try:
        with versions.hold(U) as _:
            hf = [p for p, _h in versions.served(U).holds()][0]
            l0 = json.load(open(hf))['lease_until']
            time.sleep(0.6)
            l1 = json.load(open(hf))['lease_until']
        b = new_version(f'{d}/Run_LEASE', 'L')
        b0 = b.builder()['lease_until']
        time.sleep(0.6)
        b1 = b.builder()['lease_until']
        versions.seal(b)
        time.sleep(0.3)
        b2 = b.builder()['lease_until']
        time.sleep(0.3)
        b3 = b.builder()['lease_until']
    finally:
        versions._lease_s = real_lease
    check('a hold renews its lease while held and its file is gone after', l1 > l0 and not os.path.exists(hf), (l0, l1))
    check('a version being built renews its builder\'s lease until it is sealed', b1 > b0 and b2 == b3, (b0, b1, b2, b3))
    # finding 8: one walk per publish; none for the content test of a unit's own seal
    calls = []
    real_walk = versions.walk

    def counting_walk(*a, **k):
        calls.append(a[0])
        return real_walk(*a, **k)
    versions.walk = counting_walk
    try:
        W = f'{d}/Run_WALK'
        w1 = new_version(W, 'W')
        versions.publish(W, w1)
        w2 = new_version(W, 'X')
        n0 = len(calls)
        versions.publish(W, w2)
        n_publish = len(calls) - n0
        r = TV.sealed_complete(w2.path('VERSION.json'))
        n_test = len(calls) - n0 - n_publish
    finally:
        versions.walk = real_walk
    check('a publish walks the run once (the seal; the switch checks the seal\'s record and manifest)', n_publish == 1,
          n_publish)
    check('the run_version content test walks the run again against MANIFEST.tsv (second review, finding 9)',
          r[0] and n_test == 1, (r, n_test))
    write(f'{w2.run}/added.txt', 'x')
    check('a switch to a version without the caller\'s seal record walks it (a change after the seal refuses)',
          'differs from its manifest' in (refuses(versions.switch, W, w2) or '') or 'already names' in
          (refuses(versions.switch, W, w2) or ''))
    os.remove(f'{w2.run}/added.txt')
    # finding 9: an empty stage selection refuses
    from exopipe.phase3.steps import serve as SS
    a = types.SimpleNamespace(stages=None, from_stage='retire', until=None)
    why = refuses(SS.select_stages, ('build', 'start', 'retire'), a, default_off=('retire',))
    check('--from retire without --stages retire selects nothing: refused, not a silent no-op', why and 'no stage' in why, why)
    a = types.SimpleNamespace(stages='retire', from_stage='build', until=None)
    check('--stages retire --from build selects retire', SS.select_stages(('build', 'start', 'retire'), a, ('retire',))
          == ['retire'])
    # finding 2: the status probe never takes the lock
    slot = coalesce.Slot(f'{d}/slot_probe')

    def no_lock():
        raise AssertionError('builder() took the lock')
    real_try = slot._try_lock
    slot._try_lock = no_lock
    try:
        b = slot.builder()
        ok = True
    except AssertionError:
        ok = False
    slot._try_lock = real_try
    check('Slot.builder() reads BUILDER.json and never takes the lock', ok and b is None)
    stop, outs = threading.Event(), []

    def prober():
        while not stop.is_set():
            slot.builder()
    th = threading.Thread(target=prober)
    th.start()
    try:
        for _ in range(200):
            outs.append(slot.submit(lambda: 0)['status'])
    finally:
        stop.set()
        th.join(10)
    check('200 requests with a status probe running all build (none left coalesced into no builder)',
          outs.count('built') == 200, {k: outs.count(k) for k in set(outs)})


# ---- 11. the coalescing builder wired into the sub-LST build and placement --------------------------------------------------
def test_coalescer_wired():
    from exopipe.phase3.core.record import Run
    from exopipe.phase3.lib.pages import tier_sublst
    from exopipe.phase3.lib.serve import sublst_build
    d = f'{TMP}/wired'
    U = f'{d}/Run_SLSTB'
    os.makedirs(d)
    out = f'{d}/out'
    plan = types.SimpleNamespace(U=U, U_src=U, run_name='Run_SLSTB', parity=False, pages=[], tag='first', tier='sublst')
    args = types.SimpleNamespace()
    calls = []
    real = (sublst_build.build_and_serve, sublst_build._fresh_plan)
    slot = coalesce.slot_for(U)
    try:
        sublst_build._fresh_plan = lambda a, r: types.SimpleNamespace(**dict(vars(plan), tag='fresh'))
        # a build of the run runs elsewhere (holds the slot): this invocation is coalesced, nothing built
        sublst_build.build_and_serve = lambda a, r, p, st: calls.append(p.tag)
        fd = slot._try_lock()
        with open(slot.builder_file, 'w') as f:
            json.dump(versions.me('another serve'), f)
        try:
            run = Run('serve', ['test'], out, CFG, config_sections=None)
            sublst_build.run(args, run, plan, ['build', 'views', 'test', 'carry', 'swap', 'start'])
        finally:
            os.unlink(slot.builder_file)
            os.close(fd)
        u = run.units[-1]
        check('serve --tier sublst while a build of the run runs: unit sublst_<run>_coalesced, no-input (COALESCED), nothing '
              'built, the request left for that builder', calls == [] and u.name == 'sublst_Run_SLSTB_coalesced'
              and u.status == 'no-input' and (u.reason or '').startswith('COALESCED') and slot.pending()
              and run.exit_code() == 3, (calls, u.name, u.status, u.reason))
        # the slot free: this invocation builds (the pending request and its own), and a placement during its build gets
        # exactly one follow-up, on a plan made again at its start

        def building(a, r, p, st):
            calls.append(p.tag)
            if len(calls) == 1:
                slot.request('a placement during the build')
        sublst_build.build_and_serve = building
        run = Run('serve', ['test'], out, CFG, config_sections=None)
        sublst_build.run(args, run, plan, ['build', 'start'])
        check('a free slot: the invocation builds, and a request made during its build gets one follow-up on a fresh plan',
              calls == ['first', 'fresh'] and not slot.pending(), calls)
        # a refusal inside the build is raised once the slot is released
        calls.clear()

        def refusing(a, r, p, st):
            calls.append(p.tag)
            raise Refusal('stand-in refusal outside a unit')
        sublst_build.build_and_serve = refusing
        why = refuses(sublst_build.run, args, Run('serve', ['test'], out, CFG, config_sections=None), plan, ['start'])
        fd = slot._try_lock()
        check('a build\'s refusal is raised by serve after the slot is released', why and 'stand-in' in why
              and fd is not None, why)
        if fd is not None:
            os.close(fd)
        # stages without build or start run directly
        calls.clear()
        sublst_build.build_and_serve = lambda a, r, p, st: calls.append(tuple(st))
        n = len(open(slot.journal).read().splitlines())
        sublst_build.run(args, Run('serve', ['test'], out, CFG, config_sections=None), plan, ['retire'])
        check('stages without build or start (retire) run directly, no request', calls == [('retire',)]
              and len(open(slot.journal).read().splitlines()) == n, calls)
    finally:
        sublst_build.build_and_serve, sublst_build._fresh_plan = real
    # the placement requests a build of the cumulative run beside the runs it placed
    lst = '05h'
    c = types.SimpleNamespace(out=f'{d}/pages_out', place_target=f'{d}/transients', sets=[])
    P = tier_sublst.P_of(c, lst)
    write(f'{P}/candidates/V/x.txt', 'hour')         # (the unit's sublst_page_run content test refuses this stand-in run)
    os.makedirs(f'{d}/transients')
    run = Run('pages', ['test'], f'{d}/pages_records', CFG, config_sections=None)
    tier_sublst.stage_place(c, run, {}, lst)
    u = run.units[-1]
    cum = coalesce.slot_for(os.path.join(f'{d}/transients', CFG.serve.RUN_NAME['sublst']))
    check('the sub-LST placement places the hour run and requests a build of the cumulative run beside it',
          os.path.isfile(f'{d}/transients/{os.path.basename(P)}/candidates/V/x.txt') and cum.pending()
          and u.meta.get('build_requested') == cum.d and 'placed' in json.loads(open(cum.flag).read()).get('note', ''),
          (u.status, u.reason, u.meta.get('build_requested')))


# ---- 12. the second review round ---------------------------------------------------------------------------------------------
def test_review_round2():
    from exopipe.phase3.core.record import Run
    from exopipe.phase3.lib.catalog import grouping_stage as GS
    from exopipe.phase3.lib.serve import flow, journals, sublst_build
    from exopipe.phase3.lib.serve import tests_versions as TV
    from exopipe.phase3.lib.serve import two_port as TP
    d = f'{TMP}/round2'
    os.makedirs(d)
    J = CFG.serve.SUBLST_JOURNALS[0]
    rels = {'queue': 'judgements/queue_V.txt', 'catalog': 'source_catalogs_V'}
    page = types.SimpleNamespace(key='V', family='test', program='test',
                                 at=lambda f, root=None: os.path.join(root, rels[f]) if f in rels else None)

    def build(B, tag):
        write(f'{B}/candidates/V/x.txt', tag)
        write(f'{B}/candidates/I/y.txt', tag)
        write(f'{B}/judgements/queue_V.txt', 'J1\n')
        write(f'{B}/source_catalogs_V/23MHz_sources_survivors.csv', 'source_id\nJ1\n')

    def sublst(name):
        U = f'{d}/{name}/Run_SLSTB'
        B = U + CFG.serve.SUBLST_BUILD_SUFFIX
        write(f'{U}/{J}/judgements.jsonl', '{"a": 1}\n')
        write(f'{U}/candidates/V/x.txt', 'old')
        build(B, 'new')
        write(f'{B}/{J}/judgements.jsonl', '{"a": 1}\n')
        return U, B, types.SimpleNamespace(U=U, U_src=U, run_name='Run_SLSTB', parity=False, pages=[page], family='review',
                                           refused={}, tier='sublst')

    def new_run():
        return Run('serve', ['test'], f'{d}/out', CFG, config_sections=None)

    def refusing(*a, **k):
        raise Refusal('stand-in: the switch refused')
    orig = (flow.build_page_index, flow._index_current)
    flow.build_page_index = lambda u, R, publisher=False: None
    flow._index_current = lambda R: ''
    try:
        # finding 1: no pointer log is "never served"; the move back works at the FIRST switch; a failed move back never
        # masks the refusal
        U, B, plan = sublst('first')
        state = flow.new_state()
        state.update(version=versions.place(U, B), carry_src=U, design_b=False, marks={})
        v = state['version']
        check('no POINTER_LOG.jsonl (no switch yet): ever_served is False', not os.path.exists(
            f'{versions.root_of(U)}/POINTER_LOG.jsonl') and versions.ever_served(v) is False)
        real_switch, real_unplace = versions.switch, versions.unplace
        versions.switch = refusing
        try:
            u = sublst_build.switch_stage(new_run(), plan, state)
        finally:
            versions.switch = real_switch
        check('the FIRST switch refused after the seal: the sealed build moved back to B, U still the legacy directory',
              u.status == 'refused' and 'stand-in' in (u.reason or '') and os.path.isdir(B) and not os.path.exists(v.dir)
              and not os.path.islink(U) and read(f'{U}/candidates/V/x.txt') == 'old', (u.status, u.reason))

        def failing_unplace(*a, **k):
            raise OSError(5, 'stand-in move-back failure')
        state['version'] = v = versions.place(U, B)
        versions.switch, versions.unplace = refusing, failing_unplace
        try:
            u = sublst_build.switch_stage(new_run(), plan, state)
        finally:
            versions.switch, versions.unplace = real_switch, real_unplace
        why = u.reason or ''
        check('a failed move back never masks the refusal: the switch\'s refusal is the unit\'s reason, the move-back failure '
              'added after it', u.status == 'refused' and 'stand-in: the switch refused' in why
              and 'stand-in move-back failure' in why
              and why.index('stand-in: the switch refused') < why.index('could not be moved back'), (u.status, why))
        versions.unplace(v, B, sealed=True)
        write(f'{versions.root_of(U)}/POINTER_LOG.jsonl', 'not json\n')
        check('a pointer log that exists and cannot be parsed: ever_served refuses (not decided), never guesses',
              'cannot be read' in (refuses(versions.ever_served, v) or ''))
        os.remove(f'{versions.root_of(U)}/POINTER_LOG.jsonl')
        # finding 2: the seed carry/copy reads the served journals under the journal locks; the maps are loaded before them
        events = []
        real = (versions.journal_locks, journals.sublst_carry_maps, journals.sublst_journal_snapshot,
                journals.sublst_carry_journals, journals.sublst_copy_journals, journals.sublst_carry_between, GS.mode)

        @contextlib.contextmanager
        def jl(U_, Js):
            with real[0](U_, Js):
                events.append('lock')
                yield
                events.append('unlock')
        versions.journal_locks = jl
        journals.sublst_carry_maps = lambda old, new: events.append(('maps', old, new)) or {'V': ('old', 'new')}
        journals.sublst_journal_snapshot = lambda src: events.append('snapshot') or {J: ['{"a": 1}']}
        journals.sublst_carry_journals = lambda src, B_, snapshot=None, maps=None: (
            events.append(('carry', snapshot, maps)) or ({J: {}}, {J: len(snapshot[J])}))
        journals.sublst_copy_journals = lambda src, B_: events.append('copy') or real[4](src, B_)
        journals.sublst_carry_between = lambda P, U_, marks, maps=None: events.append(('between', P, marks, maps)) or {J: 0}
        try:
            GS.mode = lambda: types.SimpleNamespace(design_b=True)
            st = flow.new_state()
            u = sublst_build.carry_swap_stage(new_run(), plan, st)
            ev_b = list(events)
            check('design B seed: maps loaded before the journal locks, the journals read under them (a consistent cut), the '
                  'carry after', u.status == 'done' and ev_b == [('maps', U, B), 'lock', 'snapshot', 'unlock',
                                                               ('carry', {J: ['{"a": 1}']}, {'V': ('old', 'new')})],
                  (u.status, u.reason, ev_b))
            versions.unplace(st['version'], B)
            events.clear()
            GS.mode = lambda: types.SimpleNamespace(design_b=False)
            st = flow.new_state()
            u = sublst_build.carry_swap_stage(new_run(), plan, st)
            check('the journal copy (not design B) runs under the journal locks', u.status == 'done'
                  and events == ['lock', 'copy', 'unlock'], (u.status, u.reason, events))
            versions.unplace(st['version'], B)
            # finding 10: the catch-up's catalogues and member maps are loaded before the locks; under them only the tail
            events.clear()
            legacy_real = os.path.realpath(U)
            state = flow.new_state()
            state.update(version=versions.place(U, B), carry_src=U, design_b=True, marks={J: 1})
            v1 = state['version']
            u = sublst_build.switch_stage(new_run(), plan, state)
            prev = versions.Version(v1.root, u.meta.get('previous') or '', 'Run_SLSTB')
            check('design-B catch-up: the maps of the old run (the legacy directory, adopted by this switch) and the new '
                  'version loaded before the locks; under them only the switch and the tail carried with those maps',
                  u.status == 'done' and events == [('maps', legacy_real, v1.run), 'lock',
                                                    ('between', prev.run, {J: 1}, {'V': ('old', 'new')}), 'unlock'],
                  (u.status, u.reason, events))
        finally:
            (versions.journal_locks, journals.sublst_carry_maps, journals.sublst_journal_snapshot,
             journals.sublst_carry_journals, journals.sublst_copy_journals, journals.sublst_carry_between, GS.mode) = real
        # finding 4: a catch-up that fails after the switch is owed; builds and switches refuse until it completes
        build(B, 'second')
        write(f'{B}/{J}/judgements.jsonl', read(f'{U}/{J}/judgements.jsonl'))              # the seed copy
        with open(f'{U}/{J}/judgements.jsonl', 'a') as f:
            f.write('{"c": 3}\n')                                                       # a label after the copy
        real_ab = journals.append_between

        def failing_ab(P, U_):
            raise OSError(5, 'stand-in catch-up failure')
        journals.append_between = failing_ab
        try:
            state = flow.new_state()
            state.update(version=versions.place(U, B), carry_src=U, design_b=False, marks={})
            v2 = state['version']
            u = sublst_build.switch_stage(new_run(), plan, state)
        finally:
            journals.append_between = real_ab
        owed = versions.owed_catchup(U)
        check('a catch-up that fails after the switch: the unit refused (SWITCHED ... OWED), U names the new version, the '
              'owed record names the old version and the marks', u.status == 'refused' and 'SWITCHED' in (u.reason or '')
              and 'OWED' in u.reason and os.path.realpath(U) == v2.run and owed and owed['from_version'] == v1.vid
              and owed['to_version'] == v2.vid and owed['from_run'] == v1.run, (u.status, u.reason, owed))
        os.makedirs(f'{d}/first/another.build')
        check('while a catch-up is owed: begin, place and switch of U refuse',
              'owes the catch-up' in (refuses(versions.begin, U) or '')
              and 'owes the catch-up' in (refuses(versions.place, U, f'{d}/first/another.build') or '')
              and 'owes the catch-up' in (refuses(versions.switch, U, v1) or ''))
        rows = {x.vid: (act, why) for x, act, why in versions.retention(U, keep=1)}
        check('while owed, retention keeps the old version (keep 1)', rows[v1.vid][0] == 'keep'
              and 'owed' in rows[v1.vid][1], rows)
        uc = sublst_build.catchup_stage(new_run(), plan)
        check('the next serve completes the owed catch-up first: the line written on the old version is in the served '
              'journal once, the record removed', uc is not None and uc.status == 'done' and versions.owed_catchup(U) is None
              and read(f'{U}/{J}/judgements.jsonl').count('"c": 3') == 1, (uc and uc.status, uc and uc.reason))
        check('nothing owed: no catch-up unit', sublst_build.catchup_stage(new_run(), plan) is None)
    finally:
        flow.build_page_index, flow._index_current = orig
    # finding 8: adoption rolls back on any failure before U points at the new version
    L = f'{d}/adopt/Run_L'
    write(f'{L}/x.txt', 'legacy')
    vl = new_version(L, 'L')
    recl = versions.seal(vl)
    real_replace = os.replace

    def failing_replace(a, b):
        if os.path.abspath(b) == os.path.abspath(L):
            raise OSError(5, 'stand-in pointer write failure')
        return real_replace(a, b)
    os.replace = failing_replace
    try:
        why = refuses(versions.switch, L, vl, record=recl)
    finally:
        os.replace = real_replace
    check('a failure after the adoption and before U points at the new version: the legacy run moved back to U, no adopted '
          'version or temporary link left', why and 'stand-in pointer write failure' in why and os.path.isdir(L)
          and not os.path.islink(L) and read(f'{L}/x.txt') == 'legacy'
          and not any(x.vid.endswith('adopted') for x in versions.all_versions(L))
          and not any(n.startswith('.Run_L.pointer') for n in os.listdir(os.path.dirname(L))), why)
    real_aj = versions._atomic_json

    def failing_aj(path, obj):
        if obj.get('state') == 'adopted':
            raise OSError(28, 'stand-in: no space for the adoption record')
        return real_aj(path, obj)
    versions._atomic_json = failing_aj
    try:
        why = refuses(versions.switch, L, vl, record=recl)
    finally:
        versions._atomic_json = real_aj
    check('a failed adoption record: the run moved back to U, refused', why and 'adoption record' in why
          and os.path.isdir(L) and not os.path.islink(L) and read(f'{L}/x.txt') == 'legacy'
          and not any(x.vid.endswith('adopted') for x in versions.all_versions(L)), why)
    sw = versions.switch(L, vl, record=recl)
    check('... and the next switch adopts and switches', os.path.islink(L) and sw['adopted'] is not None)
    # finding 9: the content test walks the run against MANIFEST.tsv
    write(f'{vl.run}/added.txt', 'x')
    r = TV.sealed_complete(vl.path('VERSION.json'))
    os.remove(f'{vl.run}/added.txt')
    r2 = TV.sealed_complete(vl.path('VERSION.json'))
    check('run_version content test: a file added after the seal fails it (the run walked against MANIFEST.tsv)',
          not r[0] and 'added' in r[2] and r2[0], (r, r2))
    # finding 7: leases - a renewal's write and fsync outside the renewer lock; renew_stop waits only for its own path
    hf, hf2 = f'{d}/lease/h.json', f'{d}/lease/h2.json'
    write(hf, json.dumps(versions.me()))
    write(hf2, json.dumps(versions.me()))
    before = read(hf)
    real_lease, real_fsync = versions._lease_s, os.fsync
    slow, synced = threading.Event(), []

    def slow_fsync(fd):
        if (threading.current_thread().name == 'versions-lease' and not slow.is_set()
                and os.path.basename(os.readlink(f'/proc/self/fd/{fd}')).startswith('.h.json.lease')):
            slow.set()
            time.sleep(0.6)
            r = real_fsync(fd)
            synced.append(time.time())
            return r
        return real_fsync(fd)
    versions._lease_s = lambda: 60.0                 # one renewal (at renew()), none after it during the test
    os.fsync = slow_fsync
    try:
        versions.renew(hf)
        slow.wait(5)
        t0 = time.time()
        versions.renew(hf2)
        versions.renew_stop(hf2)
        dt_other = time.time() - t0
        versions.renew_stop(hf)
        t_stop = time.time()
        left = sorted(os.listdir(f'{d}/lease'))
        after = read(hf)
        shutil.rmtree(f'{d}/lease')                    # what unplace does with a version directory right after its stop
        time.sleep(0.3)
        gone = not os.path.exists(f'{d}/lease')
    finally:
        os.fsync, versions._lease_s = real_fsync, real_lease
    check('a renewal is written and fsynced outside the renewer lock: during a slow fsync another path is renewed and stopped '
          'at once', slow.is_set() and dt_other < 0.3, dt_other)
    check('renew_stop of the path in flight returns only after its renewal ended: no temporary file left, the record not '
          'rewritten, the directory removable at once and not recreated', synced and synced[0] <= t_stop
          and left == ['h.json', 'h2.json'] and after == before and gone, (synced, t_stop, left))
    # findings 5 and 6: the lock files
    VR = f'{d}/locks/Run_K'
    vk = new_version(VR, 'K')
    os.makedirs(f'{vk.run}/{J}')
    versions.publish(VR, vk)
    lab = os.path.realpath(os.path.join(VR, J))
    real_lockdir = TP.LOCK_DIR
    TP.LOCK_DIR = f'{d}/legacy_locks'
    try:
        legacy = os.path.join(TP.LOCK_DIR, hashlib.md5(lab.encode()).hexdigest() + '.lock')
        with TP.post_lock(threading.Lock(), lab) as jlp:
            pass
        check('a versioned labels directory without a legacy lock file on this node: the journal lock only (none created)',
              jlp == TP.journal_lock_file(versions.root_of(VR), J) and not os.path.exists(legacy), jlp)
        write(legacy, '')
        fd = os.open(legacy, os.O_RDWR)
        fcntl.flock(fd, fcntl.LOCK_EX)                             # a POST of a hub running the code before versioned runs
        got = []

        def poster():
            with TP.post_lock(threading.Lock(), lab):
                got.append(time.time())
        t = threading.Thread(target=poster)
        t.start()
        time.sleep(0.3)
        blocked = not got
        released = time.time()
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
        t.join(5)
        check('the legacy lock exists on this node (an old-code hub\'s): a POST waits for it as well as the journal lock',
              blocked and got and got[0] >= released, (blocked, got, released))
    finally:
        TP.LOCK_DIR = real_lockdir
    old_umask = os.umask(0o077)                                     # a restrictive umask must not decide the mode
    try:
        p6 = f'{d}/perm/Run_P.versions/locks/journal_x.lock'
        os.close(TP.open_lock(p6))
    finally:
        os.umask(old_umask)
    mode, dmode = os.stat(p6).st_mode & 0o7777, os.stat(os.path.dirname(p6)).st_mode & 0o7777
    check('a lock file is created group-writable (0o664 under umask 077) in a setgid locks directory (its group inherited)',
          mode == 0o664 and dmode & 0o2000 and dmode & 0o070 == 0o070, (oct(mode), oct(dmode)))
    os.chmod(p6, 0o444)
    why = None
    try:
        os.close(TP.open_lock(p6))
    except TP.LockPermissionError as e:
        why = str(e)
    check('a lock this process may not open: a clear refusal naming the file and its owner', why and p6 in why
          and 'owned by' in why, why)
    P6 = f'{d}/perm/Run_P'
    jp = TP.journal_lock_file(versions.root_of(P6), J)
    os.close(TP.open_lock(jp))
    os.chmod(jp, 0o444)
    why = refuses(lambda: versions.journal_locks(P6, [J]).__enter__())
    check('versions.journal_locks: the same is a Refusal (not an unhandled exception)', why and jp in why and 'owned by' in why,
          why)
    os.chmod(jp, 0o664)
    os.chmod(p6, 0o664)


# ---- 13. content: the manifest's sha256 and the content test's re-reads (PR 155, coordinator on finding 9) -----------------
def test_content_hashes():
    d = f'{TMP}/content'
    U = f'{d}/Run_C'
    os.makedirs(d)
    S = CFG.serve
    v = versions.begin(U)
    write(f'{v.run}/queue.txt', 'J1\nJ2\n')
    write(f'{v.run}/cat/x.csv', 'source_id\nJ1\n')
    write(f'{v.run}/cand/a.key', 'k')
    pngs = [f'panels/p{i:03d}.png' for i in range(100)]
    for i, rel in enumerate(pngs):
        write(f'{v.run}/{rel}', f'png {i:03d}')
    rec = versions.seal(v)
    want = {e[1]: e for e in versions._read_manifest(v.path('MANIFEST.tsv'))}
    sample = versions.large_sample(pngs, rec['hash_rules'])
    check('the record carries the hash rules (16 MiB, the text and binary types, binaries sampled at the seal: 2%, at least '
          '20)',
          rec['hash_rules']['small_bytes'] == 16 * 1024 * 1024 and '.csv' in rec['hash_rules']['text']
          and '.png' in rec['hash_rules']['binary'] and rec['hash_rules']['large'] == 'sample'
          and rec['hash_rules']['sample_fraction'] == 0.02 and rec['hash_rules']['sample_min'] == 20, rec['hash_rules'])
    check('MANIFEST.tsv: the sha256 of every file not a binary (queue, CSV, key), of the binaries only their stable sample',
          all(want[r][5] == hashlib.sha256(open(f'{v.run}/{r}', 'rb').read()).hexdigest()
              for r in ('queue.txt', 'cat/x.csv', 'cand/a.key'))
          and {r for r in pngs if want[r][5]} == sample and len(sample) == 20
          and all(want[r][5] == hashlib.sha256(open(f'{v.run}/{r}', 'rb').read()).hexdigest() for r in sample)
          and rec['n_hashed'] == 23 and rec['n_binaries'] == 100 and rec['n_binaries_hashed'] == 20, (len(sample), rec))
    many = [f'panels/{i:05d}.png' for i in range(5000)]
    s1, s2 = versions.large_sample(many, rec['hash_rules']), versions.large_sample(list(reversed(many)), rec['hash_rules'])
    check('the stable sample: chosen by the path\'s hash (the same in any order), about 2% of many (5000 -> 60-140), at least '
          '20, all of fewer', s1 == s2 and 60 <= len(s1) <= 140
          and len(versions.large_sample(many[:30], rec['hash_rules'])) == 20
          and versions.large_sample(many[:7], rec['hash_rules']) == set(many[:7]), len(s1))
    st = {}
    ok, detail = versions.verify(v, stats=st)
    nb = sum(os.path.getsize(f'{v.run}/{r}') for r in ['queue.txt', 'cat/x.csv', 'cand/a.key'] + sorted(sample))
    check('the content test re-reads every file not a binary and the sample: its bytes read counted', ok
          and st['n_rehashed'] == 23 and st['bytes_read'] == nb and st['n_sample'] == 20, (detail, st))

    def same_meta(rel, fn):
        p = f'{v.run}/{rel}'
        s0 = os.stat(p)
        fn(p)
        os.utime(p, ns=(s0.st_atime_ns, s0.st_mtime_ns))
        assert os.path.getsize(p) == s0.st_size
    same_meta('queue.txt', lambda p: write(p, 'J1\nJ3\n'))
    ok, detail = versions.verify(v)
    check('a file whose content changed with its size and modification time kept fails the content test (sha256)',
          not ok and 'other content' in detail and 'queue.txt' in detail, detail)
    why = refuses(versions.switch, U, v)
    check('... and refuses the switch without the seal\'s record', why and 'differs from its manifest' in why, why)
    same_meta('queue.txt', lambda p: write(p, 'J1\nJ2\n'))
    os.utime(f'{v.run}/cat/x.csv')
    ok, detail = versions.verify(v)
    check('a file only touched (its modification time changed, the same content): re-hashed, passes', ok
          and '1 with a changed size or modification time (same content)' in detail, detail)
    sampled, other = sorted(sample)[0], sorted(set(pngs) - sample)[0]
    same_meta(sampled, lambda p: write(p, 'PNG 000'))
    ok, detail = versions.verify(v)
    check('a binary of the sample whose content changed (size and time kept) fails the content test', not ok
          and sampled in detail, detail)
    write(f'{v.run}/{sampled}', 'png ' + sampled[-7:-4])
    os.utime(f'{v.run}/{sampled}', ns=(want[sampled][3], want[sampled][3]))
    same_meta(other, lambda p: write(p, 'PNG ' + other[-7:-4]))
    st = {}
    ok, detail = versions.verify(v, stats=st)
    check('a binary outside the sample whose content changed with its size and time kept is not re-read (the sample\'s '
          'limit: 2% of the binaries per test)', ok and st['n_rehashed'] == 23, (detail, st))
    os.utime(f'{v.run}/{other}')
    ok, detail = versions.verify(v)
    check('a binary outside the sample whose modification time changed has no sha256 to compare (sample-only seal): fails',
          not ok and 'no sha256 to compare' in detail and other in detail, detail)
    write(f'{v.run}/{other}', 'png ' + other[-7:-4])
    # the seal's 'all': every binary hashed; a changed size or time is then re-hashed and judged by content
    real = S.RUN_VERSIONS_HASH_LARGE, S.RUN_VERSIONS_HASH_SMALL_BYTES
    S.RUN_VERSIONS_HASH_LARGE, S.RUN_VERSIONS_HASH_SMALL_BYTES = 'all', 100
    try:
        w = versions.begin(U)
        write(f'{w.run}/big.dat', 'x' * 200)             # over small_bytes, not a text type: a binary
        write(f'{w.run}/big.csv', 'y' * 200)             # a text type: re-hashed whatever its size
        for rel in pngs:
            write(f'{w.run}/{rel}', 'png')
        rw = versions.seal(w)
        mw = {e[1]: e for e in versions._read_manifest(w.path('MANIFEST.tsv'))}
        check('RUN_VERSIONS_HASH_LARGE all: every binary hashed at the seal; a non-text file over small_bytes is a binary, a '
              'text file of any size is not', all(mw[r][5] for r in pngs) and rw['n_binaries'] == 101
              and mw['big.dat'][5] and not versions.is_small('big.dat', 200, rw['hash_rules'])
              and versions.is_small('big.csv', 200, rw['hash_rules']), rw)
        os.utime(f'{w.run}/{other}')
        ok1, d1 = versions.verify(w)
        write(f'{w.run}/{other}', 'PNG')
        ok2, d2 = versions.verify(w)
        check('... a binary outside the sample whose time changed is re-hashed: the same content passes, another fails',
              ok1 and not ok2 and other in d2, (d1, d2))
    finally:
        S.RUN_VERSIONS_HASH_LARGE, S.RUN_VERSIONS_HASH_SMALL_BYTES = real
    # a file written while the seal reads it: refused, nothing written
    x = versions.begin(U)
    write(f'{x.run}/q.txt', 'a')
    real_sha = versions.sha256_file

    def writing(p):
        r = real_sha(p)
        with open(p, 'a') as f:
            f.write('more')
        return r
    versions.sha256_file = writing
    try:
        why = refuses(versions.seal, x)
    finally:
        versions.sha256_file = real_sha
    check('a file that changes while the seal reads it: the seal refuses, no manifest or record written',
          why and 'changed while the seal read it' in why and x.record() is None
          and not os.path.exists(x.path('MANIFEST.tsv')), why)
    rec_old = dict(rec)
    rec_old.pop('hash_rules')
    versions._atomic_json(v.path('VERSION.json'), rec_old)
    ok, detail = versions.verify(v)
    check('a record without hash rules: the content cannot be tested - fails', not ok and 'hash_rules' in detail, detail)


def main():
    shutil.rmtree(TMP, ignore_errors=True)
    os.makedirs(TMP)
    try:
        test_switch()
        test_adopt()
        test_completion()
        test_place()
        test_retention()
        test_coalesce()
        test_cli()
        test_sublst_switch()
        test_judgements_after_switch()
        test_review_findings()
        test_coalescer_wired()
        test_review_round2()
        test_content_hashes()
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    print(f'{N[0] - len(FAILS)} of {N[0]} checks passed')
    return 1 if FAILS else 0


def test_all():
    assert main() == 0


if __name__ == '__main__':
    sys.exit(main())
