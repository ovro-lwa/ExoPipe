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
   switch are appended; an abandoned placed build is moved back to the build directory by the next serve.
"""
import json
import os
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
    check('the first switch adopts the legacy directory as the previous version', ad is not None and sw['previous'] is ad
          and ad.vid.endswith('-adopted') and read(f'{ad.run}/candidates/V/x.txt') == 'legacy'
          and (ad.record() or {}).get('state') == 'adopted', (sw, ad and ad.record()))
    check('after adoption U is the pointer to the new version', os.path.islink(U) and read(f'{U}/a.txt') == 'N' * 1000)
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
    b = v.builder()
    versions._atomic_json(v.path('BUILDING.json'), dict(b, start=-1))       # its placing process is gone
    check('a placed build whose placer is gone is found', versions.placed_abandoned(U, B).vid == v.vid)
    versions.seal(v)
    check('a sealed version is never moved back', 'is sealed' in (refuses(versions.unplace, v, B) or '')
          and versions.placed_abandoned(U, B) is None)
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
    versions._atomic_json(dead.path('BUILDING.json'), dict(dead.builder(), start=-1))
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
    plan = types.SimpleNamespace(U=U, U_src=U, run_name='Run_SLSTB', parity=False, pages=[page])
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
        versions._atomic_json(w.path('BUILDING.json'), dict(w.builder(), start=-1))
        got = sublst_build.recover_placed(plan)
        check('an abandoned placed build is moved back to the build directory by the next serve',
              got is not None and got.vid == w.vid and read(f'{B}/candidates/V/x.txt') == 'newer'
              and not os.path.exists(w.dir), got)
    finally:
        flow.build_page_index, flow._index_current = orig


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
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    print(f'{N[0] - len(FAILS)} of {N[0]} checks passed')
    return 1 if FAILS else 0


def test_all():
    assert main() == 0


if __name__ == '__main__':
    sys.exit(main())
