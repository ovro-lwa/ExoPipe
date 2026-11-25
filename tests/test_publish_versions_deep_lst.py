#!/usr/bin/env python3
"""Tests of the deep and per-LST publishes on the versioned pointer (lib/serve/versions.py WORK section, lib/serve/deep.py
THE VERSIONED DEEP RUN, lib/pages/tier_sublst.py place_version, the G30 pins of lib/serve/sublst_build.py): data-free, plain
checks, runnable with python, under a minute. Scratch under $TMPDIR.

    NUMPY_MADVISE_HUGEPAGE=0 /opt/devel/pipeline/envs/py38_orca_nkosogor/bin/python3 tests/test_publish_versions_deep_lst.py

1. The work directory: work_path / work_root; writable(U) of a legacy run is U (no lock without a versions root), of a pointer
   its work directory (its writer lock taken and kept), of a pointer without one a refusal; writer_out leaves any other
   directory unchanged.
2. clone and same_content: the clone holds the run (copy-on-write where the file system makes them, else a full copy under
   serve.RUN_VERSIONS_COPY_MAX_BYTES, a larger run refused with nothing copied); a write to the clone never reaches the source;
   same_content passes a copy and fails a changed content of equal size, an added file, a changed link; the review state is
   left out.
3. snapshot and discard: the snapshot waits for the writer lock another holder has; a failed content test discards it;
   discard removes an unsealed version (logged), refuses a served one.
4. Conversion (switch with to_work): the legacy run is the work directory, U a pointer to its sealed clone; a failure to write
   the pointer moves it back; a work directory already there refuses with nothing moved.
5. stamp / pinned / moved: a switch during the block (also away and back) is seen, a log written into a legacy run is not;
   the pinned versions are held; journal_dirs; unshare_mutable.
6. The per-LST placement as a version: the first placement adopts the legacy run, the next one is a new version with the
   unchanged files hard-linked from the served one and the review state on its own inodes; the previous version is unchanged;
   a placement whose content test fails is discarded, U unchanged; PLACE_DELETE False refuses.
7. G30: the cumulative build refuses when an LST run was switched while the merge read it, not when nothing was switched (a
   log written in a legacy LST run included); the read versions are held during the merge and recorded.
8. The deep serve on a versioned run: convert (the journal lines written on the legacy run during the conversion caught up),
   a publish (snapshot of the work directory, the served journals copied in, the lines written on the old version during the
   publish caught up, the old version unchanged), a refused stage discards the snapshot, a switch of U by another serve during
   the publish refuses, a failed catch-up is owed and completed by the next serve, a partial stage selection refuses, convert
   twice is no-input, a parity convert refuses, retire runs for deep.
"""
import contextlib
import fcntl
import json
import os
import shutil
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
    check('writable of a legacy run without a versions root: U itself, no writer lock taken',
          versions.writable(U) == os.path.abspath(U) and not versions._WRITER, versions._WRITER)
    check('writer_out of a plain directory: unchanged', versions.writer_out(f'{d}/out') == f'{d}/out')
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


# ---- 3. snapshot and discard ------------------------------------------------------------------------------------------------
def test_snapshot():
    d = f'{TMP}/snap'
    U = legacy_run(f'{d}/Run_S')
    R = versions.root_of(U)
    os.makedirs(R)
    got, release = threading.Event(), threading.Event()

    def holder():                                      # another writer: its own open file, so the flock excludes
        fd = os.open(versions._writer_lock_file(os.path.realpath(R)), os.O_RDWR | os.O_CREAT)
        fcntl.flock(fd, fcntl.LOCK_EX)
        got.set()
        release.wait(5)
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    th = threading.Thread(target=holder)
    th.start()
    got.wait(5)
    threading.Timer(1.0, release.set).start()
    t0 = time.time()
    v, info = versions.snapshot(U, U, note='test')
    waited = time.time() - t0
    th.join(5)
    check('the snapshot waits for the writer lock another writer holds', waited >= 0.9, waited)
    check('the snapshot is an unsealed version holding the run', v.record() is None and versions.same_content(U, v.run)[0]
          and info['stats']['n_files'] == 3, (v.record(), info))
    versions.discard(v, note='test')
    log = [json.loads(l) for l in open(os.path.join(R, 'RETIRE_LOG.jsonl'))]
    check('discard removes the unsealed version and logs it', not os.path.exists(v.dir) and log[-1]['version'] == v.vid
          and log[-1]['discarded'], log[-1])
    with patch((versions, 'same_content', lambda a, b, **k: (False, 'stand-in difference'))):
        why = refuses(versions.snapshot, U, U, note='test')
    check('a snapshot whose content test fails is refused and discarded', why and 'stand-in difference' in why
          and not versions.all_versions(U), (why, versions.all_versions(U)))
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

    def place():
        run = Run('pages', ['test'], f'{d}/records', CFG, config_sections=None)
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
    with patch((CFG.pages_sublst, 'PLACE_DELETE', False)):
        u = place()
    check('PLACE_DELETE False refuses (a placement is a whole version)', u.status == 'refused'
          and 'PLACE_DELETE' in (u.reason or '') and versions.served(U).vid == v2.vid, (u.status, u.reason))


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
        # convert: the legacy run's journal line written during the conversion (here: while the page index is built) is
        # caught up into the version
        hooks['index'] = lambda: write(f'{real}/judgements_V/judgements.jsonl', '{"line": "during convert"}\n', mode='a')
        run = serve(['convert'])
        u = run.units[-1]
        v1 = versions.served(U)
        W = versions.work_path(U)
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
        calls.clear()
        hooks['views'] = lambda run, p: write(f'{v1.run}/judgements_V/judgements.jsonl', '{"line": "during publish"}\n',
                                              mode='a')
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
              v3.vid != v2.vid and owed and owed['from_version'] == v2.vid and owed['journals'] == ['judgements_V']
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
    finally:
        release_writers()
        shutil.rmtree(TMP, ignore_errors=True)
    print(f'{N[0] - len(FAILS)} of {N[0]} checks passed')
    return 1 if FAILS else 0


def test_all():
    assert main() == 0


if __name__ == '__main__':
    sys.exit(main())
