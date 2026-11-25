"""placement.atomic, the writers' form of core/placement (PR 177 fix round): ``with placement.atomic(dst) as tmp:`` writes a
unique temporary file beside the product and places it on a clean exit. Data-free: scratch directories under TMPDIR, a
set-group-ID directory of a second group of this user where it has one, a default ACL where the filesystem has POSIX ACLs
(UNTESTED is printed otherwise). Checks: a replaced file keeps its group, mode and ACL; a new file gets the directory's rule;
the temporary file goes on a writer exception and on a refusal; the temporary names are unique (per call and per process);
a refusal names the product and the group and is recorded as a refusal by a unit."""
import errno
import os
import shutil
import stat
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exopipe.phase3.core import placement as PL  # noqa: E402
from exopipe.phase3.core import record  # noqa: E402

TMP = os.path.join(os.environ.get('TMPDIR', tempfile.gettempdir()), 'test_placement_atomic')
ACL_OK = None
ME = os.getuid()


def _acl(entries):
    """Raw ACL bytes (version 2) from (tag, perm, id) entries; id None = undefined."""
    return struct.pack('<I', 2) + b''.join(struct.pack('<HHI', t, p, 0xFFFFFFFF if i is None else i) for t, p, i in entries)


DEFAULT = _acl([(0x01, 7, None), (0x02, 7, ME), (0x04, 7, None), (0x10, 7, None), (0x20, 5, None)])
# a product's own access ACL: rw for its owner and a named user, r for the group (mask r), nothing for other (mode 0640)
OWN = _acl([(0x01, 6, None), (0x02, 4, ME), (0x04, 4, None), (0x10, 4, None), (0x20, 0, None)])


def _other_group():
    g = [x for x in os.getgroups() if x != os.getegid()]
    return g[0] if g else None


def _dir(name, archive):
    """archive: set-group-ID, a second group where possible, the default ACL; else a plain 0775 directory."""
    global ACL_OK
    d = os.path.join(TMP, name)
    os.makedirs(d)
    os.chmod(d, 0o775)
    if archive:
        g = _other_group()
        if g is not None:
            os.chown(d, -1, g)
        os.chmod(d, 0o2775)
        try:
            os.setxattr(d, PL.ACL_DEFAULT, DEFAULT)
            ACL_OK = True
        except OSError as e:
            if e.errno not in (errno.ENOTSUP, errno.EOPNOTSUPP):
                raise
            ACL_OK = False
    return d


def _attrs(path):
    st = os.stat(path)
    return st.st_gid, PL.get_acl(path), stat.S_IMODE(st.st_mode)


def _leftovers(d):
    return sorted(n for n in os.listdir(d) if n.endswith('.tmp'))


def _write(path, text):
    with open(path, 'w') as f:
        f.write(text)
    return path


def _raises(exc, fn):
    try:
        fn()
    except exc as e:
        return e
    raise AssertionError(f'did not raise {exc.__name__}')


def _fresh():
    shutil.rmtree(TMP, ignore_errors=True)
    os.makedirs(TMP)


def _product(d, name='master_catalog.csv'):
    """An existing product with attributes of its own: the second group where there is one, its own ACL (or mode 0640)."""
    p = _write(os.path.join(d, name), 'old\n')
    g = _other_group()
    if g is not None:
        os.chown(p, -1, g)
    if ACL_OK:
        os.setxattr(p, PL.ACL_ACCESS, OWN)
    else:
        os.chmod(p, 0o640)
    return p


def test_replaced_keeps_attrs():
    """A replaced file keeps its group, mode and ACL, in an archive-like and in a plain directory; the bytes are the writer's."""
    _fresh()
    for archive in (True, False):
        d = _dir(f'd{int(archive)}', archive)
        p = _product(d)
        before = _attrs(p)
        with PL.atomic(p) as tmp, open(tmp, 'w') as f:
            assert os.path.dirname(tmp) == d                                   # beside the product
            f.write('new\n')
        assert open(p).read() == 'new\n' and _attrs(p) == before, (archive, _attrs(p), before)
        assert not _leftovers(d), _leftovers(d)


def test_new_file_directory_rule():
    """A new file gets the directory's rule (target_attrs before the write): the set-group-ID directory's group and its
    default ACL masked by 0666; a plain directory's group-write and other-read bits, never the writer's umask."""
    _fresh()
    d = _dir('arch', True)
    p = os.path.join(d, 'new.csv')
    want = PL.target_attrs(p)
    with PL.atomic(p) as tmp, open(tmp, 'w') as f:
        f.write('x\n')
    got = _attrs(p)
    assert (got[0], got[1]) == (want['gid'], want['acl']) and got[2] == want['mode'], (got, want)
    if ACL_OK:
        assert got[1] is not None and got[0] == os.stat(d).st_gid
    old = os.umask(0o077)                           # a writer umask that would give 0600
    try:
        for dmode, mode in ((0o755, 0o644), (0o775, 0o664), (0o750, 0o640)):
            e = os.path.join(TMP, 'plain%o' % dmode)
            os.makedirs(e)
            os.chmod(e, dmode)
            with PL.atomic(os.path.join(e, 'n.csv')) as tmp, open(tmp, 'w') as f:
                f.write('x\n')
            assert _attrs(os.path.join(e, 'n.csv'))[2] == mode, (oct(dmode), oct(_attrs(os.path.join(e, 'n.csv'))[2]))
    finally:
        os.umask(old)


def test_writer_exception_removes_tmp():
    """ANY exception inside the block removes the temporary file and propagates; the product is untouched (or still absent)."""
    _fresh()
    d = _dir('arch', True)
    p = _product(d)
    before = (_attrs(p), open(p).read())
    seen = []

    def boom(exc):
        with PL.atomic(p) as tmp, open(tmp, 'w') as f:
            seen.append(tmp)
            f.write('half')
            f.flush()
            assert os.path.exists(tmp)
            raise exc

    for exc in (ValueError('writer failed'), record.Refusal('a gate inside the block'), KeyboardInterrupt()):
        _raises(type(exc), lambda: boom(exc))
        assert not os.path.exists(seen[-1]) and (_attrs(p), open(p).read()) == before and not _leftovers(d)
    q = os.path.join(d, 'absent.csv')

    def boom_new():
        with PL.atomic(q) as tmp, open(tmp, 'w') as f:
            f.write('half')
            raise OSError(errno.ENOSPC, 'no space')
    _raises(OSError, boom_new)
    assert not os.path.exists(q) and not _leftovers(d)
    # a block that writes nothing refuses (nothing placed is never a silent success)
    e = _raises(PL.PlacementRefusal, lambda: _exit_clean(q))
    assert 'was not written' in str(e) and not os.path.exists(q)


def _exit_clean(q):
    with PL.atomic(q):
        pass


def test_refusal_removes_tmp_names_group_and_file():
    """A refused placement (the group cannot be given: a user that is not in the replaced product's group, as a rerun by
    another account) raises PlacementRefusal - a record.Refusal - naming the product and the group; the temporary file is
    removed and the product keeps its bytes and attributes."""
    _fresh()
    d = _dir('arch', True)
    p = _product(d)
    before = (_attrs(p), open(p).read())
    foreign = next((g for g in (0, 1, 2, 3, 4, 65534) if g not in os.getgroups() and g != os.getegid()), None)
    if os.geteuid() == 0 or foreign is None:
        print('  UNTESTED: running as root or a member of every candidate group (chown EPERM cannot be provoked)')
        return
    real = PL.target_attrs
    PL.target_attrs = lambda dst, like=None: dict(real(dst, like), gid=foreign)   # the replaced product's group, not ours
    try:
        seen = []

        def write():
            with PL.atomic(p) as tmp, open(tmp, 'w') as f:
                seen.append(tmp)
                f.write('new\n')
        e = _raises(PL.PlacementRefusal, write)
    finally:
        PL.target_attrs = real
    m = str(e)
    assert isinstance(e, record.Refusal)
    assert p in m and f'({foreign})' in m and 'replaced product' in m and 'nothing renamed' in m, m
    assert 'errno %d' % errno.EPERM in m, m
    assert not os.path.exists(seen[0]) and (_attrs(p), open(p).read()) == before and not _leftovers(d)


def test_unique_names():
    """The temporary path is in dst's directory, carries the process id, and differs per call and per process."""
    _fresh()
    d = _dir('plain', False)
    p = os.path.join(d, 'x.csv')
    names = set()
    for _ in range(2000):
        g = PL.atomic(p)
        names.add(g.__enter__())
        try:
            g.__exit__(ValueError, ValueError('stop'), None)    # leaves nothing (the block "raised")
        except ValueError:
            pass
    assert len(names) == 2000, len(names)
    assert all(os.path.dirname(n) == d and n.startswith(p + f'.{os.getpid()}.') and n.endswith('.tmp') for n in names)
    env = dict(os.environ, PYTHONPATH=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    child = subprocess.run([sys.executable, '-c', 'import sys; from exopipe.phase3.core import placement as PL; '
                            'g = PL.atomic(sys.argv[1]); print(g.__enter__())', p], env=env, stdout=subprocess.PIPE,
                           universal_newlines=True, check=True).stdout.strip()
    assert child.startswith(p + '.') and child not in names and f'.{os.getpid()}.' not in child, child
    assert not _leftovers(d)


def test_refusal_recorded_by_unit():
    """A PlacementRefusal inside a unit is recorded as a refusal (status refused, the placement reason), not an error."""
    _fresh()
    d = _dir('plain', False)

    class _Cfg:                                     # the run record needs only a config hash here (data-free)
        site_sha1 = None
        overrides = ()

        def hash(self, sections=None):
            return 'test'
    run = record.Run('tplacement', ['test'], os.path.join(TMP, 'out'), _Cfg(), config_sections=())
    u = run.unit('atomic_refused')
    real = PL.apply

    def refuse(src, want, dst=None):
        raise PL.PlacementRefusal(f'placement: {dst} cannot be given the group test - nothing renamed (temporary file {src})')
    PL.apply = refuse
    try:
        with u:
            with PL.atomic(os.path.join(d, 'p.csv')) as tmp, open(tmp, 'w') as f:
                f.write('x\n')
    finally:
        PL.apply = real
    assert u.status == 'refused' and u.reason.startswith('placement:'), (u.status, u.reason)
    assert not os.path.exists(os.path.join(d, 'p.csv')) and not _leftovers(d)


def test_tag_and_temp_glob():
    """atomic(dst, tag=) puts the tag before the pid; temp_glob(dst, tag) matches exactly the names atomic writes for that tag
    (untagged: only untagged names; 'R0' never matches 'R075'), also as <dir>/* for every file of a directory; a killed
    writer's leftover is matched; a tag that is not a name refuses (ValueError)."""
    import glob
    import signal
    _fresh()
    d = _dir('plain', False)
    p = os.path.join(d, 'PB_deep_46MHz.fits')
    made, held = {}, []                     # held: an unreferenced generator is closed, and its exit removes the file
    for tag in (None, 'R075', 'R0'):
        held.append(PL.atomic(p, tag=tag))
        made[tag] = held[-1].__enter__()
        open(made[tag], 'w').close()                           # left as a killed writer would leave it
    assert made['R075'].startswith(p + f'.R075.{os.getpid()}.') and made[None].startswith(p + f'.{os.getpid()}.'), made
    e = glob.escape(p)
    assert glob.glob(PL.temp_glob(e)) == [made[None]], glob.glob(PL.temp_glob(e))
    for tag in ('R075', 'R0'):
        assert glob.glob(PL.temp_glob(e, tag=tag)) == [made[tag]], (tag, glob.glob(PL.temp_glob(e, tag=tag)))
        assert glob.glob(PL.temp_glob(os.path.join(glob.escape(d), '*'), tag=tag)) == [made[tag]]
    for bad in ('', '0R', 'R/0', 'R.0'):
        _raises(ValueError, lambda: PL.atomic(p, tag=bad).__enter__())
    for f_ in made.values():
        os.remove(f_)
    env = dict(os.environ, PYTHONPATH=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    code = ('import os, signal, sys; from exopipe.phase3.core import placement as PL\n'
            'with PL.atomic(sys.argv[1], tag="R075") as tmp:\n'
            '    open(tmp, "w").write("half")\n'
            '    os.kill(os.getpid(), signal.SIGKILL)\n')
    rc = subprocess.run([sys.executable, '-c', code, p], env=env).returncode
    left = glob.glob(PL.temp_glob(os.path.join(glob.escape(d), '*'), tag='R075'))
    assert rc == -signal.SIGKILL and len(left) == 1 and sorted(os.listdir(d)) == [os.path.basename(left[0])], (rc, left)
    assert not glob.glob(PL.temp_glob(os.path.join(glob.escape(d), '*'), tag='R0')) and not glob.glob(PL.temp_glob(e))


def test_atomic_write_json_place():
    """record.atomic_write_json(place=True) writes inside placement.atomic: the bytes of place=False, the replaced file's
    group, mode and ACL kept, a new file the directory's rule; a refusal leaves the product and no temporary file. The
    place=False default keeps its own temporary name (<path>.tmp.<pid>) and plain rename."""
    _fresh()
    d = _dir('arch', True)
    obj = {'b': [1, 2.5, None], 'a': 'x', 'when': 'now'}
    plain = os.path.join(TMP, 'plain.json')
    record.atomic_write_json(plain, obj)
    p = _product(d, 'stack.json')
    before = _attrs(p)
    record.atomic_write_json(p, obj, place=True)
    assert open(p, 'rb').read() == open(plain, 'rb').read() and _attrs(p) == before, (_attrs(p), before)
    q = os.path.join(d, 'new.json')
    want = PL.target_attrs(q)
    record.atomic_write_json(q, obj, place=True)
    got = _attrs(q)
    assert (got[0], got[1], got[2]) == (want['gid'], want['acl'], want['mode']), (got, want)
    assert not _leftovers(d)
    foreign = next((g for g in (0, 1, 2, 3, 4, 65534) if g not in os.getgroups() and g != os.getegid()), None)
    if os.geteuid() != 0 and foreign is not None:
        real = PL.target_attrs
        PL.target_attrs = lambda dst, like=None: dict(real(dst, like), gid=foreign)
        try:
            snap = (_attrs(p), open(p, 'rb').read(), os.stat(p).st_ino)
            _raises(PL.PlacementRefusal, lambda: record.atomic_write_json(p, {'other': 1}, place=True))
        finally:
            PL.target_attrs = real
        assert (_attrs(p), open(p, 'rb').read(), os.stat(p).st_ino) == snap and not _leftovers(d) \
            and sorted(os.listdir(d)) == ['new.json', 'stack.json'], os.listdir(d)
    else:
        print('  UNTESTED: the refusal of atomic_write_json(place=True) (root, or a member of every candidate group)')
    seen = []
    real_replace = os.replace
    os.replace = lambda a, b: (seen.append(a), real_replace(a, b))
    try:
        record.atomic_write_json(os.path.join(TMP, 'rec.json'), obj)
    finally:
        os.replace = real_replace
    assert seen == [os.path.join(TMP, 'rec.json') + f'.tmp.{os.getpid()}'], seen


TESTS = [test_replaced_keeps_attrs, test_new_file_directory_rule, test_writer_exception_removes_tmp,
         test_refusal_removes_tmp_names_group_and_file, test_unique_names, test_refusal_recorded_by_unit,
         test_tag_and_temp_glob, test_atomic_write_json_place]


def main():
    try:
        for t in TESTS:
            t()
            print(f'PASS {t.__name__}')
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    print(f'ALL {len(TESTS)} PASS (POSIX ACLs {"tested" if ACL_OK else "UNTESTED"}; second group '
          f'{_other_group() if _other_group() is not None else "none: group change UNTESTED"})')


if __name__ == '__main__':
    main()
