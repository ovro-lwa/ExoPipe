"""A rename keeps the product's group and ACL (core/placement.py; FINDING-20261118-HYBRID-ROUTE-PRODUCTION-FAULTS item 8).
Data-free: scratch directories under TMPDIR with a set-group-ID bit, a second group of this user where it has one, and a
default ACL set through the extended attribute. A filesystem without POSIX ACLs prints UNTESTED for the ACL checks."""
import errno
import os
import shutil
import stat
import struct
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exopipe.phase3.core import placement as PL  # noqa: E402
from exopipe.phase3.core.record import Refusal  # noqa: E402

TMP = os.path.join(os.environ.get('TMPDIR', tempfile.gettempdir()), 'test_placement')
ACL_OK = None


def _acl(entries):
    """Raw ACL bytes (version 2) from (tag, perm, id) entries; id None = undefined."""
    return struct.pack('<I', 2) + b''.join(struct.pack('<HHI', t, p, 0xFFFFFFFF if i is None else i) for t, p, i in entries)


# the archive's default ACL: user::rwx, user:<me>:rwx, group::rwx, mask::rwx, other::r-x
DEFAULT = _acl([(0x01, 7, None), (0x02, 7, os.getuid()), (0x04, 7, None), (0x10, 7, None), (0x20, 5, None)])


def _other_group():
    """A group of this user other than its effective group (wheel on the cluster), else None."""
    g = [x for x in os.getgroups() if x != os.getegid()]
    return g[0] if g else None


def _archive_dir(name):
    """A directory like an archive Run: set-group-ID, a non-default group where possible, the default ACL."""
    global ACL_OK
    d = os.path.join(TMP, name)
    os.makedirs(d)
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


def _plain_dir(name):
    d = os.path.join(TMP, name)
    os.makedirs(d)
    os.chmod(d, 0o755)
    return d


def _write(path, text='x'):
    with open(path, 'w') as f:
        f.write(text)
    return path


def _attrs(path):
    st = os.stat(path)
    return st.st_gid, PL.get_acl(path), stat.S_IMODE(st.st_mode)


def _raises(exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc as e:
        return str(e)
    raise AssertionError(f'{fn.__name__} did not raise {exc.__name__}')


def _fresh():
    shutil.rmtree(TMP, ignore_errors=True)
    os.makedirs(TMP)


def test_inherited_acl():
    """inherited_acl reproduces what the kernel gives a file created in the directory (open(..., 'w'): mode 0o666)."""
    _fresh()
    d = _archive_dir('arch')
    if not ACL_OK:
        print('  UNTESTED: no POSIX ACLs on', TMP)
        return
    f = _write(os.path.join(d, 'sibling.fits'))
    acl, mode = PL.inherited_acl(DEFAULT)
    assert PL.get_acl(f) == acl and stat.S_IMODE(os.stat(f).st_mode) == mode == 0o664, (PL.get_acl(f), oct(mode))
    t = PL.target_attrs(os.path.join(d, 'new.fits'))
    assert (t['gid'], t['acl'], t['mode'], t['source']) == (os.stat(d).st_gid, acl, 0o664, 'directory'), t
    # a minimal default ACL gives no access ACL (the kernel stores the mode alone)
    acl2, mode2 = PL.inherited_acl(_acl([(0x01, 7, None), (0x04, 5, None), (0x20, 0, None)]))
    assert acl2 is None and mode2 == 0o640, (acl2, oct(mode2))


def test_rename_from_elsewhere():
    """A file written elsewhere and renamed in: os.replace keeps ITS group and ACL (the 2026-11-11 fault); placement.replace
    gives it the untouched sibling's."""
    _fresh()
    d, e = _archive_dir('arch'), _plain_dir('elsewhere')
    sib = _write(os.path.join(d, 'PB_deep_46MHz.fits'))
    a = _write(os.path.join(e, 'a.fits'))
    os.chmod(a, 0o600)
    os.replace(a, os.path.join(d, 'raw.fits'))
    raw = _attrs(os.path.join(d, 'raw.fits'))
    if _other_group() is not None:
        assert raw[0] != _attrs(sib)[0], 'plain rename kept the group?'
    if ACL_OK:
        assert raw[1] is None and _attrs(sib)[1] is not None
    b = _write(os.path.join(e, 'b.fits'))
    os.chmod(b, 0o600)
    w = PL.replace(b, os.path.join(d, 'filled.fits'))
    assert w['source'] == 'directory'
    got = _attrs(os.path.join(d, 'filled.fits'))
    if ACL_OK:
        assert got == _attrs(sib), (got, _attrs(sib))
    else:
        assert got[0] == _attrs(sib)[0], (got, _attrs(sib))
    assert not os.path.exists(b)


def test_replacement_keeps_existing():
    """Replacing a product keeps ITS group, ACL and mode (a named-user ACL it alone has, mode 0o640)."""
    _fresh()
    d, e = _archive_dir('arch'), _plain_dir('elsewhere')
    p = _write(os.path.join(d, 'master.fits'), 'old')
    if ACL_OK:
        own = _acl([(0x01, 6, None), (0x02, 6, os.getuid()), (0x04, 4, None), (0x10, 6, None), (0x20, 0, None)])
        os.setxattr(p, PL.ACL_ACCESS, own)
    else:
        os.chmod(p, 0o640)
    before = _attrs(p)
    n = _write(os.path.join(e, 'new.fits'), 'new')
    w = PL.replace(n, p)
    assert w['source'] == 'existing' and open(p).read() == 'new'
    assert _attrs(p) == before, (_attrs(p), before)
    # like= wins over the destination
    q = _write(os.path.join(e, 'q.fits'))
    w = PL.replace(q, os.path.join(d, 'other.fits'), like=p)
    assert w['source'] == 'like' and _attrs(os.path.join(d, 'other.fits')) == before


def test_refusals():
    """Nothing is renamed when the group or ACL cannot be given; drop=True removes the caller's temporary file."""
    _fresh()
    d, e = _archive_dir('arch'), _plain_dir('elsewhere')
    p = _write(os.path.join(d, 'product.fits'), 'old')
    assert issubclass(PL.PlacementRefusal, Refusal)
    # a group this user is not a member of (root's, read from a file it owns): chown EPERM
    if os.geteuid() != 0 and 0 not in os.getgroups() and os.stat('/').st_gid == 0:
        s = _write(os.path.join(e, 's.fits'), 'new')
        m = _raises(PL.PlacementRefusal, PL.replace, s, p, like='/')
        assert 'nothing renamed' in m and 'errno 1' in m, m
        assert open(p).read() == 'old' and os.path.exists(s)
        _raises(PL.PlacementRefusal, PL.replace, s, p, like='/', drop=True)
        assert not os.path.exists(s) and open(p).read() == 'old'
    else:
        print('  UNTESTED: running as root or in group 0 (chown EPERM cannot be provoked)')
    # an ACL that cannot be set (a filesystem without ACLs: ENOTSUP), and attributes that read back wrong
    if ACL_OK:
        s = _write(os.path.join(e, 's2.fits'), 'new')
        real = os.setxattr

        def nosup(*a, **k):
            raise OSError(errno.ENOTSUP, os.strerror(errno.ENOTSUP))
        os.setxattr = nosup
        try:
            m = _raises(PL.PlacementRefusal, PL.replace, s, os.path.join(d, 'x.fits'))
        finally:
            os.setxattr = real
        assert 'errno %d' % errno.ENOTSUP in m and not os.path.exists(os.path.join(d, 'x.fits')), m
        s = _write(os.path.join(e, 's3.fits'), 'new')
        real_chown = os.chown
        os.chown = lambda *a, **k: None
        try:
            if _other_group() is not None:
                m = _raises(PL.PlacementRefusal, PL.replace, s, os.path.join(d, 'x.fits'))
                assert 'reads back group' in m and not os.path.exists(os.path.join(d, 'x.fits')), m
        finally:
            os.chown = real_chown


def test_wired_helpers():
    """The product writers and placement helpers give the directory's (or the replaced product's) group and ACL: fitsio
    write_fits, archive verified_copy + commit_part, curves publish, hybrid write_atomic and coadd _atomic_copy."""
    from exopipe.phase3.core import fitsio
    from exopipe.phase3.lib.archive import common as AC
    from exopipe.phase3.lib.curves import publish as PUB
    from exopipe.phase3.lib.hybrid import common as HC
    from exopipe.phase3.lib.masters import coadd as CO
    _fresh()
    d, e = _archive_dir('arch'), _plain_dir('elsewhere')
    sib = _write(os.path.join(d, 'sibling.fits'))
    want = _attrs(sib)
    src = os.path.join(e, 'src.fits')
    fitsio.write_fits(src, np.zeros((4, 4), np.float32))
    os.chmod(src, 0o600)
    if ACL_OK:                                    # a source with its own ACL: copy2 would carry it into the archive
        os.setxattr(src, PL.ACL_ACCESS, _acl([(0x01, 6, None), (0x02, 4, 0), (0x04, 0, None), (0x10, 4, None),
                                              (0x20, 0, None)]))
    out = {}
    out['write_fits'] = fitsio.write_fits(os.path.join(d, 'w.fits'), np.zeros((4, 4), np.float32))
    part, _m = AC.verified_copy(src, os.path.join(d, 'a.fits'), preserve=True)
    AC.commit_part(part, os.path.join(d, 'a.fits'))
    out['archive commit_part'] = os.path.join(d, 'a.fits')
    PUB._atomic_copy(src, os.path.join(d, 'c.fits'))
    out['curves publish'] = os.path.join(d, 'c.fits')
    HC.write_atomic(os.path.join(d, 'h.fits'), np.zeros((4, 4), np.float32), None)
    out['hybrid write_atomic'] = os.path.join(d, 'h.fits')
    CO._atomic_copy(src, os.path.join(d, 'm.fits'))
    out['coadd _atomic_copy'] = os.path.join(d, 'm.fits')
    for k, p in out.items():
        got = _attrs(p)
        if ACL_OK:
            assert got == want, (k, got, want)
        else:
            assert got[0] == want[0], (k, got, want)
    # write_fits over an existing product keeps that product's mode (and ACL)
    os.chmod(out['write_fits'], 0o640) if not ACL_OK else os.setxattr(
        out['write_fits'], PL.ACL_ACCESS, _acl([(0x01, 6, None), (0x02, 6, os.getuid()), (0x04, 4, None), (0x10, 4, None),
                                                (0x20, 0, None)]))
    before = _attrs(out['write_fits'])
    fitsio.write_fits(out['write_fits'], np.ones((4, 4), np.float32), overwrite=True)
    assert _attrs(out['write_fits']) == before
    # a refused placement leaves no temporary file beside the product (drop=True)
    real = PL.apply

    def refuse(src_, want_):
        raise PL.PlacementRefusal('test')
    PL.apply = refuse
    try:
        _raises(PL.PlacementRefusal, fitsio.write_fits, os.path.join(d, 'r.fits'), np.zeros((2, 2), np.float32))
    finally:
        PL.apply = real
    assert not [n for n in os.listdir(d) if n.startswith('.r.fits')] and not os.path.exists(os.path.join(d, 'r.fits'))


TESTS = [test_inherited_acl, test_rename_from_elsewhere, test_replacement_keeps_existing, test_refusals, test_wired_helpers]


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
