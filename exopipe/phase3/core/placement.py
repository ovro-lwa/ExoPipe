"""Placing a file by rename keeps the product's group and ACL (FINDING-20261118-HYBRID-ROUTE-PRODUCTION-FAULTS item 8).

A rename keeps the inode's own group, mode and access ACL. A file written elsewhere (node-local scratch, a staging directory
without the archive's set-group-ID bit and default ACL) or given another ACL by a copy (shutil.copy2 copies the source's
extended attributes and mode) carries them into the archive: on 2026-11-11 two filled masters renamed into 18h 46 MHz became
gh:gh, mode 664, with no ACL, while the untouched files beside them are gh:pipeline with the named-user ACL, so the pipeline
user can no longer rewrite them in place.

``replace(src, dst)`` is os.replace that first gives ``src`` the attributes of the product it becomes - the group, the access
ACL and, with them, the permission bits:
  - of ``like`` when given;
  - else of the existing ``dst`` (a replacement keeps the product's group and ACL);
  - else what the destination directory gives a new file: its group when the directory has the set-group-ID bit (otherwise
    the process's effective group), and its default ACL masked by the creation mode 0o666, as the kernel derives it
    (posix_acl_create); without a default ACL the permission bits are left as written.
It REFUSES (PlacementRefusal, a record.Refusal; nothing renamed, ``src`` left where it is) when the group cannot be set (the
process is not a member of the target group: chown EPERM), the ACL cannot be set or removed (not the owner, or a filesystem
without ACLs), or the attributes read back differ. Under the parity guard both paths are checked before anything changes.
"""
import errno
import os
import stat
import struct

from exopipe.phase3.core.record import Refusal

ACL_ACCESS = 'system.posix_acl_access'
ACL_DEFAULT = 'system.posix_acl_default'
CREATE_MODE = 0o666                     # the mode open() creates a file with (Python's open, astropy's writeto)
_USER_OBJ, _GROUP_OBJ, _MASK, _OTHER = 0x01, 0x04, 0x10, 0x20
_NO_ACL = (errno.ENODATA, errno.ENOTSUP, errno.EOPNOTSUPP)


class PlacementRefusal(Refusal):
    """The file could not be given the product's group or ACL; nothing was renamed."""


def get_acl(path, name=ACL_ACCESS):
    """The raw POSIX ACL extended attribute of ``path`` (bytes), or None when it has none (or the filesystem has no ACLs)."""
    try:
        return os.getxattr(path, name)
    except OSError as e:
        if e.errno in _NO_ACL:
            return None
        raise


def _entries(acl):
    body = acl[4:]
    return [struct.unpack('<HHI', body[i:i + 8]) for i in range(0, len(body), 8)]


def acl_mode(acl):
    """The permission bits an access ACL implies (user = USER_OBJ, group = MASK or else GROUP_OBJ, other = OTHER)."""
    e = {t: p for t, p, _ in _entries(acl) if t in (_USER_OBJ, _GROUP_OBJ, _MASK, _OTHER)}
    return (e.get(_USER_OBJ, 0) << 6) | (e.get(_MASK, e.get(_GROUP_OBJ, 0)) << 3) | e.get(_OTHER, 0)


def inherited_acl(default_acl, mode=CREATE_MODE):
    """The access ACL a new file created with ``mode`` gets from a directory's default ACL (the kernel's posix_acl_create:
    USER_OBJ, OTHER and MASK - GROUP_OBJ when there is no MASK - are masked by the mode's bits). Returns (acl bytes or None,
    permission bits): None when the result is a minimal ACL (only USER_OBJ, GROUP_OBJ and OTHER), which the kernel stores as
    the mode alone."""
    ents = _entries(default_acl)
    has_mask = any(t == _MASK for t, _, _ in ents)
    out = []
    for t, p, i in ents:
        if t == _USER_OBJ:
            p &= (mode >> 6) & 7
        elif t == _OTHER:
            p &= mode & 7
        elif t == _MASK or (t == _GROUP_OBJ and not has_mask):
            p &= (mode >> 3) & 7
        out.append((t, p, i))
    acl = default_acl[:4] + b''.join(struct.pack('<HHI', *e) for e in out)
    if all(t in (_USER_OBJ, _GROUP_OBJ, _OTHER) for t, _, _ in out):
        return None, acl_mode(acl)
    return acl, acl_mode(acl)


def target_attrs(dst, like=None):
    """{'gid', 'acl' (bytes or None), 'mode' (permission bits, or None = leave as written), 'source'} the file placed at
    ``dst`` must carry (module docstring)."""
    ref = like if like is not None else (dst if os.path.lexists(dst) else None)
    if ref is not None:
        st = os.stat(ref)
        return {'gid': st.st_gid, 'acl': get_acl(ref), 'mode': stat.S_IMODE(st.st_mode) & 0o777,
                'source': 'like' if like is not None else 'existing'}
    d = os.path.dirname(os.path.abspath(dst))
    st = os.stat(d)
    gid = st.st_gid if st.st_mode & stat.S_ISGID else os.getegid()
    dacl = get_acl(d, ACL_DEFAULT)
    if dacl is None:
        return {'gid': gid, 'acl': None, 'mode': None, 'source': 'directory'}
    acl, mode = inherited_acl(dacl)
    return {'gid': gid, 'acl': acl, 'mode': mode, 'source': 'directory'}


def _attrs(path):
    st = os.stat(path)
    return st.st_gid, get_acl(path), stat.S_IMODE(st.st_mode) & 0o777


def apply(src, want):
    """Give ``src`` the group, ACL and permission bits of ``want`` (target_attrs); PlacementRefusal when it cannot."""
    gid, acl, mode = _attrs(src)
    try:
        if gid != want['gid']:
            os.chown(src, -1, want['gid'])
        if want['acl'] is not None:
            if acl != want['acl']:
                os.setxattr(src, ACL_ACCESS, want['acl'])
        else:
            if acl is not None:
                os.removexattr(src, ACL_ACCESS)
            if want['mode'] is not None and (_attrs(src)[2] != want['mode']):
                os.chmod(src, (os.stat(src).st_mode & ~0o777 & 0o7777) | want['mode'])
    except OSError as e:
        raise PlacementRefusal(f'placement: {src} cannot be given the group {want["gid"]} / ACL of the product '
                               f'({want["source"]}): {e.strerror} (errno {e.errno}) - nothing renamed') from e
    got = _attrs(src)
    exp = (want['gid'], want['acl'], want['mode'] if want['mode'] is not None else got[2])
    if want['acl'] is not None:
        exp = (exp[0], exp[1], acl_mode(want['acl']))
    if got != exp:
        raise PlacementRefusal(f'placement: {src} reads back group {got[0]} mode {got[2]:o} ACL '
                               f'{"set" if got[1] else "none"} after setting group {exp[0]} mode {exp[2]:o} ACL '
                               f'{"set" if exp[1] else "none"} ({want["source"]}) - nothing renamed')
    return want


def replace(src, dst, like=None, drop=False):
    """os.replace(src, dst) after giving ``src`` the product's group and ACL (module docstring). Returns the attributes
    applied (target_attrs). ``drop``: a refused ``src`` (the caller's own temporary copy) is removed before the refusal is
    raised, so no stale temporary file is left beside the product."""
    from exopipe.phase3.core import guard
    if guard.active():
        guard.check(src, 'placement (source: group and ACL)')        # chown / setxattr reach the file itself
        guard.check(dst, 'placement (destination)', entry=True)
    try:
        want = apply(src, target_attrs(dst, like))
    except PlacementRefusal:
        if drop:
            os.remove(src)
        raise
    os.replace(src, dst)
    return want
