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
    (posix_acl_create); without a default ACL the permission bits of directory_mode (owner read and write; group read,
    and group write only when the directory grants it; other read only when the directory grants it - never the
    source's own mode).
It REFUSES (PlacementRefusal, a record.Refusal; nothing renamed, ``src`` left where it is) when the group cannot be set (the
process is not a member of the target group: chown EPERM - e.g. a rerun by a user that is not in the group of the product it
replaces), the ACL cannot be set or removed (not the owner, or a filesystem without ACLs), or the attributes read back differ.
The refusal names the product, its temporary file and the group. Under the parity guard both paths are checked before
anything changes.

``atomic(dst)`` is the writers' form: ``with placement.atomic(dst) as tmp:`` yields a unique temporary path beside ``dst``
(``<dst>.<pid>.<random>.tmp``, or ``<dst>.<tag>.<pid>.<random>.tmp`` with ``tag``; not created), the block writes it with
the caller's own writer (open + csv, json.dump, writeto, ...), and a clean exit places it (``replace(tmp, dst, drop=True)``);
ANY exception inside the block removes the temporary file and propagates. Only a killed process leaves one behind:
``temp_glob(dst, tag)`` is the glob of those names, for the cleanup of the writer's next run, and ``sweep(dsts)`` removes
them (the ones not modified for an hour) - each name is unique, so no later write replaces it.

``atomic_many(dsts)`` is the form for a set of products installed together: ``with placement.atomic_many(dsts) as tmps:``
yields {dst: tmp}; a clean exit first gives EVERY temporary file its product's attributes (apply: where a refusal happens),
and only when all succeeded renames them in the order of ``dsts``. An attribute refusal places none of them; a rename
that fails after others succeeded leaves those placed and refuses naming them and the one that failed (renames of several
files cannot be made atomic together).

``replace_many(pairs)`` installs a set the same way when the writer chose the files' names itself (an engine's output
directory, a worker's temporary directory, a temporary name of its own) in the products' file system: every file is given
its product's attributes where it lies, then each is renamed once to its product, in order - with the same two outcomes.
"""
import contextlib
import errno
import glob
import grp
import os
import re
import stat
import struct
import sys
import time

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


def directory_mode(dmode):
    """The permission bits of a new product in a directory without a default ACL: owner read and write; group read, and group
    write only when the directory grants it; other read only when the directory grants it. The source's mode is never kept
    (a copy2 or a scratch umask would otherwise carry group write or its absence into the archive)."""
    return 0o640 | (dmode & 0o020) | (dmode & 0o004)


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
    if dacl is None:                    # no default ACL: the mode from the directory's rules, never the source's
        return {'gid': gid, 'acl': None, 'mode': directory_mode(st.st_mode), 'source': 'directory'}
    acl, mode = inherited_acl(dacl)
    return {'gid': gid, 'acl': acl, 'mode': mode, 'source': 'directory'}


def _attrs(path):
    st = os.stat(path)
    return st.st_gid, get_acl(path), stat.S_IMODE(st.st_mode) & 0o777


def _group(gid):
    try:
        return f'{grp.getgrgid(gid).gr_name} ({gid})'
    except KeyError:
        return str(gid)


def apply(src, want, dst=None):
    """Give ``src`` the group, ACL and permission bits of ``want`` (target_attrs); PlacementRefusal when it cannot (the
    message names ``dst``, the product, when given)."""
    gid, acl, mode = _attrs(src)
    if want['source'] == 'directory' and want['acl'] is None and mode & 0o100:
        want = dict(want, mode=want['mode'] | ((want['mode'] & 0o444) >> 2))   # an executable stays executable (r -> rx)
    changed = False                             # FAST PATH: nothing set -> the values read above are the read-back
    try:
        if gid != want['gid']:                  # equal (a temporary file made in a set-group-ID directory): no chown
            os.chown(src, -1, want['gid'])
            changed = True
        if want['acl'] is not None:
            if acl != want['acl']:
                os.setxattr(src, ACL_ACCESS, want['acl'])
                changed = True
        else:
            if acl is not None:
                os.removexattr(src, ACL_ACCESS)
                changed = True
                mode = _attrs(src)[2]           # the mode bits removing the ACL left
            if want['mode'] is not None and mode != want['mode']:
                os.chmod(src, (os.stat(src).st_mode & ~0o777 & 0o7777) | want['mode'])
                changed = True
    except OSError as e:
        raise PlacementRefusal(f'placement: {dst or src} cannot be given the group {_group(want["gid"])} / ACL of the '
                               f'{"replaced product" if want["source"] == "existing" else "product"} ({want["source"]}): '
                               f'{e.strerror} (errno {e.errno}) - nothing renamed (temporary file {src})') from e
    got = _attrs(src) if changed else (gid, acl, mode)     # read again only after a change (never cached across calls)
    exp = (want['gid'], want['acl'], want['mode'] if want['mode'] is not None else got[2])
    if want['acl'] is not None:
        exp = (exp[0], exp[1], acl_mode(want['acl']))
    if got != exp:
        raise PlacementRefusal(f'placement: {dst or src} reads back group {got[0]} mode {got[2]:o} ACL '
                               f'{"set" if got[1] else "none"} after setting group {_group(exp[0])} mode {exp[2]:o} ACL '
                               f'{"set" if exp[1] else "none"} ({want["source"]}) - nothing renamed (temporary file {src})')
    return want


def replace(src, dst, like=None, drop=False):
    """os.replace(src, dst) after giving ``src`` the product's group and ACL (module docstring). Returns the attributes
    applied (target_attrs). ``drop``: a refused ``src`` (the caller's own temporary copy) is removed before the refusal is
    raised, so no stale temporary file is left beside the product."""
    from exopipe.phase3.core import guard
    if guard.active():
        guard.check(src, 'placement (source: group and ACL)')        # chown / setxattr reach the file itself
    try:
        if guard.active():
            guard.check(dst, 'placement (destination)', entry=True)
        want = apply(src, target_attrs(dst, like), dst=dst)
        os.replace(src, dst)
    except BaseException:
        if drop and os.path.lexists(src):       # any failure before the rename, wherever dst is: the temporary file goes
            try:
                os.remove(src)
            except OSError:
                pass                            # the original failure is the one raised
        raise
    return want


def _remove(path):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


_HEX8 = '[0-9a-f]' * 8


def temp_glob(dst, tag=None):
    """The glob of the temporary files ``atomic(dst, tag=tag)`` writes (``<dst>[.<tag>].<pid>.<8 hex digits>.tmp``): what a
    killed writer leaves, for its next run's cleanup. ``dst`` is a glob itself (glob.escape a literal path; ``<dir>/*`` for
    every file of a directory). Without ``tag`` it does not match a tagged name whose tag starts with a letter, and a tag's
    glob matches only that tag's names."""
    return f'{dst}.{tag + "." if tag else ""}[0-9]*.{_HEX8}.tmp'


STALE_TEMP_S = 3600                     # sweep: a temporary file not modified for an hour is a leftover


def sweep(dsts, tag=None, older_than_s=STALE_TEMP_S, log=None):
    """Remove the temporary files of the products ``dsts`` (literal paths) that a writer killed between writing and placing
    left behind: exactly the names atomic / atomic_many give them (``<dst>[.<tag>].<pid>.<8 hex digits>.tmp``), and only
    those not modified for ``older_than_s`` seconds (default STALE_TEMP_S, one hour). A younger one may be a concurrent
    writer's (the V and I searches of one cell write the same generation stamp) and stays; so do ``dst`` itself, another
    product's temporary files and every other name. A writer calls it before it writes (each temporary name is unique, so no
    later write replaces a leftover). Each removal is reported through ``log`` (default: a line on stderr). Returns the paths
    removed."""
    log = log or (lambda m: print(m, file=sys.stderr, flush=True))
    if isinstance(dsts, (str, bytes, os.PathLike)):
        dsts = [dsts]
    now, out = time.time(), []
    for dst in dsts:
        dst = os.fspath(dst)
        name = re.compile(re.escape(os.path.basename(dst)) + (re.escape('.' + tag) if tag else '') + r'\.\d+\.[0-9a-f]{8}\.tmp')
        for q in sorted(glob.glob(temp_glob(glob.escape(dst), tag))):
            if not name.fullmatch(os.path.basename(q)):
                continue                        # e.g. <dst>.1.<pid>.<hex>.tmp: the temporary file of the product <dst>.1
            try:
                st = os.lstat(q)
                if stat.S_ISDIR(st.st_mode) or now - st.st_mtime < older_than_s:
                    continue
                os.remove(q)
            except FileNotFoundError:          # removed meanwhile (placed by its writer, or another sweep)
                continue
            out.append(q)
            log(f'placement: removed {q}, a temporary file of {dst} left by a writer that did not finish (not modified for '
                f'{(now - st.st_mtime) / 3600:.1f} h)')
    return out


@contextlib.contextmanager
def atomic(dst, like=None, tag=None):
    """``with atomic(dst) as tmp:`` - write ``tmp`` with the caller's own writer; a clean exit places it at ``dst``
    (replace(tmp, dst, like, drop=True): the existing file's group, mode and ACL, or the directory's rule for a new file; a
    refusal removes ``tmp``). ANY exception inside the block removes ``tmp`` and propagates. ``tmp`` is
    ``<dst>.<pid>.<random>.tmp`` in ``dst``'s directory (unique per call, never created here: a writer that refuses an
    existing file works); ``tag`` (letters, digits, '_'; first a letter) is put before the pid, for writers that share a
    directory and clean up only their own temporary files (temp_glob(dst, tag)). A block that leaves no ``tmp`` REFUSES
    (nothing to place is never a silent success)."""
    tmp = _temp_name(dst, tag)                  # (a ValueError for a malformed tag, before anything is yielded)
    try:
        yield tmp
    except BaseException:
        _remove(tmp)
        raise
    if not os.path.lexists(tmp):
        raise PlacementRefusal(f'placement: {tmp} was not written - {dst} not placed')
    replace(tmp, dst, like=like, drop=True)


def _temp_name(dst, tag=None):
    if tag is not None and not (tag[:1].isalpha() and tag.replace('_', '').isalnum()):
        raise ValueError(f'placement.atomic: tag {tag!r} is not letters, digits and \'_\' starting with a letter')
    return f'{dst}.{tag + "." if tag else ""}{os.getpid()}.{os.urandom(4).hex()}.tmp'


@contextlib.contextmanager
def atomic_many(dsts):
    """``with atomic_many(dsts) as tmps:`` - {dst: tmp} for products installed together; the block writes every ``tmp`` with
    the caller's own writer. A clean exit works in two phases:
      1. every temporary file is given its product's attributes (target_attrs + apply, as replace: the existing file's group,
         mode and ACL, or the directory's rule for a new file). A refusal here, or a temporary file not written, removes
         EVERY temporary file and places nothing;
      2. only when all succeeded, each is renamed into place in the order of ``dsts``. This phase is NOT all-or-none: a
         rename that fails refuses (PlacementRefusal) naming the destinations already placed - they stay in place, live - and
         the one that failed; the temporary files not yet renamed are removed.
    ANY exception inside the block removes every temporary file and propagates. ``dsts`` is refused before anything is
    written when two different names in it are the same path (both renamed to one file: the second would silently replace
    the first); the same name given twice is one product. Temporary names are atomic's (temp_glob(dst) matches them)."""
    order, seen = [], {}
    for d in dsts:
        k = os.path.abspath(d)
        if k in seen:
            if seen[k] != d:
                raise PlacementRefusal(f'placement: {seen[k]} and {d} are the same destination - nothing written')
            continue
        seen[k] = d
        order.append(d)
    tmps = {d: _temp_name(d) for d in order}

    def drop_all(names):
        for d in names:
            _remove(tmps[d])
    try:
        yield dict(tmps)
    except BaseException:
        drop_all(order)
        raise
    from exopipe.phase3.core import guard
    try:
        for d in order:                                        # phase 1: every attribute applied before any rename
            t = tmps[d]
            if not os.path.lexists(t):
                raise PlacementRefusal(f'placement: {t} was not written ({d})')
            if guard.active():
                guard.check(t, 'placement (source: group and ACL)')
                guard.check(d, 'placement (destination)', entry=True)
            apply(t, target_attrs(d), dst=d)
    except BaseException as e:
        drop_all(order)
        if isinstance(e, PlacementRefusal):
            raise PlacementRefusal(f'{e} - none of the {len(order)} products placed') from e
        raise
    for i, d in enumerate(order):                              # phase 2: the renames, in order
        try:
            os.replace(tmps[d], d)
        except BaseException as e:
            drop_all(order[i:])
            if not isinstance(e, OSError):
                raise
            raise PlacementRefusal(f'placement: renaming {tmps[d]} to {d} failed: {e.strerror} (errno {e.errno}); already '
                                   f'placed: {order[:i] or "none"} (those stay live); not placed: {order[i:]}') from e


def replace_many(pairs):
    """[(src, dst)]: files the writer made under names of its own (an engine's or a worker's temporary directory, a temporary
    name beside the product) ON THE PRODUCTS' FILE SYSTEM, installed together with ONE rename each (no other temporary name
    is made beside a product):
      1. every ``src`` is given its product's attributes where it lies (target_attrs(dst) + apply with its read-back, as
         replace: the existing file's group, mode and ACL, or the directory's rule for a new file). A refusal here - an
         attribute that cannot be set or does not read back, a ``src`` missing, a ``src`` on another file system than its
         ``dst`` (os.replace cannot move it) - places NONE of them;
      2. only when all succeeded, each ``src`` is renamed to its ``dst`` (os.replace) in the order of ``pairs``. This phase is
         NOT all-or-none: a rename that fails refuses (PlacementRefusal) naming the products already placed - they stay in
         place, live - the one that failed and the ones not placed.
    Each ``src`` must be the caller's own temporary file (as replace's ``drop``): on ANY failure every ``src`` not placed is
    removed, so no temporary file is left in the writer's directory or beside a product. A ``dst`` or a ``src`` named twice
    refuses before anything is changed. Returns the destinations in order."""
    from exopipe.phase3.core import guard
    pairs = [(s, d) for s, d in pairs]
    dsts = [d for _s, d in pairs]

    def drop(rest):
        for s, _d in rest:
            _remove(s)
    try:
        if (len({os.path.abspath(d) for d in dsts}) != len(dsts)
                or len({os.path.abspath(s) for s, _d in pairs}) != len(pairs)):
            raise PlacementRefusal(f'placement: a destination or a source named twice in {pairs} - nothing renamed')
        for s, d in pairs:                                     # phase 1: every attribute applied, where the file lies
            if not os.path.lexists(s):
                raise PlacementRefusal(f'placement: {s} was not written ({d})')
            if os.stat(s).st_dev != os.stat(os.path.dirname(os.path.abspath(d))).st_dev:
                raise PlacementRefusal(f'placement: {s} is not on the file system of {d} (a rename cannot move it)')
            if guard.active():
                guard.check(s, 'placement (source: group and ACL)')
                guard.check(d, 'placement (destination)', entry=True)
            apply(s, target_attrs(d), dst=d)
    except BaseException as e:
        drop(pairs)
        if isinstance(e, PlacementRefusal):
            raise PlacementRefusal(f'{e} - none of the {len(pairs)} products placed') from e
        raise
    for i, (s, d) in enumerate(pairs):                         # phase 2: one rename per product, in order
        try:
            os.replace(s, d)
        except BaseException as e:
            drop(pairs[i:])
            if not isinstance(e, OSError):
                raise
            raise PlacementRefusal(f'placement: renaming {s} to {d} failed: {e.strerror} (errno {e.errno}); already placed: '
                                   f'{dsts[:i] or "none"} (those stay live); failed: {d}; not placed: {dsts[i:]}') from e
    return dsts
