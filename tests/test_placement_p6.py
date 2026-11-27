"""Placement batch P6 (core/placement in the archive writers and per-file copies: the snapshot flow ship, the snapshot cube
ship, the flow movies, the per-night admitted products, the snapshot photometry ship, the deep search's archive frames, the deep
attempt publish, the snapshot search ship and the sub-LST search's copies): every copy is checked against its source (size and
md5) BEFORE it is placed, and placed with the destination's attributes, never the work file's; what a copy that differs or a
placement refusal does at each handler. Data-free: scratch directories under TMPDIR; a placement refusal is forced by giving
the placement a group this user is not in (UNTESTED is printed when there is none, e.g. as root); a copy that differs by
shadowing core/placement's open() for named sources. A file mode expected from the directory rule is read from
placement.target_attrs for a new name in that directory (the directory's own mode follows the umask when os.makedirs made
it): the tests hold under umask 022 and 002. Checks:
  copy     placement.copy / copy_verified: a new file takes the directory's rule (group, mode, inherited default ACL where
           setfacl works), never the source's mode 0600; a replaced file keeps its group, mode and ACL; times=True gives the
           source's times; a copy with other bytes of the same size (md5), a copy cut short after it is written (size) and a
           short read of the source each refuse (CopyNotVerified, with the copy's and the source's sizes and md5s) with the
           destination unchanged and no temporary file; a source that grows while it is read is copied as read (the copy
           checked against the bytes read, the source read once); an attribute refusal is a PlacementRefusal that is not
           CopyNotVerified, same outcome;
  sweep    one directory listing however many products of one directory are swept; the removed files are those the
           glob-based sweep of P5 removed (names with glob characters, tagged names, the <dst>.1 product's names); a
           temporary file is aged by its ctime: a copy's temporary file carrying its source's two-hour-old mtime survives;
  tree     placement.copy_tree (deep_step._copy_into and tier_sublst.copy_tree of a directory): every file verified and
           placed by its directory's rule with its source's times, every new directory by its parent's rule, links kept as
           links; a copy that differs removes what the copy made (the partial tree, new parents) and refuses; a merge keeps
           other files and a replaced file's mode, and its refusal names the files already replaced;
  convert  archive verified_copy, coadd _atomic_copy and curves publish _atomic_copy through placement: a copy that
           differs is refused with nothing placed and no temporary or fixed-name file;
  flow     flow ship_cell (G-S9): a copy that differs refuses the unit naming the file, the files shipped before it (in place)
           and the files after it (not copied); an attribute refusal the same with 'not placed'; no temporary file;
  cube     frames stage_cube's ship block (N-G8): an earlier MANIFEST.txt is removed before any cube is placed; a cube whose
           copy differs is not placed and the unit refuses naming it (its copy and source sizes and md5s recorded), the other
           cubes placed, no MANIFEST.txt; a placement refusal refuses naming what is placed and what is not copied; an
           existing cube keeps its mode; the logs and MANIFEST.txt are placed after every cube, MANIFEST.txt last, and a
           refusal of either leaves no MANIFEST.txt;
  movies   frames flow_movies: a subband whose copy differs or whose placement is refused is an item refusal (N-G10 flow
           movie ship) with none of its four files placed; the other subband's files placed with the directory's rule;
  admit    masters admit_stage's per-night write: a placement refusal refuses the night by name (gate admit.placement) with
           none of its products placed and no admissions row reached; a killed write's temporary file is swept;
  phot     photometry place_copies: the npz and the manifest placed together; a copy that differs places none; two pairs
           with one destination refuse before anything is written; measure_cell's handler refuses the unit 'SM-SHIP: ...
           (work kept: ...)';
  deep     the deep search's _write_archive_frame (a header that fails validation or a placement refusal leaves the frame
           as it was, no temporary file) and _archive_frame_publish (the archive frame takes the archive directory's mode,
           never the work frame's 0600, with the work frame's times and pixels; since 2026-11-27 it is written - the
           superseded frame's header updated with the search's cards, validated - not copied: a header that fails validation
           or a placement refusal leaves the earlier archive frame); compiled alone (the module needs the cluster's data to
           import);
  publish  deep_step stage_publish: a file of the attempt not placed removes <dest>.part and publishes nothing, the refusal
           naming what was copied; the side outputs are placed BEFORE the attempt is published, and one that is not placed
           publishes nothing; an attempt already published refuses (a Refusal, never a RuntimeError), naming a missing
           register row and how to recover;
  ship     the snapshot search ship: a copy that differs places none of the files and refuses the unit, the replaced file
           kept under _superseded and the destination unchanged; a successful --fresh ship keeps a replaced file's mode;
  sublst   the sub-LST search: copy_tree places a file by the directory's rule and refuses a copy that differs;
           seed_attempt retries a copy that differs once and refuses the second; publish rolls back on a placement refusal
           (replaced files restored, new ones removed) and says so, or says that the rollback failed."""
import ast
import builtins
import hashlib
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import textwrap
import time
import types

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')

import numpy as np  # noqa: E402
from astropy.io import fits  # noqa: E402

from exopipe.phase3.core import placement as PL  # noqa: E402
from exopipe.phase3.core import record  # noqa: E402

TMP = os.path.join(os.environ.get('TMPDIR', tempfile.gettempdir()), 'test_placement_p6')
REAL_CHANGED = PL._changed              # sweep ages by ctime; a planted leftover is aged by utime (mtime) - see main()
FOREIGN = next((g for g in (0, 1, 2, 3, 4, 65534) if g not in os.getgroups() and g != os.getegid()), None)
CAN_REFUSE = FOREIGN is not None and os.geteuid() != 0
OLD = (1700000000, 1700000123)          # (atime, mtime) of the sources: a copy with times=True carries them


def _fresh(name, mode=0o2775):
    d = os.path.join(TMP, name)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    os.chmod(d, mode)                   # set-group-ID, group-writable: the directory rule gives a new file 0664
    return d


def _write(path, data, mode=0o600, times=OLD):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        f.write(data)
    os.chmod(path, mode)
    if times:
        os.utime(path, times)
    return path


def _rule_mode(d):
    """The mode core/placement's directory rule gives a new file in the directory ``d`` (target_attrs of a new name there):
    it follows ``d``'s own mode, which follows the umask when os.makedirs made ``d``."""
    a = PL.target_attrs(os.path.join(d, '.p6-rule-probe'))
    assert a['source'] == 'directory' and a['mode'] != 0o600, a
    return a['mode']


def _bytes(n, seed):
    return np.random.RandomState(seed).bytes(n)


def _md5(p):
    return hashlib.md5(open(p, 'rb').read()).hexdigest()


def _leftovers(d):
    out = []
    for dp, _dn, fn in os.walk(d):
        out += [os.path.join(dp, f) for f in fn if f.endswith('.tmp') or f.endswith('.part') or '.tmp.' in f]
    return sorted(out)


def _state(d):
    """{relative path: (bytes md5, mode)} of every file under ``d``."""
    out = {}
    for dp, _dn, fn in os.walk(d):
        for f in fn:
            p = os.path.join(dp, f)
            out[os.path.relpath(p, d)] = (_md5(p), stat.S_IMODE(os.stat(p).st_mode))
    return out


class _Cfg:                                     # the run record needs only a config hash here (data-free)
    site_sha1 = None
    overrides = ()

    def hash(self, sections=None):
        return 'test'


def _run(d, step='test'):
    return record.Run(step, ['test'], os.path.join(d, 'out'), _Cfg(), config_sections=())


class _refuse_for:
    """Within the block, core/placement gives any destination whose name ends with ``suffix`` a group this user is not in, so
    its placement refuses (PlacementRefusal) after the temporary file is written."""

    def __init__(self, suffix):
        self.suffix, self.real = suffix, PL.target_attrs

    def __enter__(self):
        real, suffix = self.real, self.suffix
        PL.target_attrs = lambda dst, like=None: (dict(real(dst, like), gid=FOREIGN) if str(dst).endswith(suffix)
                                                  else real(dst, like))
        return self

    def __exit__(self, *a):
        PL.target_attrs = self.real
        return False


class _differs_for:
    """Within the block, core/placement's copy (copy_verified) of a source whose path ends with ``suffix`` differs from it,
    for the first ``n`` such copies (every one when None): how='flip' the first byte written to the copy changed (the same
    size: only the md5 sees it), how='cut' the copy's last 100 bytes dropped after it is written (the size sees it),
    how='short' the read of the source ends at half the file with no error (fewer bytes read than the source holds),
    how='grow' 5000 bytes appended to the source after its first block is read (a log being flushed). It shadows the
    open() of core/placement's module only."""

    def __init__(self, suffix, how='flip', n=None):
        self.suffix, self.how, self.n, self.hits = suffix, how, n, 0

    def __enter__(self):
        me, pending = self, []

        def opener(path, mode='r', *a, **kw):
            f = builtins.open(path, mode, *a, **kw)
            if mode == 'rb' and os.fspath(path).endswith(me.suffix) and (me.n is None or me.hits < me.n):
                me.hits += 1
                if me.how in ('short', 'grow'):
                    return _Read(f, me.how)
                pending.append(me.how)
            elif mode == 'xb' and pending:
                return _Write(f, pending.pop())
            return f
        PL.open = opener
        return self

    def __exit__(self, *a):
        del PL.open
        return False


class _Read:
    def __init__(self, f, how):
        self.f, self.how, self.reads = f, how, 0
        self.left = max(1, os.fstat(f.fileno()).st_size // 2) if how == 'short' else None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.f.close()
        return False

    def fileno(self):
        return self.f.fileno()

    def read(self, k=-1):
        if self.how == 'short':
            b = self.f.read(min(k, self.left))
            self.left -= len(b)
            return b
        b = self.f.read(k)
        self.reads += 1
        if self.reads == 1:
            with builtins.open(self.f.name, 'ab') as g:
                g.write(_bytes(5000, 99))
        return b


class _Write:
    def __init__(self, f, how):
        self.f, self.how, self.first = f, how, True

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.f.close()
        if self.how == 'cut':
            os.truncate(self.f.name, max(0, os.path.getsize(self.f.name) - 100))
        return False

    def fileno(self):
        return self.f.fileno()

    def write(self, b):
        if self.how == 'flip' and self.first and b:
            b = bytes([b[0] ^ 0xFF]) + bytes(b[1:])
        self.first = False
        return self.f.write(b)


def _block(rel, start, end, loop=False):
    """The statements of exopipe/phase3/<rel> from the line ``start`` up to (not including) ``end``, compiled on their own
    (each anchor must occur exactly once). loop=True runs them as the body of a one-pass loop (a ``continue`` in them ends
    it) followed by ``reached.append(1)``."""
    src = open(os.path.join(REPO, 'exopipe', 'phase3', rel)).read()
    assert src.count(start) == 1 and src.count(end) == 1, (rel, src.count(start), src.count(end))
    i = src.index(start)
    code = textwrap.dedent(src[i:src.index(end, i)])
    if loop:
        code = 'for _once in (0,):\n' + textwrap.indent(code, '    ') + '    reached.append(1)\n'
    return compile(code, f'{rel}:block', 'exec')


def _want_new(dst):
    a = PL.target_attrs(dst)
    assert a['source'] == 'directory', a
    return a['gid'], a['acl'], a['mode']


def test_copy():
    d = _fresh('copy')
    src = _write(os.path.join(d, 'work', 'a.fits'), _bytes(1 << 20, 1))
    out = os.path.join(d, 'out')
    os.makedirs(out)
    acl_dir = subprocess.run(['setfacl', '-d', '-m', f'u::rwx,g::rwx,o::r-x,u:{os.geteuid()}:rwx,m::rwx', out],
                             capture_output=True).returncode == 0 if shutil.which('setfacl') else False
    dst = os.path.join(out, 'a.fits')
    want = _want_new(dst)
    m = PL.copy(src, dst, times=True)
    assert m == _md5(src) == _md5(dst), (m, _md5(src), _md5(dst))
    assert PL._attrs(dst) == want and want[2] != 0o600, (PL._attrs(dst), want)
    assert os.stat(dst).st_mtime_ns == os.stat(src).st_mtime_ns == OLD[1] * 10 ** 9, os.stat(dst).st_mtime
    if acl_dir:
        assert want[1] is not None and PL.get_acl(dst) == want[1] and PL.get_acl(src) != want[1]
    else:
        print('UNTESTED copy: the inherited default ACL (setfacl did not set one under TMPDIR)')
    os.chmod(dst, 0o640)                                    # an existing product: its group, mode and ACL stay
    before = PL._attrs(dst)
    _write(src, _bytes(1 << 20, 2))
    PL.copy(src, dst)
    assert _md5(dst) == _md5(src) and PL._attrs(dst) == before, (PL._attrs(dst), before)
    assert abs(os.stat(dst).st_mtime - time.time()) < 600, 'times=False: the copy\'s own time'
    _write(src, _bytes(1 << 20, 3))
    for how in ('flip', 'cut', 'short'):
        keep = _state(out)
        with _differs_for('work/a.fits', how=how) as h:
            try:
                PL.copy(src, dst, times=True)
                raise AssertionError(f'{how}: a copy that differs was placed')
            except PL.CopyNotVerified as e:
                msg, got = str(e), (e.src_size, e.src_md5, e.copy_size, e.copy_md5)
        assert h.hits == 1, (how, h.hits)
        assert 'is not verified' in msg and 'nothing placed' in msg, msg
        assert ('(a short read)' in msg) == (how == 'short'), msg
        assert got[:2] == (os.path.getsize(src), _md5(src)) and got[2] and got[3], got
        assert _state(out) == keep and not _leftovers(d), (how, _state(out), _leftovers(d))
    if CAN_REFUSE:
        keep = _state(out)
        with _refuse_for('out/a.fits'):
            try:
                PL.copy(src, dst)
                raise AssertionError('an attribute refusal was placed')
            except PL.PlacementRefusal as e:
                assert not isinstance(e, PL.CopyNotVerified), e
        assert _state(out) == keep and not _leftovers(d)
    else:
        print('UNTESTED copy: an attribute refusal (no group this user is not in, or root)')
    _write(src, _bytes(1 << 20, 4))
    with _differs_for('work/a.fits', how='grow') as h:     # a source that grows while it is read: copied as read
        m = PL.copy(src, dst, times=True)
    assert h.hits == 1 and os.path.getsize(src) == (1 << 20) + 5000, os.path.getsize(src)
    assert m == _md5(dst) == _md5(src) and not _leftovers(d), (m, _md5(dst), _md5(src))
    print('ok copy: a new file takes the directory\'s rule (not the source\'s 0600), a replaced one keeps its attributes, '
          'times=True the source\'s times; a copy with other bytes (md5), cut short (size) or from a short read refuses '
          '(CopyNotVerified, with both sizes and md5s) with the destination unchanged and no temporary file; an attribute '
          'refusal likewise; a source that grows while it is read is copied as read')


def test_sweep_ctime():
    PL._changed = REAL_CHANGED
    try:
        d = _fresh('sweep_ctime')
        src = _write(os.path.join(d, 'work', 'a.log'), _bytes(5000, 5), times=(time.time() - 7200,) * 2)
        dst = os.path.join(d, 'a.log')
        planted = _write(dst + '.4194301.0badf00d.tmp', b'young\n', times=(time.time() - 7200,) * 2)
        with PL.atomic(dst) as tmp:
            PL.copy_verified(src, tmp, times=True)
            assert os.stat(tmp).st_mtime < time.time() - 7000, 'the temporary file carries its source\'s mtime'
            assert PL.sweep([dst], log=lambda m: None) == [], 'a sweep removed a temporary file changed just now'
            assert os.path.exists(tmp) and os.path.exists(planted)
        assert _md5(dst) == _md5(src) and os.stat(dst).st_mtime_ns == os.stat(src).st_mtime_ns
        PL._changed = lambda st: st.st_mtime           # the seam the planted-leftover tests use
        assert PL.sweep([dst], log=lambda m: None) == [planted]
    finally:
        PL._changed = lambda st: st.st_mtime
    print('ok sweep ctime: a temporary file with a two-hour-old mtime (a copy with times=True of an old source, and a '
          'planted one) survives a sweep - it is aged by its ctime, which copystat and utime cannot set back')


def _old_sweep_candidates(dst, tag=None):
    """The names P5's sweep (glob + fullmatch) removed for ``dst`` when every one is old enough."""
    import glob
    import re
    name = re.compile(re.escape(os.path.basename(dst)) + (re.escape('.' + tag) if tag else '') + r'\.\d+\.[0-9a-f]{8}\.tmp')
    return [q for q in sorted(glob.glob(PL.temp_glob(glob.escape(dst), tag))) if name.fullmatch(os.path.basename(q))]


def test_sweep_listing():
    d = _fresh('sweep')
    prods = [os.path.join(d, n) for n in ('a.csv', 'b[1].fits', 'a.csv.1', 'c.json')]
    names = []
    for p in prods:
        b = os.path.basename(p)
        names += [f'{b}.4194301.0badf00d.tmp', f'{b}.v1.4194302.0badf00e.tmp', f'{b}.12.0badf00d.tmp.4194303.0badf00f.tmp',
                  f'{b}.4194304.0BADF00D.tmp', f'{b}.x4194305.0badf010.tmp', f'{b}.4194306.0badf01.tmp', b]
    names += ['zz.4194307.0badf011.tmp', 'a.csvx.4194308.0badf012.tmp']
    for tag in (None, 'v1'):
        for n in names:
            _write(os.path.join(d, n), b'x\n', times=(time.time() - 7200,) * 2)
        want = [q for p in prods for q in _old_sweep_candidates(p, tag)]
        calls = []
        real = os.listdir
        os.listdir = lambda *a: calls.append(a) or real(*a)
        try:
            got = PL.sweep(prods + prods[:1], tag=tag, log=lambda m: None)
        finally:
            os.listdir = real
        assert len(calls) == 1, calls
        assert got == want and want, (tag, got, want)
        assert sorted(os.listdir(d)) == sorted(set(names) - {os.path.basename(q) for q in want}), os.listdir(d)
    print('ok sweep: one directory listing for four products (one named twice); the files removed are exactly those of the '
          'glob-based sweep, in its order, untagged and tagged')


def _tree(root):
    """A source tree: three files (mode 0600, the old times) in nested directories and a link to a file."""
    for rel, seed in (('a.bin', 1), ('sub/b.bin', 2), ('sub/deep/c.bin', 3)):
        _write(os.path.join(root, rel), _bytes(3000 + seed, 50 + seed))
    os.symlink('a.bin', os.path.join(root, 'link'))
    return root


def test_copy_tree():
    from exopipe.phase3.lib.search import deep_step as DS, tier_sublst as ST
    d = _fresh('tree')
    src = _tree(os.path.join(d, 'src'))
    files = ['a.bin', 'sub/b.bin', 'sub/deep/c.bin']
    dst = os.path.join(d, 'pub', 'Run_L')
    got = PL.copy_tree(src, dst, times=True)
    assert sorted(os.path.relpath(p, dst) for p in got) == files, got
    for rel in files:
        a, b = os.path.join(src, rel), os.path.join(dst, rel)
        assert got[b] == _md5(a) == _md5(b) and os.stat(b).st_mtime_ns == OLD[1] * 10 ** 9, rel
        assert stat.S_IMODE(os.stat(b).st_mode) == _rule_mode(os.path.dirname(b)), (rel, oct(os.stat(b).st_mode))
    assert os.path.islink(os.path.join(dst, 'link')) and os.readlink(os.path.join(dst, 'link')) == 'a.bin'
    acl = PL.get_acl(d, PL.ACL_DEFAULT) is not None
    for nd in ('pub', 'pub/Run_L', 'pub/Run_L/sub', 'pub/Run_L/sub/deep'):
        q = os.path.join(d, nd)
        st, pst = os.stat(q), os.stat(os.path.dirname(q))
        assert st.st_mode & stat.S_ISGID and st.st_gid == pst.st_gid, nd
        if not acl:
            assert stat.S_IMODE(st.st_mode) & 0o777 == PL.subdirectory_mode(pst.st_mode), (nd, oct(st.st_mode))
    if acl:
        print('UNTESTED tree: the directory mode without a default ACL (TMPDIR has one)')
    try:
        PL.copy_tree(src, dst)
        raise AssertionError('a copy_tree over an existing destination was not refused')
    except PL.PlacementRefusal as e:
        assert str(e) == f'placement: copy_tree destination {dst} exists - nothing written', e
    # a copy that differs: everything this copy made is removed (the new parents too), and it refuses
    for how in ('flip', 'short'):
        dst2 = os.path.join(d, 'new', 'x', 'Run_L')
        with _differs_for('src/sub/deep/c.bin', how=how):
            try:
                PL.copy_tree(src, dst2, times=True)
                raise AssertionError('a tree with a copy that differs was placed')
            except PL.CopyNotVerified as e:
                msg = str(e)
                assert e.src_md5 == _md5(os.path.join(src, 'sub/deep/c.bin')) and e.copy_size, vars(e)
        assert msg.startswith(f'placement: copy_tree {src} -> {dst2}: {src}/sub/deep/c.bin not copied (CopyNotVerified: '), msg
        assert f"removed what this copy made: 1 new director(ies) ['{d}/new'], 0 new file(s) outside them" in msg, msg
        assert not os.path.exists(os.path.join(d, 'new')) and not _leftovers(d), how
    # a merge (tier_sublst.copy_tree of a directory): other files kept, a replaced file keeps its mode, new ones the rule
    m = os.path.join(d, 'merge')
    _write(os.path.join(m, 'sub', 'b.bin'), b'older\n', mode=0o640)
    _write(os.path.join(m, 'keep.txt'), b'kept\n', mode=0o640)
    ST.copy_tree(src, m)
    assert _md5(os.path.join(m, 'sub', 'b.bin')) == _md5(os.path.join(src, 'sub', 'b.bin'))
    assert stat.S_IMODE(os.stat(os.path.join(m, 'sub', 'b.bin')).st_mode) == 0o640
    assert open(os.path.join(m, 'keep.txt'), 'rb').read() == b'kept\n' and os.path.islink(os.path.join(m, 'link'))
    assert stat.S_IMODE(os.stat(os.path.join(m, 'a.bin')).st_mode) == _rule_mode(m)
    ST.copy_tree(src, m)                          # again: the link kept, every file replaced in place
    m2 = os.path.join(d, 'merge2')
    _write(os.path.join(m2, 'sub', 'b.bin'), b'older\n', mode=0o640)
    with _differs_for('src/sub/deep/c.bin'):
        try:
            ST.copy_tree(src, m2)
            raise AssertionError('a merge with a copy that differs was not refused')
        except PL.CopyNotVerified as e:
            msg = str(e)
    assert (f"removed what this copy made: 1 new director(ies) ['{m2}/sub/deep'], 2 new file(s) outside them; replaced "
            f"before it (they stay): ['{m2}/sub/b.bin']") in msg, msg
    assert sorted(os.listdir(m2)) == ['sub'] and os.listdir(os.path.join(m2, 'sub')) == ['b.bin'], os.listdir(m2)
    assert not _leftovers(d), _leftovers(d)
    # deep_step._copy_into of a directory: the same placed copy
    DS._copy_into(src, os.path.join(d, 'att', 'transients'))
    assert _state(os.path.join(d, 'att', 'transients'))['sub/deep/c.bin'][0] == _md5(os.path.join(src, 'sub/deep/c.bin'))
    with _differs_for('src/a.bin'):
        try:
            DS._copy_into(src, os.path.join(d, 'att2', 'transients'))
            raise AssertionError('_copy_into placed a tree with a copy that differs')
        except PL.CopyNotVerified:
            pass
    assert not os.path.exists(os.path.join(d, 'att2')) and not _leftovers(d)
    print('ok tree: copy_tree places every file verified, by its directory\'s rule, with its source\'s times, every new '
          'directory by its parent\'s rule (set-group-ID kept), links as links; an existing destination refuses; a copy '
          'that differs (md5 or a short read) removes the partial tree and its new parents and refuses; a merge keeps other '
          'files and a replaced file\'s mode, and its refusal removes what it made and names the files it replaced; '
          'deep_step._copy_into and tier_sublst.copy_tree of a directory go through it')


def test_conversions():
    from exopipe.phase3.lib.archive import common as AC
    from exopipe.phase3.lib.curves import publish as PUB
    from exopipe.phase3.lib.masters import coadd as CO
    d = _fresh('convert')
    src = _write(os.path.join(d, 'work', 'm.fits'), _bytes(1 << 16, 7))
    out = os.path.join(d, 'out')
    os.makedirs(out)
    part, m = AC.verified_copy(src, os.path.join(out, 'a.fits'))
    assert part == os.path.join(out, 'a.fits.part') and m == _md5(src) == _md5(part)
    assert os.stat(part).st_mtime_ns == OLD[1] * 10 ** 9, 'preserve: the source\'s times (copy2)'
    try:
        AC.verified_copy(src, os.path.join(out, 'a.fits'))
        raise AssertionError('a stale .part was overwritten')
    except record.Refusal as e:
        assert 'G6: a stale ' in str(e), e
    os.remove(part)
    for how in ('flip', 'short'):
        with _differs_for('work/m.fits', how=how):
            try:
                AC.verified_copy(src, os.path.join(out, 'a.fits'))
                raise AssertionError('an archive copy that differs was not refused')
            except record.Refusal as e:
                assert str(e).startswith(f'G6: copy of {src} not verified (placement: the copy ') and \
                    str(e).endswith('- STOPPED, nothing replaced'), e
        assert os.listdir(out) == [], os.listdir(out)
    try:
        AC.verified_copy(src, os.path.join(out, 'a.fits'), want='0' * 32)
        raise AssertionError('a copy whose md5 is not the wanted one was not refused')
    except record.Refusal as e:
        assert f'(md5 {_md5(src)} != {"0" * 32})' in str(e), e
    assert os.listdir(out) == [], os.listdir(out)
    for name, fn, times in (('coadd _atomic_copy', CO._atomic_copy, True), ('curves _atomic_copy', PUB._atomic_copy, False)):
        dst = os.path.join(out, name.split()[0] + '.fits')
        stale = _write(dst + '.4194301.0badf00d.tmp', b'killed\n', times=(time.time() - 7200,) * 2)
        fn(src, dst)
        assert _md5(dst) == _md5(src) and not os.path.exists(stale), name
        assert stat.S_IMODE(os.stat(dst).st_mode) == _rule_mode(out), (name, oct(os.stat(dst).st_mode))
        assert (os.stat(dst).st_mtime_ns == OLD[1] * 10 ** 9) == times, name
        os.chmod(dst, 0o640)
        before = _state(out)
        with _differs_for('work/m.fits', how='short'):
            try:
                fn(src, dst)
                raise AssertionError(f'{name}: a copy that differs was placed')
            except PL.CopyNotVerified:
                pass
        assert _state(out) == before and sorted(os.listdir(out)) == sorted(before), (name, os.listdir(out))
    print('ok convert: archive verified_copy (the .part through copy_verified, the source\'s times; a stale .part, a copy '
          'that differs and an unwanted md5 refuse G6 with no .part left), coadd _atomic_copy and curves _atomic_copy '
          '(through placement.copy: a killed copy\'s temporary file swept, the directory rule, a copy that differs '
          'refused with the product unchanged and no temporary or fixed-name file)')


def _flow_case(d):
    fd = os.path.join(d, 'work', 'flow')
    for i in range(1, 4):
        _write(os.path.join(fd, f'flow_41MHz_{i:04d}.npy'), _bytes(4096 + i, i))
    _write(os.path.join(fd, 'flow_41MHz.csv'), b'a,b\n1,2\n')
    _write(os.path.join(fd, 'flow_41MHz.log'), b'log\n')
    run = _run(d, 'flow')
    args = types.SimpleNamespace(out=run.out_root, fresh=False, supersede=None)
    return run, args, {'41MHz': {'status': 'done', 'meta': {'flow_dir': fd}}}


def test_flow_ship():
    from exopipe.phase3.lib.flow import steps_snapshot as FS
    d = _fresh('flow')
    run, args, recs = _flow_case(d)
    fields, qa = FS.product_dirs(args, run, 'c1')
    with run.unit('ship_c1') as u:
        with _differs_for('flow_41MHz_0002.npy', how='cut'):
            FS.ship_cell(u, args, run, 'c1', ['41MHz'], recs)
    assert u.status == 'refused', (u.status, u.reason)
    assert u.reason.startswith(f'G-S9: {fields}/flow_41MHz_0002.npy not shipped (its copy differs from the source: '
                               f'placement:'), u.reason
    assert u.reason.endswith('1 file(s) shipped by this run before it stay in place, 3 file(s) after it not copied'), u.reason
    assert sorted(os.listdir(fields)) == ['flow_41MHz_0001.npy'] and not os.path.exists(qa), os.listdir(fields)
    assert not _leftovers(d), _leftovers(d)
    assert PL._attrs(os.path.join(fields, 'flow_41MHz_0001.npy'))[2] == _rule_mode(fields), 'the rule, never the work 0600'
    if CAN_REFUSE:
        with run.unit('ship_c1b') as u:
            with _refuse_for('flow_41MHz.csv'):
                FS.ship_cell(u, args, run, 'c1', ['41MHz'], recs)
        assert u.status == 'refused' and f'G-S9: {qa}/flow_41MHz.csv not shipped (not placed: placement:' in u.reason, u.reason
        assert u.reason.endswith('2 file(s) shipped by this run before it stay in place, 1 file(s) after it not '
                                 'copied'), u.reason
        assert sorted(os.listdir(fields)) == [f'flow_41MHz_{i:04d}.npy' for i in (1, 2, 3)], os.listdir(fields)
        assert os.listdir(qa) == [], os.listdir(qa)
        assert not _leftovers(d), _leftovers(d)
    else:
        print('UNTESTED flow: an attribute refusal')
    print('ok flow: a ship copy that differs refuses G-S9 before it is placed, naming the files in place before it and the '
          'files after it not copied; an attribute refusal likewise (\'not placed\'); the directory rule; no temporary file')


def _cube_ns(d, u):
    cub, dest, work = (os.path.join(d, n) for n in ('cubes', 'dest', 'work'))
    jobs = [('41MHz', 'I'), ('41MHz', 'V'), ('46MHz', 'I'), ('46MHz', 'V')]
    for i, (sb, st) in enumerate(jobs):
        _write(os.path.join(cub, f'c1_{sb}_{st}_flowcube.fits'), _bytes(20000 + i, 10 + i))
    for n in ('cube_write.log', 'cube_verify.log'):
        _write(os.path.join(work, n), n.encode() * 20)
    os.makedirs(dest, exist_ok=True)
    return dict(os=os, subprocess=subprocess, placement=PL, u=u, cell='c1', cub=cub, dest=dest, work=work, jobs=jobs,
                results={j: (0, 'OK', f'VERIFY {j[0]} {j[1]} OK') for j in jobs}, ncube=4, staged=['41MHz', '46MHz'],
                script=os.path.join(REPO, 'tests', 'x.py'), ctx=types.SimpleNamespace(run=types.SimpleNamespace(host='h')),
                _date_u=lambda: '2026-11-25')


class _copies_logged:
    """Within the block, the destinations core/placement.copy places, in order."""

    def __enter__(self):
        self.real, self.dsts = PL.copy, []
        PL.copy = lambda src, dst, times=False: self.dsts.append(os.path.basename(dst)) or self.real(src, dst, times)
        return self

    def __exit__(self, *a):
        PL.copy = self.real
        return False


def test_cube_ship():
    code = _block('lib/frames/tier_snapshot.py', "        names = [f'{cell}_{sb}_{st}_flowcube.fits' for sb, st in jobs]",
                  "        for dst, m in md5s.items():\n            u.add_output(dst, 'snapshot_cube', md5=m)")
    cubes = ['c1_41MHz_I_flowcube.fits', 'c1_41MHz_V_flowcube.fits', 'c1_46MHz_I_flowcube.fits', 'c1_46MHz_V_flowcube.fits']
    d = _fresh('cube')
    run = _run(d, 'frames')
    with run.unit('cube_ok') as u:
        ns = _cube_ns(d, u)
        old = _write(os.path.join(ns['dest'], 'c1_46MHz_V_flowcube.fits'), b'older cube\n', mode=0o640)
        _write(os.path.join(ns['dest'], 'MANIFEST.txt'), b'c1_46MHz_V_flowcube.fits md5=x copy-OK | old\nCOMPLETE c1 old\n')
        with _copies_logged() as cl:
            exec(code, ns)
    assert u.status != 'refused' and len(ns['md5s']) == 4 and not ns['mismatch'], (u.status, u.reason, ns['mismatch'])
    assert cl.dsts == cubes + ['cube_write.log', 'cube_verify.log', 'MANIFEST.txt'], cl.dsts
    assert u.meta['replaced_manifest'] == ['c1_46MHz_V_flowcube.fits md5=x copy-OK | old', 'COMPLETE c1 old'], u.meta
    man = open(os.path.join(ns['dest'], 'MANIFEST.txt')).read().splitlines()
    assert [l.split(' ')[2] for l in man[:4]] == ['copy-OK'] * 4 and man[4].startswith('COMPLETE c1 cubes=4 '), man
    assert all(l.split(' ')[1] == f'md5={_md5(os.path.join(ns["dest"], l.split(" ")[0]))}' for l in man[:4]), man
    assert stat.S_IMODE(os.stat(old).st_mode) == 0o640, 'the existing cube keeps its mode'
    rule = _rule_mode(ns['dest'])
    for n in ('c1_41MHz_I_flowcube.fits', 'MANIFEST.txt', 'cube_verify.log'):
        assert stat.S_IMODE(os.stat(os.path.join(ns['dest'], n)).st_mode) == rule, (n, 'a new file: the rule')
    assert not _leftovers(d), _leftovers(d)
    d = _fresh('cube2')
    run = _run(d, 'frames')
    with run.unit('cube_mismatch') as u:
        ns = _cube_ns(d, u)
        _write(os.path.join(ns['dest'], 'MANIFEST.txt'), b'COMPLETE c1 old\n')
        with _differs_for('c1_41MHz_V_flowcube.fits', how='short'):
            exec(code, ns)
    assert u.status == 'refused', (u.status, u.reason)
    assert u.reason.startswith("N-G8 SHIP-FAILED c1: 1 cube(s) differ on the destination - not placed: "
                               "['c1_41MHz_V_flowcube.fits'] (the copy's and the local cube's sizes and md5s in "
                               "meta.copy_mismatch); the other 3 placed in "), u.reason
    assert f"no MANIFEST.txt written - the cell is not complete; the earlier {ns['dest']}/MANIFEST.txt was removed" in u.reason
    src = os.path.join(ns['cub'], 'c1_41MHz_V_flowcube.fits')
    mm = u.meta['copy_mismatch']
    assert len(mm) == 1 and mm[0].startswith('c1_41MHz_V_flowcube.fits md5=') and ' COPY-MISMATCH ' in mm[0], mm
    assert f'copy_size={os.path.getsize(src) // 2} source_size={os.path.getsize(src)} source_md5={_md5(src)} |' in mm[0], mm
    assert 'None' not in mm[0], mm
    assert sorted(os.listdir(ns['dest'])) == ['c1_41MHz_I_flowcube.fits', 'c1_46MHz_I_flowcube.fits',
                                              'c1_46MHz_V_flowcube.fits'], os.listdir(ns['dest'])
    assert not _leftovers(d), _leftovers(d)
    d = _fresh('cube4')                         # a log whose copy differs: every cube placed, no MANIFEST.txt
    run = _run(d, 'frames')
    with run.unit('cube_log') as u:
        ns = _cube_ns(d, u)
        with _differs_for('work/cube_verify.log'):
            exec(code, ns)
    assert u.status == 'refused' and u.reason.startswith('N-G8 SHIP-FAILED c1: placement: the copy '), u.reason
    assert (f"every cube placed in {ns['dest']}, and ['cube_write.log'] of ['cube_write.log', 'cube_verify.log', "
            f"'MANIFEST.txt']; no MANIFEST.txt placed - the cell is not complete (logs in ") in u.reason, u.reason
    assert sorted(os.listdir(ns['dest'])) == sorted(cubes + ['cube_write.log']), os.listdir(ns['dest'])
    assert not _leftovers(d), _leftovers(d)
    if CAN_REFUSE:
        d = _fresh('cube3')
        run = _run(d, 'frames')
        with run.unit('cube_refused') as u:
            ns = _cube_ns(d, u)
            with _refuse_for('c1_46MHz_I_flowcube.fits'):
                exec(code, ns)
        assert u.status == 'refused' and u.reason.startswith('N-G8 SHIP-FAILED c1: placement:'), u.reason
        assert (f"placed in {ns['dest']} before it: ['c1_41MHz_I_flowcube.fits', 'c1_41MHz_V_flowcube.fits']; not copied: "
                f"['c1_46MHz_V_flowcube.fits']; no MANIFEST.txt written") in u.reason, u.reason
        assert sorted(os.listdir(ns['dest'])) == ['c1_41MHz_I_flowcube.fits', 'c1_41MHz_V_flowcube.fits']
        assert not _leftovers(d), _leftovers(d)
        d = _fresh('cube5')
        run = _run(d, 'frames')
        with run.unit('cube_manifest_refused') as u:
            ns = _cube_ns(d, u)
            with _refuse_for('dest/MANIFEST.txt'):
                exec(code, ns)
        assert u.status == 'refused' and 'no MANIFEST.txt placed - the cell is not complete' in u.reason, u.reason
        assert sorted(os.listdir(ns['dest'])) == sorted(cubes + ['cube_write.log', 'cube_verify.log'])
        assert not _leftovers(d), _leftovers(d)
    else:
        print('UNTESTED cube: an attribute refusal')
    print('ok cube: an earlier MANIFEST.txt removed before any cube is placed (its lines in the record); four cubes placed '
          '(copy-OK; an existing cube keeps its mode 0640, a new one the directory rule), then the two logs, then '
          'MANIFEST.txt last; a cube whose copy differs is not placed and N-G8 names it (copy and source sizes and md5s '
          'recorded), no MANIFEST.txt; a log not placed, or MANIFEST.txt refused, leaves no MANIFEST.txt; a placement '
          'refusal names what is placed and what is not copied; no temporary file')


def _movies_patches(TS, scratch, staged_dir):
    def engine(script, a, key, log, unit=None, refusal_codes=()):
        out = a[a.index('--out') + 1]
        sb = a[a.index('--subband') + 1]
        _write(out, _bytes(5000, 20 + int(sb[:-3])))
        _write(out.replace('.mp4', '_series.png'), b'png ' + sb.encode())
        _write(out.replace('.mp4', '_series.csv'), b'csv ' + sb.encode())
        _write(log, b'movie log ' + sb.encode())
        return (types.SimpleNamespace(rc=0, reason=''),)
    return dict(flow_dir=lambda ctx, cell: (os.path.join(staged_dir, 'flow'), 'test'), legacy_script=lambda *a: 'movie.py',
                engine_flags=lambda *a, **k: [], first_flowed_I=lambda sd: os.path.join(sd, 'hdr.fits'),
                interpreter_key=lambda *a: None, engine=engine, parallel=lambda fn, items, n: [fn(x) for x in items],
                strip_legacy_header=lambda p: None)


def test_flow_movies():
    from exopipe.phase3.lib.frames import tier_snapshot as TS
    cases = [('differs', lambda: _differs_for('flow_46MHz_series.csv'), 'is not verified')]
    if CAN_REFUSE:
        cases.append(('refused', lambda: _refuse_for('flow_46MHz.mp4'), 'placement:'))
    else:
        print('UNTESTED movies: an attribute refusal')
    for name, ctxm, says in cases:
        d = _fresh('movies_' + name)
        ship = os.path.join(d, 'ship')
        ctx = types.SimpleNamespace(S=types.SimpleNamespace(FLOW_MOVIE_DURATION_S=10, FLOW_MOVIE_PROCS=1),
                                    scratch=lambda: os.path.join(d, 'scratch'), args=types.SimpleNamespace(fresh=False))
        patches = _movies_patches(TS, ctx.scratch(), d)
        real = {k: getattr(TS, k) for k in patches}
        run = _run(d, 'frames')
        try:
            for k, v in patches.items():
                setattr(TS, k, v)
            with run.unit('movies') as u:
                with ctxm():
                    TS.flow_movies(ctx, u, 'c1', os.path.join(d, 'staged'), ['41MHz', '46MHz'], ship)
        finally:
            for k, v in real.items():
                setattr(TS, k, v)
        r = u.refused_items
        assert len(r) == 1 and r[0]['path'] == os.path.join(ship, 'flow_46MHz.mp4'), r
        assert r[0]['gate'] == 'N-G10 flow movie ship', r
        assert r[0]['reason'].startswith('flow movie 46MHz: not shipped - placement:'), r
        assert says in r[0]['reason'], r[0]['reason']
        assert sorted(os.listdir(ship)) == ['flow_41MHz.mp4', 'flow_41MHz_series.csv', 'flow_41MHz_series.png',
                                            'movie_41MHz.log'], os.listdir(ship)
        assert all(stat.S_IMODE(os.stat(os.path.join(ship, f)).st_mode) == _rule_mode(ship) for f in os.listdir(ship))
        assert [o['path'] for o in u.outputs] == [os.path.join(ship, 'flow_41MHz.mp4')], u.outputs
        assert not _leftovers(d), _leftovers(d)
    print('ok movies: a subband whose copy differs, or whose placement is refused, is an item refusal (N-G10 flow movie ship) '
          'with none of its four files placed; the other subband\'s four files placed (the directory rule)')


def test_admit_night():
    code = _block('lib/masters/admit_stages.py', '                    placement.sweep([dests[k] for k in baked])\n',
                  "                    rowd.update(verdict='admitted')", loop=True)
    keys = [('I', 'epoch'), ('I', 'sub'), ('V', 'epoch'), ('V', 'sub')]

    def ns(d, refused):
        dests = {k: os.path.join(d, 'subtracted', f'n_{k[0]}_{k[1]}.fits') for k in keys}
        os.makedirs(os.path.join(d, 'subtracted'), exist_ok=True)
        baked = {k: (dests[k], np.full((4, 4), i, np.float32), fits.Header([('ADMITTED', True)])) for i, k in enumerate(keys)}
        co = types.SimpleNamespace(epoch_standard_header=lambda p, fl, st, lst: fits.setval(p, 'EPOCHSTD', value=st))
        return dict(os=os, fits=fits, placement=PL, dests=dests, baked=baked, co=co, fl='41MHz', lst='09h', reached=[],
                    refuse_night=lambda reason, gate: refused.append((reason, gate)))
    d = _fresh('admit')
    refused = []
    g = ns(d, refused)
    stale = _write(g['dests'][('V', 'sub')] + '.4194301.0badf00d.tmp', b'killed\n', times=(time.time() - 7200,) * 2)
    exec(code, g)
    assert g['reached'] == [1] and not refused and not os.path.exists(stale), (g['reached'], refused)
    assert [fits.getheader(g['dests'][k]).get('EPOCHSTD') for k in keys] == ['I', None, 'V', None]
    assert all(stat.S_IMODE(os.stat(p).st_mode) == _rule_mode(os.path.dirname(p)) for p in g['dests'].values())
    if CAN_REFUSE:
        d = _fresh('admit2')
        refused = []
        g = ns(d, refused)
        with _refuse_for('n_V_epoch.fits'):
            exec(code, g)
        assert g['reached'] == [] and len(refused) == 1 and refused[0][1] == 'admit.placement', (g['reached'], refused)
        assert refused[0][0].startswith('its products were not placed (placement:'), refused
        assert refused[0][0].endswith('- no admissions.csv row: admitted again at the next run'), refused
        assert os.listdir(os.path.join(d, 'subtracted')) == [], os.listdir(os.path.join(d, 'subtracted'))
    else:
        print('UNTESTED admit: an attribute refusal')
    print('ok admit: the night\'s four products placed together (the epoch ones stamped before placement; a killed write\'s '
          'temporary file swept); a placement refusal refuses the night by name (admit.placement), none placed, no row')


def test_photometry():
    from exopipe.phase3.lib.photometry import steps_common as K
    d = _fresh('phot')
    work, dst = os.path.join(d, 'work'), os.path.join(d, 'prod', 'c1')
    npz = [_write(os.path.join(work, f'c1_{b}.npz'), _bytes(3000, i)) for i, b in enumerate(('Red', 'Green'))]
    man = _write(os.path.join(work, 'cell', 'manifest.csv'), b'name\nc1\n')
    pairs = [(p, os.path.join(dst, os.path.basename(p))) for p in npz] + [(man, os.path.join(dst, 'manifest.csv'))]
    try:
        K.place_copies(pairs + [(npz[1], os.path.join(dst, 'c1_Red.npz'))])
        raise AssertionError('two pairs with one destination were placed')
    except PL.PlacementRefusal as e:
        assert 'destinations named twice' in str(e) and 'nothing written' in str(e), e
    assert not os.path.exists(dst)
    with _differs_for('cell/manifest.csv', how='cut'):
        try:
            K.place_copies(pairs)
            raise AssertionError('a copy that differs was placed')
        except PL.CopyNotVerified:
            pass
    assert os.listdir(dst) == [], os.listdir(dst)
    assert K.place_copies(pairs) == [b for _a, b in pairs]
    assert all(_md5(a) == _md5(b) and os.stat(b).st_mtime_ns == OLD[1] * 10 ** 9 for a, b in pairs)
    code = _block('lib/photometry/steps_snapshot.py', '        try:            # the npz and the cell\'s manifest placed',
                  '        for p in npz:\n            u.add_output(')
    run = _run(d, 'photometry')
    with run.unit('measure.c1') as u:
        with _differs_for('c1_Green.npz'):
            exec(code, dict(os=os, K=K, placement=PL, u=u, npz=npz, dst=dst, work=work, man_name='manifest.csv',
                            mpath=os.path.join(dst, 'manifest.csv')))
    assert u.status == 'refused' and u.reason.startswith('SM-SHIP: placement: the copy '), u.reason
    assert u.reason.endswith(f'(work kept: {work})'), u.reason
    assert not _leftovers(d), _leftovers(d)
    print('ok phot: the npz and manifest placed together with their sources\' times; a copy that differs places none; two '
          'pairs with one destination refuse before anything is written; measure_cell refuses SM-SHIP (work kept)')


def _deep_writers():
    src = open(os.path.join(REPO, 'exopipe', 'phase3', 'lib', 'search', 'deep_search.py')).read()
    body = ast.parse(src).body
    fns = {n.name: n for n in body if isinstance(n, ast.FunctionDef)}
    fns.update({t.id: n for n in body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)})
    code = '\n\n'.join(ast.get_source_segment(src, fns[k]) for k in (
        '_STRUCT_CARD', '_WCS_CARD', 'SEARCH_OWNED_CARDS', 'ARCHIVE_RECORD_COLS', '_archive_root', '_archive_predecessor',
        'merged_archive_header', 'dropped_history', '_archive_record', '_archive_write_merged', '_archive_frame_publish',
        '_write_archive_frame'))
    import csv
    import fcntl
    import re
    g = dict(os=os, re=re, np=np, fits=fits, placement=PL, time=time, csv=csv, fcntl=fcntl)
    exec(compile(code, 'deep_search.py:writers', 'exec'), g)
    return g


def test_deep_search_writers():
    from exopipe.common import validate_fits_headers as VF
    g = _deep_writers()
    d = _fresh('deep')
    path = os.path.join(d, 'work', 'f_redo_resid_09h.fits')
    os.makedirs(os.path.dirname(path))
    hdr = fits.Header([('PRODNAME', 'ovro_lwa_test_frame')])
    real = VF.validate_header
    ok = types.SimpleNamespace(ok=True, errors=[])
    try:
        VF.validate_header = lambda p, level='standard': ok
        stale = _write(path + '.4194301.0badf00d.tmp', b'killed\n', times=(time.time() - 7200,) * 2)
        g['_write_archive_frame'](path, np.ones((4, 4)), hdr, '09h Red', '2025-01-01')
        assert not os.path.exists(stale) and fits.getdata(path).sum() == 16 and not _leftovers(d)
        before = _md5(path)
        ok.ok, ok.errors = False, ['NAXIS missing']
        try:
            g['_write_archive_frame'](path, np.zeros((4, 4)), hdr, '09h Red', '2025-01-01')
            raise AssertionError('a frame whose header fails validation was placed')
        except RuntimeError as e:
            assert 'archive frame header validation FAILED (NAXIS missing) - REFUSING to archive' in str(e), e
        assert _md5(path) == before and not _leftovers(d), _leftovers(d)
        ok.ok, ok.errors = True, []
        if CAN_REFUSE:
            with _refuse_for('f_redo_resid_09h.fits'):
                try:
                    g['_write_archive_frame'](path, np.zeros((4, 4)), hdr, '09h Red', '2025-01-01')
                    raise AssertionError('a refused placement was placed')
                except PL.PlacementRefusal:
                    pass
            assert _md5(path) == before and not _leftovers(d), _leftovers(d)
        else:
            print('UNTESTED deep: an attribute refusal')
    finally:
        VF.validate_header = real
    os.chmod(path, 0o600)
    os.utime(path, OLD)
    env = {k: os.environ.get(k) for k in ('P3_ARCHIVE_ROOT', 'P3_ARCHIVE_RUN', 'P3_ARCHIVE_WRITE',
                                          'P3_ARCHIVE_PREDECESSOR_ROOTS')}
    try:
        VF.validate_header = lambda p, level='standard': ok
        arch = os.path.join(d, 'archive')
        os.makedirs(arch)
        os.chmod(arch, 0o2775)
        os.environ.update(P3_ARCHIVE_ROOT=arch, P3_ARCHIVE_RUN='Run_test')
        os.environ.pop('P3_ARCHIVE_WRITE', None)
        os.environ.pop('P3_ARCHIVE_PREDECESSOR_ROOTS', None)
        dst = g['_archive_frame_publish'](path, '09h', 'Red', '2025-01-01', 'I', 'deep', '09h Red')
        assert dst == f'{arch}/09h/2025-01-01/Run_test/Red/I/deep/ovro_lwa_test_frame.fits', dst
        assert (np.array_equal(fits.getdata(dst), fits.getdata(path)) and fits.getheader(dst)['PRODNAME'] == hdr['PRODNAME']
                and stat.S_IMODE(os.stat(dst).st_mode) == _rule_mode(os.path.dirname(dst))), oct(os.stat(dst).st_mode)
        assert os.stat(dst).st_mtime_ns == OLD[1] * 10 ** 9
        before = _md5(dst)
        fits.setval(path, 'EXTRA', value=1)
        ok.ok, ok.errors = False, ['NAXIS missing']
        try:
            g['_archive_frame_publish'](path, '09h', 'Red', '2025-01-01', 'I', 'deep', '09h Red')
            raise AssertionError('an archive frame whose header fails validation was placed')
        except RuntimeError as e:
            assert 'archive frame header validation FAILED' in str(e), e
        assert _md5(dst) == before and not _leftovers(d), _leftovers(d)
        ok.ok, ok.errors = True, []
        if CAN_REFUSE:
            with _refuse_for('ovro_lwa_test_frame.fits'):
                try:
                    g['_archive_frame_publish'](path, '09h', 'Red', '2025-01-01', 'I', 'deep', '09h Red')
                    raise AssertionError('a refused placement was placed')
                except PL.PlacementRefusal:
                    pass
            assert _md5(dst) == before and not _leftovers(d), _leftovers(d)
        g['_archive_frame_publish'](path, '09h', 'Red', '2025-01-01', 'I', 'deep', '09h Red')
        assert fits.getheader(dst).get('EXTRA') == 1 and not _leftovers(d), _leftovers(d)    # the frame it superseded in place
    finally:
        VF.validate_header = real
        for k, v in env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    print('ok deep: _write_archive_frame places a valid frame (a killed write\'s temporary file swept), and a header that '
          'fails validation or a placement refusal leaves the frame as it was; _archive_frame_publish gives the archive '
          'frame the directory\'s rule (the work frame 0600), the work frame\'s times and pixels; a header that fails '
          'validation or a placement refusal leaves the earlier archive frame; no temporary file')


def test_deep_publish():
    from exopipe.phase3.lib.search import deep_step as DS
    d = _fresh('publish')
    att = os.path.join(d, 'att', 'A1')
    _write(os.path.join(att, 'transients', 'x.csv'), b'x\n')
    for n in ('c1.log', 'fit_positions.csv', 'fit_positions_summary.json'):
        _write(os.path.join(att, n), n.encode())
    item = types.SimpleNamespace(cell='c1', lst='09h', scope='Red', stokes='I')
    ctx, f = {'out': os.path.join(d, 'out')}, {'attempt': att, 'label': 'L', 'att_tag': 'A1'}
    u = types.SimpleNamespace(notes=[], note=lambda m: u.notes.append(m))
    dest = os.path.join(ctx['out'], 'attempts', 'A1')
    with _differs_for('A1/fit_positions_summary.json'):
        try:
            DS.stage_publish(u, item, ctx, f)
            raise AssertionError('an attempt with a copy that differs was published')
        except PL.PlacementRefusal as e:
            msg = str(e)
    assert msg.endswith(f"- the attempt is not published: {dest}.part (with ['transients', 'c1.log', 'fit_positions.csv'] "
                        f"copied before it) removed"), msg
    assert os.listdir(os.path.dirname(dest)) == [], os.listdir(os.path.dirname(dest))
    # the side outputs (outside the attempt) are placed BEFORE the publish rename: one that is not placed publishes nothing
    side = os.path.join(d, 'side')
    gaul = _write(os.path.join(side, 'cat', 'c1.gaul.csv'), b'gaul\n')
    _write(gaul + '.gen.json', b'{}\n')
    verd = _write(os.path.join(side, 'v110_out', 'verdicts.csv'), b'v\n')
    other = _write(os.path.join(side, 'v110_out', 'summary.json'), b's\n')
    finals = {gaul: os.path.join(ctx['out'], 'parent_catalogs', 'c1.gaul.csv'),
              verd: os.path.join(ctx['out'], 'v110_products', 'c1', 'verdicts.csv')}
    sides = [(gaul, 'parent_catalog', {}, finals[gaul]), (verd, 'fp_gate_verdicts', {}, finals[verd])]
    code = _block('lib/search/deep_step.py', "        dest = os.path.join(ctx['out'], 'attempts', "
                  "os.path.basename(f['attempt']))\n        copies, side_pairs", '        record_outputs(u, pairs, verified)')

    def run_block():
        g = dict(os=os, glob=__import__('glob'), stage_publish=DS.stage_publish, u=u, item=item, ctx=ctx, f=f, outs=[],
                 sides=sides)
        exec(code, g)
        return g
    with _differs_for('v110_out/summary.json'):
        try:
            run_block()
            raise AssertionError('a side output whose copy differs was placed and the attempt published')
        except PL.PlacementRefusal as e:
            msg = str(e)
    assert msg.endswith(f"- the attempt is not published ({dest}.part removed); the side products placed before this one "
                        f"stay: ['{finals[gaul]}', '{finals[gaul]}.gen.json', '{finals[verd]}']"), msg
    assert os.listdir(os.path.dirname(dest)) == [] and not _leftovers(d), (os.listdir(os.path.dirname(dest)), _leftovers(d))
    g = run_block()
    assert g['dest'] == dest and sorted(os.listdir(dest)) == ['c1.log', 'fit_positions.csv', 'fit_positions_summary.json',
                                                              'transients'], os.listdir(dest)
    assert [x[1] for x in g['pairs']] == [finals[gaul], finals[verd]], g['pairs']
    assert g['pairs'][1][3]['catalog'] == os.path.join(dest, 'transients', 'Run_L', 'master_catalog.csv'), g['pairs']
    assert _md5(os.path.join(os.path.dirname(finals[verd]), 'summary.json')) == _md5(other)
    assert os.path.exists(finals[gaul] + '.gen.json')
    assert stat.S_IMODE(os.stat(os.path.join(dest, 'c1.log')).st_mode) == _rule_mode(dest) and not _leftovers(d)
    assert os.stat(os.path.join(dest, 'transients', 'x.csv')).st_mtime_ns == OLD[1] * 10 ** 9
    # published but not registered (a run stopped between the rename and the register append): a Refusal naming the
    # attempt and the recovery, never a RuntimeError; with its register row: published once
    try:
        DS.stage_publish(u, item, ctx, f)
        raise AssertionError('an attempt was published twice')
    except record.Refusal as e:
        msg = str(e)
        assert not isinstance(e, RuntimeError)
    reg = os.path.join(ctx['out'], 'published_I.csv')
    assert msg.startswith(f'{dest} is published but {reg} has no row for it (cell c1, attempt A1): an earlier run stopped '
                          f'between the publish and the register append. To recover, move {dest} out of '), msg
    assert msg.endswith('and run the unit again - it is published anew'), msg
    DS.register_row(ctx['out'], item, 'A1')
    try:
        DS.stage_publish(u, item, ctx, f)
        raise AssertionError('a registered attempt was published twice')
    except record.Refusal as e:
        assert str(e) == f'{dest} exists and {reg} registers it - an attempt is published once', e
    # a killed run's unpublished .part is this unit's own scratch: removed (noted) before the copy
    os.rename(dest, os.path.join(d, 'moved_out'))
    with open(reg, 'w'):
        pass
    _write(os.path.join(dest + '.part', 'junk'), b'killed\n')
    assert DS.stage_publish(u, item, ctx, f) == dest and 'junk' not in os.listdir(dest)
    assert u.notes and u.notes[-1] == f"publish: a killed earlier run's unpublished {dest}.part removed before the copy"
    print('ok publish: a file of the attempt whose copy differs publishes nothing (<dest>.part removed; the refusal names '
          'what was copied); the side outputs are placed BEFORE the publish rename, and one whose copy differs publishes '
          'nothing (the refusal names the side outputs placed before it); a clean publish places files by the directory\'s '
          'rule with their times; a published attempt without a register row refuses naming it and the recovery, with one '
          '"published once"; a killed run\'s .part is removed with a note')


def test_search_ship():
    from exopipe.phase3.lib.search import tier_snapshot as SS
    d = _fresh('ship')
    src, dst = os.path.join(d, 'node'), os.path.join(d, 'cell')
    names = ('runinfo_41MHz_I_x.json', 'noise_41MHz_I_x.csv', 'c1.log', 'model_41MHz_median.fits')
    for i, n in enumerate(names):
        _write(os.path.join(src, n), _bytes(2000 + i, 30 + i))
    _write(os.path.join(dst, 'noise_41MHz_I_x.csv'), b'earlier\n', mode=0o640)
    run = _run(d, 'search')
    keep = _state(dst)
    with run.unit('ship_refused') as u:
        with _differs_for('node/model_41MHz_median.fits'):
            SS.ship(u, src, dst, True, run.out_root)
    assert u.status == 'refused' and u.reason.startswith('ship: placement: the copy '), (u.status, u.reason)
    assert ('; --fresh: 0 contradicted earlier outcome file(s) moved aside and 1 file(s) to be replaced copied to '
            f'{run.out_root}/_superseded/cell_') in u.reason and u.reason.endswith(f'(node-local output kept: {src})'), u.reason
    assert _state(dst) == keep and not _leftovers(d), (_state(dst), _leftovers(d))
    assert u.meta['shipped']['refused'].startswith('placement: the copy'), u.meta['shipped']
    import glob
    kept = glob.glob(os.path.join(run.out_root, '_superseded', 'cell_*', 'noise_41MHz_I_x.csv'))
    assert len(kept) == 1 and open(kept[0], 'rb').read() == b'earlier\n', kept     # the verified backup, placed by the rule
    assert stat.S_IMODE(os.stat(kept[0]).st_mode) == _rule_mode(os.path.dirname(kept[0])), oct(os.stat(kept[0]).st_mode)
    assert os.stat(kept[0]).st_mtime_ns == OLD[1] * 10 ** 9
    with run.unit('ship_ok') as u:
        assert SS.ship(u, src, dst, True, run.out_root)
    assert sorted(os.listdir(dst)) == sorted(names) and u.meta['shipped']['copied'] == 4, (os.listdir(dst), u.meta)
    modes = {n: stat.S_IMODE(os.stat(os.path.join(dst, n)).st_mode) for n in names}
    assert modes == {n: 0o640 if n.startswith('noise') else _rule_mode(dst) for n in names}, modes
    assert all(_md5(os.path.join(src, n)) == _md5(os.path.join(dst, n)) for n in names) and not _leftovers(d)
    print('ok ship: a copy that differs places none of the four files and refuses the unit (the replaced file kept under '
          '_superseded, the destination unchanged, the node-local output kept); a --fresh ship places them all, the '
          'replaced file keeping its mode 0640, the new ones the directory rule; the kept copy is verified and placed')


def test_search_sublst():
    from exopipe.phase3.lib.search import tier_sublst as ST
    d = _fresh('sublst')
    a = _write(os.path.join(d, 'pub', 'master_catalog.csv'), _bytes(3000, 40))
    b = os.path.join(d, 'att', 'master_catalog.csv')
    with _differs_for('pub/master_catalog.csv', n=1) as h:
        assert ST.seed_attempt(os.path.join(d, 'pub'), os.path.join(d, 'att')) == ['master_catalog.csv'] and h.hits == 1
    assert _md5(a) == _md5(b) and stat.S_IMODE(os.stat(b).st_mode) == _rule_mode(os.path.dirname(b)) and not _leftovers(d)
    os.remove(b)
    with _differs_for('pub/master_catalog.csv', how='short'):
        try:
            ST.seed_attempt(os.path.join(d, 'pub'), os.path.join(d, 'att'))
            raise AssertionError('a seed copied short twice was not refused')
        except record.Refusal as e:
            assert 'copied short twice' in str(e) and not isinstance(e, PL.PlacementRefusal), e
    assert not os.path.exists(b) and not _leftovers(d), _leftovers(d)
    pub, att = os.path.join(d, 'cellpub'), os.path.join(d, 'attempt')
    srcs = [_write(os.path.join(att, n), n.encode() * 50) for n in ('a.csv', 'b.csv', 'c.csv')]
    _write(os.path.join(pub, 'a.csv'), b'old a\n', mode=0o640)
    _write(os.path.join(pub, 'c.csv'), b'old c\n', mode=0o640)
    pairs = [(s, os.path.join(pub, os.path.basename(s))) for s in srcs]
    run = types.SimpleNamespace(out_root=os.path.join(d, 'out'))
    plan = {'publish': pub, 'unit': 'u1'}
    real_pf, real_ct = ST._publish_files, ST.copy_tree
    calls = []

    def ct_fail(after, rollback_fails):
        def ct(s, t):
            calls.append(t)
            if len(calls) == after + 1:
                raise PL.CopyNotVerified('placement: the copy of c.csv is not verified (test)')
            if rollback_fails and len(calls) > after + 1:
                raise OSError(5, 'simulated restore failure')
            return real_ct(s, t)
        return ct
    keep = _state(pub)
    try:
        ST._publish_files = lambda plan_, attempt_: pairs
        for rollback_fails in (False, True):
            calls.clear()
            ST.copy_tree = ct_fail(2, rollback_fails)
            try:
                ST.publish(types.SimpleNamespace(meta={}), run, plan, att)
                raise AssertionError('a publish with a copy refused was not refused')
            except PL.PlacementRefusal as e:
                msg = str(e)
            assert msg.startswith('publish: placement: the copy of c.csv is not verified (test) - '), msg
            assert msg.endswith(f'; attempt kept: {att}'), msg
            if not rollback_fails:
                assert 'the 2 file(s) published before it rolled back (replaced ones restored from ' in msg, msg
                assert _state(pub) == keep and not _leftovers(d), (_state(pub), keep)
            else:
                assert 'the rollback of the 2 file(s) published before it FAILED (OSError: ' in msg, msg
                assert f'{pub} may hold a partial publish; the replaced files are in ' in msg, msg
    finally:
        ST._publish_files, ST.copy_tree = real_pf, real_ct
    print('ok sublst: copy_tree places by the directory\'s rule; seed_attempt retries a copy that differs once and refuses a '
          'second short copy; publish rolls a placement refusal back (replaced files restored, new ones removed, the '
          'published files as before) and says so, or says the rollback failed and where the replaced files are')


def main():
    shutil.rmtree(TMP, ignore_errors=True)
    os.makedirs(TMP)
    # sweep ages a temporary file by its ctime, which utime cannot set back: the tests that plant a killed writer's
    # leftover age it by mtime through the module's seam (test_sweep_ctime tests the ctime itself)
    PL._changed = lambda st: st.st_mtime
    try:
        test_copy()
        test_sweep_listing()
        test_sweep_ctime()
        test_copy_tree()
        test_conversions()
        test_flow_ship()
        test_cube_ship()
        test_flow_movies()
        test_admit_night()
        test_photometry()
        test_deep_search_writers()
        test_deep_publish()
        test_search_ship()
        test_search_sublst()
    finally:
        PL._changed = REAL_CHANGED
        shutil.rmtree(TMP, ignore_errors=True)
    print('PASS test_placement_p6')


if __name__ == '__main__':
    main()
