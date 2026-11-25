"""Placement batch P6 (core/placement in the archive writers and per-file copies: the snapshot flow ship, the snapshot cube ship,
the flow movies, the per-night admitted products, the snapshot photometry ship, the deep search's archive frames, the deep
attempt publish, the snapshot search ship and the sub-LST search's copies): every copy is checked against its source (size and
md5) BEFORE it is placed, and placed with the destination's attributes, never the work file's; what a copy that differs or a
placement refusal does at each handler. Data-free: scratch directories under TMPDIR; a placement refusal is forced by giving
the placement a group this user is not in (UNTESTED is printed when there is none, e.g. as root); a copy that differs by
replacing shutil.copyfile for named sources. Checks:
  copy     placement.copy / copy_verified: a new file takes the directory's rule (group, mode, inherited default ACL where
           setfacl works), never the source's mode 0600; a replaced file keeps its group, mode and ACL; times=True gives the
           source's times; a copy with other bytes of the same size (md5), a copy cut short after the copy (size) and a short
           copy the safe-copy gate refuses (exopipe/_safe_copy) each refuse (CopyNotVerified) with the destination unchanged
           and no temporary file; an attribute refusal is a PlacementRefusal that is not CopyNotVerified, same outcome;
  sweep    one directory listing however many products of one directory are swept; the removed files are those the
           glob-based sweep of P5 removed (names with glob characters, tagged names, the <dst>.1 product's names);
  flow     flow ship_cell (G-S9): a copy that differs refuses the unit naming the file, the files shipped before it (in place)
           and the files after it (not copied); an attribute refusal the same with 'not placed'; no temporary file;
  cube     frames stage_cube's ship block (N-G8): a cube whose copy differs is not placed and the unit refuses naming it, the
           other cubes placed, no MANIFEST.txt; a placement refusal refuses naming what is placed and what is not copied; an
           existing cube keeps its mode;
  movies   frames flow_movies: a subband whose copy differs or whose placement is refused is an item refusal (N-G10 flow
           movie ship) with none of its four files placed; the other subband's files placed with the directory's rule;
  admit    masters admit_stage's per-night write: a placement refusal refuses the night by name (gate admit.placement) with
           none of its products placed and no admissions row reached; a killed write's temporary file is swept;
  phot     photometry place_copies: the npz and the manifest placed together; a copy that differs places none; two pairs
           with one destination refuse before anything is written; measure_cell's handler refuses the unit 'SM-SHIP: ...
           (work kept: ...)';
  deep     the deep search's _write_archive_frame (a header that fails validation or a placement refusal leaves the frame
           as it was, no temporary file) and _archive_frame_publish (the archive frame takes the archive directory's mode,
           never the work frame's 0600, with the work frame's times; a copy that differs leaves the earlier archive frame);
           compiled alone (the module needs the cluster's data to import);
  publish  deep_step stage_publish: a file of the attempt not placed removes <dest>.part and publishes nothing, the refusal
           naming what was copied; the side-output loop's refusal says the attempt IS published and which side outputs are;
  ship     the snapshot search ship: a copy that differs places none of the files and refuses the unit, the replaced file
           kept under _superseded and the destination unchanged; a successful --fresh ship keeps a replaced file's mode;
  sublst   the sub-LST search: copy_tree places a file by the directory's rule and refuses a copy that differs;
           seed_attempt retries a copy that differs once and refuses the second; publish rolls back on a placement refusal
           (replaced files restored, new ones removed) and says so, or says that the rollback failed."""
import ast
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
    """Within the block, shutil.copyfile of a source whose path ends with ``suffix`` writes a copy that differs from it, for the
    first ``n`` such copies (every one when None): how='flip' the first byte changed (the same size: only the md5 sees it),
    how='cut' the last 100 bytes dropped after the copy (the size sees it), how='gate' a read path that stops at half the
    file (exopipe's safe-copy gate refuses it: ShortCopyError, the short copy removed)."""

    def __init__(self, suffix, how='flip', n=None):
        self.suffix, self.how, self.n, self.hits = suffix, how, n, 0

    def __enter__(self):
        self.real, self.real_obj = shutil.copyfile, shutil.copyfileobj
        me = self

        def hit(src):
            if os.fspath(src).endswith(me.suffix) and (me.n is None or me.hits < me.n):
                me.hits += 1
                return True
            return False

        def copyfile(src, dst, *a, **kw):
            if me.how == 'gate' and hit(src):
                def half(fsrc, fdst, length=0):
                    fdst.write(fsrc.read(max(1, os.fstat(fsrc.fileno()).st_size // 2)))
                shutil.copyfileobj = half
                try:
                    return me.real(src, dst, *a, **kw)
                finally:
                    shutil.copyfileobj = me.real_obj
            out = me.real(src, dst, *a, **kw)
            if me.how != 'gate' and hit(src):
                if me.how == 'flip':
                    with open(dst, 'r+b') as f:
                        b = f.read(1)
                        f.seek(0)
                        f.write(bytes([b[0] ^ 0xFF]))
                else:
                    os.truncate(dst, max(0, os.path.getsize(dst) - 100))
            return out
        shutil.copyfile = copyfile
        return self

    def __exit__(self, *a):
        shutil.copyfile, shutil.copyfileobj = self.real, self.real_obj
        return False


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
    for how in ('flip', 'cut', 'gate'):
        keep = _state(out)
        with _differs_for('work/a.fits', how=how) as h:
            try:
                PL.copy(src, dst, times=True)
                raise AssertionError(f'{how}: a copy that differs was placed')
            except PL.CopyNotVerified as e:
                msg = str(e)
        assert h.hits == 1, (how, h.hits)
        assert 'is not verified' in msg and 'nothing placed' in msg, msg
        assert ('SHORT-COPY' in msg) == (how == 'gate'), msg
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
    print('ok copy: a new file takes the directory\'s rule (not the source\'s 0600), a replaced one keeps its attributes, '
          'times=True the source\'s times; a copy with other bytes (md5), cut short (size) or refused by the safe-copy gate '
          'refuses (CopyNotVerified) with the destination unchanged and no temporary file; an attribute refusal likewise')


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
    assert u.reason.startswith(f'G-S9: {fields}/flow_41MHz_0002.npy not shipped (its copy differs from the source: placement:'), u.reason
    assert u.reason.endswith('1 file(s) shipped by this run before it stay in place, 3 file(s) after it not copied'), u.reason
    assert sorted(os.listdir(fields)) == ['flow_41MHz_0001.npy'] and not os.path.exists(qa), os.listdir(fields)
    assert not _leftovers(d), _leftovers(d)
    assert PL._attrs(os.path.join(fields, 'flow_41MHz_0001.npy'))[2] == 0o664, 'the directory rule, never the work file\'s 0600'
    if CAN_REFUSE:
        with run.unit('ship_c1b') as u:
            with _refuse_for('flow_41MHz.csv'):
                FS.ship_cell(u, args, run, 'c1', ['41MHz'], recs)
        assert u.status == 'refused' and f'G-S9: {qa}/flow_41MHz.csv not shipped (not placed: placement:' in u.reason, u.reason
        assert u.reason.endswith('2 file(s) shipped by this run before it stay in place, 1 file(s) after it not copied'), u.reason
        assert sorted(os.listdir(fields)) == [f'flow_41MHz_{i:04d}.npy' for i in (1, 2, 3)] and os.listdir(qa) == [], os.listdir(qa)
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
    os.makedirs(dest, exist_ok=True)
    os.makedirs(work, exist_ok=True)
    return dict(os=os, placement=PL, u=u, cell='c1', cub=cub, dest=dest, work=work, jobs=jobs,
                results={j: (0, 'OK', f'VERIFY {j[0]} {j[1]} OK') for j in jobs})


def test_cube_ship():
    code = _block('lib/frames/tier_snapshot.py', "        names = [f'{cell}_{sb}_{st}_flowcube.fits' for sb, st in jobs]",
                  "        size = subprocess.run(['du', '-sh', dest]")
    d = _fresh('cube')
    run = _run(d, 'frames')
    with run.unit('cube_ok') as u:
        ns = _cube_ns(d, u)
        old = _write(os.path.join(ns['dest'], 'c1_46MHz_V_flowcube.fits'), b'older cube\n', mode=0o640)
        exec(code, ns)
    assert u.status != 'refused' and len(ns['md5s']) == 4 and not ns['mismatch'], (u.status, u.reason, ns['mismatch'])
    assert all(l.split(' ')[2] == 'copy-OK' for l in ns['lines']) and len(ns['lines']) == 4, ns['lines']
    assert stat.S_IMODE(os.stat(old).st_mode) == 0o640, 'the existing cube keeps its mode'
    assert stat.S_IMODE(os.stat(os.path.join(ns['dest'], 'c1_41MHz_I_flowcube.fits')).st_mode) == 0o664, 'a new cube: the rule'
    d = _fresh('cube2')
    run = _run(d, 'frames')
    with run.unit('cube_mismatch') as u:
        ns = _cube_ns(d, u)
        with _differs_for('c1_41MHz_V_flowcube.fits'):
            exec(code, ns)
    assert u.status == 'refused', (u.status, u.reason)
    assert u.reason.startswith("N-G8 SHIP-FAILED c1: 1 cube(s) differ on the destination - not placed: "
                               "['c1_41MHz_V_flowcube.fits']; the other 3 placed in "), u.reason
    assert sorted(os.listdir(ns['dest'])) == ['c1_41MHz_I_flowcube.fits', 'c1_46MHz_I_flowcube.fits',
                                              'c1_46MHz_V_flowcube.fits'], os.listdir(ns['dest'])
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
    else:
        print('UNTESTED cube: an attribute refusal')
    print('ok cube: four cubes placed (copy-OK; an existing cube keeps its mode 0640, a new one 0664); a cube whose copy '
          'differs is not placed and N-G8 names it, the other three placed, no MANIFEST.txt; a placement refusal names what '
          'is placed and what is not copied; no temporary file')


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
        assert r[0]['gate'] == 'N-G10 flow movie ship' and r[0]['reason'].startswith('flow movie 46MHz: not shipped - placement:')
        assert says in r[0]['reason'], r[0]['reason']
        assert sorted(os.listdir(ship)) == ['flow_41MHz.mp4', 'flow_41MHz_series.csv', 'flow_41MHz_series.png',
                                            'movie_41MHz.log'], os.listdir(ship)
        assert all(stat.S_IMODE(os.stat(os.path.join(ship, f)).st_mode) == 0o664 for f in os.listdir(ship))
        assert [o['path'] for o in u.outputs] == [os.path.join(ship, 'flow_41MHz.mp4')], u.outputs
        assert not _leftovers(d), _leftovers(d)
    print('ok movies: a subband whose copy differs, or whose placement is refused, is an item refusal (N-G10 flow movie ship) '
          'with none of its four files placed; the other subband\'s four files placed (0664, the directory rule)')


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
    assert all(stat.S_IMODE(os.stat(p).st_mode) == 0o664 for p in g['dests'].values())
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
    fns = {n.name: n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)}
    code = '\n\n'.join(ast.get_source_segment(src, fns[k]) for k in ('_archive_root', '_archive_frame_publish',
                                                                     '_write_archive_frame'))
    g = dict(os=os, np=np, fits=fits, placement=PL)
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
    env = {k: os.environ.get(k) for k in ('P3_ARCHIVE_ROOT', 'P3_ARCHIVE_RUN', 'P3_ARCHIVE_WRITE')}
    try:
        arch = os.path.join(d, 'archive')
        os.makedirs(arch)
        os.chmod(arch, 0o2775)
        os.environ.update(P3_ARCHIVE_ROOT=arch, P3_ARCHIVE_RUN='Run_test')
        os.environ.pop('P3_ARCHIVE_WRITE', None)
        dst = g['_archive_frame_publish'](path, '09h', 'Red', '2025-01-01', 'I', 'deep', '09h Red')
        assert dst == f'{arch}/09h/2025-01-01/Run_test/Red/I/deep/ovro_lwa_test_frame.fits', dst
        assert _md5(dst) == _md5(path) and stat.S_IMODE(os.stat(dst).st_mode) == 0o664, oct(os.stat(dst).st_mode)
        assert os.stat(dst).st_mtime_ns == OLD[1] * 10 ** 9
        before = _md5(dst)
        fits.setval(path, 'EXTRA', value=1)
        with _differs_for('f_redo_resid_09h.fits'):
            try:
                g['_archive_frame_publish'](path, '09h', 'Red', '2025-01-01', 'I', 'deep', '09h Red')
                raise AssertionError('an archive copy that differs was placed')
            except PL.CopyNotVerified:
                pass
        assert _md5(dst) == before and not _leftovers(d), _leftovers(d)
    finally:
        for k, v in env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    print('ok deep: _write_archive_frame places a valid frame (a killed write\'s temporary file swept), and a header that '
          'fails validation or a placement refusal leaves the frame as it was; _archive_frame_publish gives the archive '
          'frame the directory\'s mode 0664 (the work frame 0600) and the work frame\'s times; a copy that differs leaves the '
          'earlier archive frame; no temporary file')


def test_deep_publish():
    from exopipe.phase3.lib.search import deep_step as DS
    d = _fresh('publish')
    att = os.path.join(d, 'att', 'A1')
    _write(os.path.join(att, 'transients', 'x.csv'), b'x\n')
    for n in ('c1.log', 'fit_positions.csv', 'fit_positions_summary.json'):
        _write(os.path.join(att, n), n.encode())
    item = types.SimpleNamespace(cell='c1', lst='09h', scope='Red')
    ctx, f = {'out': os.path.join(d, 'out')}, {'attempt': att, 'label': 'L'}
    dest = os.path.join(ctx['out'], 'attempts', 'A1')
    with _differs_for('A1/fit_positions_summary.json'):
        try:
            DS.stage_publish(None, item, ctx, f)
            raise AssertionError('an attempt with a copy that differs was published')
        except PL.PlacementRefusal as e:
            msg = str(e)
    assert msg.endswith(f"- the attempt is not published: {dest}.part (with ['transients', 'c1.log', 'fit_positions.csv'] "
                        f"copied before it) removed"), msg
    assert os.listdir(os.path.dirname(dest)) == [], os.listdir(os.path.dirname(dest))
    assert DS.stage_publish(None, item, ctx, f) == dest
    assert stat.S_IMODE(os.stat(os.path.join(dest, 'c1.log')).st_mode) == 0o664 and not _leftovers(d)
    code = _block('lib/search/deep_step.py', '        try:\n            for p, cls, tctx, final in sides:',
                  '        record_outputs(u, pairs, verified)')
    s1, s2 = (_write(os.path.join(d, 'side', n), n.encode()) for n in ('s1.csv', 's2.csv'))
    finals = [os.path.join(d, 'pub_side', n) for n in ('s1.csv', 's2.csv')]
    g = dict(os=os, glob=__import__('glob'), placement=PL, _copy_into=DS._copy_into, dest=dest, f=f, outs=[], pairs=[],
             sides=[(s1, 'a', {}, finals[0]), (s2, 'b', {}, finals[1])])
    with _differs_for('side/s2.csv'):
        try:
            exec(code, g)
            raise AssertionError('a side output that differs was placed')
        except PL.PlacementRefusal as e:
            msg = str(e)
    assert msg.endswith(f"- the attempt IS published at {dest}; the side outputs before this one are in place "
                        f"(['{finals[0]}']); no register row appended"), msg
    assert os.listdir(os.path.join(d, 'pub_side')) == ['s1.csv'] and not _leftovers(d), _leftovers(d)
    print('ok publish: a file of the attempt whose copy differs publishes nothing (<dest>.part removed; the refusal names '
          'what was copied); a clean publish places files by the directory\'s rule; a side output that differs refuses '
          'saying the attempt IS published and which side outputs are in place')


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
    with run.unit('ship_ok') as u:
        assert SS.ship(u, src, dst, True, run.out_root)
    assert sorted(os.listdir(dst)) == sorted(names) and u.meta['shipped']['copied'] == 4, (os.listdir(dst), u.meta)
    modes = {n: stat.S_IMODE(os.stat(os.path.join(dst, n)).st_mode) for n in names}
    assert modes == {n: 0o640 if n.startswith('noise') else 0o664 for n in names}, modes
    assert all(_md5(os.path.join(src, n)) == _md5(os.path.join(dst, n)) for n in names) and not _leftovers(d)
    print('ok ship: a copy that differs places none of the four files and refuses the unit (the replaced file kept under '
          '_superseded, the destination unchanged, the node-local output kept); a --fresh ship places them all, the '
          'replaced file keeping its mode 0640, the new ones 0664')


def test_search_sublst():
    from exopipe.phase3.lib.search import tier_sublst as ST
    d = _fresh('sublst')
    a = _write(os.path.join(d, 'pub', 'master_catalog.csv'), _bytes(3000, 40))
    b = os.path.join(d, 'att', 'master_catalog.csv')
    with _differs_for('pub/master_catalog.csv', n=1) as h:
        assert ST.seed_attempt(os.path.join(d, 'pub'), os.path.join(d, 'att')) == ['master_catalog.csv'] and h.hits == 1
    assert _md5(a) == _md5(b) and stat.S_IMODE(os.stat(b).st_mode) == 0o664 and not _leftovers(d)
    os.remove(b)
    with _differs_for('pub/master_catalog.csv', how='gate'):
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
    try:
        test_copy()
        test_sweep_listing()
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
        shutil.rmtree(TMP, ignore_errors=True)
    print('PASS test_placement_p6')


if __name__ == '__main__':
    main()
