#!/usr/bin/env python
"""The space gate of every writer step (core/space.py, config section space; plan/CUTOVER_REMAINING_20261124.md row 35). Data
free: a synthetic mount table (two directories of the scratch area declared a lustre and an xfs mount), a stub `lfs` and a
stub statvfs. Asserts:
  - the parsers are the photometry campaign's (lib/photometry/campaign.parse_quota / parse_df are core/space's);
  - the mount: the longest mount point above a path, the last entry on a tie (/lustre: autofs, then lustre), an octal-escaped
    mount point, a path not made yet is on its existing parent's file system;
  - Lustre: `lfs quota -q -u <run user> <mount>` and `lfs df <mount>`; each floor (quota bytes, quota files, free bytes)
    refuses below it with its code (SPACE-QUOTA, SPACE-FS); a limit of 0 is no limit; an unreadable or failing lfs is
    SPACE-UNREADABLE; need_bytes / need_files / booked_bytes count against the headroom; the project quota is read only when
    space.PROJECT_ID is set and refuses SPACE-PROJECT; user= reads another user's quota;
  - another file system: statvfs's available bytes against LOCAL_MIN_FREE_BYTES (SPACE-LOCAL), a booking must fit; an
    unreadable statvfs or mount table is SPACE-UNREADABLE;
  - check(): one read per file system (two roots on one mount: one quota read), the codes of every file system;
  - Gate: require() refuses (record.Refusal) with the codes first; before_unit() reads again only after RECHECK_S; once low it
    stays shut (sticky) even after the room comes back;
  - Gate.state: the last verdict without a read;
  - the floors: section space has small defaults pending G2 (1 TiB, 100000 files, 10 TB: a 1 kB writer at 110 TB free has
    room), the photometry campaign keeps its own (2 TiB, 200000 files, 120 TB; read before every unit); the campaign's own
    read gives the same verdict, text and values as core/space.lustre with its section, and in a run with the cli's gate the
    campaign reports that gate's verdict and reads no lfs (one gate);
  - the wiring: cli.main runs the gate before a writer step's run() (exit 2, nothing written, the run summary says why), not
    for a dry run, not for a step that writes nothing, not for a step with SPACE_GATE = False (retention, notify, status,
    serve), with a step's space_gate(args, out) roots and floors (archive push/sync: the archive root too, its moves none;
    photometry --campaign: its file system and floors); record.Run.unit refuses a later unit once the room is low (the units
    before it stay done): the refused unit's record is written with its cell meta, the run ends refused even when the step
    goes on after the refusal, and the runner's per-cell handling does not take it as a cell refusal (the chain item stops);
  - the command `python -m exopipe.phase3.core.space`: exit 0 with ROOM, 2 with REFUSED and the codes.
"""
import contextlib
import io
import json
import os
import pwd
import shutil
import sys
import tempfile
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exopipe.phase3 import cli  # noqa: E402
from exopipe.phase3.config import CFG  # noqa: E402
from exopipe.phase3.core import record  # noqa: E402
from exopipe.phase3.core import space as S  # noqa: E402
from exopipe.phase3.lib.photometry import campaign as CP  # noqa: E402

FAILS = []
KIB_PER_TIB = 2 ** 30


def check(name, cond, detail=''):
    print(('PASS ' if cond else 'FAIL ') + name + ('' if cond else '  -- %s' % (detail,)), flush=True)
    if not cond:
        FAILS.append(name)


class Patch:
    def __init__(self):
        self.saved = []

    def set(self, obj, name, value):
        self.saved.append((obj, name, getattr(obj, name, None), hasattr(obj, name)))
        setattr(obj, name, value)

    def undo(self):
        for obj, name, value, had in reversed(self.saved):
            if had:
                setattr(obj, name, value)
            else:
                delattr(obj, name)
        self.saved = []


def floors(**kw):
    """A floors object of fixed test values (the photometry campaign's: 2 TiB, 200000 files, 120 TB; no local floor, no
    project), overridden by kw - the mechanics are tested against these, whatever the config's values."""
    d = dict(LFS='lfs', LFS_TIMEOUT_S=120, QUOTA_MIN_FREE_BYTES=2 * 2 ** 40, QUOTA_MIN_FREE_FILES=200000,
             FS_MIN_FREE_BYTES=120 * 10 ** 12, LOCAL_MIN_FREE_BYTES=0, PROJECT_ID=None, RECHECK_S=300)
    d.update(kw)
    return types.SimpleNamespace(**d)


def fake_lfs(kb=1 * KIB_PER_TIB, kq=0, kl=4 * KIB_PER_TIB, fl=10, fq=0, fll=10 ** 6, avail=int(200e12 / 1024), rq=0, rd=0,
             pkb=0, pkl=0, rp=0, fs=None):
    """A stub `lfs`: quota (user or project) and df answers in KiB, recording each argv."""
    calls = []

    def lfs(argv):
        calls.append(list(argv))
        mnt = fs or argv[-1]
        if argv[0] == 'quota' and argv[2] == '-p':
            return rp, ('Disk quotas for prj %s:\n %s %d 0 %d - 1 0 0 -\n' % (argv[3], mnt, pkb, pkl)), \
                ('' if rp == 0 else 'Permission denied')
        if argv[0] == 'quota':
            return rq, ('%s %d %d %d - %d %d %d -\n' % (mnt, kb, kq, kl, fl, fq, fll)) if rq == 0 else '', \
                ('' if rq == 0 else 'Permission denied')
        return rd, ('UUID 1K-blocks Used Available Use%% Mounted on\nlustre-OST0000_UUID 1 1 1 1%% %s[OST:0]\n\n'
                    'filesystem_summary:  3822424680448 3619706371072 %d  95%% %s\n' % (mnt, avail, mnt)), ''
    lfs.calls = calls
    return lfs


def fake_statvfs(free):
    def st(path):
        return types.SimpleNamespace(f_bavail=free // 4096, f_frsize=4096)
    return st


def world(d):
    """Two mount points in the scratch area: <d>/mntL (lustre, over an autofs entry of the same path) and <d>/mnt\\040x (xfs:
    an escaped space), and a mount table that says so."""
    lus, loc = os.path.join(d, 'mntL'), os.path.join(d, 'mnt x')
    for p in (lus, loc, os.path.join(lus, 'out'), os.path.join(lus, 'work'), os.path.join(loc, 'out')):
        os.makedirs(p, exist_ok=True)
    mi = os.path.join(d, 'mountinfo')
    with open(mi, 'w') as f:
        f.write('22 1 253:0 / / rw,relatime shared:1 - xfs /dev/mapper/root rw\n')
        f.write('40 22 0:40 / %s rw,relatime shared:20 - autofs systemd-1 rw\n' % lus)
        f.write('41 40 0:52 / %s rw,relatime shared:21 - lustre 10.41.0.1@o2ib:/lustre rw,flock\n' % lus)
        f.write('42 22 259:1 / %s rw,relatime shared:22 - xfs /dev/nvme0n1 rw\n' % loc.replace(' ', '\\040'))
    return lus, loc, mi


def test_parsers():
    check('the campaign\'s parsers are core/space\'s (one implementation)', CP.parse_quota is S.parse_quota
          and CP.parse_df is S.parse_df)
    q = S.parse_quota('/lustre 231098215438  237296943104 237296943104 - 36161145 40000000 40000000 -', '/lustre')
    check('lfs quota parsed (KiB and files, used and limits)',
          q == dict(kbytes=231098215438, kbytes_limit=237296943104, files=36161145, files_limit=40000000), q)
    check('lfs df: the filesystem_summary line\'s fourth field', S.parse_df(
        'UUID x\nlustre-MDT0000_UUID 9 9 9 1% /lustre[MDT:0]\nfilesystem_summary:  3 2 202712948736  95% /lustre\n')
        == 202712948736)
    check('lfs df without the summary line: None', S.parse_df('UUID x\n') is None)


def test_mount(d):
    lus, loc, mi = world(d)
    check('a Lustre root: the lustre entry over the autofs entry of the same mount point (the last on a tie)',
          S.mount_of(os.path.join(lus, 'out'), mi) == (lus, 'lustre'), S.mount_of(os.path.join(lus, 'out'), mi))
    check('a root not made yet: its existing parent\'s file system',
          S.mount_of(os.path.join(lus, 'not', 'made', 'yet'), mi) == (lus, 'lustre'))
    check('an octal-escaped mount point (\\040 = space)', S.mount_of(os.path.join(loc, 'out'), mi) == (loc, 'xfs'),
          S.mount_of(os.path.join(loc, 'out'), mi))
    check('a path under no other mount: /', S.mount_of(d, mi) == ('/', 'xfs'), S.mount_of(d, mi))
    check('an unreadable mount table: None', S.mount_of(d, os.path.join(d, 'no_such_mountinfo')) is None)


def test_lustre(d):
    lus, loc, mi = world(d)
    F = floors(QUOTA_MIN_FREE_BYTES=2 * 2 ** 40, QUOTA_MIN_FREE_FILES=200000, FS_MIN_FREE_BYTES=120 * 10 ** 12)
    user = pwd.getpwuid(os.geteuid()).pw_name
    lfs = fake_lfs()
    ok, text, vals, codes = S.lustre(lus, F, lfs=lfs)
    check('room: 3 TiB and 999990 files of quota headroom, 200 TB free (floors 2 TiB, 200000 files, 120 TB)',
          ok and not codes and vals['quota_free_bytes'] == 3 * 2 ** 40 and vals['quota_free_files'] == 999990
          and vals['fs_free_bytes'] == int(200e12 / 1024) * 1024, (ok, text, vals))
    check('the reads: the RUN USER\'s quota (%s) and lfs df of the mount point; no project read without PROJECT_ID' % user,
          lfs.calls == [['quota', '-q', '-u', user, lus], ['df', lus]], lfs.calls)
    for what, kw, code in (('quota bytes (1.9 TiB headroom)', dict(kb=int(2.1 * KIB_PER_TIB)), 'SPACE-QUOTA'),
                           ('quota files (199999 headroom)', dict(fl=10 ** 6 - 199999), 'SPACE-QUOTA'),
                           ('free bytes (119 TB)', dict(avail=int(119e12 / 1024)), 'SPACE-FS')):
        ok, text, _v, codes = S.lustre(lus, F, lfs=fake_lfs(**kw))
        check('floor %s: refused %s' % (what, code), not ok and codes == [code], (ok, text, codes))
    ok, text, _v, codes = S.lustre(lus, F, lfs=fake_lfs(kb=int(2.1 * KIB_PER_TIB), avail=int(119e12 / 1024)))
    check('two floors low: both codes', not ok and codes == ['SPACE-QUOTA', 'SPACE-FS'], codes)
    ok, text, vals, codes = S.lustre(lus, F, lfs=fake_lfs(kq=0, kl=0, fq=0, fll=0))
    check('limits of 0: no limit (headroom None), the free-space floor still applies', ok and vals['quota_free_bytes'] is None
          and vals['quota_free_files'] is None and 'unlimited' in text, (ok, text))
    for what, kw in (('quota exit 1 (Permission denied)', dict(rq=1)), ('df exit 1', dict(rd=1))):
        ok, text, _v, codes = S.lustre(lus, F, lfs=fake_lfs(**kw))
        check('unreadable (%s): SPACE-UNREADABLE' % what, not ok and codes == ['SPACE-UNREADABLE'], (text, codes))

    def boom(argv):
        raise OSError('no lfs')
    ok, text, _v, codes = S.lustre(lus, F, lfs=boom)
    check('lfs that cannot run: SPACE-UNREADABLE', not ok and codes == ['SPACE-UNREADABLE'] and 'could not run' in text, text)
    ok, text, vals, codes = S.lustre(lus, F, lfs=fake_lfs(), need_bytes=int(0.6 * 2 ** 40), booked_bytes=int(0.5 * 2 ** 40))
    check('a booking: 3 TiB headroom - 0.5 booked - 0.6 needed = 1.9 TiB < 2 TiB: SPACE-QUOTA', not ok
          and codes == ['SPACE-QUOTA'] and 'booked' in text and vals['need_bytes'] == int(0.6 * 2 ** 40), (text, codes))
    ok, text, _v, codes = S.lustre(lus, F, lfs=fake_lfs(), need_bytes=int(0.9 * 2 ** 40))
    check('... 0.9 TiB needed alone fits (2.1 TiB left)', ok, text)
    ok, text, _v, codes = S.lustre(lus, F, lfs=fake_lfs(), need_files=800000)
    check('need_files: 999990 - 800000 files < 200000: SPACE-QUOTA', not ok and codes == ['SPACE-QUOTA'], (text, codes))
    ok, text, _v, codes = S.lustre(lus, F, lfs=fake_lfs(kl=40 * KIB_PER_TIB, avail=int(130e12 / 1024)),
                                   need_bytes=11 * 10 ** 12)
    check('need_bytes counts against the free space too (130 - 11 TB < 120 TB; 39 TiB of quota headroom): SPACE-FS only',
          not ok and codes == ['SPACE-FS'],
          (text, codes))
    lfs = fake_lfs(pkb=10, pkl=3 * KIB_PER_TIB)
    ok, text, vals, codes = S.lustre(lus, floors(PROJECT_ID=1002), lfs=lfs)
    check('PROJECT_ID set: the project quota is read (lfs quota -q -p 1002) and reported', ok
          and lfs.calls[-1] == ['quota', '-q', '-p', '1002', lus] and vals['project_free_bytes'] is not None
          and 'project 1002' in text, (lfs.calls, text))
    ok, text, _v, codes = S.lustre(lus, floors(PROJECT_ID=1002), lfs=fake_lfs(pkb=int(1.5 * KIB_PER_TIB), pkl=3 * KIB_PER_TIB))
    check('... a project headroom of 1.5 TiB < 2 TiB: SPACE-PROJECT', not ok and codes == ['SPACE-PROJECT'], (text, codes))
    ok, text, _v, codes = S.lustre(lus, floors(PROJECT_ID=1002), lfs=fake_lfs(rp=1))
    check('... an unreadable project quota: SPACE-UNREADABLE', not ok and codes == ['SPACE-UNREADABLE'], (text, codes))
    lfs = fake_lfs()
    S.lustre(lus, F, lfs=lfs, user='pipeline')
    check('user=pipeline reads that user\'s quota', lfs.calls[0] == ['quota', '-q', '-u', 'pipeline', lus], lfs.calls)


def test_local_and_check(d):
    lus, loc, mi = world(d)
    F = floors(LOCAL_MIN_FREE_BYTES=100 * 10 ** 9)
    ok, text, vals, codes = S.local(os.path.join(loc, 'out'), F, statvfs=fake_statvfs(300 * 10 ** 9))
    check('local: 300 GB free >= 100 GB floor', ok and not codes and vals['free_bytes'] == 300 * 10 ** 9 // 4096 * 4096, text)
    ok, text, vals, codes = S.local(os.path.join(loc, 'out'), F, statvfs=fake_statvfs(90 * 10 ** 9))
    check('local: 90 GB free < 100 GB: SPACE-LOCAL', not ok and codes == ['SPACE-LOCAL'], (text, codes))
    ok, text, vals, codes = S.local(os.path.join(loc, 'out'), F, statvfs=fake_statvfs(300 * 10 ** 9),
                                    need_bytes=150 * 10 ** 9, booked_bytes=60 * 10 ** 9)
    check('local: a booking that does not fit (300 - 60 - 150 GB < 100 GB): SPACE-LOCAL', not ok and codes == ['SPACE-LOCAL']
          and 'booked' in text, text)

    def bad_statvfs(p):
        raise OSError('stale handle')
    ok, text, vals, codes = S.local(os.path.join(loc, 'out'), F, statvfs=bad_statvfs)
    check('local: statvfs unreadable: SPACE-UNREADABLE', not ok and codes == ['SPACE-UNREADABLE'], text)

    lfs = fake_lfs()
    ok, text, vals, codes = S.check([os.path.join(lus, 'out'), os.path.join(lus, 'work'), os.path.join(loc, 'out')],
                                    floors(), lfs=lfs, mountinfo=mi, statvfs=fake_statvfs(300 * 10 ** 9))
    check('check: two roots on one Lustre mount read it once, the local root by statvfs', ok
          and [c[0] for c in lfs.calls] == ['quota', 'df'] and len(vals) == 2
          and vals[1].get('fstype') == 'xfs', (lfs.calls, vals))
    ok, text, vals, codes = S.check([os.path.join(lus, 'out'), os.path.join(loc, 'out')],
                                    floors(LOCAL_MIN_FREE_BYTES=400 * 10 ** 9), lfs=fake_lfs(avail=int(100e12 / 1024)),
                                    mountinfo=mi, statvfs=fake_statvfs(300 * 10 ** 9))
    check('check: the codes of every file system (SPACE-FS and SPACE-LOCAL)', not ok
          and codes == ['SPACE-FS', 'SPACE-LOCAL'], codes)
    ok, text, vals, codes = S.check([os.path.join(lus, 'out')], floors(), lfs=fake_lfs(),
                                    mountinfo=os.path.join(d, 'no_such_mountinfo'))
    check('check: an unreadable mount table: SPACE-UNREADABLE', not ok and codes == ['SPACE-UNREADABLE'], text)


def test_gate(d):
    lus, loc, mi = world(d)
    state = {'avail': int(200e12 / 1024)}
    calls = []

    def lfs(argv):
        calls.append(argv)
        return fake_lfs(avail=state['avail'])(argv)
    G = S.Gate([os.path.join(lus, 'out')], floors(), recheck_s=3600, lfs=lfs, mountinfo=mi)
    G.require()
    n = len(calls)
    G.before_unit()
    G.before_unit()
    check('Gate: room at the start; before_unit within RECHECK_S reads nothing again', n == 2 and len(calls) == 2, calls)
    st = G.state()
    check('Gate.state: the last verdict, reading nothing (ok, its text, its values)', st[0] is True and 'free' in st[1]
          and st[2] and len(calls) == 2, (st, calls))
    G.recheck_s = 0
    state['avail'] = int(100e12 / 1024)
    try:
        G.before_unit()
        check('Gate: low after RECHECK_S -> Refusal', False)
    except record.Refusal as e:
        check('Gate: low after RECHECK_S -> Refusal naming SPACE-FS first', str(e).startswith('SPACE-FS: '), str(e))
    state['avail'] = int(200e12 / 1024)
    n = len(calls)
    st = G.state()
    check('Gate.state once low: not ok, the refusal text, reading nothing', st[0] is False and st[1].startswith('SPACE-FS')
          and len(calls) == n, st)
    for fn in (G.before_unit, G.require):
        try:
            fn()
            check('Gate: sticky (%s)' % fn.__name__, False)
        except record.Refusal as e:
            check('Gate: sticky - %s refuses after the room came back, without reading again' % fn.__name__,
                  str(e).startswith('SPACE-FS') and len(calls) == n, (str(e), len(calls)))
    G2 = S.Gate([os.path.join(lus, 'out')], floors(), lfs=fake_lfs(rq=1), mountinfo=mi)
    check('Gate.state before the first read: not ok', G2.state()[0] is False, G2.state())
    try:
        G2.require()
        check('Gate: unreadable quota at the start refuses', False)
    except record.Refusal as e:
        check('Gate: unreadable quota at the start refuses SPACE-UNREADABLE', str(e).startswith('SPACE-UNREADABLE'), str(e))
    try:
        S.require([os.path.join(lus, 'out')], floors(), lfs=fake_lfs(fl=10 ** 6 - 5), mountinfo=mi)
        check('require: low files refuses', False)
    except record.Refusal as e:
        check('require: low files refuses SPACE-QUOTA', str(e).startswith('SPACE-QUOTA'), str(e))


def test_floors_and_campaign(d):
    cc = CFG.photometry_campaign
    check('the floors of section space are small defaults pending G2: 1 TiB and 100000 files of quota headroom, 10 TB free',
          (CFG.space.QUOTA_MIN_FREE_BYTES, CFG.space.QUOTA_MIN_FREE_FILES, CFG.space.FS_MIN_FREE_BYTES) ==
          (2 ** 40, 100000, 10 * 10 ** 12) and 'PENDING' in (CFG.space.__doc__ or ''))
    check('the photometry campaign keeps its own floors (2 TiB, 200000 files, 120 TB) and reads before every unit '
          '(RECHECK_S 0)', (cc.QUOTA_MIN_FREE_BYTES, cc.QUOTA_MIN_FREE_FILES, cc.FS_MIN_FREE_BYTES, cc.RECHECK_S) ==
          (2 * 2 ** 40, 200000, 120 * 10 ** 12, 0))
    kb110 = int(110e12 / 1024)
    ok_s = S.lustre('/lustre', CFG.space, lfs=fake_lfs(avail=kb110), user='u', need_bytes=1000)[0]
    ok_c = S.lustre('/lustre', cc, lfs=fake_lfs(avail=kb110), user='u', need_bytes=1000)
    check('a 1 kB writer at 110 TB free: room with the floors of section space; the campaign\'s 120 TB refuses SPACE-FS',
          ok_s and not ok_c[0] and 'SPACE-FS' in ok_c[3], (ok_s, ok_c))
    check('no project quota and no local floor set by default (PROJECT_ID None, LOCAL_MIN_FREE_BYTES 0)',
          CFG.space.PROJECT_ID is None and CFG.space.LOCAL_MIN_FREE_BYTES == 0)
    check('space is not a shared section (a floor changes no product: no resume hash moves)',
          'space' not in __import__('exopipe.phase3.config', fromlist=['SHARED_SECTIONS']).SHARED_SECTIONS)
    for kw in (dict(), dict(kb=int(2.1 * KIB_PER_TIB)), dict(fl=10 ** 6 - 199999), dict(avail=int(119e12 / 1024)),
               dict(kq=0, kl=0, fq=0, fll=0), dict(rq=1), dict(rd=1)):
        a = CP.space(fake_lfs(fs=cc.SPACE_FILESYSTEM, **kw))
        ok, text, vals, _c = S.lustre(cc.SPACE_FILESYSTEM, cc, lfs=fake_lfs(fs=cc.SPACE_FILESYSTEM, **kw), user=CP.run_user())
        check('campaign gate = core/space.lustre with photometry_campaign (%s)' % (kw or 'room'), a == (ok, text, vals),
              (a, (ok, text, vals)))
    calls = []

    def counting(argv):
        calls.append(argv)
        return fake_lfs(fs=cc.SPACE_FILESYSTEM)(argv)
    G = S.Gate([cc.SPACE_FILESYSTEM], cc, lfs=counting, mountinfo=_lustre_mountinfo(d, cc.SPACE_FILESYSTEM))
    G.require()
    n = len(calls)
    got = [CP.run_space(types.SimpleNamespace(space_gate=G)) for _ in range(3)]
    check('one gate: in a run with the cli\'s space gate, the campaign reports the gate\'s verdict and reads no lfs itself',
          n == 2 and len(calls) == n and all(g == G.state() for g in got) and got[0][0] is True, (n, len(calls), got[0]))


def _lustre_mountinfo(d, path):
    """A mount table file in the test's directory ``d`` naming ``path`` a lustre mount."""
    p = os.path.join(d, 'mountinfo_lustre')
    with open(p, 'w') as f:
        f.write('1 0 8:1 / / rw - xfs /dev/sda1 rw\n41 1 0:52 / %s rw,relatime - lustre 10.0.0.1@o2ib:/fs rw\n' % path)
    return p


STEP = 'zz_space_gate_test'


def fake_step(writes=True, space_gate=None, units=2, swallow=False, hook=None):
    m = types.ModuleType('exopipe.phase3.steps.' + STEP)
    m.SUMMARY, m.TIERS, m.SELECTORS, m.CONFIG = 'test step', (), (), ()
    m.WRITES, m.PRODUCTS = writes, False
    if space_gate is not None:
        m.SPACE_GATE = space_gate
    if hook is not None:
        m.space_gate = hook
    m.add_arguments = lambda p: None
    m.default_out = lambda args: None
    m.ran = []

    def run(args, run):
        m.ran.append(True)
        if run.dry_run or not m.WRITES:
            return record.EXIT_DONE
        os.makedirs(run.out_root, exist_ok=True)
        for i in range(units):
            try:
                with run.unit('u%d' % i, cell='04h_2025-10-14', subband='55MHz') as u:
                    p = os.path.join(run.out_root, 'u%d.txt' % i)
                    with open(p, 'w') as f:
                        f.write('x')
                    u.add_output(p, 'zz_space_gate_test_file', test=False)
            except record.Refusal:
                if not swallow:          # a step that goes on to its next unit after a refusal (it must still end refused)
                    raise
        return run.exit_code()
    m.run = run
    return m


def test_wiring(d):
    P = Patch()
    reads = []
    state = {'ok': [True]}

    def fake_check(roots, floors=None, **kw):
        reads.append(list(roots))
        state['floors'] = floors
        ok = state['ok'].pop(0) if len(state['ok']) > 1 else state['ok'][0]
        return (True, 'room', [], []) if ok else (False, 'pipeline quota headroom 1.00 TiB < 2.00 TiB', [], ['SPACE-QUOTA'])
    try:
        P.set(cli, 'STEPS', list(cli.STEPS) + [(STEP, 'test step', 'tools')])
        P.set(S, 'check', fake_check)

        def go(mod, out, *extra, outcome=None):
            sys.modules['exopipe.phase3.steps.' + STEP] = mod
            err = io.StringIO()
            fd = os.open(outcome, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o600) if outcome else None
            if fd is not None:
                os.environ['EXOPIPE3_STEP_OUTCOME_FD'] = str(fd)
            try:
                with contextlib.redirect_stderr(err):
                    rc = cli.main([STEP, '--out', out, '--cores', 'none'] + list(extra))
            finally:
                CFG.reset()
                sys.modules.pop('exopipe.phase3.steps.' + STEP, None)
                os.environ.pop('EXOPIPE3_STEP_OUTCOME_FD', None)
                if fd is not None:
                    os.close(fd)
            return rc, err.getvalue()

        def outcome_of(path):
            from exopipe.phase3.core import runner
            with open(path, 'rb') as f:
                return runner.read_outcome(f)

        out = os.path.join(d, 'w_refused')
        state['ok'] = [False]
        m = fake_step()
        rc, err = go(m, out, '--work', os.path.join(d, 'w_work'))
        check('cli: low room before the step -> exit 2, the step\'s run() not called, nothing written',
              rc == 2 and not m.ran and not os.path.exists(os.path.join(out, 'u0.txt')) and 'SPACE-QUOTA' in err, (rc, err))
        check('cli: the gate reads --out and --work', reads and reads[-1] == [out, os.path.join(d, 'w_work')], reads)
        runs = os.path.join(out, '_records', STEP, '_runs')
        summ = [json.load(open(os.path.join(runs, f))) for f in os.listdir(runs)] if os.path.isdir(runs) else []
        check('cli: the run summary keeps why (reason refused: SPACE-QUOTA ..., no unit)', len(summ) == 1
              and str(summ[0].get('reason')).startswith('refused: SPACE-QUOTA') and not summ[0].get('units'), summ)
        n = len(reads)
        m = fake_step()
        rc, err = go(m, os.path.join(d, 'w_dry'), '--dry-run')
        check('cli: a dry run reads no room', len(reads) == n, reads[n:])
        m = fake_step(writes=False)
        rc, err = go(m, os.path.join(d, 'w_ro'))
        check('cli: a step that writes nothing reads no room', len(reads) == n and m.ran, (reads[n:], rc))
        m = fake_step(space_gate=False)
        rc, err = go(m, os.path.join(d, 'w_free'))
        check('cli: a step with SPACE_GATE = False (retention) reads no room', len(reads) == n and m.ran and rc == 0, (rc, err))
        check('cli: the floors of section space by default', state.get('floors') is CFG.space, state.get('floors'))
        P.set(CFG.space, 'RECHECK_S', 0)
        from exopipe.phase3.core import runner
        for swallow in (False, True):
            state['ok'] = [True, True, False]          # start, unit u0, unit u1
            out = os.path.join(d, 'w_mid_%s' % swallow)
            m = fake_step(units=3, swallow=swallow)
            oc = os.path.join(d, 'outcome_%s.json' % swallow)
            rc, err = go(m, out, outcome=oc)
            how = 'a step that goes on after a refused unit' if swallow else 'the refusal propagates'
            check('Run.unit (%s): room at the start and before u0; low before u1 -> exit 2, u0 written, u1 and u2 not'
                  % how, rc == 2 and os.path.exists(os.path.join(out, 'u0.txt'))
                  and not os.path.exists(os.path.join(out, 'u1.txt')) and not os.path.exists(os.path.join(out, 'u2.txt'))
                  and 'SPACE-QUOTA' in err, (rc, err))
            rec = os.path.join(out, '_records', STEP, 'u1.json')
            r1 = json.load(open(rec)) if os.path.exists(rec) else {}
            check('Run.unit (%s): the refused unit u1 has its record - status refused, reason SPACE-QUOTA first, its cell '
                  'and subband meta' % how, r1.get('status') == 'refused' and str(r1.get('reason')).startswith('SPACE-QUOTA')
                  and (r1.get('meta') or {}).get('cell') == '04h_2025-10-14'
                  and (r1.get('meta') or {}).get('subband') == '55MHz', r1)
            o = outcome_of(oc)
            u1 = [u for u in (o or {}).get('units', []) if u.get('unit') == 'u1']
            check('runner outcome (%s): u1 refused with its cell, the run reason refused: SPACE-QUOTA, and the runner\'s '
                  'per-cell handling does NOT take it as a cell refusal (cell_refusals None: the chain item stops, retried '
                  'when there is room, never the hour\'s cells recorded as gaps)' % how,
                  o and u1 and u1[0]['status'] == 'refused' and u1[0]['cell'] == '04h_2025-10-14'
                  and str(o.get('run_reason')).startswith('refused: SPACE-QUOTA') and runner.cell_refusals(o, rc) is None,
                  o)
            if swallow:
                check('... the step that goes on: u2 refused too (sticky), never written', any(
                    u.get('unit') == 'u2' and u.get('status') == 'refused' for u in (o or {}).get('units', [])), o)
        state['ok'] = [True]
        n = len(reads)
        m = fake_step(hook=lambda args, out: ([out, '/lustre'], 'photometry_campaign'))
        rc, err = go(m, os.path.join(d, 'w_hook'), '--work', os.path.join(d, 'w_hook_work'))
        check('cli: a step\'s space_gate(args, out) picks the roots (plus --work) and the floors section',
              rc == 0 and reads[n] == [os.path.join(d, 'w_hook'), '/lustre', os.path.join(d, 'w_hook_work')]
              and state['floors'] is CFG.photometry_campaign, (rc, reads[n:], err[-300:]))
        n = len(reads)
        m = fake_step(hook=lambda args, out: None)
        rc, err = go(m, os.path.join(d, 'w_hook_none'))
        check('cli: space_gate(args, out) returning None: no gate for this invocation', rc == 0 and len(reads) == n, rc)
        import argparse
        from exopipe.phase3.steps import archive, notify, photometry, retention, serve, status
        for mod in (notify, status, serve, retention):
            check('%s: no space gate (SPACE_GATE = False)' % mod.__name__.rsplit('.', 1)[1],
                  mod.SPACE_GATE is False and cli.space_gate_of(mod, argparse.Namespace(work=None), '/o') is None)
        A = lambda a, **k: argparse.Namespace(action=a, archive_root=None, work=None, **k)  # noqa: E731
        check('archive push and sync: gated on --out AND the archive root (paths.PHASE3_ARCHIVE_ROOT, or --archive-root)',
              cli.space_gate_of(archive, A('push'), '/o') == (['/o', os.path.abspath(CFG.paths.PHASE3_ARCHIVE_ROOT)],
                                                               'space')
              and cli.space_gate_of(archive, argparse.Namespace(action='sync', archive_root='/m', work=None), '/o')
              == (['/o', '/m'], 'space'))
        check('archive moves and audits (quarantine, retire-excluded, retire-refused, rollback, supersede, quarantine-rows, '
              'audit): no space gate', all(cli.space_gate_of(archive, A(a), '/o') is None for a in (
                  'quarantine', 'retire-excluded', 'retire-refused', 'rollback', 'supersede', 'quarantine-rows', 'audit')))
        check('photometry --campaign: --out and the campaign\'s file system, with the photometry_campaign floors; any other '
              'photometry run: --out with the space floors',
              cli.space_gate_of(photometry, argparse.Namespace(campaign=True, work=None), '/o') ==
              (['/o', CFG.photometry_campaign.SPACE_FILESYSTEM], 'photometry_campaign')
              and cli.space_gate_of(photometry, argparse.Namespace(campaign=False, work='/w'), '/o') == (['/o', '/w'],
                                                                                                          'space'))
        state['ok'] = [True]
        out = os.path.join(d, 'w_ok')
        m = fake_step()
        rc, err = go(m, out)
        check('cli: room throughout -> the step runs, exit 0', rc == 0 and os.path.exists(os.path.join(out, 'u1.txt')),
              (rc, err))
    finally:
        P.undo()
        CFG.reset()


def test_command(d):
    P = Patch()
    try:
        P.set(S, 'check', lambda roots, floors=None, **kw: (True, '%s free 300.0 GB' % roots[0], [{'free_bytes': 1}], []))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = S.main(['--path', d])
        check('python -m exopipe.phase3.core.space: exit 0, ROOM', rc == 0 and out.getvalue().startswith('ROOM: '),
              out.getvalue())
        P.set(S, 'check', lambda roots, floors=None, **kw: (False, 'low', [], ['SPACE-FS']))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = S.main(['--path', d, '--need-bytes', '5'])
        check('... exit 2, REFUSED with the code', rc == 2 and out.getvalue().startswith('REFUSED SPACE-FS: low'),
              out.getvalue())
    finally:
        P.undo()


def main():
    base = os.environ.get('EXOPIPE3_TEST_SCRATCH') or None
    d = tempfile.mkdtemp(prefix='test_space_gate_', dir=base)
    try:
        test_parsers()
        test_mount(d)
        test_lustre(d)
        test_local_and_check(d)
        test_gate(d)
        test_floors_and_campaign(d)
        test_wiring(d)
        test_command(d)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print('%d check(s) failed: %s' % (len(FAILS), FAILS) if FAILS else 'all checks passed', flush=True)
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
