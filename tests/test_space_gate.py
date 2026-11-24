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
  - the floors: the shared section's Lustre floors and lfs settings are the photometry campaign's (until Gregg sets them, G2);
    the campaign's gate gives the same verdict, text and values as core/space.lustre with its section;
  - the wiring: cli.main runs the gate before a writer step's run() (exit 2, nothing written, the run summary says why), not
    for a dry run, not for a step that writes nothing, not for a step with SPACE_GATE = False; record.Run.unit refuses a
    later unit once the room is low (the units before it stay done);
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
    """A floors object: the shared section's values, overridden by kw."""
    d = {k: getattr(CFG.space, k) for k in ('LFS', 'LFS_TIMEOUT_S', 'QUOTA_MIN_FREE_BYTES', 'QUOTA_MIN_FREE_FILES',
                                            'FS_MIN_FREE_BYTES', 'LOCAL_MIN_FREE_BYTES', 'PROJECT_ID', 'RECHECK_S')}
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
    G.recheck_s = 0
    state['avail'] = int(100e12 / 1024)
    try:
        G.before_unit()
        check('Gate: low after RECHECK_S -> Refusal', False)
    except record.Refusal as e:
        check('Gate: low after RECHECK_S -> Refusal naming SPACE-FS first', str(e).startswith('SPACE-FS: '), str(e))
    state['avail'] = int(200e12 / 1024)
    n = len(calls)
    for fn in (G.before_unit, G.require):
        try:
            fn()
            check('Gate: sticky (%s)' % fn.__name__, False)
        except record.Refusal as e:
            check('Gate: sticky - %s refuses after the room came back, without reading again' % fn.__name__,
                  str(e).startswith('SPACE-FS') and len(calls) == n, (str(e), len(calls)))
    G2 = S.Gate([os.path.join(lus, 'out')], floors(), lfs=fake_lfs(rq=1), mountinfo=mi)
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


def test_floors_and_campaign():
    cc = CFG.photometry_campaign
    same = [k for k in ('LFS', 'LFS_TIMEOUT_S', 'QUOTA_MIN_FREE_BYTES', 'QUOTA_MIN_FREE_FILES', 'FS_MIN_FREE_BYTES')
            if getattr(CFG.space, k) != getattr(cc, k)]
    check('the shared floors and lfs settings are the photometry campaign\'s (awaiting Gregg\'s values, G2)', not same, same)
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


STEP = 'zz_space_gate_test'


def fake_step(writes=True, space_gate=None, units=2):
    m = types.ModuleType('exopipe.phase3.steps.' + STEP)
    m.SUMMARY, m.TIERS, m.SELECTORS, m.CONFIG = 'test step', (), (), ()
    m.WRITES, m.PRODUCTS = writes, False
    if space_gate is not None:
        m.SPACE_GATE = space_gate
    m.add_arguments = lambda p: None
    m.default_out = lambda args: None
    m.ran = []

    def run(args, run):
        m.ran.append(True)
        if run.dry_run or not m.WRITES:
            return record.EXIT_DONE
        os.makedirs(run.out_root, exist_ok=True)
        for i in range(units):
            with run.unit('u%d' % i) as u:
                p = os.path.join(run.out_root, 'u%d.txt' % i)
                with open(p, 'w') as f:
                    f.write('x')
                u.add_output(p, 'zz_space_gate_test_file', test=False)
        return run.exit_code()
    m.run = run
    return m


def test_wiring(d):
    P = Patch()
    reads = []
    state = {'ok': [True]}

    def fake_check(roots, floors=None, **kw):
        reads.append(list(roots))
        ok = state['ok'].pop(0) if len(state['ok']) > 1 else state['ok'][0]
        return (True, 'room', [], []) if ok else (False, 'pipeline quota headroom 1.00 TiB < 2.00 TiB', [], ['SPACE-QUOTA'])
    try:
        P.set(cli, 'STEPS', list(cli.STEPS) + [(STEP, 'test step', 'tools')])
        P.set(S, 'check', fake_check)

        def go(mod, out, *extra):
            sys.modules['exopipe.phase3.steps.' + STEP] = mod
            err = io.StringIO()
            try:
                with contextlib.redirect_stderr(err):
                    rc = cli.main([STEP, '--out', out, '--cores', 'none'] + list(extra))
            finally:
                CFG.reset()
                sys.modules.pop('exopipe.phase3.steps.' + STEP, None)
            return rc, err.getvalue()

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
        P.set(CFG.space, 'RECHECK_S', 0)
        state['ok'] = [True, True, False]          # start, unit u0, unit u1
        out = os.path.join(d, 'w_mid')
        m = fake_step(units=3)
        rc, err = go(m, out)
        check('Run.unit: room at the start and before u0; low before u1 -> exit 2, u0 written, u1 and u2 not',
              rc == 2 and os.path.exists(os.path.join(out, 'u0.txt')) and not os.path.exists(os.path.join(out, 'u1.txt'))
              and not os.path.exists(os.path.join(out, 'u2.txt')) and 'SPACE-QUOTA' in err, (rc, err))
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
        test_floors_and_campaign()
        test_wiring(d)
        test_command(d)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print('%d check(s) failed: %s' % (len(FAILS), FAILS) if FAILS else 'all checks passed', flush=True)
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
