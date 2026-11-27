"""The pre-search frame quarantine's records (the refactor chat's decision of 2026-11-27, cutover row 64 step 2): a NEW round,
Run_SEARCH_QUARANTINE_<launch date>, whose manifest of record lives under archive.RECORDS_ROOT, never under paths.OPS_ROOT;
production's p3s5_state manifests and launch epoch are read only.

Checks, data-free (a scratch archive, ops root and records root; the scratch archive is paths.PHASE3_ARCHIVE_ROOT, so the
production branch of lib/archive/quarantine runs):
  round        round_for: 'launch-date' = the launch epoch's UTC date; a literal passes; a non-name refuses
  launch       launch_epoch reads archive_quarantine.LAUNCH_EPOCH_FILE under the records root; unset refuses
  apply        quarantine_for_search on the production branch moves the earlier frames into the new round, writes the round's
               record and one manifest of record per (cell, Stokes) under the records root (the content test passes on it),
               and leaves the ops root byte for byte as it was (production's manifests and launch epoch untouched, nothing
               added); the helper is never run
  new round    a round production used (its Run in the archive, no v2 record: 20261004, the 11-13 launch's 20261113), a round
               of another launch epoch, and a records root under the ops root each refuse before anything moves
  search       the search's quarantine stage (window): reads the v2 launch epoch, plans into Run_SEARCH_QUARANTINE_<launch
               date> (the only round the engine then takes a predecessor from), refuses production's launch epoch (an earlier
               round); it never names paths.OPS_ROOT
  begin-launch the operator step that writes the launch file (decision (1)): plan writes nothing; apply writes the epoch
               atomically (no temporary file left), an output of the unit with its content test; an existing file refuses, named; --retire-launch moves it first into its
               own round's records directory beside ROUND.json; a second launch on one UTC date, an occupied retirement
               place, a production records root the allowance does not open (G2) and a records root under paths.OPS_ROOT
               refuse; the narrow allowance key 'archive:search_deep' opens it; the key is protected; through the CLI
  round        archive_quarantine.ROUND_DEFAULT is 'launch-date', as search_run.QUARANTINE_ROUND (decision (3))
  round 2      fix round 2 of 2026-11-27: the run reads its launch once (a rewritten launch file switches no unit); begin-launch
               refuses a non-finite epoch, one later than now plus 60 s or earlier than the newest recorded launch, before
               anything is written (4); a launch file in a round's records directory makes the round taken (5);
               --retire-launch refuses while a deep search holds a cell lock (3); the new file is written before the old is
               retired, and a failure at the last link restores the old (6); a null record epoch is a Refusal (6)
  job-created  search_run.QUARANTINE_EPOCH 'job-created' passes in the window (and in a scratch run beside a configured
               allowance) and refuses, naming 'launch-file', whenever the run's archive root or --out is a production path
               (gate G3's test), whatever the allowance (decision (4), fix round 2 (10))
Run: python tests/test_search_quarantine_records.py
"""
import csv
import hashlib
import inspect
import json
import os
import shutil
import sys
import time
import types

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
if os.path.isfile('/lustre/gh/main/exopipe_ops/exopipe3_site.py'):       # absent on a data-free machine: no site file is loaded
    os.environ.setdefault('EXOPIPE3_SITE', '/lustre/gh/main/exopipe_ops/exopipe3_site.py')

import numpy as np                                                      # noqa: E402
from astropy.io import fits                                             # noqa: E402

from exopipe.phase3.config import CFG                                   # noqa: E402
from exopipe.phase3.core import validate                                # noqa: E402
from exopipe.phase3.core.record import Refusal                          # noqa: E402
from exopipe.phase3.lib.archive import quarantine as Q                  # noqa: E402
from exopipe.phase3.lib.search import deep_step as DS                   # noqa: E402

T = f'/tmp/exopipe3_tests_search_quarantine_records_{os.getpid()}'
ARCH, OPS, REC, OUT = (os.path.join(T, x) for x in ('archive', 'ops', 'records', 'out'))
RUN = 'Run_20260909_000000'
CELL, LST, SC = '12h_27MHz', '12h', '27MHz'
PROD_LAUNCH = 1794570845.0                       # production's p3s5_state/launch_epoch: 2026-11-13T11:54:05Z (round 20261113)
PROD_ARCH = CFG.paths.PHASE3_ARCHIVE_ROOT         # the configured (production) archive root, before setup points it at scratch
N, FAILS = [0], []


def check(name, ok, detail=None):
    N[0] += 1
    print(('PASS  ' if ok else 'FAIL  ') + name)
    if not ok:
        FAILS.append(name)
        if detail is not None:
            print('      ', str(detail)[:1500])


class Unit:
    def __init__(self):
        self.meta, self.notes, self.outputs = {}, [], []

    def check(self, *a, **k):
        pass

    def note(self, m):
        self.notes.append(m)

    def add_output(self, p, cls, **k):
        self.outputs.append((p, cls))


def frame(st, date, mtime):
    rob = 'Robust-0.75' if st == 'I' else 'Robust+0.0'
    p = os.path.join(ARCH, LST, date, RUN, SC, st, 'deep',
                     f'{SC}_{st}_deep_Taper_{rob}_dewarped_aligned_msub_ssub_LST12h_{date.replace("-", "")}T050000.fits')
    os.makedirs(os.path.dirname(p), exist_ok=True)
    fits.writeto(p, np.full((4, 4), 1.0, np.float32), overwrite=True)
    os.utime(p, (mtime, mtime))
    return p


def tree(root):
    """{relative path: (size, md5, mtime_ns)} of every file under root (a scratch directory)."""
    out = {}
    for d, _, fs in os.walk(root):
        for f in fs:
            p = os.path.join(d, f)
            st = os.stat(p)
            out[os.path.relpath(p, root)] = (st.st_size, hashlib.md5(open(p, 'rb').read()).hexdigest(), st.st_mtime_ns)
    return out


def setup():
    shutil.rmtree(T, ignore_errors=True)
    for d in (ARCH, OPS, REC, OUT):
        os.makedirs(d)
    # production's state: its launch epoch and an old round's manifest (read only), its rounds' Runs in the archive
    os.makedirs(os.path.join(OPS, 'p3s5_state', 'frame_quarantine'))
    open(os.path.join(OPS, 'p3s5_state', 'launch_epoch'), 'w').write(f'{int(PROD_LAUNCH)}\n')
    with open(os.path.join(OPS, 'p3s5_state', 'frame_quarantine', f'{CELL}_I.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(CFG.archive.QUARANTINE_COLUMNS)
        w.writerow(['/x/old.fits', '/y/old.fits', 1, 'd41d8cd98f00b204e9800998ecf8427e', CELL, 'I', '2026-10-04T00:00:00Z'])
    for r in ('20261004', '20261113'):
        os.makedirs(os.path.join(ARCH, f'Run_SEARCH_QUARANTINE_{r}', LST))
    CFG.paths.PHASE3_ARCHIVE_ROOT = ARCH
    CFG.paths.ARCHIVE_RUN = RUN
    CFG.paths.OPS_ROOT = OPS
    CFG.paths.PRODUCTION_WRITES = False          # the scratch roots are outside every production tree: G2 passes on its own
    CFG.archive.RECORDS_ROOT = REC
    CFG.search_run.PRODUCTION_WORK_ROOT = os.path.join(T, 'prod_work')    # begin-launch's cell-lock probe: never production's


def test_round_and_launch():
    ok_ = (Q.round_for(PROD_LAUNCH, 'launch-date') == '20261113' and Q.round_for(1795737600.0, 'launch-date') == '20261127'
           and Q.round_for(PROD_LAUNCH, 'T1') == 'T1')
    bad = []
    for x in ('../x', 'a/b', ''):
        try:
            Q.round_for(PROD_LAUNCH, x)
            bad.append(x)
        except Refusal:
            pass
    check('round_for: launch-date is the launch epoch\'s UTC date (production\'s 11-13 launch -> 20261113); a literal round '
          'passes; a round that is not a plain name refuses', ok_ and not bad, bad)
    lf = Q.K.expand(CFG.archive_quarantine.LAUNCH_EPOCH_FILE)
    try:
        Q.launch_epoch()
        unset = False
    except Refusal:
        unset = True
    os.makedirs(os.path.dirname(lf), exist_ok=True)
    open(lf, 'w').write('1795737600\n')
    check('launch_epoch: the v2 launch file under archive.RECORDS_ROOT (not production\'s p3s5_state/launch_epoch); unset '
          'refuses', unset and Q.launch_epoch() == 1795737600.0 and lf.startswith(REC + '/') and not Q.under_ops_root(lf), lf)


def test_apply():
    launch = round(time.time() - 20 * 86400, 3)  # 20 days ago: the later launches below are past too (begin-launch, item (4))
    rnd = Q.round_for(launch, 'launch-date')
    old = [frame(st, d, launch - 86400) for st in ('I', 'V') for d in ('2024-12-18', '2024-12-19')]
    new = frame('I', '2024-12-20', launch + 60)
    ops0 = tree(OPS)
    got = {}
    for st in ('I', 'V'):
        u = Unit()
        got[st] = (Q.quarantine_for_search(u, CELL, st, launch, rnd, apply=True, precheck=False, check_writer=False), u)
    rdir = os.path.join(REC, 'frame_quarantine', f'Run_SEARCH_QUARANTINE_{rnd}')
    mans = {st: os.path.join(rdir, f'{CELL}_{st}.csv') for st in ('I', 'V')}
    rows = {st: list(csv.DictReader(open(m))) if os.path.exists(m) else [] for st, m in mans.items()}
    moved = all(not os.path.exists(p) and os.path.exists(os.path.join(ARCH, f'Run_SEARCH_QUARANTINE_{rnd}',
                                                                      os.path.relpath(p, ARCH))) for p in old)
    rec = json.load(open(os.path.join(rdir, 'ROUND.json')))
    check('apply (production branch): the earlier frames moved into Run_SEARCH_QUARANTINE_<launch date>, the frame newer than '
          'the launch kept; the manifest of record per (cell, Stokes) under the records root, the round\'s record naming the '
          'launch epoch', moved and os.path.exists(new) and [len(rows['I']), len(rows['V'])] == [2, 2]
          and rec['round'] == rnd and rec['launch_epoch'] == launch
          and all(got[st][1].outputs == [(mans[st], 'quarantine_manifest')] for st in ('I', 'V')),
          (rnd, {k: len(v) for k, v in rows.items()}, rec, [g[1].outputs for g in got.values()]))
    r = validate.run('quarantine_manifest', mans['I'])
    check('the round\'s manifest of record passes its content test (rows_quarantined)', all(x['passed'] for x in r),
          [x['detail'] for x in r])
    check('paths.OPS_ROOT byte for byte as before the apply: production\'s manifests and launch epoch untouched, nothing added',
          tree(OPS) == ops0, sorted(set(tree(OPS)) ^ set(ops0)))
    src = inspect.getsource(Q)
    check('the production helper (p3s5_quarantine_frames.py, which writes p3s5_state) is never run: no helper call in the '
          'module; every apply takes the port\'s move loop', 'HELPER' not in src.split('def launch_epoch', 1)[1]
          and '_apply_helper' not in src and 'K.quarantine_move(' in inspect.getsource(Q.quarantine_for_search))
    return launch, rnd


def test_refusals(launch, rnd):
    extra = frame('I', '2024-12-21', launch - 86400)
    ops0, arch0 = tree(OPS), tree(ARCH)

    def refused(fn):
        try:
            fn()
            return False, 'not refused'
        except Refusal as e:
            return True, str(e)[:160]
    r = {}
    r['20261004 (production\'s first round)'] = refused(lambda: Q.quarantine_for_search(
        Unit(), CELL, 'I', launch, '20261004', apply=True, precheck=False, check_writer=False))
    r['production\'s launch epoch (round 20261113)'] = refused(lambda: Q.quarantine_for_search(
        Unit(), CELL, 'I', PROD_LAUNCH, Q.round_for(PROD_LAUNCH, 'launch-date'), apply=True, precheck=False,
        check_writer=False))
    r['the plan of round 20261004'] = refused(lambda: Q.quarantine_for_search(
        Unit(), CELL, 'I', launch, '20261004', apply=False, precheck=False, check_writer=False))
    r['this round with another launch epoch'] = refused(lambda: Q.quarantine_for_search(
        Unit(), CELL, 'I', launch + 1.0, rnd, apply=True, precheck=False, check_writer=False))
    CFG.archive.RECORDS_ROOT = os.path.join(OPS, 'records')
    try:
        r['a records root under paths.OPS_ROOT'] = refused(lambda: Q.quarantine_for_search(
            Unit(), CELL, 'I', launch + 7 * 86400, Q.round_for(launch + 7 * 86400, 'launch-date'), apply=True,
            precheck=False, check_writer=False))
    finally:
        CFG.archive.RECORDS_ROOT = REC
    check('a round production used, its plan, another launch\'s round, and a records root under paths.OPS_ROOT each refuse',
          all(v[0] for v in r.values()), r)
    check('... before anything moved or was written: the archive and paths.OPS_ROOT as they were, no records directory for '
          'the refused rounds', tree(ARCH) == arch0 and tree(OPS) == ops0 and os.path.exists(extra)
          and not os.path.exists(os.path.join(REC, 'frame_quarantine', 'Run_SEARCH_QUARANTINE_20261004'))
          and not os.path.exists(os.path.join(REC, 'frame_quarantine', 'Run_SEARCH_QUARANTINE_20261113')),
          sorted(set(tree(ARCH)) ^ set(arch0)))


def test_search_stage():
    lf = Q.K.expand(CFG.archive_quarantine.LAUNCH_EPOCH_FILE)
    launch = time.time() + 3 * 86400            # a later launch: a fresh round, every frame earlier
    open(lf, 'w').write(f'{launch!r}\n')
    rnd = Q.round_for(launch, 'launch-date')
    item = types.SimpleNamespace(cell=CELL, lst=LST, scope=SC, stokes='I', name=f'{CELL}_I')
    ctx = dict(out=OUT, t0=time.time())
    sr = CFG.search_run
    check('search_run: the round is the launch date, the epoch the v2 launch file (no key of the search names '
          'p3s5_state/launch_epoch)', sr.QUARANTINE_ROUND == 'launch-date' and sr.QUARANTINE_EPOCH == 'launch-file'
          and not hasattr(sr, 'LAUNCH_EPOCH_FILE'))
    ops0 = tree(OPS)
    u, f = Unit(), {}
    DS.stage_quarantine(u, item, ctx, f)
    plan = os.path.join(OUT, 'quarantine_plan', f'{CELL}_I.csv')
    rows = list(csv.DictReader(open(plan)))
    check('the search\'s quarantine stage (window): plans the frames older than the v2 launch into Run_SEARCH_QUARANTINE_'
          '<launch date>, the manifest of record named under the records root; the round handed on to the detect stage '
          '(the only round the engine takes a predecessor from)',
          rows and all(f'/Run_SEARCH_QUARANTINE_{rnd}/' in r['dest'] for r in rows) and ctx.get('launch') == (launch, rnd)
          and u.meta.get('quarantine_round') == rnd and u.meta.get('launch_epoch') == launch
          and DS.archive_predecessor_roots(dict(ctx, archive_root=ARCH))[1] == Q.quarantine_root(ARCH, rnd)
          and all(float(r['launch_epoch']) == launch for r in rows)
          and any(os.path.join(REC, 'frame_quarantine', f'Run_SEARCH_QUARANTINE_{rnd}', f'{CELL}_I.csv') in n for n in u.notes),
          (rows[:1], u.notes))
    open(lf, 'w').write(f'{int(PROD_LAUNCH)}\n')
    u2 = Unit()
    DS.stage_quarantine(u2, item, ctx, {})           # the same run: its launch read once, at its start
    rows2 = list(csv.DictReader(open(plan)))
    check('fix round 2 (3): the run reads the launch epoch and round ONCE (ctx \'launch\', the run\'s state): a launch file '
          'rewritten while the run goes on switches no unit\'s round - the next unit still plans into this run\'s round',
          ctx['launch'] == (launch, rnd) and u2.meta.get('quarantine_round') == rnd
          and all(f'/Run_SEARCH_QUARANTINE_{rnd}/' in r['dest'] for r in rows2), (ctx.get('launch'), u2.meta))
    try:
        DS.stage_quarantine(Unit(), item, dict(out=OUT, t0=time.time()), {})        # a new run reads the new file
        r = 'not refused'
    except Refusal as e:
        r = str(e)
    check('a new run\'s quarantine stage refuses production\'s launch epoch (round 20261113: an earlier round)',
          '20261113' in r and 'earlier' in r, r[:200])
    code = [ln.split('#', 1)[0] for ln in inspect.getsource(DS.stage_quarantine).splitlines()]
    check('paths.OPS_ROOT untouched by the stage; the stage\'s code never reads it',
          tree(OPS) == ops0 and not any('OPS_ROOT' in ln for ln in code), [ln for ln in code if 'OPS_ROOT' in ln])


def refused(fn):
    try:
        fn()
        return False, 'not refused'
    except Refusal as e:
        return True, str(e)


def test_begin_launch(launch, rnd):
    """decision (1) of 2026-11-27: the launch file is written by an operator step, never by hand."""
    from exopipe.phase3 import cli
    lf = Q.K.expand(CFG.archive_quarantine.LAUNCH_EPOCH_FILE)
    os.remove(lf)                                    # the earlier checks wrote it by hand
    ops0 = tree(OPS)
    plan = Q.begin_launch(Unit(), epoch=launch, apply=False)
    check('begin-launch plan: names the launch file, the epoch and its round; writes nothing',
          plan['launch_file'] == lf and plan['launch_epoch'] == launch and plan['round'] == rnd and not os.path.exists(lf), plan)
    Q.begin_launch(Unit(), epoch=launch, apply=True)
    left = [x for x in os.listdir(os.path.dirname(lf)) if x.endswith('.tmp')]
    check('begin-launch apply: the launch file holds the epoch (launch_epoch reads it back), no temporary file left',
          Q.launch_epoch() == launch and not left, (open(lf).read(), left))
    r = {'an existing launch file, no --retire-launch': refused(lambda: Q.begin_launch(Unit(), epoch=launch + 86400 * 2,
                                                                                         apply=True))}
    same = launch + 1.0
    r['a second launch on the same UTC date'] = refused(lambda: Q.begin_launch(Unit(), epoch=same, retire=True, apply=True))
    check('begin-launch refuses an existing launch file (naming it, pointing to --retire-launch) and a second launch on one '
          'UTC date (one round per launch), the launch file as it was',
          all(v[0] for v in r.values()) and lf in r['an existing launch file, no --retire-launch'][1]
          and '--retire-launch' in r['an existing launch file, no --retire-launch'][1] and Q.launch_epoch() == launch
          and Q.round_for(same, 'launch-date') == rnd, r)
    new2 = launch + 86400 * 2
    rdir = os.path.join(REC, 'frame_quarantine', f'Run_SEARCH_QUARANTINE_{rnd}')
    p2 = Q.begin_launch(Unit(), epoch=new2, retire=True, apply=True)
    ret = os.path.join(rdir, os.path.basename(lf))
    check('begin-launch --retire-launch: the earlier launch file moved into its own round\'s records directory beside its '
          'ROUND.json (that launch\'s epoch), the new epoch written; paths.OPS_ROOT untouched',
          p2['retired_to'] == ret and os.path.exists(os.path.join(rdir, 'ROUND.json')) and float(open(ret).read()) == launch
          and Q.launch_epoch() == new2 and tree(OPS) == ops0, (p2, os.listdir(rdir)))
    open(lf + '.x', 'w').write(f'{launch!r}\n')
    os.replace(lf + '.x', lf)                        # a launch of round rnd again: its retirement place is taken
    r2 = refused(lambda: Q.begin_launch(Unit(), epoch=new2 + 86400 * 2, retire=True, apply=True))
    check('begin-launch --retire-launch refuses when the round\'s retired launch file is there already',
          r2[0] and 'retired already' in r2[1] and Q.launch_epoch() == launch, r2)
    os.remove(lf)
    # G2: a production records root - refused without the narrow allowance, written with 'archive:search_deep'
    pr = os.path.join(T, 'prod_records')
    keep = (CFG.archive.RECORDS_ROOT, CFG.paths.PRODUCTION_ROOTS, CFG.paths.PRODUCTION_WRITES_ALLOWED)
    try:
        CFG.archive.RECORDS_ROOT = pr
        CFG.paths.PRODUCTION_ROOTS = tuple(CFG.paths.PRODUCTION_ROOTS) + (pr,)
        CFG.paths.PRODUCTION_WRITES_ALLOWED = {}
        plf = Q.K.expand(CFG.archive_quarantine.LAUNCH_EPOCH_FILE)
        g2 = refused(lambda: Q.begin_launch(Unit(), epoch=new2, apply=True))
        g2ok = g2[0] and g2[1].startswith('G2') and "'archive:search_deep'" in g2[1] and not os.path.exists(plf)
        CFG.paths.PRODUCTION_WRITES_ALLOWED = {Q.ALLOW_KEY: (pr,)}
        Q.begin_launch(Unit(), epoch=new2, apply=True)
        allowed = Q.launch_epoch() == new2 and plf.startswith(pr + '/')
        CFG.archive.RECORDS_ROOT = os.path.join(OPS, 'records')
        ops = refused(lambda: Q.begin_launch(Unit(), epoch=new2, apply=True))
    finally:
        CFG.archive.RECORDS_ROOT, CFG.paths.PRODUCTION_ROOTS, CFG.paths.PRODUCTION_WRITES_ALLOWED = keep
    check('begin-launch under a production records root: refused (G2) while paths.PRODUCTION_WRITES is False and no '
          'allowance opens it, nothing written; written when the narrow allowance opens it to \'archive:search_deep\'; a '
          'records root under paths.OPS_ROOT refuses', g2ok and allowed and ops[0] and 'OPS_ROOT' in ops[1], (g2, ops))
    check('archive_quarantine.LAUNCH_EPOCH_FILE is a protected key (no --set moves it); ROUND_DEFAULT is \'launch-date\', as '
          'search_run.QUARANTINE_ROUND (decision (3))',
          CFG.protected('archive_quarantine.LAUNCH_EPOCH_FILE') and CFG.archive_quarantine.ROUND_DEFAULT == 'launch-date'
          == CFG.search_run.QUARANTINE_ROUND)
    # the CLI: archive quarantine --begin-launch (no --cells), --retire-launch only with it
    co = os.path.join(T, 'cli_out')
    new4 = new2 + 86400 * 5
    rc = cli.main(['archive', 'quarantine', '--begin-launch', '--launch-epoch', repr(new4), '--apply', '--out', co,
                   '--cores', 'none'])
    rp = os.path.join(co, '_records', 'archive', 'begin_launch.json')
    urec = json.load(open(rp)) if os.path.exists(rp) else {'status': 'no record'}
    rc2 = cli.main(['archive', 'quarantine', '--begin-launch', '--cells', CELL, '--apply', '--out', co, '--cores', 'none'])
    rc3 = cli.main(['archive', 'quarantine', '--retire-launch', '--cells', CELL, '--out', co, '--cores', 'none'])
    rc4 = cli.main(['archive', 'quarantine', '--begin-launch', '--retire-launch', '--launch-epoch', repr(new4 + 86400 * 2),
                    '--apply', '--resume', '--out', co, '--cores', 'none'])
    check('the CLI: `exopipe3 archive quarantine --begin-launch --launch-epoch E --apply` writes the launch file (unit '
          'begin_launch, its content test passed); --begin-launch with --cells, --retire-launch without --begin-launch and '
          '--begin-launch with --resume (a done record would skip the new launch) refuse, the launch file as it was',
          rc == 0 and urec['status'] == 'done' and Q.launch_epoch() == new4 and rc2 != 0 and rc3 != 0 and rc4 != 0
          and Q.launch_epoch() == new4 and any(t['name'] == 'launch_epoch.launch_epoch_file' and t['passed']
                                              for t in urec.get('tests', [])),
          (rc, urec.get('status'), urec.get('reason'), urec.get('tests'), rc2, rc3, rc4))
    from exopipe.phase3.lib.archive import tests as AT
    bad = os.path.join(T, 'bad_launch')
    open(bad, 'w').write('nan\n')
    ct = (AT.launch_epoch_file(lf, epoch=new4), AT.launch_epoch_file(lf, epoch=new4 + 1), AT.launch_epoch_file(bad))
    check('the launch file\'s content test (never waivable): one positive finite epoch line, the epoch the unit wrote; a '
          'NaN and another epoch fail', ct[0][0] and not ct[1][0] and not ct[2][0]
          and 'launch_epoch.launch_epoch_file' in CFG.archive.NEVER_WAIVABLE, ct)
    return new4


def test_begin_launch_round2(cur):
    """The refactor chat's fix round 2 (2026-11-27), items (3)-(6): begin-launch validates the epoch before anything is written,
    a retired launch's round is taken, the retire refuses while a deep search holds a cell lock, and the new file is written
    before the old is retired, with the old restored when the last link fails."""
    lf = Q.K.expand(CFG.archive_quarantine.LAUNCH_EPOCH_FILE)
    rec0 = tree(REC)
    now = time.time()
    r = {'nan': refused(lambda: Q.begin_launch(Unit(), epoch='nan', retire=True, apply=True)),
         'inf': refused(lambda: Q.begin_launch(Unit(), epoch='inf', retire=True, apply=True)),
         'now plus 61 s': refused(lambda: Q.begin_launch(Unit(), epoch=repr(now + 61), retire=True, apply=True)),
         'earlier than the newest recorded launch': refused(lambda: Q.begin_launch(Unit(), epoch=repr(cur - 1.0), retire=True,
                                                                                    apply=True))}
    check('item (4): --launch-epoch refuses unless finite, not later than now plus archive_quarantine.LAUNCH_EPOCH_MAX_AHEAD_S '
          '(60 s), and not earlier than the newest launch v2 recorded (a round record, a retired or the live launch file) - '
          'before anything is written: the records root byte for byte as it was, the launch file as it was (a nan never '
          'linked into place)',
          all(v[0] for v in r.values()) and tree(REC) == rec0 and Q.launch_epoch() == cur
          and CFG.archive_quarantine.LAUNCH_EPOCH_MAX_AHEAD_S == 60.0 and 'finite' in r['nan'][1]
          and 'later than now' in r['now plus 61 s'][1] and 'earlier than the newest' in r['earlier than the newest recorded launch'][1],
          r)
    # item (5): a launch file in a round's records directory makes that round taken, with or without its ROUND.json
    retired = [d for d in sorted(os.listdir(os.path.join(REC, 'frame_quarantine')))
               if os.path.lexists(os.path.join(REC, 'frame_quarantine', d, os.path.basename(lf)))]
    rd = os.path.join(REC, 'frame_quarantine', retired[0]) if retired else None
    rnd0 = retired[0][len('Run_SEARCH_QUARANTINE_'):] if retired else None
    e0 = float(open(os.path.join(rd, os.path.basename(lf))).read()) if rd else None
    g = {'the retired launch\'s own epoch': refused(lambda: Q.round_gate(ARCH, rnd0, e0, rd, create=False)),
         'another epoch': refused(lambda: Q.round_gate(ARCH, rnd0, e0 + 1.0, rd, create=False))}
    bare_r = Q.round_for(cur - 86400, 'launch-date')
    bare = Q.round_records_dir(ARCH, bare_r)
    os.makedirs(bare)
    open(os.path.join(bare, os.path.basename(lf)), 'w').write(f'{cur - 86400!r}\n')      # a launch file, no ROUND.json
    g['a launch file without a ROUND.json'] = refused(lambda: Q.round_gate(ARCH, bare_r, cur - 86400, bare, create=False))
    check('item (5): round_gate treats a retired launch file (or any launch file) in the round\'s records directory as a taken '
          'round - also for the retired launch\'s own epoch and when no ROUND.json is there; the date is not reusable',
          rd is not None and all(v[0] and 'taken' in v[1] for v in g.values())
          and not os.path.exists(os.path.join(bare, CFG.archive_quarantine.ROUND_RECORD)), (retired, g))
    shutil.rmtree(bare)
    # item (3): --retire-launch refuses while a deep search holds a cell lock (a non-blocking flock probe of the work root)
    import fcntl
    wr = CFG.search_run.PRODUCTION_WORK_ROOT
    os.makedirs(wr, exist_ok=True)
    lk = os.path.join(wr, f'.{CELL}.search.lock')
    new5 = round(cur + 2 * 86400, 3)
    fh = None
    try:
        open(os.path.join(wr, '.12h_Red.search.lock'), 'w').close()               # a lock file no process holds
        fh = open(lk, 'w')
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)                           # a live search's cell lock
        held = Q.search_locks_held()
        rl = refused(lambda: Q.begin_launch(Unit(), epoch=repr(new5), retire=True, apply=True))
        plan_held = refused(lambda: Q.begin_launch(Unit(), epoch=repr(new5), retire=True, apply=False))
        untouched = Q.launch_epoch() == cur and tree(REC) == rec0
        fcntl.flock(fh, fcntl.LOCK_UN)
        fh.close()
        free = Q.search_locks_held()
        pl = Q.begin_launch(Unit(), epoch=repr(new5), retire=True, apply=False)
    finally:
        if fh is not None and not fh.closed:
            fh.close()
    check('item (3): --retire-launch refuses (plan and apply) while a deep search holds a cell lock - a non-blocking flock probe '
          'of each <search_run.PRODUCTION_WORK_ROOT>/.<cell>.search.lock (deep_step.cell_lock), released at once; a lock file no '
          'process holds does not count; nothing written; free once the lock is released',
          held == [lk] and rl[0] and plan_held[0] and 'cell lock' in rl[1] and untouched and free == []
          and pl['launch_epoch'] == new5, (held, rl, free))
    # item (6): the new file first, then the retire, then the link; a failure at the link restores the old file
    real_link = os.link
    def failing_link(src, dst, *a, **k):
        if dst == lf and src.endswith('.tmp'):
            raise OSError(5, 'injected failure at the last link')
        return real_link(src, dst, *a, **k)
    order = []
    real_open, real_unlink = os.open, os.unlink
    def spy_open(path, *a, **k):
        order.append(('open', path))
        return real_open(path, *a, **k)
    def spy_unlink(path, *a, **k):
        order.append(('unlink', path))
        return real_unlink(path, *a, **k)
    def spy_link(src, dst, *a, **k):
        order.append(('link', src, dst))
        return failing_link(src, dst, *a, **k)
    try:
        os.link, os.open, os.unlink = spy_link, spy_open, spy_unlink
        f6 = refused(lambda: Q.begin_launch(Unit(), epoch=repr(new5), retire=True, apply=True))
    finally:
        os.link, os.open, os.unlink = real_link, real_open, real_unlink
    rdir_c = Q.round_records_dir(ARCH, Q.round_for(cur, 'launch-date'))
    ret_c = os.path.join(rdir_c, os.path.basename(lf))
    left = [x for x in os.listdir(os.path.dirname(lf)) if x.endswith('.tmp')]
    steps = [o[0] for o in order if (o[0] == 'open' and o[1].endswith('.tmp')) or (o[0] == 'link')
             or (o[0] == 'unlink' and o[1] == lf)]
    check('item (6): a failure injected at the last step (linking the new launch file into place, after the old one was '
          'retired) restores the old launch file in place, the retired copy removed, no temporary file left, a Refusal naming '
          'it; the steps ran in the order new file written, old retired (link, unlink), new linked',
          f6[0] and 'restored in place' in f6[1] and Q.launch_epoch() == cur and not os.path.lexists(ret_c) and not left
          and steps[:4] == ['open', 'link', 'unlink', 'link'], (f6, steps, left))
    p6 = Q.begin_launch(Unit(), epoch=repr(new5), retire=True, apply=True)
    check('... and with no failure the same call retires the old launch and places the new one',
          Q.launch_epoch() == new5 and p6['retired_to'] == ret_c and float(open(ret_c).read()) == cur, p6)
    # item (6): a round record whose launch_epoch is null refuses (a Refusal, never a TypeError)
    rd7 = Q.round_records_dir(ARCH, Q.round_for(new5, 'launch-date'))
    os.makedirs(rd7, exist_ok=True)
    rp7 = os.path.join(rd7, CFG.archive_quarantine.ROUND_RECORD)
    with open(rp7, 'w') as fh:
        json.dump({'round': Q.round_for(new5, 'launch-date'), 'launch_epoch': None}, fh)
    n7 = {}
    for what, fn in (('begin-launch --retire-launch', lambda: Q.begin_launch(Unit(), epoch=repr(new5 + 86400), retire=True,
                                                                             apply=False)),
                     ('round_gate', lambda: Q.round_gate(ARCH, Q.round_for(new5, 'launch-date'), new5, rd7, create=False))):
        try:
            fn()
            n7[what] = 'not refused'
        except Refusal as e:
            n7[what] = 'Refusal: ' + str(e)[:400]
        except Exception as e:                     # noqa: BLE001 - the check is that it is a Refusal
            n7[what] = f'{type(e).__name__}: {e}'
    os.remove(rp7)
    check('item (6): a round record with launch_epoch null refuses as a Refusal (begin-launch, round_gate), not a TypeError',
          all(v.startswith('Refusal: ') and 'not a finite number' in v for v in n7.values()), n7)
    return new5


def test_job_created():
    """decision (4) of 2026-11-27 and fix round 2 (10): 'job-created' is for a window or scratch run only - refused whenever the
    run's archive root or output root is a production path (deep_step.production_tree, gate G3's test), whatever any allowance
    says."""
    sr = CFG.search_run
    keep = (sr.QUARANTINE_EPOCH, CFG.paths.PRODUCTION_WRITES, CFG.paths.PRODUCTION_WRITES_ALLOWED)
    t0 = 1795737600.0
    ctx = dict(out=OUT, t0=t0, archive_root=os.path.join(OUT, 'archive_mirror'))
    coll = sr.COLLECT_ROOT.format(host='node00')
    r = {}
    try:
        sr.QUARANTINE_EPOCH = 'job-created'
        window = DS.quarantine_launch_round(dict(ctx))
        CFG.paths.PRODUCTION_WRITES_ALLOWED = {Q.ALLOW_KEY: (PROD_ARCH, CFG.archive.RECORDS_ROOT), 'search:deep': (coll,)}
        beside = DS.quarantine_launch_round(dict(ctx))   # an allowance in config, this run scratch: it writes no production
        r['the production archive root, the narrow allowance opening it'] = refused(
            lambda: DS.quarantine_launch_round(dict(ctx, archive_root=PROD_ARCH)))
        r['--out the collect root, the narrow allowance opening it'] = refused(
            lambda: DS.quarantine_launch_round(dict(ctx, out=coll)))
        CFG.paths.PRODUCTION_WRITES_ALLOWED = {}
        r['the production archive root, no allowance'] = refused(
            lambda: DS.quarantine_launch_round(dict(ctx, archive_root=os.path.join(PROD_ARCH, 'x'))))
        CFG.paths.PRODUCTION_WRITES = True
        r['paths.PRODUCTION_WRITES (its archive root, the production archive)'] = refused(
            lambda: DS.quarantine_launch_round(dict(ctx, archive_root=PROD_ARCH, out=coll)))
    finally:
        sr.QUARANTINE_EPOCH, CFG.paths.PRODUCTION_WRITES, CFG.paths.PRODUCTION_WRITES_ALLOWED = keep
    code = [ln.split('#', 1)[0] for ln in inspect.getsource(DS.quarantine_launch_round).splitlines()]
    check('search_run.QUARANTINE_EPOCH \'job-created\': this run\'s start in the window and in a scratch run beside a configured '
          'allowance; refused, naming \'launch-file\', whenever the run\'s archive root or --out is a production path (the '
          'production archive ' + PROD_ARCH + ', the collect root), whatever the allowance (none, the narrow one, '
          'paths.PRODUCTION_WRITES); one test, deep_step.production_tree (G3\'s), no flag of ctx read',
          window == beside == (t0, '20261127') and all(v[0] and "'launch-file'" in v[1] for v in r.values())
          and DS.production_tree(PROD_ARCH) and any('production_tree(' in ln for ln in code)
          and not any("'production'" in ln or 'PRODUCTION_WRITES' in ln for ln in code), (window, beside, r))


def main():
    try:
        setup()
        test_round_and_launch()
        launch, rnd = test_apply()
        test_refusals(launch, rnd)
        test_search_stage()
        cur = test_begin_launch(launch, rnd)
        test_begin_launch_round2(cur)
        test_job_created()
    finally:
        CFG.reset()
        if not os.environ.get('EXOPIPE3_TEST_KEEP'):
            shutil.rmtree(T, ignore_errors=True)
    print(f'{N[0] - len(FAILS)} of {N[0]} checks passed')
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
