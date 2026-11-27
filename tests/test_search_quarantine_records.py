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
               date>, refuses production's launch epoch (an earlier round); it never names paths.OPS_ROOT
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
    launch = time.time() - 600
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
    u = Unit()
    DS.stage_quarantine(u, item, ctx, {})
    plan = os.path.join(OUT, 'quarantine_plan', f'{CELL}_I.csv')
    rows = list(csv.DictReader(open(plan)))
    check('the search\'s quarantine stage (window): plans the frames older than the v2 launch into Run_SEARCH_QUARANTINE_'
          '<launch date>, the manifest of record named under the records root',
          rows and all(f'/Run_SEARCH_QUARANTINE_{rnd}/' in r['dest'] for r in rows)
          and all(float(r['launch_epoch']) == launch for r in rows)
          and any(os.path.join(REC, 'frame_quarantine', f'Run_SEARCH_QUARANTINE_{rnd}', f'{CELL}_I.csv') in n for n in u.notes),
          (rows[:1], u.notes))
    open(lf, 'w').write(f'{int(PROD_LAUNCH)}\n')
    try:
        DS.stage_quarantine(Unit(), item, ctx, {})
        r = 'not refused'
    except Refusal as e:
        r = str(e)
    check('the search\'s quarantine stage refuses production\'s launch epoch (round 20261113: an earlier round)',
          '20261113' in r and 'earlier' in r, r[:200])
    code = [ln.split('#', 1)[0] for ln in inspect.getsource(DS.stage_quarantine).splitlines()]
    check('paths.OPS_ROOT untouched by the stage; the stage\'s code never reads it',
          tree(OPS) == ops0 and not any('OPS_ROOT' in ln for ln in code), [ln for ln in code if 'OPS_ROOT' in ln])


def main():
    try:
        setup()
        test_round_and_launch()
        launch, rnd = test_apply()
        test_refusals(launch, rnd)
        test_search_stage()
    finally:
        CFG.reset()
        if not os.environ.get('EXOPIPE3_TEST_KEEP'):
            shutil.rmtree(T, ignore_errors=True)
    print(f'{N[0] - len(FAILS)} of {N[0]} checks passed')
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
