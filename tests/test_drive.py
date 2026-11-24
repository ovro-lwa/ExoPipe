#!/usr/bin/env python3
"""Tests of the per-hour driver (exopipe/phase3/core/driver.py, `exopipe3 drive`). Data-free: a fake Slurm (tests/fake_slurm.py:
sbatch, squeue, sacct, scancel over a JSON state; the test runs the jobs) and a fake step (tests/drive_fake_step.py: unit records
only). The driver runs in this process; every Slurm job runs the real `python -m exopipe.phase3 drive stage|finally` in its own
process, which runs the fake step as the chain's step.
    TMPDIR=/fast/gh/claude_tmp /opt/devel/pipeline/envs/py38_orca_nkosogor/bin/python3 tests/test_drive.py
Checked:
 1. a chain end to end: one runner job record per hour, its stages submitted with afterok + singleton on one node, partition per
    tier, --time/--mem/--requeue on every job; the stage records the steps as the runner does; the closing job runs after the
    chain and gets the record; sacct accounting per submission; the booking released; the job is readable by `run status`;
 2. space booking: each hour on the node with the most room after the other bookings and the floor; an hour that does not fit
    waits, and starts once a booking is released; catch-up windows: newest night first, the rest wait with the reason;
 3. a failed parent: its pending children (DependencyNeverSatisfied) are cancelled by exact id and the chain is resubmitted from
    the failed stage; finished units of the earlier stages and of the failed stage are not redone (--resume);
 4. the retry limit: the hour ends 'failed', the item record says so, and the closing steps still run;
 5. a refused stage ends the hour 'refused' (pending children cancelled, closing steps run); `drive retry` runs it again from
    that stage; a stage with nothing to do (exit 3) lets the chain go on;
 6. REQUEUE: a stage killed mid-way (scancel of its exact job id while its step runs) is recorded interrupted and resubmitted;
    the new job resumes from the unit records without redoing finished units; a stage job run again after its step is done
    does nothing (requeue-safe);
 7. a refused sbatch counts no failure and is retried at the next pass; a failed squeue does nothing in that pass;
 8. discovery: only hours on or after driver.FIRST_NIGHT whose latest Phase 2 run holds driver.PHASE2_READY_FILE naming the hour;
 9. periodic jobs: submitted when due and none live, accounted, an ALARM line when one fails; every_h 0 = not scheduled;
10. one driver at a time (LOCK); the dry run (CLI) prints the sbatch command lines and submits and writes nothing.
"""
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
PY = sys.executable
FAKE = os.path.join(REPO, 'tests', 'fake_slurm.py')
STEP = os.path.join(REPO, 'tests', 'drive_fake_step.py')
sys.path.insert(0, os.path.join(REPO, 'tests'))
import fake_slurm as FS  # noqa: E402

FREE = {}


def raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc as e:
        return e
    raise AssertionError(f'{getattr(fn, "__name__", fn)} did not raise {exc.__name__}')


class Scenario:
    """A scratch directory, a fake Slurm state, a config file of the overrides (applied here and forwarded to every job
    process) and the driver's free-space reader replaced by FREE."""

    def __init__(self, name, chain, fin=True, **drv):
        self.name, self.chain, self.fin, self.drv = name, chain, fin, drv

    def __enter__(self):
        from exopipe.phase3.config import CFG
        from exopipe.phase3.core import driver
        base = os.environ.get('TMPDIR') or tempfile.gettempdir()
        self.d = tempfile.mkdtemp(prefix=f'test_drive_{self.name}_', dir=base)
        self.out = os.path.join(self.d, 'out')
        self.jobs_root = os.path.join(self.d, 'jobs')
        self.state = os.path.join(self.d, 'slurm.json')
        os.environ['FAKE_SLURM_STATE'] = self.state
        chain = [c.replace('OUT', self.out).replace('FLAG', os.path.join(self.d, 'flag')) for c in self.chain]
        lines = [f"runner.JOBS_ROOT = {self.jobs_root!r}",
                 "runner.FREE_CORES_CMD = 'echo 0-4095'",
                 f"runner.CHAINS[epoch] = {chain!r}",
                 f"runner.CHAIN_FINALLY[epoch] = {['fin --epochs {item} --record {record} --out ' + self.out] if self.fin else []!r}",
                 f"driver.STEP_HEAD = {(STEP,)!r}"]
        for k in ('SBATCH', 'SQUEUE', 'SACCT', 'SCANCEL'):
            lines.append(f"driver.{k} = {f'{PY} {FAKE} {k.lower()}'!r}")
        drv = dict(PARTITIONS={'night': 'nightp', 'backlog': 'backp'}, NODE_POOL='nodeA,nodeB', SCRATCH_BOOK_BYTES=50e9,
                   SCRATCH_FLOOR_BYTES=10e9, RETRY=1,
                   STEP_RESOURCES={'default': {'cpus': 2, 'mem': '1G', 'time': '00:05:00'}})
        drv.update(self.drv)
        per = drv.pop('PERIODIC', {'argv': ('-c', 'pass'), 'every_h': 0, 'tier': 'backlog', 'cpus': 1, 'mem': '1G',
                                   'time': '00:01:00'})
        lines.append(f"driver.PERIODIC[tess-orbits] = {per!r}")
        for k, v in drv.items():
            lines.append(f'driver.{k} = {v!r}')
        self.cfg = os.path.join(self.d, 'drive.cfg')
        with open(self.cfg, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        CFG.reset()
        CFG.authorize([], out=self.d)
        CFG.apply_override_file(self.cfg)
        CFG.set('site.SCRATCH_ROOT', dict(CFG.site.SCRATCH_ROOT, nodeA='/scratchA', nodeB='/scratchB'))
        self._free = driver.free_bytes
        driver.free_bytes = lambda host, path: FREE.get(host)
        FREE.clear()
        FREE.update(nodeA=1000e9, nodeB=1000e9)
        self.printed = []
        return self

    def __exit__(self, exc, *rest):
        from exopipe.phase3.config import CFG
        from exopipe.phase3.core import driver
        driver.free_bytes = self._free
        CFG.reset()
        if exc is None:
            shutil.rmtree(self.d, ignore_errors=True)
        else:
            print(f'scratch kept: {self.d}')

    def tick(self, epochs=(), tier='backlog', dry=False):
        from exopipe.phase3.core import driver
        ps = driver.Pass(dry=dry, forward=['--config', self.cfg], out=self.printed.append)
        n0 = len(self.printed)
        driver.tick(ps, epochs, tier)
        return self.printed[n0:]

    def run_ready(self):
        return FS.run_ready(self.state)

    def jobs(self):
        return FS.jobs(self.state)

    def hour(self, epoch):
        from exopipe.phase3.core import driver
        return driver.load_hours().get(epoch) or json.load(open(driver.hour_path(epoch, finished=True)))

    def item(self, epoch):
        from exopipe.phase3.core import runner
        return runner.item_state(self.hour(epoch)['job'], epoch)

    def ran(self):
        p = os.path.join(self.out, 'ran.log')
        return [' '.join(l.split()[:-1]) for l in open(p)] if os.path.exists(p) else []

    def count(self, text):
        return sum(1 for l in self.ran() if l == text)


CHAIN3 = ['s1 --epochs {item} --out OUT --units 2', 's2 --epochs {item} --out OUT --units 2',
          's3 --epochs {item} --out OUT --units 2']


def _cli(*args, env=None):
    e = dict(os.environ)
    e['PYTHONPATH'] = REPO + (os.pathsep + e['PYTHONPATH'] if e.get('PYTHONPATH') else '')
    e.update(env or {})
    return subprocess.run([PY, '-m', 'exopipe.phase3'] + list(args), cwd=REPO, env=e, capture_output=True, text=True,
                          timeout=300)


def test_chain_end_to_end_booking_and_window():
    from exopipe.phase3.core import driver
    with Scenario('e2e', CHAIN3, WINDOW_HOURS={'night': 24, 'backlog': 3}) as S:
        # nodeA: 100 GB free, nodeB 80 GB; each hour books 50 GB above a 10 GB floor -> two hours fit, a third waits
        FREE.update(nodeA=100e9, nodeB=80e9)
        S.tick(['03h_2026-01-01', '04h_2026-01-02', '05h_2026-01-03', '06h_2025-12-31'])
        hrs = driver.load_hours()
        assert hrs['05h_2026-01-03']['state'] == 'active' and hrs['05h_2026-01-03']['node'] == 'nodeA', hrs['05h_2026-01-03']
        assert hrs['04h_2026-01-02']['state'] == 'active' and hrs['04h_2026-01-02']['node'] == 'nodeB'
        w = hrs['03h_2026-01-01']
        assert w['state'] == 'waiting' and w['waiting_reason'].startswith('space:') and w['job'] is None, w
        assert hrs['06h_2025-12-31']['waiting_reason'].startswith('behind 03h_2026-01-01'), hrs['06h_2025-12-31']
        assert hrs['05h_2026-01-03']['booking'] == {'nodeA:/scratchA': 50e9}
        jobs = S.jobs()
        assert len(jobs) == 6, jobs
        by = {j['name']: j for j in jobs.values()}
        s = [by[f'exo3_05h_2026-01-03.s{k}'] for k in range(3)]
        assert s[0]['afterok'] == [] and s[0]['singleton'] and s[1]['afterok'] == [s[0]['id']] and s[2]['afterok'] == [s[1]['id']]
        for j in jobs.values():
            o = j['opts']
            assert o['partition'] == 'backp' and o['time'] == '00:05:00' and o['mem'] == '1G' and o['cpus-per-task'] == '2'
            assert '--requeue' in j['flags'] and o['nodelist'] in ('nodeA', 'nodeB') and o['open-mode'] == 'append'
            assert ' drive stage ' in j['cmd'] and '--cores none' in j['cmd']
        # the runner's job record of the hour
        h = S.hour('05h_2026-01-03')
        from exopipe.phase3.core import runner
        j = runner.load_job(h['job'])
        assert j['chain'] == 'epoch' and len(j['chain_steps']) == 3 and '--resume' in j['forward_argv'], j
        assert j['nodes'] == [['nodeA', 1]]
        assert driver.step_forward(['--set', 'driver.RETRY=3', '--config', 'c', '--set', 'search.X=1', '--set', 'site.A=1']) \
            == ['--config', 'c', '--set', 'search.X=1', '--set', 'site.A=1']
        # run the jobs
        res = S.run_ready()
        assert [r[2] for r in res] == ['COMPLETED'] * 6, res
        it = S.item('05h_2026-01-03')
        assert it['status'] == 'done' and [e['status'] for e in it['steps']] == ['done'] * 3, it
        assert all(e['slurm_job'] and e['host'] and e['cores'] for e in it['steps']), it['steps']
        assert S.count('s2 05h_2026-01-03 u1 done') == 1
        out = S.tick()
        assert any(l.startswith('CLOSE 05h_2026-01-03') for l in out), out
        assert driver.load_hours()['05h_2026-01-03']['state'] == 'closing'
        assert driver.load_hours()['03h_2026-01-01']['state'] == 'waiting'          # bookings held while closing
        res = S.run_ready()
        assert sorted(r[1] for r in res) == ['exo3_04h_2026-01-02.fin', 'exo3_05h_2026-01-03.fin'], res
        fin = json.load(open(os.path.join(S.out, 'fin_05h_2026-01-03.json')))
        assert fin == {'chain_status': 'done', 'status': 'done', 'steps': ['done'] * 3}, fin
        out = S.tick()
        assert any(l.startswith('FINISH 05h_2026-01-03: done') for l in out), out
        # bookings released: 03h starts on nodeA, 06h on nodeB
        hrs = driver.load_hours()
        assert '05h_2026-01-03' not in hrs and hrs['03h_2026-01-01']['node'] == 'nodeA', hrs
        assert hrs['06h_2025-12-31']['state'] == 'active' and hrs['06h_2025-12-31']['node'] == 'nodeB'
        f = S.hour('05h_2026-01-03')
        assert f['state'] == 'finished' and f['booking'] == {} and f['released'] == {'nodeA:/scratchA': 50e9}
        assert all(s['acct'] and s['acct']['state'] == 'COMPLETED' and s['acct']['maxrss'] == 2048 * 1024
                   and s['acct']['nodes'] == 'nodeA' for s in f['submissions']), f['submissions']
        # `run status` reads the hour's job
        r = _cli('run', 'status', h['job'], '--cores', 'none', '--set', f'runner.JOBS_ROOT={S.jobs_root}')
        assert r.returncode == 0 and '05h_2026-01-03' in r.stdout and 'done' in r.stdout, r.stdout + r.stderr
        # a stage job run again after its step is done does nothing (requeue-safe)
        n = len(S.ran())
        r = _cli('drive', 'stage', h['job'], '05h_2026-01-03', '0', '--cores', 'none', '--config', S.cfg)
        assert r.returncode == 0 and 'already done' in r.stdout and len(S.ran()) == n, r.stdout + r.stderr
        # `drive status`
        r = _cli('drive', 'status', '--all', '--cores', 'none', '--config', S.cfg, env={'FAKE_SLURM_STATE': S.state})
        assert r.returncode == 0 and '05h_2026-01-03' in r.stdout and 'outcome done' in r.stdout, r.stdout + r.stderr
    # catch-up windows: newest night first, per tier; night hours use the night partition
    with Scenario('window', CHAIN3, WINDOW_HOURS={'night': 1, 'backlog': 2}) as S:
        S.tick(['01h_2026-02-01', '01h_2026-02-03', '02h_2026-02-02'])
        S.tick(['09h_2026-02-01', '10h_2026-02-01'], tier='night')
        hrs = driver.load_hours()
        st = {e: (h['state'], h['waiting_reason']) for e, h in hrs.items()}
        assert st['01h_2026-02-03'][0] == 'active' and st['02h_2026-02-02'][0] == 'active', st
        assert st['01h_2026-02-01'] == ('waiting', 'window: 2 of 2 backlog hours in flight'), st
        assert st['10h_2026-02-01'][0] == 'active' and st['09h_2026-02-01'] == ('waiting', 'window: 1 of 1 night hours in flight')
        parts = {j['name'].split('.')[0]: j['opts']['partition'] for j in S.jobs().values()}
        assert parts['exo3_10h_2026-02-01'] == 'nightp' and parts['exo3_01h_2026-02-03'] == 'backp', parts
    print('  1-2 chain end to end, booking, windows: ok')


def test_failed_parent_and_retry_limit():
    from exopipe.phase3.core import driver
    chain = ['s1 --epochs {item} --out OUT --units 2', 's2 --epochs {item} --out OUT --units 2 --fail-once FLAG',
             's3 --epochs {item} --out OUT --units 2']
    with Scenario('failonce', chain) as S:
        S.tick(['07h_2026-03-01'])
        res = S.run_ready()
        assert [(r[1].split('.')[-1], r[2]) for r in res] == [('s0', 'COMPLETED'), ('s1', 'FAILED')], res
        s3 = [j for j in S.jobs().values() if j['name'].endswith('.s2')][0]
        with FS.State(S.state) as st:
            assert FS.reason(st, s3) == 'DependencyNeverSatisfied'
        out = S.tick()
        assert any(l.startswith(f'CANCEL 07h_2026-03-01 Slurm jobs {s3["id"]}') for l in out), out
        assert any(l.startswith('SUBMIT 07h_2026-03-01') and 'stages 2..3' in l and 'resubmission 1 of 1' in l
                   for l in out), out
        assert S.jobs()[s3['id']]['state'] == 'CANCELLED'
        res = S.run_ready()
        assert [r[2] for r in res] == ['COMPLETED', 'COMPLETED'], res
        assert S.count('s1 07h_2026-03-01 u0 start') == 1 and S.count('s1 07h_2026-03-01 u1 start') == 1   # stage 1 not redone
        assert S.count('s2 07h_2026-03-01 u0 start') == 1 and S.count('s2 07h_2026-03-01 u0 skip') == 1    # its done unit skipped
        assert S.count('s2 07h_2026-03-01 u1 done') == 1
        it = S.item('07h_2026-03-01')
        assert it['status'] == 'done' and it['steps'][1]['attempt'] == 2, it
        S.tick()
        S.run_ready()
        S.tick()
        assert S.hour('07h_2026-03-01')['outcome'] == 'done' and S.hour('07h_2026-03-01')['failures'] == {'1': 1}
    chain = ['s1 --epochs {item} --out OUT --units 1', 's2 --epochs {item} --out OUT --units 2 --fail-always',
             's3 --epochs {item} --out OUT --units 1']
    with Scenario('limit', chain, RETRY=1) as S:
        S.tick(['08h_2026-03-02'])
        S.run_ready()
        S.tick()                                            # resubmission 1 of 1
        res = S.run_ready()
        assert [r[2] for r in res] == ['FAILED'], res
        out = S.tick()
        assert any(l.startswith('FAIL 08h_2026-03-02 at stage 2') for l in out), out
        assert any(l.startswith('CLOSE 08h_2026-03-02: chain failed') for l in out), out
        it = S.item('08h_2026-03-02')
        assert it['status'] == 'failed' and 'stage 2 (s2)' in it['failed_reason'], it
        res = S.run_ready()
        assert [r[1] for r in res] == ['exo3_08h_2026-03-02.fin'], res
        assert json.load(open(os.path.join(S.out, 'fin_08h_2026-03-02.json')))['chain_status'] == 'failed'
        S.tick()
        h = S.hour('08h_2026-03-02')
        assert h['state'] == 'finished' and h['outcome'] == 'failed', h
        assert not [j for j in S.jobs().values() if j['state'] == 'PENDING'], 'a job left pending'
    print('  3-4 failed parent resubmitted, retry limit: ok')


def test_refused_retry_and_no_input():
    from exopipe.phase3.core import driver
    chain = ['s1 --epochs {item} --out OUT --units 1', 's2 --epochs {item} --out OUT --units 2 --rc 2 --fail-once FLAG',
             's3 --epochs {item} --out OUT --units 1 --rc 3']
    with Scenario('refused', chain) as S:
        S.tick(['11h_2026-04-01'])
        res = S.run_ready()
        assert [r[3] for r in res] == [0, 2], res                  # drive stage exits 2 for a refused step
        out = S.tick()
        assert any(l.startswith('CLOSE 11h_2026-04-01: chain refused') for l in out), out
        assert any(l.startswith('CANCEL 11h_2026-04-01') for l in out), out
        S.run_ready()
        S.tick()
        h = S.hour('11h_2026-04-01')
        assert h['state'] == 'finished' and h['outcome'] == 'refused' and h['failures'] == {}, h
        # drive retry: from the refused stage, earlier stages and finished units kept
        driver.request('retry', ['11h_2026-04-01'])
        out = S.tick()
        assert any(l.startswith('RETRY 11h_2026-04-01') for l in out) and any('stages 2..3' in l for l in out), out
        S.run_ready()
        S.tick()
        S.run_ready()
        S.tick()
        h = S.hour('11h_2026-04-01')
        assert h['outcome'] == 'done', h
        it = S.item('11h_2026-04-01')
        assert [e['status'] for e in it['steps']] == ['done', 'done', 'no-input'] and it['status'] == 'done', it
        assert S.count('s1 11h_2026-04-01 u0 start') == 1 and S.count('s2 11h_2026-04-01 u0 skip') == 1
        assert json.load(open(os.path.join(S.out, 'fin_11h_2026-04-01.json')))['chain_status'] == 'done'
    print('  5 refused, retry, no-input: ok')


def test_requeue_kill_mid_stage():
    """The requeue test: a stage killed while its step runs resumes from its records without redoing finished units."""
    chain = ['s1 --epochs {item} --out OUT --units 2', 's2 --epochs {item} --out OUT --units 4 --sleep 0.7',
             's3 --epochs {item} --out OUT --units 1']
    with Scenario('requeue', chain, RETRY=2) as S:
        S.tick(['12h_2026-05-01'])
        by = {j['name'].split('.')[-1]: j['id'] for j in S.jobs().values()}
        FS.wait(S.state, by['s0'], FS.start(S.state, by['s0']))
        p = FS.start(S.state, by['s1'])
        rec2 = os.path.join(S.out, '_records', 's2', '12h_2026-05-01_u2.json')     # units 0 and 1 done, unit 2 running
        t0 = time.time()
        while not (os.path.exists(rec2) and json.load(open(rec2))['status'] == 'running'):
            assert time.time() - t0 < 120 and p.poll() is None, 'stage 2 did not reach its second unit'
            time.sleep(0.05)
        r = subprocess.run([PY, FAKE, 'scancel', by['s1']], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        rc = FS.wait(S.state, by['s1'], p)
        assert rc != 0 and S.jobs()[by['s1']]['state'] == 'CANCELLED', rc
        it = S.item('12h_2026-05-01')
        assert it['steps'][1]['status'] == 'interrupted' and it['status'] == 'error', it
        assert S.count('s2 12h_2026-05-01 u2 interrupted') == 1, S.ran()
        out = S.tick()
        assert any('stage 2 (s2) ended interrupted; resubmission 1 of 2' in l for l in out), out
        assert S.jobs()[by['s2']]['state'] == 'CANCELLED'       # the old stage 3 job (pending on the killed one)
        res = S.run_ready()
        assert [r[2] for r in res] == ['COMPLETED', 'COMPLETED'], res
        for u in (0, 1):
            assert S.count(f's2 12h_2026-05-01 u{u} start') == 1 and S.count(f's2 12h_2026-05-01 u{u} skip') == 1, S.ran()
        assert S.count('s2 12h_2026-05-01 u2 start') == 2 and S.count('s2 12h_2026-05-01 u2 done') == 1
        assert S.count('s2 12h_2026-05-01 u3 done') == 1 and S.count('s1 12h_2026-05-01 u0 start') == 1
        it = S.item('12h_2026-05-01')
        assert it['status'] == 'done' and it['steps'][1]['attempt'] == 2, it
        S.tick()
        S.run_ready()
        S.tick()
        assert S.hour('12h_2026-05-01')['outcome'] == 'done'
    print('  6 requeue: killed stage resumed from its records: ok')


def test_sbatch_refused_squeue_failed():
    from exopipe.phase3.config import CFG
    from exopipe.phase3.core import driver
    with Scenario('sbatchfail', CHAIN3) as S:
        with FS.State(S.state) as st:
            st['refuse_sbatch'] = True
        out = S.tick(['13h_2026-06-01'])
        assert any('sbatch of 13h_2026-06-01 stage 1 refused' in l for l in out), out
        h = driver.load_hours()['13h_2026-06-01']
        assert h['state'] == 'active' and h['submissions'] == [] and h['failures'] == {}, h
        with FS.State(S.state) as st:
            st['refuse_sbatch'] = False
        out = S.tick()
        assert any('first submission from stage 1' in l for l in out) and len(S.jobs()) == 3, out
        CFG.set('driver.SQUEUE', 'false')
        n = len(S.jobs())
        S.run_ready()                                       # stage jobs end; with squeue failing nothing is acted on
        out = S.tick(['14h_2026-06-01'])
        assert any('squeue failed' in l for l in out) and len(S.jobs()) == n, out
        assert driver.load_hours()['14h_2026-06-01']['state'] == 'waiting'
    print('  7 sbatch refused, squeue failed: ok')


def test_discovery():
    from exopipe.phase3.config import CFG
    from exopipe.phase3.core import driver
    with Scenario('discover', CHAIN3, FIRST_NIGHT='2026-01-01') as S:
        p2 = os.path.join(S.d, 'p2')
        CFG.set('paths.PHASE2_ROOT', p2)

        def run(lst, date, name, rec):
            d = os.path.join(p2, lst, date, name, 'QA')
            os.makedirs(d)
            if rec is not None:
                json.dump(rec, open(os.path.join(d, 'email_deferred.json'), 'w'))
        run('07h', '2026-01-03', 'Science_20260104_000000', {'lst': '07h', 'date': '2026-01-03'})
        run('07h', '2026-01-02', 'Science_20260103_000000', None)                       # Phase 2 not ended
        run('07h', '2025-12-31', 'Science_20260101_000000', {'lst': '07h', 'date': '2025-12-31'})   # before FIRST_NIGHT
        run('08h', '2026-01-03', 'Science_20260104_000000', {'lst': '09h', 'date': '2026-01-03'})   # names another hour
        run('09h', '2026-01-03', 'Science_20260104_000000', {'lst': '09h', 'date': '2026-01-03'})
        run('09h', '2026-01-03', 'Science_20260105_000000', None)                       # the latest run has not ended
        S.tick()
        hrs = driver.load_hours()
        assert sorted(hrs) == ['07h_2026-01-03'] and hrs['07h_2026-01-03']['tier'] == 'night', hrs
        assert {j['opts']['partition'] for j in S.jobs().values()} == {'nightp'}
        S.tick()
        assert sorted(driver.load_hours()) == ['07h_2026-01-03']                         # found once
    print('  8 discovery: ok')


def test_periodic():
    per = {'argv': ('-c', 'pass'), 'every_h': 24, 'tier': 'backlog', 'cpus': 1, 'mem': '1G', 'time': '00:01:00'}
    with Scenario('periodic', CHAIN3, PERIODIC=per) as S:
        from exopipe.phase3.core import driver
        out = S.tick()
        assert any(l.startswith('PERIODIC tess-orbits: due') for l in out), out
        j = [j for j in S.jobs().values() if j['name'] == 'exo3_periodic.tess-orbits']
        assert len(j) == 1 and j[0]['opts']['partition'] == 'backp' and j[0]['opts']['time'] == '00:01:00', j
        S.tick()
        assert len(S.jobs()) == 1                                # pending: not submitted again
        S.run_ready()
        S.tick()
        st = json.load(open(os.path.join(driver.root(), 'periodic.json')))['tess-orbits']
        assert st['acct']['state'] == 'COMPLETED' and len(S.jobs()) == 1, st          # accounted, not due
        st['submitted'] = '2000-01-01T00:00:00Z'
        driver._wjson(os.path.join(driver.root(), 'periodic.json'), {'tess-orbits': st})
        S.tick()
        assert len(S.jobs()) == 2                                # due again after its period
    per = dict(per, argv=('-c', 'raise SystemExit(5)'))
    with Scenario('periodicfail', CHAIN3, PERIODIC=per) as S:
        S.tick()
        S.run_ready()
        out = S.tick()
        assert any(l.startswith('ALARM periodic tess-orbits') and 'FAILED' in l for l in out), out
    # production config: the TESS refresh daily; the reader no longer refreshes
    from exopipe.phase3.config.steps.drive import driver as D
    from exopipe.phase3.config.steps.pages import pages_deep
    assert D.PERIODIC['tess-orbits']['every_h'] == 24 and pages_deep.TESS_REFRESH_IN_READER is False
    print('  9 periodic jobs: ok')


def test_lock_and_dry_run_cli():
    from exopipe.phase3.config import CFG
    from exopipe.phase3.core import driver
    from exopipe.phase3.core.record import Refusal
    with Scenario('lock', CHAIN3) as S:
        lk = driver.Lock()
        e = raises(Refusal, driver.Lock)
        assert 'another driver holds' in str(e) and f'pid {os.getpid()}' in str(e), e
        lk.release()
        driver.Lock().release()
        # the dry run: a real CLI process, the local host's scratch disk, nothing submitted or written
        host = socket.gethostname().split('.')[0]
        cfg2 = os.path.join(S.d, 'dry.cfg')
        with open(S.cfg) as f:
            text = f.read()
        roots = dict(CFG.site.SCRATCH_ROOT, **{host: S.d})
        with open(cfg2, 'w') as f:
            f.write(text + f"site.SCRATCH_ROOT = {roots!r}\ndriver.SCRATCH_BOOK_BYTES = 1.0\ndriver.SCRATCH_FLOOR_BYTES = 0.0\n")
        jobs_before = os.listdir(S.jobs_root) if os.path.isdir(S.jobs_root) else []
        r = _cli('drive', 'tick', '--dry-run', '--epochs', '15h_2026-07-01', '--nodes', host, '--partition', 'testp',
                 '--cores', 'none', '--config', cfg2, env={'FAKE_SLURM_STATE': S.state})
        assert r.returncode == 0, r.stdout + r.stderr
        o = r.stdout
        assert f'DRY-RUN START 15h_2026-07-01 (backlog) on {host}' in o and 'DRY-RUN SUBMIT 15h_2026-07-01' in o, o
        assert o.count('--partition=testp') == 3 and o.count('--dependency=') == 3 and 'step 3: s3 --epochs 15h_2026-07-01' in o, o
        assert FS.jobs(S.state) == {}, 'the dry run submitted'
        now = os.listdir(S.jobs_root) if os.path.isdir(S.jobs_root) else []
        assert now == jobs_before == ['_driver'] and os.listdir(os.path.join(S.jobs_root, '_driver')) == ['LOCK'], now
    print('  10 lock, dry run: ok')


def main():
    t0 = time.time()
    test_chain_end_to_end_booking_and_window()
    test_failed_parent_and_retry_limit()
    test_refused_retry_and_no_input()
    test_requeue_kill_mid_stage()
    test_sbatch_refused_squeue_failed()
    test_discovery()
    test_periodic()
    test_lock_and_dry_run_cli()
    print(f'test_drive: PASS ({time.time() - t0:.0f} s)')


if __name__ == '__main__':
    main()
