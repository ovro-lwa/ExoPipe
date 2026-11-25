"""The per-hour driver: Slurm jobs for each night-hour's chain, over the runner's stage records (`exopipe3 drive`).

Design: exopipe_ops/refactor_20261012/plan/ORCHESTRATION_DESIGN_20261117.md 5.2-5.3 and plan step 3 (Gregg approved 2026-11-17;
2026-11-24: no Slurm reservations, campaigns are ordinary queued jobs). Slurm places and confines the jobs and orders the tiers
(partitions; preemption by requeue); this module decides WHAT runs, from the records.

THE HOUR. An hour is an epoch item <LST>_<date> of the chain runner.CHAINS[driver.CHAIN] (the per-hour chain 'epoch'). Its stage
records are the runner's own (core/runner.py): ONE runner job per hour, created with `runner.submit(..., start=False)` (job.json,
items.txt, items/<epoch>.json with one entry per chain step, logs/<epoch>.log), so `exopipe3 run status <job>` shows the hour.
The step commands carry --resume (forwarded to every step), so a stage that runs again skips every unit its records show done.

THE JOBS. Each chain step k of an hour is one Slurm job `exopipe3 drive stage <job> <epoch> <k>` named <prefix><epoch>.s<k>,
submitted with --dependency=afterok:<the job of step k-1>,singleton on the hour's node (one node per hour: the chain's steps
share node-local scratch, the runner's rule; an hour that has jobs keeps its node). Each job's command runs as a Slurm job step
(driver.SRUN): a cancelled, preempted or timed-out job then sends SIGTERM to the stage process, which passes it on to the
chain step once (_Term), and the step records 'interrupted'.
`drive stage` (run_stage) is requeue-safe: it does nothing for a step its record shows done, no-input or cells-refused, and
otherwise runs the step (pinned to the job's allowed cores that are not recorder cores) and records it in the item record as
the runner's chain does, cell refusals included (runner.CELL_REFUSAL_CHAINS, the 'epoch' chain: a step's exit 2 whose
refusals all name their cell is recorded 'cells-refused' and the chain goes on; every later step gets the refused cells in
record.UPSTREAM_ENV; the hour then ends 'cells-refused' with refused_cells, refused_by_cell and summary for the email). After
the chain ends (done, no-input, cells-refused, refused or failed) one closing job `drive finally <job> <epoch>` runs the
chain's closing steps (runner.CHAIN_FINALLY: the hour's email).

EACH PASS (tick, once a minute under `drive loop`):
  1. requests of `drive add` / `drive retry` (files under requests/; the loop is the only writer of the hour records);
  2. discovery (driver.FIRST_NIGHT set): every hour <LST>_<date>, date >= FIRST_NIGHT, whose latest Phase 2 run holds
     driver.PHASE2_READY_FILE (Phase 2's last write in per-hour mode) enters as a 'night' hour (an hour not ready is looked
     at again after a delay doubling up to driver.DISCOVER_BACKOFF_MAX_S);
  3. squeue once (a failed squeue: nothing is done this pass); sacct for the jobs that ended (state, exit, elapsed, peak memory,
     node: recorded per submission, requirement 8 of the design);
  4. each hour in flight, from its record: the first step not done, no-input or cells-refused is the open step. A live job
     of it: wait - unless it is pending on a node Slurm reports down or drained (sinfo): then its jobs are cancelled and the
     hour waits again with the reason recorded (rebooked on any node when none of its stages has run; else it waits for its
     node). None (it ended without finishing - error, killed, timeout, out of memory, node failure - or a parent's job failed
     and left it pending with DependencyNeverSatisfied): cancel the hour's remaining jobs (exact ids) and resubmit the chain;
     a step that had started counts one failure and the steps run again by the runner's retry rule (resume_steps), and after
     driver.RETRY failures the hour ends 'failed'. A refused step ends the hour 'refused' (its pending children are
     cancelled). An ended chain gets its closing job (a refused sbatch is retried, not counted); when that has ended the
     hour's space booking is released, and once sacct has every submission's end it moves to finished/;
  5. waiting hours are submitted in catch-up windows (driver.WINDOW_HOURS per tier, night before backlog, newest night first),
     each after its space booking: driver.SCRATCH_BOOK_BYTES on the chosen node's scratch disk (and driver.SHARED_BOOK),
     against the disk's real free space minus every other booking minus the floor; an hour that does not fit waits;
  6. periodic jobs (driver.PERIODIC: the daily TESS orbit table refresh) are submitted when due.
Each hour record is written as soon as a submission, cancellation or runner job is made, not only at the end of the pass.
Every pass with --dry-run prints the actions and the sbatch command lines, submits and cancels nothing and writes nothing.

Not here (design, other items): page-build coalescing (the publish builder: the chain's pages step is the build request); the
Phase 1 deadline alarm (Phase 1 is not driven yet); Slurm configuration (plan step 1: partitions, CpuSpecList, licences).
"""
import datetime
import fcntl
import getpass
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import time

from exopipe.phase3.core import slurm
from exopipe.phase3.core.record import UPSTREAM_ENV
from exopipe.phase3.core.env import cores_text, parse_cores

CELLS_REFUSED = 'cells-refused'           # runner.CELLS_REFUSED: a step (or item) that went on past refused cells
# a chain step after which the chain goes on: done (exit 0), no-input (exit 3: nothing to do) or cells-refused (exit 2 whose
# refusals are all per cell, in a chain of runner.CELL_REFUSAL_CHAINS: the hour's other cells go on)
DONE_STEP = ('done', 'no-input', CELLS_REFUSED)
CHAIN_END = ('done', 'no-input', 'refused', 'failed', CELLS_REFUSED)    # item statuses that end an hour (runner.TERMINAL)
TIERS = ('night', 'backlog')
IN_FLIGHT = ('active', 'closing')
# hour states: waiting -> active -> closing (the closing job) -> accounting (every submission's sacct record) -> finished
DEP_NEVER = 'DependencyNeverSatisfied'
STALE_ACCT_S = 86400                       # s: a submission sacct still does not list after this is recorded 'unknown'
_DATE_RE = re.compile(r'^\d{4}-\d{2}-\d{2}$')
_NOT_READY = {}     # epoch -> (time of its next look, delay s): discovery back-off of the hours not ready, per driver process


def _cfg():
    from exopipe.phase3.config import CFG
    return CFG


def _runner():
    from exopipe.phase3.core import runner
    return runner


def _now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def _ts(text):
    return datetime.datetime.strptime(text, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=datetime.timezone.utc).timestamp()


def _wjson(path, obj):
    from exopipe.phase3.core.record import atomic_write_json
    atomic_write_json(path, obj)


def _rjson(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _gb(b):
    return f'{b / 1e9:.0f} GB'


def _host():
    from exopipe.phase3.core import env
    return env.host()


# ---------------------------------------------------------------------------------------------------------------------
# state: hours/<epoch>.json (waiting or in flight), finished/<epoch>.json, periodic.json, requests/, LOCK
# ---------------------------------------------------------------------------------------------------------------------

def root():
    cfg = _cfg()
    return cfg.driver.ROOT or os.path.join(cfg.runner.JOBS_ROOT, '_driver')


def hour_path(epoch, finished=False):
    return os.path.join(root(), 'finished' if finished else 'hours', f'{epoch}.json')


def load_hours():
    d = os.path.join(root(), 'hours')
    if not os.path.isdir(d):
        return {}
    out = {}
    for f in sorted(os.listdir(d)):
        if f.endswith('.json'):
            h = _rjson(os.path.join(d, f))
            if h:
                out[h['epoch']] = h
    return out


def finished_epochs():
    d = os.path.join(root(), 'finished')
    return {f[:-5] for f in os.listdir(d) if f.endswith('.json')} if os.path.isdir(d) else set()


def new_hour(epoch, tier, source):
    from exopipe.phase3.core import cells
    cells.parse_epoch(epoch)
    if tier not in TIERS:
        raise ValueError(f'tier {tier!r} is not one of {TIERS}')
    return {'epoch': epoch, 'tier': tier, 'source': source, 'state': 'waiting', 'added': _now(), 'job': None,
            'node': None, 'booking': {}, 'submissions': [], 'failures': {}, 'fin_failures': 0, 'outcome': None,
            'waiting_reason': None, 'notes': []}


def _note(h, text):
    h.setdefault('notes', []).append(f'{_now()} {text}')
    del h['notes'][:-20]


def _persist(ps, h):
    """Write the hour's record now, not only at the end of the pass: each submission, cancellation and new runner job is on
    disk before anything later in the pass can fail, so the next pass never submits the hour a second time."""
    if not ps.dry and h.get('state') not in (None, 'finished'):
        _wjson(hour_path(h['epoch']), h)


class Lock:
    """One driver at a time: an exclusive flock on <root>/LOCK (Lustre is mounted with flock: coherent across nodes), held
    for the process's life (released when it exits, however it exits). The file names the holder."""

    def __init__(self):
        os.makedirs(root(), exist_ok=True)
        self.path = os.path.join(root(), 'LOCK')
        self.fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o664)
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            holder = open(self.path).read().strip()
            os.close(self.fd)
            from exopipe.phase3.core.record import Refusal
            raise Refusal(f'another driver holds {self.path} ({holder or "holder not written"}); one driver at a time')
        os.ftruncate(self.fd, 0)
        os.write(self.fd, f'host {_host()} pid {os.getpid()} since {_now()}\n'.encode())

    def release(self):
        if self.fd is not None:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            os.close(self.fd)
            self.fd = None


def request(action, epochs, tier=None):
    """`drive add` / `drive retry`: a request file the driver applies at its next pass (it is the only writer of the hour
    records). Returns its path."""
    d = os.path.join(root(), 'requests')
    p = os.path.join(d, f'{time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())}_{_host()}_{os.getpid()}_{os.urandom(4).hex()}_'
                        f'{action}.json')     # unique: two requests of one process in one second never share a file
    _wjson(p, {'action': action, 'epochs': list(epochs), 'tier': tier, 'made': _now(), 'by': getpass.getuser()})
    return p


# ---------------------------------------------------------------------------------------------------------------------
# the stage records (the runner's item record of the hour)
# ---------------------------------------------------------------------------------------------------------------------

def item_path(job, item):
    r = _runner()
    return os.path.join(r.job_dir(job), 'items', f'{r._safe(item)}.json')


def open_step(j, rec):
    """Index of the first chain step whose record is not done or no-input (None: every step is)."""
    steps = rec.get('steps') or []
    for k in range(len(j['chain_steps'])):
        if not (k < len(steps) and steps[k].get('status') in DONE_STEP):
            return k
    return None


def item_outcome(j, rec):
    """The item status once every step is done, no-input or cells-refused (the runner's rule): 'cells-refused' when a step
    refused cells (complete with refused cells, never done), 'no-input' when every step had nothing to do, else 'done'."""
    steps = (rec.get('steps') or [])[:len(j['chain_steps'])]
    if any(s.get('status') == CELLS_REFUSED for s in steps):
        return CELLS_REFUSED
    return 'no-input' if steps and all(s.get('status') == 'no-input' for s in steps) else 'done'


def close_record(j, rec, status):
    """Set the item's end status with what the runner records beside it (runner._run_item): every cell refusal of the
    item (refused_cells), one entry per refused cell (refused_by_cell, the hour's email reads it) and the one-line
    outcome (summary). The entries of this attempt are those not queued again (a later entry from an earlier attempt was
    set back to queued when the chain was resubmitted)."""
    r = _runner()
    steps_now = [e for e in rec.get('steps') or [] if e.get('status') != 'queued']
    refused = r.refused_cells_of(steps_now)
    rec.update(status=status, refused_cells=refused, refused_by_cell=r.refused_by_cell(refused),
               summary=r.chain_outcome(status, steps_now, len(j['chain_steps'])))
    return rec


def resume_steps(j, rec):
    """The chain steps a retried hour runs again, by the runner's rule (runner._run_item): every step that did not end
    done (no-input included), and in a chain of runner.CELL_REFUSAL_CHAINS (job.json cell_refusals) every step from its
    first step that ended neither done nor no-input on. Indexes, in order."""
    steps = rec.get('steps') or []
    at = len(steps)
    if j.get('cell_refusals'):
        at = next((k for k, e in enumerate(steps) if e.get('status') not in ('done', 'no-input')), len(steps))
    return [k for k in range(len(j['chain_steps'])) if k >= at or k >= len(steps) or steps[k].get('status') != 'done']


def requeue(job, item, rec, ks):
    """Set the entries of steps ``ks`` back to 'queued' (their earlier status kept as previous_status); written when one
    changed. The steps then run again (run_stage); --resume still skips their finished units."""
    steps = rec.get('steps') or []
    changed = False
    for k in ks:
        if k < len(steps) and steps[k].get('status') != 'queued':
            steps[k]['previous_status'] = steps[k].get('status')
            steps[k]['status'] = 'queued'
            changed = True
    if changed:
        _write_item(job, item, rec)
    return changed


def stage_cores():
    """(cores text, why): the cores a stage's step runs on - those of this job's allocation (the Slurm cgroup) that no data
    recorder holds (runner.FREE_CORES_CMD). Requirement 1 of the design (the recorders first): until slurm.conf withholds the
    recorder cores (CpuSpecList, plan step 1) an allocation may include them, and the step is kept off them; an allocation
    of recorder cores only, or an unreadable recorder-core list, runs nothing."""
    allowed = os.sched_getaffinity(0)
    try:
        out = subprocess.run(_cfg().runner.FREE_CORES_CMD, shell=True, capture_output=True, text=True, timeout=60)
        free = parse_cores(out.stdout) if out.returncode == 0 else set()
    except (OSError, ValueError, subprocess.TimeoutExpired):
        free = set()
    if not free:
        return None, f'the recorder-free cores of this node could not be read ({_cfg().runner.FREE_CORES_CMD})'
    use = allowed & free
    if not use:
        return None, (f'the job\'s allocation (cores {cores_text(allowed)}) holds only data-recorder cores (free: '
                      f'{cores_text(free)})')
    return cores_text(use), None


class _Term:
    """SIGTERM to the stage process: passed on to the chain step ONCE. Slurm signals only the job step's task - this
    process, run under driver.SRUN - not its children (Slurm 24.05.2 on this cluster, 2026-11-24: a srun task got SIGTERM
    at scancel and at the time limit, jobs 264227 and 264229, while the step it had started ran on to SIGKILL after
    KillWait, 30 s, job 264219; without srun even the batch script's processes got none, job 264102). The step runs in a
    session of its own (start_new_session), so a signal to this process's group never reaches it a second time: a second
    SIGTERM would end it at once, without clean-up (cli._sigterm_interrupts). The step then records 'interrupted' and
    stops; this process records it and exits 143. A step killed without a record stays 'running' with no live job, which
    the driver reads as killed (advance) and resubmits; --resume redoes only the unfinished units."""

    def __init__(self):
        self.got = False
        self.child = None
        self.sent = False

    def forward(self):
        if self.got and self.child is not None and not self.sent:
            self.sent = True
            try:
                os.kill(self.child, signal.SIGTERM)
            except ProcessLookupError:
                pass

    def __call__(self, signum, frame):
        self.got = True
        self.forward()


def _write_item(job, item, rec):
    _wjson(item_path(job, item), rec)


def run_stage(job, item, k):
    """The body of a stage's Slurm job: run chain step ``k`` of ``item`` and record it in the item record, as the runner's
    chain does (one entry per step: status, rc, started, finished, attempt, host, Slurm job). Exit 0 when the step ended done
    or no-input (afterok then releases the next stage), 2 refused, 1 otherwise. Requeue-safe: a step the record shows done
    or no-input is not run again (exit 0), and a step that runs again gets --resume (forwarded at job creation) so its
    finished units are skipped."""
    r = _runner()
    cfg = _cfg()
    j = r.load_job(job)
    if not j.get('chain_steps'):
        print(f'drive stage: job {job} is not a chain job', flush=True)
        return 1
    argvs = r.step_argvs(j, item)
    if not 0 <= k < len(argvs):
        print(f'drive stage: step {k} is outside the chain of {len(argvs)} steps', flush=True)
        return 1
    rec = r.item_state(job, item)
    steps = list(rec.get('steps') or [])
    tag = f'{item} step {k + 1}/{len(argvs)} {argvs[k][0]}'
    if k < len(steps) and steps[k].get('status') in DONE_STEP:
        print(f'===== {_now()} {tag}: already {steps[k]["status"]} (record); nothing to run', flush=True)
        return 0
    for i in range(k):
        if not (i < len(steps) and steps[i].get('status') in DONE_STEP):
            print(f'===== {_now()} {tag}: step {i + 1} is not done in the record; not run', flush=True)
            return 1
    prev = steps[k] if k < len(steps) else {}
    root_ = j.get('code_root') or j['pkg_root']
    entry = {'step': k, 'name': argvs[k][0], 'status': 'running', 'started': _now(), 'host': _host(),
             'attempt': int(prev.get('attempt', 0)) + 1, 'slurm_job': os.environ.get('SLURM_JOB_ID'),
             'slurm_restart': int(os.environ.get('SLURM_RESTART_COUNT') or 0)}
    if k < len(steps):
        steps[k] = entry
    else:
        steps.append(entry)
    rec.update(status='running', host=entry['host'], chain=j.get('chain'), steps=steps, step=k,
               log=os.path.join(r.job_dir(job), 'logs', f'{r._safe(item)}.log'), driver=True)
    rec['attempts'] = int(rec.get('attempts', 0)) + 1
    try:                                           # gate R-ROOTS on the node the stage runs on (a disk not mounted here)
        miss = r.missing_roots(j.get('chain_roots') or [])
    except Exception as e:                         # the check could not run: an error, retried by the driver's rule
        entry.update(status='error', rc=None, finished=_now(), reason=f'R-ROOTS check raised {type(e).__name__}: {e}')
        rec.update(status='error', rc=None, finished=_now())
        _write_item(job, item, rec)
        print(f'===== {_now()} {tag}: not run - {entry["reason"]}', flush=True)
        return 1
    if miss:
        why = (f'R-ROOTS: this host ({_host()}) cannot read root(s) the chain {j.get("chain")} reads: '
               + '; '.join(f'{ref} = {p}: {w}' for ref, p, w in miss))
        entry.update(status='refused', rc=2, finished=_now(), reason=why)
        rec['refusal'] = why
        close_record(j, rec, 'refused')
        rec.update(rc=2, finished=_now())
        _write_item(job, item, rec)
        print(f'===== {_now()} {tag}: REFUSED before the step - {why}', flush=True)
        return 2
    cores, why = stage_cores()
    if cores is None:
        entry.update(status='error', rc=None, finished=_now(), reason=f'placement: {why}')
        rec.update(status='error', rc=None, finished=_now())
        _write_item(job, item, rec)
        print(f'===== {_now()} {tag}: not run - {why}', flush=True)
        return 1
    full = [j['python']] + list(cfg.driver.STEP_HEAD) + argvs[k] + ['--cores', cores]
    entry.update(argv=full, cores=cores)
    rec['argv'] = full
    cellwise = bool(j.get('cell_refusals'))
    env = r._child_env(j, root_)
    env.pop(r.OUTCOME_FD_ENV, None)
    env.pop(UPSTREAM_ENV, None)
    if cellwise:
        # the cells earlier steps of the hour refused: the unit framework refuses them in this step before any work
        # (record.UPSTREAM_ENV, code UPSTREAM-REFUSED) - the runner's chain rule
        upstream = r.upstream_of(r.refused_cells_of(steps[:k]))
        if upstream:
            env[UPSTREAM_ENV] = json.dumps(upstream, sort_keys=True)
    print(f'\n===== {_now()} {_host()} slurm job {entry["slurm_job"]} (restart {entry["slurm_restart"]}) attempt '
          f'{entry["attempt"]} {tag}: {" ".join(map(shlex.quote, full))}'
          + (f'  [code {j["code_commit"]}]' if j.get('code_commit') else ''), flush=True)
    term = _Term()
    old = signal.signal(signal.SIGTERM, term)
    ofile = tempfile.TemporaryFile(prefix='exopipe3_step_outcome_')      # the step's outcome (runner.step_outcome)
    try:
        env[r.OUTCOME_FD_ENV] = str(ofile.fileno())
        p = subprocess.Popen(full, stdin=subprocess.DEVNULL, cwd=root_, env=env, pass_fds=(ofile.fileno(),),
                             start_new_session=True)
        term.child = p.pid
        term.forward()                             # a SIGTERM that came while the step was being started
        entry['pid'] = rec['pid'] = p.pid
        _write_item(job, item, rec)
        rc = p.wait()
        outcome = r.read_outcome(ofile)
    finally:
        signal.signal(signal.SIGTERM, old)
        ofile.close()
    s = 'interrupted' if term.got else r.RC_STATUS.get(rc, 'error')
    cells = r.cell_refusals(outcome, rc) if cellwise and not term.got else None
    entry.update(status=CELLS_REFUSED if cells is not None else s, rc=rc, finished=_now())
    if outcome is not None:
        entry['units'] = r.unit_counts(outcome)
    if cells is not None:
        entry['refused_cells'] = cells
        entry['upstream_refused_units'] = sum(1 for u in outcome.get('units') or [] if u.get('upstream'))
        for c in cells:
            print(f'===== {_now()} {tag}: refused cell {c["cell"]}' + (f' {c["subband"]}' if c.get('subband') else '')
                  + f' (unit {c["unit"]}, code {c["code"] or "none named"}): {c["reason"]}', flush=True)
    s = entry['status']
    if s in DONE_STEP:
        if k == len(argvs) - 1:
            close_record(j, rec, item_outcome(j, rec))
        else:
            rec['status'] = 'running'
    elif s == 'refused':
        close_record(j, rec, 'refused')
    else:
        rec['status'] = 'error'
    rec.update(rc=rc, finished=_now())
    _write_item(job, item, rec)
    print(f'===== {_now()} {tag}: {s} (exit {rc}); item {rec["status"]}', flush=True)
    if s in DONE_STEP:
        return 0
    return 2 if s == 'refused' else (143 if s == 'interrupted' else 1)


def run_finally(job, item):
    """The body of an hour's closing job: the chain's closing steps (runner.CHAIN_FINALLY), each once, with the item record
    for {record}; their exit codes are recorded under 'finally' and never change the item's status (the runner's rule).
    Requeue-safe: a closing step that has ended is not run again."""
    r = _runner()
    j = r.load_job(job)
    tmpls = j.get('chain_finally') or []
    rec = r.item_state(job, item)
    status = rec.get('status')
    if status not in CHAIN_END:
        print(f'drive finally: item {item} is {status!r}, not ended; no closing step runs', flush=True)
        return 1
    ipath = item_path(job, item)
    rec['chain_status'] = status
    fin = list(rec.get('finally') or [])
    root_ = j.get('code_root') or j['pkg_root']
    head = [j['python']] + list(_cfg().driver.STEP_HEAD)
    for k, tmpl in enumerate(tmpls):
        if k < len(fin) and fin[k].get('status') not in (None, 'running'):
            continue
        full = head + [a.replace('{item}', item).replace('{record}', ipath) for a in r._subst_vars(tmpl)] \
            + list(j.get('forward_argv') or [])
        entry = {'step': k, 'name': tmpl[0], 'argv': full, 'status': 'running', 'started': _now(), 'host': _host(),
                 'slurm_job': os.environ.get('SLURM_JOB_ID')}
        if k < len(fin):
            fin[k] = entry
        else:
            fin.append(entry)
        rec['finally'] = fin
        _write_item(job, item, rec)
        print(f'\n===== {_now()} {_host()} closing step {k + 1}/{len(tmpls)} (chain {status}): '
              f'{" ".join(map(shlex.quote, full))}', flush=True)
        term = _Term()
        old = signal.signal(signal.SIGTERM, term)
        try:
            p = subprocess.Popen(full, stdin=subprocess.DEVNULL, cwd=root_, env=r._child_env(j, root_),
                                 start_new_session=True)
            term.child = p.pid
            term.forward()
            entry['pid'] = p.pid
            _write_item(job, item, rec)
            frc = p.wait()
        finally:
            signal.signal(signal.SIGTERM, old)
        entry.update(status='interrupted' if term.got else r.RC_STATUS.get(frc, 'error'), rc=frc, finished=_now())
        _write_item(job, item, rec)
        if term.got:
            return 143
    rec['finally'] = fin
    _write_item(job, item, rec)
    return 0


# ---------------------------------------------------------------------------------------------------------------------
# Slurm submission of an hour's stages
# ---------------------------------------------------------------------------------------------------------------------

def stage_name(epoch, k):
    return f'{_cfg().driver.JOB_PREFIX}{epoch}.s{k}'


def fin_name(epoch):
    return f'{_cfg().driver.JOB_PREFIX}{epoch}.fin'


def periodic_name(name):
    return f'{_cfg().driver.JOB_PREFIX}periodic.{name}'


def _job_env(pkg_root, site_file):
    """The variables of the runner's step environment (runner._child_env: the package root first on PYTHONPATH,
    EXOPIPE3_SITE, runner.ENV_DEFAULTS) that differ from this process's; the job gets the rest from it (--export=ALL)."""
    env = _runner()._child_env({'pkg_root': pkg_root, 'site_file': site_file})
    return {k: v for k, v in env.items() if os.environ.get(k) != v}


def srun_argv(cpus):
    """The job-step prefix of a job's command (driver.SRUN; [] when ''): one task with every core of the job."""
    head = shlex.split(_cfg().driver.SRUN or '')
    return head + ['--ntasks=1', '--nodes=1', f'--cpus-per-task={int(cpus)}', '--export=ALL'] if head else []


def shell_command(pkg_root, site_file, argv, cpus=None):
    """A job's command line: cd to the package tree, then exec the command with the runner's step environment, as a job
    step (srun_argv) when ``cpus`` is given."""
    env = _job_env(pkg_root, site_file)
    argv = (srun_argv(cpus) if cpus else []) + list(argv)
    return (f'cd {shlex.quote(pkg_root)} && exec env ' + ' '.join(f'{k}={shlex.quote(v)}' for k, v in sorted(env.items()))
            + ' ' + ' '.join(map(shlex.quote, argv)))


def resources(step_name):
    res = _cfg().driver.STEP_RESOURCES
    return dict(res.get(step_name) or res['default'])


def step_forward(forward):
    """The driver's forwarded arguments without its own section's --set / --ruling-for (driver.*): the steps get the rest
    (a --config file goes to both)."""
    out, i = [], 0
    while i < len(forward):
        a = forward[i]
        if a in ('--set', '--ruling-for') and i + 1 < len(forward):
            if not forward[i + 1].startswith('driver.'):
                out += [a, forward[i + 1]]
            i += 2
            continue
        out.append(a)
        i += 1
    return out


class Pass:
    """One pass of the driver (tick): its options, the live Slurm jobs and the actions taken (printed, and in a dry run the
    only effect)."""

    def __init__(self, dry=False, nodes=None, partition=None, forward=(), code=None, out=None):
        self.dry = dry
        self.nodes_spec = nodes
        self.partition = partition
        self.forward = list(forward or ())          # to the driver's own job processes (drive stage / finally)
        self.step_forward = step_forward(self.forward)   # to the chain's steps
        self.code = code
        self.user = getpass.getuser()
        self.live = {}
        self.actions = []
        self.out = out or (lambda s: print(s, flush=True))
        self.changed = set()
        self._free = {}             # (host, path) -> free bytes, read once per pass (free_bytes runs ssh for another host)
        self._states = {}           # node -> sinfo state, read once per pass (None: sinfo failed)

    def free(self, host, path):
        if (host, path) not in self._free:
            self._free[(host, path)] = free_bytes(host, path)
        return self._free[(host, path)]

    def node_state(self, host):
        """The node's sinfo state ('' when sinfo failed or does not list it); one sinfo per pass for the pool's nodes."""
        if host not in self._states:
            ask = [n for n in dict.fromkeys(self.nodes() + [host]) if n not in self._states]
            got = slurm.node_states(ask)
            for n in ask:
                self._states[n] = (got or {}).get(n, '')
        return self._states[host]

    def act(self, text):
        self.actions.append(text)
        self.out(('DRY-RUN ' if self.dry else '') + text)

    def partition_of(self, tier):
        return self.partition or (_cfg().driver.PARTITIONS or {}).get(tier)

    def nodes(self):
        spec = self.nodes_spec or _cfg().driver.NODE_POOL
        if not spec:
            return []
        return [h for h, _ in _runner().parse_nodes(spec)]

    def hour_jobs(self, epoch):
        """{key: [(job id, state, reason)]} of the hour's live jobs; key = stage index or 'fin'."""
        pre = f'{_cfg().driver.JOB_PREFIX}{epoch}.'
        out = {}
        for jid, v in self.live.items():
            nm = v['name']
            if not nm.startswith(pre):
                continue
            tail = nm[len(pre):]
            key = 'fin' if tail == 'fin' else (int(tail[1:]) if tail[:1] == 's' and tail[1:].isdigit() else None)
            if key is not None:
                out.setdefault(key, []).append((jid, v['state'], v['reason']))
        return out


def _stage_argv(job, epoch, k, forward, jobs_root):
    return ['drive', 'stage', job, epoch, str(k), '--cores', 'none', '--set', f'runner.JOBS_ROOT={jobs_root}'] + list(forward)


def _fin_argv(job, epoch, forward, jobs_root):
    return ['drive', 'finally', job, epoch, '--cores', 'none', '--set', f'runner.JOBS_ROOT={jobs_root}'] + list(forward)


def submit_chain(ps, h, j, k0, why):
    """Submit stages k0 .. end of the hour's chain, each after the one before (afterok) and never beside another job of the
    same stage (singleton), on the hour's node. Records each submission in the hour; stops at a refused sbatch (the next
    pass resubmits from the open step)."""
    cfg = _cfg()
    r = _runner()
    epoch = h['epoch']
    part = ps.partition_of(h['tier'])
    job = h['job'] or '<new job>'
    jobs_root = j.get('jobs_root') or r.jobs_root()
    log = os.path.join(jobs_root, job, 'logs', f'{r._safe(epoch)}.log')
    prev = None
    n = len(j['chain_steps'])
    ps.act(f'SUBMIT {epoch} ({h["tier"]}, node {h["node"]}, partition {part}) stages {k0 + 1}..{n} of {n}: {why}')
    for k in range(k0, n):
        res = resources(j['chain_steps'][k][0])
        dep = ','.join(([f'afterok:{prev}'] if prev else []) + ['singleton'])
        cmd = shell_command(j['pkg_root'], j.get('site_file'),
                            [j['python'], '-m', 'exopipe.phase3'] + _stage_argv(job, epoch, k, ps.forward, jobs_root),
                            cpus=res['cpus'])
        kw = dict(name=stage_name(epoch, k), partition=part, time_limit=res['time'], mem=res['mem'], cpus=res['cpus'],
                  output=log, chdir=j['pkg_root'], nodelist=h['node'], dependency=dep,
                  licenses=cfg.driver.LICENSES or None)
        if ps.dry:
            ps.out('    ' + ' '.join(map(shlex.quote, slurm.sbatch_argv(cmd, **kw))))
            prev = f'<job of stage {k + 1}>'
            continue
        try:
            jid = slurm.submit(cmd, **kw)
        except slurm.SlurmError as e:
            _note(h, f'sbatch of stage {k + 1} refused: {e}')
            ps.out(f'  sbatch of {epoch} stage {k + 1} refused: {e}')
            break
        h['submissions'].append({'kind': 'stage', 'step': k, 'jobid': jid, 'submitted': _now(), 'why': why,
                                 'partition': part, 'node': h['node'], 'acct': None})
        _persist(ps, h)
        ps.out(f'  {stage_name(epoch, k)} -> Slurm job {jid}' + (f' (after {prev})' if prev else ''))
        prev = jid
    _persist(ps, h)
    ps.changed.add(epoch)


def _usable(ps, node):
    """False only when Slurm reports the node unusable (an unknown state counts as usable: no action on it)."""
    state = ps.node_state(node)
    return not state or slurm.node_usable(state)


def submit_finally(ps, h, j, why):
    """Submit the hour's closing job: on the hour's node, or, when Slurm reports that node unusable, on the first usable node
    of the pool (the closing steps read the item record, not the node's scratch)."""
    cfg = _cfg()
    r = _runner()
    epoch = h['epoch']
    node = h['node'] if _usable(ps, h['node']) else next((n for n in ps.nodes() if _usable(ps, n)), h['node'])
    part = ps.partition_of(h['tier'])
    jobs_root = j.get('jobs_root') or r.jobs_root()
    res = cfg.driver.FINALLY_RESOURCES
    cmd = shell_command(j['pkg_root'], j.get('site_file'),
                        [j['python'], '-m', 'exopipe.phase3'] + _fin_argv(h['job'], epoch, ps.forward, jobs_root),
                        cpus=res['cpus'])
    kw = dict(name=fin_name(epoch), partition=part, time_limit=res['time'], mem=res['mem'], cpus=res['cpus'],
              output=os.path.join(jobs_root, h['job'], 'logs', f'{r._safe(epoch)}.log'), chdir=j['pkg_root'],
              nodelist=node, dependency='singleton')
    ps.act(f'CLOSE {epoch}: chain {h["outcome"]}; closing steps {len(j["chain_finally"])} ({why})'
           + (f'; on {node}: Slurm reports {h["node"]} {ps.node_state(h["node"])}' if node != h['node'] else ''))
    if ps.dry:
        ps.out('    ' + ' '.join(map(shlex.quote, slurm.sbatch_argv(cmd, **kw))))
        return
    try:
        jid = slurm.submit(cmd, **kw)
    except slurm.SlurmError as e:
        _note(h, f'sbatch of the closing job refused: {e}')
        ps.out(f'  sbatch of {epoch} closing job refused: {e}')
        _persist(ps, h)
        return
    h['submissions'].append({'kind': 'finally', 'step': None, 'jobid': jid, 'submitted': _now(), 'why': why,
                             'partition': part, 'node': node, 'acct': None})
    _persist(ps, h)
    ps.out(f'  {fin_name(epoch)} -> Slurm job {jid}')
    ps.changed.add(epoch)


def _cancel(ps, h, ids, why):
    ids = sorted(set(ids))
    if not ids:
        return
    ps.act(f'CANCEL {h["epoch"]} Slurm jobs {",".join(ids)}: {why}')
    if ps.dry:
        return
    try:
        slurm.cancel(ids)
    except slurm.SlurmError as e:
        ps.out(f'  scancel failed: {e}')
    for i in ids:
        ps.live.pop(i, None)
    _note(h, f'cancelled {",".join(ids)}: {why}')
    _persist(ps, h)
    ps.changed.add(h['epoch'])


# ---------------------------------------------------------------------------------------------------------------------
# one hour's progress
# ---------------------------------------------------------------------------------------------------------------------

def _end_chain(ps, h, j, rec, outcome, why):
    h['outcome'] = outcome
    ps.changed.add(h['epoch'])
    live = [jid for key, js in ps.hour_jobs(h['epoch']).items() if key != 'fin' for jid, _, _ in js]
    _cancel(ps, h, live, f'the chain ended {outcome}; these stages can no longer run')
    if rec.get('status') != outcome and not ps.dry:
        close_record(j, rec, outcome)           # e.g. every step ended in an earlier pass: the record gets its end
        rec['finished'] = _now()
        _write_item(h['job'], h['epoch'], rec)
    if j.get('chain_finally'):
        h['state'] = 'closing'
        h['closing_from'] = len(h['submissions'])     # closing-job failures count only this closing's submissions
        submit_finally(ps, h, j, why)
    else:
        finish(ps, h, why)


def mark_failed(ps, h, j, rec, k, why):
    """The hour's chain ends 'failed' at step k (the driver's retries are used up): written into the item record (no job of
    the hour is live, so the driver is its only writer now)."""
    ps.act(f'FAIL {h["epoch"]} at stage {k + 1}: {why}')
    if ps.dry:
        return
    close_record(j, rec, 'failed')
    rec['failed_reason'] = f'stage {k + 1} ({j["chain_steps"][k][0]}): {why}'
    rec['finished'] = _now()
    _write_item(h['job'], h['epoch'], rec)


def advance(ps, h):
    """One hour in flight: act on its record and its live jobs (module docstring, pass step 4)."""
    cfg = _cfg()
    r = _runner()
    epoch = h['epoch']
    try:
        j = r.load_job(h['job'])
    except SystemExit as e:
        _note(h, f'runner job record unreadable: {e}')
        ps.out(f'{epoch}: runner job {h["job"]} unreadable ({e}); nothing done')
        return
    rec = r.item_state(h['job'], epoch)
    jobs = ps.hour_jobs(epoch)
    n = len(j['chain_steps'])
    if h['state'] == 'closing':
        fins = jobs.get('fin') or []
        subs = [x for x in h['submissions'][int(h.get('closing_from') or 0):] if x['kind'] == 'finally']
        fnode = subs[-1]['node'] if subs else h['node']
        if fins and all(st == 'PENDING' for _, st, _ in fins) and not _usable(ps, fnode):
            # pending on a node Slurm cannot use: cancelled and submitted again on a usable node (no failure counted)
            _cancel(ps, h, [jid for jid, _, _ in fins], f'the closing job is pending on {fnode}, which Slurm reports '
                                                         f'{ps.node_state(fnode)}')
            if not ps.dry and subs:
                h['fin_counted'] = subs[-1]['jobid']
            submit_finally(ps, h, j, f'rebooked: {fnode} is {ps.node_state(fnode)}')
            return
        if fins:
            return
        fin = rec.get('finally') or []
        nfin = len(j.get('chain_finally') or [])
        if len(fin) >= nfin and all(e.get('status') not in (None, 'running') for e in fin[:nfin]):
            finish(ps, h, 'closing steps ended')
            return
        # a failure is a closing JOB of this closing that ended without finishing the steps; a pass whose sbatch refused
        # the closing job (no new submission) submits it again without counting one
        if subs and h.get('fin_counted') != subs[-1]['jobid']:
            f = int(h.get('fin_failures', 0)) + 1
            if not ps.dry:
                h['fin_failures'], h['fin_counted'] = f, subs[-1]['jobid']
            if f > cfg.driver.RETRY:
                _note(h, f'closing job ended {f} times without finishing its steps')
                finish(ps, h, f'closing steps unfinished after {f} jobs')
                return
            why = (f'resubmit: closing job {subs[-1]["jobid"]} ended without finishing its steps (failure {f}; '
                   f'driver.RETRY {cfg.driver.RETRY})')
        else:
            why = 'the closing job is not in Slurm (its sbatch was refused): submitted again'
        submit_finally(ps, h, j, why)
        return
    status = rec.get('status')
    k = open_step(j, rec)
    steps = rec.get('steps') or []
    s = steps[k].get('status') if k is not None and k < len(steps) else None
    if k is None:
        _end_chain(ps, h, j, rec, item_outcome(j, rec), 'every stage done, no-input or cells-refused')
        return
    if status in ('refused', 'failed') or s == 'refused':
        _end_chain(ps, h, j, rec, 'refused' if s == 'refused' or status == 'refused' else 'failed',
                   f'stage {k + 1} ({j["chain_steps"][k][0]}) {s or status}')
        return
    mine = jobs.get(k) or []
    dep_never = [jid for jid, st, why in mine if why == DEP_NEVER]
    if mine and not dep_never:
        _node_unusable(ps, h, j, rec, jobs, k)     # pending on a node Slurm cannot use: cancelled, back to waiting
        return                                     # else the open stage's job is pending or running: wait
    others = [jid for key, js in jobs.items() if key != 'fin' for jid, _, _ in js]
    k0 = k
    if dep_never:
        why = f'stage {k + 1}\'s job waits on a parent job that failed ({DEP_NEVER}) although the record shows the parent done'
    elif s in ('error', 'interrupted', 'running'):
        why = f'stage {k + 1} ({j["chain_steps"][k][0]}) ended {s if s != "running" else "without recording an end (killed)"}'
    elif any(sub['kind'] == 'stage' and sub['step'] == k for sub in h['submissions']):
        why = f'stage {k + 1} has no live job and never started (cancelled or lost while pending)'
    else:
        why = f'first submission from stage {k + 1}'
    if h.get('why_next') and not (s in ('error', 'interrupted', 'running') and not dep_never):
        why = f'{h["why_next"]}; {why}'
    if not ps.dry:
        h.pop('why_next', None)
    _cancel(ps, h, others, f'resubmitting from stage {k + 1}')
    if s in ('error', 'interrupted', 'running') and not dep_never:
        # one failure per ended attempt (its start and Slurm job): a pass that could not resubmit (sbatch refused) does not
        # count the same end again
        attempt = f'{steps[k].get("slurm_job")}|{steps[k].get("started")}'
        counted = h.setdefault('counted', {})
        f = int(h['failures'].get(str(k), 0)) + (counted.get(str(k)) != attempt)
        if f > cfg.driver.RETRY:
            mark_failed(ps, h, j, rec, k, f'{why}; {f - 1} resubmissions used (driver.RETRY {cfg.driver.RETRY})')
            if not ps.dry:
                rec = r.item_state(h['job'], epoch)
            _end_chain(ps, h, j, rec, 'failed', why)
            return
        if not ps.dry:
            h['failures'][str(k)] = f
            counted[str(k)] = attempt
        why += f'; resubmission {f} of {cfg.driver.RETRY}'
        # the runner's retry rule (resume_steps): the steps that did not end done run again (no-input included), and in a
        # cell-refusal chain every step from the first that ended neither done nor no-input
        ks = resume_steps(j, rec)
        k0 = ks[0] if ks else k
        if not ps.dry:
            requeue(h['job'], epoch, rec, ks)
        if k0 != k:
            why += f'; the runner\'s retry rule runs again from stage {k0 + 1}'
    submit_chain(ps, h, j, k0, why)


def _node_unusable(ps, h, j, rec, jobs, k):
    """The open stage's job is pending on the hour's node and Slurm reports that node unusable (down, drained, draining,
    not responding, ...; slurm.node_usable): cancel the hour's jobs (exact ids) and put the hour back to waiting with the
    reason recorded. An hour none of whose stages has run is rebooked on any node; one whose stages ran there keeps the node
    (their scratch is on it) and waits for it, holding no window place. True when it acted."""
    mine = jobs.get(k) or []
    node = h.get('node')
    if not node or not mine or any(st != 'PENDING' for _, st, _ in mine):
        return False
    state = ps.node_state(node)
    if not state or slurm.node_usable(state):       # unknown (sinfo failed or does not list it) or usable: wait
        return False
    reasons = ','.join(sorted({why for _, _, why in mine}))
    bound = bool(rec.get('steps'))
    why = (f'stage {k + 1}\'s Slurm job {",".join(jid for jid, _, _ in mine)} is pending ({reasons}) on {node}, which Slurm '
           f'reports {state}')
    ids = [jid for key, js in jobs.items() if key != 'fin' for jid, _, _ in js]
    _cancel(ps, h, ids, why)
    ps.act(f'REBOOK {h["epoch"]}: {why}; ' + (f'its stages ran on {node} (scratch): it waits for that node' if bound else
                                              'no stage has run: booked again on any usable node'))
    if ps.dry:
        return True
    h.setdefault('rebooked', []).append({'time': _now(), 'node': node, 'node_state': state, 'pending_reason': reasons,
                                         'kept_node': bound})
    h.update(state='waiting', booking={}, why_next=f'rebooked after {node} became {state}')
    if not bound:
        h['node'] = None
    h['waiting_reason'] = why + (f'; waits for {node} (its stages\' scratch is there)' if bound else '; rebooking')
    _note(h, h['waiting_reason'])
    _persist(ps, h)
    ps.changed.add(h['epoch'])
    return True


def finish(ps, h, why):
    """The hour has ended: its booking is released, and it moves to finished/ once every submission has its sacct record
    (settle); until then it stays in hours/ as 'accounting' (no booking, no window place)."""
    ps.act(f'FINISH {h["epoch"]}: {h.get("outcome")}; booking released ({why})')
    if ps.dry:
        return
    h['state'] = 'accounting'
    h['finished'] = _now()
    h['released'] = dict(h.get('booking') or {})
    h['booking'] = {}
    _note(h, f'finished {h.get("outcome")}: {why}')
    ps.changed.add(h['epoch'])
    account(ps, {h['epoch']: h})
    settle(ps, h)


def settle(ps, h):
    """An 'accounting' hour whose every submission has its sacct record (account: an end state, or 'unknown' a day after
    submission) moves to finished/. True when it moved."""
    if ps.dry or h.get('state') != 'accounting':
        return False
    if any(x.get('acct') is None for x in h.get('submissions') or []):
        _persist(ps, h)
        return False
    h['state'] = 'finished'
    _wjson(hour_path(h['epoch'], finished=True), h)
    try:
        os.unlink(hour_path(h['epoch']))
    except FileNotFoundError:
        pass
    ps.changed.discard(h['epoch'])
    return True


# ---------------------------------------------------------------------------------------------------------------------
# space booking and node choice
# ---------------------------------------------------------------------------------------------------------------------

def free_bytes(host, path):
    """Free bytes of the file system holding ``path`` on ``host`` (statvfs here; on another host `stat -f` over ssh, the
    runner's ssh settings). None when it cannot be read."""
    if host in (None, '', 'localhost', _host()):
        try:
            st = os.statvfs(path)
        except OSError:
            return None
        return st.f_bavail * st.f_frsize
    env = dict(os.environ)
    env.update(_cfg().runner.SSH_PREFIX_ENV)
    try:
        r = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', host,
                            f'stat -f -c "%a %S" {shlex.quote(path)}'], capture_output=True, text=True, timeout=60, env=env)
        a, s = r.stdout.split()
        return int(a) * int(s) if r.returncode == 0 else None
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def booked(hours, key, skip=None):
    return sum(float(b) for h in hours.values() if h['epoch'] != skip and h['state'] in IN_FLIGHT
               for k, b in (h.get('booking') or {}).items() if k == key)


def pinned_node(h):
    """The node an hour must run on, or None: an hour that already has a runner job and a node keeps it (its finished
    stages' scratch is there; a retry or a rebook never moves it)."""
    return h.get('node') if h.get('job') else None


def book(ps, hours, h):
    """(node, booking, None) for the hour, or (None, None, why): the node (the hour's own node when it has one, pinned_node;
    else of ps.nodes()) that Slurm reports usable and whose scratch disk has the most room after the other hours' bookings
    and the floor, if that room holds driver.SCRATCH_BOOK_BYTES; and every driver.SHARED_BOOK path with room for its bytes.
    Free space and node states are read once per pass (Pass.free, Pass.node_state)."""
    cfg = _cfg()
    need, floor = float(cfg.driver.SCRATCH_BOOK_BYTES), float(cfg.driver.SCRATCH_FLOOR_BYTES)
    pin = pinned_node(h)
    nodes = [pin] if pin else ps.nodes()
    if not nodes:
        return None, None, 'no nodes: give --nodes or set driver.NODE_POOL'
    r = _runner()
    roots = [] if pin else r.chain_roots(cfg.driver.CHAIN, ps.step_forward)
    best, why = None, []
    for host in nodes:
        miss = r.missing_roots(roots, host)
        if miss:                                   # gate R-ROOTS: a node that cannot read a root the chain's steps read
            why.append(f'{host}: cannot read ' + ', '.join(f'{ref} = {p} ({w})' for ref, p, w in miss))
            continue
        state = ps.node_state(host)
        if state and not slurm.node_usable(state):
            why.append(f'{host}: Slurm reports it {state}')
            continue
        path = (cfg.site.SCRATCH_ROOT or {}).get(host)
        if not path:
            why.append(f'{host}: no site SCRATCH_ROOT')
            continue
        free = ps.free(host, path)
        if free is None:
            why.append(f'{host}: free space of {path} unreadable')
            continue
        key = f'{host}:{path}'
        room = free - booked(hours, key, h['epoch']) - floor
        if room < need:
            why.append(f'{host}:{path} {_gb(free)} free, {_gb(booked(hours, key, h["epoch"]))} booked, floor {_gb(floor)}, '
                       f'needs {_gb(need)}')
            continue
        if best is None or room > best[0]:
            best = (room, host, key)
    if best is None:
        return None, None, 'space: ' + '; '.join(why) + (f' (the hour keeps {pin}: its stages\' scratch is there)'
                                                         if pin else '')
    booking = {best[2]: need}
    for path, (b, fl) in (cfg.driver.SHARED_BOOK or {}).items():
        free = ps.free(None, path)
        key = f':{path}'
        if free is None or free - booked(hours, key, h['epoch']) - float(fl) < float(b):
            return None, None, (f'space: {path} {_gb(free or 0)} free, {_gb(booked(hours, key, h["epoch"]))} booked, '
                                f'floor {_gb(float(fl))}, needs {_gb(float(b))}')
        booking[key] = float(b)
    return best[1], booking, None


def _order_key(h):
    lst, _, date = h['epoch'].partition('_')
    return (date, lst)


def start_hour(ps, hours, h):
    """Book space, pick the node, create the hour's runner job (first start only) and submit its chain. False when the hour
    must wait (the reason is recorded)."""
    cfg = _cfg()
    r = _runner()
    part = ps.partition_of(h['tier'])
    if not part:
        h['waiting_reason'] = f'no Slurm partition for tier {h["tier"]} (driver.PARTITIONS; --partition)'
        return False
    node, booking, why = book(ps, hours, h)
    if node is None:
        if h.get('waiting_reason') != why:
            ps.out(f'{h["epoch"]} waits: {why}')
            ps.changed.add(h['epoch'])
        h['waiting_reason'] = why
        return False
    h.update(node=node, booking=booking, waiting_reason=None)
    ps.act(f'START {h["epoch"]} ({h["tier"]}) on {node}; booked ' + ', '.join(f'{k} {_gb(b)}' for k, b in booking.items()))
    forward = ps.step_forward + ['--resume']
    if h.get('job'):
        j = r.load_job(h['job'])
    elif ps.dry:
        steps = r.chain_steps(cfg.driver.CHAIN)
        j = {'chain_steps': steps, 'chain_finally': r.chain_finally(cfg.driver.CHAIN), 'pkg_root': r._pkg_root(),
             'python': sys.executable, 'site_file': os.environ.get('EXOPIPE3_SITE', ''), 'jobs_root': r.jobs_root(),
             'forward_argv': forward}
        ps.out(f'    runner job record (new): {r.jobs_root()}/<stamp>_drive_{h["epoch"]}, chain {cfg.driver.CHAIN} '
               f'({len(steps)} steps), forwarded {" ".join(forward)}')
        for k, a in enumerate(r.step_argvs(j, h['epoch'])):
            ps.out(f'      step {k + 1}: {" ".join(map(shlex.quote, a))}')
    else:
        h['job'] = r.submit(f'drive_{h["epoch"]}', [], [h['epoch']], [(node, 1)], chain=cfg.driver.CHAIN, start=False,
                            forward=forward, code=ps.code)
        h['state'] = 'active'
        _persist(ps, h)                            # the runner job is on disk with the hour: never a second one
        j = r.load_job(h['job'])
    h['state'] = 'active'
    hours[h['epoch']] = h
    ps.changed.add(h['epoch'])
    _persist(ps, h)
    if ps.dry and not h.get('job'):
        submit_chain(ps, h, j, 0, 'first submission')
    else:
        advance(ps, h)
    return True


def submit_waiting(ps, hours):
    """Catch-up windows: per tier (night first), the newest waiting hours up to driver.WINDOW_HOURS in flight."""
    cfg = _cfg()
    for tier in TIERS:
        cap = int((cfg.driver.WINDOW_HOURS or {}).get(tier, 0))
        inflight = sum(1 for h in hours.values() if h['tier'] == tier and h['state'] in IN_FLIGHT)
        waiting = sorted((h for h in hours.values() if h['tier'] == tier and h['state'] == 'waiting'), key=_order_key,
                         reverse=True)
        for i, h in enumerate(waiting):
            if inflight >= cap:
                why = f'window: {inflight} of {cap} {tier} hours in flight'
                for w in waiting[i:]:
                    if w.get('waiting_reason') != why:
                        w['waiting_reason'] = why
                        ps.changed.add(w['epoch'])
                break
            if start_hour(ps, hours, h):
                inflight += 1
            elif pinned_node(h):
                continue                           # it waits for its own node; the hours after it may go elsewhere
            else:
                for w in waiting[i + 1:]:
                    why = f'behind {h["epoch"]} ({h["waiting_reason"]})'
                    if w.get('waiting_reason') != why:
                        w['waiting_reason'] = why
                        ps.changed.add(w['epoch'])
                break


# ---------------------------------------------------------------------------------------------------------------------
# discovery, requests, accounting, periodic jobs
# ---------------------------------------------------------------------------------------------------------------------

def phase2_ready(lst, date):
    """(ready, phase-2 run dir or None, why): the hour's latest Phase 2 run holds driver.PHASE2_READY_FILE naming this hour."""
    from exopipe.phase3.core import inputs
    sci = inputs.latest_science_dir(lst, date)
    if not sci:
        return False, None, 'no Phase 2 run directory'
    p = os.path.join(sci, _cfg().driver.PHASE2_READY_FILE)
    if not os.path.isfile(p):
        return False, sci, f'{_cfg().driver.PHASE2_READY_FILE} absent (Phase 2 not ended in per-hour mode)'
    rec = _rjson(p, {}) or {}
    if (rec.get('lst') not in (None, lst)) or (rec.get('date') not in (None, date)):
        return False, sci, f'{p} names {rec.get("lst")} {rec.get("date")}, not this hour'
    return True, sci, None


def discover(ps, hours, known):
    cfg = _cfg()
    first = cfg.driver.FIRST_NIGHT
    if not first:
        return
    from exopipe.phase3.core import cells
    p2 = cfg.paths.PHASE2_ROOT
    now, poll, cap = time.time(), float(cfg.driver.POLL_S), float(cfg.driver.DISCOVER_BACKOFF_MAX_S)
    for lst in cells.LSTS:
        d = os.path.join(p2, lst)
        try:
            dates = sorted(x for x in os.listdir(d) if _DATE_RE.match(x) and x >= first)
        except OSError:
            continue
        for date in dates:
            epoch = f'{lst}_{date}'
            if epoch in known:
                continue
            nxt = _NOT_READY.get(epoch)
            if nxt and now < nxt[0]:
                continue                           # not ready at the last look: looked at again after its back-off delay
            ok, sci, _ = phase2_ready(lst, date)
            if not ok:
                delay = min(cap, 2 * nxt[1] if nxt else poll)
                _NOT_READY[epoch] = (now + delay, delay)
                continue
            _NOT_READY.pop(epoch, None)
            h = new_hour(epoch, 'night', 'discovered')
            h['phase2_run'] = sci
            hours[epoch] = h
            known.add(epoch)
            ps.changed.add(epoch)
            ps.act(f'DISCOVERED {epoch} (night): Phase 2 run {sci}')


def apply_requests(ps, hours, known):
    """Apply the request files of `drive add` / `drive retry`; returns their paths (removed by tick once the hours are
    saved, so a pass that fails before that applies them again)."""
    d = os.path.join(root(), 'requests')
    done = []
    if not os.path.isdir(d):
        return done
    for f in sorted(os.listdir(d)):
        if not f.endswith('.json'):
            continue
        p = os.path.join(d, f)
        q = _rjson(p)
        if not q:
            continue
        for epoch in q.get('epochs') or []:
            if q['action'] == 'add':
                add_hour(ps, hours, known, epoch, q.get('tier') or 'backlog', 'add')
            elif q['action'] == 'retry':
                retry_hour(ps, hours, epoch)
        done.append(p)
    return done


def add_hour(ps, hours, known, epoch, tier, source):
    if epoch in hours:
        ps.out(f'{epoch}: already known ({hours[epoch]["state"]}); not added')
        return
    if epoch in known:
        ps.out(f'{epoch}: already finished; use `drive retry` to run it again')
        return
    h = new_hour(epoch, tier, source)
    hours[epoch] = h
    known.add(epoch)
    ps.changed.add(epoch)
    ps.act(f'ADD {epoch} ({tier})')


def retry_hour(ps, hours, epoch):
    """A finished hour that ended refused or failed (or whose closing steps did not finish): back to waiting, its retries
    reset; its stage records are kept, so it resumes at the step that ended it."""
    if epoch in hours:
        ps.out(f'{epoch}: still {hours[epoch]["state"]}; retry applies to finished hours')
        return
    h = _rjson(hour_path(epoch, finished=True))
    if not h:
        ps.out(f'{epoch}: no finished hour of the driver; nothing to retry')
        return
    r = _runner()
    rec = r.item_state(h['job'], epoch) if h.get('job') else {}
    ks = resume_steps(r.load_job(h['job']), rec) if h.get('job') else []
    ps.act(f'RETRY {epoch}: ended {h.get("outcome")}; '
           + (f'steps {",".join(str(k + 1) for k in ks)} run again (the runner\'s retry rule)' if ks else
              'every step ended done: its closing steps run again'))
    if ps.dry:
        return
    if ks:
        requeue(h['job'], epoch, rec, ks)
        rec['status'] = 'queued'
        for k in ('chain_status', 'finally', 'failed_reason', 'summary', 'refused_cells', 'refused_by_cell'):
            rec.pop(k, None)
        _write_item(h['job'], epoch, rec)
    elif rec.get('finally') is not None:
        rec.pop('finally', None)
        _write_item(h['job'], epoch, rec)
    h.update(state='waiting', failures={}, fin_failures=0, outcome=None, booking={}, waiting_reason=None,
             why_next='retry requested (drive retry)')
    _note(h, 'retry requested')
    hours[epoch] = h
    _wjson(hour_path(epoch), h)
    os.unlink(hour_path(epoch, finished=True))


def account(ps, hours):
    """sacct for every submission that has ended (not live) and is not yet accounted: state, exit, elapsed time, peak
    memory, node - recorded in the hour (design requirement 8)."""
    subs = [(h, s) for h in hours.values() for s in h.get('submissions') or []
            if s.get('acct') is None and s['jobid'] not in ps.live]
    if not subs or ps.dry:
        return
    acct = slurm.accounting([s['jobid'] for _, s in subs])
    now = time.time()
    for h, s in subs:
        a = acct.get(s['jobid'])
        if a and a.get('state') and a['state'] not in ('PENDING', 'RUNNING', 'REQUEUED', 'COMPLETING', 'SUSPENDED'):
            s['acct'] = a
            ps.changed.add(h['epoch'])
        elif now - _ts(s['submitted']) > STALE_ACCT_S:
            s['acct'] = {'state': 'unknown (sacct lists no end a day after submission)'}
            ps.changed.add(h['epoch'])


def periodic(ps):
    """Submit each driver.PERIODIC job when due (its period since its last submission) and none of it is live; record the end
    of the last one (sacct) and print an ALARM line when it did not end COMPLETED with exit 0."""
    cfg = _cfg()
    r = _runner()
    path = os.path.join(root(), 'periodic.json')
    state = _rjson(path, {}) or {}
    changed = False
    now = time.time()
    for name, spec in sorted((cfg.driver.PERIODIC or {}).items()):
        if not spec.get('every_h'):
            continue                               # every_h 0: not scheduled
        rec = state.get(name) or {}
        live = [jid for jid, v in ps.live.items() if v['name'] == periodic_name(name)]
        if rec.get('jobid') and rec.get('acct') is None and rec['jobid'] not in ps.live and not ps.dry:
            a = slurm.accounting([rec['jobid']]).get(rec['jobid'])
            if a and a.get('state') not in (None, 'PENDING', 'RUNNING', 'REQUEUED', 'COMPLETING'):
                rec['acct'] = a
                changed = True
                if a['state'] != 'COMPLETED' or a.get('exit') != '0:0':
                    ps.out(f'ALARM periodic {name}: Slurm job {rec["jobid"]} ended {a["state"]} exit {a.get("exit")} '
                           f'(log {rec.get("log")})')
        if live:
            continue
        if rec.get('submitted') and now - _ts(rec['submitted']) < float(spec['every_h']) * 3600:
            continue
        part = ps.partition_of(spec.get('tier', 'backlog'))
        nodes = ps.nodes()
        if not part or not nodes:
            ps.out(f'periodic {name} due but not submitted: ' + ('no partition' if not part else 'no nodes'))
            continue
        log = os.path.join(root(), 'periodic', f'{name}.log')
        cmd = shell_command(r._pkg_root(), os.environ.get('EXOPIPE3_SITE', ''), [sys.executable] + list(spec['argv']))
        kw = dict(name=periodic_name(name), partition=part, time_limit=spec['time'], mem=spec['mem'], cpus=spec['cpus'],
                  output=log, chdir=r._pkg_root(), nodelist=nodes[0], dependency='singleton')
        ps.act(f'PERIODIC {name}: due (every {spec["every_h"]} h; last {rec.get("submitted") or "never"})')
        if ps.dry:
            ps.out('    ' + ' '.join(map(shlex.quote, slurm.sbatch_argv(cmd, **kw))))
            continue
        os.makedirs(os.path.dirname(log), exist_ok=True)
        try:
            jid = slurm.submit(cmd, **kw)
        except slurm.SlurmError as e:
            ps.out(f'  sbatch of periodic {name} refused: {e}')
            continue
        state[name] = {'submitted': _now(), 'jobid': jid, 'log': log, 'node': nodes[0], 'acct': None,
                       'previous': {k: rec.get(k) for k in ('submitted', 'jobid', 'acct')} if rec else None}
        changed = True
        ps.out(f'  {periodic_name(name)} -> Slurm job {jid}')
    if changed and not ps.dry:
        _wjson(path, state)


# ---------------------------------------------------------------------------------------------------------------------
# the pass
# ---------------------------------------------------------------------------------------------------------------------

def tick(ps, explicit=(), explicit_tier='backlog'):
    """One pass (module docstring). Returns the pass (its actions)."""
    hours = load_hours()
    known = set(hours) | finished_epochs()
    applied = apply_requests(ps, hours, known)
    for epoch in explicit or ():
        add_hour(ps, hours, known, epoch, explicit_tier, 'tick')
    discover(ps, hours, known)
    live = slurm.live(ps.user)
    if live is None:
        ps.out('squeue failed: no hour is advanced or submitted in this pass')
        _save(ps, hours, applied)
        return ps
    ps.live = live
    account(ps, hours)
    for h in [x for x in hours.values() if x['state'] == 'accounting']:
        settle(ps, h)
    flight = sorted((h for h in hours.values() if h['state'] in IN_FLIGHT), key=_order_key, reverse=True)
    for h in sorted(flight, key=lambda x: TIERS.index(x['tier'])):      # night first, newest night first
        advance(ps, h)
    submit_waiting(ps, hours)
    periodic(ps)
    _save(ps, hours, applied)
    if not ps.dry:
        _wjson(os.path.join(root(), 'last_pass.json'), {'time': _now(), 'host': _host(), 'pid': os.getpid(),
                                                          'actions': ps.actions[-200:]})
    return ps


def _save(ps, hours, applied=()):
    if ps.dry:
        return
    for epoch in sorted(ps.changed):
        h = hours.get(epoch)
        if h and h['state'] != 'finished':
            _wjson(hour_path(epoch), h)
    for p in applied:
        try:
            os.unlink(p)
        except FileNotFoundError:
            pass


def loop(ps_factory, interval=None):
    """`drive loop`: a pass every driver.POLL_S seconds until SIGTERM or SIGINT (the current pass finishes first). An
    exception in a pass is printed and the loop goes on (the next pass starts from the records)."""
    import traceback
    stop = {'now': False}

    def _stop(signum, frame):
        stop['now'] = True
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    interval = float(interval or _cfg().driver.POLL_S)
    while not stop['now']:
        t0 = time.time()
        try:
            tick(ps_factory())
        except (Exception, SystemExit):            # noqa: BLE001 - one bad pass never stops the driver; it is printed
            print(f'{_now()} pass failed:\n{traceback.format_exc()}', flush=True)
        while not stop['now'] and time.time() - t0 < interval:
            time.sleep(1)
    print(f'{_now()} driver loop stopped', flush=True)


# ---------------------------------------------------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------------------------------------------------

def status_lines(live=None, epochs=None, finished=False):
    r = _runner()
    hours = load_hours()
    if finished:
        d = os.path.join(root(), 'finished')
        for e in sorted(finished_epochs()):
            hours.setdefault(e, _rjson(os.path.join(d, f'{e}.json')))
    out = [f'driver root {root()}']
    lp = _rjson(os.path.join(root(), 'last_pass.json'))
    if lp:
        out.append(f'last pass {lp["time"]} on {lp["host"]} pid {lp["pid"]}')
    ps = Pass()
    ps.live = live or {}
    for e in sorted(hours, key=lambda x: _order_key(hours[x]), reverse=True):
        h = hours[e]
        if epochs and e not in epochs:
            continue
        line = f'{e:16s} {h["tier"]:8s} {h["state"]:9s} {h.get("node") or "-":11s}'
        if h.get('job'):
            try:
                j = r.load_job(h['job'])
                rec = r.item_state(h['job'], e)
                steps = rec.get('steps') or []
                n = len(j['chain_steps'])
                ndone = sum(1 for s in steps if s.get('status') in DONE_STEP)
                k = open_step(j, rec)
                line += f' steps {ndone}/{n} item {rec.get("status")}'
                if k is not None and k < len(steps):
                    line += f' at {k + 1} {steps[k].get("name")} ({steps[k].get("status")})'
            except SystemExit as ex:
                line += f' job record unreadable ({ex})'
        jobs = ps.hour_jobs(e)
        if jobs:
            line += ' live ' + ' '.join(f'{("s" + str(key + 1)) if key != "fin" else "fin"}:{jid}:{st}'
                                        + (f'({why})' if why not in ('None', '') else '')
                                        for key, js in sorted(jobs.items(), key=lambda kv: str(kv[0])) for jid, st, why in js)
        if h.get('failures'):
            line += ' failures ' + ','.join(f's{int(k) + 1}:{v}' for k, v in sorted(h['failures'].items()))
        if h.get('booking'):
            line += ' booked ' + ','.join(f'{k} {_gb(b)}' for k, b in h['booking'].items())
        if h.get('outcome'):
            line += f' outcome {h["outcome"]}'
        if h.get('waiting_reason') and h['state'] == 'waiting':
            line += f' waiting: {h["waiting_reason"]}'
        out.append(line)
    per = _rjson(os.path.join(root(), 'periodic.json'), {}) or {}
    for name, rec in sorted(per.items()):
        a = rec.get('acct') or {}
        out.append(f'periodic {name}: last submitted {rec.get("submitted")} job {rec.get("jobid")} '
                   + (f'ended {a.get("state")} exit {a.get("exit")}' if a else 'not ended'))
    return out
