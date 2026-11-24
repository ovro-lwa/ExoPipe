"""exopipe3 drive - the per-hour driver: each hour's chain as Slurm jobs over the stage records, and the periodic jobs.

    exopipe3 drive loop --nodes deep                     # the driver: a pass each driver.POLL_S s (one driver at a time)
    exopipe3 drive tick --nodes deep                     # one pass
    exopipe3 drive tick --dry-run --epochs 13h_2026-04-19 --nodes lwacalim07 --partition lowprio
                                                         # what one pass would submit for that hour: nothing submitted,
                                                         # cancelled or written
    exopipe3 drive add --epochs @backlog.txt             # backlog hours (tier 'backlog'), run newest night first
    exopipe3 drive retry --epochs 13h_2026-11-30         # a finished hour that ended refused or failed: run it again from
                                                         # the step that ended it
    exopipe3 drive status [--epochs E] [--all]           # hours waiting or in flight (--all: finished ones too), live jobs
    exopipe3 drive stage JOB ITEM K | finally JOB ITEM   # the body of a Slurm job (submitted by the driver, not by hand)

Inputs:  hours <LST>_<date> found by discovery (driver.FIRST_NIGHT; Phase 2's last write, driver.PHASE2_READY_FILE) or added
         with `drive add`; the runner's chain record of each hour (runner.JOBS_ROOT/<job>/items/<epoch>.json); squeue and sacct
         of the user's jobs. --set / --config / --ruling-for are forwarded into every step command (as `run submit` does), and
         every step command gets --resume; --parity and --injected are refused (such runs are `run submit` jobs).
Outputs: driver state under driver.ROOT (hours/, finished/, periodic.json, requests/, last_pass.json, LOCK); one runner
         job record per hour (`exopipe3 run status <job>` shows it); Slurm jobs <driver.JOB_PREFIX><epoch>.s<k> (stage k),
         .fin (the closing steps) and periodic.<name>.
Failure modes: a second driver (refused: LOCK); squeue failing (that pass does nothing); no partition for a tier, no node,
no space (the hour waits, with the reason in `drive status`); a stage ending error / killed / timeout / out of memory (the
chain is resubmitted from it, driver.RETRY times, then the hour ends 'failed' and its closing steps run); a refused stage
(the hour ends 'refused'). Design: core/driver.py.
"""
SUMMARY = 'per-hour driver: Slurm jobs per night-hour chain over the stage records; periodic jobs'
TIERS = None
SELECTORS = ()
WRITES = False
CONFIG = ('driver', 'runner')


def add_arguments(p):
    p.add_argument('action', choices=['loop', 'tick', 'add', 'retry', 'status', 'stage', 'finally'])
    p.add_argument('job', nargs='?', help='runner job id (stage, finally)')
    p.add_argument('item', nargs='?', help='the hour <LST>_<date> (stage, finally)')
    p.add_argument('step', nargs='?', type=int, help='chain step index, 0-based (stage)')
    p.add_argument('--epochs', help='hours <LST>_<date>: list or @file (add, retry, status; tick: add them as backlog first)')
    p.add_argument('--tier', default='backlog', choices=['backlog', 'night'], help='tier of hours added (add, tick --epochs)')
    p.add_argument('--nodes', help='nodes an hour may run on: a site pool name or host[,host] (default driver.NODE_POOL)')
    p.add_argument('--partition', help='Slurm partition of every job (default driver.PARTITIONS per tier)')
    p.add_argument('--code', metavar='COMMIT', help='every step of a newly started hour runs from a read-only worktree of '
                                                   'COMMIT (the runner\'s --code)')
    p.add_argument('--interval', type=float, help='s between passes (loop; default driver.POLL_S)')
    p.add_argument('--all', action='store_true', help='finished hours too (status)')


def default_out(args):
    return '/tmp'


def _epochs(spec):
    from exopipe.phase3.core import cells
    try:
        return cells.expand_epochs(spec)
    except ValueError as e:
        raise SystemExit(f'refused: {e}')


def _pass(args, dry):
    from exopipe.phase3.core import driver
    from exopipe.phase3.steps.run import forward_argv
    return driver.Pass(dry=dry, nodes=args.nodes, partition=args.partition, forward=forward_argv(args), code=args.code)


def run(args, run):
    from exopipe.phase3.core import driver, slurm
    a = args.action
    if getattr(args, 'injected', False) or getattr(args, 'parity', False):
        raise SystemExit('refused: the driver runs the production chain; an injected or parity run is a `run submit` job')
    if a == 'stage':
        if not (args.job and args.item and args.step is not None):
            raise SystemExit('drive stage needs JOB ITEM K')
        return driver.run_stage(args.job, args.item, args.step)
    if a == 'finally':
        if not (args.job and args.item):
            raise SystemExit('drive finally needs JOB ITEM')
        return driver.run_finally(args.job, args.item)
    if a == 'status':
        import getpass
        live = slurm.live(getpass.getuser())
        for line in driver.status_lines(live, _epochs(args.epochs) if args.epochs else None, finished=args.all):
            print(line)
        if live is None:
            print('(squeue failed: live jobs not shown)')
        return 0
    if a in ('add', 'retry'):
        if not args.epochs:
            raise SystemExit(f'drive {a} needs --epochs')
        eps = _epochs(args.epochs)
        if args.dry_run:
            print(f'DRY-RUN drive {a}: {", ".join(eps)}')
            return 0
        p = driver.request(a, eps, args.tier if a == 'add' else None)
        print(f'drive {a}: {len(eps)} hour(s) requested ({p}); the driver applies it at its next pass')
        return 0
    eps = _epochs(args.epochs) if args.epochs else ()
    if a == 'tick':
        if args.dry_run:
            driver.tick(_pass(args, True), eps, args.tier)
            return 0
        lock = driver.Lock()
        try:
            driver.tick(_pass(args, False), eps, args.tier)
        finally:
            lock.release()
        return 0
    if a == 'loop':
        if args.dry_run:
            raise SystemExit('drive loop has no dry run: use `drive tick --dry-run`')
        lock = driver.Lock()
        try:
            if eps:
                driver.request('add', eps, args.tier)
            driver.loop(lambda: _pass(args, False), args.interval)
        finally:
            lock.release()
        return 0
    raise SystemExit(f'unknown action {a}')
