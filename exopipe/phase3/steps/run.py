"""exopipe3 run - the multi-node runner (replaces fleets, queues, waiters and dated drivers).

    exopipe3 run submit --name vsearch --nodes lwacalim09:8 --retry 1 -- search --tier deep --stokes V --cells @list
    exopipe3 run submit --after 20261013T010000Z_masters --nodes lwacalim09:4 -- frames --tier deep --cells @list
    exopipe3 run status [JOB]          # one job (items by status) or the latest jobs
    exopipe3 run hold JOB | release JOB | drain JOB --host lwacalim09 | retry JOB [--which failed,refused]
    exopipe3 run reorder JOB --first 09h_Full,09h_Red

Inputs:  a step command line after '--'; items from --cells/--items (any item grammar: list, @file, JOB:STATUS,
         record:STATUS:STEP@OUT). Use {item} in the step command where the item goes; otherwise --cells <item> is
         appended.
Outputs: a job record under runner.JOBS_ROOT (job.json, items.txt, items/<item>.json, logs/<item>.log).
Failure modes: unknown job; a --gpu job on a non-GPU host (refused); nodes unreachable (worker never starts - visible
in supervisor.log and `run status`). An item's status is its step's exit code (0 done, 2 refused, 3 no-input, other
error; errors retried --retry times, then failed).
"""
import sys

SUMMARY = 'multi-node runner'
TIERS = None
SELECTORS = ()
WRITES = False


def add_arguments(p):
    p.add_argument('action', choices=['submit', 'status', 'hold', 'release', 'drain', 'retry', 'reorder', 'list',
                                      'supervise', 'worker'])
    p.add_argument('job', nargs='?', help='job id (status/hold/release/drain/retry/reorder)')
    p.add_argument('--name', help='job name (submit)')
    p.add_argument('--nodes', help='host:lanes[,host:lanes] or a site pool name (submit)')
    p.add_argument('--items', help='items (any item grammar); default: the --cells value inside the step command')
    p.add_argument('--after', help='wait until this job is finished before starting (submit)')
    p.add_argument('--retry', type=int, default=0, help='retries of an item that ends in error (submit)')
    p.add_argument('--gpu', action='store_true', help='every node must be a GPU host of the site file (submit)')
    p.add_argument('--host', help='host to drain (drain)')
    p.add_argument('--which', default='failed', help='statuses to re-queue (retry), e.g. failed,refused')
    p.add_argument('--first', help='items to move to the front (reorder)')
    p.add_argument('--lane', type=int, default=0, help=argparse_suppress())


def argparse_suppress():
    import argparse
    return argparse.SUPPRESS


def default_out(args):
    return '/tmp'


def run(args, run):
    from exopipe.phase3.core import runner, cells
    a = args.action
    if a == 'submit':
        argv = list(args.step_argv)
        if not argv:
            raise SystemExit('submit needs the step command after --')
        spec = args.items
        if spec is None and '--cells' in argv:
            i = argv.index('--cells')
            spec = argv[i + 1]
            argv = argv[:i] + argv[i + 2:]
        if spec is None:
            raise SystemExit('give --items or --cells in the step command')
        its = cells.expand(spec)
        job = runner.submit(args.name, argv, its, runner.parse_nodes(args.nodes or ''), after=args.after,
                            retry=args.retry, gpu=args.gpu)
        print(f'job {job}: {len(its)} items; record {runner.job_dir(job)}')
        return 0
    if a == 'list' or (a == 'status' and not args.job):
        for jb in runner.list_jobs():
            st, by = runner.job_state(jb)
            print(f'{jb:50s} {st:9s} {by}')
        return 0
    if a == 'status':
        st, by = runner.job_state(args.job)
        print(f'{args.job}: {st} {by}')
        for it in runner.items(args.job):
            s = runner.item_state(args.job, it)
            print(f"  {it:20s} {s.get('status'):9s} {s.get('host', ''):12s} attempts {s.get('attempts', 0)} "
                  f"rc {s.get('rc', '')} {s.get('finished', s.get('started', ''))}")
        return 0
    if a == 'hold':
        runner.hold(args.job, True)
    elif a == 'release':
        runner.hold(args.job, False)
    elif a == 'drain':
        runner.drain(args.job, args.host)
    elif a == 'retry':
        print(f'{runner.retry(args.job, tuple(args.which.split(",")))} items re-queued')
    elif a == 'reorder':
        print(f'{runner.reorder(args.job, args.first.split(","))} items moved to the front')
    elif a == 'supervise':
        runner.supervise(args.job)
    elif a == 'worker':
        runner.worker(args.job, args.lane)
    return 0
