"""exopipe3 run - the multi-node runner (replaces fleets, queues, waiters and dated drivers).

    exopipe3 run submit --name vsearch --nodes lwacalim09:8 --retry 1 -- search --tier deep --stokes V --cells @list
    exopipe3 run submit --after 20261013T010000Z_masters --nodes lwacalim09:4 -- frames --tier deep --cells @list
    exopipe3 run submit --chain deep-cell --items @cells.txt --nodes lwacalim09:2     # runner.CHAINS in config.py
    exopipe3 run submit --code 1fd3460 --set search.X=1 --nodes refactor -- search --cells @list --out /fast/gh/p
    exopipe3 run status [JOB]          # one job (items by status; chain items show the step reached) or the latest jobs
    exopipe3 run hold JOB | release JOB | drain JOB --host lwacalim09 | retry JOB [--which failed,refused]
    exopipe3 run reorder JOB --first 09h_Full,09h_Red
    exopipe3 run submit --no-start ...; exopipe3 run start JOB      # create the job record now, start it later

Inputs:  a step command line after '--', or --chain NAME (runner.CHAINS: step command templates run in order per
         item); items from --cells/--items (any item grammar: list, @file, JOB:STATUS, record:STATUS:STEP@OUT). Use
         {item} in the step command where the item goes; otherwise --cells <item> is appended.
         --set / --config / --ruling-for / --parity given to `run submit` BEFORE '--' are forwarded into every step
         command (and recorded in job.json); protected keys and unknown ledger IDs are refused at submit, ruled keys are
         checked by each step against its own --out.
         --code COMMIT: every step process runs from a read-only git worktree of that commit (one per commit, reused,
         under runner.CODE_WORKTREES_ROOT); the commit is recorded in job.json and in every item record.
Outputs: a job record under runner.JOBS_ROOT (job.json, items.txt, items/<item>.json, logs/<item>.log).
Failure modes: unknown job; unknown chain; --code not a commit of the repository (refused); a --gpu job on a non-GPU
host (refused); nodes unreachable (worker never starts - visible in supervisor.log and `run status`). An item's status
is its step's exit code (0 done, 2 refused, 3 no-input, other error; errors retried --retry times, then failed); a
chain item stops at its first non-zero exit with that step's status, and a retried chain item resumes at that step.
"""
import os

SUMMARY = 'multi-node runner'
TIERS = None
SELECTORS = ()
WRITES = False
CONFIG = ('runner',)      # config sections this step reads (plus config.SHARED_SECTIONS)


def add_arguments(p):
    p.add_argument('action', choices=['submit', 'start', 'status', 'hold', 'release', 'drain', 'retry', 'reorder',
                                      'list', 'supervise', 'worker'])
    p.add_argument('job', nargs='?', help='job id (start/status/hold/release/drain/retry/reorder)')
    p.add_argument('--name', help='job name (submit)')
    p.add_argument('--nodes', help='host:lanes[,host:lanes] or a site pool name (submit)')
    p.add_argument('--items', help='items (any item grammar); default: the --cells value inside the step command')
    p.add_argument('--chain', help='run each item through the steps of runner.CHAINS[NAME] in order (submit)')
    p.add_argument('--code', metavar='COMMIT', help='run every step process from a read-only worktree of COMMIT '
                                                   '(submit)')
    p.add_argument('--no-start', action='store_true', help='create the job record only; `run start JOB` starts it')
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


def forward_argv(args):
    """The arguments of `run submit` forwarded into every step command: --set, --config (absolute), --ruling-for,
    --parity."""
    fwd = []
    for s in args.set or ():
        fwd += ['--set', s]
    if args.config:
        fwd += ['--config', os.path.abspath(args.config)]
    for r in getattr(args, 'ruling_for', None) or ():
        fwd += ['--ruling-for', r]
    if getattr(args, 'parity', False):
        fwd += ['--parity']
    return fwd


def _status_line(j, it, s):
    line = (f"  {it:20s} {s.get('status'):9s} {s.get('host', ''):12s} attempts {s.get('attempts', 0)} "
            f"rc {s.get('rc', '')} {s.get('finished', s.get('started', ''))}")
    if j.get('chain_steps'):
        steps = s.get('steps') or []
        n_done = sum(1 for x in steps if x.get('status') == 'done')
        line += f"  steps {n_done}/{len(j['chain_steps'])}"
        last = steps[-1] if steps else None
        if last and last.get('status') != 'done':
            line += f" at step {last.get('step', 0) + 1} {last.get('name', '')} ({last.get('status')})"
    return line


def run(args, run):
    from exopipe.phase3.core import runner, cells
    a = args.action
    if a == 'submit':
        argv = list(args.step_argv)
        if not argv and not args.chain:
            raise SystemExit('submit needs the step command after --, or --chain NAME')
        spec = args.items
        if spec is None and '--cells' in argv:
            i = argv.index('--cells')
            spec = argv[i + 1]
            argv = argv[:i] + argv[i + 2:]
        if spec is None:
            raise SystemExit('give --items or --cells in the step command')
        its = cells.expand(spec)
        job = runner.submit(args.name, argv, its, runner.parse_nodes(args.nodes or ''), after=args.after,
                            retry=args.retry, gpu=args.gpu, start=not args.no_start, chain=args.chain,
                            forward=forward_argv(args), code=args.code)
        j = runner.load_job(job)
        print(f'job {job}: {len(its)} items' + (f'; chain {args.chain} ({len(j["chain_steps"])} steps)'
                                                 if args.chain else '')
              + (f'; code {j["code_commit"][:12]} at {j["code_root"]}' if j.get('code_commit') else '')
              + (f'; forwarded {" ".join(j["forward_argv"])}' if j['forward_argv'] else '')
              + f'; record {runner.job_dir(job)}' + ('; not started (run start JOB)' if args.no_start else ''))
        return 0
    if a == 'list' or (a == 'status' and not args.job):
        for jb in runner.list_jobs():
            st, by = runner.job_state(jb)
            print(f'{jb:50s} {st:9s} {by}')
        return 0
    if a == 'status':
        st, by = runner.job_state(args.job)
        j = runner.load_job(args.job)
        print(f'{args.job}: {st} {by}' + (f"  chain {j['chain']}" if j.get('chain') else '')
              + (f"  code {j['code_commit'][:12]}" if j.get('code_commit') else ''))
        for it in runner.items(args.job):
            print(_status_line(j, it, runner.item_state(args.job, it)))
        return 0
    if a == 'start':
        runner.start_supervisor(args.job)
        print(f'job {args.job}: supervisor started')
    elif a == 'hold':
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
