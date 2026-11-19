"""exopipe3 run - the multi-node runner (replaces fleets, queues, waiters and dated drivers).

    exopipe3 run submit --name vsearch --nodes lwacalim09:8 --retry 1 -- search --tier deep --stokes V --cells @list
    exopipe3 run submit --after 20261013T010000Z_masters --nodes lwacalim09:4 -- frames --tier deep --cells @list
    exopipe3 run submit --chain deep-cell --items @cells.txt --nodes lwacalim09:2     # runner.CHAINS in config.py
    exopipe3 run submit --code 1fd3460 --set search.X=1 --nodes refactor -- search --cells @list --out /fast/gh/p
    exopipe3 run status [JOB]          # one job (items by status; chain items show the step reached) or the latest jobs
    exopipe3 run hold JOB | release JOB | drain JOB --host lwacalim09 | retry JOB [--which failed,refused]
    exopipe3 run reorder JOB --first 09h_Full,09h_Red
    exopipe3 run submit --no-start ...; exopipe3 run start JOB      # create the job record now, start it later
    exopipe3 run update --trigger 13h_Green,13h_Full,15h_Green --hybrid-out H --v3-out M3 --v5-out M5 --plan DIR
                                       # a partial update as one plan (lib/update/plan.py; runner.UPDATE_ORDER): --plan
                                       # writes plan.json / plan.csv only; without it the jobs are submitted in order
    exopipe3 run update --from-exclusions --hybrid-out H --v3-out M3 --v5-out M5 --plan DIR
                                       # the trigger = every installed master holding a night excluded now (exit 3: none)

Inputs:  a step command line after '--', or --chain NAME (runner.CHAINS: step command templates run in order per
         item); items from --cells/--items (any item grammar: list, @file, JOB:STATUS, record:STATUS:STEP@OUT). Use
         {item} in the step command where the item goes; otherwise --cells <item> is appended.
         --set / --config / --ruling-for / --parity / --injected given to `run submit` BEFORE '--' are forwarded into
         every step command (and recorded in job.json); protected keys and unknown ledger IDs are refused at submit, ruled
         keys are checked by each step against its own --out. An injected run (core/injected.py) is submitted from an
         ordinary shell with `run submit --injected`: `run submit` refuses an EXOPIPE3_INJECTED it only inherits (the steps
         it launches on other nodes would not get it), and inside an injected run's processes it cannot run at all (its job
         records live under runner.JOBS_ROOT, which the injected-run guard refuses).
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
                                      'list', 'supervise', 'worker', 'update'])
    p.add_argument('job', nargs='?', help='job id (start/status/hold/release/drain/retry/reorder)')
    p.add_argument('--name', help='job name (submit)')
    p.add_argument('--nodes', help='host:lanes[,host:lanes] or a site pool name (submit)')
    p.add_argument('--items', help='items (any item grammar); default: the --cells value inside the step command')
    p.add_argument('--chain', help='run each item through the steps of runner.CHAINS[NAME] in order (submit)')
    p.add_argument('--code', metavar='COMMIT', help='run every step process from a read-only worktree of COMMIT '
                                                   '(submit; update, default the clean HEAD of the package tree)')
    p.add_argument('--no-start', action='store_true', help='create the job record only; `run start JOB` starts it')
    p.add_argument('--after', help='wait until this job is finished before starting (submit)')
    p.add_argument('--retry', type=int, default=0, help='retries of an item that ends in error (submit)')
    p.add_argument('--gpu', action='store_true', help='every node must be a GPU host of the site file (submit)')
    p.add_argument('--host', help='host to drain (drain)')
    p.add_argument('--which', default='failed', help='statuses to re-queue (retry), e.g. failed,refused')
    p.add_argument('--first', help='items to move to the front (reorder)')
    p.add_argument('--lane', type=int, default=0, help=argparse_suppress())
    u = p.add_argument_group('update (a partial update of the hybrid route; spec plan/specs/hybrid.md 3.7)')
    u.add_argument('--trigger', help='the changed master cells <LST>_<scope>: list or @file (update)')
    u.add_argument('--from-exclusions', action='store_true',
                   help='the trigger = every installed deep master holding a night data_decisions.csv excludes now (update)')
    u.add_argument('--round', metavar='YYYYMMDD', help='the quarantine round of every push of the plan (update; default today, UTC)')
    u.add_argument('--hybrid-out', help='the standing hybrid work run (update): --resume skips its unchanged units')
    u.add_argument('--v3-out', help='the standing mosaic --generation v3 work run (update)')
    u.add_argument('--v5-out', help='the standing mosaic --generation v5 work run (update)')
    u.add_argument('--plan', metavar='DIR', help='write plan.json and plan.csv to DIR and stop (update)')
    u.add_argument('--apply', action='store_true', help='submit install and the pushes too (update); without it the plan stops '
                                                        'before install; each push class must be in the cutover allowance')


def argparse_suppress():
    import argparse
    return argparse.SUPPRESS


def default_out(args):
    return '/tmp'


def forward_argv(args):
    """The arguments of `run submit` forwarded into every step command: --set, --config (absolute), --ruling-for,
    --parity, --injected (core/injected.py)."""
    fwd = []
    for s in args.set or ():
        fwd += ['--set', s]
    if args.config:
        fwd += ['--config', os.path.abspath(args.config)]
    for r in getattr(args, 'ruling_for', None) or ():
        fwd += ['--ruling-for', r]
    if getattr(args, 'parity', False):
        fwd += ['--parity']
    if getattr(args, 'injected', False):
        fwd += ['--injected']
    return fwd


def submit_injected_gate(args):
    """An injected run's jobs (core/injected.py): `run submit --injected` from an ordinary shell forwards the flag to every step
    command. Refused: submitting inside an injected run's processes (the write guard they inherit refuses the job records
    under runner.JOBS_ROOT), and an EXOPIPE3_INJECTED this process only inherits (the steps launched on other nodes over ssh
    would run as ordinary runs)."""
    from exopipe.phase3.core import guard, injected
    from exopipe.phase3.core.record import Refusal
    if injected.active() and guard.active():
        raise Refusal(f'run submit inside an injected run\'s processes ({injected.reason()}): the job records live under '
                      f'runner.JOBS_ROOT, outside the run\'s scratch roots - submit from an ordinary shell with `run submit '
                      f'--injected`')
    if injected.active() and not getattr(args, 'injected', False):
        raise Refusal(f'this environment carries EXOPIPE3_INJECTED ({injected.reason()}) but `run submit` has no --injected: '
                      f'the steps it launches on other nodes would run as ordinary runs - pass --injected (forwarded to every '
                      f'step command), or unset EXOPIPE3_INJECTED')


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


def update(args):
    """`run update`: the plan (lib/update/plan.py) written to --plan, or submitted as runner jobs in order."""
    import time
    from exopipe.phase3.core import runner
    from exopipe.phase3.lib.update import plan as UP
    from exopipe.phase3.core.record import NoInput, Refusal
    if bool(args.trigger) == bool(args.from_exclusions):
        raise SystemExit('run update needs exactly one of --trigger <LST>_<scope>[,...] (the changed master cells) and '
                         '--from-exclusions')
    outs = {k: getattr(args, k) for k in UP.OUTS}
    ev = fp = None
    try:
        if args.from_exclusions:
            UP.check_outs(outs)                         # before the master audit's minutes
            trig, ev, n, fp = UP.from_exclusions(workers=getattr(args, 'workers', None))
            if not trig:
                raise NoInput(f'none of the {n} installed deep masters holds a night data_decisions.csv excludes now (sha1 '
                              f'{fp["sha1"][:12]}, {fp["n_exclude_rows"]} exclude rows): nothing to update')
            print(f'--from-exclusions: {len(trig)} cell(s) of {n} masters hold an excluded night: ' + ', '.join(
                f'{c} ({", ".join(sorted({e["date"] for e in ev if e["cell"] == c}))})' for c in trig))
        elif args.trigger.startswith('@'):
            with open(args.trigger[1:]) as f:
                trig = f.read().replace(',', ' ').split()
        else:
            trig = [c.strip() for c in args.trigger.split(',') if c.strip()]
        pl = UP.build(trig, outs, apply=args.apply, rnd=args.round)
        if ev is not None:
            pl.update(trigger_from='exclusions', trigger_evidence=ev, exclusions=fp)
        rows = UP.predicted(pl)
    except (ValueError, OSError, Refusal) as e:
        raise SystemExit(f'refused: {e}')
    stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    if args.plan:
        p = UP.write(pl, rows, args.plan)
        print(f'update plan {p}: cells {", ".join(pl["cells"])}; '
              + '; '.join(f'{e["name"]} {len(e["items"])}' + (' (needs --apply)' if e.get('needs_apply') and not pl['apply'] else '')
                          for e in pl['entries'] if e['items'])
              + f'; predicted push files {len(rows)} ({sum(r["action"] == UP.PUBLISH for r in rows)} publish); '
              f'quarantine round {pl["round"]}')
        return 0
    if not args.nodes:
        raise SystemExit('refused: run update submits jobs; give --nodes (or --plan DIR to write the plan only)')
    code = args.code or _clean_head(runner)
    pl['code'] = code
    jobs = UP.submit(pl, runner.parse_nodes(args.nodes), f'update_{stamp}', code)     # every record first: a refusal starts none
    if not jobs:
        raise SystemExit('refused: the plan has no entry to submit')
    UP.write(pl, rows, os.path.join(runner.job_dir(jobs[0][1]), 'update_plan'))
    UP.start(jobs)
    for name, job in jobs:
        print(f'{name}: job {job}')
    print(f'code {code}; ' + ('every entry submitted' if pl['apply'] else
                              'stopped before the first entry that writes the installed tree or the archive (no --apply)'))
    return 0


def _clean_head(runner):
    """The commit of the package tree run update submits from, refused when the tree has uncommitted changes (the jobs run
    from a read-only worktree of that commit, so a later checkout or merge never changes their code mid-plan)."""
    repo = runner._pkg_root()
    st = runner._git(repo, 'status', '--porcelain', '--untracked-files=no', timeout=120)
    if st.returncode != 0 or st.stdout.strip():
        raise SystemExit(f'refused: the package tree {repo} has uncommitted changes (or git failed); commit them or give --code')
    return runner._git(repo, 'rev-parse', 'HEAD', timeout=60).stdout.strip()


def run(args, run):
    from exopipe.phase3.core import runner, cells
    a = args.action
    if a == 'submit':
        argv = list(args.step_argv)
        if not argv and not args.chain:
            raise SystemExit('submit needs the step command after --, or --chain NAME')
        submit_injected_gate(args)
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
    if a == 'update':
        return update(args)
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
