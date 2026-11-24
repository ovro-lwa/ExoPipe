"""Configuration of `exopipe3 drive` - the per-hour driver: Slurm jobs for each night-hour's chain, and the periodic jobs.

Design: exopipe_ops/refactor_20261012/plan/ORCHESTRATION_DESIGN_20261117.md sections 5.2-5.3 and plan step 3 (Gregg approved
2026-11-17 "Proceed asap!"; 2026-11-24: no Slurm reservations, campaigns are ordinary queued jobs). Per run: --set driver.KEY=VALUE
or --config FILE; `exopipe3 config --section driver` lists the keys. Host names live in the site file (NODE_POOLS), never here.
"""
from exopipe.phase3.config import _Section


class driver(_Section):
    """`exopipe3 drive`: one small process (`drive loop`) that each minute finds the hours to run, submits each hour's chain
    (runner.CHAINS[CHAIN]) as Slurm jobs linked by afterok, reads progress from the runner's chain record of the hour (the
    stage records), resubmits a stage that ended without done, no-input or refused, books disk space before a chain is
    submitted, and submits the periodic jobs."""
    NULLABLE_KEYS = ('ROOT', 'FIRST_NIGHT', 'NODE_POOL')
    # Driver state: hours/<epoch>.json (one per hour in flight or waiting; it holds the hour's Slurm submissions and its
    # space booking), finished/<epoch>.json, periodic.json, requests/ (drive add / retry), LOCK (one driver at a time).
    # None = <runner.JOBS_ROOT>/_driver, beside the runner's job records (the runner lists only jobs, never '_' entries).
    ROOT = None
    CHAIN = 'epoch'              # runner.CHAINS name of the per-hour chain (DESIGN 12.1); its closing steps: runner.CHAIN_FINALLY
    # Automatic discovery of new hours: an hour <LST>_<date> with date >= FIRST_NIGHT whose latest Phase 2 run directory
    # (core/inputs.latest_science_dir) holds PHASE2_READY_FILE. None = no discovery: hours enter only by `drive add`. Set at
    # the cutover (the first continuous night): every earlier hour is backlog, added by list.
    FIRST_NIGHT = None
    # Phase 2's last write of an hour run in per-hour mode (--defer_email: the hour's one email is sent at the end of Phase 3,
    # steps/notify.py). Mirror of exopipe/phase2/phase2_science_agg.py DEFERRED_EMAIL_FILE under the run's QA/.
    PHASE2_READY_FILE = 'QA/email_deferred.json'
    # Slurm partition per tier (design 5.2): 'night' = the real-time tier (hours found by discovery), 'backlog' = tier 0 (hours
    # added by list, and the periodic jobs). None = not configured (plan step 1 creates the partitions): the driver refuses to
    # submit that tier.
    PARTITIONS = {'night': None, 'backlog': None}
    # Nodes an hour's chain may run on: a site NODE_POOLS name or host[,host] (runner.parse_nodes). One node per hour, chosen
    # at submission as the node with the most booked-for room on its scratch disk; every stage of the hour runs there (the
    # runner's chain rule: one node runs every step of an item, which share node-local scratch). None = give --nodes.
    NODE_POOL = None
    # Catch-up windows (design 5.3): hours in flight at once per tier (submitted and not finished), newest night first. Hours
    # beyond the window wait in hours/ (MaxJobCount is 10,000 jobs; a chain is about 23 jobs).
    WINDOW_HOURS = {'night': 24, 'backlog': 4}
    # Resubmissions of one stage of an hour that ended without done, no-input or refused (error, killed, timeout, node
    # failure, out of memory), counted per stage; after them the hour ends 'failed' and its closing steps run.
    RETRY = 2
    # Space booking (design 5.3, requirement 9): before an hour's chain is submitted, SCRATCH_BOOK_BYTES are booked on the
    # chosen node's scratch disk (site SCRATCH_ROOT[host]), and refused unless the disk's real free space minus every other
    # booking minus SCRATCH_FLOOR_BYTES holds them. The bookings are the hours' records (released when an hour finishes).
    # SCRATCH_BOOK_BYTES is a provisional default: plan step 2 (one hour of Phase 2-3 run as Slurm jobs) measures the per-hour
    # scratch peak and sets it. An hour that has submitted jobs keeps its node on every later booking (its scratch is there).
    SCRATCH_BOOK_BYTES = 520e9   # B per hour: /fast peak of the 10-s/1-min search per hour of data (design section 2 table)
    SCRATCH_FLOOR_BYTES = 400e9  # B kept free on that disk (the 400 GB /fast floor of the speed-up runs, plan/STAGE_MAP_20261111.md)
    # Destination bookings on shared storage: {path: (B per hour, B floor)}, checked with statvfs on the driver's host.
    # Empty until the per-hour product volume is measured (design plan step 2: one hour of Phase 2-3 as Slurm jobs).
    SHARED_BOOK = {}
    # Slurm resources of a stage job by the chain step's name (its first word: search, frames, ...); 'default' for the rest.
    # Every job gives --time and --mem (DefaultTime unset, DefMemPerNode UNLIMITED). Not measured: plan step 2 reads each
    # stage's peak memory and time from sacct, which the driver records per submission.
    STEP_RESOURCES = {'default': {'cpus': 16, 'mem': '100G', 'time': '06:00:00'}}
    FINALLY_RESOURCES = {'cpus': 2, 'mem': '8G', 'time': '00:30:00'}   # the closing job (the hour's email)
    # sbatch --licenses of every stage job: 'lustre_heavy:1' once slurm.conf declares Licenses=lustre_heavy:6 (design 5.1,
    # plan step 1). '' = none (no licence is configured today; a request for an unknown licence is refused by sbatch).
    LICENSES = ''
    # Slurm job names: <prefix><epoch>.s<k> (stage k, 0-based), <prefix><epoch>.fin (closing job), <prefix>periodic.<name>.
    # The driver finds an hour's live jobs by these names (squeue), and links a stage to its parent with afterok plus
    # singleton (two jobs of one stage never run at once).
    JOB_PREFIX = 'exo3_'
    # How a chain step runs: <runner job's python> STEP_HEAD <step argv> (the runner's chain: python -m exopipe.phase3 <step>).
    STEP_HEAD = ('-m', 'exopipe.phase3')
    POLL_S = 60                  # s between two passes of `drive loop` (design 5.3: each minute)
    # Discovery back-off: an hour whose Phase 2 run is not ready is looked at again after a delay that starts at POLL_S and
    # doubles up to this cap (so a date that never becomes ready costs one Lustre look per cap, not one per pass); the cap is
    # also the longest wait between Phase 2's ready file and the hour's discovery.
    DISCOVER_BACKOFF_MAX_S = 900     # s
    # Periodic jobs: name -> argv after the interpreter, period (hours; 0 = not scheduled), tier, resources. Each is submitted
    # when its period has passed since its last submission and no job of it is live; one that ends other than COMPLETED with
    # exit 0 prints an ALARM line in the driver's log. The TESS orbit table refresh runs
    # daily (plan/CUTOVER_READINESS.md TESS item: complete when the table reaches now + 14 days). The readers keep refreshing
    # it themselves (pages_deep.TESS_REFRESH_IN_READER True) until the cutover starts `drive loop`; that switch is the cutover's.
    PERIODIC = {'tess-orbits': {'argv': ('-m', 'exopipe.phase3.lib.pages.tess_orbits'), 'every_h': 24, 'tier': 'backlog',
                                'cpus': 1, 'mem': '2G', 'time': '00:15:00'}}
    # Slurm commands (a command line each; the data-free tests point them at tests/fake_slurm.py)
    SBATCH = 'sbatch'
    SQUEUE = 'squeue'
    SACCT = 'sacct'
    SCANCEL = 'scancel'
    SINFO = 'sinfo'              # node states: a pending job's node that is down or drained is rebooked (core/driver.advance)
    # Each job's command runs as a job step under SRUN (with --ntasks=1 --cpus-per-task=<the job's cpus> --export=ALL): a
    # cancelled, preempted or timed-out job then sends SIGTERM to the step's task (the stage process, which passes it on to
    # the chain step once; core/driver._Term), before SIGKILL after KillWait. Without a job step the batch script's processes
    # got no SIGTERM (2026-11-24, Slurm 24.05.2: job 264102 SIGKILLed after 30 s; with srun, jobs 264227 and 264229 got
    # SIGTERM at scancel and at the time limit). '' = no step (the command runs in the batch script).
    SRUN = 'srun'
