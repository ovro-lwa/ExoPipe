#!/usr/bin/env python3
"""ONE SNAPSHOT CELL (one LST hour of one night), the fleet's products, PIPELINED.

A port of the cell body of exopipe_ops/snapshot_fleet_node_v10.sh (lines 71-433; "v10:N" below names its lines) onto the
scheduler of schedule.py. The fleet runs the stages of a cell one after another and holds a node for ~4.5 hours at a load of
5-8 on 26 free cores (exopipe_ops/refactor_20261012/plan/STAGE_MAP_20261111.md section 1). Here the same engine commands, with
the same arguments, start as soon as what they read exists:

    stage -> regrid -> Master and zero-visibility gates                     (before anything else, as the fleet)
    flow:<sb>            FLOW_PER_GPU processes per GPU (two give byte-identical frames and ~1.4x the throughput); a flow
                         takes whichever GPU has a free slot (the fields are identical on every GPU and node)
    fields:<sb>          the subband's flow fields into the flow tree            after flow:<sb>
    search:<sb>          10-s and 1-minute search, both Stokes                   after flow:<sb>
    curves:<sb>:I, :V    light curves (read the search's admission table)        after search:<sb>, fields:<sb>; V after I
    cube:<sb>:I, :V      flowed-frame cube, write then verify                    after flow:<sb>
    flowmovie:<sb>                                                               after flow:<sb>
    subbands             the fleet's completeness guard (v10:207-241)            after every search
    wideband:V, :I       Red / Green / Blue (read every member's admission)      after subbands
    widebands            the wideband completeness guard (v10:248-276)           after both
    framemovies          the 19 frame movies                                     after every flow
    ship                 search products, models, flow products, light curves, cubes + MANIFEST (v10:340-429)

PRIORITY (the order build_tasks() plans in): the flows, then the searches, the guard and the wideband search - the chain that
decides when the cell can end - then the frame movies, light curves, cubes and flow movies. The tasks of that chain HOLD what
they wait for (schedule.py), so the work of an earlier subband never keeps a GPU slot idle or a search waiting: in the first
ten-band run (2026-11-11) the flows were planned subband by subband and the last one ended at 38 minutes instead of ~20.

WHAT FAILS THE CELL is what fails it in the fleet: staging incomplete, a broken or missing Master, an incomplete flow, too few
frames, an unaccounted subband or wideband search (and here also a failed regrid, which the fleet does not notice). Then
nothing is shipped: the work directory is kept for inspection, the staging is removed (as the fleet does on every refusal),
and the flow fields the light curves needed during the run are moved aside (snapshot_flow/<cell>.failed_<time>).
Movies, light curves and cubes do not fail the cell (v10: FRAMEMOVIES-FAILED, LIGHTCURVE-FAILED, CUBE-INCOMPLETE are logged
and the cell is DONE); here each is recorded with its status in the cell record, never silently.
A RUN STARTS CLEAN: a ship directory of this cell under --out (an earlier run's products) refuses the run. An exception, a
SIGTERM or a Ctrl-C stops the engine processes (each runs in its own process group), and the record is written with the
status failed or interrupted - there is always a cell_<cell>.json.

WHERE IT WRITES: only under --out (D), --stage-root, --cutout-root and --cubes-root, none of which may be, be inside, or
contain a production tree (PROTECTED below; symbolic links resolved): D/stage/<cell> (staging), D/work/<cell> (the fleet's node-local output),
D/snapshot_fleet_out/<cell>, D/snapshot_flow/<cell>, D/snapshot_lightcurves/<cell>, D/snapshot_movies/<cell>,
D/snapshot_cubes/<cell>, D/photometry (cutout cubes, written by the search as it runs - as in the fleet), D/cell_<cell>.json (every task's outcome and times) and
D/cell_<cell>.log (the fleet's log lines). The record holds, per task, the wall time and the CPU time of its engine
processes (cpu_s; cpu_s / seconds = the cores it really used) and their peak memory (maxrss_gb): what the core and memory
charges below should follow. Production roots are not offered: the fleet remains production's driver until the
snapshot tier is cut over.

NOT PORTED: the injection battery hook (v10:291-338; a study, not part of the per-hour chain) and the 1-minute pause switch
(lifted 2026-10-08).

    python cell.py --cell 04h_2024-12-19 --out /fast/gh/x --phase1-root /lustre/pipeline/snapshots_gpu_pipeline
    python cell.py --cell ... --out ... --subbands "18MHz 23MHz 27MHz 32MHz 36MHz" --wideband yes --plan     # print the plan only

Standard library only; Python 3.6+. The engines run under their own interpreters (--py36, --flowpy).
"""
import argparse
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import schedule                                                                     # noqa: E402

ALL_SUBBANDS = ['18MHz', '23MHz', '27MHz', '32MHz', '36MHz', '41MHz', '46MHz', '50MHz', '55MHz', '59MHz',
                '64MHz', '69MHz', '73MHz', '78MHz', '82MHz']                        # v10:48
GROUPS = ('Red', 'Green', 'Blue')                                                   # v10:262
OPS = '/lustre/gh/main/exopipe_ops'
# production's code, state and product trees, and the fleet's node-local ones: never a place this driver writes to
PROTECTED = (OPS, '/lustre/gh/main/clv15', '/lustre/gh/main/exopipe', '/lustre/gh/main/phase3', '/lustre/pipeline',
             '/fast/gh/main/phase3', '/fast/pipeline')
ON_GRID_NAXIS = 1508                                                                # v10:131
# memory a task holds, GB, by the subband's grid (1508 / 2358 / 3122 px): the peak resident memory of the engines on
# 04h 2024-12-19, from the cell records (maxrss_gb, 2026-11-11), with a little room: search 5.6 / 12.7 / 21.1; cube writer
# 12.2 / 29.5 / 51.6 until 2026-11-12. Since the writer fills one cube frame by frame (phase3_snapshot_cube.write, memory
# only): 8.4 GB at 1508 px and 35.3 GB at 3122 px, MEASURED on 04h 2024-12-19 18 MHz and 82 MHz Stokes I (lwacalim03, the
# cube's own verify step under the py36 interpreter 6.1 and 25.2 GB), and in two full cells of 2026-11-12 (records
# cell_x1 / cell_x2, all 30 cube tasks each): 8.4-8.5 GB at 1508 px, 20.3-20.7 GB at 2358 px, 35.2-35.4 GB at 3122 px
MEM_SEARCH = {'low': 6, 'mid': 14, 'high': 23}
MEM_CUBE = {'low': 10, 'mid': 22, 'high': 38}
MEM_WIDEBAND = 28
# cores a task is CHARGED, as a share of its thread count: what the engines were measured to use on 04h 2024-12-19 (11-11,
# cpu_s / seconds of the cell records): a search with 8 threads 3.9-5.3 cores; the frame movies with 8 threads 3.0; light
# curves 2.7-6.9 with 8 threads (their default is 4, charged in full); the wideband search with 12 threads about 2 while
# it reads and its 12 only for the minutes it accumulates. A charge is an AVERAGE: for those minutes the threads of the
# running tasks can exceed the core list - products do not depend on it. Charged at the thread count, the two wideband searches and the frame movies took the whole budget of the
# first full cell one after another (20 minutes) and left the cube writers two cores.
CORE_SHARE = {'search': 0.65, 'wideband': 0.5, 'movies': 0.5, 'curves': 1.0}
CUTOUTS_GB, CUBES_GB = 83, 39          # what one full cell writes: photometry cutout cubes, flowed-frame cubes (04h 2024-12-19)


def charge(kind, threads):
    return max(1, int(round(threads[kind] * CORE_SHARE[kind])))


def band_class(sb):
    f = int(sb[:-3])
    return 'low' if f <= 36 else ('mid' if f <= 59 else 'high')


def build_tasks(subbands, wideband, frame_movies, cubes, threads):
    """The task graph of one cell: [(name, after, needs)] in PRIORITY order (the scheduler starts the first that fits). Pure:
    no I/O, so it is tested without data. First the chain that decides when the cell can end - flows, searches, the guard,
    the wideband search - then what only has to be finished by then. Flows go high band first: the last subbands to finish
    their flow are then the cheapest to search, which keeps the tail after the last flow short. A flow needs one slot of the
    pooled 'gpu' budget (which device is chosen when it starts)."""
    order = sorted(subbands, key=lambda s: -int(s[:-3]))
    tasks = [('flow:' + sb, (), {'gpu': 1, 'cores': 1}) for sb in order]
    tasks += [('fields:' + sb, ('flow:' + sb,), {'cores': 1}) for sb in order]
    tasks += [('search:' + sb, ('flow:' + sb,), {'cores': charge('search', threads), 'mem_gb': MEM_SEARCH[band_class(sb)]}) for sb in order]
    tasks.append(('subbands', tuple('search:' + sb for sb in order), {'cores': 1}))
    if wideband:
        for st in ('V', 'I'):                                                       # v10:242: V then I
            tasks.append(('wideband:' + st, ('subbands',), {'cores': charge('wideband', threads), 'mem_gb': MEM_WIDEBAND}))
        tasks.append(('widebands', ('wideband:V', 'wideband:I'), {'cores': 1}))
    if frame_movies:
        tasks.append(('framemovies', tuple('flow:' + sb for sb in order), {'cores': charge('movies', threads)}))
    for sb in order:
        tasks.append(('curves:%s:I' % sb, ('search:' + sb, 'fields:' + sb), {'cores': charge('curves', threads)}))
        tasks.append(('curves:%s:V' % sb, ('curves:%s:I' % sb,), {'cores': charge('curves', threads)}))
    if cubes:
        for sb in order:
            for st in ('I', 'V'):
                tasks.append(('cube:%s:%s' % (sb, st), ('flow:' + sb,), {'cores': 1, 'mem_gb': MEM_CUBE[band_class(sb)]}))
    tasks += [('flowmovie:' + sb, ('flow:' + sb,), {'cores': 1}) for sb in order]
    return tasks


FATAL_PREFIXES = ('flow:', 'search:', 'subbands', 'wideband')        # these fail the cell, as in the fleet; the rest are recorded
HOLD_PREFIXES = FATAL_PREFIXES                                       # ... and they are the chain that decides when the cell ends:
#                                                                      while one waits, nothing planned after it takes what it lacks


class Cell(object):
    def __init__(self, a):
        self.a = a
        self.cell = a.cell
        self.lst, self.date = a.cell.split('_', 1)
        self.subbands = a.subbands.split() if a.subbands else list(ALL_SUBBANDS)
        bad = [s for s in self.subbands if s not in ALL_SUBBANDS]
        if bad:
            raise SystemExit('REFUSED: unknown subband(s) %s' % bad)
        self.full = set(self.subbands) == set(ALL_SUBBANDS)
        yes = {'yes': True, 'no': False, 'auto': None}
        # a subset skips the wideband search and the frame movies unless asked for (v10: P3_FLEET_SUBSET)
        self.wideband = self.full if yes[a.wideband] is None else yes[a.wideband]
        self.frame_movies = self.full if yes[a.frame_movies] is None else yes[a.frame_movies]
        D = os.path.abspath(a.out)
        _refuse_protected(D, '--out')
        if a.stage_root:
            _refuse_protected(a.stage_root, '--stage-root')
        self.gpus = a.gpus.split()
        self.threads = dict(search=a.search_threads, curves=a.curves_threads, wideband=a.wideband_threads, movies=a.movie_threads)
        if not self.gpus or a.flow_per_gpu < 1 or min(self.threads.values()) < 1 or a.movie_procs < 1:
            raise SystemExit('REFUSED: no GPU, or a flow, thread or process count below 1 (--gpus %r --flow-per-gpu %d, threads %s)'
                             % (a.gpus, a.flow_per_gpu, self.threads))
        self.D = D
        self.stage = os.path.join(os.path.abspath(a.stage_root) if a.stage_root else os.path.join(D, 'stage'), self.cell)
        self.out = os.path.join(D, 'work', self.cell)                 # the fleet's $OUT
        self.flowdir = os.path.join(self.out, 'flow')                 # v10:143
        self.ship = os.path.join(D, 'snapshot_fleet_out', self.cell)
        self.flowtree = os.path.join(D, 'snapshot_flow')
        self.lcout = os.path.join(D, 'work', 'lightcurves', self.cell)
        self.lcship = os.path.join(D, 'snapshot_lightcurves', self.cell)
        self.movies = os.path.join(D, 'snapshot_movies', self.cell)
        self.movies_work = os.path.join(D, 'work', 'movies', self.cell)
        self.cubework = os.path.join(D, 'work', 'cubes', self.cell)
        # the two bulk products (83 GB of cutout cubes and 39 GB of flowed-frame cubes per cell) may go to another filesystem,
        # as production's do (Lustre); the first full cell kept them beside its staging and ran the node's disk to its floor
        for opt, root in (('--cutout-root', a.cutout_root), ('--cubes-root', a.cubes_root)):
            if root:
                _refuse_protected(root, opt)
        self.cubes = os.path.join(os.path.abspath(a.cubes_root) if a.cubes_root else os.path.join(D, 'snapshot_cubes'), self.cell)
        self.cutouts = os.path.abspath(a.cutout_root) if a.cutout_root else os.path.join(D, 'photometry')
        self.logpath = os.path.join(D, 'cell_%s.log' % self.cell)
        self.record = OrderedDict(cell=self.cell, subbands=self.subbands, started=None, status='running', steps=OrderedDict())
        self._lock = threading.Lock()
        self._gpu_load = OrderedDict((g, 0) for g in self.gpus)        # flows running per device
        self._cpu = {}                                                 # scheduler thread name -> CPU seconds of its engine processes
        self._rss = {}                                                 # ... -> peak resident memory of one of them, GB
        self._children = set()                                         # engine processes running now (each its own process group)
        self.active = list(self.subbands)                              # the subbands with staged frames (set in prepare)
        self._own_stage = False                                        # True once THIS run has begun to stage (then it removes it)
        self.cores = None
        self.cores_budget = self.mem_budget_gb = None                  # set in prepare(), from the node unless given
        self.zero_vis = set()

    # ---------------------------------------------------------------- small helpers
    def log(self, msg):
        line = '%s %s' % (time.strftime('%a %b %e %H:%M:%S UTC %Y', time.gmtime()), msg)
        with self._lock:
            with open(self.logpath, 'a') as f:
                f.write(line + '\n')
        print(line, flush=True)

    def env(self, **more):
        e = dict(os.environ)
        for k in list(e):
            if k.startswith(('P3_', 'SNAP_')):          # nothing inherited decides a product (as core/legacy.run)
                del e[k]
        # P3_SNAP_FLOW=1 is NOT set here: the fleet sets it for the two search commands only (v10:65, :243). With it the
        # engine's frame discovery looks for flowed copies and refuses, which is right for a search and wrong for staging
        e.update(NUMPY_MADVISE_HUGEPAGE='0', PYTHONUNBUFFERED='1',
                 P3_SNAP_CUTOUT_ROOT=self.cutouts, P3_SNAP_FLOWTREE=self.flowtree,
                 P3_SNAP_SAMPLES_DIR=self.a.samples_dir, P3_SNAP_LEGACY_ROOT=self.a.legacy_root,
                 P3_SNAP_DETECT_THREADS=str(self.a.search_threads), P3_SNAP_GROUP_THREADS=str(self.a.wideband_threads),
                 P3_SNAP_LC_THREADS=str(self.a.curves_threads), P3_SNAP_MOVIE_THREADS=str(self.a.movie_threads),
                 P3_SNAP_MOVIE_PROCS=str(self.a.movie_procs))
        if self.a.phase1_root:
            e['P3_SNAP_PHASE1_ROOT'] = self.a.phase1_root
        if self.a.guard:                                 # the refactor's write guard: a write outside --out is refused in the child
            e['EXOPIPE3_PARITY_ROOT'] = os.pathsep.join(sorted({self.D, os.path.dirname(self.stage), self.cutouts}))
            e['PYTHONPATH'] = self.a.guard + (os.pathsep + e['PYTHONPATH'] if e.get('PYTHONPATH') else '')
        e.update({k: str(v) for k, v in more.items()})
        return e

    def pinned(self, argv):
        return (['taskset', '-c', self.cores] if self.cores else []) + list(argv)

    def call(self, argv, logfile, env=None, append=False, pin=True):
        """Run one engine command; its stdout and stderr go to logfile. Returns the exit code. The CPU time of the process
        (and of the children it waited for) is added to the running task's account."""
        os.makedirs(os.path.dirname(logfile), exist_ok=True)
        with open(logfile, 'ab' if append else 'wb') as f:
            p = subprocess.Popen(self.pinned(argv) if pin else list(argv), stdout=f, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, env=env or self.env(), start_new_session=True)
            with self._lock:
                self._children.add(p.pid)
            try:
                _, status, ru = os.wait4(p.pid, 0)
            finally:
                with self._lock:
                    self._children.discard(p.pid)
        p.returncode = os.WEXITSTATUS(status) if os.WIFEXITED(status) else -os.WTERMSIG(status)
        who = threading.current_thread().name
        with self._lock:
            self._cpu[who] = self._cpu.get(who, 0.0) + ru.ru_utime + ru.ru_stime
            self._rss[who] = max(self._rss.get(who, 0.0), ru.ru_maxrss / 2.0 ** 20)      # GB (Linux reports kB)
        return p.returncode

    def stop_children(self, grace=30):
        """Terminate every running engine process group (SIGTERM, then SIGKILL after `grace` seconds)."""
        import signal
        for sig, wait in ((signal.SIGTERM, grace), (signal.SIGKILL, 10)):
            with self._lock:
                pids = list(self._children)
            if not pids:
                return
            for pid in pids:
                try:
                    os.killpg(pid, sig)
                except OSError:
                    pass
            t_end = time.time() + wait
            while time.time() < t_end:
                with self._lock:
                    if not self._children:
                        return
                time.sleep(0.2)

    def eng(self, name):
        return os.path.join(self.a.engine, name)

    def master(self, sb):
        return os.path.join(self.a.masters, '%s_%s' % (self.lst, sb), 'coadd_final', 'I_deep_Taper_R0_%s.fits' % sb)

    def staged(self, sb):
        return sorted(p for p in glob.glob(os.path.join(self.stage, sb, '*-image.fits')))

    def py36(self, code, *argv):
        return subprocess.run([self.a.py36, '-c', code] + list(argv), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              universal_newlines=True, env=self.env())

    # ---------------------------------------------------------------- before the scheduler: v10:93-203
    def prepare(self):
        a = self.a
        for d in (self.stage, self.out, self.flowdir, os.path.dirname(self.logpath)):
            os.makedirs(d, exist_ok=True)
        if os.listdir(self.out) != ['flow'] or os.listdir(self.flowdir):
            raise SystemExit('REFUSED: the work directory %s is not empty (an earlier run) - give a new --out' % self.out)
        # A RUN STARTS CLEAN: products of an earlier run of this cell would be mixed in silently (every ship copy is cp -n)
        old = [d for d in (self.ship, self.lcship, self.movies, self.cubes, os.path.join(self.flowtree, self.cell)) if os.path.exists(d)]
        if old:
            raise SystemExit('REFUSED %s products of an earlier run of this cell: %s - give a new --out%s, or remove them'
                             % (self.cell, ' '.join(old), ' (and a new --cubes-root: one of them is there)' if self.a.cubes_root and self.cubes in old else ''))
        for d in (self.lcout, self.cubework, self.movies_work):        # this driver's own work copies (v10:367, :396)
            shutil.rmtree(d, ignore_errors=True)
        if a.cores:
            self.cores = a.cores
        else:
            r = subprocess.run(['bash', os.path.join(OPS, 'node_free_cores.sh')], stdout=subprocess.PIPE, universal_newlines=True)
            self.cores = r.stdout.strip()                 # leave the data-recorder cores alone (v10:47)
            if r.returncode != 0 or not self.cores:       # never unpinned: the engines would land on the recorders' cores
                raise SystemExit('REFUSED %s no core list from node_free_cores.sh (exit %d) - give --cores' % (self.cell, r.returncode))
        self.cores_budget = a.cores_budget or _count_cores(self.cores)
        self.mem_budget_gb = a.mem_budget_gb or int(0.8 * _mem_available_gb())
        need = max(MEM_CUBE[band_class(sb)] if a.cubes else MEM_SEARCH[band_class(sb)] for sb in self.subbands)
        need = max(need, MEM_WIDEBAND if self.wideband else 0)
        most = max(charge(k, self.threads) for k in CORE_SHARE)
        self.log('CLAIM %s (cell.py: subbands %d, gpus [%s] x %d flows, cores %s; budgets: %d cores, %d GB)'
                 % (self.cell, len(self.subbands), ' '.join(self.gpus), a.flow_per_gpu, self.cores, self.cores_budget, self.mem_budget_gb))
        if self.mem_budget_gb < need or self.cores_budget < most:
            raise SystemExit('REFUSED %s the budgets (%d cores, %d GB) are below the largest single task (%d threads, %d GB)'
                             % (self.cell, self.cores_budget, self.mem_budget_gb, most, need))
        # WHO ELSE IS ON THE NODE: recorded, so that a run's times can be read (the nightly GPU imaging uses the GPUs of several
        # nodes from 13:41 UTC for about four hours; the first ten-band run of this driver overlapped it)
        try:
            r = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader'], stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL, universal_newlines=True, timeout=30)
            others = len([l for l in r.stdout.splitlines() if l.strip()]) if r.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            others = None
        self.record['node'] = OrderedDict(name=os.uname()[1], load_at_start=round(os.getloadavg()[0], 1), gpu_processes_at_start=others,
                                          cores_budget=self.cores_budget, mem_budget_gb=self.mem_budget_gb)
        if others:
            self.log('GPU-SHARED %s %d other process(es) on the GPUs at the start - this run\'s flow times are not a benchmark' % (self.cell, others))
        # DISK GUARD (v10:93-99)
        for what, root, need in (('--cutout-root', a.cutout_root, CUTOUTS_GB), ('--cubes-root', a.cubes_root, CUBES_GB)):
            if root:                                       # the bulk products of one cell must fit where they are sent
                os.makedirs(root, exist_ok=True)
                have = shutil.disk_usage(root).free // 2 ** 30
                if have < 2 * need:
                    raise SystemExit('DISK-REFUSED %s only %d GB free under %s %s (a cell writes ~%d GB there; twice that is asked)'
                                     % (self.cell, have, what, root, need))
        free = shutil.disk_usage(os.path.dirname(self.stage)).free // 2 ** 30
        if free < a.min_free_gb:
            raise SystemExit('DISK-REFUSED %s only %d GB free under %s (need %d)' % (self.cell, free, self.stage, a.min_free_gb))
        # STAGE (v10:100-124): one file per (Stokes, integration), chosen by the engine's own discovery; copy, funpack, count
        t0 = time.time()
        self._own_stage = True
        code = ('import sys; sys.path.insert(0, %r); import phase3_snapshot_search as s\n'
                'fr = s.discover_frames(sys.argv[1], sys.argv[2], sys.argv[3])\n'
                'for st in ("I", "V"):\n'
                '    for ts, p in fr[st]: print(p)\n') % a.engine
        from concurrent.futures import ThreadPoolExecutor
        nlist = 0
        for sb in self.subbands:
            d = os.path.join(self.stage, sb)
            os.makedirs(d, exist_ok=True)
            r = self.py36(code, self.lst, self.date, sb)
            if r.returncode != 0:
                raise SystemExit('STAGE-REFUSED %s %s: frame discovery failed: %s' % (self.cell, sb, r.stderr.strip().splitlines()[-1][:300] if r.stderr.strip() else 'exit %d' % r.returncode))
            paths = [p for p in r.stdout.splitlines() if p.startswith('/')]
            with open(os.path.join(d, '.filelist'), 'w') as f:
                f.write(''.join(p + '\n' for p in paths))
            nlist += len(paths)

            def copy(p, d=d):
                dst = os.path.join(d, os.path.basename(p))
                if not os.path.exists(dst):                               # cp -n
                    shutil.copyfile(p, dst + '.part')
                    if os.path.getsize(dst + '.part') != os.path.getsize(p):   # Lustre has made silent short copies (2026-11-09)
                        raise RuntimeError('short copy of %s' % p)
                    os.replace(dst + '.part', dst)
            with ThreadPoolExecutor(max_workers=a.copy_procs) as ex:
                list(ex.map(copy, paths))

            def unpack(f):
                o = f[:f.rindex('.')]
                with open(o + '.tmp', 'wb') as out:
                    rc = subprocess.call([a.funpack, '-S', f], stdout=out, stderr=subprocess.DEVNULL)
                if rc == 0:
                    os.replace(o + '.tmp', o)
                    os.remove(f)
            packed = glob.glob(os.path.join(d, '*.fs')) + glob.glob(os.path.join(d, '*.fz'))
            with ThreadPoolExecutor(max_workers=a.funpack_procs) as ex:
                list(ex.map(unpack, packed))
        nst = sum(len(self.staged(sb)) for sb in self.subbands)
        nleft = sum(len(glob.glob(os.path.join(self.stage, sb, pat))) for sb in self.subbands for pat in ('*.fs', '*.fz', '*.tmp', '*.part'))
        if nleft or nst != nlist:
            raise SystemExit('STAGE-REFUSED %s decompressed %d of %d listed frames, %d compressed/tmp leftovers' % (self.cell, nst, nlist, nleft))
        self.log('STAGED %s frames=%d (%.0f s)' % (self.cell, nst, time.time() - t0))
        self.record['steps']['stage'] = dict(status='done', seconds=round(time.time() - t0, 1), frames=nst)
        # REGRID (v10:125-141): pilot-era frames onto the subband's OWN Master grid
        t0, nreg = time.time(), 0
        for sb in self.subbands:
            fl = self.staged(sb)
            if not fl:
                continue
            r = self.py36('import sys\nfrom astropy.io import fits\nprint(fits.getheader(sys.argv[1])["NAXIS1"])', fl[0])
            if r.stdout.strip() not in ('', str(ON_GRID_NAXIS)):
                if not os.path.exists(self.master(sb)):
                    self.log('REGRID-SKIPPED %s %s no Master of its own - frames left on their grid' % (self.cell, sb))
                    continue
                rc = self.call([a.py36, self.eng('phase3_snapshot_regrid.py'), os.path.join(self.stage, sb), self.master(sb), '--nproc', '8'],
                               os.path.join(self.out, 'regrid_%s.log' % sb))
                if rc != 0:                              # the fleet goes on with frames on the wrong grid; here the cell is refused
                    raise SystemExit('REFUSED %s REGRID-FAILED %s (exit %d, see %s)' % (self.cell, sb, rc, os.path.join(self.out, 'regrid_%s.log' % sb)))
                nreg += 1
        if nreg:
            self.log('REGRIDDED %s %d subbands to the Master grid' % (self.cell, nreg))
        self.record['steps']['regrid'] = dict(status='done', seconds=round(time.time() - t0, 1), regridded=nreg)
        # MASTERS (v10:147-182): a broken link is a fault; no Master = the cell is not searched
        broken = [sb for sb in self.subbands if os.path.islink(self.master(sb)) and not os.path.exists(self.master(sb))]
        if broken:
            raise SystemExit('REFUSED %s broken Master link(s): %s' % (self.cell, ' '.join(broken)))
        nomaster = [sb for sb in self.subbands if not os.path.exists(self.master(sb))]
        if nomaster:
            raise SystemExit('REFUSED %s no Taper R0 Master: %s (the hourly-screen fallback is retired)' % (self.cell, ' '.join(nomaster)))
        # ZERO-VISIBILITY FRAMES (v10:157-176): refused, not missing - left out of the expected and the flowed counts
        r = self.py36('import glob, sys\nfrom astropy.io import fits\n'
                      'for p in glob.glob(sys.argv[1] + "/*/*-image.fits"):\n'
                      '    try:\n        v = fits.getheader(p).get("WSCNVIS", 1)\n    except Exception:\n        continue\n'
                      '    if v is not None and float(v) == 0:\n        print(p)\n', self.stage)
        if r.returncode != 0:
            raise SystemExit('REFUSED %s the zero-visibility probe failed: %s' % (self.cell, (r.stderr.strip().splitlines() or ['exit %d' % r.returncode])[-1][:300]))
        self.zero_vis = set(p for p in r.stdout.splitlines() if p.startswith('/'))
        if self.zero_vis:
            self.log('ZERO-VIS %s %d staged frames with WSCNVIS = 0 refused (not expected to flow)' % (self.cell, len(self.zero_vis)))
        if nst < a.min_frames:                             # v10:200-203 (tested before the flow here: nothing to gain by flowing first)
            raise SystemExit('REFUSED %s only %d decompressed frames (need %d) - not searching' % (self.cell, nst, a.min_frames))
        # a subband without a single frame this hour is not staged: nothing is planned for it and nothing is expected of it
        # (the fleet's guards and its cube count read the staged subbands only: v10:207-241, :400-402)
        self.active = [sb for sb in self.subbands if self.staged(sb)]
        if len(self.active) < len(self.subbands):
            self.log('NO-FRAMES %s %s: not staged, not planned' % (self.cell, ' '.join(sb for sb in self.subbands if sb not in self.active)))

    # ---------------------------------------------------------------- the tasks
    def t_flow(self, sb):
        with self._lock:                                               # the device with the fewest flows running (first on a tie)
            gpu = min(self._gpu_load, key=lambda g: self._gpu_load[g])
            self._gpu_load[gpu] += 1
        try:
            e = self.env(CUDA_VISIBLE_DEVICES=gpu, XLA_PYTHON_CLIENT_PREALLOCATE='false', XLA_PYTHON_CLIENT_MEM_FRACTION=self.a.xla_mem_fraction)
            rc = self.call([self.a.flowpy, self.eng('phase3_snapshot_flow.py'), '--stage', os.path.join(self.stage, sb),
                            '--reference', self.master(sb), '--subband', sb, '--out', self.flowdir],
                           os.path.join(self.flowdir, 'flow_%s.log' % sb), env=e)
        finally:
            with self._lock:
                self._gpu_load[gpu] -= 1
        # FLOW COMPLETE for this subband (v10:192-199, per subband): every staged frame but the zero-visibility ones is flowed
        want = [p for p in self.staged(sb) if not p.endswith('_flowed.fits') and p not in self.zero_vis]
        missing = [p for p in want if not os.path.exists(p[:-5] + '_flowed.fits')]
        if rc != 0 or missing:
            raise RuntimeError('FLOW-INCOMPLETE %s %s (GPU %s): exit %d, %d of %d frames without a flowed copy' % (self.cell, sb, gpu, rc, len(missing), len(want)))
        # DISK: on XFS the flow's output files keep their speculative preallocation long after they are closed - measured
        # 2026-11-11 on lwacalim00: 104 MB on disk for a 39 MB flowed frame (2.7x), 232 GB of slack for five 3122-px subbands,
        # which took /fast below its floor. Setting each file to its own length releases it; content and times are unchanged
        freed = sum(release_preallocation(p[:-5] + '_flowed.fits') for p in want)
        if freed > 2 ** 30:
            self.log('TRIMMED %s %s %.0f GB of preallocation released from %d flowed frames' % (self.cell, sb, freed / 2 ** 30, len(want)))

    def t_fields(self, sb):
        d = os.path.join(self.flowtree, self.cell)                    # v10:346-348: cp -n flow_*.npy to the flow tree
        os.makedirs(d, exist_ok=True)
        for p in glob.glob(os.path.join(self.flowdir, 'flow_%s_*.npy' % sb)):
            dst = os.path.join(d, os.path.basename(p))
            if not os.path.exists(dst):
                shutil.copyfile(p, dst + '.part')
                os.replace(dst + '.part', dst)

    def t_search(self, sb):
        # the engine's exit code is not the verdict (excluded = 3, a night-rule refusal = 1, both accounted for): the outcome is
        # judged by the completeness guard from the records the engine wrote, as in the fleet (v10:207-241)
        self.call([self.a.py36, '-u', self.eng('phase3_snapshot_search.py'), '--lst', self.lst, '--date', self.date, '--subband', sb,
                   '--stokes', 'VI', '--mode', 'median', '--thresh', '5.0', '--out', self.out,
                   '--stage-root', os.path.join(self.stage, sb), '--nthreads', str(min(8, self.a.search_threads))],   # the fleet's 8, never above the thread count
                  os.path.join(self.out, 'search_%s.log' % sb),
                  env=self.env(P3_SNAP_FLOW='1'))
        self.trim_band(sb)

    def trim_band(self, band):
        """Release the preallocation of one band's cutout cubes after the only task that writes them has ended (49 of
        118 GB in the first full cell). ONLY in this run's own cutout tree (<out>/photometry), which no other cell writes:
        an external --cutout-root may be shared between cells and is another filesystem's business - nothing is touched
        there. The engine's layout is <root>/<sample>/<band>/<source>/<file> (phase3_snapshot_cutouts.py)."""
        if self.a.cutout_root:
            return 0
        freed = 0
        for p in glob.glob(os.path.join(self.cutouts, '*', band, '*', '*.fits')):
            freed += release_preallocation(p)
        if freed > 2 ** 30:
            self.log('TRIMMED %s %s cutouts: %.0f GB of preallocation released' % (self.cell, band, freed / 2 ** 30))
        return freed

    def t_curves(self, sb, st):
        rc = self.call([self.a.py36, '-u', self.eng('phase3_snapshot_lightcurves.py'), 'extract', '--cell', self.cell, '--subband', sb,
                        '--stage', os.path.join(self.stage, sb), '--proto-root', os.path.dirname(self.out),
                        '--master-catalogs', self.a.master_catalogs, '--out', self.lcout, '--stokes', st],
                       os.path.join(self.out, 'lightcurves_%s_%s.log' % (sb, st)), append=True,
                       # V runs after I here, so a missing I table means I failed: V refuses at once instead of polling for an hour
                       env=self.env(P3_LC_WAIT_I_S='0'))
        if rc != 0:
            raise RuntimeError('LIGHTCURVE-FAILED %s %s %s (exit %d)' % (self.cell, sb, st, rc))

    def t_cube(self, sb, st):
        cub = os.path.join(self.cubework, 'cubes')
        os.makedirs(cub, exist_ok=True)
        rc = self.call([self.a.flowpy, self.eng('phase3_snapshot_cube.py'), 'write', '--stage', os.path.join(self.stage, sb), '--cell', self.cell,
                        '--subband', sb, '--stokes', st, '--flows', self.flowdir, '--out', cub],
                       os.path.join(self.cubework, 'cube_write_%s_%s.log' % (sb, st)))
        if rc == 0:
            rc = self.call([self.a.py36, self.eng('phase3_snapshot_cube.py'), 'verify', os.path.join(cub, '%s_%s_%s_flowcube.fits' % (self.cell, sb, st)),
                            '--stage', os.path.join(self.stage, sb)], os.path.join(self.cubework, 'cube_verify_%s_%s.log' % (sb, st)))
        if rc != 0:
            raise RuntimeError('cube %s %s %s not written or not verified (exit %d)' % (self.cell, sb, st, rc))
        release_preallocation(os.path.join(cub, '%s_%s_%s_flowcube.fits' % (self.cell, sb, st)))

    def t_flowmovie(self, sb):
        fl = sorted(glob.glob(os.path.join(self.stage, sb, '*Clean-Snapshot-*-image_flowed.fits')) +
                    glob.glob(os.path.join(self.stage, sb, '*Pilot-Snapshot-*-I-image_flowed.fits')))       # v10:278
        argv = [self.a.flowpy, self.eng('phase3_snapshot_flow_movie.py'), '--flows', self.flowdir, '--subband', sb]
        argv += ['--header-fits', fl[0]] if fl else []
        rc = self.call(argv + ['--out', os.path.join(self.flowdir, 'flow_%s.mp4' % sb)], os.path.join(self.flowdir, 'movie_%s.log' % sb))
        if rc != 0:
            raise RuntimeError('flow movie %s %s failed (exit %d)' % (self.cell, sb, rc))

    def t_subbands(self):
        """The fleet's COMPLETENESS GUARD (v10:207-241): every staged subband has BOTH Stokes runinfo files, or the night rule's
        refusal record, or is excluded in data_decisions.csv (read by the engine's own reader); anything else fails the cell."""
        r = self.py36('import sys; sys.path.insert(0, %r); import phase3_snapshot_search as s\n'
                      'print(" ".join(sorted(s.excluded_subbands(sys.argv[1], sys.argv[2]))))' % self.a.engine, self.lst, self.date)
        excluded = set(r.stdout.split())
        nri = len(glob.glob(os.path.join(self.out, 'runinfo_*MHz_*_median.json')))
        nnr, nex, miss, refused = 0, 0, [], []
        staged = self.active
        for sb in staged:
            if sb in excluded:
                nex += 2
                continue
            for st in ('V', 'I'):
                if os.path.exists(os.path.join(self.out, 'runinfo_%s_%s_median.json' % (sb, st))):
                    pass
                elif os.path.exists(os.path.join(self.out, 'nightrefused_%s_%s_median.json' % (sb, st))):
                    nnr += 1; refused.append('%s_%s' % (sb, st))
                else:
                    miss.append('%s_%s' % (sb, st))
        self.log('SUBBANDS %s runinfo=%d/%d' % (self.cell, nri, 2 * len(staged)))
        if miss:
            raise RuntimeError('SEARCH-INCOMPLETE %s runinfo=%d + night-rule refusals=%d of %d expected; unaccounted: %s - NOT shipped'
                               % (self.cell, nri, nnr, 2 * len(staged), ' '.join(miss)))
        if nex:
            self.log('EXCLUDED-SUBBANDS %s %d of %d searches excluded by data_decisions (accounted, not searched): %s'
                     % (self.cell, nex, 2 * len(staged), ' '.join(sorted(excluded & set(staged)))))
        if nnr:
            self.log('NIGHT-REFUSED %s %d of %d searches refused by the night rule (recorded, shipped as refusals): %s'
                     % (self.cell, nnr, 2 * len(staged), ' '.join(refused)))
        if nri == 0 and nnr == 0:
            raise RuntimeError('FLEET-ABORT %s produced no runinfo - systemic failure' % self.cell)

    def t_wideband(self, st):
        self.call([self.a.py36, '-u', self.eng('phase3_snapshot_group_search.py'), '--lst', self.lst, '--date', self.date, '--stokes', st,
                   '--thresh', '5.0', '--out', self.out, '--stage-root-fmt', os.path.join(self.stage, '%s'),
                   '--nthreads', str(min(8, self.a.wideband_threads))],
                  os.path.join(self.out, 'group_%s.log' % st), env=self.env(P3_SNAP_FLOW='1'))   # judged by the guard below

    def t_widebands(self):
        """WIDEBAND COMPLETENESS (v10:248-276): each Stokes' outcome record exists and Red, Green and Blue are each searched (with
        its runinfo), refused by the night rule (with its record), excluded, without usable members, or too few frames."""
        bad = []
        for st in ('V', 'I'):
            p = os.path.join(self.out, 'wideband_outcome_%s.json' % st)
            if not os.path.exists(p):
                bad.append('%s:no-outcome-record' % st)
                continue
            g = json.load(open(p)).get('groups', {})
            for grp in GROUPS:
                s = g.get(grp, 'missing')
                ok = (s in ('excluded', 'no-members', 'too-few-frames')
                      or (s == 'searched' and os.path.exists(os.path.join(self.out, 'runinfo_%s_%s_median.json' % (grp, st))))
                      or (s == 'night-refused' and os.path.exists(os.path.join(self.out, 'nightrefused_%s_%s_median.json' % (grp, st)))))
                if not ok:
                    bad.append('%s_%s:%s' % (grp, st, s))
        self.log('WIDEBAND %s runinfo=%d' % (self.cell, len(glob.glob(os.path.join(self.out, 'runinfo_*.json')))))
        for grp in GROUPS:                                             # both Stokes' searches have ended: their cutouts are whole
            self.trim_band(grp)
        if bad:
            raise RuntimeError('SEARCH-INCOMPLETE %s wideband group search unaccounted: %s - NOT shipped' % (self.cell, ' '.join(bad)))

    def t_framemovies(self):
        # v10:284-289. Made in the work tree and shipped with everything else: the fleet makes them after its two guards, so a
        # failed cell has no movies and no MOVIES_DONE
        os.makedirs(self.movies_work, exist_ok=True)
        log = os.path.join(self.movies_work, 'movies.log')
        rc = self.call([self.a.flowpy, '-u', self.eng('phase3_snapshot_frame_movie.py'), self.cell, self.movies_work, '--frames-root', self.stage], log)
        total = [l for l in open(log, errors='replace') if l.startswith('TOTAL')]
        if rc != 0 or not total:
            raise RuntimeError('FRAMEMOVIES-FAILED %s (see %s)' % (self.cell, log))
        self.log('FRAMEMOVIES %s %d %s' % (self.cell, len(glob.glob(os.path.join(self.movies_work, '*.mp4'))), total[-1].strip()))

    # ---------------------------------------------------------------- ship (v10:340-429), after every task
    @staticmethod
    def _cp_n(src, dstdir):
        os.makedirs(dstdir, exist_ok=True)
        dst = os.path.join(dstdir, os.path.basename(src))
        if not os.path.exists(dst):                                    # cp -n
            shutil.copyfile(src, dst + '.part')
            os.replace(dst + '.part', dst)

    def ship_all(self, outcomes):
        for pat in ('*.csv', '*.json', '*.txt', '*.log'):
            for p in glob.glob(os.path.join(self.out, pat)):
                self._cp_n(p, self.ship)
        models = glob.glob(os.path.join(self.out, 'model_*_median.fits'))
        for p in models:
            self._cp_n(p, self.ship)
        self.log('MODELS %s shipped=%d' % (self.cell, len(models)))
        for pat in ('*.mp4', '*_series.png', '*_series.csv', 'flow_*.csv', '*.log'):
            for p in glob.glob(os.path.join(self.flowdir, pat)):
                self._cp_n(p, os.path.join(self.ship, 'flow'))
        self.log('FLOWPRODUCTS %s movies=%d fields=%d' % (self.cell, len(glob.glob(os.path.join(self.ship, 'flow', '*.mp4'))),
                                                         len(glob.glob(os.path.join(self.flowtree, self.cell, '*.npy')))))
        # frame movies: the mp4 files, the log, then the marker (v10:286)
        if outcomes.get('framemovies') == 'done':
            for p in glob.glob(os.path.join(self.movies_work, '*.mp4')) + [os.path.join(self.movies_work, 'movies.log')]:
                self._cp_n(p, self.movies)
            open(os.path.join(self.movies, 'MOVIES_DONE'), 'w').close()
            shutil.rmtree(self.movies_work, ignore_errors=True)
        # light curves (v10:379-382)
        for p in glob.glob(os.path.join(self.lcout, '*_lightcurves.fits')):
            self._cp_n(p, self.lcship)
        nlc, nsb = len(glob.glob(os.path.join(self.lcship, '*_I_lightcurves.fits'))), len(self.active)
        self.log('LIGHTCURVES %s I-tables=%d/%d' % (self.cell, nlc, nsb))
        if nlc < nsb:
            self.log('LIGHTCURVES-INCOMPLETE %s %d of %d subbands - the chain will re-measure those from frames' % (self.cell, nlc, nsb))
        shutil.rmtree(self.lcout, ignore_errors=True)                  # v10:383
        # cubes (v10:391-429): all or nothing over the subbands with flowed frames (v10:400-402) - 2 per subband, each verified,
        # copied and checked by md5, then the MANIFEST
        csbs = [sb for sb in self.active if glob.glob(os.path.join(self.stage, sb, '*_flowed.fits'))]
        if self.a.cubes:
            cub = os.path.join(self.cubework, 'cubes')
            want = ['%s_%s_%s_flowcube.fits' % (self.cell, sb, st) for sb in csbs for st in ('I', 'V')]
            ok = [n for n in want if outcomes.get('cube:%s:%s' % tuple(n[len(self.cell) + 1:-len('_flowcube.fits')].split('_'))) == 'done'
                  and os.path.exists(os.path.join(cub, n))]
            verify = {}
            for sb in csbs:
                for st in ('I', 'V'):
                    p = os.path.join(self.cubework, 'cube_verify_%s_%s.log' % (sb, st))
                    for l in (open(p, errors='replace') if os.path.exists(p) else ()):
                        if l.startswith('VERIFY') and ' -> OK' in l:
                            verify['%s_%s_%s_flowcube.fits' % (self.cell, sb, st)] = l.strip()
            if not want or len(ok) != len(want) or len(verify) != len(want):
                self.log('CUBE-INCOMPLETE %s cubes=%d verify_OK=%d of %d - not shipped (logs %s)' % (self.cell, len(ok), len(verify), len(want), self.cubework))
                return 'cubes-incomplete'
            os.makedirs(self.cubes, exist_ok=True)
            lines, nbad = [], 0
            for n in want:
                shutil.copyfile(os.path.join(cub, n), os.path.join(self.cubes, n))
                m1, m2 = _md5(os.path.join(cub, n)), _md5(os.path.join(self.cubes, n))
                nbad += m1 != m2
                lines.append('%s md5=%s %s | %s' % (n, m2, 'copy-OK' if m1 == m2 else 'COPY-MISMATCH', verify[n]))
            if nbad:
                self.log('CUBE-SHIP-FAILED %s %d cubes differ after the copy' % (self.cell, nbad))
                try:                                                   # never a partial cube directory under its own name
                    os.rename(self.cubes, '%s.failed_%s' % (self.cubes, time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())))
                except OSError as e:
                    self.log('SET-ASIDE-FAILED %s %s: %s' % (self.cell, self.cubes, e))
                return 'cubes-ship-failed'
            lines.append('COMPLETE %s cubes=%d subbands=%d node=%s %s writer=cell.py' % (self.cell, len(want), len(csbs), os.uname()[1],
                                                                                        time.strftime('%a %b %e %H:%M:%S UTC %Y', time.gmtime())))
            with open(os.path.join(self.cubes, 'MANIFEST.txt'), 'w') as f:
                f.write('\n'.join(lines) + '\n')
            for kind in ('write', 'verify'):
                with open(os.path.join(self.cubes, 'cube_%s.log' % kind), 'w') as f:
                    for sb in csbs:
                        for st in ('I', 'V'):
                            p = os.path.join(self.cubework, 'cube_%s_%s_%s.log' % (kind, sb, st))
                            f.write(open(p, errors='replace').read() if os.path.exists(p) else '')
            self.log('CUBES %s %d cubes verified + shipped' % (self.cell, len(want)))
            shutil.rmtree(self.cubework, ignore_errors=True)           # v10:425 (kept when incomplete or the copy failed: the logs)
        return 'shipped'

    # ---------------------------------------------------------------- run
    def scheduler(self):
        a = self.a
        budgets = {'cores': self.cores_budget, 'mem_gb': self.mem_budget_gb, 'gpu': len(self.gpus) * a.flow_per_gpu}
        s = schedule.Scheduler(budgets)
        fns = {}
        for sb in self.active:
            fns['flow:' + sb] = lambda sb=sb: self.t_flow(sb)
            fns['fields:' + sb] = lambda sb=sb: self.t_fields(sb)
            fns['search:' + sb] = lambda sb=sb: self.t_search(sb)
            fns['flowmovie:' + sb] = lambda sb=sb: self.t_flowmovie(sb)
            for st in ('I', 'V'):
                fns['curves:%s:%s' % (sb, st)] = lambda sb=sb, st=st: self.t_curves(sb, st)
                fns['cube:%s:%s' % (sb, st)] = lambda sb=sb, st=st: self.t_cube(sb, st)
        fns.update({'subbands': self.t_subbands, 'wideband:V': lambda: self.t_wideband('V'), 'wideband:I': lambda: self.t_wideband('I'),
                    'widebands': self.t_widebands, 'framemovies': self.t_framemovies})
        soft = {}                                                      # what a non-fatal task really did

        def wrap(name):
            fn = fns[name]
            if name.startswith(FATAL_PREFIXES):
                return fn

            def run():                                                 # movies, light curves, cubes: recorded, never fatal (as v10)
                try:
                    r = fn()
                    if r not in (None, 0, True):
                        raise RuntimeError('returned %r' % (r,))
                except Exception as e:
                    soft[name] = str(e)[:300]
                    self.log('%s-FAILED %s: %s' % (name.split(':')[0].upper(), self.cell, soft[name]))
            return run
        for name, after, needs in build_tasks(self.active, self.wideband, self.frame_movies, a.cubes, self.threads):
            s.add(name, wrap(name), after=after, needs=needs, hold=name.startswith(HOLD_PREFIXES))
        return s, soft

    def run(self):
        """Run the cell. Returns 0 (done), 1 (failed), 2 (refused before any search), 130 (interrupted). Whatever happens the
        record D/cell_<cell>.json is written, no engine process is left running, and the staging is removed (--keep-stage keeps it)."""
        import signal
        t0 = time.time()
        self.record['started'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(t0))
        handlers = {}
        if threading.current_thread() is threading.main_thread():
            def on_signal(signum, frame):
                raise _Interrupted('signal %d' % signum)
            for sg in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
                handlers[sg] = signal.signal(sg, on_signal)
        rc = 1
        try:
            rc = self._run(t0)
        except SystemExit as e:                                        # a refusal before any search: logged and recorded, never silent
            self.log(str(e))
            self.record.update(status='refused', refusal=str(e))
            rc = 2
        except (_Interrupted, KeyboardInterrupt) as e:
            self._stopped_at = time.time()
            self.stop_children()
            self._record_partial(t0)
            self.log('CELL-INTERRUPTED %s (%s) - engine processes stopped, nothing shipped; work kept in %s' % (self.cell, e or 'keyboard', self.out))
            self.record['status'] = 'interrupted'
            self._set_aside()
            rc = 130
        except Exception as e:                                         # a defect or a full disk: never a silent traceback only
            self._stopped_at = time.time()
            self.stop_children()
            self._record_partial(t0)
            self.log('CELL-FAILED %s %s: %s - work kept in %s' % (self.cell, type(e).__name__, str(e)[:300], self.out))
            self.record.update(status='failed', error='%s: %s' % (type(e).__name__, str(e)[:500]))
            self._set_aside()
            rc = 1
        finally:
            for sg, h in handlers.items():
                signal.signal(sg, h)
            if self._own_stage and not self.a.keep_stage:              # v10:431 and every refusal and failure path of the fleet;
                shutil.rmtree(self.stage, ignore_errors=True)          # never the staging of an earlier run this one refused on
            self.record['elapsed_s'] = round(time.time() - t0, 1)
            os.makedirs(self.D, exist_ok=True)
            with open(os.path.join(self.D, 'cell_%s.json' % self.cell), 'w') as f:
                json.dump(self.record, f, indent=1)
        return rc

    def _record_partial(self, t0):
        """The rows of the tasks that had finished when the run stopped (the first full cell was stopped three minutes
        before its end and its record held no task at all)."""
        s = getattr(self, '_sched', None)
        if s is not None:
            try:
                self._record_steps(dict(s.outcomes), t0)
            except Exception as e:
                self.log('RECORD-PARTIAL-FAILED %s %s' % (self.cell, e))

    def _record_steps(self, out, t0):
        """One row per task outcome in the record: status, times since the cell's start, the CPU seconds and peak memory of
        its engine processes. Also called with what had finished when the run is interrupted or fails."""
        soft = getattr(self, '_soft', {})
        for name, o in list(out.items()):
            d = o.as_dict()
            d['seconds'] = round(d['seconds'], 1)
            for k in ('started', 'ended'):
                d[k] = None if d[k] is None else round(d[k] - t0, 1)   # seconds since the cell's start
            if name in soft:
                d['status'], d['detail'] = 'failed-not-fatal', soft[name]
            stopped = getattr(self, '_stopped_at', None)
            if stopped is not None and o.ended is not None and o.ended >= stopped:
                # its engine process was stopped with the run (a search or a wideband search returns without an error of
                # its own - their verdict is the completeness guard's): never 'done'
                d['status'], d['detail'] = 'interrupted', 'stopped with the run'
            if 'task-' + name in self._cpu:
                d['cpu_s'] = round(self._cpu['task-' + name], 1)
                d['maxrss_gb'] = round(self._rss.get('task-' + name, 0.0), 1)
            self.record['steps'][name] = d

    def _set_aside(self):
        """A cell that did not complete ships nothing: what the run put into a ship tree is moved aside, marked as failed
        (the flow fields the light curves read during the run; a ship that stopped half way)."""
        stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
        for d in (os.path.join(self.flowtree, self.cell), self.ship, self.lcship, self.movies, self.cubes):
            if os.path.isdir(d):
                try:
                    os.rename(d, '%s.failed_%s' % (d, stamp))
                except OSError as e:
                    self.log('SET-ASIDE-FAILED %s %s: %s' % (self.cell, d, e))

    def _run(self, t0):
        self.prepare()
        s, soft = self.scheduler()
        self._sched, self._soft = s, soft
        t1 = time.time()
        out = s.run()
        self._record_steps(out, t0)
        failed = [n for n, o in out.items() if o.status == 'failed']
        flows = [o for n, o in out.items() if n.startswith('flow:') and o.ended]
        if flows:
            self.log('FLOW %s %d subbands flowed, last at %.0f min' % (self.cell, len(flows), (max(o.ended for o in flows) - t0) / 60))
        if failed:
            self.record['status'] = 'failed'
            for n in failed:
                self.log('%s' % out[n].detail)
            self.log('CELL-FAILED %s (%s) - nothing shipped; work kept in %s' % (self.cell, ', '.join(failed), self.out))
            self._set_aside()
            return 1
        outcomes = {n: ('failed-not-fatal' if n in soft else o.status) for n, o in out.items()}
        self.record['ship'] = self.ship_all(outcomes)
        self.record['status'] = 'done'
        self.log('DONE %s runinfo=%d nightrefused=%d elapsed=%dmin (scheduler %dmin) node=%s'
                 % (self.cell, len(glob.glob(os.path.join(self.out, 'runinfo_*.json'))),
                    len(glob.glob(os.path.join(self.out, 'nightrefused_*.json'))), (time.time() - t0) / 60, (time.time() - t1) / 60, os.uname()[1]))
        return 0


class _Interrupted(BaseException):
    """SIGTERM / SIGINT / SIGHUP in the main thread (BaseException: no task wrapper may swallow it)."""


def _refuse_protected(path, what):
    r = os.path.realpath(path)
    for p in PROTECTED:
        pr = os.path.realpath(p)
        if r == '/' or r == pr or r.startswith(pr + '/') or pr.startswith(r + '/'):
            raise SystemExit('REFUSED: %s %s is, is inside, or contains a production tree (%s); this driver writes only to its own root'
                             % (what, path, p))


def release_preallocation(path):
    """Give back the blocks a filesystem holds for `path` beyond its length (XFS speculative preallocation): the file is
    set to its own length, which changes no byte, and its access and modification times are put back. Returns the bytes
    released (0 when there was nothing, on a filesystem without the effect, or when the file cannot be touched)."""
    try:
        st = os.stat(path)
        before = st.st_blocks * 512
        if before <= st.st_size + (1 << 20):                 # nothing worth a truncate
            return 0
        os.truncate(path, st.st_size)
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))
        return max(0, before - os.stat(path).st_blocks * 512)
    except OSError:
        return 0


def _count_cores(spec):
    """'0-23,30-31' -> 26"""
    n = 0
    for part in spec.split(','):
        lo, _, hi = part.strip().partition('-')
        n += int(hi or lo) - int(lo) + 1
    return n


def _mem_available_gb():
    """The memory this driver may count on, whole GB (0 when it cannot be read): MemAvailable bounded by the memory limit
    of the process's cgroups - core/nodemem.py, loaded by its path (this driver runs as a script outside the package).
    Until 2026-11-12 this read MemAvailable alone, which does not know the cap the nodes put on all user sessions
    together (lwacalim00: 400 GiB of 503)."""
    import importlib.util
    try:
        spec = importlib.util.spec_from_file_location(
            'exopipe3_nodemem', os.path.join(HERE, os.pardir, os.pardir, 'core', 'nodemem.py'))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.mem_available_gb() or 0
    except Exception:            # a copy of this driver without core/nodemem.py beside it: 0, so the driver REFUSES on
        return 0                 # its budgets with a recorded line (or runs on --mem-budget-gb), never a bare traceback


def _md5(path, block=1 << 22):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(block), b''):
            h.update(b)
    return h.hexdigest()


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--cell', required=True, help='<LST>_<date>, e.g. 04h_2024-12-19')
    ap.add_argument('--out', required=True, help='the root everything is written under (never a production tree)')
    ap.add_argument('--subbands', default='', help='space-separated; default all 15')
    ap.add_argument('--wideband', choices=('auto', 'yes', 'no'), default='auto', help='auto = only with all 15 subbands (the fleet\'s subset rule)')
    ap.add_argument('--frame-movies', choices=('auto', 'yes', 'no'), default='auto')
    ap.add_argument('--no-cubes', dest='cubes', action='store_false')
    ap.add_argument('--keep-stage', action='store_true', help='do not delete the staging at the end')
    ap.add_argument('--plan', action='store_true', help='print the task graph and stop')
    ap.add_argument('--stage-root', default=None)
    ap.add_argument('--cutout-root', default=None, help='where the search writes the photometry cutout cubes (default <out>/photometry)')
    ap.add_argument('--cubes-root', default=None, help='where the flowed-frame cubes are shipped (default <out>/snapshot_cubes)')
    ap.add_argument('--phase1-root', default=None, help='P3_SNAP_PHASE1_ROOT for the engines (e.g. the GPU pipeline frame root)')
    ap.add_argument('--masters', default='/lustre/pipeline/exopipe/phase3/snapshot_masters')
    ap.add_argument('--master-catalogs', default=OPS + '/snapshot_master_catalogs')
    ap.add_argument('--engine', default=HERE)
    ap.add_argument('--legacy-root', default='/lustre/gh/main/clv15', help='the unported modules of the light curves')
    ap.add_argument('--samples-dir', default='/lustre/gh/main/clv15', help='the target lists')
    ap.add_argument('--guard', default='', help='directory of the refactor\'s parity sitecustomize: children refuse writes outside --out')
    ap.add_argument('--py36', default='/opt/devel/pipeline/envs/development/bin/python')
    ap.add_argument('--flowpy', default='/opt/devel/pipeline/envs/py310_orca/bin/python')
    ap.add_argument('--funpack', default='/opt/devel/pipeline/envs/development/bin/funpack')
    ap.add_argument('--gpus', default='0 1')
    ap.add_argument('--flow-per-gpu', type=int, default=2)
    ap.add_argument('--xla-mem-fraction', default='0.45')
    ap.add_argument('--cores', default='', help='taskset core list for the engines; default: the node\'s free cores (node_free_cores.sh)')
    ap.add_argument('--cores-budget', type=int, default=0, help='cores the tasks may use together; default: the size of the core list')
    ap.add_argument('--mem-budget-gb', type=int, default=0, help='memory the tasks may hold together; default: 80%% of what is available at the start')
    ap.add_argument('--search-threads', type=int, default=8)
    ap.add_argument('--wideband-threads', type=int, default=12, help='threads of one wideband search (two run side by side)')
    ap.add_argument('--curves-threads', type=int, default=4)
    ap.add_argument('--movie-threads', type=int, default=8)
    ap.add_argument('--movie-procs', type=int, default=6)
    ap.add_argument('--copy-procs', type=int, default=6, help='files copied from Lustre at a time (REVIEW.md: at most 6)')
    ap.add_argument('--funpack-procs', type=int, default=16)
    ap.add_argument('--min-free-gb', type=int, default=800)
    ap.add_argument('--min-frames', type=int, default=1000)
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    c = Cell(a)
    if a.plan:
        for name, after, needs in build_tasks(c.subbands, c.wideband, c.frame_movies, a.cubes, c.threads):
            print('%-22s after %-40s needs %s%s' % (name, ', '.join(after) or '-', needs, '  (holds)' if name.startswith(HOLD_PREFIXES) else ''))
        return 0
    return c.run()


if __name__ == '__main__':
    sys.exit(main())
