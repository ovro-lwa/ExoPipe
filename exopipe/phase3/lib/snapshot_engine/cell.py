#!/usr/bin/env python3
"""ONE SNAPSHOT CELL (one LST hour of one night), the fleet's products, PIPELINED.

A port of the cell body of exopipe_ops/snapshot_fleet_node_v10.sh (lines 71-433; "v10:N" below names its lines) onto the
scheduler of schedule.py. The fleet runs the stages of a cell one after another and holds a node for ~4.5 hours at a load of
5-8 on 26 free cores (exopipe_ops/refactor_20261012/plan/STAGE_MAP_20261111.md section 1). Here the same engine commands, with
the same arguments, start as soon as what they read exists:

    stage -> regrid -> Master and zero-visibility gates                     (before anything else, as the fleet)
    flow:<sb>            FLOW_PER_GPU processes per GPU (two give byte-identical frames and ~1.4x the throughput)
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

WHAT FAILS THE CELL is what fails it in the fleet: staging incomplete, a broken or missing Master, an incomplete flow, too few
frames, an unaccounted subband or wideband search. Then nothing is shipped and the work directory is kept for inspection.
Movies, light curves and cubes do not fail the cell (v10: FRAMEMOVIES-FAILED, LIGHTCURVE-FAILED, CUBE-INCOMPLETE are logged
and the cell is DONE); here each is recorded with its status in the cell record, never silently.

WHERE IT WRITES: only under --out (D): D/stage/<cell> (staging), D/work/<cell> (the fleet's node-local output),
D/snapshot_fleet_out/<cell>, D/snapshot_flow/<cell>, D/snapshot_lightcurves/<cell>, D/snapshot_movies/<cell>,
D/snapshot_cubes/<cell>, D/photometry (cutout cubes), D/cell_<cell>.json (every task's outcome and times) and
D/cell_<cell>.log (the fleet's log lines). Production roots are not offered: the fleet remains production's driver until the
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
ON_GRID_NAXIS = 1508                                                                # v10:131
# memory a task holds, GB, by the subband's grid (1508 / 2358 / 3122 px): measured maxima of the engines on 04h 2024-12-19
# (search 5.4 / 12.8 / 21.0; cube writer ~15 / 40 / 70; wideband search 27 per Stokes)
MEM_SEARCH = {'low': 6, 'mid': 14, 'high': 23}
MEM_CUBE = {'low': 15, 'mid': 40, 'high': 70}
MEM_WIDEBAND = 28


def band_class(sb):
    f = int(sb[:-3])
    return 'low' if f <= 36 else ('mid' if f <= 59 else 'high')


def build_tasks(subbands, gpus, wideband, frame_movies, cubes, threads):
    """The task graph of one cell: [(name, after, needs)] in priority order. Pure: no I/O, so it is tested without data.
    Flows go high band first: the last subbands to finish their flow are then the cheapest to search, which keeps the tail
    after the last flow short. Subbands are dealt to the GPUs in turn, as the fleet does (v10:144-145)."""
    order = sorted(subbands, key=lambda s: -int(s[:-3]))
    tasks = []
    for k, sb in enumerate(order):
        g = 'gpu%s' % gpus[k % len(gpus)]
        c = band_class(sb)
        tasks.append(('flow:' + sb, (), {g: 1, 'cores': 1}))
        tasks.append(('fields:' + sb, ('flow:' + sb,), {'cores': 1}))
        tasks.append(('search:' + sb, ('flow:' + sb,), {'cores': threads['search'], 'mem_gb': MEM_SEARCH[c]}))
        tasks.append(('curves:%s:I' % sb, ('search:' + sb, 'fields:' + sb), {'cores': threads['curves']}))
        tasks.append(('curves:%s:V' % sb, ('curves:%s:I' % sb,), {'cores': threads['curves']}))
        if cubes:
            for st in ('I', 'V'):
                tasks.append(('cube:%s:%s' % (sb, st), ('flow:' + sb,), {'cores': 1, 'mem_gb': MEM_CUBE[c]}))
        tasks.append(('flowmovie:' + sb, ('flow:' + sb,), {'cores': 1}))
    tasks.append(('subbands', tuple('search:' + sb for sb in order), {'cores': 1}))
    if wideband:
        for st in ('V', 'I'):                                                       # v10:242: V then I
            tasks.append(('wideband:' + st, ('subbands',), {'cores': threads['wideband'], 'mem_gb': MEM_WIDEBAND}))
        tasks.append(('widebands', ('wideband:V', 'wideband:I'), {'cores': 1}))
    if frame_movies:
        tasks.append(('framemovies', tuple('flow:' + sb for sb in order), {'cores': threads['movies']}))
    return tasks


FATAL_PREFIXES = ('flow:', 'search:', 'subbands', 'wideband')        # these fail the cell, as in the fleet; the rest are recorded


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
        for p in (OPS + '/snapshot_fleet_out', '/lustre/pipeline/exopipe/phase3', '/lustre/gh/main/phase3'):
            if D == p or D.startswith(p + '/'):
                raise SystemExit('REFUSED: --out %s is inside a production tree (%s); this driver writes only to its own root' % (D, p))
        self.D = D
        self.stage = os.path.join(os.path.abspath(a.stage_root) if a.stage_root else os.path.join(D, 'stage'), self.cell)
        self.out = os.path.join(D, 'work', self.cell)                 # the fleet's $OUT
        self.flowdir = os.path.join(self.out, 'flow')                 # v10:143
        self.ship = os.path.join(D, 'snapshot_fleet_out', self.cell)
        self.flowtree = os.path.join(D, 'snapshot_flow')
        self.lcout = os.path.join(D, 'work', 'lightcurves', self.cell)
        self.lcship = os.path.join(D, 'snapshot_lightcurves', self.cell)
        self.movies = os.path.join(D, 'snapshot_movies', self.cell)
        self.cubework = os.path.join(D, 'work', 'cubes', self.cell)
        self.cubes = os.path.join(D, 'snapshot_cubes', self.cell)
        self.cutouts = os.path.join(D, 'photometry')
        self.logpath = os.path.join(D, 'cell_%s.log' % self.cell)
        self.gpus = a.gpus.split()
        self.threads = dict(search=a.search_threads, curves=a.curves_threads, wideband=a.wideband_threads, movies=a.movie_threads)
        self.record = OrderedDict(cell=self.cell, subbands=self.subbands, started=None, status='running', steps=OrderedDict())
        self._lock = threading.Lock()
        self.cores = None
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
                 P3_SNAP_DETECT_THREADS=str(self.a.search_threads), P3_SNAP_GROUP_THREADS=str(self.a.group_threads),
                 P3_SNAP_LC_THREADS=str(self.a.curves_threads), P3_SNAP_MOVIE_THREADS=str(self.a.movie_threads),
                 P3_SNAP_MOVIE_PROCS=str(self.a.movie_procs))
        if self.a.phase1_root:
            e['P3_SNAP_PHASE1_ROOT'] = self.a.phase1_root
        if self.a.guard:                                 # the refactor's write guard: a write outside --out is refused in the child
            e['EXOPIPE3_PARITY_ROOT'] = self.D
            e['PYTHONPATH'] = self.a.guard + (os.pathsep + e['PYTHONPATH'] if e.get('PYTHONPATH') else '')
        e.update({k: str(v) for k, v in more.items()})
        return e

    def pinned(self, argv):
        return (['taskset', '-c', self.cores] if self.cores else []) + list(argv)

    def call(self, argv, logfile, env=None, append=False, pin=True):
        """Run one engine command; its stdout and stderr go to logfile. Returns the exit code."""
        os.makedirs(os.path.dirname(logfile), exist_ok=True)
        with open(logfile, 'ab' if append else 'wb') as f:
            return subprocess.call(self.pinned(argv) if pin else list(argv), stdout=f, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, env=env or self.env())

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
        if a.cores:
            self.cores = a.cores
        else:
            r = subprocess.run(['bash', os.path.join(OPS, 'node_free_cores.sh')], stdout=subprocess.PIPE, universal_newlines=True)
            self.cores = r.stdout.strip() or None         # leave the data-recorder cores alone (v10:47)
        self.log('CLAIM %s (cell.py: subbands %d, gpus [%s] x %d flows, cores %s)' % (self.cell, len(self.subbands), ' '.join(self.gpus),
                                                                                    a.flow_per_gpu, self.cores))
        # DISK GUARD (v10:93-99)
        free = shutil.disk_usage(os.path.dirname(self.stage)).free // 2 ** 30
        if free < a.min_free_gb:
            raise SystemExit('DISK-REFUSED %s only %d GB free under %s (need %d)' % (self.cell, free, self.stage, a.min_free_gb))
        # STAGE (v10:100-124): one file per (Stokes, integration), chosen by the engine's own discovery; copy, funpack, count
        t0 = time.time()
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
                nreg += rc == 0
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
        self.zero_vis = set(p for p in r.stdout.splitlines() if p.startswith('/'))
        if self.zero_vis:
            self.log('ZERO-VIS %s %d staged frames with WSCNVIS = 0 refused (not expected to flow)' % (self.cell, len(self.zero_vis)))
        if nst < a.min_frames:                             # v10:200-203 (tested before the flow here: nothing to gain by flowing first)
            raise SystemExit('REFUSED %s only %d decompressed frames (need %d) - not searching' % (self.cell, nst, a.min_frames))

    # ---------------------------------------------------------------- the tasks
    def t_flow(self, sb, gpu):
        e = self.env(CUDA_VISIBLE_DEVICES=gpu, XLA_PYTHON_CLIENT_PREALLOCATE='false', XLA_PYTHON_CLIENT_MEM_FRACTION=self.a.xla_mem_fraction)
        rc = self.call([self.a.flowpy, self.eng('phase3_snapshot_flow.py'), '--stage', os.path.join(self.stage, sb),
                        '--reference', self.master(sb), '--subband', sb, '--out', self.flowdir],
                       os.path.join(self.flowdir, 'flow_%s.log' % sb), env=e)
        # FLOW COMPLETE for this subband (v10:192-199, per subband): every staged frame but the zero-visibility ones is flowed
        want = [p for p in self.staged(sb) if not p.endswith('_flowed.fits') and p not in self.zero_vis]
        missing = [p for p in want if not os.path.exists(p[:-5] + '_flowed.fits')]
        if rc != 0 or missing:
            raise RuntimeError('FLOW-INCOMPLETE %s %s: exit %d, %d of %d frames without a flowed copy' % (self.cell, sb, rc, len(missing), len(want)))

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
                   '--stage-root', os.path.join(self.stage, sb), '--nthreads', '8'], os.path.join(self.out, 'search_%s.log' % sb),
                  env=self.env(P3_SNAP_FLOW='1'))

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
        staged = [sb for sb in self.subbands if self.staged(sb)]
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
                   '--thresh', '5.0', '--out', self.out, '--stage-root-fmt', os.path.join(self.stage, '%s'), '--nthreads', '8'],
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
        if bad:
            raise RuntimeError('SEARCH-INCOMPLETE %s wideband group search unaccounted: %s - NOT shipped' % (self.cell, ' '.join(bad)))

    def t_framemovies(self):
        os.makedirs(self.movies, exist_ok=True)                        # v10:284-289
        log = os.path.join(self.movies, 'movies.log')
        rc = self.call([self.a.flowpy, '-u', self.eng('phase3_snapshot_frame_movie.py'), self.cell, self.movies, '--frames-root', self.stage], log)
        total = [l for l in open(log, errors='replace') if l.startswith('TOTAL')]
        if rc != 0 or not total:
            raise RuntimeError('FRAMEMOVIES-FAILED %s (see %s)' % (self.cell, log))
        open(os.path.join(self.movies, 'MOVIES_DONE'), 'w').close()
        self.log('FRAMEMOVIES %s %d %s' % (self.cell, len(glob.glob(os.path.join(self.movies, '*.mp4'))), total[-1].strip()))

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
        # light curves (v10:379-382)
        for p in glob.glob(os.path.join(self.lcout, '*_lightcurves.fits')):
            self._cp_n(p, self.lcship)
        nlc, nsb = len(glob.glob(os.path.join(self.lcship, '*_I_lightcurves.fits'))), len(self.subbands)
        self.log('LIGHTCURVES %s I-tables=%d/%d' % (self.cell, nlc, nsb))
        if nlc < nsb:
            self.log('LIGHTCURVES-INCOMPLETE %s %d of %d subbands - the chain will re-measure those from frames' % (self.cell, nlc, nsb))
        # cubes (v10:391-429): all or nothing - 2 per subband, each verified, copied and checked by md5, then the MANIFEST
        if self.a.cubes:
            cub = os.path.join(self.cubework, 'cubes')
            want = ['%s_%s_%s_flowcube.fits' % (self.cell, sb, st) for sb in self.subbands for st in ('I', 'V')]
            ok = [n for n in want if outcomes.get('cube:%s:%s' % tuple(n[len(self.cell) + 1:-len('_flowcube.fits')].split('_'))) == 'done'
                  and os.path.exists(os.path.join(cub, n))]
            verify = {}
            for sb in self.subbands:
                for st in ('I', 'V'):
                    p = os.path.join(self.cubework, 'cube_verify_%s_%s.log' % (sb, st))
                    for l in (open(p, errors='replace') if os.path.exists(p) else ()):
                        if l.startswith('VERIFY') and ' -> OK' in l:
                            verify['%s_%s_%s_flowcube.fits' % (self.cell, sb, st)] = l.strip()
            if len(ok) != len(want) or len(verify) != len(want):
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
                return 'cubes-ship-failed'
            lines.append('COMPLETE %s cubes=%d subbands=%d node=%s %s writer=cell.py' % (self.cell, len(want), len(self.subbands), os.uname()[1],
                                                                                        time.strftime('%a %b %e %H:%M:%S UTC %Y', time.gmtime())))
            with open(os.path.join(self.cubes, 'MANIFEST.txt'), 'w') as f:
                f.write('\n'.join(lines) + '\n')
            for kind in ('write', 'verify'):
                with open(os.path.join(self.cubes, 'cube_%s.log' % kind), 'w') as f:
                    for sb in self.subbands:
                        for st in ('I', 'V'):
                            p = os.path.join(self.cubework, 'cube_%s_%s_%s.log' % (kind, sb, st))
                            f.write(open(p, errors='replace').read() if os.path.exists(p) else '')
            self.log('CUBES %s %d cubes verified + shipped' % (self.cell, len(want)))
        return 'shipped'

    # ---------------------------------------------------------------- run
    def scheduler(self):
        a = self.a
        budgets = {'cores': a.cores_budget, 'mem_gb': a.mem_budget_gb}
        budgets.update({'gpu%s' % g: a.flow_per_gpu for g in self.gpus})
        s = schedule.Scheduler(budgets)
        order = sorted(self.subbands, key=lambda x: -int(x[:-3]))
        gpu_of = {sb: self.gpus[k % len(self.gpus)] for k, sb in enumerate(order)}
        fns = {}
        for sb in self.subbands:
            fns['flow:' + sb] = lambda sb=sb: self.t_flow(sb, gpu_of[sb])
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
        for name, after, needs in build_tasks(self.subbands, self.gpus, self.wideband, self.frame_movies, a.cubes, self.threads):
            s.add(name, wrap(name), after=after, needs=needs)
        return s, soft

    def run(self):
        t0 = time.time()
        self.record['started'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(t0))
        try:
            self.prepare()
        except SystemExit as e:                                        # a refusal before any search: logged and recorded, never silent
            self.log(str(e))
            self.record.update(status='refused', refusal=str(e), elapsed_s=round(time.time() - t0, 1))
            with open(os.path.join(self.D, 'cell_%s.json' % self.cell), 'w') as f:
                json.dump(self.record, f, indent=1)
            return 2
        s, soft = self.scheduler()
        t1 = time.time()
        out = s.run()
        for name, o in out.items():
            d = o.as_dict()
            d['seconds'] = round(d['seconds'], 1)
            for k in ('started', 'ended'):
                d[k] = None if d[k] is None else round(d[k] - t0, 1)   # seconds since the cell's start
            if name in soft:
                d['status'], d['detail'] = 'failed-not-fatal', soft[name]
            self.record['steps'][name] = d
        failed = [n for n, o in out.items() if o.status == 'failed']
        flows = [o for n, o in out.items() if n.startswith('flow:') and o.ended]
        if flows:
            self.log('FLOW %s %d subbands flowed, last at %.0f min' % (self.cell, len(flows), (max(o.ended for o in flows) - t0) / 60))
        if failed:
            self.record['status'] = 'failed'
            for n in failed:
                self.log('%s' % out[n].detail)
            self.log('CELL-FAILED %s (%s) - nothing shipped; work kept in %s' % (self.cell, ', '.join(failed), self.out))
        else:
            outcomes = {n: ('failed-not-fatal' if n in soft else o.status) for n, o in out.items()}
            self.record['ship'] = self.ship_all(outcomes)
            self.record['status'] = 'done'
            if not self.a.keep_stage:
                shutil.rmtree(self.stage, ignore_errors=True)          # v10:431
            self.log('DONE %s runinfo=%d nightrefused=%d elapsed=%dmin (scheduler %dmin) node=%s'
                     % (self.cell, len(glob.glob(os.path.join(self.out, 'runinfo_*.json'))),
                        len(glob.glob(os.path.join(self.out, 'nightrefused_*.json'))), (time.time() - t0) / 60, (time.time() - t1) / 60, os.uname()[1]))
        self.record['elapsed_s'] = round(time.time() - t0, 1)
        with open(os.path.join(self.D, 'cell_%s.json' % self.cell), 'w') as f:
            json.dump(self.record, f, indent=1)
        return 0 if not failed else 1


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
    ap.add_argument('--cores-budget', type=int, default=26)
    ap.add_argument('--mem-budget-gb', type=int, default=250)
    ap.add_argument('--search-threads', type=int, default=8)
    ap.add_argument('--group-threads', type=int, default=12)
    ap.add_argument('--wideband-threads', type=int, default=12)
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
        for name, after, needs in build_tasks(c.subbands, c.gpus, c.wideband, c.frame_movies, a.cubes, c.threads):
            print('%-22s after %-40s needs %s' % (name, ', '.join(after) or '-', needs))
        return 0
    return c.run()


if __name__ == '__main__':
    sys.exit(main())
