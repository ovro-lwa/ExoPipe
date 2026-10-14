"""exopipe3 search --tier sublst - the 10-minute (sub-LST) search, one unit per (LST-band cell, Stokes).

Called by steps/search.py (the tier dispatcher) through add_arguments(p) and run(args, run).

Unit of work: an LST-band cell <LST>_<band> (the deep-cell grammar; band = one of the 15 subbands or Red/Green/Blue - this
tier has no Full) and one Stokes; record _records/search/sublst-<arm>_<cell>_<S>[_<night>].json (the arm is in the unit name
so the A, B and BI searches of one cell never share a record, spec K11). Items (--cells): an LST-band cell 09h_18MHz (every
night; patterns 09h_* match the 18 bands), a unit name 09h_18MHz_V / sublst-B_09h_18MHz_V (that Stokes), or a NIGHT CELL
(core.cells.parse_night_cell) 09h_2025-01-02 (every band, that night) / 09h_2025-01-02_18MHz (one band, that night); --lsts
09h = every band of the LST; --epochs 09h_2025-01-02 (the common per-hour selector) = every band of that night (with
--cells/--lsts also given, those only filter the bands); --scopes / --stokes / --exclude-cells filter.

Two operating modes (DESIGN 12):
- intermittent (a search pass over an LST, today's sublst_search_pass.sh B): every night's windows of the cell are searched;
  the pass starts clean (NOCAT:62-68): an existing published <cell>_<S>/ refuses unless --fresh, which moves it to
  <out>/_superseded/ first.
- per hour (a night cell or --epochs <LST>_<date>): that night's windows only, run_deep_cell included_dates = the night's
  window keys (every admission gate still runs on the whole cell), appended to a copy of the cell's PUBLISHED records
  (catalogue, ledgers, refusal records, persisted sigma-stack maps); the publish is transactional (the replaced cumulative
  files are backed up under <out>/_superseded/, the published cell is verified, and a failure rolls it back). Needs the
  cell's full search published first; a night already in the published image ledger refuses. search_sublst.EPOCHS_MODE =
  'refuse' turns per-hour mode off (spec G4).

Stages (search_sublst.STAGES; --stages/--from/--until of the step pick a subset, run in this order):
  preflight  inputs present and current: the arm's residual tree (NO_WINDOWS.txt = nothing to search, exit 3), >= MIN_EPOCHS
             windows (else the engine's P3-POLICY-SKIP: exit 3), the co-add copy's I master (R0) and its COADDOP (and the V
             master's) = MASTER_GEN, residuals_provisional.csv, the v110 null tables (V, md5-pinned), the engine interpreter;
             the admitted night set (core.exclusions; excluded windows are dropped inside the engine, E9, and listed in the
             record); the lease (search_sublst.LEASE: per-node memory lease, or 'lst-mutex' = today's one-LST search slot);
             free space on the attempt filesystem (FAST_MIN_FREE_GB: a node condition = error, retried by the runner).
  detect     lib/search/deep_search.run_deep_cell(lst, band, label='SLST<hh><arm>', stokes=S, cadence='10min') IN PLACE in a
             fresh child process run by core.legacy.run (site PYTHON['default']; module mode, this file's --engine-child
             entry, its arguments in <attempt>/.engine_spec.json; the parity guard and the child's code fingerprint come with it) whose environment is built from config:
             every inherited P3_* variable cleared, then P3_TENMIN_ROOT, P3_COADD_ROOT, P3_MASTER_GEN, P3_TABLE_PREFIX=sublst_search,
             P3_TABLE_RECORD_ONLY=1, P3_REFINE_OFF=1, P3_ARCHIVE_WRITE=0, P3_FLOW_ROOT/P3_STAGING_ROOT/P3_[IV]RESID_ROOT in
             the attempt dir, and the engine's tunables (search_sublst). Log: <attempt>/<cell>.log.
  fit        the FITTED position of every catalogue row on the frame the search wrote (core.fitting.detection_position,
             tier sublst, K = fitting.K_TEMPLATE[sublst,S]; deep systematic blanked = error model A; '0.0' written as '0.1')
             -> sidecar fit_positions.csv + fit_positions_summary.json (content test F1-F4 of sublst_fit_positions.py).
             The catalogue keeps the peak position; catalog's fitpos stays the operative fit in the window (spec K14).
  verify     the content test of every product (pre-publish): catalogue, gate ledger (zero in-search 'table' removals -
             the record-only rule, SP:49-50), image ledger, refusal records, sigma-stack maps, search frames, stamps.
  publish    copy Run_<label>/, transients/, <cell>.log, catdir/, v110_out/ (+ the fit sidecar) to
             <out>/search_<arm>/<cell>_<S>/ (SP:53-55) and record every product there; the attempt is removed on success
             (SP:57), kept on any failure.

Inputs:  <paths.SUBLST_ROOT>/<TREE_SUBDIRS[arm]>/<cell>/{subtracted/<YYYY-MM-DDTkk>_<S>_sub_<band>.fits, bin_masters/,
         frames.csv, NO_WINDOWS.txt}; <paths.SUBLST_ROOT>/coadd_<LST>/<cell>/{coadd_final/{I_deep_Taper_R0,I_deep_Taper_R075,V_deep_Taper_R0,PB_deep}_<band>.fits,
         coadd_final/weights.csv, residuals_provisional.csv}; data_decisions.csv; the bright-parent table file (record-only);
         the v110 null tables (V).
Outputs (product classes): transients/Run_<label>/master_catalog.csv (detections), v_deep_ledger.csv (gate_ledger),
         v_deep_images.csv (image_ledger), refusals/<cell>_<kind>_excluded.csv (refusal_record), sigstack/*.fits
         (noise_map), Run_<label>/{iresid,vresid}/<cell>/*_redo_resid[_pbcorr]_<band>.fits (search_frame),
         transients/Run_<label>/candidates/ (candidate_stamps), catdir/master_R075_<cell>.gaul.csv (master_catalog, masters' class),
         v110_out/V_top100_fp_gate_p3.csv (fp_gate_verdicts, V), fit_positions.csv (detection_positions); --replay-attempt
         or a run without publish also writes verify_report.json (sublst_verify_report); <cell>.log (sublst_search_log).
Failure modes: exit 3 = no window (NO_WINDOWS.txt) / fewer than MIN_EPOCHS windows / incremental night with no window;
         exit 2 = a gate refused (the engine's REFUSING/REFUSED RuntimeError, a missing master or table, COADDOP, a pinned
         md5, an in-search table removal, a content test, an override of an engine-locked key, a production --out while
         paths.PRODUCTION_WRITES is False); exit 1 = engine error (traceback in <attempt>/<cell>.log; the attempt is kept),
         too little free disk, a lease not obtained within LEASE_WAIT_S.
"""
import contextlib
import csv
import glob
import hashlib
import json
import math
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import uuid

TIER = 'sublst'
CONFIG = ('search_sublst', 'fitting', 'runner')       # the sections this tier reads (plus config.SHARED_SECTIONS)
_CELL_RE = re.compile(r'^(?:sublst-(?P<arm>A|B|BI)_)?(?P<lst>\d\dh)_(?P<band>\d+MHz|Red|Green|Blue|Full)'
                      r'(?:_(?P<stokes>[IV]))?(?:_(?P<date>\d{4}-\d\d-\d\d))?$')
_SUMMARY_RE = re.compile(r'^\s*(\d\dh) (\S+): (\d+) epochs -> (\d+) candidates', re.M)
# the engine modules whose source is the code of a detect (sha1 recorded as inputs: DESIGN 11.5 code identity)
ENGINE_SOURCES = ('lib/search/deep_search.py', 'lib/search/identification.py', 'lib/search/search_i.py',
                  'lib/search/search_v.py', 'lib/search/veto.py', 'lib/search/candidates.py',
                  'lib/search/parent_curve_table.py', 'lib/search/c50.py', 'tenmin_frames.py', 'phase3_tenmin.py',
                  'core/fitting.py', 'core/residuals.py')
REFUSAL_COLUMNS = {'nightimage': 4, 'epochgate': 4, 'sanity': 6, 'sanity_I': 6, 'sanity_V': 6, 'searchtier': 7,
                   'stale_input': 5, 'beam': 6, 'extraction_I': 7, 'extraction_V': 7}     # deep_search.py:1109-1585


# =====================================================================================================================
# shared helpers (tier_snapshot imports these)
# =====================================================================================================================

def cfg():
    from exopipe.phase3.config import CFG
    return CFG


def pkg_root():
    """The repository root holding the exopipe package this module runs from."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))


def phase3_dir():
    return os.path.join(pkg_root(), 'exopipe', 'phase3')


def refusal(msg):
    from exopipe.phase3.core.record import Refusal
    return Refusal(msg)


def no_input(msg):
    from exopipe.phase3.core.record import NoInput
    return NoInput(msg)


def utc_stamp():
    return time.strftime('%Y%m%dT%H%M%S', time.gmtime())


def stage_list(args, order, optional=(), tier=''):
    """The stages to run, in ``order``: --stages a,b (any of order + optional), else ``order``; then --from / --until.
    Reads the step's shared flags under the dests stages, from_stage (or from_/stage_from), until (or until_stage)."""
    allowed = tuple(order) + tuple(optional)
    full = list(order) + [s for s in optional if s not in order]
    spec = getattr(args, 'stages', None)
    if spec:
        want = [s.strip() for s in str(spec).split(',') if s.strip()]
        bad = [s for s in want if s not in allowed]
        if bad:
            raise refusal(f'--stages {",".join(bad)}: not a stage of tier {tier} (stages: {", ".join(allowed)})')
        out = [s for s in full if s in want]
    else:
        out = list(order)
    frm = next((getattr(args, n) for n in ('from_stage', 'from_', 'stage_from', 'frm') if getattr(args, n, None)), None)
    until = next((getattr(args, n) for n in ('until', 'until_stage', 'stage_until') if getattr(args, n, None)), None)
    for name, v in (('--from', frm), ('--until', until)):
        if v and v not in allowed:
            raise refusal(f'{name} {v}: not a stage of tier {tier} (stages: {", ".join(allowed)})')
    if frm:
        out = [s for s in out if full.index(s) >= full.index(frm)]
    if until:
        out = [s for s in out if full.index(s) <= full.index(until)]
    if not out:
        raise refusal(f'no stage selected for tier {tier} (--stages/--from/--until)')
    return out


def check_overrides(section, tier):
    """Refuse an override of a <tier> entry of another search section, which this tier does not read (it would be a
    silent no-op): the tier's values live in its own section. (Engine-hard-coded keys and pinned inputs are the section's
    PROTECTED_KEYS, which config.py itself refuses to override.)"""
    c = cfg()
    for o in c.overrides:
        key = o[0]
        sec = key.split('[', 1)[0].partition('.')[0]
        if sec.startswith('search') and sec != section and f'[{tier}' in key:
            raise refusal(f'{key}: the {tier} tier reads its values from config section {section}, not {sec} - override '
                          f'{section}.<KEY> instead (exopipe3 config --section {section})')


def production_guard(out, extra_roots, run_dry):
    """While paths.PRODUCTION_WRITES is False, refuse an --out under a production root (paths.PRODUCTION_ROOTS and this
    tier's own production trees, spec G3/K3 - e.g. paths.SNAPSHOT_FLEET_OUT is not in PRODUCTION_ROOTS). A dry run writes
    nothing and is allowed."""
    c = cfg()
    if run_dry or c.paths.PRODUCTION_WRITES:
        return
    real = os.path.realpath(out)
    for root in tuple(c.paths.PRODUCTION_ROOTS) + tuple(extra_roots):
        r = os.path.realpath(root)
        if real == r or real.startswith(r.rstrip('/') + '/'):
            raise refusal(f'--out {out} is under the production root {root} and paths.PRODUCTION_WRITES is False '
                          f'(refactor window): pass --out to a parity/scratch root')


def site_python(key):
    """The site interpreter of ``key`` (core.legacy.interpreter; vocabulary default/flow/py36/casa/tess); refuses when the
    key is unset or the interpreter is not on this node."""
    from exopipe.phase3.core import legacy
    py = legacy.interpreter(key)
    if not os.path.exists(py):
        raise refusal(f'site PYTHON[{key!r}] = {py} does not exist on {socket.gethostname()}')
    return py


def free_gb(path):
    p = path
    while p and not os.path.exists(p):
        p = os.path.dirname(p)
    st = os.statvfs(p or '/')
    return st.f_bavail * st.f_frsize / 2 ** 30


def wait_free(path, need_gb, wait_s, what):
    """A node condition: free space on ``path``'s filesystem >= need_gb, waiting up to wait_s; else an ERROR (retryable,
    never a refusal or a done state - INCIDENT-FAST-FULL-SILENT-SHIPPING-FAILURES)."""
    t0 = time.time()
    while True:
        f = free_gb(path)
        if f >= need_gb:
            return f
        if time.time() - t0 >= wait_s:
            raise RuntimeError(f'{what}: only {f:.0f} GB free on the filesystem of {path} (need {need_gb} GB) - a node '
                               f'condition, not a result: retry on a node with room')
        time.sleep(min(60, max(1, wait_s)))


_MD5 = {}


def md5_file(path):
    st = os.stat(path)
    key = (path, st.st_size, st.st_mtime_ns)
    if key not in _MD5:
        h = hashlib.md5()
        with open(path, 'rb') as f:
            for blk in iter(lambda: f.read(1 << 20), b''):
                h.update(blk)
        _MD5[key] = h.hexdigest()
    return _MD5[key]


def git_head(path):
    try:
        return subprocess.run(['git', '-C', path, 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True,
                              timeout=20).stdout.strip() or 'unknown'
    except Exception:
        return 'unknown'


def has_tests(cls):
    from exopipe.phase3.core import validate
    try:
        validate.tests_for(cls)
        return True
    except KeyError:
        return False


def _untested(u, cls, path, n=1):
    """A product of a class with no registered content test REFUSES the unit (README 4: every class written has a content
    test; review 2026-10-13 - was a note, so a missing test module passed silently)."""
    u.meta.setdefault('untested_products', []).append({'path': os.path.abspath(path), 'class': cls, 'n': n})
    u.check(f'content_test_registered:{cls}', False, value=n, cls=cls, path=path,
            detail=f'no content test registered for product class {cls!r} ({n} file(s), e.g. {path})')


def record_output(u, path, cls, **ctx):
    """add_output when the class has a registered content test; otherwise the unit is refused (_untested)."""
    if has_tests(cls):
        return u.add_output(path, cls, **ctx)
    _untested(u, cls, path)
    return None


def record_outputs(u, paths, cls, **ctx):
    """record_output for many files of one class; without a registered test the unit is refused (one check)."""
    paths = list(paths)
    if not paths:
        return 0
    if has_tests(cls):
        for p in paths:
            u.add_output(p, cls, **ctx)
        return len(paths)
    _untested(u, cls, paths[0], n=len(paths))
    return len(paths)


def precheck(u, path, cls, label='pre-publish'):
    """Run the class's content tests on a file without recording it as an output (the verify stage, before publish)."""
    from exopipe.phase3.core import validate
    if not has_tests(cls):
        _untested(u, cls, path)
        return False
    ok = True
    for r in validate.run(cls, path):
        u.check(f"{label}:{r['name']}", r['passed'], value=r['value'], detail=r['detail'], cls=cls, path=path)
        ok = ok and r['passed']
    return ok


def record_failure(run, name, exc, **meta):
    """A unit that raised BEFORE its record was opened (preflight: an unreadable header, the exclusion list, no scratch root)
    is recorded with that outcome - refused (Refusal), no-input (NoInput) or error - so the run's exit code and records show
    it (README 3-4). Before the review fix (2026-10-13) it was only printed: the unit was missing from the run and the exit
    code ignored it (0 when the other units were done, 3 when it was the only one)."""
    if any(u.name == name for u in list(run.units)):
        return
    try:
        with run.unit(name, **meta):
            raise exc
    except Exception:                    # errors re-raise from the record's __exit__ (status 'error', record written)
        pass


def dry_exit(run, what='unit(s)'):
    """Exit code of a dry run (the plan): 2 when any planned unit would be refused by a gate, 3 when every unit has no
    input, else 0 - the same rule as record.Run.exit_code, with 'planned' counted as done."""
    by = {}
    for u in run.units:
        by[u.status] = by.get(u.status, 0) + 1
    print(f'\ndry run: {len(run.units)} {what}: ' + ', '.join(f'{k} {v}' for k, v in sorted(by.items())))
    st = [u.status for u in run.units]
    if not st or all(x == 'no-input' for x in st):
        return 3
    return 2 if 'refused' in st else 0


def legacy_call(target, args, python, env, log, unit=None, cwd=None, module=False):
    """core.legacy.run (the ONE child runner: site interpreter, inherited P3_* cleared, parity guard, child code
    fingerprint into the unit record) with ``env`` (None values dropped). The outcome is decided by the caller from the
    files and the exit status, never from log text (refusal_text=())."""
    from exopipe.phase3.core import legacy
    if log:
        os.makedirs(os.path.dirname(os.path.abspath(log)), exist_ok=True)
    return legacy.run(target, list(args), python=python, module=module,
                      env={k: str(v) for k, v in env.items() if v is not None}, cwd=cwd, log=log, unit=unit,
                      refusal_codes=(), refusal_text=())


def log_tail(path, n=5):
    try:
        with open(path, 'rb') as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 20000))
            lines = f.read().decode('utf-8', 'replace').strip().splitlines()
        return ' | '.join(lines[-n:])
    except OSError:
        return ''


def copy_tree(src, dst):
    """rsync -a semantics for one item: a directory is merged into dst (files replaced, extra files kept), a file
    replaced; mtimes preserved."""
    if os.path.isdir(src):
        shutil.copytree(src, dst, symlinks=True, dirs_exist_ok=True)
    else:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        tmp = f'{dst}.tmp.{os.getpid()}'
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)


# ---------------------------------------------------------------------------------------------------------------------
# Leases (DESIGN 11.7 "named leases declared per step in config"; the runner has none yet, so the step holds them)
# ---------------------------------------------------------------------------------------------------------------------

def mem_total_gb():
    with open('/proc/meminfo') as f:
        for line in f:
            if line.startswith('MemTotal:'):
                return int(line.split()[1]) / 2 ** 20
    return 0.0


def lease_root(out_root):
    """Where lease files live: search_sublst.LEASE_ROOT, else <runner.JOBS_ROOT>/_leases; in a parity run (write guard
    active) <out>/_leases, the only shared place the guard allows."""
    c = cfg()
    root = c.search_sublst.LEASE_ROOT or os.path.join(c.runner.JOBS_ROOT, '_leases')
    try:
        from exopipe.phase3.core import guard
        if guard.active() and not guard.is_allowed(root):
            root = os.path.join(out_root, '_leases')
    except Exception:
        pass
    return root


class Lease:
    """A named lease held by one unit, with heartbeats; holders are files <root>/<name>/holders/<id>.json written under a
    mkdir lock (atomic on Lustre). Modes:
      'node-memory': at most ``capacity`` live holders (a per-node counting lease, name carries the host);
      'lst-mutex'  : every live holder has the same ``group`` (today's search slot: one LST's searches at a time).
    A holder is stale when, on this host, its pid is gone, or when its heartbeat is older than stale_s; stale holders
    are removed by the next taker (NOCAT:41-42 reclaimed a slot whose owner process was gone)."""

    def __init__(self, root, name, mode, group='', capacity=1, poll_s=120, wait_s=0, heartbeat_s=60, stale_s=900,
                 info=None):
        self.dir = os.path.join(root, name)
        self.hdir = os.path.join(self.dir, 'holders')
        self.mode, self.group, self.capacity = mode, group, int(capacity)
        self.poll_s, self.wait_s, self.heartbeat_s, self.stale_s = poll_s, wait_s, heartbeat_s, stale_s
        self.host = socket.gethostname()
        self.id = f'{self.host}_{os.getpid()}_{utc_stamp()}_{uuid.uuid4().hex[:8]}'
        self.path = os.path.join(self.hdir, f'{self.id}.json')
        self.info = dict(info or {})
        self._stop = threading.Event()
        self._thread = None
        self.waited_s = 0.0

    # -- lock ----------------------------------------------------------------------------------------------------------
    @contextlib.contextmanager
    def _lock(self):
        os.makedirs(self.hdir, exist_ok=True)
        lk = os.path.join(self.dir, '.lock')
        t0 = time.time()
        while True:
            try:
                os.mkdir(lk)
                break
            except FileExistsError:
                try:
                    if time.time() - os.stat(lk).st_mtime > 120:      # a lock older than 2 min is a dead taker's
                        os.rmdir(lk)
                        continue
                except FileNotFoundError:
                    continue
                if time.time() - t0 > 600:
                    raise RuntimeError(f'lease lock {lk} held for more than 600 s')
                time.sleep(0.2)
        try:
            yield
        finally:
            try:
                os.rmdir(lk)
            except OSError:
                pass

    def _holders(self):
        out = []
        for p in glob.glob(os.path.join(self.hdir, '*.json')):
            try:
                with open(p) as f:
                    h = json.load(f)
                h['_path'], h['_mtime'] = p, os.stat(p).st_mtime
            except (OSError, ValueError):
                continue
            out.append(h)
        return out

    def _stale(self, h):
        if h.get('host') == self.host:
            try:
                os.kill(int(h.get('pid', -1)), 0)
            except ProcessLookupError:
                return True
            except (PermissionError, ValueError, OverflowError):
                pass
        return time.time() - h['_mtime'] > self.stale_s

    def _admit(self, live):
        if self.mode == 'node-memory':
            return len(live) < self.capacity
        if self.mode == 'lst-mutex':
            return all(h.get('group') == self.group for h in live)
        return True

    def try_acquire(self):
        with self._lock():
            live = []
            for h in self._holders():
                if self._stale(h):
                    try:
                        os.remove(h['_path'])
                    except OSError:
                        pass
                else:
                    live.append(h)
            if not self._admit(live):
                return False, live
            rec = dict(self.info, host=self.host, pid=os.getpid(), group=self.group, mode=self.mode,
                       taken=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
            tmp = self.path + '.tmp'
            with open(tmp, 'w') as f:
                json.dump(rec, f)
            os.replace(tmp, self.path)
            return True, live

    def acquire(self, say=print):
        t0 = time.time()
        said = False
        while True:
            ok, live = self.try_acquire()
            if ok:
                self.waited_s = time.time() - t0
                self._thread = threading.Thread(target=self._beat, daemon=True)
                self._thread.start()
                return self
            if not said:
                say(f'lease {os.path.basename(self.dir)} ({self.mode}{", group " + self.group if self.group else ""}): '
                    f'waiting - held by {", ".join(sorted(h.get("unit", "?") + "@" + h.get("host", "?") for h in live))}')
                said = True
            if self.wait_s and time.time() - t0 >= self.wait_s:
                raise RuntimeError(f'lease {self.dir} not obtained within {self.wait_s} s (holders: '
                                   f'{", ".join(h.get("unit", "?") for h in live)})')
            time.sleep(self.poll_s)

    def _beat(self):
        while not self._stop.wait(self.heartbeat_s):
            try:
                os.utime(self.path)
            except OSError:
                pass

    def release(self):
        self._stop.set()
        try:
            os.remove(self.path)
        except OSError:
            pass

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *exc):
        self.release()
        return False


def make_lease(section, out_root, unit, lst, reserve_gb=None):
    """The lease of one unit from a config section with LEASE, MEM_RESERVE_GB, NODE_MEM_MARGIN_GB, LEASE_* keys."""
    s = getattr(cfg(), section)
    mode = s.LEASE
    if mode == 'none':
        return None
    if mode not in ('node-memory', 'lst-mutex'):
        raise refusal(f'{section}.LEASE = {mode!r}: node-memory, lst-mutex or none')
    reserve = float(reserve_gb if reserve_gb is not None else s.MEM_RESERVE_GB)
    host = socket.gethostname()
    if mode == 'node-memory':
        cap = int(max(0.0, mem_total_gb() - s.NODE_MEM_MARGIN_GB) // reserve) if reserve > 0 else 10 ** 6
        if cap < 1:
            raise refusal(f'{host}: {mem_total_gb():.0f} GB of memory less the {s.NODE_MEM_MARGIN_GB} GB margin holds no '
                          f'{reserve:.0f} GB unit ({section}.MEM_RESERVE_GB)')
        name = f'{section}-mem-{host}'
    else:
        cap = 1
        name = f'{section}-slot'
    return Lease(lease_root(out_root), name, mode, group=lst, capacity=cap, poll_s=s.LEASE_POLL_S, wait_s=s.LEASE_WAIT_S,
                 heartbeat_s=s.LEASE_HEARTBEAT_S, stale_s=s.LEASE_STALE_S, info={'unit': unit, 'reserve_gb': reserve})


# =====================================================================================================================
# the sub-LST tier
# =====================================================================================================================

def add_arguments(p):
    g = p.add_argument_group('tier sublst (search --tier sublst)')
    g.add_argument('--sublst-arm', choices=('A', 'B', 'BI'), default=None,
                   help='residual tree searched: A = tenmin_root, B = tenmin_root_x2 (the production set), BI = B with '
                        'injections (default search_sublst.ARM); sets the label SLST<hh><arm> and <out>/search_<arm>/')
    g.add_argument('--sublst-keep-attempt', action='store_true',
                   help='keep the node-local attempt dir after a published success (default: removed, as SP:57)')


def default_out(args):
    return cfg().paths.SUBLST_ROOT            # published copies under <root>/search_<arm>/ (SP:24)


def _arm(args):
    return getattr(args, 'sublst_arm', None) or cfg().search_sublst.ARM


def _label(lst, arm):
    return cfg().search_sublst.RUN_LABEL.format(hh=lst[:2], arm=arm, lst=lst)


def _unit_name(arm, cell, stokes, date=None):
    return f'sublst-{arm}_{cell}_{stokes}' + (f'_{date}' if date else '')


def tree_root(arm):
    c = cfg()
    return os.path.join(c.paths.SUBLST_ROOT, c.search_sublst.TREE_SUBDIRS[arm])


def coadd_root(lst):
    c = cfg()
    return os.path.join(c.paths.SUBLST_ROOT, c.search_sublst.COADD_SUBDIR.format(lst=lst))


def expand_units(args):
    """[(cell, lst, band, stokes, date_or_None)] in pass order (LST, night, SEARCH_ORDER, STOKES_ORDER).

    --cells tokens: an LST-band cell 09h_18MHz (every night: the intermittent pass; patterns 09h_* over the 18 bands), a unit
    name 09h_18MHz_V / sublst-B_09h_18MHz_V (that Stokes), or a NIGHT CELL (core.cells.parse_night_cell) 09h_2025-01-02 (every
    band, that night) / 09h_2025-01-02_18MHz (one band, that night) = the per-hour, incremental search. --lsts 09h = every band
    of the LST. --epochs 09h_2025-01-02 (the common per-hour selector, core.cells.expand_epochs) = every band of that night;
    with --cells/--lsts also given, those only filter the bands. --scopes / --stokes / --exclude-cells filter."""
    from exopipe.phase3.core import cells as C
    s = cfg().search_sublst
    bands = tuple(s.BANDS)
    universe = tuple(f'{l}_{b}' for l in C.LSTS for b in bands)
    scopes = set(C.expand(getattr(args, 'scopes', None))) if getattr(args, 'scopes', None) else None
    stokes_all = tuple(s.STOKES_ORDER)
    want_st = [x.strip() for x in str(getattr(args, 'stokes', None) or '').replace('VI', 'V,I').replace('IV', 'I,V')
               .split(',') if x.strip()]
    for x in want_st:
        if x not in ('I', 'V'):
            raise refusal(f'--stokes {x}: I or V')
    excl = set(C.expand(getattr(args, 'exclude_cells', None), universe=universe, kind='cell')) \
        if getattr(args, 'exclude_cells', None) else set()

    def need_epochs():
        if s.EPOCHS_MODE != 'incremental':
            raise refusal(f'per-hour sub-LST search (a night cell or --epochs): search_sublst.EPOCHS_MODE = '
                          f'{s.EPOCHS_MODE!r} (spec G4); "incremental" searches one night of a cell appended to its '
                          f'published attempt')

    def band_ok(tok, band):
        if band not in bands:
            raise refusal(f'{tok}: band {band} is not searched in the sub-LST tier (bands: {", ".join(bands)})')

    items = []           # (cell, stokes or None, date or None)
    for tok in (C.expand(args.cells, universe=universe, kind='sub-LST cell') if getattr(args, 'cells', None) else []):
        m = _CELL_RE.match(tok)
        if m:
            band_ok(tok, m['band'])
            if m['date']:
                need_epochs()
            items.append((f'{m["lst"]}_{m["band"]}', m['stokes'], m['date']))
            continue
        try:
            lst, date, sb = C.parse_night_cell(tok)
        except ValueError:
            raise refusal(f'{tok!r}: not a sub-LST item (<LST>_<band>[_<S>], or a night cell <LST>_<YYYY-MM-DD>[_<band>]; '
                          f'bands {", ".join(bands)})')
        need_epochs()
        if sb:
            band_ok(tok, sb)
        for b in ([sb] if sb else bands):
            items.append((f'{lst}_{b}', None, date))
    if getattr(args, 'lsts', None):
        for lst in C.expand(args.lsts, universe=C.LSTS, kind='LST'):
            for b in bands:
                items.append((f'{lst}_{b}', None, None))
    if getattr(args, 'epochs', None):
        need_epochs()
        filt = {}
        for cell, st, d in items:
            if d is None:
                filt.setdefault(cell, set()).add(st)
        items = [it for it in items if it[2] is not None]
        for tok in C.expand_epochs(args.epochs):
            lst, date = C.parse_epoch(tok)
            for b in bands:
                cell = f'{lst}_{b}'
                if filt and cell not in filt:
                    continue
                for st in sorted(filt.get(cell, {None}), key=str):
                    items.append((cell, st, date))
    if not items:
        raise refusal('tier sublst needs --cells, --lsts or --epochs (nothing selected)')
    out, seen = [], set()
    for cell, st, date in items:
        lst, band = cell.split('_', 1)
        if cell in excl or (scopes and band not in scopes):
            continue
        for S in ([st] if st else stokes_all):
            if want_st and S not in want_st:
                continue
            key = (cell, S, date)
            if key not in seen:
                seen.add(key)
                out.append((cell, lst, band, S, date))
    order = {b: i for i, b in enumerate(s.SEARCH_ORDER)}
    sto = {x: i for i, x in enumerate(s.STOKES_ORDER)}
    out.sort(key=lambda t: (t[1], t[4] or '', order.get(t[2], 99), sto.get(t[3], 9)))
    return out


def _windows(tree_cell, band, stokes):
    keys = []
    for p in sorted(glob.glob(os.path.join(tree_cell, 'subtracted', f'*_{stokes}_sub_{band}.fits'))):
        m = re.search(r'(\d{4}-\d\d-\d\d(?:T\d\d)?)_[IV]_sub_', os.path.basename(p))
        if m:
            keys.append(m.group(1))
    return keys


def _cards(path, *keys):
    from exopipe.phase3.core import record
    c = record.fits_primary_cards(path)
    return {k: c.get(k) for k in keys}


def preflight(args, run, cell, lst, band, stokes, date):
    """The plan of one unit: paths, windows, admitted nights, inputs, and the first gate that stops it (refuse / no-input).
    Reads only headers and directory listings."""
    from exopipe.phase3.core import exclusions
    c = cfg()
    s = c.search_sublst
    arm = _arm(args)
    if arm not in s.TREE_SUBDIRS:
        raise refusal(f'arm {arm}: no tree in search_sublst.TREE_SUBDIRS ({", ".join(s.TREE_SUBDIRS)})')
    label = _label(lst, arm)
    tree = os.path.join(tree_root(arm), cell)
    coadd = os.path.join(coadd_root(lst), cell)
    pub = os.path.join(run.out_root, s.PUBLISH_SUBDIR.format(arm=arm), f'{cell}_{stokes}')
    plan = dict(cell=cell, lst=lst, band=band, stokes=stokes, arm=arm, label=label, tree=tree, coadd=coadd, publish=pub,
                date=date, inputs=[], gates=[], refuse=None, noinput=None, included_dates=None, windows=[],
                admitted=[], excluded_nights=[])

    def gate(name, ok, detail):
        plan['gates'].append((name, bool(ok), detail))
        return ok

    nowin = os.path.join(tree, 'NO_WINDOWS.txt')
    if os.path.exists(nowin):
        with open(nowin) as f:
            first = f.readline().strip()
        plan['noinput'] = f'NO_WINDOWS.txt: {first}'
        gate('tree.has_windows', False, plan['noinput'])
        return plan
    keys = _windows(tree, band, 'I')     # the engine's epoch list: the Stokes I windows (deep_search.py:1019-1022)
    plan['windows'] = keys
    if not gate('E5.epochs_found', bool(keys), f'{len(keys)} windows in {tree}/subtracted'):
        plan['refuse'] = (f'E5: ZERO epochs found under {tree}/subtracted (inputs missing/unshipped) - the engine refuses '
                          f'a 0-epoch cell')
        return plan
    if not gate('E6.min_epochs', len(keys) >= s.MIN_EPOCHS, f'{len(keys)} windows, minimum {s.MIN_EPOCHS}'):
        plan['noinput'] = f'P3-POLICY-SKIP: only {len(keys)} windows (minimum {s.MIN_EPOCHS}, cadence 10min)'
        return plan
    nights = sorted({k[:10] for k in keys})
    ex = exclusions.load()
    plan['admitted'] = ex.admitted(lst, band, nights)
    plan['excluded_nights'] = ex.excluded_constituents(lst, band, nights)
    mI = os.path.join(coadd, 'coadd_final', f'I_deep_Taper_R0_{band}.fits')
    if not gate('E1.master_I', os.path.exists(mI), mI):
        plan['refuse'] = f'E1: no I master at {mI} - inputs missing or unshipped (the engine refuses the silent no-op)'
        return plan
    op = _cards(mI, 'COADDOP')['COADDOP']
    if not gate('E2.coaddop_I', op == s.MASTER_GEN, f'COADDOP={op!r}, required {s.MASTER_GEN!r}'):
        plan['refuse'] = f'E2: {mI} has COADDOP={op!r}, expected {s.MASTER_GEN!r} (superseded master generation)'
        return plan
    mV = os.path.join(coadd, 'coadd_final', f'V_deep_Taper_R0_{band}.fits')
    if os.path.exists(mV):
        opv = _cards(mV, 'COADDOP')['COADDOP']
        if not gate('E2.coaddop_V', opv == s.MASTER_GEN, f'COADDOP={opv!r}, required {s.MASTER_GEN!r}'):
            plan['refuse'] = f'E2: {mV} has COADDOP={opv!r}, expected {s.MASTER_GEN!r}'
            return plan
    elif stokes == 'V':
        gate('E33.master_V', False, f'no V master {mV}')
        plan['refuse'] = (f'E33: no V master at {mV} - no Stokes V residual can be built, and the engine refuses a run in '
                          f'which epochs entered the search and no image row was written (deep_search.py:2083-2091)')
        return plan
    rp = os.path.join(coadd, 'residuals_provisional.csv')
    if not gate('E14.search_tier_table', os.path.exists(rp), rp):
        plan['refuse'] = f'E14: residuals_provisional.csv missing at {rp} - the search-tier gate refuses the cell'
        return plan
    if stokes == 'V':
        nt = s.V110_NULL_TRIALS
        if not gate('G2.v110_null_trials', os.path.exists(nt), nt):
            plan['refuse'] = (f'G2/E30: v110 null-trial table {nt} missing on {socket.gethostname()} - the frozen V FP '
                              f'gate would crash and refuse the cell (K6)')
            return plan
        m = md5_file(nt)
        if not gate('G2.v110_null_trials_md5', m == s.V110_NULL_TRIALS_MD5, f'md5 {m}'):
            plan['refuse'] = f'G2: {nt} md5 {m} != pinned {s.V110_NULL_TRIALS_MD5} (a pin changes only by calibrate install)'
            return plan
        pa = s.V110_PA_NULL
        if os.path.exists(pa):
            m = md5_file(pa)
            if not gate('G2.v110_pa_null_md5', m == s.V110_PA_NULL_MD5, f'md5 {m}'):
                plan['refuse'] = f'G2: {pa} md5 {m} != pinned {s.V110_PA_NULL_MD5}'
                return plan
        else:
            gate('v110_pa_null', not s.PA_NULL_REQUIRED, f'{pa} missing: the PA term of v110 is dropped (B-68, today)')
            if s.PA_NULL_REQUIRED:
                plan['refuse'] = f'G2: {pa} missing and search_sublst.PA_NULL_REQUIRED is True'
                return plan
    if date:                              # per-hour (incremental) mode
        inc = [k for k in keys if k[:10] == date]
        plan['included_dates'] = inc
        if not gate('epoch.windows', bool(inc), f'{len(inc)} windows of {date}'):
            plan['noinput'] = f'no window of {date} in {tree}/subtracted (per-hour item {lst}_{date})'
            return plan
        prev_led = os.path.join(pub, 'transients', f'Run_{label}', 'v_deep_images.csv')
        if not gate('epoch.published_attempt', os.path.exists(prev_led), prev_led):
            plan['refuse'] = (f'per-hour mode appends to the published attempt of {cell} {stokes}; none at {pub} - run the '
                              f'full search of the cell first')
            return plan
        with open(prev_led, newline='') as f:
            done = {r['date'] for r in csv.DictReader(f) if r.get('stokes') == stokes}
        dup = sorted(set(inc) & done)
        if not gate('epoch.not_yet_searched', not dup, f'already in the published image ledger: {dup}'):
            plan['refuse'] = (f'windows {dup} of {date} are already in the published attempt {pub}: a re-search of a night '
                              f'is a full run of the cell (--fresh), never a second append')
            return plan
    # inputs (fingerprinted at run time; listed here for the plan)
    ins = [p for p in sorted(glob.glob(os.path.join(tree, 'subtracted', f'*_{stokes}_sub_{band}.fits')))]
    if stokes == 'V':
        ins += sorted(glob.glob(os.path.join(tree, 'subtracted', f'*_I_sub_{band}.fits')))   # the epoch list is I's
    ins += sorted(glob.glob(os.path.join(tree, 'bin_masters', '*.fits')))
    ins += [os.path.join(tree, 'frames.csv')]
    ins += [mI, os.path.join(coadd, 'coadd_final', f'I_deep_Taper_R075_{band}.fits'), mV,
            os.path.join(coadd, 'coadd_final', f'PB_deep_{band}.fits'),
            os.path.join(coadd, 'coadd_final', 'weights.csv'), rp, c.paths.DATA_DECISIONS,
            os.path.join(phase3_dir(), '..', 'common', 'parent_curve_tables.json')]
    if stokes == 'V':
        ins += [s.V110_NULL_TRIALS, s.V110_PA_NULL]
    ins += [os.path.join(phase3_dir(), x) for x in ENGINE_SOURCES]
    plan['inputs'] = [os.path.normpath(p) for p in ins if os.path.exists(p)]
    return plan


def engine_env(plan, attempt):
    """search.md 7.3 (sub-LST rows): the engine environment, built from config only."""
    s = cfg().search_sublst
    label = plan['label']
    return {
        'XLA_PYTHON_CLIENT_PREALLOCATE': 'true' if s.XLA_PREALLOCATE else 'false',
        'P3_TENMIN_ROOT': tree_root(plan['arm']),
        'P3_COADD_ROOT': coadd_root(plan['lst']),
        'P3_MASTER_GEN': s.MASTER_GEN,
        'P3_TABLE_RECORD_ONLY': '1' if s.TABLE_RECORD_ONLY else None,
        'P3_TABLE_PREFIX': s.TABLE_PREFIX or None,
        'P3_REFINE_OFF': None if s.REFINE_X2 else '1',
        'P3_ARCHIVE_WRITE': '1' if s.ARCHIVE_WRITE else '0',
        'P3_FLOW_ROOT': os.path.join(attempt, 'flow'),
        'P3_STAGING_ROOT': attempt,
        'P3_VRESID_ROOT': os.path.join(attempt, f'Run_{label}', 'vresid'),
        'P3_IRESID_ROOT': os.path.join(attempt, f'Run_{label}', 'iresid'),
        'P3_TENMIN_MIN_ELEVATION': repr(float(s.MIN_ELEVATION_DEG)),
        'P3_MIN_EPOCHS': str(int(s.MIN_EPOCHS)),
        'P3_SIZE_FLOOR_BEAMS': repr(float(s.SIZE_FLOOR_BEAMS)),
        'P3_AXIS_REMOVE': repr(float(s.AXIS_REMOVE)),
        'P3_SANITY_I_X': repr(float(s.SANITY_I_X)),
        'P3_SEARCH_TIER_SIGMA': repr(float(s.SEARCH_TIER_SIGMA)),
        'P3_BEAM_GATE_RATIO': repr(float(s.BEAM_GATE_RATIO)),
        'P3_V110_TIMEOUT': str(int(s.V110_TIMEOUT_S)),
        'P3_TABLE_ELEV': '1' if s.TABLE_ELEV else '0',
        'P3_TABLE_ELEV_BANDS': s.TABLE_ELEV_BANDS,
        'P3_TABLE_XM_EXEMPT': '1' if s.TABLE_XM_EXEMPT else '0',
        'TMPDIR': os.path.join(attempt, 'tmp'),
    }


# The child process (core.legacy.run, module mode: `python -m exopipe.phase3.lib.search.tier_sublst --engine-child SPEC`):
# one call of the engine exactly as SP:44-48 (sys.argv=['sublst']), from the branch, with the legacy roots appended and the
# old-name finder installed (core.env.legacy; MOVE_LOG decision 1). It writes a result record; exit 0 done, 2 refused, 1 error.
# SPEC is the path of a JSON file in the attempt (review 2026-10-13): core.legacy.run writes the argv as the log's header line,
# and an inline JSON carried the refusal markers ('REFUSING') into <cell>.log, so the sublst_search_log content test (SP:51:
# no REFUSING line) refused every completed search. An inline JSON object is still accepted.
def engine_child(argj):
    import traceback
    if argj.lstrip().startswith('{'):
        a = json.loads(argj)
    else:
        with open(argj) as f:
            a = json.load(f)
    sys.argv = ['sublst']

    def _w(o):
        with open(a['result'] + '.tmp', 'w') as f:
            json.dump(o, f)
        os.replace(a['result'] + '.tmp', a['result'])
    try:
        from exopipe.phase3.core import env as _env
        _env.legacy()
        from exopipe.phase3.lib.search.deep_search import run_deep_cell
        kw = dict(label=a['label'], stokes=a['stokes'], cadence=a['cadence'])
        if a['included_dates']:
            kw['included_dates'] = a['included_dates']
        rc = run_deep_cell(a['lst'], a['band'], **kw)
        print('SUBLST-SEARCH-DONE %s %s' % (a['cell'], a['stokes']), flush=True)
        _w({'ok': True, 'rc': rc})
        return 0
    except BaseException as e:
        if isinstance(e, SystemExit) and e.code in (0, None):
            raise
        msg = str(e)
        refused = isinstance(e, RuntimeError) and any(m in msg for m in a['markers'])
        traceback.print_exc()
        sys.stdout.flush()
        _w({'ok': False, 'type': type(e).__name__, 'message': msg[:4000], 'refusal': refused})
        return 2 if refused else 1


def _run_root(attempt, label):
    return os.path.join(attempt, 'transients', f'Run_{label}')


def _read_csv(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return []
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def detect(run, u, plan, attempt):
    """The engine call in a child process (core.legacy.run). Returns the facts dict; raises Refusal / NoInput /
    RuntimeError."""
    c = cfg()
    s = c.search_sublst
    site_python(s.ENGINE_PYTHON)
    os.makedirs(os.path.join(attempt, 'tmp'), exist_ok=True)
    if plan['date']:                     # per-hour: seed the attempt with the published attempt's cumulative records
        src = _run_root(plan['publish'], plan['label'])
        dst = _run_root(attempt, plan['label'])
        os.makedirs(dst, exist_ok=True)
        for name in ('master_catalog.csv', 'v_deep_ledger.csv', 'v_deep_images.csv', 'sigstack', 'refusals'):
            if os.path.exists(os.path.join(src, name)):
                copy_tree(os.path.join(src, name), os.path.join(dst, name))
        u.note(f'per-hour mode: attempt seeded from {src} (catalogue, ledgers, refusal records, sigma-stack maps)')
    log = os.path.join(attempt, f'{plan["cell"]}.log')
    result = os.path.join(attempt, '.engine_result.json')
    a = dict(lst=plan['lst'], band=plan['band'], cell=plan['cell'], label=plan['label'], stokes=plan['stokes'],
             cadence=s.CADENCE, included_dates=plan['included_dates'], markers=list(s.REFUSAL_MARKERS), result=result)
    env = dict(engine_env(plan, attempt), PYTHONPATH=pkg_root())
    u.meta['engine'] = dict(python_key=s.ENGINE_PYTHON, cwd=attempt, log=log,
                            env={k: v for k, v in sorted(env.items()) if v is not None})
    spec = os.path.join(attempt, '.engine_spec.json')      # never inline: the argv is the log's header line (engine_child)
    with open(spec + '.tmp', 'w') as f:
        json.dump(a, f)
    os.replace(spec + '.tmp', spec)
    res_p = legacy_call('exopipe.phase3.lib.search.tier_sublst', ['--engine-child', spec], s.ENGINE_PYTHON, env,
                        log, unit=u, cwd=attempt, module=True)
    u.meta['engine'].update(exit=res_p.rc, wall_s=res_p.wall_s)
    res = None
    if os.path.exists(result):
        with open(result) as f:
            res = json.load(f)
    if res is None:
        raise RuntimeError(f'engine process ended with status {res_p.rc} and no result record (killed? out of memory?); '
                           f'attempt kept: {attempt}; log tail: {log_tail(log)}')
    if not res.get('ok'):
        if res.get('refusal'):
            raise refusal(f'{res["message"]} (engine refusal; attempt kept: {attempt})')
        raise RuntimeError(f'engine {res.get("type")}: {res.get("message")} (attempt kept: {attempt}; log {log})')
    rr = _run_root(attempt, plan['label'])
    imgs = [r for r in _read_csv(os.path.join(rr, 'v_deep_images.csv')) if r.get('stokes') == plan['stokes']]
    if plan['date']:
        imgs = [r for r in imgs if r.get('date', '')[:10] == plan['date']]
    txt = ''
    try:
        with open(log, errors='replace') as f:
            txt = f.read()
    except OSError:
        pass
    summ = _SUMMARY_RE.findall(txt)
    facts = dict(rc=res.get('rc'), image_rows=len(imgs), summary=(' '.join(summ[-1]) if summ else ''))
    u.meta['engine'].update(facts)
    if res.get('rc') == 0 and not imgs:
        skip = re.findall(r'(P3-POLICY-SKIP[^\n]*|INCREMENTAL - nothing admitted[^\n]*)', txt)
        raise no_input(f'the engine searched no image of {plan["cell"]} {plan["stokes"]}: '
                       f'{skip[-1] if skip else "no image row written"} (attempt {attempt})')
    return facts


# ---------------------------------------------------------------------------------------------------------------------
# fit stage (port of exopipe_ops/sublst_fit_positions.py as a sidecar; spec 1.4 row 4, 7.2 'fit')
# ---------------------------------------------------------------------------------------------------------------------

FIT_EXTRA = ['frame', 'frame_sha1', 'fit_frac_px']       # the detection_positions schema: cand_id + DET_COLS + these


def frame_rel(label, r):
    sub = 'iresid' if r['stokes'] == 'I' else 'vresid'
    return f"Run_{label}/{sub}/{r['lst']}_{r['band']}/{r['date']}_{r['stokes']}_redo_resid_{r['band']}.fits"


def cutout_check(d, w, cut_path, ra, dec):
    """'' when the search's own residual cutout of this detection equals the frame's pixels at its location (NaN as 0,
    value for value) and the detection's peak pixel is the cutout's centre (+- 1 px); else the reason (SFP:74-99)."""
    import numpy as np
    from astropy.io import fits
    from astropy.wcs import WCS
    if not os.path.exists(cut_path):
        return 'no search cutout'
    with fits.open(cut_path, memmap=False) as hc:
        cdat = np.squeeze(hc[0].data).astype(np.float64)
        wc = WCS(hc[0].header).celestial
    ra0, de0 = [float(v) for v in wc.pixel_to_world_values(0, 0)]
    x0, y0 = [float(v) for v in w.world_to_pixel_values(ra0, de0)]
    xi, yi = int(round(x0)), int(round(y0))
    if not (abs(x0 - xi) < 1e-3 and abs(y0 - yi) < 1e-3):
        return f'cutout grid is not the frame grid (offset {x0 - xi:+.4f}, {y0 - yi:+.4f} px)'
    ny, nx = cdat.shape
    H, W = d.shape
    ref = np.full((ny, nx), np.nan)
    ys, xs, ye, xe = max(0, yi), max(0, xi), min(H, yi + ny), min(W, xi + nx)
    if ye <= ys or xe <= xs:
        return 'cutout lies outside the frame'
    ref[ys - yi:ye - yi, xs - xi:xe - xi] = d[ys:ye, xs:xe]
    if not np.array_equal(np.nan_to_num(ref), np.nan_to_num(cdat)):
        return (f'cutout pixels differ from the frame (max |diff| '
                f'{float(np.nanmax(np.abs(np.nan_to_num(ref) - np.nan_to_num(cdat)))):.4g})')
    xd, yd = [float(v) for v in w.world_to_pixel_values(ra, dec)]
    if abs((xd - xi) - (nx - 1) / 2.0) > 1.0 or abs((yd - yi) - (ny - 1) / 2.0) > 1.0:
        return f'the detection is not at the cutout centre ({xd - xi:.1f}, {yd - yi:.1f} px in a {nx} x {ny} cutout)'
    return ''


def frac_px(w, ra, dec):
    x, y = [float(t) for t in w.world_to_pixel_values(float(ra), float(dec))]
    return math.hypot(x - round(x), y - round(y))


def _fit_frame(job):
    """One frame (SFP fit_frame): memory-mapped, its LST at MJD-MID, every detection's cutout checked and fitted."""
    root, rel, cutroot, rows, how = job
    import numpy as np
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy.time import Time
    import astropy.units as au
    from exopipe.phase3.core import fitting
    s = cfg().search_sublst
    path = os.path.join(root, rel)
    h1 = hashlib.sha1()
    with open(path, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 22), b''):
            h1.update(blk)
    sha = h1.hexdigest()
    out = []
    with fits.open(path, memmap=True) as hd:
        h = hd[0].header
        d = np.squeeze(hd[0].data)
        w = WCS(h).celestial
        lstd = Time(h['MJD-MID'], format='mjd').sidereal_time('apparent', longitude=s.FIT_LST_LON_DEG * au.deg).deg
        for i, r in rows:
            ra, dec, snr = float(r['ra']), float(r['dec']), float(r['snr'])
            cp_rel = f"transients/Run_{how['label']}/candidates/{r['cand_id']}/{r['cand_id']}_resid.fits"
            chk = cutout_check(d, w, os.path.join(cutroot, cp_rel), ra, dec)
            v, sg = fitting.forced_value_rms(d, w, ra, dec)       # verbatim deep_forced_radio.forced_value_rms (1d5d559)
            if how['function'] == 'legacy':
                import deep_forced_radio as DFR                     # exopipe_ops in place (spec Q11 fallback)
                o = DFR.detection_position(d, w, h, ra, dec, v, sg, snr, r['band'], r['stokes'], lstd,
                                           k_template=tuple(how['k'][r['stokes']]))
            else:
                o = fitting.detection_position(d, w, h, ra, dec, v, sg, snr, r['band'], r['stokes'], lstd,
                                               k=tuple(how['k'][r['stokes']]), tier='sublst')
            for col in s.FIT_BLANKED:
                o[col] = ''
            floor = ''
            for col in ('pos_stat_maj_arcsec', 'pos_stat_min_arcsec'):
                if o['pos_kind'] in ('gaussian', 'template') and o[col] == '0.0':
                    o[col] = s.FIT_STAT_FLOOR
                    floor = f'stat below 0.05 arcsec written as {s.FIT_STAT_FLOOR}'
            fpx = f"{frac_px(w, o['pos_ra'], o['pos_dec']):.4f}" if o['pos_kind'] in ('gaussian', 'template') else ''
            out.append((i, dict(cand_id=r['cand_id'], stokes=r['stokes'], frame=rel, frame_sha1=sha, fit_frac_px=fpx,
                                _cutout=cp_rel, _check=chk or 'equal', _in_frac=frac_px(w, ra, dec), _floor=floor, **o)))
    return out


def fit_sidecar(root, label, out_dir, prev_dir=None, only_dates=None, nproc=None, say=print):
    """Fit every row of <root>/transients/Run_<label>/master_catalog.csv on its frame under <root> (frames of rows not
    re-fitted are taken from prev_dir's sidecar, per-hour mode). Writes <out_dir>/fit_positions.csv and
    fit_positions_summary.json; returns (csv path, json path, summary). Content test F1-F4 refuse."""
    from exopipe.phase3.core import fitting
    c = cfg()
    s = c.search_sublst
    cat = os.path.join(_run_root(root, label), 'master_catalog.csv')
    with open(cat, newline='') as f:
        rd = csv.reader(f)
        hdr = next(rd)
        raw = [x for x in rd]
    if 'pos_kind' in hdr:
        raise refusal(f'{cat} already carries pos_kind - the fit stage fits the peak-position record once')
    ix = {k: hdr.index(k) for k in ('cand_id', 'lst', 'date', 'band', 'stokes', 'ra', 'dec', 'snr')}
    rows = [{k: x[j] for k, j in ix.items()} for x in raw]
    if any(len(x) != len(hdr) for x in raw):
        raise refusal(f'{cat} has rows whose length is not the header\'s')
    prev = {}
    if prev_dir and os.path.exists(os.path.join(prev_dir, 'fit_positions.csv')):
        prev = {r['cand_id']: r for r in _read_csv(os.path.join(prev_dir, 'fit_positions.csv'))}
    k = {S: fitting.k_template('sublst', S) for S in ('I', 'V')}
    how = dict(label=label, function=s.FIT_FUNCTION, k=k)
    if s.FIT_FUNCTION not in ('core', 'legacy'):
        raise refusal(f'search_sublst.FIT_FUNCTION = {s.FIT_FUNCTION!r}: core or legacy')
    byf, missing, res = {}, [], {}
    for i, r in enumerate(rows):
        if only_dates is not None and r['date'] not in only_dates and r['cand_id'] in prev:
            res[i] = dict({kk: prev[r['cand_id']].get(kk, '') for kk in ['cand_id'] + fitting.DET_COLS + FIT_EXTRA},
                          stokes=r['stokes'], _check='equal (previous fit kept)', _floor='', _in_frac=1.0)
            continue
        rel = frame_rel(label, r)
        base = root if os.path.exists(os.path.join(root, rel)) else (prev_dir if prev_dir and
                                                                     os.path.exists(os.path.join(prev_dir, rel)) else None)
        if base is None:
            missing.append(f"{r['cand_id']}: no frame {rel}")
            continue
        byf.setdefault((base, rel), []).append((i, r))
    if missing:
        raise refusal(f'F1: {len(missing)} detection(s) without the frame their search wrote, e.g. {missing[:3]}')
    jobs = sorted(((b, rel, b, rs, how) for (b, rel), rs in byf.items()), key=lambda j: -len(j[3]))
    t0 = time.time()
    n = nproc or s.FIT_NPROC
    if jobs:
        if n > 1 and len(jobs) > 1:
            import multiprocessing as mp
            with mp.get_context('fork').Pool(min(n, len(jobs))) as pool:
                for out in pool.imap_unordered(_fit_frame, jobs, chunksize=1):
                    for i, rec in out:
                        res[i] = rec
        else:
            for j in jobs:
                for i, rec in _fit_frame(j):
                    res[i] = rec
    bad = []
    if sorted(res) != list(range(len(rows))):
        bad.append(f'F2: {len(res)} fitted records for {len(rows)} rows')
    nocut = [res[i]['cand_id'] + ': ' + res[i]['_check'] for i in sorted(res) if not res[i]['_check'].startswith('equal')]
    if nocut:
        bad.append(f'F1: {len(nocut)} detection(s) whose frame is not the frame the search read, e.g. {nocut[:3]}')
    f3 = []
    for i in sorted(res):
        o, r = res[i], rows[i]
        if o['pos_kind'] not in ('gaussian', 'template', 'peak'):
            f3.append(f"{o['cand_id']}: pos_kind {o['pos_kind']!r}")
            continue
        try:
            sm, sn = float(o['pos_stat_maj_arcsec']), float(o['pos_stat_min_arcsec'])
        except ValueError:
            sm = sn = float('nan')
        if not (math.isfinite(sm) and math.isfinite(sn) and sm > 0 and sn > 0):
            f3.append(f"{o['cand_id']}: statistical error {o['pos_stat_maj_arcsec']!r}/{o['pos_stat_min_arcsec']!r}")
        if any(o.get(col, '') != '' for col in s.FIT_BLANKED):
            f3.append(f"{o['cand_id']}: the deep systematic was not blanked")
        if abs(float(o['peak_ra']) - float(r['ra']) % 360.0) > 1e-5 or abs(float(o['peak_dec']) - float(r['dec'])) > 1e-5:
            f3.append(f"{o['cand_id']}: peak_ra/peak_dec {o['peak_ra']},{o['peak_dec']} is not the input {r['ra']},{r['dec']}")
    if f3:
        bad.append(f'F3: {len(f3)} row(s), e.g. {f3[:3]}')
    fp = [float(res[i]['fit_frac_px']) for i in res if res[i].get('fit_frac_px') not in ('', None)]
    on_ctr = sum(1 for x in fp if x < s.FIT_ON_CENTRE_PX)
    frac_on = on_ctr / len(fp) if fp else 0.0
    if fp and frac_on > s.FIT_ON_CENTRE_MAX_FRAC:
        bad.append(f'F4: {on_ctr} of {len(fp)} fitted rows ({100 * frac_on:.2f} per cent) within {s.FIT_ON_CENTRE_PX} px '
                   f'of a pixel centre')
    kinds, per_frame = {}, {}
    for i in res:
        kk = f"{res[i]['stokes']}:{res[i]['pos_kind']}"
        kinds[kk] = kinds.get(kk, 0) + 1
        pf = per_frame.setdefault(res[i]['frame'], [0, 0])
        pf[0] += 1
        pf[1] += res[i]['pos_kind'] == 'peak'
    summ = dict(stage='search fit (sublst sidecar)', catalogue=cat, rows=len(rows), frames=len(byf),
                fit_function=s.FIT_FUNCTION, k_template={S: list(v) for S, v in k.items()}, kinds=dict(sorted(kinds.items())),
                fitted_rows=len(fp), fitted_on_pixel_centre=on_ctr,
                stat_written_at_record_precision=sum(1 for i in res if res[i].get('_floor')),
                cutouts_equal=len(res) - len(nocut),
                input_rows_on_pixel_centre=sum(1 for i in res if res[i].get('_in_frac', 1.0) < s.FIT_ON_CENTRE_PX),
                attempt=root,
                peak_fallback_fraction_per_frame={fr: round(v[1] / v[0], 4) for fr, v in sorted(per_frame.items())},
                peak_fallback_fraction=(round(sum(v[1] for v in per_frame.values()) /
                                              max(1, sum(v[0] for v in per_frame.values())), 4)),
                refitted_dates=sorted(only_dates) if only_dates is not None else 'all', seconds=round(time.time() - t0, 1),
                refused=bad)
    if bad:
        raise refusal('fit content test failed (sidecar not written): ' + '; '.join(bad))
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, 'fit_positions.csv')
    out_json = os.path.join(out_dir, 'fit_positions_summary.json')
    cols = ['cand_id'] + fitting.DET_COLS + FIT_EXTRA
    with open(out_csv + '.part', 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        wr.writeheader()
        wr.writerows(res[i] for i in sorted(res))
    os.replace(out_csv + '.part', out_csv)
    with open(out_json + '.part', 'w') as f:
        json.dump(summ, f, indent=1)
    os.replace(out_json + '.part', out_json)
    say(f"fit: {len(rows)} detections on {len(byf)} frames: {summ['kinds']}; {len(fp)} fitted, {on_ctr} on a pixel "
        f"centre; peak-fallback fraction {summ['peak_fallback_fraction']}")
    return out_csv, out_json, summ


# ---------------------------------------------------------------------------------------------------------------------
# products of an attempt / published dir, by class
# ---------------------------------------------------------------------------------------------------------------------

def products(root, plan):
    """[(path, class, ctx)] of the products under an attempt or published dir (published layout = attempt layout). ctx
    tells the class tests the tier, unit and the master a parent catalogue was made on."""
    label, cell, S, band = plan['label'], plan['cell'], plan['stokes'], plan['band']
    unit = dict(tier=TIER, stokes=S, cell=cell)
    rr = _run_root(root, label)
    out = []
    for name, cls in (('master_catalog.csv', 'detections'), ('v_deep_ledger.csv', 'gate_ledger'),
                      ('v_deep_images.csv', 'image_ledger')):
        p = os.path.join(rr, name)
        if os.path.exists(p):
            out.append((p, cls, unit))
    for p in sorted(glob.glob(os.path.join(rr, 'refusals', '*_excluded.csv'))):
        out.append((p, 'refusal_record', unit))
    for p in sorted(glob.glob(os.path.join(rr, 'sigstack', '*_sigstack.fits'))):
        out.append((p, 'noise_map', unit))
    sub = 'iresid' if S == 'I' else 'vresid'
    for p in sorted(glob.glob(os.path.join(root, f'Run_{label}', sub, cell, f'*_{S}_redo_resid*_{band}.fits'))):
        out.append((p, 'search_frame', unit))
    if os.path.isdir(os.path.join(rr, 'candidates')) and os.listdir(os.path.join(rr, 'candidates')):
        out.append((os.path.join(rr, 'candidates'), 'candidate_stamps', unit))
    master = os.path.join(plan['coadd'], 'coadd_final', f'I_deep_Taper_R075_{band}.fits')
    for cand in (os.path.join(root, 'catdir'), os.path.join(root, 'flow', f'{cell}_10min', 'catdir')):
        g = sorted(glob.glob(os.path.join(cand, '*.gaul.csv')))
        if g:
            out += [(p, 'master_catalog', dict(master=master, tier=TIER)) for p in g]
            break
    for cand in (os.path.join(root, 'v110_out'), os.path.join(root, 'flow', f'{cell}_10min', 'v110_out')):
        p = os.path.join(cand, 'V_top100_fp_gate_p3.csv')
        if os.path.exists(p):
            out.append((p, 'fp_gate_verdicts', dict(unit, catalog=os.path.join(rr, 'master_catalog.csv'))))
            break
    p = os.path.join(root, 'fit_positions.csv')
    if os.path.exists(p):
        out.append((p, 'detection_positions', dict(unit, attempt=root)))
    for p in sorted(glob.glob(os.path.join(root, f'{cell}.log')) + glob.glob(os.path.join(root, f'{cell}_????-??-??.log'))):
        out.append((p, 'sublst_search_log', {}))
    return out


def verify(u, root, plan, label='verify'):
    """The content test of every product under root (precheck: recorded as checks, not as outputs)."""
    from exopipe.phase3.core import validate
    ok = True
    n = 0
    for p, cls, ctx in products(root, plan):
        n += 1
        if not has_tests(cls):
            _untested(u, cls, p)
            ok = False
            continue
        for r in validate.run(cls, p, **ctx):
            u.check(f"{label}:{r['name']}", r['passed'], value=r['value'], detail=r['detail'], cls=cls, path=p)
            ok = ok and r['passed']
    u.check(f'{label}:products_found', n > 0, value=n, detail=f'products under {root}')
    return ok and n > 0


def _publish_files(plan, attempt):
    """[(src file, dst file)] of a publish (PUBLISH_ITEMS + the fit sidecar), rsync -a layout (SP:53-55)."""
    s = cfg().search_sublst
    pub = plan['publish']
    pairs = []
    items = [it.format(label=plan['label'], cell=plan['cell'], arm=plan['arm']) for it in s.PUBLISH_ITEMS]
    items += ['fit_positions.csv', 'fit_positions_summary.json']
    for rel in items:
        src = os.path.join(attempt, rel)
        dst = os.path.join(pub, os.path.basename(rel))
        if plan['date'] and rel == f"{plan['cell']}.log":
            dst = os.path.join(pub, f"{plan['cell']}_{plan['date']}.log")     # a per-hour append keeps the pass's log
        if os.path.isdir(src):
            for dp, _dn, fn in os.walk(src):
                for f in fn:
                    sp = os.path.join(dp, f)
                    pairs.append((sp, os.path.join(dst, os.path.relpath(sp, src))))
        elif os.path.exists(src):
            pairs.append((src, dst))
    return pairs


def publish(u, run, plan, attempt):
    """Copy the attempt's products to <out>/search_<arm>/<cell>_<S>/ (SP:53-55). Returns (pub, rollback) - rollback()
    restores the published dir as it was before this publish (per-hour mode: the published cumulative files are backed up
    under <out>/_superseded/<unit>_<UTC>/ first; new files are removed)."""
    pub = plan['publish']
    pairs = _publish_files(plan, attempt)
    backup = os.path.join(run.out_root, '_superseded', f'{plan["unit"]}_{utc_stamp()}')
    saved, created = [], []
    for src, dst in pairs:
        if os.path.exists(dst):
            b = os.path.join(backup, os.path.relpath(dst, pub))
            os.makedirs(os.path.dirname(b), exist_ok=True)
            shutil.copy2(dst, b)
            saved.append((b, dst))
        else:
            created.append(dst)
        copy_tree(src, dst)

    def rollback():
        for b, dst in saved:
            copy_tree(b, dst)
        for dst in created:
            try:
                os.remove(dst)
            except OSError:
                pass
    u.meta['published'] = dict(dir=pub, files=len(pairs), replaced=len(saved), backup=backup if saved else None)
    return pub, rollback


def _write_report(u, out_dir, root, plan):
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, 'verify_report.json')
    rep = dict(root=root, unit=u.name, tier=TIER, written=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
               tests=[{k: t.get(k) for k in ('name', 'passed', 'value', 'detail', 'path', 'cls')} for t in u.tests])
    with open(p + '.part', 'w') as f:
        json.dump(rep, f, indent=1, default=str)
    os.replace(p + '.part', p)
    return p


def _print_plan(plan, stages, attempt):
    print(f"\n[{plan['unit']}] {plan['cell']} Stokes {plan['stokes']} arm {plan['arm']} label {plan['label']}"
          + (f" per-hour night {plan['date']}" if plan['date'] else ' (every night)'))
    print(f"  stages: {', '.join(stages)}")
    print(f"  tree {plan['tree']}  ({len(plan['windows'])} windows; nights {len({k[:10] for k in plan['windows']})}, "
          f"admitted {len(plan['admitted'])}, excluded now {plan['excluded_nights'] or 'none'})")
    if plan['included_dates']:
        print(f"  included windows: {', '.join(plan['included_dates'])}")
    print(f"  co-add copy {plan['coadd']}")
    for name, ok, det in plan['gates']:
        print(f"  gate {'PASS' if ok else 'STOP'} {name}: {det}")
    if plan['refuse']:
        print(f"  -> REFUSED: {plan['refuse']}")
    elif plan['noinput']:
        print(f"  -> NO-INPUT: {plan['noinput']}")
    else:
        print(f"  inputs: {len(plan['inputs'])} files fingerprinted")
        print(f"  attempt {attempt}")
        print(f"  engine (core.legacy.run, site PYTHON[{cfg().search_sublst.ENGINE_PYTHON!r}]): run_deep_cell("
              f"{plan['lst']!r}, {plan['band']!r}, label={plan['label']!r}, stokes={plan['stokes']!r}, cadence='10min'"
              + (f", included_dates=<{len(plan['included_dates'])} windows>" if plan['included_dates'] else '') + ')')
        print(f"  lease: {cfg().search_sublst.LEASE} ({cfg().search_sublst.MEM_RESERVE_GB} GB per unit)")
        if 'publish' in stages:
            print(f"  publish -> {plan['publish']}" + (' (exists: --fresh moves it to _superseded)'
                                                      if os.path.isdir(plan['publish']) and not plan['date'] else ''))


def _prior_attempt(run, unit):
    from exopipe.phase3.core import record
    rec = record.load_record(run.out_root, run.step, unit)
    if not rec:
        return None, None
    m = rec.get('meta') or {}
    if m.get('attempt') and os.path.isdir(m['attempt']) and not m.get('attempt_removed'):
        return m['attempt'], rec
    pub = (m.get('published') or {}).get('dir')
    if pub and os.path.isdir(pub):
        return pub, rec
    return None, rec


def run_unit(args, run, cell, lst, band, stokes, date, stages, work_root):
    s = cfg().search_sublst
    arm = _arm(args)
    unit = _unit_name(arm, cell, stokes, date)
    replay = getattr(args, 'replay_attempt', None)
    plan = preflight(args, run, cell, lst, band, stokes, date)
    plan['unit'] = unit
    attempt = os.path.join(work_root, s.WORK_SUBDIR.format(arm=arm), f'{cell}_{stokes}_{utc_stamp()}')
    # an intermittent pass starts clean (NOCAT:62-68): a published dir of an earlier search would be merged into silently
    if not date and 'publish' in stages and os.path.isdir(plan['publish']) and not run.fresh and not plan['refuse'] \
            and not plan['noinput']:
        plan['gates'].append(('publish.starts_clean', False, f'{plan["publish"]} exists'))
        plan['refuse'] = (f'{plan["publish"]} holds the published search of an earlier run: a pass starts clean (NOCAT:62-68) '
                          f'- --fresh moves it to {run.out_root}/_superseded/ first, or search one night with --epochs')
    if run.dry_run:
        _print_plan(plan, stages, attempt)
        if replay:
            print(f'  replay attempt (read-only): {replay}')
        u = run.unit(unit, tier=TIER)
        u.status = 'refused' if plan['refuse'] else ('no-input' if plan['noinput'] else 'planned')
        u.reason = plan['refuse'] or plan['noinput'] or ''
        return
    if run.resume and 'detect' in stages and not plan['refuse'] and not plan['noinput']:
        ok, why = run.resume_check(unit, plan['inputs'], plan['admitted'])
        if ok:
            u = run.unit(unit, tier=TIER)
            u.status, u.reason = 'skipped-valid', why
            print(f'[{unit}] resume: {why}')
            return
    with run.unit(unit, tier=TIER, cell=cell, stokes=stokes, arm=arm, label=plan['label'], date=date) as u:
        u.meta.update(stages=stages, tree=plan['tree'], coadd=plan['coadd'], windows=len(plan['windows']),
                      excluded_nights=plan['excluded_nights'], gates=plan['gates'],
                      included_dates=plan['included_dates'])
        for name, ok, det in plan['gates']:
            if ok or not (plan['noinput'] and not plan['refuse']):
                u.check(f'preflight:{name}', ok, detail=det)
        if plan['refuse']:
            u.refuse(plan['refuse'])
        if plan['noinput']:
            raise no_input(plan['noinput'])
        u.set_admitted(plan['admitted'])
        if 'detect' in stages:
            for p in plan['inputs']:
                u.add_input(p)
        if plan['excluded_nights']:
            u.note(f"data_decisions-excluded nights {plan['excluded_nights']}: their windows are dropped inside the engine "
                   f"(E9; printed, written to no refusal file - recorded here, UNACCOUNTED_EPOCHS='record')")
        # ---- where the stages read from ---------------------------------------------------------------------------
        root = None
        if replay:
            if 'detect' in stages or 'publish' in stages:
                u.refuse('--replay-attempt runs fit and/or verify on an existing attempt read-only: drop detect/publish '
                         'from --stages')
            root = os.path.abspath(replay)
            if not os.path.isdir(_run_root(root, plan['label'])):
                u.refuse(f'--replay-attempt {root}: no transients/Run_{plan["label"]}/ (not an attempt of this label)')
            u.add_input(os.path.join(_run_root(root, plan['label']), 'master_catalog.csv'))
        elif 'detect' not in stages:
            root, _rec = _prior_attempt(run, unit)
            if root is None:
                u.refuse(f'stages {",".join(stages)} without detect need the previous detect record of {unit} under '
                         f'{run.out_root}/_records/search/ naming its attempt or published dir (none found)')
            u.note(f'stages read the attempt named by the previous record: {root}')
        lease = None
        if 'detect' in stages:
            wait_free(work_root, s.FAST_MIN_FREE_GB, s.DISK_WAIT_S, f'{unit}')
            lease = make_lease('search_sublst', run.out_root, unit, lst)
            if lease is not None:
                lease.acquire(say=lambda m: print(f'[{unit}] {m}', flush=True))
                u.meta['lease'] = dict(mode=lease.mode, dir=lease.dir, capacity=lease.capacity,
                                       waited_s=round(lease.waited_s, 1))
        try:
            if 'detect' in stages:
                os.makedirs(attempt, exist_ok=True)
                u.meta['attempt'] = attempt
                print(f'[{unit}] detect: attempt {attempt}', flush=True)
                facts = detect(run, u, plan, attempt)
                print(f"[{unit}] detect: {facts['summary'] or 'engine returned ' + str(facts['rc'])}; "
                      f"{facts['image_rows']} image rows", flush=True)
                root = attempt
        finally:
            if lease is not None:
                lease.release()
        own = root == attempt                      # this run's own attempt (else a replay / earlier attempt: read only)
        side = attempt if own else os.path.join(run.out_root, 'replay' if replay else 'restage', unit)
        if 'fit' in stages:
            cat = os.path.join(_run_root(root, plan['label']), 'master_catalog.csv')
            if not os.path.exists(cat):
                # the engine creates master_catalog.csv with its first candidate row (candidates.py:277-282): a search with
                # 0 candidates has none (01h_50MHz V, published 2026-10-08: '3 epochs -> 0 candidates'). Nothing to fit -
                # never an error; but a catalogue missing after the engine reported candidates refuses (review 2026-10-13)
                n_eng = (u.meta.get('engine') or {}).get('rc')
                if n_eng:
                    u.refuse(f'fit: no {cat} although the engine reported {n_eng} candidate(s)')
                u.note(f'fit: no master_catalog.csv under {root} (0 detections) - nothing to fit, no sidecar written')
            else:
                fit_sidecar(root, plan['label'], side, prev_dir=plan['publish'] if date else None,
                            only_dates=set(plan['included_dates']) if date else None,
                            say=lambda m: print(f'[{unit}] {m}', flush=True))
                if not own:
                    record_output(u, os.path.join(side, 'fit_positions.csv'), 'detection_positions', tier=TIER,
                                  stokes=stokes, cell=cell, attempt=root)
        if 'verify' in stages and not (own and date and 'publish' in stages):
            # per-hour mode verifies AFTER the transactional publish (the attempt holds only the new night's frames)
            ok = verify(u, root, plan, label='verify')
            if not own or 'publish' not in stages:
                record_output(u, _write_report(u, side, root, plan), 'sublst_verify_report')
            if not ok:
                u.note(f'verify failed: nothing published; attempt kept: {root}')
                return
        if 'publish' in stages:
            if not own:
                u.refuse('publish copies this run\'s own attempt only')
            if not date and os.path.isdir(plan['publish']):          # --fresh (checked in the plan): move the old one aside
                old = os.path.join(run.out_root, '_superseded', f'{unit}_{utc_stamp()}')
                os.makedirs(os.path.dirname(old), exist_ok=True)
                os.rename(plan['publish'], old)
                u.note(f'--fresh: the earlier published search moved to {old}')
            pub, rollback = publish(u, run, plan, attempt)
            if date and 'verify' in stages:
                if not verify(u, pub, plan, label='verify'):
                    rollback()
                    u.note(f'verify of the published cell failed after the per-hour append: publish rolled back; '
                           f'attempt kept: {attempt}')
                    return
            for p, cls, ctx in products(pub, plan):
                record_output(u, p, cls, **ctx)
            print(f'[{unit}] published -> {pub}', flush=True)
            if not u.failed_tests() and not (getattr(args, 'sublst_keep_attempt', False) or s.KEEP_ATTEMPT_ON_SUCCESS):
                shutil.rmtree(attempt, ignore_errors=True)
                u.meta['attempt_removed'] = True
        elif own:
            for p, cls, ctx in products(attempt, plan):
                record_output(u, p, cls, **ctx)


def run(args, run):
    """Tier sublst: one unit per (cell, Stokes) - see the module docstring."""
    c = cfg()
    check_overrides('search_sublst', TIER)
    if getattr(args, 'inject', None):
        raise refusal('--inject on tier sublst: the injected residual tree is built outside this step (OPS sublst_inject.py, '
                      'not in MOVE_PLAN; spec Q9) - search an injected tree with --sublst-arm BI')
    production_guard(run.out_root, (c.paths.SUBLST_ROOT,), run.dry_run)
    stages = stage_list(args, c.search_sublst.STAGES, tier=TIER)
    units = expand_units(args)
    from exopipe.phase3.core import fitsio
    if run.dry_run:
        work_root = getattr(args, 'work', None) or (c.site.SCRATCH_ROOT or {}).get(socket.gethostname()) \
            or '<node-local scratch: --work or site SCRATCH_ROOT>'
    else:
        work_root = fitsio.scratch_root(getattr(args, 'work', None))
    print(f'search --tier sublst: {len(units)} unit(s), arm {_arm(args)}, stages {",".join(stages)}, out {run.out_root}, '
          f'lease {c.search_sublst.LEASE}', flush=True)
    workers = max(1, int(getattr(args, 'workers', None) or 1))
    jobs = [(args, run, cell, lst, band, S, date, stages, work_root) for cell, lst, band, S, date in units]
    def one(j):
        name = _unit_name(_arm(args), j[2], j[5], j[6])
        try:
            run_unit(*j)
        except Exception as e:                  # recorded as the unit's error (exit 1); the other units still run
            record_failure(run, name, e, tier=TIER)
            print(f'[{name}] ERROR {type(e).__name__}: {e}', flush=True)
    if workers == 1 or run.dry_run:
        for j in jobs:
            one(j)
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(one, jobs))
    if run.dry_run:
        return dry_exit(run, 'unit(s)')
    return None


if __name__ == '__main__':          # the engine child of detect (core.legacy.run module mode); never a user command
    if len(sys.argv) == 3 and sys.argv[1] == '--engine-child':
        sys.exit(engine_child(sys.argv[2]))
    sys.exit('this module is run through `exopipe3 search --tier sublst`')
