"""Shared machinery of `exopipe3 install` (steps/install.py): roots and their re-rooting under --out, the gate helper,
the in-place call context of the production install module, the action log, the manifest, md5 helpers.

During the refactor window the install itself is the production module phase3_install.py, moved by pure git mv to
exopipe/phase3/lib/install/install.py ("PI") and called IN PLACE (DESIGN 11.1). PI reads module globals (ROOT,
INSTALL_ROUND, WQ, QROOT_NAME, RECORD, FREEZE_MARKER, CELL_FILES, CELL_TOP, CELL_DIRS) and environment variables
(P3_ARCHIVE_RUN, P3_ARCHIVE_ROOT, P3_INSTALL_FREE_FLOOR_GB, P3_COADD_ROOT, read at call time by PI and
coadd_full_pipeline); ``pi_context`` sets them from config for one call and restores them afterwards. These are the
only environment writes of the step; each call's settings are recorded in the unit's notes.
"""
import contextlib
import csv
import hashlib
import os
import re
import time

from exopipe.phase3.core.record import Refusal

RCOLS = ['time', 'cell', 'step', 'path', 'dest', 'md5', 'bytes', 'detail']   # PI's action-log columns (phase3_install.py:60)
MCOLS = ['section'] + RCOLS                                                # the attempt manifest: 'plan' rows + 'action' rows
# action-log steps that MOVE or WRITE something in the working tree, the quarantine or the archive: a failure after any of
# them is a half-moved cell (error, `--stages recover`), before them a refusal that leaves the cell untouched
MOVE_STEPS = ('working-quarantine', 'installed', 'cell-file', 'cell-dir', 'epoch-gate-record', 'archive-quarantine',
              'screen-published', 'epoch-derived')

# Gates, in evaluation order (spec section 5). WAIVABLE: evaluated by this step only (not re-evaluated inside PI) and not
# structural; everything evaluated inside PI cannot be waived during the window (PI raises).
GATES = ('H', 'R', 'G-1', 'G-1b', 'G1', 'G1-EXACT', 'CUB', 'RB', 'RB-FILES', 'CELL', 'G-MEM', 'G-PROD', 'G1b', 'Z',
         'G2', 'G3', 'STG', 'G0-GEN', 'G-DEST', 'a0', 'EG', 'G4', 'G0', 'SET', 'b', 'AQ', 'd', 'e', 'V', 'Q')
WAIVABLE = ('H', 'G1-EXACT', 'Q')
NEVER_WAIVABLE = ('G0', 'G0-GEN', 'b', 'a0', 'SET', 'AQ', 'd', 'e', 'G-1b', 'G-DEST')        # DESIGN 11.6
# content tests that are never waivable (digest / set mismatches, excluded constituents)
NEVER_WAIVABLE_TESTS = ('install_manifest.rows_match_files', 'archive_cell.audit', 'installed_cell.exclusion_conformance',
                        'screen_table.install_cell_row', 'screen_table.rows')

# PI RuntimeError text -> the gate it is (phase3_install.py raise sites; coadd_full_pipeline for G0)
_PI_GATE_TEXT = (
    ('P3_ARCHIVE_RUN unset', 'R'), ('freeze is not confirmed', 'G-1'), ('snapshot_masters links resolve', 'G-1b'),
    ('no content-test record', 'G1'), ('content record (D1-D6)', 'CUB'), ('lack MDWINTP', 'CUB'),
    ('needs --rebuild-root with REBUILD_CONTENT_OK', 'RB'), ('master membership violates', 'G-MEM'),
    ('has no coadd_final/', 'G-PROD'), ('does not carry what the rulings require', 'G-PROD'),
    ('no applied-screen record', 'G1b'), ('two generation files map', 'G2'), ('GB needed on the working tree', 'G3'),
    ('.install_staging exists', 'STG'), ('staging copy of', 'a0'), ('IN THE MASTER fail the epoch gate', 'EG'),
    ('fails the epoch content check', 'G4'), ('EXCLUSION CONFORMANCE REFUSED', 'G0'),
    ('working quarantine destination exists', 'G-DEST'), ('archive quarantine destination exists', 'G-DEST'),
    ('screen quarantine destination exists', 'G-DEST'), ('working tree after install != the generation', 'SET'),
    ('archive quarantine move of', 'AQ'), ('archive audit', 'd'), ('published screen copy not verified', 'e'),
)


def cfg():
    from exopipe.phase3.config import CFG
    return CFG


def utcnow():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def pi_gate_of(text):
    """The gate a PI RuntimeError message names (or 'PI' when it names none)."""
    for frag, gate in _PI_GATE_TEXT:
        if frag in text:
            return gate
    return 'PI'


# ---------------------------------------------------------------------------------------------------------------------
# roots
# ---------------------------------------------------------------------------------------------------------------------

class Roots:
    """Where this run reads and writes. Production values (paths.COADD_SB_ROOT, paths.PHASE3_ARCHIVE_ROOT,
    install.WORKING_QUARANTINE_FMT, install.RECORD_CSV, install_enqueue.SEARCH_STATE_ROOT; no --out), or every write
    destination re-rooted under --out: <out>/coadd_sb, <out>/archive, <out>/working_quarantine_<round>,
    <out>/install_record.csv, <out>/p3s5_state. Inputs (generations, rebuild roots, data_decisions.csv, the screen record
    membership.verify_products reads, the snapshot link tree, the freeze marker) are read in place and never written."""

    def __init__(self, out, rerooted):
        C = cfg()
        ci, ce = C.install, C.install_enqueue
        self.out = os.path.abspath(out)
        self.rerooted = bool(rerooted)
        self.run = C.paths.ARCHIVE_RUN
        if self.rerooted:
            self.working = os.path.join(self.out, 'coadd_sb')
            self.archive = os.path.join(self.out, 'archive')
            self.wq_fmt = os.path.join(self.out, 'working_quarantine_{round}')
            self.record_csv = os.path.join(self.out, 'install_record.csv')
            self.state = os.path.join(self.out, 'p3s5_state')
        else:
            self.working = C.paths.COADD_SB_ROOT
            self.archive = C.paths.PHASE3_ARCHIVE_ROOT
            self.wq_fmt = ci.WORKING_QUARANTINE_FMT
            self.record_csv = ci.RECORD_CSV
            self.state = ce.SEARCH_STATE_ROOT
        if not ci.ARCHIVE_QUARANTINE_FMT.startswith('{archive}/'):
            raise Refusal(f'install.ARCHIVE_QUARANTINE_FMT {ci.ARCHIVE_QUARANTINE_FMT!r} must start with "{{archive}}/" '
                          f'(phase3_install names the archive quarantine <archive>/<QROOT_NAME>)')
        self.records = os.path.join(self.out, '_records', 'install')

    # quarantine roots of a round
    def wq(self, rnd):
        return self.wq_fmt.format(round=rnd)

    def aq_name(self, rnd):
        return cfg().install.ARCHIVE_QUARANTINE_FMT[len('{archive}/'):].format(round=rnd)

    def aq(self, rnd):
        return os.path.join(self.archive, self.aq_name(rnd))

    def cdir(self, cell):
        return os.path.join(self.working, cell)

    def masters_dir(self, lst, scope):
        return cfg().install.ARCHIVE_MASTERS_FMT.format(archive=self.archive, run=self.run, lst=lst, scope=scope)

    def night_glob(self, lst, scope):
        return cfg().install.ARCHIVE_NIGHT_GLOB.format(archive=self.archive, run=self.run, lst=lst, scope=scope)

    def screen_dir(self):
        return cfg().install.SCREEN_DIR_FMT.format(archive=self.archive, run=self.run)

    def screen_table(self):
        return os.path.join(self.screen_dir(), cfg().install.SCREEN_TABLE)

    def screen_path(self, lst, scope):
        return os.path.join(self.screen_dir(), cfg().install.SCREEN_FILE_FMT.format(lst=lst, scope=scope))


def roots_for_working(working_root):
    """(working root, archive root) pair of a working root, by the re-rooting convention of ``Roots``."""
    P = cfg().paths
    w = os.path.abspath(working_root)
    if w == os.path.abspath(P.COADD_SB_ROOT):
        return w, P.PHASE3_ARCHIVE_ROOT
    return w, os.path.join(os.path.dirname(w), 'archive')


def roots_for_archive(archive_root):
    P = cfg().paths
    a = os.path.abspath(archive_root)
    if a == os.path.abspath(P.PHASE3_ARCHIVE_ROOT):
        return P.COADD_SB_ROOT, a
    return os.path.join(os.path.dirname(a), 'coadd_sb'), a


def is_rerooted(args_out):
    """--out given and not the production working root itself."""
    if not args_out:
        return False
    return os.path.abspath(args_out) != os.path.abspath(cfg().paths.COADD_SB_ROOT)


# ---------------------------------------------------------------------------------------------------------------------
# start-of-run checks
# ---------------------------------------------------------------------------------------------------------------------

# shared roots the install derives every production destination from (decision 3): an override of any of them would let the
# default --out (paths.COADD_SB_ROOT) leave the production guard while other destinations stayed production
SHARED_ROOT_KEYS = ('paths.COADD_SB_ROOT', 'paths.PHASE3_ARCHIVE_ROOT', 'paths.ARCHIVE_RUN', 'paths.OPS_ROOT')


def check_protected_overrides():
    """Refuse an override of the shared roots install writes under (the config's PROTECTED_KEYS already refuses the install
    sections' own protected keys at --set time)."""
    C = cfg()
    for o in C.overrides:
        base = o[0].split('[', 1)[0]
        if base in SHARED_ROOT_KEYS:
            raise Refusal(f'{base} is a production root install derives its destinations from: exopipe3 install refuses an '
                          f'override of it (re-root every write with --out instead)')


def check_window_consistency(PI):
    """Keys whose value the in-place phase3_install HARD-CODES: an override cannot be honoured during the window."""
    C = cfg().install
    fixed = {'GEN_TOKENS': ({'mdewarp': '_mdewarp', 'rebuild': ''}, 'canon(), phase3_install.py:89-90'),
             'OMIT_VCLEAN_PATTERN': ('V_deep_Taper_R0_clean', 'phase3_install.py:267'),
             'EPOCH_MASTER_STEM': ({'I': 'I_deep_Taper_R075', 'V': 'V_deep_Taper_R0'}, 'phase3_install.py:326'),
             'STAGING_SPACE_FACTOR': (1.05, 'phase3_install.py:279'),
             'STAGING_DIRNAME': ('.install_staging', 'phase3_install.py:288'),
             'ARCHIVE_SKIP_TOKENS': (('_msub',), 'phase3_install.py:393'),
             'AUDIT_CELL_FILES': (('weights.csv', 'night_image_gate.csv', 'epoch_gate.csv', 'stage2_verdict.csv'),
                                  'phase3_install.py:526'),
             'SNAPSHOT_MASTERS_LINK_TREE': ('/lustre/pipeline/exopipe/phase3/snapshot_masters', 'phase3_install.py:203'),
             'ARCHIVE_MASTERS_FMT': ('{archive}/Coadd/{run}/{lst}/{scope}', 'phase3_install.py:375'),
             'ARCHIVE_NIGHT_GLOB': ('{archive}/{lst}/*/{run}/{scope}/[IV]/deep/*.fits', 'phase3_install.py:393'),
             'SCREEN_DIR_FMT': ('{archive}/mdewarp_screens/{run}', 'phase3_install.py:131-132'),
             'SCREEN_TABLE': ('mdewarp_screens.csv', 'phase3_install.py:137'),
             'SCREEN_FILE_FMT': ('{lst}_{scope}_warp_screen.fits', 'phase3_install.py:151'),
             'SCREEN_STATUS_MAXLEN': (120, 'phase3_install.py:185')}
    bad = [f'install.{k} = {getattr(C, k)!r} (phase3_install uses {v!r}, {src})' for k, (v, src) in fixed.items()
           if getattr(C, k) != v]
    try:
        from exopipe.phase3.lib.dewarp import master_dewarp as MD
        dw = getattr(cfg().sections().get('dewarp'), 'RESAMPLE_METHOD', None)
        if dw is not None and dw != MD.RESAMPLE_METHOD:
            bad.append(f'dewarp.RESAMPLE_METHOD = {dw!r} (phase3_install\'s cubic gate reads master_dewarp.RESAMPLE_METHOD '
                       f'{MD.RESAMPLE_METHOD!r}, phase3_install.py:217-222)')
    except ImportError:
        pass
    if not C.REQUIRE_FREEZE_MARKER:
        bad.append('install.REQUIRE_FREEZE_MARKER = False (phase3_install checks the marker unconditionally, '
                   'phase3_install.py:199-200; B-14 (a) takes effect at absorption)')
    if bad:
        raise Refusal('refactor window: the in-place phase3_install cannot honour these values: ' + '; '.join(bad))


def check_waivers(run_obj):
    """--waive names: gates of this step must be WAIVABLE; never-waivable content tests refused (DESIGN 11.6)."""
    for test, _ in run_obj.waivers:
        if test.startswith('install.'):
            g = test[len('install.'):]
            if g not in GATES:
                raise Refusal(f'--waive {test}: no install gate {g!r} (gates: {", ".join(GATES)})')
            if g in NEVER_WAIVABLE:
                raise Refusal(f'--waive {test}: gate {g} is never waivable (DESIGN 11.6: excluded constituent, digest or '
                              f'set mismatch, freeze verification, quarantine collision)')
            if g not in WAIVABLE:
                raise Refusal(f'--waive {test}: gate {g} is evaluated inside phase3_install (called in place during the '
                              f'refactor window) and cannot be waived until absorption; waivable: {", ".join(WAIVABLE)}')
        elif test in NEVER_WAIVABLE_TESTS:
            raise Refusal(f'--waive {test}: content test {test} is never waivable (DESIGN 11.6)')


# ---------------------------------------------------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------------------------------------------------

class Gates:
    """Record each gate as a check ``install.<GATE>`` on the unit; a failed gate refuses (Refusal) unless --waive covers a
    WAIVABLE gate. With collect=True (dry run) failures are collected so every verdict is printed; ``finish()`` then
    refuses with all of them."""

    def __init__(self, u, run_obj, collect=False, echo=None):
        self.u = u
        self.run = run_obj
        self.collect = collect
        self.failed = []
        self.echo = echo

    def check(self, gate, ok, detail='', value=None):
        name = f'install.{gate}'
        ok = bool(ok)
        waived = None
        if not ok and gate in WAIVABLE and gate not in NEVER_WAIVABLE:
            waived = self.run.waived(name)
        self.u.check(name, ok, value=value, detail=str(detail)[:2000])
        if self.echo:
            self.echo(f"  gate {gate:9s} {'PASS' if ok else ('WAIVED' if waived else 'REFUSE')}  {str(detail)[:300]}")
        if ok or waived:
            if waived:
                self.u.note(f'gate {gate} waived: {waived}')
            return True
        if self.collect:
            self.failed.append((gate, detail))
            return False
        raise Refusal(f'{gate}: {detail}')

    def finish(self):
        if self.failed:
            raise Refusal('; '.join(f'{g}: {str(d)[:300]}' for g, d in self.failed))


# ---------------------------------------------------------------------------------------------------------------------
# in-place call context
# ---------------------------------------------------------------------------------------------------------------------

def pi_module():
    from exopipe.phase3.lib.install import install as PI
    return PI


@contextlib.contextmanager
def pi_context(PI, roots, rnd=None):
    """Set PI's module globals and the environment it and coadd_full_pipeline read, for one call; restore afterwards.
    Yields the settings (recorded by the caller)."""
    C = cfg().install
    g = {'ROOT': roots.working, 'RECORD': roots.record_csv, 'FREEZE_MARKER': C.FREEZE_MARKER,
         'CELL_FILES': tuple(C.CELL_FILES), 'CELL_TOP': tuple(C.CELL_TOP), 'CELL_DIRS': tuple(C.CELL_DIRS)}
    if rnd:
        g.update(INSTALL_ROUND=rnd, WQ=roots.wq(rnd), QROOT_NAME=roots.aq_name(rnd))
    env = {'P3_ARCHIVE_RUN': roots.run, 'P3_ARCHIVE_ROOT': roots.archive, 'P3_INSTALL_FREE_FLOOR_GB': repr(float(C.FREE_FLOOR_GB)),
           'P3_COADD_ROOT': roots.working}
    if rnd:
        env['P3_INSTALL_ROUND'] = rnd
    saved_g = {k: getattr(PI, k) for k in g}
    saved_e = {k: os.environ.get(k) for k in env}
    try:
        for k, v in g.items():
            setattr(PI, k, v)
        os.environ.update(env)
        yield dict(globals=g, env=env)
    finally:
        for k, v in saved_g.items():
            setattr(PI, k, v)
        for k, v in saved_e.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@contextlib.contextmanager
def captured_stdout():
    """Capture what in-place modules print (PI, coadd) so it goes into the unit notes, not mixed into the plan output."""
    import io
    import sys
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        yield buf
    finally:
        sys.stdout = old


# ---------------------------------------------------------------------------------------------------------------------
# action log and manifest
# ---------------------------------------------------------------------------------------------------------------------

def log_offset(path):
    try:
        return os.path.getsize(path)
    except FileNotFoundError:
        return 0


def action_rows(path, offset, cell):
    """Rows of the action log appended after byte ``offset`` for ``cell`` (other cells may append concurrently)."""
    import io
    if not os.path.exists(path):
        return []
    with open(path, newline='') as f:
        f.seek(offset)
        text = f.read()
    rows = list(csv.DictReader(io.StringIO(text, newline=''), fieldnames=RCOLS))
    return [r for r in rows if r.get('cell') == cell and r.get('time') != 'time']


def write_manifest(path, plan_rows, act_rows=()):
    """The attempt manifest: plan rows (written BEFORE staging) and the action-log rows of the attempt. Atomic."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f'{path}.tmp.{os.getpid()}'
    with open(tmp, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=MCOLS, extrasaction='ignore')
        w.writeheader()
        for r in plan_rows:
            w.writerow(dict(r, section='plan'))
        for r in act_rows:
            w.writerow(dict(r, section='action'))
    os.replace(tmp, path)
    return path


def keep_previous(path):
    """Move an earlier attempt's file aside (<name>.<mtime UTC>), never overwrite it."""
    if os.path.exists(path):
        stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime(os.path.getmtime(path)))
        base, ext = os.path.splitext(path)
        dst = f'{base}.{stamp}{ext}'
        n = 1
        while os.path.exists(dst):
            dst = f'{base}.{stamp}_{n}{ext}'
            n += 1
        os.rename(path, dst)
        return dst
    return None


# ---------------------------------------------------------------------------------------------------------------------
# misc
# ---------------------------------------------------------------------------------------------------------------------

def md5(path, block=None):
    block = block or cfg().install.MD5_BLOCK_BYTES
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(block), b''):
            h.update(b)
    return h.hexdigest()


def md5_many(paths, threads=None):
    """{path: md5 or None (missing)} read in parallel (I/O bound)."""
    from concurrent.futures import ThreadPoolExecutor

    def one(p):
        try:
            return p, md5(p)
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            return p, None
    paths = list(dict.fromkeys(paths))
    if not paths:
        return {}
    with ThreadPoolExecutor(max_workers=max(1, min(threads or cfg().install.MD5_THREADS, len(paths)))) as ex:
        return dict(ex.map(one, paths))


def host():
    """This host's short name, the key of the site tables (core.env.host: the domain part stripped)."""
    from exopipe.phase3.core import env
    return env.host()


def local_to_host(path):
    """True when ``path`` lies under a node-local root of this host (site NODE_LOCAL_ROOTS, core.env.node_local_holder)."""
    from exopipe.phase3.core import env
    return env.node_local_holder(path) == env.host()


def nearest_existing(path):
    p = os.path.abspath(path)
    while not os.path.exists(p) and p != '/':
        p = os.path.dirname(p)
    return p


_ROUND_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_-]*$')


def valid_round(name):
    return bool(_ROUND_RE.match(name or ''))


def admitted_in_master(weights_csv, lst, scope):
    """In-master nights of a weights.csv admitted by data_decisions (core.exclusions): the unit's admitted night set."""
    from exopipe.phase3.core import exclusions
    try:
        with open(weights_csv) as f:
            dates = [r['date'] for r in csv.DictReader(f) if r.get('in_master') == '1']
    except FileNotFoundError:
        return []
    return exclusions.load().admitted(lst, scope, dates)


def adopt_record(u, rec, reason):
    """--resume: the unit's earlier record is valid now (record.resume_check passed, outputs re-tested): carry its inputs,
    admitted nights, outputs and tests into this run's record without redoing the work."""
    u.inputs.update(rec.get('inputs') or {})
    if rec.get('admitted') is not None:
        u.set_admitted(rec['admitted'])
    for o in rec.get('outputs') or []:
        u.add_output(o['path'], o['class'], test=False)
    for t in rec.get('tests') or []:
        u.tests.append(t)
    u.meta.update(rec.get('meta') or {})
    u.note(f'resumed (--resume): {reason}; record of run {rec.get("run_id")}')
