"""Cutout-generation campaigns: `exopipe3 photometry --tier snapshot --campaign` (config photometry_campaign).

A campaign regenerates the 10-s photometry cutouts of a whole generation from the rescaled flow cubes and measures its stage 1,
so a future generation runs in v2 without dated drivers. Port of the generation-6 drivers (exopipe_ops/gen6_20261117/; plan
refactor_20261012/plan/GEN6_PORT_PLAN_20261123.md pull request 4):

  generation()       the generation written, photometry_campaign.GENERATION: the cutout root (CUTOUT_ROOT_FMT) and the stage-1
                     directory under --out (STAGE1_DIR_FMT) derive from it. Never the served generation (CP-GEN): a campaign
                     writes a new generation; serving it is the serve step's switch.
  enumerate_units()  the campaign's (cell, band) units from the flow cubes (gen6_build_jobs_DRAFT.py :46-104): every cube
                     directory <cell> (one level, os.listdir; no walk) and every subband with both Stokes cubes. Excluded when
                     core/exclusions (the snapshot tiers' per-night read, fresh) OR the subband engine's own reader
                     (excluded_subbands, read by cutouts_probe.py) says so; HELD when the two readers disagree, only one Stokes
                     cube exists, the cell's cube MANIFEST has no COMPLETE line, or a cube has no rescale record
                     (lib/photometry/fluxscale.records). The widebands Red, Green, Blue of every cell with a subband unit (the
                     group search refuses an excluded or member-less group itself, gen6_wb_regen_DRAFT.sh).
  CodePin            the engine code pinned at the campaign start (gen6_cutout_regen_DRAFT.sh :40, :51-53): sha1 of the engines and
                     the files beside them; the modules each engine child imported join the pin when first seen (the unit
                     records' legacy_code). A change refuses that unit and every later one (CP-PIN).
  space()            the run user's quota headroom (bytes and files, `lfs quota -q -u <user>`) and the file system's free space
                     (the filesystem_summary line of `lfs df`), before each unit (gen6_space_ok.sh). Below a floor, or unreadable:
                     the unit refuses and every later one (CP-SPACE).
  io_guard_wait()    the recorders' I/O guard: wait while it is raised, before each unit and each stage-1 cell (a yield).
  readiness()        per cell and band kind (subbands 'sb', widebands 'wb'): stage 1 runs only when every cutouts unit of the kind
                     has a terminal record - done, or no input for a known reason (C-EXCLUDED; C-NOCUT whose engine counts all
                     promise no file: the night rule, nothing in view). A refused, failed or missing unit, or a held subband,
                     HOLDS THAT CELL'S KIND ONLY (SM-HELD); the other cells go on. A kind whose units all wrote nothing ends
                     with the outcome 'no wideband' / 'no subband' (SM-NOBAND, no input; nothing measured) - not done, not
                     failed (gen6_stage1_wb_no_wideband_20261122.py: 01h, 02h, 04h, 08h of 2024-12-26).
  check()            the campaign's complete-when content test, one command anyone can run (read only):
                     python -m exopipe.phase3.lib.photometry.campaign check --out <snapshot forced root> [--cells ...]
                     every enumerated unit has a done record whose recorded checks passed and whose manifest's files exist with
                     their sizes (--retest: the manifest's content tests run again), or a terminal no-input record; every
                     cell's stage 1 per kind is done (manifest tests MM1 + MM2) or 'no <kind>'.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

from exopipe.phase3.core import record
from exopipe.phase3.core.record import NoInput, Refusal
from exopipe.phase3.lib.photometry import steps_common as K

STOKES = ('V', 'I')
KIND_NAME = {'sb': 'subband', 'wb': 'wideband'}


def _cc():
    return K.cfg().photometry_campaign


# ---------------------------------------------------------------------------------------------------------------------
# the generation written (one config value)
# ---------------------------------------------------------------------------------------------------------------------

def generation():
    """photometry_campaign.GENERATION, refused (CP-GEN) unless it names a generation (GENERATION_RE) other than the served
    one (photometry.GENERATION)."""
    g, served = _cc().GENERATION, K.cfg().photometry.GENERATION
    if not isinstance(g, str) or not re.match(_cc().GENERATION_RE, g):
        raise Refusal('CP-GEN: photometry_campaign.GENERATION %r does not name a generation (%s)' % (g, _cc().GENERATION_RE))
    if g == served:
        raise Refusal('CP-GEN: photometry_campaign.GENERATION %s is the served generation (photometry.GENERATION) - a campaign '
                      'writes a new generation; a regeneration of the served one is --in-place' % g)
    return g


def cutout_root():
    return _cc().CUTOUT_ROOT_FMT.format(generation=generation())


def stage1_dir():
    return _cc().STAGE1_DIR_FMT.format(generation=generation())


def kind_bands(kind):
    from exopipe.phase3.lib.photometry import steps_snapshot as SS
    return SS.kind_bands(kind)


def kind_of(band):
    return 'sb' if 'MHz' in band else 'wb'


# ---------------------------------------------------------------------------------------------------------------------
# enumeration
# ---------------------------------------------------------------------------------------------------------------------

class Enumeration:
    """The campaign's units [(cell, band)], subbands then widebands per cell; excluded [{cell, subband, core, engine}]; held
    [{cell, subband, reason}] (not cut, and they hold their cell's subband stage 1); cells (those with a unit or a held
    subband); counts of the cube root."""

    def __init__(self):
        self.units, self.excluded, self.held, self.cells = [], [], [], []
        self.cube_root, self.n_cube_dirs, self.n_complete, self.engine_read = None, 0, 0, False

    def to_dict(self):
        return dict(cube_root=self.cube_root, n_cube_dirs=self.n_cube_dirs, n_complete=self.n_complete,
                    engine_read=self.engine_read, cells=self.cells, units=[list(u) for u in self.units],
                    excluded=self.excluded, held=self.held)

    @classmethod
    def from_dict(cls, d):
        e = cls()
        for k in ('cube_root', 'n_cube_dirs', 'n_complete', 'engine_read', 'cells', 'excluded', 'held'):
            setattr(e, k, d.get(k, getattr(e, k)))
        e.units = [tuple(u) for u in d.get('units', [])]
        return e

    def bands_of(self, cell, kind):
        return [b for c, b in self.units if c == cell and kind_of(b) == kind]

    def held_of(self, cell):
        return [h for h in self.held if h['cell'] == cell]


def _fluxscale():
    try:
        from exopipe.phase3.lib.photometry import fluxscale
    except ImportError as e:
        raise Refusal('CP-ENUM: the rescale records reader (lib/photometry/fluxscale.py) is not in this tree (%s) - the '
                      'campaign cannot list its units' % e)
    return fluxscale


def cube_name(cell, sb, stokes):
    return '%s_%s_%s_flowcube.fits' % (cell, sb, stokes)


def cube_cells(cube_root):
    """The night cells with a directory under the cube root (one os.listdir; names <LST>_<date>)."""
    from exopipe.phase3.core import cells as C
    out = []
    for n in sorted(os.listdir(cube_root)):
        try:
            C.parse_epoch(n)
        except ValueError:
            continue
        if os.path.isdir(os.path.join(cube_root, n)):
            out.append(n)
    return out


def enumerate_units(cells=None, cube_root=None, rec=None, ex=None, engine_excluded=None, complete=None):
    """The campaign's Enumeration. ``cells``: restrict to these night cells (a named cell without a cube directory is held);
    ``rec``: the rescale records {cube path: record} (default fluxscale.records()); ``ex``: the exclusion read (default
    core/exclusions.for_tier('snapshot', fresh=True)); ``engine_excluded``: {cell: subbands the subband engine's reader
    excludes} (cutouts_probe.py; None: not read - core/exclusions alone, the read-only listing of check());
    ``complete``: cell -> its cube MANIFEST has a COMPLETE line (default fluxscale._manifest_complete)."""
    from exopipe.phase3.core import exclusions
    root = cube_root or K.cfg().paths.SNAPSHOT_CUBES_ROOT
    if rec is None:
        fx = _fluxscale()
        try:
            rec = fx.records()
        except fx.RecordsError as e:
            raise Refusal('CP-ENUM: %s' % e)
    if complete is None:
        fx = _fluxscale()
        complete = lambda cell: fx._manifest_complete(cell, root)      # noqa: E731
    ex = ex or exclusions.for_tier('snapshot', fresh=True)
    e = Enumeration()
    e.cube_root, e.engine_read = root, engine_excluded is not None
    present = cube_cells(root)
    e.n_cube_dirs = len(present)
    todo = present if cells is None else [c for c in cells if c in set(present)]
    for c in ([] if cells is None else [c for c in cells if c not in set(present)]):
        e.held.append(dict(cell=c, subband='*', reason='no cube directory under %s' % root))
    wb = kind_bands('wb')
    for cell in todo:
        lst, date = cell.split('_', 1)
        names = set(os.listdir(os.path.join(root, cell)))
        done_cell = bool(complete(cell))
        e.n_complete += done_cell
        if engine_excluded is not None and cell not in engine_excluded:
            raise Refusal('CP-ENUM: the subband engine\'s exclusion reader returned nothing for %s' % cell)
        n_sub = 0
        for sb in kind_bands('sb'):
            have = [S for S in STOKES if cube_name(cell, sb, S) in names]
            if not have:
                continue
            if len(have) != len(STOKES):
                e.held.append(dict(cell=cell, subband=sb, reason='only the Stokes %s cube' % '+'.join(have)))
                continue
            core = bool(ex.is_excluded(lst, sb, date))
            eng = None if engine_excluded is None else sb in engine_excluded[cell]
            if eng is not None and eng != core:
                e.held.append(dict(cell=cell, subband=sb, reason='PHOT-EXCL-READERS: core/exclusions %s, the subband engine\'s '
                                   'excluded_subbands %s' % (core, eng)))
                continue
            if core or eng:
                e.excluded.append(dict(cell=cell, subband=sb, core=core, engine=eng))
                continue
            miss = [S for S in STOKES if os.path.normpath(os.path.join(root, cell, cube_name(cell, sb, S))) not in rec]
            if not done_cell or miss:
                e.held.append(dict(cell=cell, subband=sb, reason='; '.join(
                    (['cube MANIFEST without a COMPLETE line'] if not done_cell else [])
                    + (['no rescale record of the %s cube' % '+'.join(miss)] if miss else []))))
                continue
            e.units.append((cell, sb))
            n_sub += 1
        if n_sub:
            e.units.extend((cell, b) for b in wb)
    e.cells = sorted({c for c, _ in e.units} | {h['cell'] for h in e.held})
    return e


def engine_exclusions(run, cells):
    """{cell: set of subbands} the subband engine's own exclusion reader excludes (cutouts_probe.py 'run' mode, one child for
    every cell: steps_cutouts.EngineReads). A failed read refuses (CP-ENUM)."""
    from exopipe.phase3.lib.photometry import steps_cutouts as SC
    reads = SC.EngineReads(run, list(cells), [], False, [])
    if reads.error:
        raise Refusal('CP-ENUM: %s' % reads.error)
    bad = [c for c in cells if 'error' in reads.cells.get(c, {'error': 1})]
    if bad:
        raise Refusal('CP-ENUM: the subband engine\'s exclusion reader failed for %d cells (e.g. %s)' % (len(bad), bad[0]))
    return {c: set(reads.cells[c]['excluded']) for c in cells}


# ---------------------------------------------------------------------------------------------------------------------
# the code pin, the space gate, the I/O guard
# ---------------------------------------------------------------------------------------------------------------------

def pin_files():
    """The engine files the campaign pins: photometry_campaign.PIN_ENGINES and the PIN_SIBLINGS beside the subband engine."""
    cc = _cc()
    files = [K.engine(k) for k in cc.PIN_ENGINES]
    base = os.path.dirname(K.engine('cutouts_subband'))
    for n in cc.PIN_SIBLINGS:
        p = os.path.join(base, n)
        if not os.path.isfile(p):
            raise Refusal('CP-PIN: %s (photometry_campaign.PIN_SIBLINGS) is not beside the subband engine in %s' % (n, base))
        files.append(p)
    return [os.path.abspath(f) for f in files]


class CodePin:
    """{file: sha1[:12]} of the engine code at the campaign start (K.code_sha12, the fingerprint core/legacy.run records)."""

    def __init__(self, files):
        self.files = {}
        for f in files:
            h = K.code_sha12(f)
            if h is None:
                raise Refusal('CP-PIN: the engine file %s cannot be read at the campaign start' % f)
            self.files[os.path.abspath(f)] = h
        self.started = dict(self.files)

    def changed(self):
        """'file: pinned -> now' of every pinned file whose bytes changed since the pin."""
        out = []
        for f, h in sorted(self.files.items()):
            now = K.code_sha12(f)
            if now != h:
                out.append('%s: %s -> %s' % (f, h, now or 'unreadable'))
        return out

    def absorb(self, fingerprint):
        """Add a unit's child fingerprint {file: sha1[:12]} (its record's legacy_code): a file first seen joins the pin; a
        file the pin holds in another version, or one the unit's children ran in two versions ('mixed:'), is returned."""
        bad = []
        for f, h in sorted((fingerprint or {}).items()):
            f = os.path.abspath(f)
            if str(h).startswith('mixed:'):
                bad.append('%s ran in two versions in one unit (%s)' % (f, h))
            elif f not in self.files:
                self.files[f] = h
            elif self.files[f] != h:
                bad.append('%s: pinned %s, the unit ran %s' % (f, self.files[f], h))
        return bad


def run_user():
    """The user this process runs as (the quota that the products count against)."""
    import pwd
    return pwd.getpwuid(os.geteuid()).pw_name


def _lfs(argv):
    cc = _cc()
    p = subprocess.run([cc.LFS] + argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True,
                       timeout=cc.LFS_TIMEOUT_S)
    return p.returncode, p.stdout, p.stderr


def parse_quota(text, fs):
    """{'kbytes', 'kbytes_limit', 'files', 'files_limit'} of `lfs quota -q -u <user> <fs>` (KiB; a limit None: none set - the
    smaller non-zero of the soft and hard limits otherwise). None when the line is not there."""
    toks = text.split()
    if fs not in toks:
        return None
    v = toks[toks.index(fs) + 1:toks.index(fs) + 9]
    if len(v) < 8:
        return None
    try:
        kb, kq, kl, fl, fq, fll = (int(x.rstrip('*')) for x in (v[0], v[1], v[2], v[4], v[5], v[6]))
    except ValueError:
        return None
    lim = lambda a, b: min([x for x in (a, b) if x > 0] or [0]) or None     # noqa: E731
    return dict(kbytes=kb, kbytes_limit=lim(kq, kl), files=fl, files_limit=lim(fq, fll))


def parse_df(text):
    """The available KiB of the filesystem_summary line of `lfs df <fs>` (its fourth field); None without that line."""
    for ln in text.splitlines():
        f = ln.split()
        if f and f[0] == 'filesystem_summary:' and len(f) >= 4:
            try:
                return int(f[3])
            except ValueError:
                return None
    return None


def space(lfs=None):
    """(ok, text, values): the run user's quota headroom (bytes, files) and the file system's free bytes against the floors of
    photometry_campaign. Unreadable is not ok. ``lfs``: argv -> (rc, stdout, stderr) (tests)."""
    cc = _cc()
    lfs = lfs or _lfs
    fs, user = cc.SPACE_FILESYSTEM, run_user()
    try:
        rq, oq, eq = lfs(['quota', '-q', '-u', user, fs])
        rd, od, ed = lfs(['df', fs])
    except (OSError, subprocess.SubprocessError) as e:
        return False, 'lfs quota / lfs df could not run (%s: %s)' % (type(e).__name__, e), {}
    q = parse_quota(oq, fs) if rq == 0 else None
    av = parse_df(od) if rd == 0 else None
    if q is None or av is None:
        return False, 'lfs quota -u %s %s / lfs df %s unreadable (exit %s, %s; %s, %s)' % (
            user, fs, fs, rq, (eq or oq).strip()[:200], rd, (ed or 'no filesystem_summary line').strip()[:200]), {}
    vals = dict(user=user, filesystem=fs, quota_free_bytes=None if q['kbytes_limit'] is None else
                (q['kbytes_limit'] - q['kbytes']) * 1024, quota_free_files=None if q['files_limit'] is None else
                q['files_limit'] - q['files'], fs_free_bytes=av * 1024)
    low = []
    if vals['quota_free_bytes'] is not None and vals['quota_free_bytes'] < cc.QUOTA_MIN_FREE_BYTES:
        low.append('quota headroom of %s %.2f TiB < %.2f TiB' % (user, vals['quota_free_bytes'] / 2 ** 40,
                                                                 cc.QUOTA_MIN_FREE_BYTES / 2 ** 40))
    if vals['quota_free_files'] is not None and vals['quota_free_files'] < cc.QUOTA_MIN_FREE_FILES:
        low.append('quota headroom of %s %d files < %d files' % (user, vals['quota_free_files'], cc.QUOTA_MIN_FREE_FILES))
    if vals['fs_free_bytes'] < cc.FS_MIN_FREE_BYTES:
        low.append('%s free %.1f TB < %.1f TB' % (fs, vals['fs_free_bytes'] / 1e12, cc.FS_MIN_FREE_BYTES / 1e12))
    text = '%s quota headroom %s, %s files; %s free %.1f TB' % (
        user, 'unlimited' if vals['quota_free_bytes'] is None else '%.2f TiB' % (vals['quota_free_bytes'] / 2 ** 40),
        'unlimited' if vals['quota_free_files'] is None else vals['quota_free_files'], fs, vals['fs_free_bytes'] / 1e12)
    return not low, ('; '.join(low) + ' (%s)' % text) if low else text, vals


def io_guard_raised():
    """The file that raises the recorders' I/O guard for this host, else None: its stop file with a line starting with
    IO_GUARD_STOP_PREFIX, or the alert file present (an unreadable stop file does not raise it, as `grep -qs`)."""
    cc = _cc()
    kw = dict(ops=K.cfg().paths.OPS_ROOT, host=K.hostname())
    stop = cc.IO_GUARD_STOP_FILE.format(**kw)
    try:
        with open(stop, errors='replace') as f:
            if any(ln.startswith(cc.IO_GUARD_STOP_PREFIX) for ln in f):
                return stop
    except OSError:
        pass
    alert = cc.IO_GUARD_ALERT_FILE.format(**kw)
    return alert if os.path.exists(alert) else None


def io_guard_wait(sleep=time.sleep):
    """Wait while the I/O guard is raised (polled every IO_GUARD_POLL_S); returns the seconds waited."""
    t0, n = time.time(), 0
    while True:
        why = io_guard_raised()
        if not why:
            if n:
                K.say('CP-IOGUARD: lowered after %.0f s' % (time.time() - t0))
            return time.time() - t0
        if n == 0:
            K.say('CP-IOGUARD: the recorders\' I/O guard is raised (%s) - waiting, polled every %d s'
                  % (why, _cc().IO_GUARD_POLL_S))
        n += 1
        sleep(_cc().IO_GUARD_POLL_S)


# ---------------------------------------------------------------------------------------------------------------------
# the campaign of one invocation
# ---------------------------------------------------------------------------------------------------------------------

class Campaign:
    """The campaign of this invocation: generation, roots, enumeration, code pin. ``gate(u)`` before each unit; ``after(u)``
    once a cutouts unit ran (its child fingerprints join the pin). A pin change or low space stops the campaign: that unit
    and every later one refuse."""

    def __init__(self, gen, enum, pin, space_fn=None, sleep=time.sleep):
        self.generation, self.enum, self.pin = gen, enum, pin
        self.space_fn, self.sleep = space_fn or space, sleep
        self.stopped = None
        self.record_path = None

    def refusal(self):
        io_guard_wait(self.sleep)
        if self.stopped:
            return '%s (campaign stopped earlier in this run)' % self.stopped
        ch = self.pin.changed() if self.pin is not None else []
        if ch:
            self.stopped = 'CP-PIN: the engine code changed since the campaign started: %s' % '; '.join(ch[:5])
            return self.stopped
        ok, text, _ = self.space_fn()
        if not ok:
            self.stopped = 'CP-SPACE: %s' % text
            return self.stopped
        return None

    def gate(self, u):
        r = self.refusal()
        if r:
            u.refuse(r + ' - not started')

    def after(self, u):
        if self.pin is None:
            return
        bad = self.pin.absorb((u.meta or {}).get('legacy_code'))
        if bad and not self.stopped:
            self.stopped = 'CP-PIN: %s ran another engine version than the campaign pinned: %s' % (u.name, '; '.join(bad[:5]))


def scope_cells(args):
    """The night cells named by --cells / --epochs (<LST>_<date>; a band suffix refuses: a campaign covers whole cells), minus
    --exclude-cells; None when none is named (every cube directory)."""
    from exopipe.phase3.core import cells as C
    spec = ','.join(x for x in (getattr(args, 'cells', None), getattr(args, 'epochs', None)) if x)
    if not spec:
        return None
    root = K.cfg().paths.SNAPSHOT_CUBES_ROOT
    uni = cube_cells(root) if os.path.isdir(root) else []
    out = []
    for it in C.expand(spec, universe=uni, kind='night cell'):
        lst, date, sb = C.parse_night_cell(it)
        if sb:
            raise Refusal('CP-SCOPE: --campaign covers whole night cells; %s names a band' % it)
        out.append('%s_%s' % (lst, date))
    if getattr(args, 'exclude_cells', None):
        drop = set(C.expand(args.exclude_cells, universe=uni, kind='night cell'))
        out = [c for c in out if c not in drop]
    return out


def gate_args(args, stages):
    """CP-GEN / CP-SCOPE / CP-STAGES: what --campaign takes. Its roots come from the generation (no --cutout-root, --stage1);
    it covers whole cells and every band (no --scopes, --index, --manifest-name), the snapshot samples (no --targets,
    --planets), a new generation (no --in-place); its stages are cutouts and measure (the evaluation is stage 2)."""
    if args.tier != 'snapshot':
        raise Refusal('CP-SCOPE: --campaign is a snapshot-tier campaign (--tier snapshot)')
    for flag, attr in (('--cutout-root', 'cutout_root'), ('--stage1', 'stage1'), ('--scopes', 'scopes'),
                       ('--index', 'index'), ('--manifest-name', 'manifest_name'), ('--targets', 'targets'),
                       ('--planets', 'planets'), ('--in-place', 'in_place')):
        if getattr(args, attr, None):
            raise Refusal('CP-SCOPE: --campaign takes no %s (its roots derive from photometry_campaign.GENERATION; it covers '
                          'every band of whole cells for the snapshot samples, into a new generation)' % flag)
    bad = [s for s in stages if s not in ('cutouts', 'measure')]
    if bad:
        raise Refusal('CP-STAGES: a campaign runs the stages cutouts and measure, not %s' % ','.join(bad))
    generation()


def start(run, args):
    """The Campaign of this invocation (started once: enumeration with both exclusion readers, code pin, space, start
    record <out>/_records/photometry/<RECORD_SUBDIR>/<generation>_<run id>.json). A dry run lists without the engine's reader
    and writes nothing."""
    camp = getattr(args, 'photometry_campaign', None)
    if camp is not None:
        return camp
    gen = generation()
    root = cutout_root()
    K.gate_write_root(root, 'the campaign cutout root')
    cells = scope_cells(args)
    eng = None
    if not run.dry_run:
        listed = cells if cells is not None else cube_cells(K.cfg().paths.SNAPSHOT_CUBES_ROOT)
        eng = engine_exclusions(run, listed) if listed else {}
    enum = enumerate_units(cells, engine_excluded=eng)
    pin = CodePin(pin_files())
    camp = Campaign(gen, enum, pin)
    ok, text, vals = camp.space_fn()
    K.plan(run, '[campaign] generation %s: cutouts %s, stage 1 %s; %d units (%d subband, %d wideband) in %d cells; %d '
                '(cell, subband) excluded, %d held; %d of %d cube directories COMPLETE; engine reader %s; %d engine files '
                'pinned; space: %s' % (gen, root, os.path.join(run.out_root, stage1_dir()), len(enum.units),
                                       sum(kind_of(b) == 'sb' for _, b in enum.units),
                                       sum(kind_of(b) == 'wb' for _, b in enum.units), len(enum.cells), len(enum.excluded),
                                       len(enum.held), enum.n_complete, enum.n_cube_dirs,
                                       'read' if enum.engine_read else 'not read (dry run)', len(pin.files), text))
    if not ok:
        if run.dry_run:
            K.would_refuse(run, 'campaign', 'CP-SPACE: %s' % text)
        else:
            raise Refusal('CP-SPACE: %s - the campaign does not start' % text)
    if not run.dry_run:
        d = os.path.join(run.out_root, '_records', run.step, _cc().RECORD_SUBDIR)
        os.makedirs(d, exist_ok=True)
        camp.record_path = os.path.join(d, '%s_%s.json' % (gen, run.run_id))
        record.atomic_write_json(camp.record_path, dict(
            run.header(), generation=gen, cutout_root=root, stage1=os.path.join(run.out_root, stage1_dir()),
            started=record.utcnow(), user=run_user(), host=K.hostname(), scope=cells, pin=pin.files, space=vals,
            enumeration=enum.to_dict()), durable=True)
    args.photometry_campaign = camp
    return camp


def latest_record(out_root, gen):
    """The newest campaign start record of generation ``gen`` under ``out_root`` (None when there is none)."""
    d = os.path.join(out_root, '_records', 'photometry', _cc().RECORD_SUBDIR)
    names = sorted(n for n in (os.listdir(d) if os.path.isdir(d) else ()) if n.startswith(gen + '_') and n.endswith('.json'))
    if not names:
        return None
    with open(os.path.join(d, names[-1])) as f:
        return json.load(f)


# ---------------------------------------------------------------------------------------------------------------------
# stage-1 readiness per cell and band kind
# ---------------------------------------------------------------------------------------------------------------------

def unit_state(out_root, cell, band):
    """(state, text) of the cutouts unit of (cell, band) from its record: 'done'; 'none' - no input for a known reason
    (C-EXCLUDED; C-NOCUT whose engine counts all promise no file); 'held' - anything else (no record, refused, error, another
    no-input reason such as C-NOCUBE)."""
    from exopipe.phase3.lib.photometry import steps_cutouts as SC
    name = 'cutouts.%s.%s' % (cell, band)
    rec = record.load_record(out_root, 'photometry', name)
    if rec is None:
        return 'held', '%s: no record' % name
    st, why = rec.get('status'), (rec.get('reason') or '').strip()
    if st == 'done':
        return 'done', name
    if st == 'no-input':
        if why.startswith('C-EXCLUDED'):
            return 'none', '%s: excluded' % name
        out = (rec.get('meta') or {}).get('engine_outcome') or {}
        if why.startswith('C-NOCUT') and out and all(SC.expected_files(out.get(S), kind_of(band) == 'wb') == 0
                                                    for S in STOKES):
            return 'none', '%s: no cutout (engine counts V %s, I %s)' % (name, out.get('V'), out.get('I'))
    return 'held', '%s: %s %s' % (name, st, why[:240])


def readiness(out_root, enum, cell, kind):
    """(outcome, detail) of the stage 1 of ``cell`` for band kind ``kind``: 'ready'; 'held' (a unit of the kind not terminal,
    or - subbands - a held subband); 'no subband' / 'no wideband' (every unit of the kind terminal and none wrote a cutout);
    'empty' (the campaign has no unit of the kind in the cell)."""
    bands = enum.bands_of(cell, kind)
    held = ['%s %s: held at enumeration (%s)' % (h['cell'], h['subband'], h['reason'])
            for h in (enum.held_of(cell) if kind == 'sb' else [])]
    if not bands and not held:
        return 'empty', 'no %s unit in %s' % (KIND_NAME[kind], cell)
    states = [unit_state(out_root, cell, b) for b in bands]
    held += [t for s, t in states if s == 'held']
    if held:
        return 'held', '; '.join(held)
    if all(s == 'none' for s, _ in states):
        return 'no %s' % KIND_NAME[kind], '; '.join(t for _, t in states)
    return 'ready', '%d %s units terminal (%d done)' % (len(states), KIND_NAME[kind], sum(s == 'done' for s, _ in states))


def stage1_cell(run, args, camp, cell, kind, root, stage1):
    """The stage 1 of one cell and band kind in a campaign: held (SM-HELD, refused), no cutout of the kind (SM-NOBAND, no
    input, outcome recorded), or measured (steps_snapshot.measure_cell into the kind's manifest, the campaign gate first)."""
    from exopipe.phase3.lib.photometry import steps_snapshot as SS
    bands = list(kind_bands(kind))
    name = SS.unit_name(cell, bands)
    outcome, detail = readiness(run.out_root, camp.enum, cell, kind)
    if outcome == 'empty':
        return []
    man = _cc().STAGE1_MANIFEST[kind]
    if run.dry_run:
        K.plan(run, '[measure]   unit %s: %s (%s)' % (name, outcome, detail[:300]))
        return []
    if outcome == 'ready':
        files = {b: SS.cell_files(cell, root, None, b) for b in bands}
        return SS.measure_cell(run, args, cell, bands, files, root, stage1, man, kind_manifest=True, gate=camp.gate)
    with run.unit(name, tier='snapshot', stage='measure', cell=cell, bands=bands, cutout_root=root, manifest=man,
                  campaign=camp.generation, outcome=outcome) as u:
        if outcome == 'held':
            u.refuse('SM-HELD: the stage 1 of %s (%s) waits for its cutouts units - %s' % (cell, KIND_NAME[kind], detail[:900]))
        raise NoInput('SM-NOBAND: %s - every %s unit of %s is terminal and none wrote a cutout: %s'
                      % (outcome, KIND_NAME[kind], cell, detail[:900]))
    return run.units[-1:]


# ---------------------------------------------------------------------------------------------------------------------
# complete-when content test
# ---------------------------------------------------------------------------------------------------------------------

def check_done(rec, band, retest=False):
    """Problems of a done cutouts record: a failed recorded test; a required check (photometry_campaign.COMPLETE_CHECKS, and
    COMPLETE_STOKES_CHECKS per Stokes the engine counted cutouts for) not recorded; an engine count no file count can be held
    to; not exactly one cutout manifest; a manifest row whose file is missing or has another size; a Stokes with a positive
    engine count and no row. ``retest``: the manifest's content tests run again."""
    from exopipe.phase3.core import validate
    from exopipe.phase3.lib.photometry import steps_cutouts as SC
    from exopipe.phase3.lib.photometry import tests_cutouts as TC
    cc = _cc()
    bad = []
    tests = rec.get('tests') or []
    failed = [t['name'] for t in tests if not t.get('passed') and not t.get('waived')]
    if failed:
        bad.append('failed tests %s' % failed[:5])
    passed = {t['name'] for t in tests if t.get('passed')}
    bad += ['check %s not recorded' % n for n in cc.COMPLETE_CHECKS if n not in passed]
    out = (rec.get('meta') or {}).get('engine_outcome') or {}
    want = {S: SC.expected_files(out.get(S), kind_of(band) == 'wb') for S in STOKES}
    bad += ['engine count %s %r promises no file count' % (S, out.get(S)) for S in STOKES if want[S] is None]
    for S in STOKES:
        if want[S]:
            bad += ['check %s not recorded' % n.format(stokes=S) for n in cc.COMPLETE_STOKES_CHECKS
                    if n.format(stokes=S) not in passed]
    mans = [o for o in rec.get('outputs') or [] if o.get('class') == 'photometry_cutout_manifest']
    if len(mans) != 1:
        return bad + ['%d cutout manifests recorded (one expected)' % len(mans)]
    o = mans[0]
    ctx = o.get('ctx') or {}
    root = ctx.get('root') or (rec.get('meta') or {}).get('cutout_root')
    if not os.path.isfile(o['path']):
        return bad + ['manifest %s missing' % o['path']]
    _, rows = TC.read_manifest(o['path'])
    per = {S: 0 for S in STOKES}
    miss = []
    for r in rows:
        p = os.path.join(root, r['file'])
        try:
            size = os.stat(p).st_size
        except OSError:
            miss.append('%s missing' % r['file'])
            continue
        if size != int(r['bytes']):
            miss.append('%s: %d bytes, manifest %s' % (r['file'], size, r['bytes']))
        per[r['stokes']] = per.get(r['stokes'], 0) + 1
    if miss:
        bad.append('%d of %d manifest files missing or of another size (e.g. %s)' % (len(miss), len(rows), miss[0]))
    bad += ['Stokes %s: engine count %s, no file in the manifest' % (S, out.get(S)) for S in STOKES if want[S] and not per[S]]
    if retest:
        bad += ['%s: %s' % (t['name'], t['detail']) for t in validate.run(o['class'], o['path'], **ctx) if not t['passed']]
    return bad


def check_stage1(out_root, stage1, enum, cell, kind):
    """(state, problem or '') of the stage 1 of (cell, kind): 'done' (a done measure record of the kind's bands, its manifest
    passing MM1 + MM2), 'no <kind>' (its SM-NOBAND record), 'empty', or incomplete."""
    from exopipe.phase3.lib.photometry import steps_snapshot as SS
    from exopipe.phase3.lib.photometry import tests as T
    outcome, detail = readiness(out_root, enum, cell, kind)
    if outcome == 'empty':
        return 'empty', ''
    if outcome == 'held':
        return 'held', 'stage 1 %s %s held: %s' % (cell, kind, detail[:300])
    name = SS.unit_name(cell, list(kind_bands(kind)))
    rec = record.load_record(out_root, 'photometry', name)
    if rec is None:
        return 'missing', 'stage 1 %s %s: no record %s (%s)' % (cell, kind, name, outcome)
    meta = rec.get('meta') or {}
    if outcome != 'ready':
        if rec.get('status') == 'no-input' and meta.get('outcome') == outcome:
            return outcome, ''
        return 'missing', 'stage 1 %s %s: outcome %s, record %s %s' % (cell, kind, outcome, rec.get('status'), meta.get('outcome'))
    if rec.get('status') != 'done' or not set(kind_bands(kind)) <= set(meta.get('bands_done') or []):
        return 'missing', 'stage 1 %s %s: record %s (%s)' % (cell, kind, rec.get('status'), (rec.get('reason') or '')[:200])
    mp = os.path.join(stage1, cell, _cc().STAGE1_MANIFEST[kind])
    if not os.path.isfile(mp):
        return 'missing', 'stage 1 %s %s: no manifest %s' % (cell, kind, mp)
    for fn in (T.mm1, T.mm2):
        ok, _, det = fn(mp)
        if not ok:
            return 'missing', 'stage 1 %s %s: %s %s' % (cell, kind, os.path.basename(mp), det)
    return 'done', ''


def check(out_root, enum, stage1=None, retest=False):
    """(complete, counts, problems) of a campaign: every unit of ``enum`` terminal (done records pass check_done), no held
    (cell, subband), and - with ``stage1`` - every cell's stage 1 per kind done or 'no <kind>'."""
    counts = {'units': len(enum.units), 'done': 0, 'none': 0, 'incomplete': 0, 'held_enumeration': len(enum.held)}
    problems = ['held at enumeration: %s %s (%s)' % (h['cell'], h['subband'], h['reason']) for h in enum.held]
    for cell, band in enum.units:
        state, text = unit_state(out_root, cell, band)
        if state == 'done':
            bad = check_done(record.load_record(out_root, 'photometry', 'cutouts.%s.%s' % (cell, band)), band, retest)
            if bad:
                state, text = 'incomplete', 'cutouts.%s.%s: done record, but %s' % (cell, band, '; '.join(bad)[:600])
        counts[state if state in ('done', 'none') else 'incomplete'] += 1
        if state not in ('done', 'none'):
            problems.append(text)
    if stage1:
        for cell in enum.cells:
            for kind in ('sb', 'wb'):
                st, why = check_stage1(out_root, stage1, enum, cell, kind)
                key = 'stage1_%s' % st.replace(' ', '_')
                counts[key] = counts.get(key, 0) + 1
                if why:
                    problems.append(why)
    return not problems, counts, problems


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m exopipe.phase3.lib.photometry.campaign',
                                 description='The complete-when content test of a cutout-generation campaign (read only).')
    sub = ap.add_subparsers(dest='cmd')
    c = sub.add_parser('check', help='every unit terminal and its files present; stage 1 per cell and kind done or no <kind>')
    c.add_argument('--out', required=True, help='the campaign\'s --out (records under <out>/_records/photometry)')
    c.add_argument('--cells', help='night cells <LST>_<date> (comma list, patterns, @FILE); default every cube directory')
    c.add_argument('--generation', help='default photometry_campaign.GENERATION')
    c.add_argument('--retest', action='store_true', help='run every done unit\'s manifest content tests again (reads every '
                   'listed cutout)')
    c.add_argument('--no-stage1', action='store_true', help='check the cutouts units only')
    c.add_argument('--max-problems', type=int, default=200, help='problem lines printed (default 200)')
    a = ap.parse_args(argv)
    if a.cmd != 'check':
        ap.print_help()
        return 2
    if a.generation:
        K.cfg().set('photometry_campaign.GENERATION', a.generation, source='--generation')
    try:
        gen = generation()
        cells = scope_cells(argparse.Namespace(cells=a.cells, epochs=None, exclude_cells=None))
        enum = enumerate_units(cells)
    except Refusal as e:
        print('CANNOT CHECK: %s' % e, flush=True)
        return 2
    out = os.path.abspath(a.out)
    stage1 = None if a.no_stage1 else os.path.join(out, stage1_dir())
    ok, counts, problems = check(out, enum, stage1, a.retest)
    started = latest_record(out, gen)
    if started is not None:
        su = {tuple(u) for u in started['enumeration']['units'] if cells is None or u[0] in cells}
        now = set(enum.units)
        if su != now:
            print('UNITS CHANGED since the campaign start record %s: %d added (e.g. %s), %d removed (e.g. %s)' % (
                started.get('run_id'), len(now - su), sorted(now - su)[:3], len(su - now), sorted(su - now)[:3]), flush=True)
    print('GENERATION %s; cutouts %s; records %s; stage 1 %s' % (gen, cutout_root(), out, stage1 or 'not checked'))
    print('COUNTS %s' % json.dumps(counts, sort_keys=True))
    for p in problems[:a.max_problems]:
        print('PROBLEM %s' % p)
    if len(problems) > a.max_problems:
        print('... %d more problems' % (len(problems) - a.max_problems))
    print('COMPLETE' if ok else 'INCOMPLETE: %d problems' % len(problems), flush=True)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
