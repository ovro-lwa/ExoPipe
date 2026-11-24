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
                     records' legacy_code). The pin lives in the campaign start record (rewritten as files join it); a later
                     invocation under the same --out (--resume among them) continues that campaign only with the same cutout
                     root and engine code, else it refuses (CP-GEN, CP-PIN) unless --new-campaign starts a new one. A unit
                     whose children ran another version than the pin (or two, 'mixed:') is recorded refused CP-PIN (not a
                     gap: it holds its cell) and the campaign stops; a unit recorded under another pin is cut again.
  space()            the run user's quota headroom (bytes and files, `lfs quota -q -u <user>`) and the file system's free space
                     (the filesystem_summary line of `lfs df`), before each unit (gen6_space_ok.sh), with this section's
                     floors. ONE gate: in a run that is not a dry run, cli.main's space gate reads it (steps/photometry.
                     space_gate: the SPACE_FILESYSTEM and these floors, before every unit, RECHECK_S 0) and run_space()
                     reports that gate's last verdict without reading lfs again; only a dry run (no gate) reads it, space().
                     Below a floor, or unreadable: the run does not start (SPACE-*), a unit refuses and the step stops
                     (record.Run.unit), and the campaign's own check refuses CP-SPACE and every later unit.
  io_guard_wait()    the recorders' I/O guard: wait while it is raised, before each unit and each stage-1 cell (a yield).
  gap_kind()         GAPS (R-GREGG-20261027-LSRECON-SCALE-BOUND-0P5-2P0-LEVEL-REFUSALS-STAY-GAPS-55MHZ-06H-R0-N-MINUS-1, verbatim
                     "Refusals by the large-scale level test stay refused, as gaps, never forced"): a unit whose inputs a gate
                     refused (photometry_campaign.GAP_REFUSED: C-FLUXSCALE before the engine) or that found no input by a gap
                     code (GAP_NOINPUT: C-NOCUBE) is terminal and listed - never rerun into a pass by an override.
  readiness()        per cell and band kind (subbands 'sb', widebands 'wb'): stage 1 runs only when every cutouts unit of the kind
                     has a terminal record - done, no input for a known reason (C-EXCLUDED; C-NOCUT whose engine counts all
                     promise no file: the night rule, nothing in view), or a no-input gap. A failed, missing or non-gap refused
                     unit, or a held subband, HOLDS THAT CELL'S KIND ONLY (SM-HELD, incomplete); the other cells go on. A
                     refused gap holds it too (SM-HELD), as a gap: the kind's stage 1 is not run. A kind whose units all wrote
                     nothing ends with the outcome 'no wideband' / 'no subband' (SM-NOBAND, no input; nothing measured) - not
                     done, not failed (gen6_stage1_wb_no_wideband_20261122.py: 01h, 02h, 04h, 08h of 2024-12-26).
  check()            the campaign's complete-when content test, one command anyone can run (read only):
                     python -m exopipe.phase3.lib.photometry.campaign check --out <snapshot forced root> [--cells ...]
                     against the newest start record under --out (its generation, cutout root and pin), enumerating with both
                     exclusion readers as the campaign does: every enumerated unit has a record of THIS campaign (meta campaign
                     and cutout_root; a record of another run under the same --out is incomplete) that is done - its required
                     checks passed (COMPLETE_* of the band kind), its manifest's files exist with their sizes (--retest: its
                     outputs' content tests run again), its children's code is the pin's -, a terminal no-input record, or a
                     gap; every cell's stage 1 per kind is done from the cutouts the cell holds now (the measure record's
                     cutout listing digest; a re-cut since is incomplete) with manifest tests MM1 + MM2, 'no <kind>' or not
                     run for a gap. Exit 0 "COMPLETE with N gaps (E excluded, R night-rule refused, Z with no cutout; not
                     gaps)" (each gap a GAP line: unit, code, reason), 1 INCOMPLETE (PROBLEM lines), 2 cannot check.
"""
import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import time

from exopipe.phase3.core import record
from exopipe.phase3.core import space as S
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
        self.cube_root, self.n_cube_dirs, self.n_scoped, self.n_complete, self.engine_read = None, 0, 0, 0, False

    def to_dict(self):
        return dict(cube_root=self.cube_root, n_cube_dirs=self.n_cube_dirs, n_scoped=self.n_scoped, n_complete=self.n_complete,
                    engine_read=self.engine_read, cells=self.cells, units=[list(u) for u in self.units],
                    excluded=self.excluded, held=self.held)

    @classmethod
    def from_dict(cls, d):
        e = cls()
        for k in ('cube_root', 'n_cube_dirs', 'n_scoped', 'n_complete', 'engine_read', 'cells', 'excluded', 'held'):
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


def enumerate_units(cells=None, cube_root=None, rec=None, ex=None, engine_excluded=None, complete=None, present=None):
    """The campaign's Enumeration. ``present``: the cube directories (cube_cells, listed once by the caller; default listed
    here). ``cells``: restrict to these night cells (a named cell without a cube directory is held);
    ``rec``: the rescale records {cube path: record} (default fluxscale.records()); ``ex``: the exclusion read (default
    core/exclusions.for_tier('snapshot', fresh=True)); ``engine_excluded``: {cell: subbands the subband engine's reader
    excludes} (cutouts_probe.py; None: not read - core/exclusions alone, a dry run's listing; start() and check() read it);
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
    present = cube_cells(root) if present is None else list(present)
    e.n_cube_dirs = len(present)
    todo = present if cells is None else [c for c in cells if c in set(present)]
    e.n_scoped = len(todo)
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
    """The user this process runs as (the quota that the products count against; core/space.run_user)."""
    return S.run_user()


def _lfs(argv):
    cc = _cc()
    return S.lfs_runner(cc.LFS, cc.LFS_TIMEOUT_S)(argv)


parse_quota = S.parse_quota          # `lfs quota -q -u <user> <fs>` (core/space.parse_quota)
parse_df = S.parse_df                # the filesystem_summary line of `lfs df <fs>` (core/space.parse_df)


def run_space(run):
    """The campaign's space verdict (ok, text, values) in ``run``: with cli.main's space gate (``run.space_gate``; every run
    that is not a dry run - steps/photometry.space_gate gives it this section's file system and floors, read before every
    unit) the gate's last verdict, reading nothing (one gate, no second lfs read); without it (a dry run) space(), once per
    call."""
    gate = getattr(run, 'space_gate', None)
    return gate.state() if gate is not None else space()


def space(lfs=None):
    """(ok, text, values): the run user's quota headroom (bytes, files) and the file system's free space against the floors of
    photometry_campaign - core/space.lustre with this section's SPACE_FILESYSTEM, LFS, LFS_TIMEOUT_S and floors (what the
    cli's gate reads for a campaign). Unreadable is not ok. ``lfs``: argv -> (rc, stdout, stderr) (tests). A run with the
    cli's gate asks run_space() instead."""
    cc = _cc()
    ok, text, vals, _codes = S.lustre(cc.SPACE_FILESYSTEM, cc, lfs=lfs or _lfs, user=run_user())
    return ok, text, vals


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

    def __init__(self, gen, enum, pin, space_fn=None, sleep=time.sleep, root=None):
        self.generation, self.enum, self.pin = gen, enum, pin
        self.cutout_root = root or cutout_root()
        self.space_fn, self.sleep = space_fn or space, sleep
        self.stopped = None
        self.record_path, self.record = None, None      # the start record this invocation wrote (its pin kept current)

    def refusal(self):
        if self.stopped:                # a stopped campaign refuses at once (no I/O-guard wait first)
            return '%s (campaign stopped earlier in this run)' % self.stopped
        io_guard_wait(self.sleep)
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
        """Once a cutouts unit ran: its children's code fingerprint joins the pin (a file first seen; the start record is
        rewritten so a later invocation and the check see it). A unit whose children ran another version than the pin, or two
        ('mixed:'), does not stay done: its record is refused CP-PIN (incomplete, it holds its cell) and the campaign stops."""
        if self.pin is None:
            return
        n0 = len(self.pin.files)
        bad = self.pin.absorb((u.meta or {}).get('legacy_code'))
        if len(self.pin.files) != n0 and self.record_path:
            self.record['pin'] = dict(self.pin.files)
            record.atomic_write_json(self.record_path, self.record, durable=True)
        if bad:
            why = 'CP-PIN: %s ran another engine version than the campaign pinned: %s' % (u.name, '; '.join(bad[:5]))
            refuse_after(u, why)
            if not self.stopped:
                self.stopped = why

    def stale(self, run, name):
        """True when the existing record of unit ``name`` is not this campaign's (belongs) or its children's code is not the
        pin's (code_mismatch): such a unit is cut again, never resumed."""
        rec = record.load_record(run.out_root, run.step, name)
        if rec is None:
            return False
        return (not belongs(rec, (self.generation, self.cutout_root))
                or bool(code_mismatch(self.pin.files if self.pin is not None else {}, (rec.get('meta') or {}).get('legacy_code'))))


def refuse_after(u, why):
    """Rewrite the closed unit ``u``'s record as refused ``why`` (its earlier outcome kept in its notes)."""
    run = u.run
    rec = record.load_record(run.out_root, run.step, u.name)
    if rec is not None:
        rec['notes'] = list(rec.get('notes') or []) + ['%s %s - refused after the unit by the campaign: %s' % (
            rec.get('status'), (rec.get('reason') or '')[:200], why[:300])]
        rec['status'], rec['reason'] = 'refused', why
        record.atomic_write_json(record.record_path(run.out_root, run.step, u.name), rec, durable=True)
    u.status, u.reason = 'refused', why


def belongs(rec, ident):
    """True when a unit record is this campaign's: ``ident`` (generation, cutout root) equals its meta's campaign and
    cutout_root. A record of another run under the same --out (a served-generation regeneration, --in-place, another root)
    is not."""
    meta = rec.get('meta') or {}
    return (meta.get('campaign') == ident[0]
            and os.path.normpath(str(meta.get('cutout_root') or '')) == os.path.normpath(ident[1]))


def code_mismatch(pin, fingerprint):
    """How a unit's children's code fingerprint {file: sha1[:12]} (its record's legacy_code) differs from the campaign pin:
    a file in two versions ('mixed:'), in another version, or not in the pin."""
    bad = []
    for f, h in sorted((fingerprint or {}).items()):
        f = os.path.abspath(f)
        if str(h).startswith('mixed:'):
            bad.append('%s ran in two versions (%s)' % (f, h))
        elif f not in pin:
            bad.append('%s (%s) is not in the campaign pin' % (f, h))
        elif pin[f] != h:
            bad.append('%s: pinned %s, the unit ran %s' % (f, pin[f], h))
    return bad


def continue_refusal(prior, root, pin):
    """Why this invocation cannot continue the campaign of start record ``prior`` under the same --out (None when it can):
    another cutout root (CP-GEN), or engine code other than its pin - a pinned file changed, or a file pinned now that it
    did not pin (CP-PIN). A campaign's units come from one engine version; --new-campaign starts a new one."""
    rid = prior.get('run_id')
    if os.path.normpath(str(prior.get('cutout_root') or '')) != os.path.normpath(root):
        return ('CP-GEN: campaign %s of generation %s writes %s; this invocation\'s cutout root is %s - a new campaign under '
                'this --out takes --new-campaign' % (rid, prior.get('generation'), prior.get('cutout_root'), root))
    old = prior.get('pin') or {}
    bad = ['%s: pinned %s, now %s' % (f, h, K.code_sha12(f) or 'unreadable') for f, h in sorted(old.items())
           if K.code_sha12(f) != h]
    bad += ['%s (%s) pinned now, not by campaign %s' % (f, h, rid) for f, h in sorted(pin.files.items()) if f not in old]
    if bad:
        return ('CP-PIN: the engine code differs from the pin of campaign %s (started %s): %s - its units come from one engine '
                'version; a new campaign under this --out takes --new-campaign (its check then calls every unit done under '
                'the old pin incomplete, and the run cuts them again)' % (rid, prior.get('started'), '; '.join(bad[:5])))
    return None


def scope_cells(args, present=None):
    """The night cells named by --cells / --epochs (<LST>_<date>; a band suffix refuses: a campaign covers whole cells), minus
    --exclude-cells; None when none is named (every cube directory). ``present``: the cube directories (patterns expand over
    them; default listed here). A named cell without a cube directory stays in the scope: the enumeration holds it."""
    from exopipe.phase3.core import cells as C
    spec = ','.join(x for x in (getattr(args, 'cells', None), getattr(args, 'epochs', None)) if x)
    if not spec:
        return None
    root = K.cfg().paths.SNAPSHOT_CUBES_ROOT
    uni = list(present) if present is not None else (cube_cells(root) if os.path.isdir(root) else [])
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
    try:
        K.gate_write_root(root, 'the campaign cutout root')
    except Refusal as e:            # a dry run reports it (exit 2) and still lists the campaign
        if not run.dry_run:
            raise
        K.would_refuse(run, 'campaign', str(e))
    croot = K.cfg().paths.SNAPSHOT_CUBES_ROOT
    present = cube_cells(croot)         # listed once: the scope's patterns, the engine reader's cells, the enumeration
    cells = scope_cells(args, present)
    eng = None
    if not run.dry_run:                 # the engine's reader probes only cells with a cube directory (a named one without is held)
        have = set(present)
        probe = [c for c in (cells if cells is not None else present) if c in have]
        eng = engine_exclusions(run, probe) if probe else {}
    enum = enumerate_units(cells, engine_excluded=eng, present=present)
    pin = CodePin(pin_files())
    prior = latest_record(run.out_root, gen)
    new = bool(getattr(args, 'new_campaign', False))
    cont = None
    if prior is not None and not new:   # a later invocation continues the campaign under this --out: same root, same pin
        why = continue_refusal(prior, root, pin)
        if why and not run.dry_run:
            raise Refusal(why)
        if why:
            K.would_refuse(run, 'campaign', why)
        else:
            pin.files, cont = dict(prior.get('pin') or {}), prior.get('run_id')
    camp = Campaign(gen, enum, pin, root=root, space_fn=lambda: run_space(run))   # the cli's gate when the run has one
    ok, text, vals = camp.space_fn()
    which = ('continues campaign %s' % cont if cont else
             'a new campaign over %s (--new-campaign)' % prior.get('run_id') if prior and new else 'a new campaign')
    K.plan(run, '[campaign] generation %s: cutouts %s, stage 1 %s; %d units (%d subband, %d wideband) in %d cells; %d '
                '(cell, subband) excluded, %d held; %d of the %d cube directories in scope COMPLETE (%d under the cube root); '
                'engine reader %s; %d engine files pinned (%s); space: %s' % (
                    gen, root, os.path.join(run.out_root, stage1_dir()), len(enum.units),
                    sum(kind_of(b) == 'sb' for _, b in enum.units), sum(kind_of(b) == 'wb' for _, b in enum.units),
                    len(enum.cells), len(enum.excluded), len(enum.held), enum.n_complete, enum.n_scoped, enum.n_cube_dirs,
                    'read' if enum.engine_read else 'not read (dry run)', len(pin.files), which, text))
    if not ok:
        if run.dry_run:
            K.would_refuse(run, 'campaign', 'CP-SPACE: %s' % text)
        else:
            raise Refusal('CP-SPACE: %s - the campaign does not start' % text)
    if not run.dry_run:
        d = os.path.join(run.out_root, '_records', run.step, _cc().RECORD_SUBDIR)
        os.makedirs(d, exist_ok=True)
        camp.record_path = os.path.join(d, '%s_%s.json' % (gen, run.run_id))
        camp.record = dict(run.header(), generation=gen, cutout_root=root, stage1=os.path.join(run.out_root, stage1_dir()),
                           started=record.utcnow(), user=run_user(), host=K.hostname(), scope=cells, pin=dict(pin.files),
                           space=vals, continues=cont, new_campaign_over=(prior.get('run_id') if prior and new else None),
                           enumeration=enum.to_dict())
        record.atomic_write_json(camp.record_path, camp.record, durable=True)
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


def units_changed(started, units, cells=None):
    """(added, removed): the units of a check's enumeration against a campaign start record, over the cells both scopes
    cover (the record's ``scope``, the check's ``cells``; None = every cell)."""
    scope = started.get('scope')
    then = {tuple(u) for u in started['enumeration']['units'] if cells is None or u[0] in cells}
    now = {tuple(u) for u in units if (scope is None or u[0] in scope) and (cells is None or u[0] in cells)}
    return sorted(now - then), sorted(then - now)


# ---------------------------------------------------------------------------------------------------------------------
# stage-1 readiness per cell and band kind
# ---------------------------------------------------------------------------------------------------------------------

def gap_kind(status, reason):
    """'refused' / 'no-input' when a unit's outcome is a recorded GAP (photometry_campaign.GAP_REFUSED / GAP_NOINPUT: a gate
    refused its inputs; R-GREGG-20261027-LSRECON-SCALE-BOUND-0P5-2P0-LEVEL-REFUSALS-STAY-GAPS-55MHZ-06H-R0-N-MINUS-1), else
    None."""
    cc = _cc()
    code = reason.split(':', 1)[0].strip()
    if status == 'refused' and code in cc.GAP_REFUSED and (cc.GAP_REFUSED[code] is None or cc.GAP_REFUSED[code] in reason):
        return 'refused'
    if status == 'no-input' and code in cc.GAP_NOINPUT:
        return 'no-input'
    return None


NONE_TAGS = ('excluded', 'night rule', 'no cutout')


def _none_tag(band, out):
    """Why a C-NOCUT unit whose counts promise no file has none: 'excluded' (the group search's exit 3), 'night rule' (the
    subband engine's 'None', the group search's 'refused: ...'), else 'no cutout' (a count 0, no member cube)."""
    vals = [str(out.get(S)).strip() for S in STOKES]
    if 'excluded' in vals:
        return 'excluded'
    if any((v == 'None' and kind_of(band) == 'sb') or v.startswith('refused: ') for v in vals):
        return 'night rule'
    return 'no cutout'


def unit_state(out_root, cell, band, ident):
    """(state, text) of the cutouts unit of (cell, band) from its record: 'done'; 'none' - no cutout by a data decision or the
    engine's outcome (C-EXCLUDED; C-NOCUT whose engine counts all promise no file, the night rule among them; the text after
    '<unit>: ' starts with its NONE_TAGS tag); 'gap' - a gate refused its inputs (gap_kind 'refused': holds its cell's stage 1
    of the band kind); 'gap-noinput' - no input by a gap code (C-NOCUBE: holds nothing); 'held' - anything else, incomplete
    (no record, a record of another run than this campaign - ``ident`` (generation, cutout root), belongs -, an error, an
    output or content refusal, an operational refusal, another no-input reason)."""
    from exopipe.phase3.lib.photometry import steps_cutouts as SC
    name = 'cutouts.%s.%s' % (cell, band)
    rec = record.load_record(out_root, 'photometry', name)
    if rec is None:
        return 'held', '%s: no record' % name
    if not belongs(rec, ident):
        meta = rec.get('meta') or {}
        return 'held', ('%s: a record of another run, not this campaign\'s (%s %s, campaign %s, cutout root %s; this campaign '
                        '%s, %s)' % (
            name, rec.get('status'), (rec.get('reason') or '')[:80], meta.get('campaign'), meta.get('cutout_root'), ident[0],
            ident[1]))
    st, why = rec.get('status'), (rec.get('reason') or '').strip()
    if st == 'done':
        return 'done', name
    gap = gap_kind(st, why)
    if gap:
        return 'gap' if gap == 'refused' else 'gap-noinput', '%s: %s %s' % (name, st, why[:400])
    if st == 'no-input':
        if why.startswith('C-EXCLUDED'):
            return 'none', '%s: excluded (%s)' % (name, why[:120])
        out = (rec.get('meta') or {}).get('engine_outcome') or {}
        if why.startswith('C-NOCUT') and out and all(SC.expected_files(out.get(S), kind_of(band) == 'wb') == 0
                                                    for S in STOKES):
            return 'none', '%s: %s (engine counts V %s, I %s)' % (name, _none_tag(band, out), out.get('V'), out.get('I'))
    return 'held', '%s: %s %s' % (name, st, why[:240])


def readiness(out_root, enum, cell, kind, ident):
    """(outcome, detail) of the stage 1 of ``cell`` for band kind ``kind``: 'ready'; 'held' (a unit of the kind not terminal,
    or - subbands - a held subband: incomplete); 'gap' (every unit terminal, but a gate refused the inputs of one or more -
    the stage 1 of the kind is not run, a gap itself); 'no subband' / 'no wideband' (every unit of the kind terminal and none
    wrote a cutout, no refused gap among them); 'empty' (the campaign has no unit of the kind in the cell). A no-input gap
    (C-NOCUBE) holds nothing: the stage 1 measures the bands that have cutouts. ``ident``: (generation, cutout root) of the
    campaign (unit_state)."""
    bands = enum.bands_of(cell, kind)
    held = ['%s %s: held at enumeration (%s)' % (h['cell'], h['subband'], h['reason'])
            for h in (enum.held_of(cell) if kind == 'sb' else [])]
    if not bands and not held:
        return 'empty', 'no %s unit in %s' % (KIND_NAME[kind], cell)
    states = [unit_state(out_root, cell, b, ident) for b in bands]
    held += [t for s, t in states if s == 'held']
    if held:
        return 'held', '; '.join(held)
    gaps = [t for s, t in states if s == 'gap']
    if gaps:
        return 'gap', '; '.join(gaps)
    if all(s in ('none', 'gap-noinput') for s, _ in states):
        return 'no %s' % KIND_NAME[kind], '; '.join(t for _, t in states)
    return 'ready', '%d %s units terminal (%d done)' % (len(states), KIND_NAME[kind], sum(s == 'done' for s, _ in states))


def stage1_cell(run, args, camp, cell, kind, root, stage1):
    """The stage 1 of one cell and band kind in a campaign: held (SM-HELD, refused), no cutout of the kind (SM-NOBAND, no
    input, outcome recorded), or measured (steps_snapshot.measure_cell into the kind's manifest, the campaign gate first)."""
    from exopipe.phase3.lib.photometry import steps_snapshot as SS
    bands = list(kind_bands(kind))
    name = SS.unit_name(cell, bands)
    outcome, detail = readiness(run.out_root, camp.enum, cell, kind, (camp.generation, root))
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
        if outcome == 'gap':
            u.refuse('SM-HELD: the stage 1 of %s (%s) is not run - a gate refused the inputs of its cutouts units (gaps, never '
                     'forced): %s' % (cell, KIND_NAME[kind], detail[:900]))
        raise NoInput('SM-NOBAND: %s - every %s unit of %s is terminal and none wrote a cutout: %s'
                      % (outcome, KIND_NAME[kind], cell, detail[:900]))
    return run.units[-1:]


# ---------------------------------------------------------------------------------------------------------------------
# complete-when content test
# ---------------------------------------------------------------------------------------------------------------------

def check_done(rec, band, retest=False, pin=None):
    """Problems of a done cutouts record: a failed recorded test; a required check not recorded (photometry_campaign.
    COMPLETE_CHECKS, COMPLETE_KIND_CHECKS of the band kind, COMPLETE_STOKES_CHECKS per Stokes the engine counted cutouts for,
    COMPLETE_GEN5_CHECKS per such Stokes of a subband of the generation-5 comparison sample); an engine count no file count can
    be held to; not exactly one output of each class of COMPLETE_OUTPUTS, or one missing; a manifest row whose file is missing
    or has another size; a Stokes with a positive engine count and no row; with ``pin`` (the campaign's, {file: sha1[:12]}),
    children's code other than the pin (code_mismatch). ``retest``: every required output's content tests run again."""
    from exopipe.phase3.core import validate
    from exopipe.phase3.lib.photometry import steps_cutouts as SC
    from exopipe.phase3.lib.photometry import tests_cutouts as TC
    cc = _cc()
    kind = kind_of(band)
    bad = []
    tests = rec.get('tests') or []
    failed = [t['name'] for t in tests if not t.get('passed') and not t.get('waived')]
    if failed:
        bad.append('failed tests %s' % failed[:5])
    passed = {t['name'] for t in tests if t.get('passed')}
    meta = rec.get('meta') or {}
    out = meta.get('engine_outcome') or {}
    want = {S: SC.expected_files(out.get(S), kind == 'wb') for S in STOKES}
    bad += ['engine count %s %r promises no file count' % (S, out.get(S)) for S in STOKES if want[S] is None]
    need = list(cc.COMPLETE_CHECKS) + list(cc.COMPLETE_KIND_CHECKS[kind])
    for S in STOKES:
        if want[S]:
            need += [n.format(stokes=S) for n in cc.COMPLETE_STOKES_CHECKS[kind]]
            if kind == 'sb' and meta.get('generation5_sample'):
                need += [n.format(stokes=S) for n in cc.COMPLETE_GEN5_CHECKS]
    bad += ['check %s not recorded' % n for n in need if n not in passed]
    if pin is not None:
        bad += ['engine code: %s' % x for x in code_mismatch(pin, meta.get('legacy_code'))[:3]]
    outs = {}
    for o in rec.get('outputs') or []:
        outs.setdefault(o.get('class'), []).append(o)
    for cls in cc.COMPLETE_OUTPUTS[kind]:
        if len(outs.get(cls, [])) != 1:
            bad.append('%d %s outputs recorded (one expected)' % (len(outs.get(cls, [])), cls))
        elif not os.path.isfile(outs[cls][0]['path']):
            bad.append('%s %s missing' % (cls, outs[cls][0]['path']))
    mans = outs.get('photometry_cutout_manifest') or []
    if len(mans) != 1 or not os.path.isfile(mans[0]['path']):
        return bad
    o = mans[0]
    ctx = o.get('ctx') or {}
    root = ctx.get('root') or meta.get('cutout_root')
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
        for cls in cc.COMPLETE_OUTPUTS[kind]:
            for x in outs.get(cls, [])[:1]:
                if os.path.isfile(x['path']):
                    bad += ['%s: %s' % (t['name'], t['detail']) for t in validate.run(x['class'], x['path'], **(x.get('ctx') or {}))
                            if not t['passed']]
    return bad


def check_stage1(out_root, stage1, enum, cell, kind, ident):
    """(state, problem or '') of the stage 1 of (cell, kind): 'done' (a done measure record of the kind's bands, measured
    from the cutouts the cell holds now - the record's cutout_listing_sha1 equals the listing's, as measure_cell's resume
    compares it -, its manifest passing MM1 + MM2), 'no <kind>' (its SM-NOBAND record), 'gap' (not run: a gate refused the
    inputs of a cutouts unit of the kind; the text names them - a gap, not a problem), 'empty', or incomplete (a re-cut
    since stage 1 among them: the next run measures it again)."""
    from exopipe.phase3.lib.photometry import steps_snapshot as SS
    from exopipe.phase3.lib.photometry import tests as T
    outcome, detail = readiness(out_root, enum, cell, kind, ident)
    if outcome == 'empty':
        return 'empty', ''
    if outcome == 'gap':
        return 'gap', 'stage 1 %s %s not run: %s' % (cell, kind, detail[:400])
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
    files = [p for b in kind_bands(kind) for p in SS.cell_files(cell, ident[1], None, b)]
    lsha = SS.listing_sha1(files) if files else ''
    if os.path.normpath(str(meta.get('cutout_root') or '')) != os.path.normpath(ident[1]) \
            or meta.get('cutout_listing_sha1') != lsha:
        return 'missing', ('stage 1 %s %s: measured from other cutouts than the cell holds now under %s (re-cut since; %d '
                           'cutouts now) - the next run measures it again' % (cell, kind, ident[1], len(files)))
    mp = os.path.join(stage1, cell, _cc().STAGE1_MANIFEST[kind])
    if not os.path.isfile(mp):
        return 'missing', 'stage 1 %s %s: no manifest %s' % (cell, kind, mp)
    for fn in (T.mm1, T.mm2):
        ok, _, det = fn(mp)
        if not ok:
            return 'missing', 'stage 1 %s %s: %s %s' % (cell, kind, os.path.basename(mp), det)
    return 'done', ''


def check(out_root, enum, started, stage1=None, retest=False):
    """(complete, counts, problems, gaps) of a campaign whose start record is ``started`` (its generation, cutout root and
    pin): every unit of ``enum`` terminal - done (this campaign's record, passing check_done with the pin), no cutout
    ('none': counted by NONE_TAGS tag), or a GAP (a gate refused its inputs: listed, never a problem) - no (cell, subband)
    held at the unit list (its inputs never tested, a disagreement of the two exclusion readers among them), and - with
    ``stage1`` - every cell's stage 1 per kind done, 'no <kind>' or not run for a gap. Complete = no problem; the gaps are
    reported beside it."""
    ident = (started['generation'], started['cutout_root'])
    pin = dict(started.get('pin') or {})
    counts = {'units': len(enum.units), 'done': 0, 'none': 0, 'gap': 0, 'incomplete': 0, 'held_enumeration': len(enum.held)}
    counts.update({'none_%s' % t.replace(' ', '_'): 0 for t in NONE_TAGS})
    problems = ['held at enumeration: %s %s (%s)' % (h['cell'], h['subband'], h['reason']) for h in enum.held]
    gaps = []
    for cell, band in enum.units:
        state, text = unit_state(out_root, cell, band, ident)
        if state == 'none':
            tag = text.split(': ', 1)[1]
            counts['none_%s' % next(t for t in NONE_TAGS if tag.startswith(t)).replace(' ', '_')] += 1
        if state == 'done':
            bad = check_done(record.load_record(out_root, 'photometry', 'cutouts.%s.%s' % (cell, band)), band, retest, pin)
            if bad:
                state, text = 'incomplete', 'cutouts.%s.%s: done record, but %s' % (cell, band, '; '.join(bad)[:600])
        key = {'done': 'done', 'none': 'none', 'gap': 'gap', 'gap-noinput': 'gap'}.get(state, 'incomplete')
        counts[key] += 1
        if key == 'gap':
            gaps.append(text)
        elif key == 'incomplete':
            problems.append(text)
    if stage1:
        for cell in enum.cells:
            for kind in ('sb', 'wb'):
                st, why = check_stage1(out_root, stage1, enum, cell, kind, ident)
                key = 'stage1_%s' % st.replace(' ', '_')
                counts[key] = counts.get(key, 0) + 1
                if why:
                    (gaps if st == 'gap' else problems).append(why)
    return not problems, counts, problems, gaps


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m exopipe.phase3.lib.photometry.campaign',
                                 description='The complete-when content test of a cutout-generation campaign (read only).')
    sub = ap.add_subparsers(dest='cmd')
    c = sub.add_parser('check', help='every unit terminal (done with its files present, no cutout, or a gap); stage 1 per cell '
                       'and kind done, no <kind>, or not run for a gap')
    c.add_argument('--out', required=True, help='the campaign\'s --out (records under <out>/_records/photometry)')
    c.add_argument('--cells', help='night cells <LST>_<date> (comma list, patterns, @FILE); default every cube directory')
    c.add_argument('--generation', help='default photometry_campaign.GENERATION')
    c.add_argument('--retest', action='store_true', help='run the content tests of every done unit\'s manifest and flux-scale '
                   'record again (reads every listed cutout)')
    c.add_argument('--no-stage1', action='store_true', help='check the cutouts units only')
    c.add_argument('--max-problems', type=int, default=200, help='problem lines printed (default 200)')
    c.add_argument('--work', help='directory of the engine exclusion reader\'s child log (default a temporary directory, '
                   'removed afterwards; nothing is written under --out)')
    a = ap.parse_args(argv)
    if a.cmd != 'check':
        ap.print_help()
        return 2
    if a.generation:
        K.cfg().set('photometry_campaign.GENERATION', a.generation, source='--generation')
    out = os.path.abspath(a.out)
    try:
        gen = generation()
        started = latest_record(out, gen)
        if started is None:
            raise Refusal('no campaign start record of generation %s under %s' % (gen, os.path.join(
                out, '_records', 'photometry', _cc().RECORD_SUBDIR)))
        present = cube_cells(K.cfg().paths.SNAPSHOT_CUBES_ROOT)
        cells = scope_cells(argparse.Namespace(cells=a.cells, epochs=None, exclude_cells=None), present)
        have = set(present)
        probe = [c for c in (cells if cells is not None else present) if c in have]
        work = a.work or tempfile.mkdtemp(prefix='campaign_check_', dir=os.environ.get('TMPDIR') or None)
        try:                            # both exclusion readers, as the campaign enumerates (a disagreement holds)
            run = record.Run('photometry', ['campaign check'] + list(sys.argv[1:] if argv is None else argv), work, K.cfg())
            eng = engine_exclusions(run, probe) if probe else {}
        finally:
            if not a.work:
                shutil.rmtree(work, ignore_errors=True)
        enum = enumerate_units(cells, engine_excluded=eng, present=present)
    except Refusal as e:
        print('CANNOT CHECK: %s' % e, flush=True)
        return 2
    stage1 = None if a.no_stage1 else os.path.join(out, stage1_dir())
    ok, counts, problems, gaps = check(out, enum, started, stage1, a.retest)
    added, removed = units_changed(started, enum.units, cells)
    if added or removed:
        print('UNITS CHANGED since the campaign start record %s: %d added (e.g. %s), %d removed (e.g. %s)' % (
            started.get('run_id'), len(added), added[:3], len(removed), removed[:3]), flush=True)
    print('GENERATION %s; campaign %s (started %s); cutouts %s; records %s; stage 1 %s' % (
        gen, started.get('run_id'), started.get('started'), started.get('cutout_root'), out, stage1 or 'not checked'))
    print('COUNTS %s' % json.dumps(counts, sort_keys=True))
    for g in gaps:
        print('GAP %s' % g)
    for p in problems[:a.max_problems]:
        print('PROBLEM %s' % p)
    if len(problems) > a.max_problems:
        print('... %d more problems' % (len(problems) - a.max_problems))
    print(summary(ok, counts, problems, gaps), flush=True)
    return 0 if ok else 1


def summary(ok, counts, problems, gaps):
    """The complete-when line: COMPLETE or INCOMPLETE (problems), the gaps, and the units without a cutout counted by why
    (excluded, the night rule's refusal, another zero count) - none of them a gap."""
    none = '%d excluded, %d night-rule refused, %d with no cutout; not gaps' % (
        counts.get('none_excluded', 0), counts.get('none_night_rule', 0), counts.get('none_no_cutout', 0))
    if not ok:
        return 'INCOMPLETE: %d problems, %d gaps (%s)' % (len(problems), len(gaps), none)
    return 'COMPLETE with %d gaps (%s)' % (len(gaps), none)


if __name__ == '__main__':
    sys.exit(main())
