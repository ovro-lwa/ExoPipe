"""Exclusions - the ONLY reader of data_decisions.csv (PHASE3_ARCHITECTURE.md sections 2 and 4; CLAUDE.md 2).

data_decisions.csv is Gregg's: read only, never edited by code. Exclusion is absolute and total - co-adds AND searches
and every downstream product; no code path re-admits an excluded cell.

The rule (identical to the pre-refactor co-add rule, exopipe/phase3/coadd_full_pipeline.py is_excluded, which every
tier must share):
- SUBBAND scope (e.g. '18MHz'): excluded iff that exact (LST, subband, date) row says 'exclude'.
- BAND scope (Red, Green, Blue, Full): excluded iff any of ITS OWN member subbands is excluded, except subbands in
  ``CFG.exclusions.SUBBAND_ONLY`` (18 MHz: Gregg 2026-06-29 - an 18-only exclusion drops the 18 MHz image but keeps
  its Red/Full band image) and rows tagged ``[PERBAND]`` (Gregg 2026-08-26: a per-image exclusion excludes only its
  own scope); or an explicit row for the band scope exists. Per-band rule: Gregg 2026-08-31 (a night bad only above
  50 MHz keeps its Red band).
- A missing or unreadable file is a HARD REFUSAL, never an empty set (ruling C-09, Gregg 2026-08-11).
- The scope of a read (``CFG.exclusions.SCOPES``; production coadd_full_pipeline.load_exclusions(scope), exopipe a8b56d3)
  decides which rows count. None, the default (the co-adds, the deep tier, every reader that names no scope), skips the
  rows tagged ``CFG.exclusions.SNAPSHOT_ONLY_TAG`` (DECIDED-20261113-SNAPSHOT-OPEN-ITEMS-LOWSKY-CONFORMANCE-17H22H), so
  the deep tier is unchanged by them; 'snapshot' (the snapshot, 1-minute and 10-minute tiers on Phase 1 frames) and
  'snapshot-gpu' honor them; 'snapshot-gpu' also skips the rows tagged ``CFG.exclusions.GPU_READMIT_TAG`` (Gregg
  2026-11-10). Any other scope REFUSES. The snapshot tiers read per night (``PerNight``; production
  coadd_full_pipeline.snapshot_exclusion_test, exopipe 36351ca): 'snapshot-gpu' on a cell whose 10-s frames are the GPU
  pipeline's (``gpu_frame_cells``), 'snapshot' on every other night. ``for_tier(tier)`` is the read of a tier's readers.

Every product that depends on an admitted night set records ``Exclusions.fingerprint()`` and the admitted nights, so
an exclusion change makes every cached or adopted result older than it stale (deep requirement (d), 2026-10-13).
"""
import csv
import hashlib
import os

from exopipe.phase3.core.record import Refusal


def _norm_scope(s):
    return str(s).replace('MHz', '').strip()


class Exclusions:
    """The parsed exclusion list. ``Exclusions()`` reads ``CFG.paths.DATA_DECISIONS``; ``scope`` is the read's scope
    (module docstring)."""

    def __init__(self, path=None, scope=None):
        from exopipe.phase3.config import CFG
        if scope not in CFG.exclusions.SCOPES:
            raise Refusal(f'REFUSING: unknown exclusion scope {scope!r} (known: {CFG.exclusions.SCOPES})')
        self.scope = scope
        self.path = path or CFG.paths.DATA_DECISIONS
        self.subband_only = {_norm_scope(s) for s in CFG.exclusions.SUBBAND_ONLY}
        self.perband_tag = CFG.exclusions.PERBAND_TAG
        self.band_members = {b: {_norm_scope(s) for s in subs} for b, subs in CFG.instrument.SUBBAND_BANDS.items()}
        self.band_members['Full'] = set().union(*self.band_members.values())
        if not os.path.exists(self.path):
            raise Refusal(f'exclusion list not found at {self.path} - never proceed with an empty exclusion set '
                          f'(ruling C-09, Gregg 2026-08-11)')
        self.rows = set()          # (lst, scope without 'MHz', date) with verdict 'exclude' that this read's scope counts
        self.keys_any = set()      # the same key for EVERY row, any verdict (a 'keep' is Gregg's readmission): a rule that
                                   # must not re-propose a decided night skips these (propagate_date_outliers.py:40-42)
        self.perband = set()       # the subset tagged [PERBAND]
        self.reasons = {}
        with open(self.path, 'rb') as f:
            raw = f.read()
        self._sha1 = hashlib.sha1(raw).hexdigest()
        try:
            rd = csv.DictReader(raw.decode('utf-8').splitlines())
            for r in rd:
                key = (r['lst'].strip(), _norm_scope(r['subband']), r['date'].strip())
                self.keys_any.add(key)
                if (r.get('verdict') or '').strip() != 'exclude':
                    continue
                if scope is None and CFG.exclusions.SNAPSHOT_ONLY_TAG in (r.get('reason') or ''):
                    continue
                if scope == 'snapshot-gpu' and CFG.exclusions.GPU_READMIT_TAG in (r.get('reason') or ''):
                    continue
                self.rows.add(key)
                self.reasons[key] = r.get('reason', '')
                if self.perband_tag in (r.get('reason') or ''):
                    self.perband.add(key)
        except Exception as e:
            raise Refusal(f'exclusion list {self.path} unreadable ({e}) - refusing (ruling C-09)')

    # ---- the rule -----------------------------------------------------------------------------------------------
    def is_excluded(self, lst, scope, date):
        fs = _norm_scope(scope)
        if fs in self.band_members:
            bad = {s for (l, s, d) in self.rows if l == lst and d == date and (l, s, d) not in self.perband}
            return bool(self.band_members[fs] & (bad - self.subband_only)) or (lst, fs, date) in self.rows
        return (lst, fs, date) in self.rows

    def excluded_constituents(self, lst, scope, dates):
        """The subset of ``dates`` excluded for (lst, scope). Non-empty for a master's constituents = the master
        ingested data that data_decisions drops: callers (publish, search) MUST refuse."""
        return sorted(d for d in dates if self.is_excluded(lst, scope, d))

    def admitted(self, lst, scope, dates):
        return sorted(d for d in dates if not self.is_excluded(lst, scope, d))

    def is_whole_night_excluded(self, lst, date):
        return all(self.is_excluded(lst, b, date) for b in ('Red', 'Green', 'Blue'))

    def excluded_nights(self, lst, scope):
        """Every date with a row that excludes (lst, scope) under the rule (dates seen in the file for this LST)."""
        dates = {d for (l, s, d) in self.rows if l == lst}
        return sorted(d for d in dates if self.is_excluded(lst, scope, d))

    def fingerprint(self):
        return {'path': self.path, 'sha1': self._sha1, 'n_exclude_rows': len(self.rows)}

    # ---- compatibility with the pre-refactor API (coadd_full_pipeline.load_exclusions/is_excluded) ---------------
    def as_legacy_set(self):
        return set(self.rows)


_CACHE = {}


def load(path=None, scope=None):
    """The process-wide Exclusions of one scope (re-read when the file changes)."""
    from exopipe.phase3.config import CFG
    p = path or CFG.paths.DATA_DECISIONS
    try:
        m = os.stat(p).st_mtime_ns
    except FileNotFoundError:
        m = None
    hit = _CACHE.get((p, scope))
    if hit is None or hit[0] != m:
        _CACHE[(p, scope)] = (m, Exclusions(p, scope=scope))
    return _CACHE[(p, scope)][1]


def gpu_frame_cells(path=None):
    """{(LST, date)} whose 10-s frames are the observatory GPU pipeline's (``CFG.paths.SNAPSHOT_GPU_FRAME_CELLS``; production
    coadd_full_pipeline.gpu_frame_cells, exopipe 36351ca). A missing file is an empty set: every night then reads scope
    'snapshot', the stricter one."""
    from exopipe.phase3.config import CFG
    p = path or CFG.paths.SNAPSHOT_GPU_FRAME_CELLS
    out = set()
    if os.path.exists(p):
        with open(p) as fh:
            for ln in fh:
                f = ln.split()
                if len(f) >= 2 and not ln.startswith('#'):
                    out.add((f[0], f[1]))
    return out


def refuse_frame_overrides():
    """Refuses when the environment overrides the GPU-frame cell list (P3_SNAP_GPU_CELLS) or the frame root
    (P3_SNAP_PHASE1_ROOT, run-wide): the in-place snapshot engines (clv15 phase3_snapshot_search.frames_from_gpu) and the
    co-add port follow them, ``gpu_frame_cells`` reads ``CFG.paths.SNAPSHOT_GPU_FRAME_CELLS``, so their per-night verdicts
    on a re-admitted row would differ within one run."""
    from exopipe.phase3.config import CFG
    for var, want in (('P3_SNAP_GPU_CELLS', CFG.paths.SNAPSHOT_GPU_FRAME_CELLS),
                      ('P3_SNAP_PHASE1_ROOT', CFG.paths.PHASE1_ROOT)):
        v = os.environ.get(var)
        if v and os.path.realpath(v) != os.path.realpath(want):
            raise Refusal(f'REFUSING: {var}={v} in the environment (config: {want}) - the in-place snapshot engines would '
                          f'choose the exclusion scope per night from it and the per-night read here would not; unset it')


class PerNight:
    """The snapshot tiers' read, per night (production coadd_full_pipeline.snapshot_exclusion_test, exopipe 36351ca, and the
    per-cell readers of the 10-minute tree, wideband, stacker, job lists, catalog/publish gate and snapshot group search):
    scope 'snapshot-gpu' on a GPU-frame cell (``gpu_frame_cells``), 'snapshot' on every other night. The verdicts of
    Exclusions; ``path`` is the 'snapshot' read's, ``gpu_path`` the cell list read (None when ``gpu_cells`` is given), and
    ``fingerprint()`` covers both. ``fresh``: read the file now, not the process cache."""

    def __init__(self, path=None, gpu_cells=None, fresh=False):
        from exopipe.phase3.config import CFG
        self.gpu_path = None
        if gpu_cells is None:
            refuse_frame_overrides()
            self.gpu_path = CFG.paths.SNAPSHOT_GPU_FRAME_CELLS
        self.snapshot = Exclusions(path, scope='snapshot') if fresh else load(path, scope='snapshot')
        self.gpu = Exclusions(path, scope='snapshot-gpu') if fresh else load(path, scope='snapshot-gpu')
        self.gpu_cells = gpu_frame_cells() if gpu_cells is None else set(gpu_cells)
        self.path = self.snapshot.path

    def of(self, lst, date):
        """The Exclusions of one (LST, night)."""
        return self.gpu if (lst, date) in self.gpu_cells else self.snapshot

    def is_excluded(self, lst, scope, date):
        return self.of(lst, date).is_excluded(lst, scope, date)

    def excluded_constituents(self, lst, scope, dates):
        return sorted(d for d in dates if self.is_excluded(lst, scope, d))

    def admitted(self, lst, scope, dates):
        return sorted(d for d in dates if not self.is_excluded(lst, scope, d))

    def is_whole_night_excluded(self, lst, date):
        return self.of(lst, date).is_whole_night_excluded(lst, date)

    def fingerprint(self):
        """The 'snapshot' read's (the whole file's sha1) and the GPU-frame cell set's: either changes verdicts."""
        cells = '\n'.join(sorted(f'{l} {d}' for l, d in self.gpu_cells)).encode()
        return dict(self.snapshot.fingerprint(), gpu_frame_cells=hashlib.sha1(cells).hexdigest(),
                    n_gpu_frame_cells=len(self.gpu_cells))


def for_tier(tier, fresh=False, path=None):
    """The read of a tier's readers: the default (``Exclusions``) for the deep tier or no tier, ``PerNight`` for the snapshot
    tiers (``CFG.exclusions.SNAPSHOT_SCOPE_TIERS``: the snapshot 10-s and 1-minute tiers, the sub-LST 10-minute tier). Any
    other tier refuses (a reader of an unknown tier must not fall back to the default silently). ``fresh``: read now, not
    the process cache; ``path``: another exclusion list than ``CFG.paths.DATA_DECISIONS`` (tests)."""
    from exopipe.phase3.config import CFG
    if tier in (None, 'deep'):
        return Exclusions(path) if fresh else load(path)
    if tier in CFG.exclusions.SNAPSHOT_SCOPE_TIERS:
        return PerNight(path, fresh=fresh)
    raise Refusal(f'REFUSING: no exclusion read for tier {tier!r} (deep, or one of {CFG.exclusions.SNAPSHOT_SCOPE_TIERS})')


def is_excluded(lst, scope, date):
    return load().is_excluded(lst, scope, date)


def excluded_constituents(lst, scope, dates):
    return load().excluded_constituents(lst, scope, dates)
