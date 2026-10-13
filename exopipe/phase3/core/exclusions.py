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
    """The parsed exclusion list. ``Exclusions()`` reads ``CFG.paths.DATA_DECISIONS``."""

    def __init__(self, path=None):
        from exopipe.phase3.config import CFG
        self.path = path or CFG.paths.DATA_DECISIONS
        self.subband_only = {_norm_scope(s) for s in CFG.exclusions.SUBBAND_ONLY}
        self.perband_tag = CFG.exclusions.PERBAND_TAG
        self.band_members = {b: {_norm_scope(s) for s in subs} for b, subs in CFG.instrument.SUBBAND_BANDS.items()}
        self.band_members['Full'] = set().union(*self.band_members.values())
        if not os.path.exists(self.path):
            raise Refusal(f'exclusion list not found at {self.path} - never proceed with an empty exclusion set '
                          f'(ruling C-09, Gregg 2026-08-11)')
        self.rows = set()          # (lst, scope without 'MHz', date) with verdict 'exclude'
        self.perband = set()       # the subset tagged [PERBAND]
        self.reasons = {}
        with open(self.path, 'rb') as f:
            raw = f.read()
        self._sha1 = hashlib.sha1(raw).hexdigest()
        try:
            rd = csv.DictReader(raw.decode('utf-8').splitlines())
            for r in rd:
                if (r.get('verdict') or '').strip() != 'exclude':
                    continue
                key = (r['lst'].strip(), _norm_scope(r['subband']), r['date'].strip())
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


def load(path=None):
    """The process-wide Exclusions (re-read when the file changes)."""
    from exopipe.phase3.config import CFG
    p = path or CFG.paths.DATA_DECISIONS
    try:
        m = os.stat(p).st_mtime_ns
    except FileNotFoundError:
        m = None
    hit = _CACHE.get(p)
    if hit is None or hit[0] != m:
        _CACHE[p] = (m, Exclusions(p))
    return _CACHE[p][1]


def is_excluded(lst, scope, date):
    return load().is_excluded(lst, scope, date)


def excluded_constituents(lst, scope, dates):
    return load().excluded_constituents(lst, scope, dates)
