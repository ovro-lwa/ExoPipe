"""DEEP OMISSION - one stage shared by the short tiers (10-s, 1-minute, 10-minute): what the deep transient search refused
is omitted from their transient search, per hour, band and Stokes, unless the tier's own data clear it.

Rulings. Gregg 2026-10-24 ~20:30 UTC, verbatim: "Omission from the deep transient search is sufficient for omission from the
10s/1-min/10-min search, assessed on a per hour, per band basis" and "Approved with one critical exception - if removing noisy
frames is addressing the cause of the issues with the deep image, then there is no need to flag the full hour in the
10-s/1-min/10-min pipeline" (R-GREGG-20261024-DEEP-OMISSION-APPLIES-TO-SHORT-TIERS-WITH-FRAME-EXCEPTION); the design "All
approved. Release." ~23:45 UTC, the exception "unless the rate is anomalously high", each gate in its own Stokes scope, the
deep's processing failures not propagated (R-GREGG-20261024-DEEP-OMISSION-DESIGN-APPROVED-AND-10MIN-HOLD-RELEASED); the rebuild
from the merged refusal record (R-GREGG-20261031-SNAPSHOT-RESEARCH-LC-DEEPOMIT-GPU-RELEASE). Production: exopipe_ops
sublst_deep_omission.py + sublst_deep_omission_drop.py (10-minute; c6d2c9e, 0d4f1a9), snapshot_deep_omission.py and step 0 of
snapshot_combined_view.py (10-s and 1-minute; 37546f6) - ported here line for line (DECISIONS and DROP below); the parity
test is tests/test_deep_omission.py.

DECISIONS, for every (LST, band, night, Stokes) the deep search refused (load_refusals: the LST-band's LATEST attempt, the
propagating gates catalog_omission.PROPAGATE, each in its own Stokes scope - searchtier by its Stokes column, sanity_I -> I,
sanity_V -> V, the others both) and each tier T of the short tier:
  (1) noise: the night as the tier admitted it - 10-s / 1-minute: the median global rms of the ADMITTED rows of the search's
      noise table (<SNAPSHOT_FLEET_OUT>/<LST>_<night>/noise_<band>_<S>_median[_1min].csv); 10-minute: the median central rms of
      the windows the search SEARCHED (v_deep_images.csv rows, night and residual from the window tree's frames.csv) - must be
      <= median + SIGMA x 1.4826 MAD over the LST's nights with that record (fewer than 3 nights: no threshold, not passed);
  (2) rate: the night's served detections per admitted row / window <= median + SIGMA x max(1.4826 MAD, RATE_FLOOR_PER_NIGHT /
      the median rows or windows per night) over those nights (fewer than 3 nights: passed);
  KEEP when both pass, else OMIT; no record of the night in the tier: ABSENT (nothing searched, nothing to drop).
  WIDEBAND: a deep-refused band group night is decided the same way on its own record; a subband whose verdict is one of
  catalog_omission.WIDEBAND_MEMBER_VERDICTS for a night (production's OMIT or ABSENT - the ABSENT part pending Gregg, as the rate
  floor RATE_FLOOR_PER_NIGHT; WIDEBAND_EXCEPT excepted) also omits that night's detections of the band group it is a MEMBER of
  - 10-s / 1-minute: by the cell's wideband_members_<S>.csv, both tiers; 10-minute: by each wideband window's own WBMEMB card.
  (A deep refusal of the Full band never omits anything here: the short tiers have no Full band, so its decisions are ABSENT,
  and Full is a member of no group - production's reading, Q review F11.)
DROP (the data level, before the cumulative catalogs, identities and pages):
  10-s / 1-minute (the combined view's step 0 and its per-tier omission parts): every served batch identity's detections (the
      identity's review_dir detections.csv rows; a detection = cand_id + peak position) whose (LST, subband or band, night,
      Stokes, tier) is OMIT are omitted; an identity left with none is dropped; per tier, a batch source left with none is
      dropped, a partial one whose best detection went is re-pointed to its remaining detection of largest |S/N| (its survivor
      gates NOT re-run - recorded; see OWED below), member rows and panels of omitted keys are dropped. This stage COMPUTES the
      drop and WRITES it as the kept product deep_omission_kept.csv; the drop is APPLIED where the combined view is assembled -
      the snapshot tier's cumulative step `serve --tier snapshot --family comb` (lib/serve/snapshot_view.py, the port of
      snapshot_combined_view.py; refactor chat 2026-11-03), which reads the kept product instead of recomputing it, checks the
      served set against its digests and refuses any row that does not apply exactly (Q review F2, N2).
  10-minute: the cumulative merge drops the OMIT rows per LST before concatenation (sublst_deep_omission_drop.py IN PLACE,
      reading the decision table this stage writes; `serve --tier sublst` build passes it as SUBLST_DEEP_OMISSION). This stage
      records the detections that drop removes (the member rows of every LST run whose key is OMIT).

Where it sits. Production applies the rule where the cumulative product is assembled, after each unit's catalogue chain and
pages (the served detections the rate condition counts exist only then) and before the cumulative catalogues, identities and
pages: the combined snapshot view (snapshot_combined_view.py step 0) and the cumulative 10-minute merge (sublst_cumulative.sh).
Here: `exopipe3 catalog --tier snapshot --stages omit --omit-runs RUN,RUN..` (the served batch runs of the view) and
`exopipe3 catalog --tier sublst --stages omit --lsts ...` (the LST runs of the cumulative); the 10-minute cumulative build calls
tenmin_table() before its merge; the snapshot tier's cumulative step (serve --tier snapshot --family comb: its omit stage calls
run_snapshot, its build applies deep_omission_kept.csv).

Products (in the stage's output directory --out; production names): deep_omission.csv (class omission_decisions: one row per
decision, production's columns and formats - byte for byte), deep_omission_dropped.csv (class omission_record: every omitted
detection and every consequence, with the decision text of its own key, column 'why'; 10-s / 1-minute: the combined view's
record columns + why; 10-minute: the merge drop's), deep_omission_summary.txt (line 1 = the page line) and .json (counts);
10-s / 1-minute also deep_omission_kept.csv (class omission_kept, KEPT_COLS: what the cumulative step applies, every row with
the served run and the catalogue file it acts on - 'detection kept' (each kept detection of each kept identity: the merge's
input), 'source dropped', 'source re-pointed' (fields: the survivors row's new values, JSON), 'member row dropped', 'panel
dropped'; its content test traces every row to the decisions, the record and the summary, refuses a detection key kept twice,
and checks kept + omitted against the served set's fingerprint in the summary json - order-free multiset digests of the
served, kept and omitted detections, key 'digests', which the cumulative step also checks against the served runs as it
reads them; Q fix verification N2).
Records: <out>/_records/catalog/<label>.omit.json - every refusal file read (and the larger groups - noise tables, member lists,
window tables, served tables - as content digests in meta input_digests). --omit-refusals DIR is an override of the RULED
catalog_omission.REFUSALS through the config (as --set: --ruling-for or a non-production --out; recorded), Q review F8.

Gates (refuse, exit 2): G1 the refusal record missing, empty or holding a row not of its file's LST, or (an ordinary run) lying
under an injected run's start record, however it was named (Q fix verification N1); G2 a served run without its
catalogue (or a 10-minute LST run without its master_catalog.csv); G3 a served identity without detections, or a detection in
two identities; G4 a per-tier source or member row of no served identity; G5 kept + omitted != served; G6 served detections in
an ABSENT key (catalog_omission.ABSENT_WITH_SERVED = 'refuse', the refactor's PROPOSED value pending Gregg: the exception test
cannot be evaluated - never 'nothing to drop'; 'production' keeps them and counts them, as production does - the parity file's
value; the site is marked WAVE2 G6).

Per instant (Gregg 2026-10-18: a veto removes that instant's detections across frequency only, never other times or tiers;
2026-10-24: the omission is assessed "on a per hour, per band basis"): every removal names its OWN key - the (LST, night) hour
that holds the detection, its own band (or, across frequency only, the band group of an omitted member subband of the same
night), its own Stokes and its own tier (a 10-s decision never removes a 1-minute detection) - and carries the decision texts
of that key (column 'why', from why_part / member_gates / *_member_reason below, the templates the per-instant content test
reads: tests_per_instant.omission_per_instant, O1-O3, registered on omission_record).

OWED, NOT DONE - the re-run of a partial source's survivor gates (Q review F6). plan/CUTOVER_READINESS.md, "DEEP OMISSION CARRIES
TO THE SHORTER TIERS", Port: "Partial sources re-run their survivor gates, which today's drops do not" - the reading of Gregg's
2026-10-24 ruling (R-GREGG-20261024-DEEP-OMISSION-APPLIES-TO-SHORT-TIERS-WITH-FRAME-EXCEPTION, verbatim: "Omission from the deep
transient search is sufficient for omission from the 10s/1-min/10-min search") that an omitted detection was never searched, so
a source must stand on its remaining detections. Production does not: it re-points and records - exopipe_ops
sublst_deep_omission_drop.py:17-18 (its docstring): "its survivor gates are NOT re-run (recorded) - it stays visible, so no real
source is lost"; clv15 fix_ledger IMPLEMENTED-DEEP-OMISSION-DROP-IN-THE-10-MINUTE-CUMULATIVE-20261025: "(its survivor gates are
not re-run, recorded)"; IMPLEMENTED-DEEP-OMISSION-10S-1MIN-IN-THE-COMBINED-SNAPSHOT-VIEW-20261025: "(survivor gates not re-run,
recorded)" (attribution corrected, Q fix verification N4). What is
owed, per partial source (some detections omitted, at least one kept - re-pointed or not), on its KEPT detections only, by each
tier's own rule for a source that lost instants:
  10-s / 1-minute: the snapshot chain's re-build of a source after a per-instant veto (clv15 phase3_snapshot_chain.py 9465ae6,
      :3888-3935 "SOME INSTANTS SURVIVE: the source is re-built on them": route_repeats, the reflection-band floor
      reflection_band_source_gate, the seed rule (best |S/N| >= the seed floor), the anchor anchor_on_members, then the route
      pass - a source that qualifies for no route is removed), on the identity and each of its per-tier sources;
  10-minute: the per-tier catalogue's survivor gates (ops/build_source_catalogs_v3.py, config catalog_associate): |best S/N| >=
      SURVIVOR_SNR (7.0) or promoted (E recomputed on the kept members < E_PROMOTE), and the 7-8 sigma size gate of an
      unpromoted single (SIZE_GATE_7_8), inside the merge's drop (sublst_deep_omission_drop.py, in place).
A source that fails is dropped (recorded 'source dropped', note 'survivor gates failed on its kept detections'), its page,
panels and member rows with it; one that passes keeps its re-pointed fields. It changes the served product against production,
so it is an amendment and needs Gregg's ruling first (DECISIONS_PENDING entry "Deep omission: the survivor-gate re-run of a
partial source", named in refactor_reviews_20261103/Q_deepomit_fixes.md); not implemented in this port for that reason and
because it needs the chain's source re-build ported. Where it would act: every identity with both an omitted and a kept
detection (a 'detection' row in the record and 'detection kept' rows in the kept product), the 'source re-pointed' rows among
them (B4X+B6H: none on the 10-25 or today's inputs - no partial source lost its best detection).

PRODUCTION BUGS CARRIED FOR PARITY (Q review F1, F4; production's owner fixes them first, then this port follows - the decision
table stays byte-identical to production until then; each site is marked '# PRODUCTION BUG F1/F4, pending the production fix'
and a test pins today's behavior, so the later change is visible):
  F1 the wideband consequence is keyed on the night's union of members, not on the detection's own frame / block / window: a
     wideband detection whose own frame was stacked without the omitted subband is dropped too (decide_snapshot, decide_tenmin);
  F4 deep_value / deep_threshold are written in the order the refusal files were listed (glob, unsorted), the gates sorted, so
     value i need not belong to gate i, and deep_threshold_meaning follows the first sorted gate (_base, refusal_files, _meaning).
  F10 (text, production's): with fewer than 3 nights there is no noise threshold (nan, never passed) and the reason still reads
     RMS_ABOVE ('tier rms above threshold') - production's byte-identical table text and the 10-minute merge's 'why', so it is
     not changed here (_verdict).
"""
import collections
import csv
import glob
import hashlib
import io
import itertools
import json
import os
import re
import sys
import time

TIER_SNAPSHOT, TIER_TENMIN = 'snapshot', 'sublst'

# ---- the decision texts (production's words; the per-instant content test reads these f-strings and constants with ast) ----
PASSES = 'passes both'
RMS_ABOVE = 'tier rms above threshold'
RATE_HIGH = 'served rate anomalously high'
TENMIN_NO_WINDOWS = 'no 10-minute tier windows for this band and Stokes (nothing to drop)'
TENMIN_NIGHT_ABSENT = 'the night has no admitted window in the tier (already absent)'
SANITY_MEANING = 'the cell median of the statistic (the gate refuses above 10 x it)'
GATE_MEANING = "the gate's own threshold"
SNAPSHOT_RATE_DEF = ("served detections of the night (every batch identity's detections given to the combined view) per "
                     "admitted row of the tier's noise table, that (LST, band, Stokes, tier)")
TENMIN_RATE_DEF = ('served detections of the night (published per-LST run master_catalog) per searched window (v_deep_images.csv '
                   'rows), that (LST, band, Stokes)')


def snapshot_absent_reason(tier):
    return f'no searched {tier} data of this night, band and Stokes in the tier (nothing to drop)'


def member_word(band, verdict):
    return f'{band} {verdict}'


def member_gates(members):
    return f'member {members}'


def snapshot_member_reason(members, group):
    return f"wideband-member: {members} this night, a member of the cell's {group} frames"


def tenmin_member_reason(members):
    return f'wideband-member: {members} this night, in the WBMEMB of its wideband windows'


def why_part(gates, reason):
    """One OMIT decision's text in the record's 'why' (production's WHY: '%s: %s' % (gates, reason))."""
    return f'{gates}: {reason}'


WHY_SEP = ' | '
SNAPSHOT_PARTIAL_NOTE = 'best detection omitted; re-pointed (survivor gates not re-run)'
SNAPSHOT_COLS = ['lst', 'band', 'night', 'stokes', 'gates', 'deep_value', 'deep_threshold', 'deep_attempt', 'tier', 'n_admitted',
                 'tier_median_rms', 'tier_rms_threshold', 'rms_pass', 'rate', 'rate_threshold', 'rate_pass', 'verdict', 'reason',
                 'deep_threshold_meaning', 'rate_definition']
TENMIN_COLS = ['lst', 'band', 'night', 'stokes', 'gates', 'deep_value', 'deep_threshold', 'deep_attempt', 'n_windows',
               'tier_median_rms', 'tier_rms_threshold', 'rms_pass', 'rate', 'rate_threshold', 'rate_pass', 'verdict', 'reason',
               'deep_threshold_meaning', 'rate_definition']
EMPTY_SNAPSHOT_HEADER = 'lst,band,night,stokes,tier,verdict,reason\n'      # snapshot_deep_omission.write, no rows
EMPTY_TENMIN_HEADER = 'lst,band,night,stokes,verdict,reason\n'             # sublst_deep_omission.py main, no rows
# the combined view's record columns (snapshot_combined_view.py, its header when a detection is omitted) + 'why'
RECORD_COLS = ['kind', 'batch', 'identity', 'cand_id', 'lst', 'band', 'night', 'stokes', 'tier', 'snr', 'source', 'note', 'why']
TENMIN_RECORD_COLS = ['kind', 'lst', 'stokes', 'view', 'source_id', 'band', 'cand_id', 'night', 'why', 'detail']
# the KEPT PRODUCT of the 10-s / 1-minute drop (deep_omission_kept.csv, class omission_kept; Q review F2): what the snapshot
# tier's cumulative step (the combined view's port) applies. Every row names the served run and the file it acts on:
#   'detection kept'      run, batch, identity, cand_id, ra_peak / dec_peak (the detection's key dkey), lst, band (subband or
#                         band), night, stokes, tier, snr - each kept detection of each kept identity (the merge's input)
#   'source dropped'      run, batch, identity, catalog (<cdir>/<band>_sources_survivors.csv), source_id, stokes, tier, note
#   'source re-pointed'   ... source_id, cand_id / snr = its new best; fields = JSON of the survivors row's new values
#                         (best_cand_id, best_snr, ts, subband, peak_jy_best, deep_omission); note = the record's
#   'member row dropped'  run, batch, catalog (<cdir>/<band>_source_members.csv), source_id, cand_id, stokes, tier, the key
#                         (lst, band, night) when its own key is OMIT, else note 'identity left with no detection'
#   'panel dropped'       run, batch, catalog (the source's review_dir), source_id, stokes, tier, the key, note = the file
KEPT_COLS = ['kind', 'run', 'batch', 'identity', 'catalog', 'source_id', 'cand_id', 'ra_peak', 'dec_peak', 'lst', 'band', 'night',
             'stokes', 'tier', 'snr', 'note', 'fields']
KEPT_KINDS = ('detection kept', 'source dropped', 'source re-pointed', 'member row dropped', 'panel dropped')
# a snapshot detection's key from its cand_id (snapshot_combined_view.py CIDK) and a panel's (PANEL); a 10-minute detection's
# (sublst_deep_omission_drop.CID)
SNAP_CID = re.compile(r'^(\d\dh)_(\d{4}-\d\d-\d\d)_([A-Za-z0-9]+)_([IV])_(10s|1min)_')
SNAP_PANEL = re.compile(r'^rank\d+_(\d\dh)_(\d{4}-\d\d-\d\d)_([A-Za-z0-9]+?)\d{14}')
TENMIN_CID = re.compile(r'^([IV])\w*?_(\d\dh)_(\d{4}-\d\d-\d\d)T\d\d_([^_]+)_\d+$')
MEMBER_SOURCE = re.compile(r'^(.*)_(\d{2}h_\d{4}-\d{2}-\d{2})_([^_]+)$')

csv.field_size_limit(10 ** 9)


def _cfg():
    from exopipe.phase3.config import CFG
    return CFG


def _c():
    return _cfg().catalog_omission


def groups():
    """{band group: [member subbands]} (instrument.SUBBAND_BANDS; production GROUPS, drift-tested)."""
    return {g: list(v) for g, v in _cfg().instrument.SUBBAND_BANDS.items()}


def refuse(msg):
    from exopipe.phase3.core.record import Refusal
    raise Refusal(msg)


# =====================================================================================================================
# reading: every file once, its content digest kept for the record
# =====================================================================================================================

class Reader:
    """Reads text files once, keeping (size, sha1) of each by group for the unit record (no second read). ``prefetch`` reads
    many files (and ``listdirs`` lists many directories) on catalog_omission.READ_WORKERS threads ahead of their use."""

    def __init__(self, workers=None):
        self.files = collections.defaultdict(dict)       # group -> {path: (size, sha1)}
        self.workers = workers or _c().READ_WORKERS
        self._pre = {}                                   # path -> bytes | None (absent), read ahead

    @staticmethod
    def _read(path):
        try:
            with open(path, 'rb') as f:
                return f.read()
        except FileNotFoundError:
            return None

    def prefetch(self, paths):
        from concurrent.futures import ThreadPoolExecutor
        todo = [p for p in dict.fromkeys(paths) if p not in self._pre]
        with ThreadPoolExecutor(self.workers) as ex:
            for p, b in zip(todo, ex.map(self._read, todo, chunksize=64)):
                self._pre[p] = b

    def listdirs(self, dirs):
        """{directory: sorted names} of every directory (read ahead in parallel)."""
        from concurrent.futures import ThreadPoolExecutor
        dirs = list(dict.fromkeys(dirs))
        with ThreadPoolExecutor(self.workers) as ex:
            return dict(zip(dirs, ex.map(lambda d: sorted(os.listdir(d)) if os.path.isdir(d) else None, dirs, chunksize=64)))

    def text(self, path, group, missing_ok=False):
        b = self._pre.pop(path) if path in self._pre else self._read(path)
        if b is None:
            if missing_ok:
                return None
            raise FileNotFoundError(path)
        self.files[group][path] = (len(b), hashlib.sha1(b).hexdigest())
        return b.decode('utf-8')

    def rows(self, path, group, missing_ok=True):
        t = self.text(path, group, missing_ok=missing_ok)
        return [] if t is None else list(csv.DictReader(io.StringIO(t, newline='')))

    def reader_rows(self, path, group):
        return list(csv.reader(io.StringIO(self.text(path, group), newline='')))

    def digest(self, group):
        """(n files, sha256 over the sorted (path, size, sha1)) of one group."""
        h = hashlib.sha256()
        items = sorted(self.files.get(group, {}).items())
        for p, (n, s) in items:
            h.update(f'{p}\0{n}\0{s}\n'.encode())
        return len(items), h.hexdigest()

    def record(self, u, roots):
        """Every group into the unit record: a group of up to 300 files file by file (inputs), a larger one as its content
        digest (meta input_digests: root, files, sha256 over (path, size, sha1)) with its root's fingerprint as an input."""
        for g, files in sorted(self.files.items()):
            if len(files) <= 300:
                for p in sorted(files):
                    u.add_input(p)
            else:
                n, d = self.digest(g)
                root = roots.get(g) or os.path.commonpath(list(files))
                u.meta.setdefault('input_digests', {})[g] = {'root': root, 'files': n, 'content_sha256': d}
                u.add_input(root)


# =====================================================================================================================
# DECISIONS
# =====================================================================================================================

def mad_thr(vals, sigma=None):
    """median + SIGMA x 1.4826 MAD of the finite values; nan with fewer than 3 (sublst_deep_omission.mad_thr)."""
    import numpy as np
    sigma = _c().SIGMA if sigma is None else sigma
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    if v.size < 3:
        return np.nan
    m = np.median(v)
    return m + sigma * 1.4826 * np.median(np.abs(v - m))


def _rate_floor_problem(v):
    import math
    if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or v < 0:
        return f'catalog_omission.RATE_FLOOR_PER_NIGHT must be a finite number >= 0 (detections per night), not {v!r}'
    return None


def _member_verdicts_problem(v):
    try:
        v = tuple(v)
    except TypeError:
        v = (v,)
    if 'OMIT' not in v or not set(v) <= {'OMIT', 'ABSENT'}:
        return ("catalog_omission.WIDEBAND_MEMBER_VERDICTS must be ('OMIT',) or ('OMIT', 'ABSENT') (the card: a subband "
                f"omitted drops the night's wideband detections that include it), not {v!r}")
    return None


def _absent_mode_problem(v):
    if v not in ('refuse', 'production'):
        return f"catalog_omission.ABSENT_WITH_SERVED must be 'refuse' or 'production', not {v!r}"
    return None


def check_values(cfg=None):
    """[problems] of the catalog_omission values the stage reads only when it runs - RATE_FLOOR_PER_NIGHT,
    WIDEBAND_MEMBER_VERDICTS (production's values, pending Gregg) and ABSENT_WITH_SERVED (proposed). cli.main refuses the step on
    any (exit 2, before the run, as B-30 / B-32 do for theirs): an invalid value given with --set or --config never starts a run
    that would stop late with an error, and the floor is checked even where no cell has 3 nights (Q fix verification N3)."""
    if cfg is None:
        from exopipe.phase3.config import CFG as cfg
    c = cfg.catalog_omission
    return [p for p in (_rate_floor_problem(c.RATE_FLOOR_PER_NIGHT), _member_verdicts_problem(c.WIDEBAND_MEMBER_VERDICTS),
                        _absent_mode_problem(c.ABSENT_WITH_SERVED)) if p]


def rate_floor():
    """catalog_omission.RATE_FLOOR_PER_NIGHT (production's value, pending Gregg; 0 = the card's rule without a floor)."""
    v = _c().RATE_FLOOR_PER_NIGHT
    bad = _rate_floor_problem(v)
    if bad:
        raise ValueError(bad)
    return float(v)


def member_verdicts():
    """catalog_omission.WIDEBAND_MEMBER_VERDICTS (production's ('OMIT', 'ABSENT'), the ABSENT part pending Gregg): the verdicts
    of a member subband night that omit its band group's detections. 'OMIT' is always one (the card's rule)."""
    v = tuple(_c().WIDEBAND_MEMBER_VERDICTS)
    bad = _member_verdicts_problem(v)
    if bad:
        raise ValueError(bad)
    return v


def rate_thr(rates, rows_per_night):
    """median + SIGMA x max(1.4826 MAD, RATE_FLOOR_PER_NIGHT / max(median rows per night, 1)); nan with fewer than 3 rates."""
    import numpy as np
    c = _c()
    rv = np.asarray(list(rates), float)
    if rv.size < 3:
        return np.nan
    nwm = float(np.median(list(rows_per_night)))
    return np.median(rv) + c.SIGMA * max(1.4826 * np.median(np.abs(rv - np.median(rv))), rate_floor() / max(nwm, 1.0))


def refusal_files(rdir, lst):
    """The record's files of one LST, in the order production reads them (glob - the directory's listing order, unsorted)."""
    # PRODUCTION BUG F4, pending the production fix (sublst_deep_omission.load_refusals reads glob order; Q review F4): this order
    # becomes the order of deep_value / deep_threshold in _base, while the gates are written sorted - kept for byte parity
    return glob.glob(os.path.join(rdir, '%s_*_excluded.csv' % lst))


def load_refusals(rdir, lst, reader=None):
    """{(band, night, S): [(gate, value, threshold, attempt)]} of the LST's propagating refusals, the LST-band's latest attempt
    only (sublst_deep_omission.load_refusals; a row's attempt is its last field)."""
    reader = reader or Reader()
    prop = tuple(_c().PROPAGATE)
    rows, latest = collections.defaultdict(list), {}
    for f in refusal_files(rdir, lst):
        b = os.path.basename(f)[len(lst) + 1:-len('_excluded.csv')]
        band, gate = b.split('_', 1)
        for r in reader.reader_rows(f, 'refusals'):
            if not r or r[0] != lst:
                if r:
                    refuse(f'G1: the refusal record {f} holds a row not of its LST {lst}: {",".join(r)[:120]}')
                continue
            att = r[-1]
            latest[band] = max(latest.get(band, ''), att)
            rows[(band, gate)].append(r)
    out = collections.defaultdict(list)
    for (band, gate), rr in rows.items():
        if gate not in prop:
            continue
        for r in rr:
            if r[-1] != latest[band]:
                continue
            night = r[2]
            if gate == 'searchtier':
                S = [r[3]]
                val, thr = r[4], r[5]
            elif gate == 'sanity_I':
                S = ['I']
                val, thr = (r[3], r[4]) if len(r) >= 6 else ('', '')
            elif gate == 'sanity_V':
                S = ['V']
                val, thr = (r[3], r[4]) if len(r) >= 6 else ('', '')
            else:
                S = ['I', 'V']
                val, thr = (r[3], r[4]) if len(r) >= 6 else ('', '')
            for s in S:
                out[(band, night, s)].append((gate, val, thr, latest[band]))
    return out


def _base(lst, band, night, S, gates):
    # PRODUCTION BUG F4, pending the production fix (sublst_deep_omission.py:155-156, snapshot_deep_omission.decide): gates are
    # written sorted, deep_value / deep_threshold in the record's read order (refusal_files), so value i need not belong to gate
    # i (00h 27 MHz 2025-10-15 I: gates 'beam;searchtier', deep_value '2895.8;3.96' - beam is 3.96). Kept for byte parity;
    # pinned by tests/test_deep_omission.py test_production_bugs_pinned.
    return dict(lst=lst, band=band, night=night, stokes=S, gates=';'.join(sorted({g[0] for g in gates})),
                deep_value=';'.join(g[1] for g in gates), deep_threshold=';'.join(g[2] for g in gates),
                deep_attempt=gates[0][3])


def _meaning(r):
    # PRODUCTION BUG F4, pending the production fix: the meaning follows the FIRST SORTED gate, which may be another gate than
    # the threshold it describes (e.g. 'beam;sanity' reads GATE_MEANING for the sanity gate's cell median too)
    g0 = r['gates'].split(';')[0] if r['gates'] else ''
    return '' if not r['deep_threshold'] else SANITY_MEANING if g0.startswith('sanity') else GATE_MEANING


def _verdict(rp, ap):
    # PRODUCTION TEXT F10 (Q review, information only): with fewer than 3 nights the noise threshold is nan, rp is False and the
    # reason reads RMS_ABOVE although there was no threshold - production's byte-identical table text (and the 10-minute merge's
    # 'why'), so it changes only with production's (pinned by test_production_bugs_pinned)
    return ('KEEP' if (rp and ap) else 'OMIT',
            PASSES if (rp and ap) else (RMS_ABOVE if not rp else RATE_HIGH))


class SnapshotRecords:
    """The snapshot search's per-cell records of one fleet root: noise statistics per (LST, band, Stokes, tier) and the cells'
    wideband members (snapshot_deep_omission.night_stats / wb_members, cached per instance)."""

    def __init__(self, proto, reader):
        self.proto, self.reader = proto, reader
        self._ns, self._wm = {}, {}

    def cells(self, lst):
        """The LST's cell directories <proto>/<LST>_<night>, sorted (snapshot_deep_omission.night_stats' glob)."""
        if ('cells', lst) not in self._ns:
            self._ns[('cells', lst)] = sorted(glob.glob(os.path.join(self.proto, '%s_*' % lst)))
        return self._ns[('cells', lst)]

    def prefetch(self, lst, keys):
        """Read ahead (in parallel) the noise tables of ``keys`` [(band, S)] in both tiers over every cell of the LST - a
        missing file is remembered as missing; the decisions then read them in production's order (each one once)."""
        c = _c()
        self.reader.prefetch(os.path.join(d, c.SNAPSHOT_NOISE[T].format(band=b, stokes=S)) for d in self.cells(lst)
                             for b, S in sorted(set(keys)) for T in c.SNAPSHOT_TIERS)

    def night_stats(self, lst, band, S, T):
        """{night: (median admitted global rms Jy/beam, n admitted rows)} over every searched night of the LST with this table."""
        import numpy as np
        k = (lst, band, S, T)
        if k not in self._ns:
            out = {}
            name = _c().SNAPSHOT_NOISE[T].format(band=band, stokes=S)
            for d in self.cells(lst):
                night = os.path.basename(d)[len(lst) + 1:]
                text = self.reader.text(os.path.join(d, name), 'snapshot_noise', missing_ok=True)
                if text is None:                  # no table: the night is not one of the tier's nights for this band
                    continue
                v = []
                for r in csv.DictReader(io.StringIO(text, newline='')):
                    if str(r.get('admitted', '')) in ('1', 'True') and r.get('global_rms_jyb') not in ('', None):
                        try:
                            v.append(float(r['global_rms_jyb']))
                        except ValueError:
                            pass
                out[night] = (float(np.median(v)) if v else np.nan, len(v))
            self._ns[k] = out
        return self._ns[k]

    def members(self, lst, night, S):
        """{group: set of member subbands} of the cell's wideband frames (the search's wideband_members_<S>.csv)."""
        k = (lst, night, S)
        if k not in self._wm:
            p = os.path.join(self.proto, '%s_%s' % (lst, night), _c().SNAPSHOT_MEMBERS.format(stokes=S))
            out = collections.defaultdict(set)
            for r in self.reader.rows(p, 'snapshot_members'):
                out[r['group']].add(r['subband'])
            self._wm[k] = out
        return self._wm[k]


def decide_snapshot(lsts, served, rdir=None, proto=None, reader=None):
    """The 10-s and 1-minute decisions (snapshot_deep_omission.decide): rows in production's order and formats.
    served = {(lst, night, band, stokes, tier): n served detections}."""
    import numpy as np
    c = _c()
    reader = reader or Reader()
    rdir = rdir or c.REFUSALS
    recs = SnapshotRecords(proto or _cfg().paths.SNAPSHOT_FLEET_OUT, reader)
    G = groups()
    rows = []
    for lst in lsts:
        refs = load_refusals(rdir, lst, reader)
        recs.prefetch(lst, [(band, S) for band, _n, S in refs])
        lrows = []
        for (band, night, S), gates in sorted(refs.items()):
            base = _base(lst, band, night, S, gates)
            for T in c.SNAPSHOT_TIERS:
                st = recs.night_stats(lst, band, S, T)
                if night not in st or st[night][1] == 0:
                    lrows.append(dict(base, tier=T, n_admitted='', tier_median_rms='', tier_rms_threshold='', rms_pass='', rate='',
                                      rate_threshold='', rate_pass='', verdict='ABSENT', reason=snapshot_absent_reason(T)))
                    continue
                thr = mad_thr([m for m, _ in st.values()])
                rate = {nt: served.get((lst, nt, band, S, T), 0) / float(n) for nt, (m, n) in st.items() if n > 0}
                rthr = rate_thr(rate.values(), [n for m, n in st.values() if n > 0])
                med, n = st[night]
                rp = bool(np.isfinite(med) and np.isfinite(thr) and med <= thr)
                ap = bool(not np.isfinite(rthr) or rate[night] <= rthr)
                verdict, reason = _verdict(rp, ap)
                lrows.append(dict(base, tier=T, n_admitted=n, tier_median_rms='%.4g' % med, tier_rms_threshold='%.4g' % thr,
                                  rms_pass=int(rp), rate='%.4g' % rate[night], rate_threshold='%.4g' % rthr, rate_pass=int(ap),
                                  verdict=verdict, reason=reason))
        # wideband consequences (WIDEBAND_EXCEPT excepted): a subband night not KEPT in a tier (its verdict one of
        # WIDEBAND_MEMBER_VERDICTS - production's OMIT or ABSENT, pending Gregg) omits that night's band-group detections when
        # the subband is a member of the cell's wideband frames of that group (both tiers share the frames)
        # PRODUCTION BUG F1, pending the production fix (snapshot_deep_omission.py:106-119; Q review F1): the members are the
        # cell's UNION over every frame (wideband_members_<S>.csv by group, its ts ignored), so a group detection whose own frame
        # was stacked without the omitted subband is omitted too; kept for parity, pinned by test_production_bugs_pinned
        mv = member_verdicts()
        wb = collections.defaultdict(set)
        for r in lrows:
            if r['verdict'] in mv and r['band'] not in G and r['band'] not in c.WIDEBAND_EXCEPT:
                mem = recs.members(lst, r['night'], r['stokes'])
                for g, sbs in G.items():
                    if r['band'] in sbs and r['band'] in mem.get(g, ()):
                        wb[(g, r['night'], r['stokes'], r['tier'])].add(member_word(r['band'], r['verdict']))
        for (g, night, S, T), why in sorted(wb.items()):
            m = ', '.join(sorted(why))
            lrows.append(dict(lst=lst, band=g, night=night, stokes=S, gates=member_gates(m), deep_value='', deep_threshold='',
                              deep_attempt='', tier=T, n_admitted='', tier_median_rms='', tier_rms_threshold='', rms_pass='',
                              rate='', rate_threshold='', rate_pass='', verdict='OMIT', reason=snapshot_member_reason(m, g)))
        for r in lrows:
            r['deep_threshold_meaning'] = _meaning(r)
            r['rate_definition'] = SNAPSHOT_RATE_DEF
        rows += lrows
    return rows


class TenminRecords:
    """The 10-minute search's records (sublst_deep_omission.tier_windows / window_rms / served_counts / wideband_windows /
    wbmemb): the searched windows of a cell and Stokes, their central rms (a cache entry while its window's size and mtime
    match, else the page-context table under the same check, else recomputed in place), the published runs' served counts,
    the wideband windows' WBMEMB cards. Window-rms caches are per LST, window_rms_<LST>.csv (production's layout): the stage's
    own under ``cache_dir`` (written), production's under ``seed_dir`` (read only)."""

    def __init__(self, reader, cache_dir=None, seed_dir=None, frame_rms=None):
        cfg = _cfg()
        cs = cfg.catalog_sublst
        self.reader = reader
        self.root = cfg.paths.SUBLST_ROOT
        self.arm = cs.ARM
        self.tree = cs.TENMIN_TREE.format(root=self.root, tree=cs.ARM_TREES[self.arm])
        self.search = cs.SEARCH_DIR.format(root=self.root, arm=self.arm)
        self.frame_rms = frame_rms if frame_rms is not None else _c().TENMIN_FRAME_RMS
        self.cache_dir, self.seed_dir = cache_dir, seed_dir
        self.n = collections.Counter()           # window rms by source: seed, cache, page-context, recomputed, not-finite
        self._fr = None
        self._wbw, self._wbm, self._served = {}, {}, {}

    def load_cache(self, lst):
        """{window path: rms} of the LST's cache files whose window still has the size and mtime it was measured at
        (sublst_deep_omission.decide; the seed first, the stage's own cache over it)."""
        cache = {}
        for d, tag in ((self.seed_dir, 'seed'), (self.cache_dir, 'cache')):
            f = os.path.join(d, 'window_rms_%s.csv' % lst) if d else None
            if not f or not os.path.exists(f):
                continue
            for r in csv.DictReader(open(f, newline='')):
                try:
                    st = os.stat(r['path'])
                except OSError:
                    continue
                if r.get('size') == str(st.st_size) and r.get('mtime') == str(int(st.st_mtime)):
                    cache[r['path']] = (float(r['rms']), tag)
        return {k: v for k, (v, _t) in cache.items()}, collections.Counter(t for _v, t in cache.values())

    def write_cache(self, lst, cache):
        """The LST's cache file (every window measured or read this run; sublst_deep_omission.decide), atomically."""
        if not self.cache_dir:
            return
        os.makedirs(self.cache_dir, exist_ok=True)
        f = os.path.join(self.cache_dir, 'window_rms_%s.csv' % lst)
        with open(f + '.tmp', 'w', newline='') as o:
            w = csv.writer(o)
            w.writerow(['path', 'size', 'mtime', 'rms'])
            for k, v in cache.items():
                try:
                    st = os.stat(k)
                    w.writerow([k, st.st_size, int(st.st_mtime), v])
                except OSError:
                    pass
        os.replace(f + '.tmp', f)

    def run_dir(self, lst):
        return os.path.join(_cfg().paths.TRANSIENTS_ROOT, _c().TENMIN_RUN.format(hh=lst[:-1], arm=self.arm))

    def windows(self, lst, band):
        """{S: [(key, night, sub_path)]} the search SEARCHED (v_deep_images.csv rows), night and residual from frames.csv
        (night = frames.csv 'date', never the window key's date); None without frames.csv."""
        f = os.path.join(self.tree, '%s_%s' % (lst, band), 'frames.csv')
        if not os.path.exists(f):
            return None
        meta = {}
        for r in self.reader.rows(f, 'tenmin_frames', missing_ok=False):
            meta[(r['key'], r['stokes'])] = (r['date'], r['sub_path'])
        out = {}
        for S in ('I', 'V'):
            vi = os.path.join(self.search, '%s_%s_%s' % (lst, band, S), 'transients', 'Run_SLST%s%s' % (lst[:-1], self.arm),
                              'v_deep_images.csv')
            if not os.path.exists(vi):
                continue
            keys = [r['date'] for r in self.reader.rows(vi, 'tenmin_searched', missing_ok=False)]
            out[S] = [(k, meta[(k, S)][0], meta[(k, S)][1]) for k in keys if (k, S) in meta]
        return out

    def rms(self, path, cache):
        """The searched window's central rms (sublst_deep_omission.window_rms): the cache, else the page-context table while
        size and mtime match, else recomputed with page_context_tenmin.central_rms_file (exopipe_ops, in place); nan when
        unreadable (counted)."""
        import numpy as np
        if path in cache:
            self.n['cache'] += 1
            return cache[path]
        if self._fr is None:
            self._fr = {}
            if self.frame_rms and os.path.exists(self.frame_rms):
                for r in self.reader.rows(self.frame_rms, 'tenmin_frame_rms', missing_ok=False):
                    self._fr[r['path']] = (r['size'], r['mtime'], r['rms'])
        v = np.nan
        try:
            st = os.stat(path)
            rec = self._fr.get(path)
            if rec and rec[0] == str(st.st_size) and rec[1] == str(int(st.st_mtime)):
                v = float(rec[2])
                self.n['page-context'] += 1
            else:
                from exopipe.phase3.core import env
                ops = env.legacy_root('exopipe_ops')
                if ops not in sys.path:
                    sys.path.insert(0, ops)
                from page_context_tenmin import central_rms_file
                v = central_rms_file(path)
                self.n['recomputed'] += 1
        except Exception:
            v = np.nan
        if not np.isfinite(v):
            self.n['not-finite'] += 1
        cache[path] = v
        return v

    def served(self, lst):
        """({(band, window key, S): n served detections} of the published LST run's master_catalog.csv, run exists)."""
        if lst not in self._served:
            f = os.path.join(self.run_dir(lst), 'master_catalog.csv')
            n = collections.defaultdict(int)
            ok = os.path.exists(f)
            if ok:
                for r in self.reader.rows(f, 'tenmin_served', missing_ok=False):
                    n[(r['band'], r['date'], r['stokes'])] += 1
            self._served[lst] = (n, ok)
        return self._served[lst]

    def wideband_windows(self, lst, g):
        """{(night, S): [(key, sub_path)]} of the tier's wideband windows of (LST, group)."""
        if (lst, g) not in self._wbw:
            out = collections.defaultdict(list)
            f = os.path.join(self.tree, '%s_%s' % (lst, g), 'frames.csv')
            if os.path.exists(f):
                for r in self.reader.rows(f, 'tenmin_frames', missing_ok=False):
                    out[(r['date'], r['stokes'])].append((r['key'], r['sub_path']))
            self._wbw[(lst, g)] = out
        return self._wbw[(lst, g)]

    def wbmemb(self, path):
        """The member subbands of a wideband window from its own WBMEMB card ('' -> every subband of every group)."""
        if path not in self._wbm:
            from astropy.io import fits
            try:
                v = str(fits.getheader(path).get('WBMEMB', ''))
            except Exception:
                v = ''
            self._wbm[path] = set(v.split(',')) if v else set(sum(groups().values(), []))
        return self._wbm[path]


def decide_tenmin(lst, rdir=None, recs=None, reader=None):
    """The 10-minute decisions of one LST (sublst_deep_omission.decide): rows in production's order and formats."""
    import numpy as np
    c = _c()
    reader = reader or (recs.reader if recs else Reader())
    recs = recs or TenminRecords(reader)
    G = groups()
    refs = load_refusals(rdir or c.REFUSALS, lst, reader)
    if not refs:
        return []
    served, have_run = recs.served(lst)
    cache, loaded = recs.load_cache(lst)
    recs.n.update({'loaded-' + k: v for k, v in loaded.items()})
    rows, tw = [], {}
    for (band, night, S), gates in sorted(refs.items()):
        if band not in tw:
            tw[band] = recs.windows(lst, band)
        W = tw[band]
        base = _base(lst, band, night, S, gates)
        if W is None or S not in W:
            rows.append(dict(base, n_windows='', tier_median_rms='', tier_rms_threshold='', rms_pass='', rate='', rate_threshold='',
                             rate_pass='', verdict='ABSENT', reason=TENMIN_NO_WINDOWS))
            continue
        per_night = collections.defaultdict(list)
        for key, nt, path in W[S]:
            per_night[nt].append((key, path))
        med = {nt: np.nanmedian([recs.rms(p, cache) for _, p in v]) for nt, v in per_night.items()}
        rate = {nt: sum(served.get((band, k, S), 0) for k, _ in v) / len(v) for nt, v in per_night.items()}
        thr = mad_thr(list(med.values()))
        rthr = rate_thr(rate.values(), [len(v) for v in per_night.values()])
        if night not in per_night:
            rows.append(dict(base, n_windows=0, tier_median_rms='', tier_rms_threshold='%.4g' % thr, rms_pass='', rate='',
                             rate_threshold='%.4g' % rthr, rate_pass='', verdict='ABSENT', reason=TENMIN_NIGHT_ABSENT))
            continue
        rp = bool(np.isfinite(med[night]) and np.isfinite(thr) and med[night] <= thr)
        ap = bool(not have_run or (np.isfinite(rthr) and rate[night] <= rthr) or not np.isfinite(rthr))
        verdict, reason = _verdict(rp, ap)
        rows.append(dict(base, n_windows=len(per_night[night]), tier_median_rms='%.4g' % med[night],
                         tier_rms_threshold='%.4g' % thr, rms_pass=int(rp), rate='%.4g' % rate[night],
                         rate_threshold='%.4g' % rthr, rate_pass=int(ap), verdict=verdict, reason=reason))
    recs.write_cache(lst, cache)
    # wideband consequences (WIDEBAND_EXCEPT excepted): a deep-refused subband night the tier does not KEEP - its verdict one of
    # WIDEBAND_MEMBER_VERDICTS (production's OMIT or ABSENT, pending Gregg) - omits that night's wideband detections of a group
    # WHEN the subband is a member of that night's wideband windows by their own WBMEMB card; a night with no wideband window has
    # nothing to drop. One row per (group, night, Stokes).
    # PRODUCTION BUG F1, pending the production fix (sublst_deep_omission.py:192-211; Q review F1): ONE window of the night whose
    # WBMEMB holds the subband omits the group's whole night - every window's detections, also those of a window stacked without
    # it (each window's own WBMEMB is read, but only to decide the night); kept for parity, pinned by test_production_bugs_pinned
    mv = member_verdicts()
    wb = collections.defaultdict(set)
    for r in [x for x in rows if x['verdict'] in mv and x['band'] not in G and x['band'] not in c.WIDEBAND_EXCEPT]:
        for g, mem in G.items():
            if r['band'] not in mem:
                continue
            for key, path in recs.wideband_windows(lst, g).get((r['night'], r['stokes']), []):
                if r['band'] in recs.wbmemb(path):
                    wb[(g, r['night'], r['stokes'])].add(member_word(r['band'], r['verdict']))
                    break
    for (g, night, S), why in sorted(wb.items()):
        m = ', '.join(sorted(why))
        nwin = len(recs.wideband_windows(lst, g).get((night, S), []))
        rows.append(dict(lst=lst, band=g, night=night, stokes=S, gates=member_gates(m), deep_value='', deep_threshold='',
                         deep_attempt='', n_windows=nwin, tier_median_rms='', tier_rms_threshold='', rms_pass='', rate='',
                         rate_threshold='', rate_pass='', verdict='OMIT', reason=tenmin_member_reason(m)))
    for r in rows:
        r['deep_threshold_meaning'] = _meaning(r)
    return rows


def omitted_keys(rows):
    """{(lst, band, night, stokes, tier)} of the OMIT rows (tier '10min' for a 10-minute table)."""
    return {(r['lst'], r['band'], r['night'], r['stokes'], r.get('tier') or '10min') for r in rows if r['verdict'] == 'OMIT'}


def why_of(rows):
    """{(lst, band, night, stokes, tier): the 'why' text} of the OMIT rows (WHY_SEP between two decisions of one key)."""
    out = collections.defaultdict(list)
    for r in rows:
        if r['verdict'] == 'OMIT':
            out[(r['lst'], r['band'], r['night'], r['stokes'], r.get('tier') or '10min')].append(why_part(r['gates'], r['reason']))
    return {k: WHY_SEP.join(v) for k, v in out.items()}


def write_decisions(path, rows, tier):
    """The decision table in production's layout: snapshot - the union of the rows' keys (snapshot_deep_omission.write);
    10-minute - the first row's keys with rate_definition (sublst_deep_omission.py main). Atomic."""
    with open(path + '.tmp', 'w', newline='') as fh:
        if not rows:
            fh.write(EMPTY_SNAPSHOT_HEADER if tier == TIER_SNAPSHOT else EMPTY_TENMIN_HEADER)
        elif tier == TIER_SNAPSHOT:
            hdr = []
            for r in rows:
                hdr += [k for k in r if k not in hdr]
            w = csv.DictWriter(fh, fieldnames=hdr)
            w.writeheader()
            w.writerows(rows)
        else:
            rows = [dict(r, rate_definition=TENMIN_RATE_DEF) for r in rows]
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    os.replace(path + '.tmp', path)


# =====================================================================================================================
# DROP - 10-s and 1-minute (the combined view's step 0 and the per-tier consequences)
# =====================================================================================================================

def batch_tag(run):
    """Run_SNAPB4P_20261018 -> B4P (snapshot_combined_view.py TAG)."""
    return re.sub(r'^Run_SNAP|_\d{8}$', '', os.path.basename(run.rstrip('/')))


def dkey(d):
    """A detection exactly: the frame + the peak position (snapshot_combined_view.dkey)."""
    return (d['cand_id'], d.get('ra_peak') or d['ra'], d.get('dec_peak') or d['dec'])


def okey(d):
    """A snapshot detection's omission key (lst, subband or band, night, stokes, tier)."""
    return (d['lst'], d.get('subband') or d['band'], d['date'], d['stokes'], d['tier'])


def served_identities(runs, reader):
    """[(run, identity, [(file, row)], [detections])] of every served batch identity (snapshot_combined_view.py step 1:
    the rows of source_catalogs/*_sources_survivors.csv by source_id; detections = every row of each row's review_dir
    detections.csv, one per dkey). G2 a run without source_catalogs; G3 an identity without detections."""
    out = []
    for r in runs:
        if not os.path.isdir(os.path.join(r, 'source_catalogs')):
            refuse(f'G2: {r} has no source_catalogs (not a served batch run)')
        ids = collections.OrderedDict()
        for p in sorted(glob.glob(os.path.join(r, 'source_catalogs', '*_sources_survivors.csv'))):
            for row in reader.rows(p, 'served_catalogs', missing_ok=False):
                ids.setdefault(row['source_id'], []).append((os.path.basename(p), row))
        reader.prefetch(os.path.join(r, row['review_dir'], 'detections.csv') for grs in ids.values() for _, row in grs)
        for J, grs in ids.items():
            dets = collections.OrderedDict()
            for _, row in grs:
                for d in reader.rows(os.path.join(r, row['review_dir'], 'detections.csv'), 'served_detections'):
                    dets.setdefault(dkey(d), d)
            if not dets:
                refuse(f'G3: {batch_tag(r)} identity {J} has no detections')
            out.append((r, J, grs, list(dets.values())))
    return out


def served_counts(pending):
    """{(lst, night, band, stokes, tier): n} - decide_snapshot's served argument (the combined view's re-keyed Counter)."""
    c = collections.Counter(okey(d) for p in pending for d in p[3])
    return {(k[0], k[2], k[1], k[3], k[4]): n for k, n in c.items()}, sum(c.values())


def apply_snapshot(pending, OM, why):
    """The identity-level drop (the combined view's step 0): (kept [(run, identity, rows, kept detections)], dropped
    {(run, identity)}, record rows). G3 a kept detection in two identities; G5 conservation."""
    kept, dropped, rec = [], set(), []
    n_all = sum(len(p[3]) for p in pending)
    for r, J, grs, dets in pending:
        keep = []
        for d in dets:
            k = okey(d)
            if k in OM:
                rec.append(dict(kind='detection', batch=batch_tag(r), identity=J, cand_id=d['cand_id'], lst=d['lst'], band=k[1],
                                night=d['date'], stokes=d['stokes'], tier=d['tier'], snr=d['snr'], why=why[k]))
            else:
                keep.append(d)
        if not keep:
            dropped.add((r, J))
            rec.append(dict(kind='identity dropped', batch=batch_tag(r), identity=J))
            continue
        kept.append((r, J, grs, keep))
    n_om = sum(1 for x in rec if x['kind'] == 'detection')
    n_in = sum(len(p[3]) for p in kept)
    if n_in + n_om != n_all:
        refuse(f'G5: omission record {n_om} + kept {n_in} != served {n_all}')
    keys = collections.Counter(dkey(d) for p in kept for d in p[3])
    dup = [k for k, n in keys.items() if n > 1]
    if dup:
        refuse(f'G3: {len(dup)} detections in two batch identities (e.g. {dup[:2]}) - the batches are not cell-disjoint')
    return kept, dropped, rec


def per_tier_snapshot(runs, pending, dropped, OM, why, reader):
    """The per-tier consequences (the combined view's step 4b, its omission parts): per Stokes, tier and batch, a source of a
    dropped identity is dropped; a source left with no detection is dropped; a partial one whose best detection went is
    re-pointed (its survivor gates not re-run); a member row of a dropped identity or an omitted key is dropped; a panel of an
    omitted key is not linked (every kept source's directory is listed, as the view does). Returns (record rows,
    {(run, cdir, source_id): re-pointed fields}, the kept product's per-tier rows - KEPT_COLS, each naming the served run and the
    catalogue file or page directory it acts on). G4 a source or member row of no served identity."""
    ids = collections.defaultdict(set)
    for r, J, _g, _d in pending:
        ids[r].add(J)
    cats = []                          # (S, T, cdir, run, kind, band, rows) in the order the record is written
    for S in ('V', 'I'):
        for T in _c().SNAPSHOT_TIERS:
            cdir = f'source_catalogs_{S}_{T}'
            for r in runs:
                for kind, suf in (('sources', '_sources_survivors.csv'), ('members', '_source_members.csv')):
                    for p in sorted(glob.glob(os.path.join(r, cdir, '*' + suf))):
                        cats.append((S, T, cdir, r, kind, os.path.basename(p)[:-len(suf)],
                                     reader.rows(p, 'served_catalogs', missing_ok=False)))
    srcs = [os.path.join(r, row['review_dir']) for S, T, cdir, r, kind, band, rows in cats if kind == 'sources'
            for row in rows if (r, row['identity_id']) not in dropped]
    reader.prefetch(os.path.join(d, 'detections.csv') for d in srcs)
    rec, repoint, drops, listing = [], {}, [], None
    for S, T, cdir, r, kind, band, rows in cats:
        tag = batch_tag(r)
        run = os.path.abspath(r)
        if kind == 'sources':
            catalog = f'{cdir}/{band}_sources_survivors.csv'
            for row in rows:
                if (r, row['identity_id']) in dropped:
                    rec.append(dict(kind='source dropped', batch=tag, identity=row['identity_id'], source=row['source_id'],
                                    stokes=S, tier=T, band=band, note='identity left with no detection'))
                    drops.append(dict(kind='source dropped', run=run, batch=tag, identity=row['identity_id'], catalog=catalog,
                                      source_id=row['source_id'], stokes=S, tier=T, note='identity left with no detection'))
                    continue
                if row['identity_id'] not in ids[r]:
                    refuse(f'G4: {tag} {cdir} source {row["source_id"]} of identity {row["identity_id"]} is in no served identity')
                src = os.path.join(r, row['review_dir'])
                ad = reader.rows(os.path.join(src, 'detections.csv'), 'served_detections')
                kd = [d for d in ad if okey(d) not in OM]
                if not kd:
                    rec.append(dict(kind='source dropped', batch=tag, identity=row['identity_id'], source=row['source_id'],
                                    stokes=S, tier=T, band=band, note='every detection omitted'))
                    drops.append(dict(kind='source dropped', run=run, batch=tag, identity=row['identity_id'], catalog=catalog,
                                      source_id=row['source_id'], stokes=S, tier=T, note='every detection omitted'))
                    continue
                if len(kd) < len(ad) and row.get('best_cand_id') in {d['cand_id'] for d in ad} - {d['cand_id'] for d in kd}:
                    top = max(kd, key=lambda d: abs(float(d['snr'])))
                    note = 'best %s (%s) omitted; re-pointed to %s (%s)' % (row.get('best_cand_id'), row.get('best_snr'),
                                                                          top['cand_id'], top['snr'])
                    rec.append(dict(kind='source partial', batch=tag, identity=row['identity_id'], source=row['source_id'],
                                    stokes=S, tier=T, band=band, note=note))
                    fields = dict(best_cand_id=top['cand_id'], best_snr=top['snr'], ts=top['ts'], subband=top['subband'],
                                  peak_jy_best='', deep_omission=SNAPSHOT_PARTIAL_NOTE)
                    repoint[(r, cdir, row['source_id'])] = fields
                    drops.append(dict(kind='source re-pointed', run=run, batch=tag, identity=row['identity_id'], catalog=catalog,
                                      source_id=row['source_id'], cand_id=top['cand_id'], snr=top['snr'], stokes=S, tier=T,
                                      note=note, fields=json.dumps(fields, sort_keys=True)))
                if listing is None:                   # every kept source's directory, listed ahead (once)
                    listing = reader.listdirs(srcs)
                names = listing.get(src)
                for fn in names if names is not None else sorted(os.listdir(src)):
                    if fn in ('detections.csv', 'astrometry.png'):
                        continue
                    pm = SNAP_PANEL.match(fn)
                    if pm:
                        k = (pm.group(1), pm.group(3), pm.group(2), S, T)
                        if k in OM:
                            rec.append(dict(kind='panel', batch=tag, source=row['source_id'], lst=k[0], band=k[1], night=k[2],
                                            stokes=S, tier=T, note=fn, why=why[k]))
                            drops.append(dict(kind='panel dropped', run=run, batch=tag, identity=row['identity_id'],
                                              catalog=row['review_dir'], source_id=row['source_id'], lst=k[0], band=k[1],
                                              night=k[2], stokes=S, tier=T, note=fn))
            continue
        catalog = f'{cdir}/{band}_source_members.csv'
        for row in rows:
            mm = MEMBER_SOURCE.match(row['source_id'])
            ck = SNAP_CID.match(row.get('cand_id', ''))
            k = (ck.group(1), ck.group(3), ck.group(2), ck.group(4), ck.group(5)) if ck else None
            if k in OM:                       # its own key omitted
                rec.append(dict(kind='member row', batch=tag, source=row['source_id'], cand_id=row.get('cand_id', ''),
                                lst=k[0], band=k[1], night=k[2], stokes=S, tier=T, why=why[k]))
                drops.append(dict(kind='member row dropped', run=run, batch=tag, identity=mm.group(1) if mm else '',
                                  catalog=catalog, source_id=row['source_id'], cand_id=row.get('cand_id', ''), lst=k[0],
                                  band=k[1], night=k[2], stokes=S, tier=T))
                continue
            if mm and (r, mm.group(1)) in dropped:     # a consequence: its identity has no detection left
                rec.append(dict(kind='member row', batch=tag, source=row['source_id'], cand_id=row.get('cand_id', ''),
                                band=band, stokes=S, tier=T, note='identity left with no detection'))
                drops.append(dict(kind='member row dropped', run=run, batch=tag, identity=mm.group(1), catalog=catalog,
                                  source_id=row['source_id'], cand_id=row.get('cand_id', ''), stokes=S, tier=T,
                                  note='identity left with no detection'))
                continue
            if mm is None or mm.group(1) not in ids[r]:
                refuse(f'G4: {tag} {cdir} member source {row["source_id"]} is in no served identity')
    return rec, repoint, drops


def kept_rows(kept):
    """The kept product's 'detection kept' rows (KEPT_COLS) of apply_snapshot's kept identities, generated (a full view keeps
    millions): every kept detection of every kept identity - the identities and detections the cumulative step merges (its key
    dkey: cand_id + ra_peak / dec_peak)."""
    for r, J, _grs, dets in kept:
        run, tag = os.path.abspath(r), batch_tag(r)
        for d in dets:
            c, ra, dec = dkey(d)
            yield dict(kind='detection kept', run=run, batch=tag, identity=J, cand_id=c, ra_peak=ra, dec_peak=dec,
                       lst=d['lst'], band=okey(d)[1], night=d['date'], stokes=d['stokes'], tier=d['tier'], snr=d['snr'])


# ---- the served set's fingerprint (Q fix verification N2) -----------------------------------------------------------------
# The kept product's content test could not see a kept detection listed twice, one the served product never had, or one moved
# to another identity: it checked each kept row against the decisions and the record, never against the served set. The writer
# now puts ORDER-FREE MULTISET DIGESTS into deep_omission_summary.json (key 'digests'): of the served detections, the kept ones,
# the omitted ones - each item (batch, identity, cand_id, ra_peak, dec_peak) - and of the omitted ones' (batch, identity,
# cand_id), the record's detection rows (they carry no peak). The digest is additive (the count and the sum of the items'
# sha256 modulo 2**256), so the content test streams the kept rows once and checks kept + omitted = served, and the cumulative
# step re-reads the served runs and checks served = the summary's (the served product has not changed since the drop).

class MultisetDigest:
    """An order-free, additive digest of a multiset of text items: '<count>:<sum of their sha256 modulo 2**256, 64 hex>'."""
    MOD = 1 << 256

    def __init__(self, text=None):
        self.n, self.s = 0, 0
        if text:
            n, _, s = str(text).partition(':')
            self.n, self.s = int(n), int(s, 16)

    def add(self, item):
        self.n += 1
        self.s = (self.s + int.from_bytes(hashlib.sha256(item.encode('utf-8')).digest(), 'big')) % self.MOD
        return self

    def __add__(self, other):
        out = MultisetDigest()
        out.n, out.s = self.n + other.n, (self.s + other.s) % self.MOD
        return out

    def __eq__(self, other):
        return isinstance(other, MultisetDigest) and (self.n, self.s) == (other.n, other.s)

    def __ne__(self, other):          # Python 3.6 children import this module too
        return not self == other

    def text(self):
        return '%d:%064x' % (self.n, self.s)


def detection_item(batch, identity, k):
    """A served detection as a digest item: batch tag, identity, cand_id, ra_peak, dec_peak (k = dkey), unit-separated."""
    return '\x1f'.join((batch, identity) + tuple(k))


def served_digests(pending, OM):
    """{'served', 'kept', 'omitted', 'omitted_cid'} digests (texts) of the served identities ``pending`` under the OMIT keys OM:
    every detection, the kept ones, the omitted ones (detection_item) and the omitted ones' (batch, identity, cand_id)."""
    D = {k: MultisetDigest() for k in ('served', 'kept', 'omitted', 'omitted_cid')}
    for r, J, _grs, dets in pending:
        tag = batch_tag(r)
        for d in dets:
            item = detection_item(tag, J, dkey(d))
            D['served'].add(item)
            if okey(d) in OM:
                D['omitted'].add(item)
                D['omitted_cid'].add('\x1f'.join((tag, J, d['cand_id'])))
            else:
                D['kept'].add(item)
    if D['kept'] + D['omitted'] != D['served']:            # additive by construction; a broken digest refuses here
        refuse('G5: the kept and omitted digests do not add up to the served digest')
    return {k: v.text() for k, v in D.items()}


# =====================================================================================================================
# DROP - 10-minute: the detections the in-place merge drop removes (sublst_deep_omission_drop.filter_rows, member rows)
# =====================================================================================================================

def tenmin_detections(lsts, OM, why, recs, reader):
    """Record rows of every member row of the LST runs whose (lst, band, night, stokes) is OMIT - the 'detection' rows
    sublst_deep_omission_drop.filter_rows writes (source_catalogs_<S>_<sb|wb>/*_source_members.csv; Stokes = the table's)."""
    rec = []
    keys4 = {k[:4]: k for k in OM}
    for L in lsts:
        R = recs.run_dir(L)
        for S in ('I', 'V'):
            for T in ('sb', 'wb'):
                for p in sorted(glob.glob(os.path.join(R, f'source_catalogs_{S}_{T}', '*_source_members.csv'))):
                    for r in reader.rows(p, 'tenmin_members', missing_ok=False):
                        m = TENMIN_CID.match(r['cand_id'] or '')
                        if not m:
                            continue
                        k = (m.group(2), m.group(4), m.group(3), S)
                        if k in keys4:
                            rec.append(dict(kind='detection', lst=L, stokes=S, view=T, source_id=r['source_id'], band=r['band'],
                                            cand_id=r['cand_id'], night=k[2], why=why[keys4[k]], detail=''))
    return rec


# =====================================================================================================================
# the stage
# =====================================================================================================================

def write_record(path, rows, cols):
    with open(path + '.tmp', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='raise')
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in cols})
    os.replace(path + '.tmp', path)


def page_line(n):
    """Line 1 of the summary: the page line (production's words)."""
    return '%d detections omitted by the deep-omission rule (Gregg 2026-10-24)' % n


def absent_gate(rows, served_of):
    """G6: the ABSENT keys holding served detections, [(key, n)]. catalog_omission.ABSENT_WITH_SERVED 'refuse' refuses on any
    (the exception test cannot be evaluated; the rule is never applied as 'nothing to drop'); 'production' keeps them, as
    production does, and the summary counts them."""
    bad = []
    for r in rows:
        if r['verdict'] == 'ABSENT':
            k = (r['lst'], r['band'], r['night'], r['stokes'], r.get('tier') or '10min')
            n = served_of(k)
            if n:
                bad.append((k, n))
    mode = _c().ABSENT_WITH_SERVED
    if _absent_mode_problem(mode):
        raise ValueError(_absent_mode_problem(mode))
    if bad and mode == 'refuse':    # WAVE2 G6: the refactor's PROPOSED gate, pending Gregg (production keeps; parity file 'production')
        refuse(f'G6: {len(bad)} deep-refused keys the tier has no record of hold served detections (e.g. {bad[:3]}) - the '
               f'exception test cannot be evaluated')
    return bad


def run_snapshot(runs, out, rdir=None, proto=None, u=None, label='omit'):
    """The 10-s / 1-minute stage on the served batch runs ``runs``: decisions, the drop, its record and the kept product (what
    the snapshot tier's cumulative step applies) into ``out``. Returns the summary dict. Gates refuse (core.record.Refusal)."""
    c = _c()
    rdir = rdir or c.REFUSALS
    proto = proto or _cfg().paths.SNAPSHOT_FLEET_OUT
    _record_gate(rdir)
    reader = Reader()
    t0 = time.time()
    pending = served_identities(runs, reader)
    served, n_all = served_counts(pending)
    lsts = sorted({k[0] for k in served})
    dec = decide_snapshot(lsts, served, rdir=rdir, proto=proto, reader=reader)
    absent = absent_gate(dec, lambda k: served.get((k[0], k[2], k[1], k[3], k[4]), 0))
    OM, why = omitted_keys(dec), why_of(dec)
    kept, dropped, rec = apply_snapshot(pending, OM, why)
    digests = served_digests(pending, OM)          # N2: the served set's fingerprint, checked by the kept product's test
    prec, _repoint, drops = per_tier_snapshot(runs, pending, dropped, OM, why, reader)
    rec += prec
    os.makedirs(out, exist_ok=True)
    write_decisions(os.path.join(out, 'deep_omission.csv'), dec, TIER_SNAPSHOT)
    write_record(os.path.join(out, 'deep_omission_dropped.csv'), rec, RECORD_COLS)
    kept_kinds = collections.Counter()              # Q review F2: the drop, written for the cumulative step that applies it

    def counted(rows):
        for x in rows:
            kept_kinds[x['kind']] += 1
            yield x
    write_record(os.path.join(out, 'deep_omission_kept.csv'), counted(itertools.chain(kept_rows(kept), drops)), KEPT_COLS)
    kinds = collections.Counter(x['kind'] for x in rec)
    n_om = kinds.get('detection', 0)
    summ = dict(tier=TIER_SNAPSHOT, label=label, runs=[os.path.abspath(r) for r in runs], batches=[batch_tag(r) for r in runs],
                refusals=os.path.abspath(rdir), proto=os.path.abspath(proto), lsts=lsts, served=n_all, omitted=n_om,
                kept=n_all - n_om, decisions=len(dec), omit_keys=len(OM),
                omit_keys_by_tier={t: sum(1 for k in OM if k[4] == t) for t in c.SNAPSHOT_TIERS},
                verdicts=dict(collections.Counter(r['verdict'] for r in dec)), record=dict(sorted(kinds.items())),
                kept_product=dict(sorted(kept_kinds.items())),
                identities=len(pending), identities_kept=len(kept), absent_with_served=len(absent), digests=digests,
                seconds=round(time.time() - t0, 1))
    _write_summary(out, summ, n_om, 'served detections %d, kept %d; decisions %d (%d OMIT keys); %s' % (
        n_all, n_all - n_om, len(dec), len(OM), ', '.join('%s %d' % kv for kv in sorted(kinds.items()))))
    if u is not None:
        reader.record(u, {'refusals': rdir, 'snapshot_noise': proto, 'snapshot_members': proto,
                          'served_catalogs': os.path.commonpath([os.path.abspath(r) for r in runs]),
                          'served_detections': os.path.commonpath([os.path.abspath(r) for r in runs])})
    return summ


def tenmin_table(lsts, out_csv, rdir=None, cache_dir=None, seed_dir=None, reader=None, frame_rms=None):
    """The 10-minute decision table of ``lsts`` into ``out_csv`` - what the cumulative merge reads (SUBLST_DEEP_OMISSION).
    Window-rms caches: the stage's own under ``cache_dir`` (window_rms_<LST>.csv, written), production's under ``seed_dir``
    (read only). Returns (rows, TenminRecords). G1 the refusal record; G2 an LST without its published run."""
    c = _c()
    rdir = rdir or c.REFUSALS
    _record_gate(rdir)
    reader = reader or Reader()
    recs = TenminRecords(reader, cache_dir=cache_dir, seed_dir=seed_dir, frame_rms=frame_rms)
    rows = []
    for L in lsts:
        if not os.path.exists(os.path.join(recs.run_dir(L), 'master_catalog.csv')):
            refuse(f'G2: no published 10-minute run {recs.run_dir(L)}/master_catalog.csv for {L}')
        rows += decide_tenmin(L, rdir=rdir, recs=recs, reader=reader)
    write_decisions(out_csv, rows, TIER_TENMIN)
    return rows, recs


def tenmin_served_of(recs):
    """served(key) of a 10-minute decision key (lst, band, night, stokes, '10min'): the LST run's master_catalog rows of that
    band and Stokes whose window key's date part is the night (key = night + 'T' + bin; sublst_deep_omission_drop.py)."""
    def f(k):
        return sum(n for (b, wk, s), n in recs.served(k[0])[0].items() if b == k[1] and s == k[3] and wk[:10] == k[2])
    return f


def run_tenmin(lsts, out, rdir=None, seed_dir=None, u=None, label='omit', frame_rms=None):
    """The 10-minute stage on the published LST runs of ``lsts``: the decision table, the detections the cumulative merge's
    drop removes (their record) into ``out``. Window rms: production's cache catalog_omission.TENMIN_RMS_CACHE_SEED is read as
    a seed (``seed_dir`` '' = none); the stage's own cache under <out>/cache. Returns the summary dict."""
    c = _c()
    rdir = rdir or c.REFUSALS
    reader = Reader()
    t0 = time.time()
    seed_dir = c.TENMIN_RMS_CACHE_SEED if seed_dir is None else (seed_dir or None)
    os.makedirs(out, exist_ok=True)
    rows, recs = tenmin_table(lsts, os.path.join(out, 'deep_omission.csv'), rdir=rdir, cache_dir=os.path.join(out, 'cache'),
                              seed_dir=seed_dir, reader=reader, frame_rms=frame_rms)
    OM, why = omitted_keys(rows), why_of(rows)
    absent = absent_gate(rows, tenmin_served_of(recs))
    rec = tenmin_detections(lsts, OM, why, recs, reader)
    write_record(os.path.join(out, 'deep_omission_dropped.csv'), rec, TENMIN_RECORD_COLS)
    summ = dict(tier=TIER_TENMIN, label=label, lsts=list(lsts), refusals=os.path.abspath(rdir), omitted=len(rec),
                decisions=len(rows), omit_keys=len(OM), verdicts=dict(collections.Counter(r['verdict'] for r in rows)),
                absent_with_served=len(absent), window_rms=dict(sorted(recs.n.items())), rms_cache_seed=seed_dir,
                seconds=round(time.time() - t0, 1))
    _write_summary(out, summ, len(rec), 'decision table %s; %d (lst, band, night, Stokes) omitted'
                   % (os.path.join(out, 'deep_omission.csv'), len(OM)))
    if u is not None:
        reader.record(u, {'refusals': rdir, 'tenmin_frames': recs.tree, 'tenmin_searched': recs.search,
                          'tenmin_served': _cfg().paths.TRANSIENTS_ROOT, 'tenmin_members': _cfg().paths.TRANSIENTS_ROOT})
    return summ


def injected_record_gate(rdir):
    """G1, the injected-run guard on the refusal record (Q fix verification N1): an ORDINARY run never takes its decisions from
    an injected run's refusal record (core/record.injected_runs: an injected run's start record in the directory or an
    ancestor), however the record was named - --omit-refusals, --set / --config catalog_omission.REFUSALS, or a caller's rdir
    (the 10-minute cumulative build's tenmin_table). The flag route is also checked by core/injected.check_flags before the
    step runs; this check covers the config route and every caller. An injected run may read it."""
    from exopipe.phase3.core import injected, record
    if injected.active():
        return
    hit = record.injected_runs(rdir)
    if hit:
        root, f, why = hit[0]
        refuse(f'G1 {injected.LABEL}: the deep refusal record {rdir}: {root} holds an injected run\'s products ({f}: {why}) - '
               f'an ordinary run never takes its omission decisions from an injected run (catalog_omission.REFUSALS, however '
               f'it was set)')


def _record_gate(rdir):
    """G1: the refusal record exists and holds files (a rule that cannot be checked is never applied as 'nothing to drop'),
    and an ordinary run's record is no injected run's (injected_record_gate)."""
    injected_record_gate(rdir)
    if not os.path.isdir(rdir):
        refuse(f'G1: no deep refusal record {rdir}')
    if not glob.glob(os.path.join(rdir, '*_excluded.csv')):
        refuse(f'G1: the deep refusal record {rdir} holds no *_excluded.csv file')


def _write_summary(out, summ, n_om, line2):
    with open(os.path.join(out, 'deep_omission_summary.txt.tmp'), 'w') as fh:
        fh.write(page_line(n_om) + '\n' + line2 + '\n')
    os.replace(os.path.join(out, 'deep_omission_summary.txt.tmp'), os.path.join(out, 'deep_omission_summary.txt'))
    with open(os.path.join(out, 'deep_omission_summary.json.tmp'), 'w') as fh:
        json.dump(summ, fh, indent=1, default=str)
    os.replace(os.path.join(out, 'deep_omission_summary.json.tmp'), os.path.join(out, 'deep_omission_summary.json'))


# =====================================================================================================================
# the stage inside `exopipe3 catalog` (--tier snapshot|sublst --stages omit)
# =====================================================================================================================

STAGE = 'omit'


def add_arguments(p):
    p.add_argument('--omit-runs', metavar='RUN,RUN|@FILE',
                   help='--stages omit, tier snapshot: the served batch runs of the view (directories, comma list or @file); '
                        'tier sublst takes --lsts (the published Run_SLST<hh>B runs)')
    p.add_argument('--omit-refusals', metavar='DIR',
                   help='--stages omit: the deep refusal record - an override of the RULED catalog_omission.REFUSALS (as --set '
                        'catalog_omission.REFUSALS=DIR: needs --ruling-for catalog_omission.REFUSALS=<ledger ID> or a --out '
                        'outside every production root; recorded in the run record)')


def default_out(args):
    """Production location of the stage's products: 10-s / 1-minute <SNAPSHOT_REVIEW_ROOT>/deep_omission/<label>, 10-minute
    <TRANSIENTS_ROOT>/deep_omission (sublst_cumulative.sh DOD). Refused while paths.PRODUCTION_WRITES is False: give --out."""
    cfg = _cfg()
    if getattr(args, 'tier', None) == TIER_SNAPSHOT:
        return os.path.join(cfg.paths.SNAPSHOT_REVIEW_ROOT, 'deep_omission', getattr(args, 'run_label', None) or 'Run_SNAPOMIT')
    return os.path.join(cfg.paths.TRANSIENTS_ROOT, 'deep_omission')


def wants(args):
    """True when the catalog step was asked for this stage (--stages omit, alone)."""
    st = getattr(args, 'stages', None)
    return bool(st) and [s.strip() for s in str(st).split(',') if s.strip()] == [STAGE]


def _runs_arg(args):
    v = getattr(args, 'omit_runs', None)
    if not v:
        raise SystemExit('error: catalog --tier snapshot --stages omit needs --omit-runs RUN,RUN|@FILE (the served batch runs)')
    if v.startswith('@'):
        items = [ln.split('#', 1)[0].strip() for ln in open(v[1:])]
    else:
        items = [x.strip() for x in v.split(',')]
    root = _cfg().paths.SNAPSHOT_REVIEW_ROOT
    return [x if os.path.isabs(x) else os.path.join(root, x) for x in items if x]


def refusals_arg(args):
    """The deep refusal record of the stage: catalog_omission.REFUSALS, a RULED key (the record Gregg's rulings name). --omit-
    refusals DIR is an override of it THROUGH THE CONFIG (CFG.set, as --set catalog_omission.REFUSALS=DIR): allowed with
    --ruling-for catalog_omission.REFUSALS=<ledger ID> or into a --out outside every production root (the CLI's authorize), and
    written into the run record (config_overrides, ruled_overrides) - never a bypass of the ruling (Q review F8). A refused
    override refuses the run (exit 2) before any unit."""
    v = getattr(args, 'omit_refusals', None)
    if v:
        from exopipe.phase3.config import ConfigRefusal
        try:
            _cfg().set('catalog_omission.REFUSALS', os.path.abspath(v), source='--omit-refusals')
        except ConfigRefusal as e:
            refuse(f'--omit-refusals {v}: {e}')
    return _c().REFUSALS


def run(args, run):
    """`exopipe3 catalog --tier snapshot|sublst --stages omit` - one unit '<label>.omit' (label = --run, else from the batch
    tags / 'Run_SLSTOMIT'). A gate refuses the unit (exit 2)."""
    tier = args.tier
    if tier not in (TIER_SNAPSHOT, TIER_TENMIN):
        raise SystemExit(f'error: --stages omit is a stage of the short tiers (snapshot, sublst), not {tier}')
    rdir = refusals_arg(args)
    injected_record_gate(rdir)               # N1: before the dry-run plan too, whichever route set the record
    out = os.path.abspath(run.out_root)
    if tier == TIER_SNAPSHOT:
        runs = _runs_arg(args)
        label = getattr(args, 'run_label', None) or 'Run_SNAPOMIT_' + '_'.join(batch_tag(r) for r in runs)[:120]
    else:
        from exopipe.phase3.core import cells
        raw = getattr(args, 'lsts', None)
        if not raw:
            raise SystemExit('error: catalog --tier sublst --stages omit needs --lsts (the LSTs of the cumulative)')
        lsts = [cells.parse_lst(x) for x in cells.expand(raw, universe=cells.LSTS, kind='LST')]
        label = getattr(args, 'run_label', None) or 'Run_SLSTOMIT'
    if run.dry_run:
        print(f'[{label}.{STAGE}] tier {tier}: refusal record {rdir}; out {out}; '
              + (f'runs {" ".join(runs)}' if tier == TIER_SNAPSHOT else f'LSTs {" ".join(lsts)}'))
        return 0
    with run.unit(f'{label}.{STAGE}', tier=tier, stage=STAGE, refusals=rdir) as u:
        if tier == TIER_SNAPSHOT:
            summ = run_snapshot(runs, out, rdir=rdir, u=u, label=label)
        else:
            summ = run_tenmin(lsts, out, rdir=rdir, u=u, label=label)
        u.meta['summary'] = summ
        u.note('deep omission (%s): %d decisions, %d OMIT keys, %d detections omitted%s' % (
            tier, summ['decisions'], summ['omit_keys'], summ['omitted'],
            (' of %d served' % summ['served']) if 'served' in summ else ''))
        dec = os.path.join(out, 'deep_omission.csv')
        u.add_output(dec, 'omission_decisions', tier=tier)
        u.add_output(os.path.join(out, 'deep_omission_dropped.csv'), 'omission_record', tier=tier, decisions=dec)
        if tier == TIER_SNAPSHOT:          # the drop the snapshot tier's cumulative step applies (Q review F2)
            u.add_output(os.path.join(out, 'deep_omission_kept.csv'), 'omission_kept', tier=tier, decisions=dec)
    return None
