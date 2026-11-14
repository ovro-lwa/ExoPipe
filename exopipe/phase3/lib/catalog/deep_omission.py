"""DEEP OMISSION - one stage shared by the short tiers (10-s, 1-minute, 10-minute): what the deep transient search refused
is omitted from their transient search, per hour, band and Stokes, unless the tier's own data clear it.

Rulings. Gregg 2026-10-24 ~20:30 UTC, verbatim: "Omission from the deep transient search is sufficient for omission from the
10s/1-min/10-min search, assessed on a per hour, per band basis" and "Approved with one critical exception - if removing noisy
frames is addressing the cause of the issues with the deep image, then there is no need to flag the full hour in the
10-s/1-min/10-min pipeline" (R-GREGG-20261024-DEEP-OMISSION-APPLIES-TO-SHORT-TIERS-WITH-FRAME-EXCEPTION); the design "All
approved. Release." ~23:45 UTC, the exception "unless the rate is anomalously high", each gate in its own Stokes scope, the
deep's processing failures not propagated (R-GREGG-20261024-DEEP-OMISSION-DESIGN-APPROVED-AND-10MIN-HOLD-RELEASED); the rebuild
from the merged refusal record (R-GREGG-20261031-SNAPSHOT-RESEARCH-LC-DEEPOMIT-GPU-RELEASE). Production: exopipe_ops
sublst_deep_omission.py + sublst_deep_omission_drop.py (10-minute; c6d2c9e, 0d4f1a9; dde1760 gate pairing, edf3263 re-judging,
35aad1a approval marker), snapshot_deep_omission.py, snapshot_rejudge.py and step 0 / section 0b of snapshot_combined_view.py
(10-s and 1-minute; 37546f6; dde1760 per image, fail closed, gate pairing; dbc40a6 Card 2, re-judging, the re-point fix, the rate
population) - ported here line for line (DECISIONS and DROP below; lib/catalog/rejudge.py the re-judging); the parity test is
tests/test_deep_omission.py at the production pin tests/production_pin.py (exopipe_ops fa90670).

DECISIONS, for every (LST, band, night, Stokes) the deep search refused (load_refusals: the LST-band's LATEST attempt, the
propagating gates catalog_omission.PROPAGATE, each in its own Stokes scope - searchtier by its Stokes column, sanity_I -> I,
sanity_V -> V, the others both) and each tier T of the short tier:
  (1) noise: the night as the tier admitted it - 10-s / 1-minute: the median global rms of the ADMITTED rows of the search's
      noise table (<SNAPSHOT_FLEET_OUT>/<LST>_<night>/noise_<band>_<S>_median[_1min].csv); 10-minute: the median central rms of
      the windows the search SEARCHED (v_deep_images.csv rows, night and residual from the window tree's frames.csv) - must be
      <= median + SIGMA x 1.4826 MAD over the LST's nights with that record (fewer than 3 nights: no threshold, not passed);
  (2) rate: the night's served detections per admitted row / window <= median + SIGMA x max(1.4826 MAD, RATE_FLOOR_PER_NIGHT /
      the median rows or windows per night) over those nights (fewer than 3 nights: passed); 10-s / 1-minute (dbc40a6,
      R-REFACTOR-20261104-OMISSION-RATE-POPULATION-SEARCHED-NIGHTS): the population is the nights of the cells the view's batches
      searched (chain_summary.json cells) - a searched night with nothing served is a true zero, one outside the view excluded;
      a decided night outside them has no served rate (blank) and passes; the Card 2 detections are not counted as served;
  KEEP when both pass, else OMIT; no record of the night in the tier: ABSENT (nothing searched, nothing to drop).
  WIDEBAND: a deep-refused band group night is decided the same way on its own record; a subband whose verdict is one of
  catalog_omission.WIDEBAND_MEMBER_VERDICTS for a night (production's OMIT or ABSENT - ruled 2026-11-04 by delegation, as the rate
  floor RATE_FLOOR_PER_NIGHT; WIDEBAND_EXCEPT excepted) also omits that night's detections of the band group it is a MEMBER of
  - 10-s / 1-minute: by the cell's wideband_members_<S>.csv, both tiers, PER IMAGE since dde1760 (the frame exception, Gregg
    2026-10-24, decided 2026-11-04 by the refactor chat on his principle: a wideband detection is omitted only when its own
    10-s frame / 1-minute block holds the omitted subband - the row carries wb_omitted_members; an image absent from the member
    table is omitted, fail closed, and refused here, G7); a cell with no member table (searched before 2026-10-07) takes a
    not-KEPT subband its per-subband pass admitted frames of (admission_<sb>.csv) to be in every image of its group that night
    (a whole-night row, fail closed); 10-minute: by each wideband window's own WBMEMB card (the 10-minute per-window rule is
    production-first, review T9).
  (A deep refusal of the Full band never omits anything here: the short tiers have no Full band, so its decisions are ABSENT,
  and Full is a member of no group - production's reading, Q review F11.)
DROP (the data level, before the cumulative catalogs, identities and pages):
  10-s / 1-minute (the combined view's step 0 and its per-tier omission parts): every served batch identity's detections (the
      identity's review_dir detections.csv rows; a detection = cand_id + peak position) whose (LST, subband or band, night,
      Stokes, tier) is OMIT are omitted; an identity left with none is dropped; per tier, a batch source left with none is
      dropped, a partial one whose best detection - the one of its own (ts, subband), dbc40a6 - went is re-pointed to its
      remaining detection of largest |S/N|, member rows and panels of omitted keys are dropped. CARD 2 (dbc40a6): a 10-s Stokes
      V detection under Stokes I Master flux > 20 x |V| (each batch's card2_vleak.csv, checked against itself, G8) leaves the
      view at every site the omission acts on. RE-JUDGING (dbc40a6): every batch's served sources are re-judged per tier on the
      detections they keep with the chain's own route functions (lib/catalog/rejudge.py, one child per batch, self-checked on
      the batch's record and served_tiers.csv, G9); a (batch identity, tier) that no longer qualifies loses its per-tier source
      rows, member rows and panels (deep_omission_rejudged.csv = production's rejudged.csv). This stage COMPUTES the
      drop and WRITES it as the kept product deep_omission_kept.csv; the drop is APPLIED where the combined view is assembled -
      the snapshot tier's cumulative step `serve --tier snapshot --family comb` (lib/serve/snapshot_view.py, the port of
      snapshot_combined_view.py; refactor chat 2026-11-03), which reads the kept product instead of recomputing it, checks the
      served set against its digests and refuses any row that does not apply exactly (Q review F2, N2).
  10-minute: the cumulative merge drops the OMIT rows per LST before concatenation (sublst_deep_omission_drop.py IN PLACE,
      reading the decision table this stage writes; `serve --tier sublst` build passes it as SUBLST_DEEP_OMISSION). This stage
      records the detections that drop removes (the member rows of every LST run whose key is OMIT) and the partial survivors
      its RE-JUDGING removes (edf3263, tenmin_rejudge: the builder's survivor rule on the members kept), refusing those the drop
      would refuse (G10) and, when any is removed, without Gregg's go (catalog_omission.TENMIN_REJUDGE_APPROVED, 35aad1a; G11).

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
10-s / 1-minute also deep_omission_served.csv (class omission_served: the served counts per (LST, night, band, Stokes, tier) the
decisions were made with, the Card 2 detections not counted - the build re-decides with them to show its decisions current,
V_combview review F5), deep_omission_rejudged.csv (class omission_rejudged: production's rejudged.csv, the touched (batch source,
tier) of the re-judging, dbc40a6; the children's jobs, results and logs in _rejudge/) and
deep_omission_kept.csv (class omission_kept, KEPT_COLS: what the cumulative step applies, every row with
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
under an injected run's start record, however it was named (Q fix verification N1); G4b a batch run whose per-tier survivors do
not hold exactly the identities its served_tiers.csv serves in each tier, or without that file; G4c a kept per-tier source whose
page holds no detection (detections.csv missing or empty: an incomplete batch run, never booked as a deep omission) (both
V_combview review F4); G2 a served run without its
catalogue (or a 10-minute LST run without its master_catalog.csv); G3 a served identity without detections, or a detection in
two identities; G4 a per-tier source or member row of no served identity; G5 kept + omitted != served; G6 served detections in
an ABSENT key (catalog_omission.ABSENT_WITH_SERVED = 'refuse', RULED R-REFACTOR-20261104-ABSENT-WITH-SERVED-REFUSE: the exception test
cannot be evaluated - never 'nothing to drop'; 'production' keeps them and counts them, as production does - the parity file's
value; the site is marked WAVE2 G6). Since the V2 follow-up round (exopipe_ops dde1760, dbc40a6, edf3263, 35aad1a): G2b a batch
run without its chain_summary.json cells (the rate population); G4d a partial source whose own (ts, subband) detection is not on
its page (production refuses: its re-point has no best); G7 a lookup of a per-image key finding no image in the cell's member
table (production omits such a detection, fail closed - here a key-format mismatch refuses, the review's T2); G8 Card 2: a batch
without card2_vleak.csv, a file disagreeing with itself, or a served flagged row matching no served detection by its key (T3);
G9 a re-judging child refusing (the chain or served_tiers.csv not reproduced, the record incomplete) or failing, or its inputs
missing; G10 (10-minute) a partial survivor whose survival rests on a statistic the drop does not re-derive; G11 (10-minute)
re-judging removing a source without Gregg's go. The V3 round: the 10-minute re-judging's 7-8 sigma gate is the builder's under
the configured size rule (tenmin_size_gate: none under 'envelope' - the rule judged every detection at the search - production's
1.7 beams under 'production', the parity file; the merge's in-place drop gets the same through P3_SIZE_GATE_7_8), and G12 refuses
LSTs whose search jobs were searched under another rule than the configured one (tenmin_rule_check); a re-judged-out source's
members are not recorded one by one and production's page line counts them (summary rejudged_members).

Per instant (Gregg 2026-10-18: a veto removes that instant's detections across frequency only, never other times or tiers;
2026-10-24: the omission is assessed "on a per hour, per band basis"): every removal names its OWN key - the (LST, night) hour
that holds the detection, its own band (or, across frequency only, the band group of an omitted member subband of the same
night), its own Stokes and its own tier (a 10-s decision never removes a 1-minute detection) - and carries the decision texts
of that key (column 'why', from why_part / member_gates / *_member_reason below, the templates the per-instant content test
reads: tests_per_instant.omission_per_instant, O1-O3, registered on omission_record).

DONE, the re-run of a partial source's survivor gates (Q review F6 OWED; Gregg 2026-11-04 R-GREGG-20261104-DEEP-OMISSION-SHORT-
TIERS-ANSWERS, "I see reflections coinciding with other types of sources, including possibly real ones, so this makes sense",
approved as one change in production and the refactor, the count shown to him before serving - his go recorded 2026-11-04T15:30Z):
the 10-s / 1-minute re-judging (production dbc40a6 snapshot_rejudge.py, ported lib/catalog/rejudge.py) and the 10-minute one
(production edf3263, in the merge's drop in place; predicted here, tenmin_rejudge).

PRODUCTION BUGS CARRIED FOR PARITY (Q review F1, F4, F10): F1 (the wideband consequence keyed on the night's union of members)
and F4 (deep_value / deep_threshold in the record's listing order) are FIXED in production (exopipe_ops dde1760; 10-s / 1-minute
F1, both tiers' F4) and here. Still carried, pinned by test_production_bugs_pinned:
  F1 (10-minute): ONE window of the night whose WBMEMB holds the subband omits the group's whole night (dde1760 left it; the
     review's T9: production first);
  F4 (meaning): deep_threshold_meaning follows the first (sorted) gate (_meaning);
  F10 (text, production's): with fewer than 3 nights there is no noise threshold (nan, never passed) and the reason still reads
     RMS_ABOVE ('tier rms above threshold') - production's byte-identical table text and the 10-minute merge's 'why' (_verdict).
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
import shutil
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
    """The 10-s / 1-minute member reason BEFORE exopipe_ops dde1760 (2026-11-04): the whole night's wideband detections omitted.
    No longer written; the per-instant templates keep it for decision tables written before (served views of 11-02 and earlier)."""
    return f"wideband-member: {members} this night, a member of the cell's {group} frames"


def snapshot_member_image_reason(members, group):
    """A member-derived wideband row applied PER IMAGE (the frame exception; exopipe_ops dde1760)."""
    return f'wideband-member: {members} this night - omitted only in the {group} images that include it (frame exception)'


def snapshot_member_failclosed_reason(members, group):
    """A member-derived wideband row of a cell with no wideband member table (fail closed; exopipe_ops dde1760)."""
    return (f'wideband-member: {members} this night - the cell has no wideband member table (searched before 2026-10-07): '
            f'every {group} image of the night is taken to include it (fail closed)')


def member_list(why):
    """wb_omitted_members of a per-image member row: the omitted member subbands, sorted by frequency (production's
    ' '.join(sorted({w.split()[0] for w in why}, key=lambda b: int(b[:-3]))))."""
    return ' '.join(sorted({w.split()[0] for w in why}, key=lambda b: int(b[:-3])))


def tenmin_member_reason(members):
    return f'wideband-member: {members} this night, in the WBMEMB of its wideband windows'


def why_part(gates, reason):
    """One OMIT decision's text in the record's 'why' (production's WHY: '%s: %s' % (gates, reason))."""
    return f'{gates}: {reason}'


WHY_SEP = ' | '
SNAPSHOT_PARTIAL_NOTE = 'best detection omitted; re-pointed (survivor gates not re-run)'   # before dbc40a6 (the 11-02 rule)
# the RE-POINT of a partial source (exopipe_ops dbc40a6; the refactor port review's defect RP fixed): its best detection is the one
# of its own (ts, subband); when that is removed the source is re-pointed to the largest-|S/N| detection it keeps
REPOINT_NOTE = 'best detection removed (deep omission or Card 2); re-pointed to the largest-|SNR| detection it keeps'
REPOINT_FIELDS = ('best_cand_id', 'best_snr', 'ts', 'subband', 'fit_verdict', 'peak_jy_best', 'deep_omission')
SOURCE_EMPTY_NOTE = 'every detection omitted'
SOURCE_EMPTY_CARD2_NOTE = 'every detection omitted or removed by Card 2'
IDENTITY_EMPTY_NOTE = 'identity left with no detection'


def repoint_record_note(row, top):
    """The record's note of a re-pointed partial source (snapshot_combined_view.py dbc40a6)."""
    return 'best %s %s (%s) removed; re-pointed to %s (%s)' % (row.get('subband'), row.get('ts'), row.get('best_snr'),
                                                              top['cand_id'], top['snr'])
SNAPSHOT_COLS = ['lst', 'band', 'night', 'stokes', 'gates', 'deep_value', 'deep_threshold', 'deep_attempt', 'tier', 'n_admitted',
                 'tier_median_rms', 'tier_rms_threshold', 'rms_pass', 'rate', 'rate_threshold', 'rate_pass', 'verdict', 'reason',
                 'deep_threshold_meaning', 'rate_definition']
SNAPSHOT_WB_COL = 'wb_omitted_members'      # dde1760: a member-derived row's omitted member subbands (per image); the union
                                           # header ends with it once a member row is written (its key sits before 'reason')
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
#   'source re-pointed'   ... source_id, cand_id / snr = its new best; fields = JSON of the survivors row's new values in
#                         production's order (REPOINT_FIELDS, then rejudged_routes when it still qualifies after re-judging);
#                         note = the record's (dbc40a6: the best = the detection of the row's own (ts, subband))
#   'source routes'       a partial source whose best detection is kept, still qualifying after re-judging: fields =
#                         {"rejudged_routes": its routes on the members it keeps} (dbc40a6)
#   'source re-judged'    run, batch, identity, catalog (sources file), source_id, stokes, tier, note = the re-judging's (its
#                         (batch identity, tier) no longer qualifies on the detections it keeps; dbc40a6)
#   'member row re-judged'  run, batch, identity, catalog (members file), source_id, cand_id, stokes, tier (dbc40a6)
#   'card2 member row dropped'  ... a member row of a frame where every detection of its batch identity is a Card 2 one
#   'card2 panel dropped' run, batch, catalog (the review_dir), source_id, stokes, tier, note = the file (10-s Stokes V, Card 2)
#   'member row dropped'  run, batch, catalog (<cdir>/<band>_source_members.csv), source_id, cand_id, stokes, tier, the key
#                         (lst, band, night) when its own key is OMIT, else note 'identity left with no detection'
#   'panel dropped'       run, batch, catalog (the source's review_dir), source_id, stokes, tier, the key, note = the file
KEPT_COLS = ['kind', 'run', 'batch', 'identity', 'catalog', 'source_id', 'cand_id', 'ra_peak', 'dec_peak', 'lst', 'band', 'night',
             'stokes', 'tier', 'snr', 'note', 'fields']
KEPT_KINDS = ('detection kept', 'source dropped', 'source re-pointed', 'source routes', 'source re-judged', 'member row dropped',
              'member row re-judged', 'card2 member row dropped', 'panel dropped', 'card2 panel dropped')
# a snapshot detection's key from its cand_id (snapshot_combined_view.py CIDK) and a panel's (PANEL); a 10-minute detection's
# (sublst_deep_omission_drop.CID)
SNAP_CID = re.compile(r'^(\d\dh)_(\d{4}-\d\d-\d\d)_([A-Za-z0-9]+)_([IV])_(10s|1min)_')
SNAP_PANEL = re.compile(r'^rank\d+_(\d\dh)_(\d{4}-\d\d-\d\d)_([A-Za-z0-9]+?)(\d{14})')   # group 4: the image's ts (dde1760)
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


class OmissionRefusal(Exception):
    """A gate of this module refusing in a STANDALONE load (the combined view's and the re-judging's Python 3.6 children load
    this file without the exopipe package): the caller turns it into its own refusal."""


def refuse(msg):
    try:
        from exopipe.phase3.core.record import Refusal
    except ImportError:
        raise OmissionRefusal(msg)
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
    WIDEBAND_MEMBER_VERDICTS (production's values, ruled 2026-11-04) and ABSENT_WITH_SERVED (the refactor's 'refuse', ruled 2026-11-04). cli.main refuses the step on
    any (exit 2, before the run, as B-30 / B-32 do for theirs): an invalid value given with --set or --config never starts a run
    that would stop late with an error, and the floor is checked even where no cell has 3 nights (Q fix verification N3)."""
    if cfg is None:
        from exopipe.phase3.config import CFG as cfg
    c = cfg.catalog_omission
    return [p for p in (_rate_floor_problem(c.RATE_FLOOR_PER_NIGHT), _member_verdicts_problem(c.WIDEBAND_MEMBER_VERDICTS),
                        _absent_mode_problem(c.ABSENT_WITH_SERVED)) if p]


def rate_floor():
    """catalog_omission.RATE_FLOOR_PER_NIGHT (production's value, ruled 2026-11-04 by delegation, R-GREGG-20261104-DEEP-OMISSION-SHORT-
    TIERS-ANSWERS; 0 = the card's rule without a floor)."""
    v = _c().RATE_FLOOR_PER_NIGHT
    bad = _rate_floor_problem(v)
    if bad:
        raise ValueError(bad)
    return float(v)


def member_verdicts():
    """catalog_omission.WIDEBAND_MEMBER_VERDICTS (production's ('OMIT', 'ABSENT'), ruled 2026-11-04 by delegation): the verdicts
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
    """The record's files of one LST, in the order production reads them (glob - the directory's listing order, unsorted). Since
    exopipe_ops dde1760 the order only decides the order of two rows of ONE gate name (the gates sorted stably, _base)."""
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
    """The decision row's gate fields: names, values and thresholds in ONE order - the gates sorted by name, stably, none
    de-duplicated (exopipe_ops dde1760, both tiers: the Q review's F4 fixed; before, the names were sorted and de-duplicated and
    the values / thresholds in the record's read order, so a value could sit beside another gate's name)."""
    gs = sorted(gates, key=lambda g: g[0])
    return dict(lst=lst, band=band, night=night, stokes=S, gates=';'.join(g[0] for g in gs),
                deep_value=';'.join(g[1] for g in gs), deep_threshold=';'.join(g[2] for g in gs), deep_attempt=gates[0][3])


def _meaning(r):
    # PRODUCTION TEXT (Q review F4, the part dde1760 kept): the meaning follows the FIRST (sorted) gate, which may be another gate
    # than a threshold it describes (e.g. 'beam;sanity' reads GATE_MEANING for the sanity gate's cell median too)
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

    def __init__(self, proto, reader, names=None):
        """names: {'noise': {tier: pattern}, 'members': pattern, 'admission': pattern} - the file names (default the config's
        catalog_omission.SNAPSHOT_NOISE / SNAPSHOT_MEMBERS / SNAPSHOT_ADMISSION; a standalone load passes them)."""
        self.proto, self.reader = proto, reader
        self._ns, self._wm = {}, {}
        if names is None:
            c = _c()
            names = dict(noise=dict(c.SNAPSHOT_NOISE), members=c.SNAPSHOT_MEMBERS, admission=c.SNAPSHOT_ADMISSION)
        self.names = names

    def cells(self, lst):
        """The LST's cell directories <proto>/<LST>_<night>, sorted (snapshot_deep_omission.night_stats' glob)."""
        if ('cells', lst) not in self._ns:
            self._ns[('cells', lst)] = sorted(glob.glob(os.path.join(self.proto, '%s_*' % lst)))
        return self._ns[('cells', lst)]

    def prefetch(self, lst, keys):
        """Read ahead (in parallel) the noise tables of ``keys`` [(band, S)] in both tiers over every cell of the LST - a
        missing file is remembered as missing; the decisions then read them in production's order (each one once)."""
        self.reader.prefetch(os.path.join(d, self.names['noise'][T].format(band=b, stokes=S)) for d in self.cells(lst)
                             for b, S in sorted(set(keys)) for T in sorted(self.names['noise']))

    def night_stats(self, lst, band, S, T):
        """{night: (median admitted global rms Jy/beam, n admitted rows)} over every searched night of the LST with this table."""
        import numpy as np
        k = (lst, band, S, T)
        if k not in self._ns:
            out = {}
            name = self.names['noise'][T].format(band=band, stokes=S)
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

    def member_rows(self, lst, night, S):
        """The rows of the cell's wideband_members_<S>.csv (the search's), or None without the file - read once."""
        k = ('rows', lst, night, S)
        if k not in self._wm:
            p = os.path.join(self.proto, '%s_%s' % (lst, night), self.names['members'].format(stokes=S))
            t = self.reader.text(p, 'snapshot_members', missing_ok=True)
            self._wm[k] = None if t is None else list(csv.DictReader(io.StringIO(t, newline='')))
        return self._wm[k]

    def has_members(self, lst, night, S):
        """True when the cell has a wideband member table of Stokes S (searched since the frame-time fix of 2026-10-07)."""
        return self.member_rows(lst, night, S) is not None

    def members(self, lst, night, S):
        """{group: set of member subbands} of the cell's wideband frames (the search's wideband_members_<S>.csv)."""
        k = (lst, night, S)
        if k not in self._wm:
            out = collections.defaultdict(set)
            for r in self.member_rows(lst, night, S) or ():
                out[r['group']].add(r['subband'])
            self._wm[k] = out
        return self._wm[k]

    def searched(self, lst, night, sb):
        """True when the cell's per-subband pass admitted at least one frame of sb (admission_<sb>.csv, joint over Stokes;
        snapshot_deep_omission.searched_sb) - dde1760's fail-closed rule for a cell without a member table."""
        k = ('admitted', lst, night, sb)
        if k not in self._wm:
            p = os.path.join(self.proto, '%s_%s' % (lst, night), self.names['admission'].format(band=sb))
            rows = self.reader.rows(p, 'snapshot_admission')
            self._wm[k] = any(str(r.get('admitted', '')) in ('1', 'True') for r in rows)
        return self._wm[k]

    def image_members(self, lst, night, S, T):
        """{(group, image id): set of member subbands} of the cell's wideband images of tier T (snapshot_deep_omission
        .wb_image_members): 10-s - the wideband frame's ts; 1-minute - the block of the frame's integration midpoint
        (clv15 phase3_snapshot_search.integration_minute_key + '00', in place), a block's members the union over its frames."""
        k = ('images', lst, night, S, T)
        if k not in self._wm:
            out = collections.defaultdict(set)
            mk = minute_key() if T != '10s' else None
            for r in self.member_rows(lst, night, S) or ():
                iid = r['ts'] if T == '10s' else mk(r['date_obs']) + '00'
                out[(r['group'], iid)].add(r['subband'])
            self._wm[k] = out
        return self._wm[k]


_MINUTE_KEY = []


def minute_key():
    """clv15 phase3_snapshot_search.integration_minute_key (in place: the 1-minute block of a frame, as the search builds it)."""
    if not _MINUTE_KEY:
        root = _legacy_clv15()
        if root not in sys.path:
            sys.path.insert(0, root)
        from phase3_snapshot_search import integration_minute_key
        _MINUTE_KEY.append(integration_minute_key)
    return _MINUTE_KEY[0]


def _legacy_clv15():
    """The clv15 root (core.env.legacy_root; a standalone load under Python 3.6 takes the site's default)."""
    try:
        from exopipe.phase3.core import env
        return env.legacy_root('clv15')
    except ImportError:
        return os.environ.get('EXOPIPE3_CLV15', '/lustre/gh/main/clv15')


def decide_snapshot(lsts, served, rdir=None, proto=None, reader=None, cells=None):
    """The 10-s and 1-minute decisions (snapshot_deep_omission.decide, exopipe_ops dbc40a6): rows in production's order and
    formats. served = {(lst, night, band, stokes, tier): n served detections} (the combined view's served set less the Card 2
    detections). cells (dbc40a6; R-REFACTOR-20261104-OMISSION-RATE-POPULATION-SEARCHED-NIGHTS): the cells the view's batches
    searched ('<LST>_<night>'); given, the RATE condition's population is those nights only - a searched night with nothing
    served is a true zero, a night of a batch outside the view is excluded - and a decided night outside them has no served
    rate (rate blank, the condition passes); None = every night with a noise table (before dbc40a6). The rms condition always
    reads every night with a noise table."""
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
        inview = (lambda nt: True) if cells is None else (lambda nt, _l=lst: '%s_%s' % (_l, nt) in cells)
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
                rate = {nt: served.get((lst, nt, band, S, T), 0) / float(n) for nt, (m, n) in st.items() if n > 0 and inview(nt)}
                rthr = rate_thr(rate.values(), [n for nt, (m, n) in st.items() if n > 0 and inview(nt)])
                med, n = st[night]
                rp = bool(np.isfinite(med) and np.isfinite(thr) and med <= thr)
                ap = bool(not np.isfinite(rthr) or night not in rate or rate[night] <= rthr)
                verdict, reason = _verdict(rp, ap)
                lrows.append(dict(base, tier=T, n_admitted=n, tier_median_rms='%.4g' % med, tier_rms_threshold='%.4g' % thr,
                                  rms_pass=int(rp), rate=('%.4g' % rate[night]) if night in rate else '',
                                  rate_threshold='%.4g' % rthr, rate_pass=int(ap), verdict=verdict, reason=reason))
        # wideband consequences (WIDEBAND_EXCEPT excepted): a subband night not KEPT in a tier (its verdict one of
        # WIDEBAND_MEMBER_VERDICTS - production's OMIT or ABSENT, ruled 2026-11-04) omits that night's band-group detections whose
        # OWN image holds the subband (the frame exception, exopipe_ops dde1760: the row carries wb_omitted_members and is
        # applied per image - Omission.key / image_omitted); a cell with no member table of the Stokes (searched before
        # 2026-10-07) takes a subband its per-subband pass admitted frames of to be in EVERY image of the group that night - a
        # whole-night row, fail closed (wb_omitted_members empty)
        mv = member_verdicts()
        wb, nomt = collections.defaultdict(set), set()
        for r in lrows:
            if r['verdict'] in mv and r['band'] not in G and r['band'] not in c.WIDEBAND_EXCEPT:
                mem = recs.members(lst, r['night'], r['stokes'])
                has_t = recs.has_members(lst, r['night'], r['stokes'])
                for g, sbs in G.items():
                    if r['band'] not in sbs:
                        continue
                    if r['band'] in mem.get(g, ()):
                        wb[(g, r['night'], r['stokes'], r['tier'])].add(member_word(r['band'], r['verdict']))
                    elif not has_t and recs.searched(lst, r['night'], r['band']):        # no member table: fail closed
                        wb[(g, r['night'], r['stokes'], r['tier'])].add(member_word(r['band'], r['verdict']))
                        nomt.add((g, r['night'], r['stokes'], r['tier']))
        for (g, night, S, T), why in sorted(wb.items()):
            m = ', '.join(sorted(why))
            per_image = (g, night, S, T) not in nomt
            lrows.append(dict(lst=lst, band=g, night=night, stokes=S, gates=member_gates(m), deep_value='', deep_threshold='',
                              deep_attempt='', tier=T, n_admitted='', tier_median_rms='', tier_rms_threshold='', rms_pass='',
                              rate='', rate_threshold='', rate_pass='', verdict='OMIT',
                              wb_omitted_members=member_list(why) if per_image else '',
                              reason=(snapshot_member_image_reason(m, g) if per_image else
                                      snapshot_member_failclosed_reason(m, g))))
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
        self._st = {}                            # {window path: (size, int mtime) or None}: its status, read once per run

    def stats(self, paths):
        """{path: (size, int mtime)} of the paths that exist: ONE os.stat per path, on catalog_omission.TENMIN_STAT_THREADS
        threads (a searched window is a link to another Lustre file: about 70 ms a status call when cold, 47,433 cached
        windows on 2026-11-14 - the 50-110 min of the table when they are read one after another, as production does;
        measured 4.0 ms each on 16 threads). The values are what the serial calls return; remembered for write_cache."""
        todo = [p for p in dict.fromkeys(paths) if p not in self._st]

        def one(p):
            try:
                st = os.stat(p)
                return p, (st.st_size, int(st.st_mtime))
            except OSError:
                return p, None
        n = max(1, int(getattr(_c(), 'TENMIN_STAT_THREADS', 1) or 1))
        if n > 1 and len(todo) > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(n) as ex:
                self._st.update(ex.map(one, todo))
        else:
            self._st.update(one(p) for p in todo)
        return {p: self._st[p] for p in paths if self._st.get(p) is not None}

    def load_cache(self, lst):
        """{window path: rms} of the LST's cache files whose window still has the size and mtime it was measured at
        (sublst_deep_omission.decide; the seed first, the stage's own cache over it). The windows' status is read once per
        path, on threads (``stats``), not once per cache row in turn: the same rows pass."""
        cache, rows = {}, []
        for d, tag in ((self.seed_dir, 'seed'), (self.cache_dir, 'cache')):
            f = os.path.join(d, 'window_rms_%s.csv' % lst) if d else None
            if not f or not os.path.exists(f):
                continue
            rows += [(r, tag) for r in csv.DictReader(open(f, newline=''))]
        st = self.stats([r['path'] for r, _t in rows])
        for r, tag in rows:
            s = st.get(r['path'])
            if s is None:
                continue
            if r.get('size') == str(s[0]) and r.get('mtime') == str(s[1]):
                cache[r['path']] = (float(r['rms']), tag)
        return {k: v for k, (v, _t) in cache.items()}, collections.Counter(t for _v, t in cache.values())

    def write_cache(self, lst, cache):
        """The LST's cache file (every window measured or read this run; sublst_deep_omission.decide), atomically. A window's
        size and mtime are the ones its value was validated or measured at in this run (``stats`` / ``rms``) - production
        reads the status again here, a second pass over every window that returns the same values unless a window was
        rewritten during the run (and then production records the new status beside the OLD value); a window with no status
        of this run is read here, as production."""
        if not self.cache_dir:
            return
        os.makedirs(self.cache_dir, exist_ok=True)
        f = os.path.join(self.cache_dir, 'window_rms_%s.csv' % lst)
        st = self.stats(list(cache))
        with open(f + '.tmp', 'w', newline='') as o:
            w = csv.writer(o)
            w.writerow(['path', 'size', 'mtime', 'rms'])
            for k, v in cache.items():
                if k in st:
                    w.writerow([k, st[k][0], st[k][1], v])
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
            s = self._st.get(path)                               # read on threads before the median (decide_tenmin)
            if s is None:
                st = os.stat(path)                               # (a path without a status: raises, as before)
                s = self._st[path] = (st.st_size, int(st.st_mtime))   # the status this value belongs to (write_cache)
            rec = self._fr.get(path)
            if rec and rec[0] == str(s[0]) and rec[1] == str(s[1]):
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
        recs.stats([path for _k, _n, path in W[S] if path not in cache])   # the windows no cache row holds: their status
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
    # WIDEBAND_MEMBER_VERDICTS (production's OMIT or ABSENT, ruled 2026-11-04) - omits that night's wideband detections of a group
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
    """{(lst, band, night, stokes, tier)} of the WHOLE-NIGHT OMIT rows (tier '10min' for a 10-minute table). A member-derived
    wideband row with wb_omitted_members (10-s / 1-minute, exopipe_ops dde1760) is applied per image: omitted_wb, Omission."""
    return {(r['lst'], r['band'], r['night'], r['stokes'], r.get('tier') or '10min') for r in rows
            if r['verdict'] == 'OMIT' and not (r.get(SNAPSHOT_WB_COL) or '').strip()}


def omitted_wb(rows):
    """{(lst, group, night, stokes, tier): frozenset of omitted member subbands} of the member-derived wideband rows applied per
    image (snapshot_deep_omission.omitted_wb)."""
    return {(r['lst'], r['band'], r['night'], r['stokes'], r['tier']): frozenset(r[SNAPSHOT_WB_COL].split()) for r in rows
            if r['verdict'] == 'OMIT' and (r.get(SNAPSHOT_WB_COL) or '').strip()}


def omit_keys_all(rows):
    """Every OMIT key, whole-night or per image (the record's 'why' and the counts by tier read both)."""
    return {(r['lst'], r['band'], r['night'], r['stokes'], r.get('tier') or '10min') for r in rows if r['verdict'] == 'OMIT'}


def panel_image(digits):
    """A panel name's 14 digits (YYYYMMDDHHMMSS) as the image id YYYYMMDD_HHMMSS (snapshot_combined_view.py dde1760)."""
    return digits[:8] + '_' + digits[8:]


class Omission:
    """The decisions applied to a detection, member row or panel (snapshot_combined_view.py dde1760 omitted / d_omitted): a
    whole-night OMIT key, or a per-image key whose image (a 10-s frame's ts, a 1-minute block's ts) holds an omitted member
    subband - an image absent from the cell's member table is omitted (fail closed, never silently kept; production's
    wb_image_omitted). The absent images are COUNTED by key (V_combview review T2: a key-format mismatch would omit every such
    detection silently; production does not count them) - Omission.absent, gated by the stage."""

    def __init__(self, rows, recs):
        self.OM, self.OMWB = omitted_keys(rows), omitted_wb(rows)
        self.recs = recs
        self.absent = collections.Counter()          # key -> lookups of an image the member table does not hold

    def image_omitted(self, key, iid):
        om = self.OMWB.get(key)
        if not om:
            return False
        lst, g, night, S, T = key
        mem = self.recs.image_members(lst, night, S, T).get((g, iid))
        if mem is None:
            self.absent[key] += 1
            return True
        return bool(mem & om)

    def key(self, k, iid):
        return k in self.OM or (k in self.OMWB and self.image_omitted(k, iid))

    def det(self, d):
        return self.key(okey(d), d['ts'])



# =====================================================================================================================
# CARD 2 PER FRAME in the 10-s tier (exopipe_ops dbc40a6 snapshot_rejudge.card2_removed + snapshot_combined_view.py section 0b;
# Gregg 2026-11-04 "Approved.", R-GREGG-20261104-CARD2-PER-FRAME-SNAPSHOT-10S-APPROVED): a Stokes V 10-s detection under Stokes I
# Master flux > 20 x |V| (ratio catalog_omission.CARD2_RATIO 0.05) leaves the view at every site the omission acts on and is not
# counted as served in the omission's rate test. The batches predate the chain's own Card 2 (clv15 1279bd4), so the view applies
# it AFTER the chain, from each batch's card2_vleak.csv (phase3_snapshot_card2.py's back-fill), checked against itself.
# =====================================================================================================================

def card2_removed(run, reader=None, ratio=0.05, name='card2_vleak.csv'):
    """(set of det_uid, {det_uid: (cand_id, ra_peak, dec_peak, identity_jname)}, counts) of the batch's 10-s Stokes V detections
    Card 2 rejects, from <run>/card2_vleak.csv after checking the file against itself (snapshot_rejudge.card2_removed): every
    row's ratio recomputed from its own peak_jyb and i_here_jy, the flag from the ratio (within half a unit of the file's 5
    decimals of the threshold the file's own flag stands, counted). G8 refuses a file that disagrees with itself, or none."""
    p = os.path.join(run, name)
    if reader is not None:
        text = reader.text(p, 'card2', missing_ok=True)
    else:
        try:
            with open(p, newline='') as fh:
                text = fh.read()
        except FileNotFoundError:
            text = None
    if text is None:
        refuse('G8 Card 2: %s has no %s (Card 2 cannot be applied)' % (run, name))
    rem, info, n = set(), {}, collections.Counter()
    for r in csv.DictReader(io.StringIO(text, newline='')):
        n['rows'] += 1
        v, ih = r['v_over_i_here'], r['i_here_jy']
        if v != '':
            vv = abs(float(r['peak_jyb'])) / float(ih)
            if abs(vv - float(v)) > max(1e-4, 1e-3 * vv):
                refuse('G8 Card 2: %s row %s: v_over_i_here %s but |peak|/i_here = %.5f' % (p, r['det_uid'], v, vv))
            if abs(float(v) - ratio) <= 0.5e-5:
                n['at_threshold_within_rounding'] += 1
            elif float(r['card2_ratio']) != ratio or (r['card2_would_remove'] == '1') != (float(v) <= ratio):
                refuse('G8 Card 2: %s row %s: card2_would_remove %s disagrees with its ratio %s at %s'
                       % (p, r['det_uid'], r['card2_would_remove'], v, r['card2_ratio']))
            if float(r['card2_ratio']) != ratio:
                refuse('G8 Card 2: %s row %s: ratio %s, not %s' % (p, r['det_uid'], r['card2_ratio'], ratio))
        elif r['card2_would_remove'] == '1':
            refuse('G8 Card 2: %s row %s: flagged without a ratio' % (p, r['det_uid']))
        if r['tier'] == '10s' and r['card2_would_remove'] == '1':
            rem.add(r['det_uid'])
            info[r['det_uid']] = (r['cand_id'], r['ra_peak'], r['dec_peak'], r['identity_jname'], r['served'] == '1')
            n['removed_10s'] += 1
            n['removed_10s_served'] += int(r['served'] == '1')
        elif r['tier'] == '10s' and v == '':
            n['not_evaluated_10s'] += 1
    return rem, info, n


def nk(cand, ra, dec):
    """A detection key at the record's 5 decimals - card2_vleak.csv's and detections.csv's (snapshot_combined_view.py nk)."""
    return (cand, '%.5f' % float(ra), '%.5f' % float(dec))


def det_nk(d):
    return nk(d['cand_id'], d.get('ra_peak') or d['ra'], d.get('dec_peak') or d['dec'])


class Card2:
    """Card 2 over the view's batch runs (snapshot_combined_view.py dbc40a6 section 0b): per run its det_uids (the re-judging's
    input), the detection keys of every batch (C2DK, one set, as production), the counts; det(d) is d_card2; jc / jc_card2 are
    the per-(run, identity, frame) counts (a member row or a panel, which carry no position, is a Card 2 one when every
    detection of its batch identity at its frame is)."""

    def __init__(self, runs, reader=None, ratio=0.05, name='card2_vleak.csv'):
        self.uids, self.info = {}, {}
        self.DK = set()
        self.n = collections.Counter()
        for r in runs:
            u_, info_, n_ = card2_removed(r, reader, ratio, name)
            self.uids[r] = u_
            self.info[r] = info_
            self.DK.update(nk(c_, a_, b_) for c_, a_, b_, _j, _s in info_.values())
            self.n.update(n_)
        self.JC_ALL, self.JC_C2 = collections.Counter(), collections.Counter()

    def det(self, d):
        return bool(self.DK) and d.get('tier') == '10s' and d.get('stokes') == 'V' and det_nk(d) in self.DK

    def jc_card2(self, r, J, cand):
        return self.JC_C2[(r, J, cand)] > 0 and self.JC_C2[(r, J, cand)] == self.JC_ALL[(r, J, cand)]

    def n_mixed(self):
        return sum(1 for k_, v_ in self.JC_C2.items() if v_ != self.JC_ALL[k_])

    def served_keys(self, r):
        """[(det_uid, key)] of the run's SERVED Card 2 rows (card2_vleak.csv served = 1) - the T3 count's side of the file."""
        return [(u, nk(c_, a_, b_)) for u, (c_, a_, b_, _j, s_) in sorted(self.info[r].items()) if s_]


def card2_coverage(card2, pending):
    """V_combview review T3: every SERVED Card 2 row of each batch's card2_vleak.csv must match a served detection of that batch
    (the key nk of the row and of the detection), omitted or not - a row whose key is formatted otherwise would leave Card 2
    unapplied silently, and production's T7 reads the same keys. Returns (matched served rows, the file's served count, [rows
    matching no served detection])."""
    keys = collections.defaultdict(set)
    for r, J, _g, dets in pending:
        for d in dets:
            if d.get('tier') == '10s' and d.get('stokes') == 'V':
                keys[r].add(det_nk(d))
    matched, total, miss = 0, 0, []
    for r in card2.uids:
        for u, k in card2.served_keys(r):
            total += 1
            if k in keys[r]:
                matched += 1
            else:
                miss.append((batch_tag(r), u, k))
    return matched, total, miss


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


def served_counts(pending, card2=None):
    """({(lst, night, band, stokes, tier): n}, n_all) - decide_snapshot's served argument (the combined view's re-keyed Counter)
    and every served detection. With card2 (dbc40a6) the Card 2 detections are not counted as served in the rate test - they
    leave the view whatever the omission decides - but n_all counts them."""
    c = collections.Counter(okey(d) for p in pending for d in p[3] if card2 is None or not card2.det(d))
    return {(k[0], k[2], k[1], k[3], k[4]): n for k, n in c.items()}, sum(len(p[3]) for p in pending)


CARD2_WHY = 'card2: Stokes I Master flux under it > 20 x |V| (Card 2 per frame, 10 s; Gregg 2026-11-04)'
ALSO_CARD2 = 'also card2'          # the note of an omitted detection Card 2 would remove too (not counted as served, K5)


def apply_snapshot(pending, om, why, card2):
    """The identity-level drop (the combined view's step 0 and section 0b, dbc40a6): a detection on an omitted key (per image)
    is omitted, else one Card 2 removes is removed ('card2 detection'), else kept; an identity left with none is dropped.
    Returns (kept [(run, identity, rows, kept detections)], dropped {(run, identity)}, record rows). G3 a kept detection in two
    identities; G5 conservation (omitted + Card 2 + kept = served). Fills card2.JC_ALL / JC_C2 (the frame counts)."""
    kept, dropped, rec = [], set(), []
    n_all = sum(len(p[3]) for p in pending)
    for r, J, grs, dets in pending:
        keep = []
        tag = batch_tag(r)
        for d in dets:
            k = okey(d)
            if om.det(d):
                rec.append(dict(kind='detection', batch=tag, identity=J, cand_id=d['cand_id'], lst=d['lst'], band=k[1],
                                night=d['date'], stokes=d['stokes'], tier=d['tier'], snr=d['snr'], why=why[k],
                                note=ALSO_CARD2 if card2.det(d) else ''))
            elif card2.det(d):
                rec.append(dict(kind='card2 detection', batch=tag, identity=J, cand_id=d['cand_id'], lst=d['lst'], band=k[1],
                                night=d['date'], stokes=d['stokes'], tier=d['tier'], snr=d['snr'], why=CARD2_WHY))
                card2.JC_C2[(r, J, d['cand_id'])] += 1
            else:
                keep.append(d)
            card2.JC_ALL[(r, J, d['cand_id'])] += 1
        if not keep:
            dropped.add((r, J))
            rec.append(dict(kind='identity dropped', batch=tag, identity=J))
            continue
        kept.append((r, J, grs, keep))
    n_om = sum(1 for x in rec if x['kind'] == 'detection')
    n_c2 = sum(1 for x in rec if x['kind'] == 'card2 detection')
    n_in = sum(len(p[3]) for p in kept)
    if n_in + n_om + n_c2 != n_all:
        refuse(f'G5: omission record {n_om} + Card 2 record {n_c2} + kept {n_in} != served {n_all}')
    keys = collections.Counter(dkey(d) for p in kept for d in p[3])
    dup = [k for k, n in keys.items() if n > 1]
    if dup:
        refuse(f'G3: {len(dup)} detections in two batch identities (e.g. {dup[:2]}) - the batches are not cell-disjoint')
    return kept, dropped, rec


class Rejudged:
    """The re-judging's results over the view's batch runs (snapshot_combined_view.py dbc40a6, the REJ_* of section 0b): per
    (run, identity, tier) re-judged OUT (it qualified, it no longer does, and it kept members: 'every member removed' is the
    omission's) with its note, the routes of one that still qualifies, the rows of rejudged.csv in production's order, and the
    counts the view's summary reports."""

    def __init__(self):
        self.OUT, self.NOTE, self.ROUTES = set(), {}, {}
        self.ROWS = []                       # rejudged.csv: [(batch, row items)]
        self.N, self.RM = collections.Counter(), collections.Counter()
        self.G3D = 0
        self.absent = collections.Counter()  # images absent from a member table the re-judging's omission met (T2)

    def add(self, run, res):
        tag = batch_tag(run)
        for J, T, items in res['rejudge']:
            row = collections.OrderedDict(items)
            self.ROWS.append(collections.OrderedDict([('batch', tag)] + list(row.items())))
            if row['qualifies_before'] == 1 and row['qualifies_after'] == 1:
                self.ROUTES[(run, J, T)] = row['routes_after']
            if row['qualifies_before'] == 1 and row['qualifies_after'] == 0 and row['reason_after'] != 'every member removed':
                self.OUT.add((run, J, T))
                self.NOTE[(run, J, T)] = 're-judged on %d of %d members (%s removed): %s' % (
                    row['n_members'] - row['n_removed'], row['n_members'], row['removed_by'], row['reason_after'])
        self.N.update(res['counts'])
        self.G3D += res['g3d_neighbours']
        for k, v in res['n_removed']:
            self.RM[k] += v
        for k, v in res.get('absent_images') or ():
            self.absent[tuple(k)] += v

    @classmethod
    def from_csv(cls, path, runs):
        """The re-judging's OUT / NOTE / ROUTES / ROWS read back from its rejudged table (the omit product's
        deep_omission_rejudged.csv: production's rejudged.csv) - the combined view's build applies them (it cannot re-judge).
        The counts stay the omit summary's."""
        rej = cls()
        by_tag = {batch_tag(r): r for r in runs}
        with open(path, newline='') as f:
            for x in csv.DictReader(f):
                r = by_tag.get(x['batch'])
                if r is None:
                    refuse(f'G-K re-judging: {path} names batch {x["batch"]}, no run of the view')
                row = collections.OrderedDict((k, v) for k, v in x.items() if k != 'batch')
                for k in ('qualifies_before', 'qualifies_after', 'n_members', 'n_removed'):
                    row[k] = int(row[k])
                rej.add(r, dict(rejudge=[[row['identity_jname'], row['tier'], list(row.items())]], counts={}, n_removed=[],
                                g3d_neighbours=0))
        return rej

    def summary(self, per_tier_rows_dropped):
        """The view summary's 'rejudge' (snapshot_combined_view.py dbc40a6), in its order."""
        return collections.OrderedDict([
            ('served_sources_reproduced', self.N['served sources']), ('touched', self.N['served (source, tier) touched']),
            ('still_qualify', self.N['still qualifies']), ('rejudged_out', len(self.OUT)),
            ('every_member_removed', self.N['every member removed']), ('per_tier_rows_dropped', per_tier_rows_dropped),
            ('removed_detections_in_records', dict(self.RM)), ('g3d_removals_sharing_a_group_with_card2', self.G3D)])


REJUDGED_FILE = 'deep_omission_rejudged.csv'    # the omit product's copy of the view's rejudged.csv (production's columns)


def per_tier_snapshot(runs, pending, dropped, om, why, reader, card2, rej):
    """The per-tier consequences (the combined view's step 4b and section 0b, dbc40a6): per Stokes, tier and batch, a source of a
    dropped identity is dropped; a (batch identity, tier) re-judged out loses its source rows ('source re-judged') and member
    rows ('member row re-judged'); a source left with no detection after the omission (per image) and Card 2 is dropped; a
    PARTIAL one whose best detection - the one of its own (ts, subband) - went is re-pointed to its kept detection of largest
    |S/N| (best_cand_id '<identity>_<lst>_<date>_<band>', fit_verdict, the note; the refactor port review's defect RP fixed),
    and one still qualifying after re-judging gets rejudged_routes; a member row of a dropped identity or an omitted key
    (per image) is dropped, one of a frame where every detection of its batch identity is a Card 2 one too ('card2 member
    row'); a panel of an omitted key (per image) is not linked, nor a 10-s Stokes V one of such a frame ('card2 panel').
    Returns (record rows, the kept product's per-tier rows - KEPT_COLS, each naming the served run and the catalogue file or
    page directory it acts on, the number of per-tier source rows re-judged out). G4 a source or member row of no served
    identity; G4c a kept source whose page holds no detection; G4d a partial source whose own (ts, subband) detection is
    not on its page (production refuses: the re-point could not name its best)."""
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
            for row in rows if (r, row['identity_id']) not in dropped and (r, row['identity_id'], T) not in rej.OUT]
    reader.prefetch(os.path.join(d, 'detections.csv') for d in srcs)
    rec, drops, listing = [], [], None
    n_rej_rows = 0
    for S, T, cdir, r, kind, band, rows in cats:
        tag = batch_tag(r)
        run = os.path.abspath(r)
        if kind == 'sources':
            catalog = f'{cdir}/{band}_sources_survivors.csv'
            for row in rows:
                J = row['identity_id']
                if (r, J) in dropped:
                    rec.append(dict(kind='source dropped', batch=tag, identity=J, source=row['source_id'],
                                    stokes=S, tier=T, band=band, note=IDENTITY_EMPTY_NOTE))
                    drops.append(dict(kind='source dropped', run=run, batch=tag, identity=J, catalog=catalog,
                                      source_id=row['source_id'], stokes=S, tier=T, note=IDENTITY_EMPTY_NOTE))
                    continue
                if (r, J, T) in rej.OUT:
                    rec.append(dict(kind='source re-judged', batch=tag, identity=J, source=row['source_id'], stokes=S, tier=T,
                                    band=band, note=rej.NOTE[(r, J, T)]))
                    drops.append(dict(kind='source re-judged', run=run, batch=tag, identity=J, catalog=catalog,
                                      source_id=row['source_id'], stokes=S, tier=T, note=rej.NOTE[(r, J, T)]))
                    n_rej_rows += 1
                    continue
                if J not in ids[r]:
                    refuse(f'G4: {tag} {cdir} source {row["source_id"]} of identity {J} is in no served identity')
                src = os.path.join(r, row['review_dir'])
                ad = reader.rows(os.path.join(src, 'detections.csv'), 'served_detections')
                if not ad:      # G4c (V_combview review F4): an incomplete input is never booked as a deep omission
                    refuse(f'G4c: {tag} {cdir} source {row["source_id"]}: its page {row["review_dir"]} holds no detection '
                           f'(detections.csv missing or empty) - an incomplete batch run; production would record it as "every '
                           f'detection omitted" by the deep rule')
                kd = [d for d in ad if not om.det(d) and not card2.det(d)]
                if not kd:
                    note = SOURCE_EMPTY_CARD2_NOTE if any(card2.det(d) for d in ad) else SOURCE_EMPTY_NOTE
                    rec.append(dict(kind='source dropped', batch=tag, identity=J, source=row['source_id'],
                                    stokes=S, tier=T, band=band, note=note))
                    drops.append(dict(kind='source dropped', run=run, batch=tag, identity=J, catalog=catalog,
                                      source_id=row['source_id'], stokes=S, tier=T, note=note))
                    continue
                if len(kd) < len(ad):
                    bd = [d for d in ad if d['ts'] == row.get('ts') and d['subband'] == row.get('subband')]
                    if not bd:
                        refuse(f'G4d: {tag} {cdir} source {row["source_id"]} names a best detection (ts {row.get("ts")}, subband '
                               f'{row.get("subband")}) absent from its detections.csv')
                    kdk = {dkey(d) for d in kd}
                    fields = collections.OrderedDict()
                    note = ''
                    top = None
                    if all(dkey(d) not in kdk for d in bd):
                        top = max(kd, key=lambda d: abs(float(d['snr'])))
                        note = repoint_record_note(row, top)
                        rec.append(dict(kind='source partial', batch=tag, identity=J, source=row['source_id'],
                                        stokes=S, tier=T, band=band, note=note))
                        fields.update([('best_cand_id', '%s_%s_%s_%s' % (J, top['lst'], top['date'], band)),
                                       ('best_snr', top['snr']), ('ts', top['ts']), ('subband', top['subband']),
                                       ('fit_verdict', top.get('fit_verdict', '')), ('peak_jy_best', ''),
                                       ('deep_omission', REPOINT_NOTE)])
                    if (r, J, T) in rej.ROUTES:
                        fields['rejudged_routes'] = rej.ROUTES[(r, J, T)]
                    if fields:
                        drops.append(dict(kind='source re-pointed' if top is not None else 'source routes', run=run, batch=tag,
                                          identity=J, catalog=catalog, source_id=row['source_id'],
                                          cand_id=top['cand_id'] if top is not None else '', snr=top['snr'] if top is not None else '',
                                          stokes=S, tier=T, note=note, fields=json.dumps(fields)))
                if listing is None:                   # every kept source's directory, listed ahead (once)
                    listing = reader.listdirs(srcs)
                names = listing.get(src)
                for fn in names if names is not None else sorted(os.listdir(src)):
                    if fn in ('detections.csv', 'astrometry.png'):
                        continue
                    pm = SNAP_PANEL.match(fn)
                    if not pm:
                        continue
                    k = (pm.group(1), pm.group(3), pm.group(2), S, T)
                    if om.key(k, panel_image(pm.group(4))):
                        rec.append(dict(kind='panel', batch=tag, source=row['source_id'], lst=k[0], band=k[1], night=k[2],
                                        stokes=S, tier=T, note=fn, why=why[k]))
                        drops.append(dict(kind='panel dropped', run=run, batch=tag, identity=J, catalog=row['review_dir'],
                                          source_id=row['source_id'], lst=k[0], band=k[1], night=k[2], stokes=S, tier=T, note=fn))
                        continue
                    if S == 'V' and T == '10s' and card2.JC_C2:
                        cand = '%s_%s_%s_%s_%s_%s_%s' % (pm.group(1), pm.group(2), pm.group(3), S, T, pm.group(4)[:8],
                                                         pm.group(4)[8:])
                        if card2.jc_card2(r, J, cand):
                            rec.append(dict(kind='card2 panel', batch=tag, source=row['source_id'], stokes=S, tier=T, band=k[1],
                                            note=fn, why=CARD2_WHY))
                            drops.append(dict(kind='card2 panel dropped', run=run, batch=tag, identity=J,
                                              catalog=row['review_dir'], source_id=row['source_id'], stokes=S, tier=T, note=fn))
            continue
        catalog = f'{cdir}/{band}_source_members.csv'
        for row in rows:
            mm = MEMBER_SOURCE.match(row['source_id'])
            ck = SNAP_CID.match(row.get('cand_id', ''))
            k = (ck.group(1), ck.group(3), ck.group(2), ck.group(4), ck.group(5)) if ck else None
            if k is not None and om.key(k, row['cand_id'][ck.end():]):          # its own key omitted (per image)
                rec.append(dict(kind='member row', batch=tag, source=row['source_id'], cand_id=row.get('cand_id', ''),
                                lst=k[0], band=k[1], night=k[2], stokes=S, tier=T, why=why[k]))
                drops.append(dict(kind='member row dropped', run=run, batch=tag, identity=mm.group(1) if mm else '',
                                  catalog=catalog, source_id=row['source_id'], cand_id=row.get('cand_id', ''), lst=k[0],
                                  band=k[1], night=k[2], stokes=S, tier=T))
                continue
            if mm and (r, mm.group(1)) in dropped:     # a consequence: its identity has no detection left
                rec.append(dict(kind='member row', batch=tag, source=row['source_id'], cand_id=row.get('cand_id', ''),
                                band=band, stokes=S, tier=T, note=IDENTITY_EMPTY_NOTE))
                drops.append(dict(kind='member row dropped', run=run, batch=tag, identity=mm.group(1), catalog=catalog,
                                  source_id=row['source_id'], cand_id=row.get('cand_id', ''), stokes=S, tier=T,
                                  note=IDENTITY_EMPTY_NOTE))
                continue
            if mm and (r, mm.group(1), T) in rej.OUT:   # its (batch identity, tier) re-judged out
                rec.append(dict(kind='member row re-judged', batch=tag, source=row['source_id'], cand_id=row.get('cand_id', ''),
                                stokes=S, tier=T, band=band))
                drops.append(dict(kind='member row re-judged', run=run, batch=tag, identity=mm.group(1), catalog=catalog,
                                  source_id=row['source_id'], cand_id=row.get('cand_id', ''), stokes=S, tier=T))
                continue
            if mm and card2.jc_card2(r, mm.group(1), row.get('cand_id', '')):    # every detection of its frame a Card 2 one
                rec.append(dict(kind='card2 member row', batch=tag, source=row['source_id'], cand_id=row.get('cand_id', ''),
                                stokes=S, tier=T, band=band, why=CARD2_WHY))
                drops.append(dict(kind='card2 member row dropped', run=run, batch=tag, identity=mm.group(1), catalog=catalog,
                                  source_id=row['source_id'], cand_id=row.get('cand_id', ''), stokes=S, tier=T))
                continue
            if mm is None or mm.group(1) not in ids[r]:
                refuse(f'G4: {tag} {cdir} member source {row["source_id"]} is in no served identity')
    complete_batch_runs(runs, cats, reader)   # G4b (V_combview review F4), after G4 / G4c: a stray source names its own gate
    return rec, drops, n_rej_rows


SERVED_TIERS = 'served_tiers.csv'          # a batch run's per-tier serving table (the chain's phase3_snapshot_tierserve)


def served_tiers(run, reader):
    """{tier: {identity}} that the batch run's served_tiers.csv serves in each tier (chain_outcome 'served', qualifies 1), or
    None without the file."""
    t = reader.text(os.path.join(run, SERVED_TIERS), 'served_catalogs', missing_ok=True)
    if t is None:
        return None
    out = collections.defaultdict(set)
    for r in csv.DictReader(io.StringIO(t, newline='')):
        if r.get('chain_outcome') == 'served' and str(r.get('qualifies')) == '1':
            out[r['tier']].add(r['identity_jname'])
    return out


def complete_batch_runs(runs, cats, reader):
    """G4b (V_combview review F4): each batch run's per-tier survivors hold exactly the identities its served_tiers.csv serves in
    that tier - so a per-tier catalogue lost or cut short (an interrupted copy, a re-serve that stopped halfway) refuses instead of
    a source vanishing from the view unrecorded. Held exactly on all 36 batches of Run_SNAPCOMB_20261102 (2026-11-04).
    ``cats``: per_tier_snapshot's (S, T, cdir, run, kind, band, rows)."""
    for r in runs:
        st = served_tiers(r, reader)
        if st is None:
            refuse(f'G4b: {batch_tag(r)} has no {SERVED_TIERS}: its per-tier catalogues cannot be shown complete')
        for T in _c().SNAPSHOT_TIERS:
            have = {row['identity_id'] for S, T2, cdir, r2, kind, band, rows in cats if r2 == r and T2 == T and kind == 'sources'
                    for row in rows}
            want = st.get(T, set())
            if have != want:
                refuse(f'G4b: {batch_tag(r)} tier {T}: its per-tier survivors hold {len(have - want)} identities {SERVED_TIERS} does '
                       f'not serve (e.g. {sorted(have - want)[:2]}) and lack {len(want - have)} it serves (e.g. '
                       f'{sorted(want - have)[:2]}) - an incomplete or inconsistent batch run')


SERVED_COLS = ['lst', 'night', 'band', 'stokes', 'tier', 'n']


def write_served(path, served):
    """deep_omission_served.csv: the served counts the decisions were made with ({(lst, night, band, stokes, tier): n}, the
    rate test's input) - so the build can re-decide from the current refusal record and fleet records and show the decisions
    current (V_combview review F5), and the content test can tie them to the kept and omitted detections."""
    with open(path + '.tmp', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(SERVED_COLS)
        for k in sorted(served):
            w.writerow(list(k) + [served[k]])
    os.replace(path + '.tmp', path)


def read_served(path):
    """{(lst, night, band, stokes, tier): n} of a deep_omission_served.csv."""
    with open(path, newline='') as f:
        return {(r['lst'], r['night'], r['band'], r['stokes'], r['tier']): int(r['n']) for r in csv.DictReader(f)}


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


def served_digests(pending, om, card2):
    """{'served', 'kept', 'omitted', 'omitted_cid', 'card2'} digests (texts) of the served identities ``pending``: every
    detection, the kept ones, the omitted ones (per image; detection_item) and their (batch, identity, cand_id), the ones Card 2
    alone removes (dbc40a6) - kept + omitted + card2 = served."""
    D = {k: MultisetDigest() for k in ('served', 'kept', 'omitted', 'omitted_cid', 'card2')}
    for r, J, _grs, dets in pending:
        tag = batch_tag(r)
        for d in dets:
            item = detection_item(tag, J, dkey(d))
            D['served'].add(item)
            if om.det(d):
                D['omitted'].add(item)
                D['omitted_cid'].add('\x1f'.join((tag, J, d['cand_id'])))
            elif card2.det(d):
                D['card2'].add(item)
            else:
                D['kept'].add(item)
    if D['kept'] + D['omitted'] + D['card2'] != D['served']:      # additive by construction; a broken digest refuses here
        refuse('G5: the kept, omitted and Card 2 digests do not add up to the served digest')
    return {k: v.text() for k, v in D.items()}


# =====================================================================================================================
# DROP - 10-minute: the detections the in-place merge drop removes (sublst_deep_omission_drop.filter_rows, member rows)
# =====================================================================================================================

def tenmin_detections(lsts, OM, why, recs, reader):
    """Record rows of the in-place merge drop (sublst_deep_omission_drop.py, 35aad1a): every member row of the LST runs whose
    (lst, band, night, stokes) is OMIT - the 'detection' rows filter_rows writes (source_catalogs_<S>_<sb|wb>/*_source_members.csv;
    Stokes = the table's) - and every PARTIAL survivor the drop's RE-JUDGING removes (edf3263; tenmin_rejudge) - its 'source
    re-judged' rows; a re-judged-out source's members get no 'detection' row (production's filter_rows drops them with the
    source before its omission test - fixed in the V3 round: the V2 port listed its omitted members too). Returns (rows,
    {(lst, S, T, source_id, band): reason} of the re-judged-out sources, the member rows of those sources - production's
    page line counts them with the omitted detections)."""
    rec, rej, gate, n_rjm = [], {}, None, 0
    keys4 = {k[:4]: k for k in OM}
    for L in lsts:
        R = recs.run_dir(L)
        left, gone = collections.defaultdict(list), collections.defaultdict(list)
        dets = []
        for S in ('I', 'V'):
            for T in ('sb', 'wb'):
                for p in sorted(glob.glob(os.path.join(R, f'source_catalogs_{S}_{T}', '*_source_members.csv'))):
                    for r in reader.rows(p, 'tenmin_members', missing_ok=False):
                        m = TENMIN_CID.match(r['cand_id'] or '')
                        k = (m.group(2), m.group(4), m.group(3), S) if m else None
                        (gone if k in keys4 else left)[(S, T, r['source_id'], r['band'])].append(r)
                        if k in keys4:
                            dets.append(dict(kind='detection', lst=L, stokes=S, view=T, source_id=r['source_id'], band=r['band'],
                                             cand_id=r['cand_id'], night=k[2], why=why[keys4[k]], detail=''))
        # the members of a source re-judged out leave with it and are not recorded one by one: production's filter_rows skips
        # them before its omission test (sublst_deep_omission_drop.py, REJ before 'omitted'; the 'members_rejudged' counts)
        n_rec = len(rec)
        rec += dets
        # RE-JUDGING of every partial survivor (sublst_deep_omission_drop._ctx, edf3263), once per LST: on the members it keeps
        if not any(gone.values()):
            continue
        master = {}
        mp = os.path.join(R, 'master_catalog.csv')
        if os.path.exists(mp):
            for r in reader.rows(mp, 'tenmin_served', missing_ok=False):
                master[r['cand_id']] = r
        if gate is None:
            gate = tenmin_size_gate()[0]
        for S in ('I', 'V'):
            for T in ('sb', 'wb'):
                for p in sorted(glob.glob(os.path.join(R, f'source_catalogs_{S}_{T}', '*_sources_survivors.csv'))):
                    for r in reader.rows(p, 'tenmin_survivors', missing_ok=False):
                        key = (S, T, r['source_id'], r['band'])
                        if key in gone and left.get(key):
                            reason = tenmin_rejudge(L, key, r, left[key], master, gate)
                            if reason:
                                rej[(L,) + key] = reason
                                rec.append(dict(kind='source re-judged', lst=L, stokes=S, view=T, source_id=r['source_id'],
                                                band=r['band'], cand_id='', night='',
                                                why='re-judged on the members it keeps: %s' % reason,
                                                detail='identity %s' % r.get('identity_id', '')))
        if any(k[0] == L for k in rej):
            rec[n_rec:] = [x for x in rec[n_rec:] if x['kind'] != 'detection'
                           or (L, x['stokes'], x['view'], x['source_id'], x['band']) not in rej]
            # production's page line counts every member row of a re-judged-out source with the omitted detections (its
            # 'members_rejudged_<S>_<T>' counts, kept and omitted members alike, sublst_deep_omission_drop.finish)
            n_rjm += sum(len(left.get(k[1:], ())) + len(gone.get(k[1:], ())) for k in rej if k[0] == L)
    return rec, rej, n_rjm


def _tsnr(r, col='det_snr'):
    try:
        return abs(float(r.get(col) or 'nan'))
    except ValueError:
        return 0.0


def tenmin_size_gate():
    """(gate in beams, the merge's environment, the rule): the 7-8 sigma size/axis gate the 10-minute RE-JUDGING applies to a
    partial survivor's remaining single - the catalogue builder's own under the configured 10-minute size rule
    (search_sublst.SIZE_RULE; lib/search/size_rule.catalog_env), so the re-judging judges a remaining single as the builder of
    the same catalogues judged it and the tiers judge one way (the refactor chat, the V3 round 2026-11-04). Under 'envelope'
    (the refactor's default, R-GREGG-20261022-ALL-APPROVED-SIZE-GATE-READMISSION-SUBAGENTS) the builder's fixed gate is
    replaced by the rule: every detection a catalogue holds was kept by the rule's SNR-calibrated size and axis test at the
    search, so the remaining single carries the rule's verdict (kept) and no fixed gate acts - (inf, {P3_SIZE_GATE_7_8: inf},
    'envelope'), the environment the merge passes to its in-place drop (sublst_deep_omission_drop.SIZE_GATE_7_8 reads
    P3_SIZE_GATE_7_8). Under 'production' (the parity file's search_sublst.SIZE_RULE) production's fixed
    catalog_associate.SIZE_GATE_7_8 (1.7 beams) and nothing set - the drop's own default (P3_SIZE_GATE_7_8 unset, cleared
    from any inherited environment by core.legacy.run)."""
    from exopipe.phase3.lib.search import size_rule as SZ
    rule = SZ.rule_value(SZ.configured('10min')[0], 'search_sublst.SIZE_RULE')
    env = SZ.catalog_env({'rule': rule})
    v = env.get('P3_SIZE_GATE_7_8')
    return (float(v) if v is not None else float(_cfg().catalog_associate.SIZE_GATE_7_8)), env, rule


def tenmin_rule_check(lsts):
    """G12 (the V3 round): the re-judging applies the builder's 7-8 sigma gate of the CONFIGURED 10-minute size rule
    (tenmin_size_gate), so every search job of the LSTs must have been searched under that rule - under 'envelope' the
    remaining single's verdict is the rule's at the search (a job searched under production's fixed rule gave its detections no
    verdict of the rule: 2026-11-04, production's 08h J0645+4024, a 7.07-sigma single of 2.32 beams, would be kept unjudged);
    under 'production' the fixed gate presumes catalogues the rule never judged. Each LST's jobs (catalog_sublst.SEARCH_DIR/
    <lst>_<band>_<S>/transients/Run_SLST<hh><arm>/size_rule.json; none = production) must hold ONE rule, the configured one
    (lib/search/size_rule.one_rule, as the catalogue's S-G7 requires before it builds); under 'envelope' an LST without any job
    refuses too (the verdict cannot be shown). Returns {lst: rule}."""
    from exopipe.phase3.lib.search import size_rule as SZ
    cs = _cfg().catalog_sublst
    search = cs.SEARCH_DIR.format(root=_cfg().paths.SUBLST_ROOT, arm=cs.ARM)
    try:
        want = SZ.configured_record('10min')
    except SZ.RuleRefusal as e:
        refuse(f'G12: the configured 10-minute size rule cannot be read: {e}')
    out = {}
    for L in lsts:
        runs = [os.path.join(j, 'transients', 'Run_SLST%s%s' % (L[:-1], cs.ARM))
                for j in sorted(glob.glob(os.path.join(search, '%s_*_[IV]' % L)))]
        runs = [r for r in runs if os.path.isdir(r)]
        if not runs and want['rule'] == 'envelope':
            refuse(f'G12: no 10-minute search job of {L} under {search} - the size rule its catalogue was judged by cannot be '
                   f'shown, so its remaining singles cannot be re-judged with it')
        try:
            out[L] = SZ.one_rule(runs, want=want, what=f'the {L} 10-minute search jobs in {search}')['rule']
        except SZ.RuleRefusal as e:
            refuse(f'G12 the re-judging\'s size rule: {e}')
    return out


def tenmin_rejudge(L, key, r, lf, master, gate=None):
    """'' when a PARTIAL 10-minute survivor still survives on the members it keeps, else the reason - the merge drop's re-judging
    (sublst_deep_omission_drop._rejudge, exopipe_ops edf3263; Gregg 2026-11-04, R-GREGG-20261104-DEEP-OMISSION-SHORT-TIERS-
    ANSWERS) with build_source_catalogs_v3's own survivor rule: |best SNR| >= catalog_associate.SURVIVOR_SNR (7), or a
    promotion that still holds (RPT / REP / XREP need >= 2 members, void with one; E:<sample>, PRIORITY, MB hold while the
    best is unchanged), then the 7-8 sigma size/axis gate ``gate`` (tenmin_size_gate: the builder's own under the configured
    size rule - none under 'envelope', production's 1.7 beams under 'production'; the merge passes the same value to its drop)
    for an unpromoted single without a cross-match. G10 refuses where survival rests on a statistic the members left change and
    that is not re-derived (a repeat or cross-match E, a multi-band test) - production refuses the build there."""
    ca = _cfg().catalog_associate
    THR, GATE = float(ca.SURVIVOR_SNR), (tenmin_size_gate()[0] if gate is None else float(gate))
    b = max(lf, key=_tsnr)
    a, n = _tsnr(b), len(lf)
    labels = [x for x in (r.get('promotion') or '').split('+') if x]
    void = [x for x in labels if x in ('RPT', 'REP', 'XREP') and n < 2]
    unc = [x for x in labels if x not in void]
    if a < THR:
        if unc:
            refuse('G10 re-judging: %s %s %s %s keeps %d member(s) at |SNR| %.2f < %g and survives only by %s, which the members '
                   'left change (not re-derived here)' % (L, key[0], key[2], key[3], n, a, THR, '+'.join(unc)))
        return 'no route on %d remaining member(s): |SNR| %.2f < %g, promotion %s void' % (n, a, THR, r.get('promotion') or 'none')
    if n == 1 and not (r.get('xm_best') or '').strip() and 7.0 <= a < 8.0:
        mr = master.get(b['cand_id'])
        if mr is None:
            refuse('G10 re-judging: %s %s %s %s best %s has no master_catalog row (size gate)' % (L, key[0], key[2], key[3],
                                                                                              b['cand_id']))
        try:
            sz, ax = float(mr.get('size_beams') or 'nan'), float(mr.get('axis_norm') or 'nan')
        except ValueError:
            sz, ax = float('nan'), float('nan')
        if (sz == sz and sz > GATE) or (ax == ax and ax > GATE):
            if unc:
                refuse('G10 re-judging: %s %s %s %s: the 7-8 sigma size/axis gate removes its remaining single unless %s still '
                       'holds (not re-derived here)' % (L, key[0], key[2], key[3], '+'.join(unc)))
            return ('7-8 sigma size/axis gate on its remaining single (|SNR| %.2f, size %.2f, axis %.2f beams; > %g; unpromoted, no '
                    'cross-match)' % (a, sz, ax, GATE))
    return ''


def tenmin_rejudge_gate(rej):
    """G11 (exopipe_ops 35aad1a): a 10-minute stage whose re-judging removes a source refuses without Gregg's go on the count
    recorded (catalog_omission.TENMIN_REJUDGE_APPROVED non-empty) - the merge's drop refuses the same build in place."""
    ap = _c().TENMIN_REJUDGE_APPROVED
    if rej and not (ap and os.path.isfile(ap) and os.path.getsize(ap) > 0):
        k = sorted(rej)[0]
        refuse(f'G11: re-judging removes {len(rej)} 10-minute source(s) (e.g. {k}) - Gregg\'s go on the count is required first '
               f'({ap})')


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
    if bad and mode == 'refuse':    # WAVE2 G6: ruled R-REFACTOR-20261104-ABSENT-WITH-SERVED-REFUSE (production keeps: parity file)
        refuse(f'G6: {len(bad)} deep-refused keys the tier has no record of hold served detections (e.g. {bad[:3]}) - the '
               f'exception test cannot be evaluated')
    return bad


def view_cells(runs, reader):
    """The cells the view's batches searched - the union of each run's chain_summary.json 'cells' (a list or a dict; the
    rate population of the decisions, snapshot_combined_view.py dbc40a6 CELLS_VIEW). G2b a run without them."""
    cells = set()
    for r in runs:
        t = reader.text(os.path.join(r, 'chain_summary.json'), 'chain_summary', missing_ok=True)
        cs = json.loads(t).get('cells') if t is not None else None
        if cs is None:
            refuse(f'G2b: {batch_tag(r)} has no chain_summary.json cells: the rate population (the cells its batch searched) '
                   f'cannot be read')
        cells.update(cs if isinstance(cs, list) else list(cs))
    return cells


def card2_vintage(runs, reader):
    """{batch tag: 'chain' | 'view'}: where each batch's Card 2 ran (V_combview review T7) - 'chain' when its chain_summary.json
    has the chain's own Card 2 counts (clv15 1279bd4: per frame after the per-frame fit, BEFORE the tie and the 3-D gate), else
    'view' (a batch chained before: the view applies Card 2 after its chain). A view mixing both applies Card 2 in two orders;
    recorded in the omit summary, not a gate (production records nothing)."""
    out = collections.OrderedDict()
    for r in runs:
        t = reader.text(os.path.join(r, 'chain_summary.json'), 'chain_summary', missing_ok=True)
        out[batch_tag(r)] = 'chain' if (t is not None and 'card2' in json.loads(t)) else 'view'
    return out


def rejudge_runs(runs, dec_path, proto, card2, work, u=None):
    """The re-judging of every batch run: lib/catalog/rejudge.py, one CHILD per run under catalog_omission.REJUDGE_PYTHON
    (production's interpreter; clv15's chain and tier-serving modules in place), at most catalog_omission.REJUDGE_PROCS at a
    time (production 8 in fork workers on lwacalim07's disk; the Lustre rule: 6 readers; separate processes, so no fork of a
    large parent - V_combview review T6), each with the decisions (per image), the fleet root and its run's Card 2 det_uids;
    job, result and log under ``work``. Returns Rejudged (in the runs' order). G9 a child that refused (a self-check: the chain
    not reproduced, served_tiers.csv not reproduced, the record incomplete) or failed."""
    from concurrent.futures import ThreadPoolExecutor
    from exopipe.phase3.core import legacy
    c = _c()
    for r in runs:                   # the inputs each child reads: refused here, before any child runs, by name
        if not os.path.isfile(os.path.join(r, SERVED_TIERS)):
            refuse(f'G4b: {batch_tag(r)} has no {SERVED_TIERS}: its per-tier catalogues cannot be shown complete (nor its sources '
                   f're-judged)')
        miss = [f for f in ('detection_record.csv.gz', 'chain_summary.json') if not os.path.isfile(os.path.join(r, f))]
        if miss:
            refuse(f'G9: {batch_tag(r)} has no {", ".join(miss)}: its served sources cannot be re-judged')
    os.makedirs(work, exist_ok=True)
    names = dict(noise=dict(c.SNAPSHOT_NOISE), members=c.SNAPSHOT_MEMBERS, admission=c.SNAPSHOT_ADMISSION)
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rejudge.py')
    jobs = []
    for r in runs:
        tag = batch_tag(r)
        res = os.path.join(work, f'{tag}.result.json')
        if os.path.exists(res):
            os.remove(res)
        job = dict(run=os.path.abspath(r), decisions=os.path.abspath(dec_path), proto=os.path.abspath(proto), names=names,
                   card2=sorted(card2.uids[r]), clv15=_legacy_clv15(), samples_dir=c.REJUDGE_SAMPLES_DIR or _legacy_clv15(),
                   deep_omission_py=os.path.abspath(__file__), result=res)
        jp = os.path.join(work, f'{tag}.job.json')
        with open(jp, 'w') as f:
            json.dump(job, f, indent=1)
        jobs.append((r, jp, res, os.path.join(work, f'{tag}.log')))

    def one(j):
        r, jp, res, log = j
        return legacy.run(script, [jp], python=c.REJUDGE_PYTHON, log=log, unit=u, timeout=c.REJUDGE_TIMEOUT_S,
                          refusal_codes=(2,))
    with ThreadPoolExecutor(max(1, min(int(c.REJUDGE_PROCS), len(jobs) or 1))) as ex:
        results = list(ex.map(one, jobs))
    rej = Rejudged()
    for (r, jp, res, log), lr in zip(jobs, results):
        if not lr.ok or not os.path.exists(res):
            refuse(f'G9: the re-judging of {batch_tag(r)} {"refused" if lr.refused else "failed"}: {lr.reason} (log {log})')
        with open(res) as f:
            x = json.load(f)
        if os.path.abspath(x['run']) != os.path.abspath(r):
            refuse(f'G9: re-judging result {res} is of {x["run"]}, not {r}')
        rej.add(r, x)
    return rej


def run_snapshot(runs, out, rdir=None, proto=None, u=None, label='omit'):
    """The 10-s / 1-minute stage on the served batch runs ``runs``: Card 2 (each batch's card2_vleak.csv), the decisions (the
    rate population = the view's searched cells), the re-judging (one child per batch), the drop, its record, the re-judging's
    table and the kept product (what the snapshot tier's cumulative step applies) into ``out``. Returns the summary dict. Gates
    refuse (core.record.Refusal)."""
    c = _c()
    rdir = rdir or c.REFUSALS
    proto = proto or _cfg().paths.SNAPSHOT_FLEET_OUT
    _record_gate(rdir)
    reader = Reader()
    t0 = time.time()
    pending = served_identities(runs, reader)
    card2 = Card2(runs, reader, ratio=c.CARD2_RATIO, name=c.SNAPSHOT_CARD2)
    c2_match, c2_total, c2_miss = card2_coverage(card2, pending)
    if c2_miss:          # V_combview review T3: a served Card 2 row matching no served detection would be applied silently not
        refuse(f'G8 Card 2: {len(c2_miss)} of {c2_total} served Card 2 rows of card2_vleak.csv match no served detection of '
               f'their batch by (cand_id, ra_peak, dec_peak) at 5 decimals (e.g. {c2_miss[:2]}) - a key mismatch, never applied '
               f'silently')
    served, n_all = served_counts(pending, card2)
    cells = view_cells(runs, reader)
    lsts = sorted({k[0] for k in served})
    dec = decide_snapshot(lsts, served, rdir=rdir, proto=proto, reader=reader, cells=cells)
    absent = absent_gate(dec, lambda k: served.get((k[0], k[2], k[1], k[3], k[4]), 0))
    why = why_of(dec)
    recs = SnapshotRecords(proto, reader)
    om = Omission(dec, recs)
    # the re-judging's jobs, results and logs: beside the product while it runs (a refused stage writes no product; the logs stay
    # there for the refusal's reader), moved into it as <out>/_rejudge when the product is written
    work = out.rstrip('/') + '.rejudge'
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    dec_work = os.path.join(work, 'deep_omission.csv')
    write_decisions(dec_work, dec, TIER_SNAPSHOT)
    rej = rejudge_runs(runs, dec_work, proto, card2, work, u=u)
    kept, dropped, rec = apply_snapshot(pending, om, why, card2)
    digests = served_digests(pending, om, card2)       # N2: the served set's fingerprint, checked by the kept product's test
    prec, drops, n_rej_rows = per_tier_snapshot(runs, pending, dropped, om, why, reader, card2, rej)
    rec += prec
    absent_images = om.absent + rej.absent
    if absent_images:    # V_combview review T2: production omits such a detection (fail closed) - a key-format mismatch
        refuse(f'G7: {sum(absent_images.values())} lookups of a per-image wideband key found no image in the cell\'s member table '
               f'(e.g. {sorted(absent_images)[:2]}) - the image ids of the detections and of the member table differ; production '
               f'would omit every one of them silently')
    os.makedirs(out, exist_ok=True)
    shutil.rmtree(os.path.join(out, '_rejudge'), ignore_errors=True)
    os.rename(work, os.path.join(out, '_rejudge'))
    write_decisions(os.path.join(out, 'deep_omission.csv'), dec, TIER_SNAPSHOT)
    write_served(os.path.join(out, 'deep_omission_served.csv'), served)      # F5: what the decisions were made with
    write_record(os.path.join(out, 'deep_omission_dropped.csv'), rec, RECORD_COLS)
    write_rejudged(os.path.join(out, REJUDGED_FILE), rej.ROWS)
    kept_kinds = collections.Counter()              # Q review F2: the drop, written for the cumulative step that applies it

    def counted(rows):
        for x in rows:
            kept_kinds[x['kind']] += 1
            yield x
    write_record(os.path.join(out, 'deep_omission_kept.csv'), counted(itertools.chain(kept_rows(kept), drops)), KEPT_COLS)
    kinds = collections.Counter(x['kind'] for x in rec)
    n_om, n_c2 = kinds.get('detection', 0), kinds.get('card2 detection', 0)
    OMALL = omit_keys_all(dec)
    summ = dict(tier=TIER_SNAPSHOT, label=label, runs=[os.path.abspath(r) for r in runs], batches=[batch_tag(r) for r in runs],
                refusals=os.path.abspath(rdir), proto=os.path.abspath(proto), lsts=lsts, served=n_all, omitted=n_om, card2=n_c2,
                card2_served=sum(1 for p in pending for d in p[3] if card2.det(d)),
                kept=n_all - n_om - n_c2, decisions=len(dec), omit_keys=len(om.OM), omit_wb_image_keys=len(om.OMWB),
                omit_keys_by_tier={t: sum(1 for k in OMALL if k[4] == t) for t in c.SNAPSHOT_TIERS},
                verdicts=dict(collections.Counter(r['verdict'] for r in dec)), record=dict(sorted(kinds.items())),
                kept_product=dict(sorted(kept_kinds.items())),
                identities=len(pending), identities_kept=len(kept), absent_with_served=len(absent), digests=digests,
                cells=sorted(cells),
                card2_counts=collections.OrderedDict([
                    ('ratio', c.CARD2_RATIO), ('removed_10s_v_records', card2.n['removed_10s']),
                    ('removed_10s_v_served', card2.n['removed_10s_served']), ('not_evaluated_10s_v', card2.n['not_evaluated_10s']),
                    ('removed_from_view', n_c2), ('mixed_identity_frames_kept', card2.n_mixed()),
                    ('at_threshold_within_rounding', card2.n['at_threshold_within_rounding']),
                    ('served_rows_matched', c2_match)]),
                rejudge=rej.summary(n_rej_rows), rejudge_counts=dict(rej.N),
                rejudge_inputs=rejudge_inputs(runs), card2_vintage=card2_vintage(runs, reader),
                absent_images=sum(absent_images.values()), seconds=round(time.time() - t0, 1))
    _write_summary(out, summ, n_om, 'served detections %d, kept %d; decisions %d (%d OMIT keys + %d wideband per image); %s' % (
        n_all, n_all - n_om - n_c2, len(dec), len(om.OM), len(om.OMWB), ', '.join('%s %d' % kv for kv in sorted(kinds.items()))))
    if u is not None:
        reader.record(u, {'refusals': rdir, 'snapshot_noise': proto, 'snapshot_members': proto, 'snapshot_admission': proto,
                          'served_catalogs': os.path.commonpath([os.path.abspath(r) for r in runs]),
                          'served_detections': os.path.commonpath([os.path.abspath(r) for r in runs]),
                          'card2': os.path.commonpath([os.path.abspath(r) for r in runs]),
                          'chain_summary': os.path.commonpath([os.path.abspath(r) for r in runs])})
        for r in runs:
            for f in ('detection_record.csv.gz', 'served_tiers.csv'):
                u.add_input(os.path.join(r, f))
    return summ


REJUDGE_INPUT_FILES = ('detection_record.csv.gz', 'chain_summary.json', 'served_tiers.csv', 'card2_vleak.csv')


def rejudge_inputs(runs):
    """{batch tag: {file: [size, mtime_ns]}} of each run's re-judging and Card 2 inputs - recorded in the omit summary, so the
    cumulative step refuses an omit product whose re-judging inputs changed since (V_combview review T5 / F5: the build cannot
    re-derive the re-judging)."""
    out = collections.OrderedDict()
    for r in runs:
        d = collections.OrderedDict()
        for f in REJUDGE_INPUT_FILES:
            try:
                st = os.stat(os.path.join(r, f))
                d[f] = [st.st_size, st.st_mtime_ns]
            except OSError:
                d[f] = None
        out[batch_tag(r)] = d
    return out


def write_rejudged(path, rows):
    """rejudged.csv in production's format (snapshot_combined_view.py dbc40a6: write(path, list(REJ_ROWS[0].keys()) if REJ_ROWS
    else ['batch'], REJ_ROWS) - csv.DictWriter, extrasaction 'ignore')."""
    with open(path + '.tmp', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ['batch'], extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    os.replace(path + '.tmp', path)


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
    rules = tenmin_rule_check(lsts)                     # G12: the LSTs' jobs searched under the configured size rule
    rec, rej, n_rjm = tenmin_detections(lsts, OM, why, recs, reader)
    tenmin_rejudge_gate(rej)
    write_record(os.path.join(out, 'deep_omission_dropped.csv'), rec, TENMIN_RECORD_COLS)
    n_det = sum(1 for x in rec if x['kind'] == 'detection')
    g78, _env, g_rule = tenmin_size_gate()
    summ = dict(tier=TIER_TENMIN, label=label, lsts=list(lsts), refusals=os.path.abspath(rdir), omitted=n_det,
                rejudged=len(rej), rejudged_members=n_rjm, rejudge_size_gate=dict(rule=g_rule, gate_beams=str(g78),
                                                                                  lst_rules=rules),
                decisions=len(rows), omit_keys=len(OM),
                verdicts=dict(collections.Counter(r['verdict'] for r in rows)),
                absent_with_served=len(absent), window_rms=dict(sorted(recs.n.items())), rms_cache_seed=seed_dir,
                seconds=round(time.time() - t0, 1))
    _write_summary(out, summ, n_det + n_rjm, 'decision table %s; %d (lst, band, night, Stokes) omitted; %d sources re-judged out'
                   ' (%d member rows)' % (os.path.join(out, 'deep_omission.csv'), len(OM), len(rej), n_rjm))
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
            u.add_output(os.path.join(out, 'deep_omission_served.csv'), 'omission_served', tier=tier)
            u.add_output(os.path.join(out, REJUDGED_FILE), 'omission_rejudged', tier=tier)
            u.add_output(os.path.join(out, 'deep_omission_kept.csv'), 'omission_kept', tier=tier, decisions=dec)
    return None
