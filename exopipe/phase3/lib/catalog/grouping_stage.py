"""The catalog step's GROUPING (refactor item F.3): design B (lib/catalog/grouping.py) wired into `exopipe3 catalog` before
every multiplicity-dependent gate, with ONE multiples rule on every tier.

RULINGS
  Gregg 2026-10-26, R-GREGG-20261026-GROUPING-DESIGN-B-APPROVED-IDENTITY-097-MULTIPLES-085-DEEPV-IDENTITY-ONLY-PREVIEW-THEN-REFACTOR
  (verbatim: "All three grouping decisions are recommended according to your recommendations."); the decision page he approved:
  design B "runs once on all detections (all bands, nights and LSTs), before every gate that depends on multiplicity";
  "Identities (one row per source on the pages) use the strict level, P >= 0.97"; "The multiples level decides which detections
  count as repeats and skip the singles gates"; "Multiples skip three gates: the sigma-stack singleton gate, the 7-8 sigma size
  gate for unpromoted singles, and the 7 sigma floor for unpromoted sources. Singles keep every gate."; "Your labels. They carry
  to the new identities by member detections, through the existing carry mechanism. Conflicts are flagged, never resolved
  silently."
  Gregg 2026-10-27, R-GREGG-20261027-HUB-BOTH-LISTENERS-MULTIPLES-CHAIN-FORCED-EVERY-TIER, verbatim: "Which ever approach is taken
  (e.g. dropping <7-sigma is fine), let's be consistent." - ONE multiples chain on the 10-minute and deep tiers.

CONFIG (catalog_grouping; production's values in parity/production_20261013.cfg, the code sites marked WAVE2 GROUPING)
  GROUPING        'design-b' | 'production'      MULTIPLES_RULE  'one-rule' | 'production'   (the two must agree: mode())
  REPEAT_RUNGS    'multiples-level' | 'production' (under the one rule: whether the builder's repeat rungs RPT / REP / XREP may
                  promote a single-window identity; ruled reading of the decision page: no - the multiples level decides repeats)

WHERE IT RUNS
  deep       stage 'group' of the deep chain (after 'select'; one unit '<label>.group.<S>' per Stokes): every detection of the
             fitted conformant catalogue (the select stage's input master_catalog.csv) of that Stokes, with the frames' beams
             (frame_beam_bmaj.csv); the state (stable IDs) at <run>/<STATE_SUBDIR>/. Then 'associate' runs the builder with the
             grouping (lib/catalog/wraps/builder_grouping.py) and 'fork' (production's multiples_fork.py) ends no-input.
  10-minute  `exopipe3 catalog --tier sublst --stages group --group-omission O [--lsts ...]` (one unit '<label>.group'):
             ONE tier-wide grouping of every LST (the state is the tier's, per Stokes - not one per LST): every detection of
             the published LST runs (their master_catalog.csv + pertie_detections.csv, concatenated) AFTER the deep
             omission, per Stokes, into catalog_grouping.SUBLST_DIR (or --out; the state beside it); --group-input DIR groups
             DIR/master_catalog.csv + DIR/pertie_detections.csv instead (e.g. a frozen cumulative), with the same omission.
             THE OMISSION (production's order, sublst_cumulative.sh:99-107: the decision table (:99), the merge with its drop
             (:101, sublst_cumulative.py + sublst_deep_omission_drop.filter_rows), then the master identity (:103-107), whose
             test M1, sublst_master_identity.py:19, requires the grouping to hold exactly the build's master_catalog.csv
             after the drop): O is the 10-minute omit stage's output (`exopipe3 catalog --tier sublst --stages omit --lsts
             ... --out O`, its deep_omission.csv; a file path names the table itself); every row in an OMIT (LST, band,
             night, Stokes) cell leaves the input exactly as production's drop removes it from master_catalog.csv and
             pertie_detections.csv (wraps/builder_grouping.Omission). The re-judging removals (tenmin_rejudge) take
             sources, member rows and pages only: production keeps their detections in the detection tables
             (sublst_deep_omission_drop.py:22-30, :180-183), so the grouping keeps them too. A missing, unreadable or
             incomplete table, or one without an LST of the input, REFUSES (S-G28); the run records the table's path and
             sha256 (meta 'omission') and the rows dropped (grouping_input/omission_dropped.csv). Every LST's build stage
             then reads the grouping and DROPS the rows of the same OMIT cells from its builder input (the grouping does not
             hold them; wraps/builder_grouping.py, DEEP OMISSION) - any other selected row the grouping does not hold
             refuses (S-G18). THE ORDER for the all-nights pass: every LST's chain --until veto, the omit stage over all
             LSTs, the group stage with --group-omission and --group-lst-dir <LST>=<its catalogue directory> for every LST,
             then every LST's chain --from build (lib/catalog/incremental_sublst.FIRST_RUN). KNOWN LIMIT: a regroup can
             rename or merge identities of LSTs built before it; their published catalogues still carry the old names (the
             runner's start line records the grouping table's sha1) - the cumulative build does not yet refuse a mix of
             tables (report: what remains).
PRODUCTS (per Stokes S, directory grouping_<S>/ of the stage's output)
  detections.csv   one row per detection of the input (DET_COLS): its catalog fingerprint, identity and multiples-level IDs and
                   display names, the multiples group's multiplicity (10-minute windows / deep nights) and multiple flag
  identities.csv   one row per identity (IDENT_COLS): its multiples-level link, counts, bands, best detection, name position
  multiples.csv    one row per multiples-level group (MULT_COLS): its identities, counts, multiplicity, multiple flag
  summary.json     the module's run summary + the inputs and the mode;   events.jsonl  this run's events (grouping.append_events)
  the state        <state dir>/grouping_<tier>_<S>_state.json (GroupingState; saved only after the products pass their content
                   tests) and <state dir>/grouping_<tier>_<S>_events.jsonl (every run's events, appended)
  Content tests (lib/catalog/tests_grouping.py): class grouping_table (detections.csv + its siblings) and grouping_state.
LABELS (carry_judgements): a journal of Gregg's judgements keyed by the OLD page ids is carried to the new identities by member
  detections: a new identity whose old judged pages all agree on (confidence, reason, reason 2) takes that judgement UNCHANGED
  (carried_* fields record from where); disagreement is a CONFLICT - reported, never carried; an old judgement that reaches no
  detection is reported. Output: judgements.jsonl (refuses to write over a non-empty journal) + carry_report.csv - the
  preview's mechanism (grouping_20261026/preview/build_preview.py carry_labels), generalised.
"""
import collections
import csv
import glob
import hashlib
import json
import math
import os
import re
import shutil
import time

from exopipe.phase3.core import placement
from exopipe.phase3.lib.catalog import grouping as G

STAGE = 'group'
GROUPINGS = ('design-b', 'production')
RULES = ('one-rule', 'production')
RUNGS = ('multiples-level', 'production')
DETECTIONS, IDENTITIES, MULTIPLES, SUMMARY, EVENTS = ('detections.csv', 'identities.csv', 'multiples.csv', 'summary.json',
                                                      'events.jsonl')
DET_COLS = ('cand_id', 'stokes', 'band', 'lst', 'date', 'key', 'night', 'ra', 'dec', 'snr', 'identity_id', 'identity_name',
            'multiples_id', 'multiples_name', 'multiplicity', 'multiple')
IDENT_COLS = ('identity_id', 'identity_name', 'multiples_id', 'multiples_name', 'multiple', 'n_det', 'n_windows', 'n_nights',
              'n_lsts', 'bands', 'best_cand_id', 'best_snr', 'name_ra', 'name_dec')
MULT_COLS = ('multiples_id', 'multiples_name', 'multiple', 'multiplicity', 'n_identities', 'identity_ids', 'identity_names',
             'n_det', 'n_windows', 'n_nights', 'n_lsts')
# the builder's extra source columns under design B (lib/catalog/wraps/builder_grouping.py writes them; tests read them)
SOURCE_COLS = ('group_identity_id', 'multiples_id', 'multiples_name', 'multiple', 'multiples_multiplicity',
               'multiples_n_identities', 'multiples_identities')
CARRY_COLS = ('new_source', 'new_identity_id', 'n_detections', 'old_sources', 'outcome', 'verdicts')   # production's
#                                                       carry report (deep_carry_judgements.py) + the stable identity ID
TIER_OF = {'deep': 'deep', 'sublst': '10min', '10min': '10min'}
csv.field_size_limit(1 << 30)


class StageRefusal(RuntimeError):
    """A grouping-stage gate refused (its text names the gate)."""


def _cfg():
    from exopipe.phase3.config import CFG
    return CFG


# =====================================================================================================================
# the mode (config)
# =====================================================================================================================

class Mode:
    def __init__(self, grouping, rule, rungs):
        self.grouping, self.rule, self.rungs = grouping, rule, rungs
        self.design_b = grouping == 'design-b'

    def as_dict(self):
        return dict(grouping=self.grouping, multiples_rule=self.rule, repeat_rungs=self.rungs)

    def __repr__(self):
        return f'Mode({self.grouping}, {self.rule}, {self.rungs})'


def mode(cfg=None):
    """The configured grouping mode; REFUSES an unknown value or a GROUPING / MULTIPLES_RULE pair that does not agree (design B's
    multiples level IS the one rule's multiplicity, and production's two chains have no meaning on design B's identities)."""
    g = (cfg or _cfg()).catalog_grouping
    grouping, rule, rungs = g.GROUPING, g.MULTIPLES_RULE, g.REPEAT_RUNGS
    if grouping not in GROUPINGS:
        raise StageRefusal(f'catalog_grouping.GROUPING = {grouping!r} is not one of {GROUPINGS} - REFUSING')
    if rule not in RULES:
        raise StageRefusal(f'catalog_grouping.MULTIPLES_RULE = {rule!r} is not one of {RULES} - REFUSING')
    if rungs not in RUNGS:
        raise StageRefusal(f'catalog_grouping.REPEAT_RUNGS = {rungs!r} is not one of {RUNGS} - REFUSING')
    if (grouping == 'production') != (rule == 'production'):
        raise StageRefusal(f'catalog_grouping.GROUPING = {grouping!r} with MULTIPLES_RULE = {rule!r}: the two must agree - the '
                           f"one rule's multiplicity IS design B's multiples level, and production's two multiples chains act on "
                           f"production's sources (set both to 'production', as the parity file does, or neither) - REFUSING")
    return Mode(grouping, rule, rungs)


def design_b(cfg=None):
    return mode(cfg).design_b


def group_dir(out, stokes):
    return os.path.join(out, f'grouping_{stokes}')


def sublst_dir(cfg=None):
    """catalog_grouping.SUBLST_DIR with {transients_root} = paths.TRANSIENTS_ROOT."""
    cfg = cfg or _cfg()
    return cfg.catalog_grouping.SUBLST_DIR.format(transients_root=cfg.paths.TRANSIENTS_ROOT)


# =====================================================================================================================
# inputs
# =====================================================================================================================

def load_det(tier, stokes, master_catalog, side):
    """det of one tier and Stokes (lib/catalog/grouping_inputs): 10-minute master_catalog.csv + pertie_detections.csv; deep
    master_catalog.csv + frame_beam_bmaj.csv. A missing input REFUSES."""
    from exopipe.phase3.lib.catalog import grouping_inputs as GI
    for p in (master_catalog, side):
        if not p or not os.path.exists(p):
            raise StageRefusal(f'grouping input {p} is missing - REFUSING')
    if tier == '10min':
        return GI.load_tenmin(master_catalog, side, stokes)
    if tier == 'deep':
        return GI.load_deep(master_catalog, side, stokes)
    raise StageRefusal(f'grouping: tier {tier!r} is not 10min or deep - REFUSING')


def stokes_rows(master_catalog, stokes):
    """[(cand_id, row)] of the Stokes' rows of a catalogue, in file order (the content test's independent read)."""
    with open(master_catalog, newline='') as f:
        return [(r['cand_id'], r) for r in csv.DictReader(f) if r.get('stokes') == stokes]


# =====================================================================================================================
# the products
# =====================================================================================================================

def _write_csv(path, cols, rows):
    # the directory's group and ACL, or the replaced file's (core/placement.py)
    with placement.atomic(path) as tmp, open(tmp, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(cols))
        w.writeheader()
        w.writerows(rows)
    return path


def _date_of(key):
    return key.split('|', 1)[1] if '|' in key else ''


def tables(det, res, tier, stokes):
    """(detections rows, identities rows, multiples rows) of a GroupingResult res on det."""
    import numpy as np
    n = len(res.cand_id)
    snr = np.asarray(det['snr'], float)
    ra = det['ra_raw'] if 'ra_raw' in det else det['ra']
    de = det['dec_raw'] if 'dec_raw' in det else det['dec']
    keys = [str(k) for k in det['key']]
    nights = [str(x) for x in det['night']]
    lsts = [str(x) for x in det['lst']]
    bands = [str(x) for x in det['band']]
    ident = [str(x) for x in res.identity_id]
    mult = [str(x) for x in res.multiples_id]
    rows = []
    for j in range(n):
        rows.append(dict(cand_id=str(res.cand_id[j]), stokes=stokes, band=bands[j], lst=lsts[j], date=_date_of(keys[j]),
                         key=keys[j], night=nights[j], ra=repr(float(ra[j])), dec=repr(float(de[j])), snr=repr(float(snr[j])),
                         identity_id=ident[j], identity_name=res.names[ident[j]], multiples_id=mult[j],
                         multiples_name=res.names[mult[j]], multiplicity=int(res.multiplicity[j]), multiple=int(res.multiple[j])))
    mem = collections.defaultdict(list)
    for j, i in enumerate(ident):
        mem[i].append(j)
    mmem = collections.defaultdict(list)
    for j, m in enumerate(mult):
        mmem[m].append(j)
    seq = G._seq_of
    irows = []
    for i in sorted(mem, key=seq):
        js = mem[i]
        b = max(js, key=lambda j: (abs(snr[j]), -j))
        pr, pd = G.mean_position(det['ra'][js], det['dec'][js], det['C'][js], snr[js])
        m = mult[js[0]]
        irows.append(dict(identity_id=i, identity_name=res.names[i], multiples_id=m, multiples_name=res.names[m],
                          multiple=int(res.multiple[js[0]]), n_det=len(js), n_windows=len({keys[j] for j in js}),
                          n_nights=len({nights[j] for j in js}), n_lsts=len({lsts[j][:3] for j in js}),
                          bands='+'.join(sorted({bands[j] for j in js})), best_cand_id=str(res.cand_id[b]),
                          best_snr=repr(float(snr[b])), name_ra=repr(pr), name_dec=repr(pd)))
    mrows = []
    for m in sorted(mmem, key=seq):
        js = mmem[m]
        ids = sorted({ident[j] for j in js}, key=seq)
        mrows.append(dict(multiples_id=m, multiples_name=res.names[m], multiple=int(res.multiple[js[0]]),
                          multiplicity=int(res.multiplicity[js[0]]), n_identities=len(ids), identity_ids=';'.join(ids),
                          identity_names=';'.join(res.names[i] for i in ids), n_det=len(js),
                          n_windows=len({keys[j] for j in js}), n_nights=len({nights[j] for j in js}),
                          n_lsts=len({lsts[j][:3] for j in js})))
    return rows, irows, mrows


def write_products(det, res, tier, stokes, out_dir, inputs, mode_, d=None):
    """Write the products into directory d (default grouping_<S>/ under out_dir; run_group writes a temporary directory and
    renames it into place only after the content tests pass): the three tables, summary.json and this run's events. Returns
    ({name: path}, summary)."""
    d = d or group_dir(out_dir, stokes)
    os.makedirs(d, exist_ok=True)
    rows, irows, mrows = tables(det, res, tier, stokes)
    paths = dict(detections=_write_csv(os.path.join(d, DETECTIONS), DET_COLS, rows),
                 identities=_write_csv(os.path.join(d, IDENTITIES), IDENT_COLS, irows),
                 multiples=_write_csv(os.path.join(d, MULTIPLES), MULT_COLS, mrows))
    ev = os.path.join(d, EVENTS)
    if os.path.exists(ev):
        os.remove(ev)
    G.append_events(ev, res)
    paths['events'] = ev
    summ = dict(res.summary, tier=tier, stokes=stokes, inputs=inputs, mode=mode_.as_dict(),
                counts=dict(detections=len(rows), identities=len(irows), multiples_groups=len(mrows),
                            multiples=sum(1 for r in mrows if r['multiple']),
                            identities_in_multiples=sum(1 for r in irows if r['multiple']),
                            detections_in_multiples=sum(1 for r in rows if r['multiple'])))
    p = os.path.join(d, SUMMARY)
    with placement.atomic(p) as tmp, open(tmp, 'w') as f:
        json.dump(summ, f, indent=1, sort_keys=True, default=str)
    paths['summary'] = p
    return paths, summ


def state_path(state_dir, tier, stokes):
    return G.GroupingState(tier, stokes).path(state_dir)


def history_path(state_dir, tier, stokes):
    return os.path.join(state_dir, f'grouping_{tier}_{stokes}_events.jsonl')


def _utc():
    return time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())


class _Lock:
    """An exclusive flock on ``path`` held for a with-block: a second holder REFUSES (StageRefusal, naming the holder written
    into the file) - two grouping runs never share a state (review 2026-11-04, F2). ``wait_s`` > 0: poll for up to that many
    seconds first (the per-hour 10-minute items, lib/catalog/incremental_sublst.night_lock); default 0, refuse at once."""

    def __init__(self, path, what, wait_s=0):
        self.path, self.what, self.fd, self.wait_s = path, what, None, float(wait_s or 0)

    def __enter__(self):
        import fcntl
        import time
        os.makedirs(os.path.dirname(self.path) or '.', exist_ok=True)
        self.fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o664)
        t_end = time.time() + max(0.0, self.wait_s)
        while True:
            try:
                fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except (BlockingIOError, PermissionError):
                if time.time() < t_end:
                    time.sleep(min(10.0, max(0.1, t_end - time.time())))
                    continue
                try:
                    holder = os.pread(self.fd, 400, 0).decode('utf-8', 'replace').strip()
                except OSError:
                    holder = '?'
                os.close(self.fd)
                self.fd = None
                raise StageRefusal(f'{self.what}: the lock {self.path} is held by another grouping run ({holder or "holder unknown"})'
                                   + (f' after {self.wait_s:.0f} s of waiting' if self.wait_s else '')
                                   + ' - two runs never group one state at once - REFUSING') from None
        os.ftruncate(self.fd, 0)
        os.pwrite(self.fd, f'pid {os.getpid()} host {os.uname()[1]} since {_utc()}\n'.encode(), 0)
        return self

    def __exit__(self, *a):
        import fcntl
        if self.fd is not None:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            os.close(self.fd)
            self.fd = None


def state_lock_path(state_dir, tier, stokes):
    return os.path.join(state_dir, f'.grouping_{tier}_{stokes}.lock')


def run_group(det, tier, stokes, out_dir, state_dir, run_label, inputs, group_mode=None, new_state=False, check=None,
              reserved_names=None):
    """Group det (one tier and Stokes) against the saved state and replace grouping_<S>/ only when every check passed
    (review 2026-11-04, F2): the products are written into a TEMPORARY directory beside it; ``check`` (a callable (paths) ->
    [failures]: the content tests) runs there; on a failure the directory is renamed grouping_<S>.refused_<UTC> (kept for
    inspection, never at the path the builds read) and nothing else changes; on a pass the state is saved (atomic) with its
    event history, then the previous grouping_<S>/ is moved to grouping_<S>.superseded_<UTC> and the new one renamed into
    place. The whole run holds an exclusive lock on the state (<state dir>/.grouping_<tier>_<S>.lock; a second run refuses).
    reserved_names: the display names no identity here may take (default: those of the tier's other Stokes in out_dir now,
    present_names; the cross-Stokes link, lib/catalog/stokes_link.py). A run that groups both Stokes passes naming_order's
    (THE NAMING RULE).
    Returns (paths, summary, result) with the final paths and the table's sha1 (paths['detections_sha1'])."""
    m = mode()
    if not m.design_b:
        raise StageRefusal('grouping stage under catalog_grouping.GROUPING = production - nothing to do (the caller ends no-input)')
    final = group_dir(out_dir, stokes)
    with _Lock(state_lock_path(state_dir, tier, stokes), f'grouping {tier} {stokes}'):
        try:
            st = G.GroupingState.load(state_dir, tier, stokes, new=new_state)
        except G.GroupingRefused as e:
            raise StageRefusal(str(e)) from None
        try:
            res = G.group_catalog(det, tier, stokes, state=st, run_label=run_label, mode=group_mode,
                                  reserved_names=present_names(out_dir, other_stokes(stokes)) if reserved_names is None
                                  else reserved_names)
        except G.GroupingRefused as e:
            raise StageRefusal(str(e)) from None
        tmp = f'{final}.tmp.{os.getpid()}'
        if os.path.exists(tmp):
            shutil.rmtree(tmp)
        paths, summ = write_products(det, res, tier, stokes, out_dir, inputs, m, d=tmp)
        # the state this run hands on, written BESIDE the products first (the content test reads it), then saved
        paths['state_next'] = res.state.save(os.path.join(tmp, 'state_next'))
        prev = state_path(state_dir, tier, stokes)
        if os.path.exists(prev):
            os.makedirs(os.path.join(tmp, 'state_previous'), exist_ok=True)
            paths['state_previous'] = shutil.copy2(prev, os.path.join(tmp, 'state_previous'))
        if check is not None:
            bad = check(paths)
            if bad:
                kept = f'{final}.refused_{_utc()}'
                os.rename(tmp, kept)
                raise StageRefusal('grouping products failed their content tests: ' + '; '.join(bad[:6]) + f' - the state is '
                                   f'NOT advanced and {final} is unchanged (the refused products: {kept}) - REFUSING')
        state_file = res.state.save(state_dir)
        hp = history_path(state_dir, tier, stokes)
        G.append_events(hp, res)
        if os.path.exists(final):
            os.rename(final, f'{final}.superseded_{_utc()}')
        os.rename(tmp, final)
    paths = {k: (final + v[len(tmp):] if isinstance(v, str) and v.startswith(tmp) else v) for k, v in paths.items()}
    paths.update(state=state_file, history=hp, detections_sha1=table_digest(paths['detections']))
    return paths, summ, res


def other_stokes(stokes):
    return {'I': 'V', 'V': 'I'}[stokes]


def naming_order(stokes):
    """(the Stokes in grouping order, {S: reserved_names for run_group}) of a run that groups ``stokes`` - THE NAMING RULE
    (a display name names one (Stokes, identity) of the tier; the cross-Stokes link, lib/catalog/stokes_link.py):
      both Stokes in one run   the names are decided from this run's groupings only: I is grouped first and reserves nothing
                               (I keeps its short names), then V reserves I's NEW names (run_group's default, present_names,
                               read after I's table is in place), so V takes the long form on a collision. Deterministic: an
                               unchanged dataset gives unchanged names run after run, in whatever order the Stokes were asked.
      one Stokes               it reserves the other Stokes' present names (run_group's default): after a run of both on the
                               same data that gives the same names, and a name is never held in both Stokes."""
    if {'I', 'V'} <= set(stokes):
        return sorted(stokes, key=lambda S: S != 'I'), {'I': frozenset()}
    return list(stokes), {}


def present_names(out_dir, stokes):
    """the identity display names of Stokes `stokes` in grouping_<S>/ under out_dir now (identities.csv; none: empty) - what
    the other Stokes' grouping must not reuse (display_names reserved)"""
    p = os.path.join(group_dir(out_dir, stokes), IDENTITIES)
    if not os.path.exists(p):
        return frozenset()
    with open(p, newline='') as f:
        return frozenset(r['identity_name'] for r in csv.DictReader(f))


def read_detections(path):
    """{cand_id: row} of a detections.csv (a refusal names a duplicate cand_id)."""
    out = {}
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            if r['cand_id'] in out:
                raise StageRefusal(f'{path}: cand_id {r["cand_id"]} twice - REFUSING')
            out[r['cand_id']] = r
    return out


def table_digest(path):
    import hashlib
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


# =====================================================================================================================
# the builder with the grouping (lib/catalog/wraps/builder_grouping.py): what a stage hands it
# =====================================================================================================================

WRAP_ENV = ('EXOPIPE3_GROUPING_MODE', 'EXOPIPE3_GROUPING_TABLE', 'EXOPIPE3_GROUPING_TIER', 'EXOPIPE3_GROUPING_RUNGS')


def builder_runner():
    """Path of the builder runner (self-contained: loaded by path in the child, whose `exopipe` can be production's)."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wraps', 'builder_grouping.py')


UNIT_LOCK = '.grouping.lock'              # the 10-minute unit's lock in its output (the shared state's) directory


def _latest_group_record(grouping_dir, stokes):
    """(finished, record path, record) of the LATEST group unit record of grouping_dir that grouped Stokes ``stokes``."""
    recs = []
    for p in sorted(glob.glob(os.path.join(grouping_dir, '_records', 'catalog', '*.json'))):
        try:
            with open(p) as f:
                r = json.load(f)
        except (OSError, ValueError):
            continue
        meta = r.get('meta') or {}
        if meta.get('stage') == STAGE and stokes in (meta.get('stokes') or ()):
            recs.append((r.get('finished') or '', p, r))
    if not recs:
        raise StageRefusal(f'no record of a 10-minute grouping run of Stokes {stokes} under {grouping_dir}/_records/catalog - '
                           f'run `exopipe3 catalog --tier sublst --stages group` - REFUSING')
    return max(recs, key=lambda x: (x[0], x[1]))


def recorded_inputs(grouping_dir, stokes, with_table=False):
    """({LST: all-nights catalog directory}, [(LST, night catalog directory), ...]) the grouping the builds read was assembled
    from: the record recorded_table reads (same refusals: the latest run must be done and its table unchanged). A night's delta
    build (lib/catalog/tier_sublst._build_delta) takes its appended nights from here, so the builder is never handed other
    nights than the grouping was. with_table: also recorded_table's (table, sha1), from the SAME record (one read: a grouping
    run finishing in between cannot give the table of one run and the nights of another)."""
    rec = _latest_group_record(grouping_dir, stokes)
    tab = recorded_table(grouping_dir, stokes, _rec=rec)
    meta = rec[2].get('meta') or {}
    inp = meta.get('input')
    out = (dict(inp) if isinstance(inp, dict) else {}), [(L, D) for L, D in (meta.get('input_nights') or [])]
    return out + (tab,) if with_table else out


def recorded_table(grouping_dir, stokes, _rec=None):
    """(table path, sha1) of the 10-minute grouping the LST builds read (review 2026-11-04, F2): the group unit records of
    grouping_dir (<dir>/_records/catalog/*.json of stage 'group') that grouped Stokes ``stokes`` - the LATEST must be done and
    have recorded this table (grouping_<S>/detections.csv) with the sha1 it has now. A refused latest run, a table its run did
    not record, or a table changed since REFUSES (StageRefusal) - a build never reads a grouping no successful run made."""
    table = os.path.join(os.path.abspath(grouping_dir), f'grouping_{stokes}', DETECTIONS)
    fin, p, r = _rec or _latest_group_record(grouping_dir, stokes)   # _rec: recorded_inputs' one read of the record
    if r.get('status') != 'done':
        raise StageRefusal(f'the latest 10-minute grouping run of Stokes {stokes} ({os.path.basename(p)}, {fin}) ended '
                           f'{r.get("status")}: {(r.get("reason") or "")[:200]} - group again before building - REFUSING')
    t = ((r.get('meta') or {}).get('tables') or {}).get(stokes)
    if not t or os.path.abspath(t.get('path', '')) != table:
        raise StageRefusal(f'the latest grouping run ({os.path.basename(p)}) recorded no table of Stokes {stokes} at {table} - '
                           f'REFUSING')
    if not os.path.exists(table):
        raise StageRefusal(f'{table} is missing although {os.path.basename(p)} recorded it - REFUSING')
    sha = table_digest(table)
    if sha != t.get('sha1'):
        raise StageRefusal(f'{table} changed since its grouping run recorded it (sha1 {sha[:12]} vs {str(t.get("sha1"))[:12]}) - '
                           f'REFUSING')
    return table, sha


FROZEN_MODES = ('', 'record', 'delta', 'passthrough')     # the runner's EXOPIPE3_FROZEN_MODE (its FROZEN STATISTICS text)


def builder_env(table, tier, mode_=None, sha1=None, cfg=None, frozen='', frozen_record=None, frozen_touch='', removed_dirs=(),
                omission=None):
    """The environment of the runner: the grouping table of the key's Stokes and its expected sha1 (the runner refuses a table
    that changed after the stage checked it), the tier, the repeat rungs, and the accepted builders (catalog_grouping.
    BUILDER_SHA1: path-normalized sha1s of the builders the patches were written for; review 2026-11-04, F7).
    frozen: the frozen-statistics mode of the per-hour delta build - '' (none), 'record' (an all-nights build writes
    builder_frozen.json), 'delta' (a night's build on the record ``frozen_record``), 'passthrough' (the check of the hooks).
    removed_dirs: the run directories whose removal tables (selection / ephemeris) the build's record reads
    (wraps/builder_grouping.removed_before_builder); empty = the directory of the build's input. The frozen names and
    EXOPIPE3_REMOVED_DIRS are ALWAYS set, so a value inherited from the caller's environment never reaches a build.
    omission: {path, sha256} of the deep-omission decision table the grouping recorded (recorded_omission; the 10-minute
    tier: the runner drops its OMIT cells from the builder input and refuses a 10-minute build without one); the two names
    are ALWAYS set, empty for none."""
    if frozen not in FROZEN_MODES:
        raise StageRefusal(f'builder_env: frozen={frozen!r} is not one of {FROZEN_MODES} - REFUSING')
    if (frozen == 'delta') != bool(frozen_record):
        raise StageRefusal(f'builder_env: frozen={frozen!r} with frozen_record={frozen_record!r}: a delta build names the '
                           f'all-nights build\'s record, no other build does - REFUSING')
    m = mode_ or mode(cfg)
    g = (cfg or _cfg()).catalog_grouping
    env = {'EXOPIPE3_GROUPING_MODE': 'design-b', 'EXOPIPE3_GROUPING_TABLE': os.path.abspath(table),
           'EXOPIPE3_GROUPING_TIER': TIER_OF.get(tier, tier), 'EXOPIPE3_GROUPING_RUNGS': m.rungs,
           'EXOPIPE3_GROUPING_BUILDER_SHA1': ','.join(g.BUILDER_SHA1),
           'EXOPIPE3_GROUPING_PROVENANCE': '1',     # grouping_table.json beside the catalogues (the cumulative's one-table gate)
           'EXOPIPE3_FROZEN_MODE': frozen, 'EXOPIPE3_FROZEN_RECORD': os.path.abspath(frozen_record) if frozen_record else '',
           'EXOPIPE3_FROZEN_TOUCH': frozen_touch or '',
           'EXOPIPE3_FROZEN_BUILDER_SHA1': ','.join(g.FROZEN_BUILDER_SHA1),
           'EXOPIPE3_REMOVED_DIRS': os.pathsep.join(os.path.abspath(d) for d in removed_dirs),
           'EXOPIPE3_GROUPING_OMISSION': (omission or {}).get('path') or '',
           'EXOPIPE3_GROUPING_OMISSION_SHA256': (omission or {}).get('sha256') or ''}
    if sha1:
        env['EXOPIPE3_GROUPING_TABLE_SHA1'] = sha1
    return env


def builder_line(text):
    """The runner's machine line ('GROUPING-BUILDER ...') from a log tail, or ''."""
    for ln in reversed((text or '').splitlines() if isinstance(text, str) else list(text or ())):
        if ln.startswith('GROUPING-BUILDER '):
            return ln
    return ''


def covers(table_path, selected_csv, omission=None):
    """[problems]: rows of the builder's input (a selected_<K>.csv) the grouping table does not hold, or holds with another
    fingerprint (band, LST, date, catalogue ra / dec / S/N) - a stale grouping (the builder runner refuses the same).
    omission ({path, sha256}, recorded_omission): rows of its OMIT cells are not problems - the runner drops them from the
    builder input (the grouping holds none of them)."""
    T = read_detections(table_path)
    om = None
    if omission:
        try:
            om = _bg().Omission(omission['path'], omission['sha256'])
        except _bg().Refused as e:
            return [f'S-G28 {e}']
    bad = []
    with open(selected_csv, newline='') as f:
        for r in csv.DictReader(f):
            if om is not None:
                try:
                    if om.omitted(r['cand_id'], r.get('stokes'), r.get('lst', '')):
                        continue
                except _bg().Refused as e:
                    bad.append(f'S-G28 {e}')
                    break
            e = T.get(r['cand_id'])
            if e is None:
                bad.append(f'{r["cand_id"]}: not in the grouping')
            elif (e['band'], e['lst'], e['date']) != (r['band'], r['lst'], r['date']) or any(
                    float(e[k]) != float(r[k]) for k in ('ra', 'dec', 'snr')):
                bad.append(f'{r["cand_id"]}: grouped with another fingerprint')
            if len(bad) >= 20:
                break
    return bad


# =====================================================================================================================
# the deep identity stage under design B (lib/catalog/deep_chain.st_identity, WAVE2 GROUPING)
# =====================================================================================================================
# Production's identity stage (exopipe_ops p3s3_identity.py) re-groups the per-tier survivors by single linkage across tiers AND
# Stokes and rewrites identity_id in place: under design B it would replace the ruled identities (the pages group their rows by
# identity_id - one review entry per identity - and the serve's deep judgement carry keys the new sources by it). Under design B
# the stage writes the same four columns from design B's grouping instead, per Stokes (design B groups every band, wideband and
# subband, of one Stokes): identity_id = the identity's display name (as the builder wrote it), identity_bands / identity_tiers /
# identity_n over the identity's survivors rows of the Stokes' tiers. No identity spans two Stokes (design B is per Stokes; the
# one-name-across-Stokes rule of R-CROSS-TIER-SINGLE-IDENTITY has no design-B counterpart yet - an open question; the pages'
# cross-tier / polarization links, by position, still show an I and a V identity of one source).

def _stokes_of_dir(d):
    k = os.path.basename(d).replace('source_catalogs_', '')
    return k.split('_')[0] if k[:1] in ('I', 'V') and '_' in k else None


def identity_writeback(tier_files, union_files, tables):
    """Write identity_id / identity_bands / identity_tiers / identity_n from design B into every per-tier survivors file
    (tier_files: source_catalogs_<S>_<t>/*_sources_survivors.csv) and the union copies (union_files: source_catalogs/, rows
    keyed by (stokes, source_id, band)). tables: {Stokes: grouping detections.csv}. A row whose identity_id is not the
    grouping's identity of its best detection, or a tier file of a Stokes without a table, REFUSES (the builds were not made
    with this grouping). Returns counts."""
    T = {S: read_detections(p) for S, p in tables.items()}
    per = collections.defaultdict(list)         # (S, identity name) -> [(tier, band)]
    rows_of, bad = {}, []
    for f in sorted(tier_files):
        d = os.path.dirname(f)
        S = _stokes_of_dir(d)
        if S not in T:
            raise StageRefusal(f'{f}: no design-B grouping of Stokes {S} - REFUSING')
        tier = os.path.basename(d).replace('source_catalogs_', '')
        cols, rr = None, []
        with open(f, newline='') as fh:
            rd = csv.DictReader(fh)
            cols, rr = list(rd.fieldnames or []), list(rd)
        rows_of[f] = (cols, rr, S, tier)
        for r in rr:
            e = T[S].get(r.get('best_cand_id', ''))
            if e is None or r.get('identity_id') != e['identity_name'] or r.get('source_id') != e['identity_name']:
                bad.append(f'{os.path.basename(d)}/{os.path.basename(f)} {r.get("source_id")}: identity_id '
                           f'{r.get("identity_id")!r}, the grouping names its best detection {r.get("best_cand_id")} '
                           f'{e["identity_name"] if e else "(not grouped)"!r}')
                continue
            per[(S, e['identity_name'])].append((tier, r['band']))
    if bad:
        raise StageRefusal(f'{len(bad)} survivors row(s) are not named by their design-B identity, e.g. {bad[:2]} - the builds '
                           f'were not made with this grouping - REFUSING')
    val = {k: ('+'.join(sorted({b for _, b in v})), '+'.join(sorted({t for t, _ in v})), str(len(v))) for k, v in per.items()}
    n_rows = 0
    for f, (cols, rr, S, tier) in rows_of.items():
        for c in ('identity_tiers', 'identity_n'):
            if c not in cols:
                cols.append(c)
        for r in rr:
            b_, t_, n_ = val[(S, r['identity_id'])]
            r.update(identity_bands=b_, identity_tiers=t_, identity_n=n_)
        if rr:
            _write_csv(f, cols, rr)
        n_rows += len(rr)
    n_union = 0
    for f in sorted(union_files):
        with open(f, newline='') as fh:
            rd = csv.DictReader(fh)
            cols, rr = list(rd.fieldnames or []), list(rd)
        if not rr:
            continue
        for c in ('identity_tiers', 'identity_n'):
            if c not in cols:
                cols.append(c)
        for r in rr:
            k = (r.get('stokes'), r.get('identity_id') or r.get('source_id'))
            if k not in val:
                raise StageRefusal(f'{f}: union row {r.get("stokes")} {r.get("source_id")} {r.get("band")} has no design-B '
                                   f'identity among the tier survivors - REFUSING')
            r.update(identity_bands=val[k][0], identity_tiers=val[k][1], identity_n=val[k][2])
        _write_csv(f, cols, rr)
        n_union += len(rr)
    return dict(tier_rows=n_rows, union_rows=n_union, identities={S: sum(1 for k in val if k[0] == S) for S in T},
                cross_tier={S: sum(1 for k, v in val.items() if k[0] == S and '+' in v[1]) for S in T})


def identity_failures(path, tables):
    """[failures] of a survivors file (per-tier or union) against design B: identity_id = the grouping's identity name of the
    row's best detection (and = source_id), identity_n >= 1, and every row of one identity with the same bands / tiers / n."""
    T = {S: read_detections(p) for S, p in tables.items()}
    d = os.path.dirname(os.path.abspath(path))
    union = os.path.basename(d) == 'source_catalogs'
    with open(path, newline='') as fh:
        rr = list(csv.DictReader(fh))
    bad, seen = [], {}
    for r in rr:
        S = r.get('stokes') if union else _stokes_of_dir(d)
        e = T.get(S, {}).get(r.get('best_cand_id', ''))
        if e is None or r.get('identity_id') != e['identity_name'] or r.get('source_id') != e['identity_name']:
            bad.append(f'{r.get("source_id")} {r.get("band")}: identity_id {r.get("identity_id")!r} is not the design-B identity')
            continue
        try:
            n = int(r.get('identity_n') or 0)
        except ValueError:
            n = 0
        if n < 1:
            bad.append(f'{r.get("source_id")}: identity_n {r.get("identity_n")!r}')
        k = (S, r['identity_id'])
        v = (r.get('identity_bands'), r.get('identity_tiers'), r.get('identity_n'))
        if seen.setdefault(k, v) != v:
            bad.append(f'{r.get("source_id")}: identity columns differ between its rows ({v} vs {seen[k]})')
    return bad


# =====================================================================================================================
# the one rule against today's outcome (a record: which detections the rule serves, rescues and drops)
# =====================================================================================================================

def builder_outcomes(cat_dirs, selected):
    """{cand_id: (outcome, source key, promotion)} of every row of the builder inputs ``selected`` (selected_<K>.csv paths),
    read from the builder's own files in ``cat_dirs`` (source_catalogs_<K>/): <band>_sigma_stack_singleton_removed.csv,
    <band>_source_members.csv (every source), <band>_sources_master.csv (|best| >= 5), <band>_sources_removed_gate78.csv.
    outcome: 'survivor' | 'sigma-stack singleton' | '7-8 sigma size/axis (source)' | 'source below 7 sigma, unpromoted' |
    'unaccounted'. The source key is (source_id, band, lst_cells) - one per build (a 10-minute cumulative concatenates LSTs)."""
    thr = 7.0
    rem, mem, master, g78 = set(), {}, {}, set()
    for d in cat_dirs:
        for p in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            fp = os.path.join(d, p)
            if p.endswith('_sigma_stack_singleton_removed.csv'):
                with open(fp, newline='') as f:
                    rem.update(r['cand_id'] for r in csv.DictReader(f))
            elif p.endswith('_source_members.csv'):
                with open(fp, newline='') as f:
                    for r in csv.DictReader(f):
                        lst = r['cand_id'].split('_')[1] if r['cand_id'].count('_') >= 3 else ''
                        mem[r['cand_id']] = (r['source_id'], r['band'], lst)
            elif p.endswith('_sources_master.csv'):
                with open(fp, newline='') as f:
                    for r in csv.DictReader(f):
                        for L in (r.get('lst_cells') or '').split('+'):
                            master[(r['source_id'], r['band'], L[:3])] = r
            elif p.endswith('_sources_removed_gate78.csv'):
                with open(fp, newline='') as f:
                    for r in csv.DictReader(f):
                        for L in (r.get('lst_cells') or '').split('+'):
                            g78.add((r['source_id'], r['band'], L[:3]))
    out = {}
    for sp in selected:
        with open(sp, newline='') as f:
            for r in csv.DictReader(f):
                c = r['cand_id']
                if c in rem:
                    out[c] = ('sigma-stack singleton', None, '')
                    continue
                k = mem.get(c)
                if k is None:
                    out[c] = ('unaccounted', None, '')
                    continue
                k = (k[0], k[1], (k[2] or r.get('lst', ''))[:3])
                if k in g78:
                    out[c] = ('7-8 sigma size/axis (source)', k, '')
                    continue
                s = master.get(k)
                if s is None:
                    out[c] = ('unaccounted', k, '')
                    continue
                prom = s.get('promotion', '') or ''
                ok = abs(float(s['best_snr'])) >= thr or bool(prom)
                out[c] = ('survivor' if ok else f'source below {thr:g} sigma, unpromoted', k, prom)
    return out


REPEAT_RUNG_NAMES = ('RPT', 'REP', 'XREP')


def one_rule_outcome(row, sel, outcome, prom, sstack_thr, stokes, rungs):
    """The one rule's verdict of one detection: (served 0/1, reason). row: its detections.csv row; sel: its builder-input row or
    None (removed before the builder); outcome / prom: production's builder outcome and its source's promotion."""
    if sel is None:
        return 0, 'not in the builder input (removed before the builder: selection / ephemeris)'
    if int(row['multiple']):
        if outcome == 'unaccounted':
            return 0, 'unaccounted by the builder'
        return 1, 'member of a multiple (the three multiplicity gates skipped)'
    if stokes == 'I' and sstack_thr is not None:
        try:
            ss = float(sel.get('sstack') or '')
        except ValueError:
            ss = None
        if ss is not None and abs(float(sel['snr'])) < sstack_thr * max(ss, 1.0):
            return 0, 'single: sigma-stack singleton test fails'
    if outcome == '7-8 sigma size/axis (source)':
        return 0, 'single: 7-8 sigma size/axis gate (its source)'
    if outcome == 'unaccounted':
        return 0, 'unaccounted by the builder'
    if outcome in ('survivor', 'sigma-stack singleton') or outcome.startswith('source below'):
        lab = [x for x in (prom or '').split('+') if x]
        if rungs == 'multiples-level':
            lab = [x for x in lab if x.split(':')[0] not in REPEAT_RUNG_NAMES]
        return None, '+'.join(lab)          # the floor is judged per identity by the caller
    return 0, outcome


def one_rule_record(table_path, cat_dirs, selected, stokes, sstack_thr=7.0, rungs='multiples-level', thr=7.0):
    """A RECORD of the one rule against TODAY'S builder outcome (production's grouping and gates), per detection of the
    grouping: rows (cand_id, identity, multiples, multiple, today's outcome, the rule's verdict) and counts - served today and
    under the rule, rescued (in a multiple, removed today by a multiplicity gate), dropped (served today, a single the rule
    removes: the sigma-stack test, the identity floor with the repeat rungs following the multiples level). The single-window
    identity's floor: its best |S/N| among its members that pass the sigma-stack test >= thr, or a non-repeat rung of a member's
    source. An estimate on production's sources (the builder with the grouping forms identity rows; this table compares
    outcomes, it is not the catalogue)."""
    T = read_detections(table_path)
    sel = {}
    for sp in selected:
        with open(sp, newline='') as f:
            for r in csv.DictReader(f):
                sel[r['cand_id']] = r
    out = builder_outcomes(cat_dirs, selected)
    pending = collections.defaultdict(list)
    verdict = {}
    for c, r in T.items():
        o, _, prom = out.get(c, ('not in the builder input', None, ''))
        s = sel.get(c)
        v, why = one_rule_outcome(r, s, o, prom, sstack_thr, stokes, rungs)
        if v is None:
            pending[r['identity_id']].append((c, why))
        else:
            verdict[c] = (v, why)
    for i, items in pending.items():
        best = max(abs(float(sel[c]['snr'])) for c, _ in items)
        rungs_ = sorted({x for _, w in items for x in w.split('+') if x})
        ok = best >= thr or bool(rungs_)
        for c, _ in items:
            verdict[c] = (1, f'single: identity floor passes ({"|S/N| %.2f" % best if best >= thr else "+".join(rungs_)})') if ok \
                else (0, f'single: identity below {thr:g} sigma, unpromoted (repeat rungs follow the multiples level)'
                      if rungs == 'multiples-level' else f'single: identity below {thr:g} sigma, unpromoted')
    rows, cnt = [], collections.Counter()
    for c, r in T.items():
        o = out.get(c, ('not in the builder input', None, ''))[0]
        today = int(o == 'survivor')
        v, why = verdict[c]
        rows.append(dict(cand_id=c, identity_id=r['identity_id'], identity_name=r['identity_name'],
                         multiples_id=r['multiples_id'], multiple=r['multiple'], today=o, served_today=today, served_rule=v,
                         rule=why))
        cnt['served_today'] += today
        cnt['served_rule'] += v
        if v and not today:
            cnt['rescued'] += 1
            cnt[f'rescued: {o}'] += 1
        if today and not v:
            cnt['dropped'] += 1
            cnt[f'dropped: {why}'] += 1
    ids_today = {r['identity_id'] for r in rows if r['served_today']}
    ids_rule = {r['identity_id'] for r in rows if r['served_rule']}
    mult_rule = {r['multiples_id'] for r in rows if r['served_rule'] and int(r['multiple'])}
    cnt['identities_with_a_served_detection_today'] = len(ids_today)
    cnt['identities_with_a_served_detection_rule'] = len(ids_rule)
    cnt['multiples_with_a_served_detection_rule'] = len(mult_rule)
    return rows, dict(cnt)


# =====================================================================================================================
# labels: carry Gregg's judgements to the new identities by member detections
# =====================================================================================================================

def read_journal(path, lines=None):
    """{source_id: the latest 'classify' record of scope 'source'} of a judgements.jsonl (a 'retract' removes it) - the
    journal's semantics (the review server's; serve/journals.latest_carry); ``lines`` (a list of text lines) instead of the
    file."""
    J = {}
    for line in (lines if lines is not None else open(path)):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get('action') == 'retract':
            J.pop(r.get('source_id'), None)
        elif r.get('scope', 'source') == 'source' and r.get('action') == 'classify':
            J[r['source_id']] = r
    return J


def _lst_of(cand_id):
    """the LST of a production cand_id ('<S><n>_<LST>_<date>_<band>_<seq>'), else ''"""
    parts = str(cand_id).split('_')
    return parts[1] if len(parts) >= 4 and len(parts[1]) == 3 and parts[1].endswith('h') else ''


def _member_dets(d, with_identity=False):
    """[(page id, detection)] of a catalogue directory: every member of a survivors row (page id = identity_id, else
    source_id); a detection = dict(cand_id, band, lst, date, ra, dec) (+ name, id: the row's source_id and group_identity_id)"""
    if not os.path.isdir(d):
        return []
    rows = {}
    for p in sorted(os.listdir(d)):
        if p.endswith('_sources_survivors.csv') and not p.startswith('ALL_'):
            with open(os.path.join(d, p), newline='') as f:
                for r in csv.DictReader(f):
                    rows[(r['source_id'], r['band'])] = r
    out = []
    for p in sorted(os.listdir(d)):
        if p.endswith('_source_members.csv'):
            with open(os.path.join(d, p), newline='') as f:
                for m in csv.DictReader(f):
                    r = rows.get((m['source_id'], m.get('band') or p.split('_')[0]))
                    if r is None:
                        continue
                    det = dict(cand_id=m['cand_id'], band=m.get('band') or r['band'], lst=_lst_of(m['cand_id']),
                               date=m.get('det_date', ''), ra=float(m['det_ra']), dec=float(m['det_dec']))
                    if with_identity:
                        det.update(name=r['source_id'], id=r.get('group_identity_id', ''))
                    out.append(((r.get('identity_id') or '').strip() or r['source_id'], det))
    return out


def page_members(cat_dirs):
    """{page id: [detection, ...]} of the pages a set of catalogue directories serves - page id = the survivors row's
    identity_id, else its source_id (the review page groups rows so: classify_transients.py) - with every member detection of
    the row's (source_id, band) (dicts: cand_id, band, lst, date, ra, dec). The membership the judgements were made on."""
    out = collections.defaultdict(list)
    seen = set()
    for d in cat_dirs:
        for pg, det in _member_dets(d):
            if (pg, det['cand_id']) not in seen:
                seen.add((pg, det['cand_id']))
                out[pg].append(det)
    return dict(out)


def identities_of_catalogs(cat_dirs):
    """[detection with name / id] of design-B catalogue directories: every member detection of a survivors row, named by its
    identity (the builder with the grouping names every source by it; group_identity_id holds the stable ID)."""
    out, seen = [], set()
    for d in cat_dirs:
        for _pg, det in _member_dets(d, with_identity=True):
            if det['cand_id'] not in seen:
                seen.add(det['cand_id'])
                out.append(det)
    return out


def table_detections(table):
    """[detection with name / id] of a grouping detections.csv"""
    return [dict(cand_id=c, band=r['band'], lst=r['lst'], date=r['date'], ra=float(r['ra']), dec=float(r['dec']),
                 name=r['identity_name'], id=r['identity_id']) for c, r in read_detections(table).items()]


def half_beam_arcsec(band, cfg=None):
    """half the nominal beam of a band's class (catalog_conform.NOMINAL_BEAM_ARCSEC; a subband takes its class from
    catalog.BAND_CLASS): the carry's match radius (production's deep carry: half the cell master's beam)"""
    cfg = cfg or _cfg()
    beams = cfg.catalog_conform.NOMINAL_BEAM_ARCSEC
    if band in beams:
        return 0.5 * float(beams[band])
    for cls, members in cfg.catalog.BAND_CLASS.items():
        if band in members:
            return 0.5 * float(beams[cls])
    raise StageRefusal(f'carry: band {band!r} has no class and no nominal beam - REFUSING')


class _Matcher:
    """old detection -> the new detections it is: same band, LST and date (the 10-minute date carries its window bin; a deep
    date is the night) and within half the band's nominal beam - the deep carry's rule (deep_carry_judgements.py: night and
    position within half a beam), robust to a re-search that re-numbers candidates; every new detection within the radius is
    reached, as there"""

    def __init__(self, new):
        self.by_key = collections.defaultdict(list)
        for d in new:
            self.by_key[(d['band'], d['lst'][:3], d['date'])].append(d)
        self.tol = {}

    def __call__(self, d):
        if isinstance(d, str):
            raise StageRefusal('carry: an old member without its fingerprint (band, LST, date, position) - REFUSING')
        b = d['band']
        if b not in self.tol:
            self.tol[b] = half_beam_arcsec(b)
        out = []
        for n in self.by_key.get((b, d['lst'][:3], d['date']), ()):
            dra = ((n['ra'] - d['ra'] + 180.0) % 360.0 - 180.0) * math.cos(math.radians(d['dec']))
            if math.hypot(dra, n['dec'] - d['dec']) * 3600.0 <= self.tol[b]:
                out.append(n)
        return out


_VERDICT_KEYS = ('judge_confidence', 'judge_fp_reason', 'judge_fp_reason_2')
MERGE_CONFLICTS = 'merge_conflicts.jsonl'      # = lib/serve/label_identity.MERGE_CONFLICTS


def _verdict(r):
    return (r.get('judge_confidence'), r.get('judge_fp_reason', ''), r.get('judge_fp_reason_2', ''))


def _vnorm(r):
    return tuple(r.get(k) or '' for k in _VERDICT_KEYS)


def _conflict_key(rec):
    """a merge-conflict record's identity: its page, stable ID and the verdicts it lists (time aside)"""
    return (rec.get('source_id'), rec.get('group_identity_id'),
            frozenset(_vnorm(v) + (v.get('source', ''), v.get('identity', ''), str(v.get('ts', '')))
                      for v in rec.get('verdicts', [])))


def _open_conflicts(journal, J):
    """{old page id: its newest merge-conflict record} of the labels directory of ``journal`` (merge_conflicts.jsonl, written
    by the carry and by the two-port server; lib/serve/label_identity.py) that the page's final verdict J does not settle
    (label_identity.conflict_settled); {} without the file"""
    p = os.path.join(os.path.dirname(journal), MERGE_CONFLICTS) if journal else None
    if not p or not os.path.exists(p):
        return {}
    from exopipe.phase3.lib.serve.label_identity import conflict_settled
    last = {}
    with open(p) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r.get('source_id'):
                    last[r['source_id']] = r
    return {s0: r for s0, r in last.items() if not conflict_settled(r, [J[s0]] if s0 in J else [])}


def _latest_state(path):
    """{source_id: its latest source-scope record} of a journal file (classify sets, retract clears; any writer, carried or not)
    - the state the review server shows"""
    st = {}
    if not os.path.exists(path):
        return st
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get('action') == 'retract':
                st.pop(r.get('source_id'), None)
            elif r.get('scope', 'source') == 'source' and r.get('action') == 'classify':
                st[r['source_id']] = r
    return st


def carry_judgements(journal, old_members, new, out_dir, run_id, queue_name='', carried_from_run='', on_page=None,
                     lines=None, append=False, since=0):
    """Carry the judgements of ``journal`` (judgements.jsonl, keyed by the OLD page ids) to the NEW identities by member
    detections - the existing carry rule (exopipe_ops deep_carry_judgements.py; the preview's carry_labels): ``old_members`` =
    {old page id: [detection dicts: cand_id, band, lst, date, ra, dec]} (the membership the judgements were made on, e.g.
    page_members); ``new`` = the new detections named by identity (a grouping detections.csv path, or identities_of_catalogs);
    an old detection reaches every new detection of the same band, LST and date within half the band's nominal beam. Each old
    page's judgement is its FINAL state in the journal (the lines in order: a classify sets it, a retract clears it). A new
    identity reached by judged old pages that all agree on (judge_confidence, judge_fp_reason, judge_fp_reason_2) takes the
    judgement UNCHANGED (source_id = its display name; carried_from_run / carried_from_source / shared_detections and the
    'CARRIED from' note); a disagreement is a CONFLICT - reported, NOT carried, and kept with every verdict in
    out_dir/merge_conflicts.jsonl, which the served page shows at its top marked AMBIGUOUS (lib/serve/label_identity.py
    ambiguous_html; R-GREGG-20261124-CROSS-STOKES-LINK-TEST-APPROVED: "When the merged identities carry conflicting verdicts,
    show both and mark the source ambiguous; never pick one"); an old judgement whose detections reach no new detection is
    reported. An UNSETTLED merge conflict of the old labels directory (<dirname(journal)>/merge_conflicts.jsonl, the newest
    record of an old page; label_identity.conflict_settled against that page's final verdict) is carried like a judgement:
    its verdicts join those of the old pages reaching the same new identity, so the new identity is a CONFLICT again, its
    record listing every verdict - the ambiguity lasts until a reviewer judges the page, never dropped by a rebuild.
    Writes out_dir/judgements.jsonl and out_dir/carry_report.csv (production's columns + new_identity_id).
    append=False: the SEED carry - it REFUSES a non-empty journal (the carry only seeds).
    append=True (review 2026-11-04, F1): ``lines`` is the FULL old journal and ``since`` the number of its lines the seed carry
    read. Every new identity an old page of the lines after ``since`` reaches is RE-DERIVED from the whole journal - the final
    state of EVERY judged old page reaching it, seed pages included - and compared with what the new journal shows now:
    one verdict -> a carried classify (unless it already holds exactly that verdict, those old pages and that note); a disagreement -> a CONFLICT row and, when the
    identity holds a carried judgement, a retract (never an override: the outcome does not depend on when a label was made);
    no judged page left (all retracted) -> a retract of its carried judgement. An identity judged directly on the new page (a
    record without carried_from_run) is never overridden ('not carried: judged on the new page'). Returns the counts."""
    newdets = table_detections(new) if isinstance(new, str) else list(new)
    match = _Matcher(newdets)
    raw = lines if lines is not None else open(journal).read().splitlines()
    J = read_journal(None, lines=raw)
    old_conf = _open_conflicts(journal, J)               # old page -> its unsettled merge-conflict record
    touched_pages = set()
    if append:
        for ln in raw[since:]:
            if ln.strip():
                sid = json.loads(ln).get('source_id')
                if sid:
                    touched_pages.add(sid)
        touched_pages |= set(old_conf)                   # re-derived each time (an identical record is not written twice)
    reach = {}                                           # old page -> {(name, id): shared detections}
    for s0 in sorted(set(J) | touched_pages | set(old_conf)):
        cnt = collections.Counter()
        for d in old_members.get(s0, []):
            for n in {(x['name'], x['id']) for x in match(d)}:
                cnt[n] += 1
        reach[s0] = cnt
    owner = collections.defaultdict(set)                 # new identity -> the judged old pages (final state) reaching it
    for s0 in sorted(set(J) | set(old_conf)):
        for n in reach[s0]:
            owner[n].add(s0)
    ndet = collections.Counter((d['name'], d['id']) for d in newdets)
    now = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())

    def carried_record(name, iid, olds, share, shared):
        s0 = max(sorted(o for o in olds if o in J), key=lambda s: share[s])
        r = dict(J[s0])
        # presentation p1, as production's carry tools write it (deep_carry_judgements.py:106, snapshot_carry_judgements.py:158):
        # the review server keys a page's state by (source_id, presentation_id), and the new queue presents a carried identity
        # first as p1 - so the carried judgement shows there, and a retract (which names no presentation: p1) clears it
        r.update(source_id=name, run_id=run_id, presentation_id='p1', ts=now, carried_from_run=carried_from_run,
                 carried_from_source=';'.join(sorted(olds)), shared_detections=shared,
                 new_source_detections=ndet[(name, iid)], group_identity_id=iid,
                 judge_note=('CARRIED from %s %s (%s matched by window and position within half a beam; grouping identity %s '
                             'with %d detections)' % (carried_from_run, ','.join(sorted(olds)), shared, iid, ndet[(name, iid)])
                             + (' | original note: %s' % J[s0]['judge_note'] if J[s0].get('judge_note') else '')))
        if queue_name:
            r['queue_name'] = queue_name
        for k in ('src_ra', 'src_dec'):
            r.pop(k, None)
        return r

    def retract_record(name, iid, why):
        return dict(action='retract', scope='source', source_id=name, presentation_id='p1', run_id=run_id, ts=now,
                    group_identity_id=iid, retract_carried_from_run=carried_from_run, retract_reason=why)

    recs, rows, conflicts = [], [], []
    os.makedirs(out_dir, exist_ok=True)
    jp, rp = os.path.join(out_dir, 'judgements.jsonl'), os.path.join(out_dir, 'carry_report.csv')
    cnt = collections.Counter()
    if not append:
        if os.path.exists(jp) and os.path.getsize(jp) > 0:
            raise StageRefusal(f'{jp} exists and is not empty - the carry only seeds an empty journal - REFUSING')
        targets = sorted(owner)
        unmatched = [s0 for s0 in sorted(J) if not reach[s0]]
        cur = {}
    else:
        targets = sorted({n for p in touched_pages for n in reach.get(p, ())})
        cur = _latest_state(jp)
        done = set()
        if os.path.exists(rp):
            with open(rp, newline='') as f:
                done = {r['old_sources'] for r in csv.DictReader(f) if r['outcome'].startswith('old judgement')}
        unmatched = [s0 for s0 in sorted(touched_pages) if s0 in J and not reach[s0] and s0 not in done]
    lost_conf = [s0 for s0 in sorted(old_conf) if not reach[s0]]
    have_conf = set()
    if append and os.path.exists(os.path.join(out_dir, MERGE_CONFLICTS)):
        with open(os.path.join(out_dir, MERGE_CONFLICTS)) as f:
            have_conf = {_conflict_key(json.loads(x)) for x in f if x.strip()}
    for (name, iid) in targets:
        olds = owner.get((name, iid), set())
        share = {s0: reach[s0][(name, iid)] for s0 in olds}
        shared = ';'.join('%s:%d/%d' % (s0, k, len(old_members.get(s0, []))) for s0, k in sorted(share.items()))
        verdicts = {_verdict(J[s]) for s in olds if s in J}
        prior = [v for s in sorted(olds) if s in old_conf for v in old_conf[s].get('verdicts', [])]
        kinds = {_vnorm(J[s]) for s in olds if s in J} | {_vnorm(v) for v in prior}
        base = dict(new_source=name, new_identity_id=iid, n_detections=ndet[(name, iid)], old_sources=shared)
        now_rec = cur.get(name)
        if now_rec is not None and 'carried_from_run' not in now_rec:     # judged on the new page: never overridden
            rows.append(dict(base, outcome='not carried: judged on the new page', verdicts='/'.join(
                str(x) for x in _verdict(now_rec))))
            cnt['own'] += 1
            continue
        if len(verdicts) > 1 or (prior and len(kinds) > 1):
            rows.append(dict(base, outcome='CONFLICT - not carried',
                             verdicts='|'.join('/'.join(str(x) for x in v) for v in sorted(kinds if prior else verdicts,
                                                                                               key=str))))
            vs, seen = [], set()
            for v in [dict({k: J[s0].get(k, '') or '' for k in _VERDICT_KEYS + ('judge_note',)}, source=s0,
                           identity=J[s0].get('group_identity_id', ''), ts=J[s0].get('ts', '')) for s0 in sorted(olds)
                      if s0 in J] + [dict(v) for v in prior]:
                k = _vnorm(v) + (v.get('source', ''), v.get('identity', ''), str(v.get('ts', '')))
                if k not in seen:
                    seen.add(k)
                    vs.append(v)
            rec = dict(ts=now, origin='carry', source_id=name, group_identity_id=iid, run_id=run_id,
                       carried_from_run=carried_from_run, verdicts=vs)
            if prior:
                rec['carried_conflict_from'] = ';'.join(sorted(s0 for s0 in olds if s0 in old_conf))
            if _conflict_key(rec) not in have_conf:
                conflicts.append(rec)
                have_conf.add(_conflict_key(rec))
            if now_rec is not None:                     # a carried judgement the old pages no longer agree on
                recs.append(retract_record(name, iid, 'CONFLICT: the old pages reaching this identity now disagree'))
                cnt['retracted'] += 1
            cnt['conflicts'] += 1
            continue
        if not verdicts:                                 # every judged old page reaching it was retracted
            if now_rec is not None:
                recs.append(retract_record(name, iid, 'the old pages reaching this identity retracted their judgements'))
                rows.append(dict(base, outcome='retracted: its old pages retracted their judgements', verdicts=''))
                cnt['retracted'] += 1
            continue
        r = carried_record(name, iid, olds, share, shared)
        # 'unchanged' compares the note too (F1 residual, CUTOVER_READINESS 2026-11-04: "a changed note with an unchanged
        # verdict between carry and swap is dropped"): the carried note quotes the old page's note, so a note edited in the
        # window re-carries the identity
        if now_rec is not None and _verdict(now_rec) == _verdict(r) and \
                now_rec.get('carried_from_source') == r['carried_from_source'] and \
                now_rec.get('judge_note', '') == r.get('judge_note', ''):
            cnt['unchanged'] += 1
            continue
        recs.append(r)
        rows.append(dict(base, outcome='carried', verdicts='/'.join(str(x) for x in _verdict(r))))
        cnt['carried'] += 1
    for s0 in unmatched:
        rows.append(dict(new_source='', new_identity_id='', n_detections=0, old_sources=s0,
                         outcome='old judgement reached no new detection', verdicts=J[s0].get('judge_confidence', '')))
    for s0 in lost_conf:
        rows.append(dict(new_source='', new_identity_id='', n_detections=0, old_sources=s0,
                         outcome='old merge conflict reached no new detection',
                         verdicts='|'.join('/'.join(_vnorm(v)) for v in old_conf[s0].get('verdicts', []))))
    with open(jp, 'a' if append else 'w') as f:
        for r in recs:
            f.write(json.dumps(r) + '\n')
    if conflicts or not append:
        with open(os.path.join(out_dir, MERGE_CONFLICTS), 'a' if append else 'w') as f:
            for r in conflicts:
                f.write(json.dumps(r, sort_keys=True) + '\n')
    if append and os.path.exists(rp):
        with open(rp, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=list(CARRY_COLS)).writerows(rows)
    else:
        _write_csv(rp, CARRY_COLS, rows)
    return dict(judged_old=len(J), carried=cnt['carried'], conflicts=cnt['conflicts'], unmatched=len(unmatched),
                new_identities_touched=len(targets), retracted=cnt['retracted'], unchanged=cnt['unchanged'],
                judged_on_new_page=cnt['own'], lines=len(raw), between_lines=(len(raw) - since) if append else 0,
                ambiguous_kept=len(conflicts), conflicts_carried=sum(1 for r in conflicts if r.get('carried_conflict_from')),
                conflicts_lost=len(lost_conf),
                carried_on_page=(sum(1 for r in recs if r.get('action') != 'retract' and r['source_id'] in on_page)
                                 if on_page is not None else None))


# =====================================================================================================================
# `exopipe3 catalog --tier sublst --stages group` (the 10-minute cumulative's grouping)
# =====================================================================================================================

def add_arguments(p):
    p.add_argument('--group-input', metavar='DIR',
                   help='--stages group, tier sublst: group DIR/master_catalog.csv + DIR/pertie_detections.csv (e.g. a frozen '
                        'cumulative) instead of the published LST runs of --lsts')
    p.add_argument('--group-new-state', action='store_true',
                   help='the first grouping run of a tier and Stokes: start the stable IDs (refused where a state exists; '
                        'without it a missing state refuses - IDs never restart at 1 silently)')
    p.add_argument('--group-mode', choices=('batch', 'incremental', 'regroup'),
                   help="grouping.group_catalog mode: default 'batch' for a new state, else 'incremental' (pure appends; a "
                        "re-searched window refuses, naming 'regroup'); 'regroup' recomputes the partition and carries the IDs")
    p.add_argument('--group-omission', metavar='DIR|CSV',
                   help='tier sublst (required): the deep-omission decision table the grouping input is filtered by - the '
                        '10-minute omit stage\'s --out (`--tier sublst --stages omit --lsts <every grouped LST> --out DIR`: '
                        'DIR/deep_omission.csv) or the table itself; production\'s order, decision table then drop then '
                        'identities (sublst_cumulative.sh:99-107)')
    p.add_argument('--group-stokes', default='I,V', metavar='I,V', help='Stokes to group (default both)')
    p.add_argument('--group-lst-dir', action='append', default=[], metavar='LST=DIR',
                   help='tier sublst: take this LST\'s detections from its CATALOGUE directory DIR (master_catalog.csv, '
                        'pertie/pertie_detections.csv) instead of its published run - the LST being catalogued, grouped '
                        'before its build stage (repeatable)')
    p.add_argument('--group-night-dir', action='append', default=[], metavar='LST=DIR',
                   help='tier sublst, the per-hour catalog build: ALSO take the detections of a night appended to this LST '
                        'since its all-nights build, from the night\'s catalog directory DIR (master_catalog.csv, pertie/'
                        'pertie_detections.csv); repeatable, in the order the nights were appended (the newest last)')


def wants(args):
    st = getattr(args, 'stages', None)
    return (bool(st) and [s.strip() for s in str(st).split(',') if s.strip()] == [STAGE]
            and getattr(args, 'tier', None) == 'sublst')


def default_out(args):
    return sublst_dir()


def lst_run(lst, cfg=None):
    """The published LST run of the cumulative (catalog_omission.TENMIN_RUN under paths.TRANSIENTS_ROOT, the served arm)."""
    cfg = cfg or _cfg()
    return os.path.join(cfg.paths.TRANSIENTS_ROOT, cfg.catalog_omission.TENMIN_RUN.format(hh=lst[:2], arm=cfg.catalog_sublst.ARM))


def _pertie_of(d):
    for p in (os.path.join(d, 'pertie_detections.csv'), os.path.join(d, 'pertie', 'pertie_detections.csv')):
        if os.path.exists(p):
            return p
    return os.path.join(d, 'pertie_detections.csv')


def assemble_tenmin_input(lsts, out_dir, cfg=None, lst_dirs=None, night_dirs=None, omission=None):
    """The 10-minute grouping input: the master_catalog.csv and pertie_detections.csv rows of every published LST run of
    ``lsts`` (header union, LST order, the run's row order) AFTER the deep omission ``omission`` (an Omission, omission_of):
    the rows of its OMIT cells are dropped per LST before the concatenation, as production's merge drops them before its
    identities (sublst_cumulative.sh:99-107; the build drops the same rows from its input). Writes
    out_dir/grouping_input/{master_catalog,pertie_detections}.csv and the dropped rows' record omission_dropped.csv; returns
    (their paths, {lst: run dir}); omission.dropped counts the rows dropped per file. No omission, a run without either
    file, an LST the decision table does not hold, or a cand_id in two runs, REFUSES.
    night_dirs (the per-hour catalog build): [(LST, directory)], the catalog directories of the nights appended since the LSTs'
    last all-nights builds - each one's rows follow its LST's run, in the order given, so the earlier nights' rows keep their
    places and the grouping's incremental mode sees only the newest night's detections as new. A night row whose cand_id
    the LST's run (or an earlier night) already holds with the same content is the same detection (kept once); with another
    content it REFUSES, as two runs do."""
    if omission is None:
        raise StageRefusal('S-G28 no deep-omission decision table: the 10-minute grouping holds the detections left after the '
                           'deep omission (--group-omission <the omit stage\'s --out>) - REFUSING, never "no omission"')
    bad = sorted(set(lsts) - set(omission.lsts))
    if bad:
        raise StageRefusal(f'S-G28 the deep-omission decision table {omission.path} holds no decision of {bad} (its LSTs: '
                           f'{", ".join(omission.lsts)}): run the omit stage over every grouped LST - REFUSING')
    d = os.path.join(out_dir, 'grouping_input')
    os.makedirs(d, exist_ok=True)
    lst_dirs = dict(lst_dirs or {})
    bad = sorted(set(lst_dirs) - set(lsts))
    if bad:
        raise StageRefusal(f'--group-lst-dir names LST(s) {bad} outside --lsts {lsts} - REFUSING')
    nights = [(L, os.path.abspath(D)) for L, D in (night_dirs or [])]
    bad = sorted({L for L, _ in nights} - set(lsts))
    if bad:
        raise StageRefusal(f'--group-night-dir names LST(s) {bad} outside --lsts {lsts} - REFUSING')
    if len(set(nights)) != len(nights):
        raise StageRefusal('--group-night-dir names a night directory twice - REFUSING')
    runs = {L: lst_dirs.get(L) or lst_run(L, cfg) for L in lsts}
    srcs = [x for L, R in runs.items() for x in [(L, R, L)] + [(L2, D, f'{L2} night directory {D}') for L2, D in nights if L2 == L]]
    out, gone = [], []
    for name in ('master_catalog.csv', 'pertie_detections.csv'):
        hdr, rows, seen = [], [], {}
        for L, R, what in srcs:
            p = os.path.join(R, name) if name == 'master_catalog.csv' else _pertie_of(R)
            if not os.path.exists(p):
                raise StageRefusal(f'{what}: no {p} - ' + ('publish the LST before grouping the cumulative' if what == L else
                                                           'a night directory holds the night\'s catalog and window-tie rows'))
            with open(p, newline='') as f:
                rd = csv.DictReader(f)
                hdr += [k for k in (rd.fieldnames or []) if k not in hdr]
                for r in rd:
                    c = r.get('cand_id')
                    k = _omitted(omission, c, r.get('stokes'), L)
                    if k:                         # production's filter_rows: the run's rows of an OMIT cell go
                        gone.append(dict(file=name, source=p, cand_id=c, lst=k[0], band=k[1], night=k[2], stokes=k[3]))
                        continue
                    if c in seen:
                        if seen[c] != (L, r):
                            raise StageRefusal(f'{name}: cand_id {c} in {seen[c][0]} and {what} - REFUSING')
                        continue
                    seen[c] = (L, r)
                    rows.append(r)
        out.append(_write_csv(os.path.join(d, name), hdr, rows))
    _record_dropped(d, gone, omission)
    return out[0], out[1], runs


OMISSION_DROPPED = 'omission_dropped.csv'      # grouping_input/: the rows the deep omission dropped from the grouping input
OMISSION_DROPPED_COLS = ['file', 'source', 'cand_id', 'lst', 'band', 'night', 'stokes']


def _omitted(omission, cand_id, stokes, lst):
    try:
        return omission.omitted(cand_id, stokes, lst)
    except _bg().Refused as e:
        raise StageRefusal(f'S-G28 {e}')


def _record_dropped(d, gone, omission):
    _write_csv(os.path.join(d, OMISSION_DROPPED), OMISSION_DROPPED_COLS, gone)
    omission.dropped = dict(collections.Counter(g['file'] for g in gone))
    return omission.dropped


def _bg():
    """wraps/builder_grouping (stdlib only): the ONE reading of the deep-omission decision table (Omission), the builder's
    and the grouping's."""
    from exopipe.phase3.lib.catalog.wraps import builder_grouping
    return builder_grouping


def omission_of(path):
    """The 10-minute deep-omission decision table named by --group-omission (the omit stage's --out directory: its
    deep_omission.csv; or the table's own path) as an Omission (one read: sha256 and verdicts from the same bytes). A missing
    or unreadable table, or one without the columns lst, band, night, stokes, verdict or without a row, REFUSES (S-G28)."""
    if not path:
        raise StageRefusal('S-G28 no deep-omission decision table: the 10-minute grouping needs --group-omission <the omit '
                           'stage\'s --out> (`exopipe3 catalog --tier sublst --stages omit --lsts <every grouped LST> --out '
                           'DIR`) - REFUSING, never "no omission"')
    p = os.path.abspath(path)
    if os.path.isdir(p):
        p = os.path.join(p, OMISSION_TABLE)
    try:
        return _bg().Omission(p)
    except _bg().Refused as e:
        raise StageRefusal(f'S-G28 {e}')


OMISSION_TABLE = 'deep_omission.csv'           # the omit stage's decision table in its --out (deep_omission.run_tenmin)


def filter_input(src, out_dir, omission):
    """--group-input DIR under the deep omission: DIR's master_catalog.csv and pertie_detections.csv without the rows of
    the OMIT cells (each row's LST from its own lst column) into out_dir/grouping_input/; returns their paths. A row LST the
    table does not hold REFUSES (S-G28)."""
    d = os.path.join(out_dir, 'grouping_input')
    os.makedirs(d, exist_ok=True)
    out, gone = [], []
    for name in ('master_catalog.csv', 'pertie_detections.csv'):
        p = os.path.join(src, name)
        if not os.path.exists(p):
            raise StageRefusal(f'--group-input {src}: no {name} - REFUSING')
        with open(p, newline='') as f:
            rd = csv.DictReader(f)
            hdr, rows = list(rd.fieldnames or []), []
            for r in rd:
                L = r.get('lst') or ''
                if L not in omission.lsts:
                    raise StageRefusal(f'S-G28 the deep-omission decision table {omission.path} holds no decision of {L!r} '
                                       f'({r.get("cand_id")} of {p}) - REFUSING')
                k = _omitted(omission, r.get('cand_id'), r.get('stokes'), L)
                if k:
                    gone.append(dict(file=name, source=p, cand_id=r.get('cand_id'), lst=k[0], band=k[1], night=k[2],
                                     stokes=k[3]))
                else:
                    rows.append(r)
        out.append(_write_csv(os.path.join(d, name), hdr, rows))
    _record_dropped(d, gone, omission)
    return out[0], out[1]


def recorded_omission(grouping_dir, stokes, _rec=None):
    """{path, sha256} of the deep-omission decision table the LATEST 10-minute grouping run of Stokes ``stokes`` in
    grouping_dir recorded (meta 'omission'): what every build of that grouping drops. A latest run that is not done, one that
    recorded no table (grouped before the grouping followed the omission), and a table missing or changed since REFUSE
    (StageRefusal) - a build never drops by another table than its grouping's, and never drops nothing for want of one."""
    fin, p, r = _rec or _latest_group_record(grouping_dir, stokes)
    if r.get('status') != 'done':
        raise StageRefusal(f'the latest 10-minute grouping run of Stokes {stokes} ({os.path.basename(p)}, {fin}) ended '
                           f'{r.get("status")} - group again before building - REFUSING')
    om = (r.get('meta') or {}).get('omission') or {}
    if not om.get('path') or not om.get('sha256'):
        raise StageRefusal(f'S-G28 the latest grouping run of Stokes {stokes} ({os.path.basename(p)}) recorded no deep-omission '
                           f'decision table (it grouped before the grouping followed the omission): group again with '
                           f'--group-omission - REFUSING')
    try:
        with open(om['path'], 'rb') as f:
            now = hashlib.sha256(f.read()).hexdigest()
    except OSError as e:
        raise StageRefusal(f'S-G28 the deep-omission decision table {om["path"]} that {os.path.basename(p)} recorded is missing '
                           f'or unreadable ({e}) - REFUSING')
    if now != om['sha256']:
        raise StageRefusal(f'S-G28 the deep-omission decision table {om["path"]} changed since {os.path.basename(p)} recorded it '
                           f'(sha256 {now[:12]} vs {om["sha256"][:12]}) - group again - REFUSING')
    return dict(path=om['path'], sha256=om['sha256'])


def run(args, run):
    """`exopipe3 catalog --tier sublst --stages group`: one unit '<label>.group' grouping each Stokes of the cumulative - the
    published LST runs of --lsts (default: every LST), or --group-input DIR."""
    m = mode()
    label = getattr(args, 'run_label', None) or 'Run_SLSTB'
    out = os.path.abspath(run.out_root)
    src = getattr(args, 'group_input', None)
    stokes = [s.strip() for s in (getattr(args, 'group_stokes', None) or 'I,V').split(',') if s.strip()]
    from exopipe.phase3.core import cells
    raw = getattr(args, 'lsts', None)
    lsts = [cells.parse_lst(x) for x in cells.expand(raw or ','.join(cells.LSTS), universe=cells.LSTS, kind='LST')]
    if run.dry_run:
        print(f'[{label}.{STAGE}] tier sublst: grouping {m!r}; input '
              + (src if src else f'the published LST runs {lsts[0]}..{lsts[-1]} ({len(lsts)})')
              + f' after the deep omission {getattr(args, "group_omission", None) or "(none named: the unit refuses, S-G28)"}'
              + f'; out {out} (state in the same directory); Stokes {",".join(stokes)}')
        return 0
    from exopipe.phase3.core.record import NoInput
    with run.unit(f'{label}.{STAGE}', tier='sublst', stage=STAGE) as u:
        u.meta['mode'] = m.as_dict()
        u.meta['stokes'] = stokes
        if not m.design_b:
            raise NoInput("catalog_grouping.GROUPING = 'production': production groups per LST in the builder and joins LSTs "
                          "by J-name (the parity file) - no grouping stage")
        lock = _Lock(os.path.join(out, UNIT_LOCK), 'the 10-minute grouping')
        try:
            lock.__enter__()                  # the whole unit: one grouping of the shared 10-minute state at a time (F2)
        except StageRefusal as e:
            u.refuse(str(e))
        try:
            _run_tenmin_unit(args, u, m, label, out, src, stokes, lsts, cells)
        finally:
            lock.__exit__()
    return None


def _run_tenmin_unit(args, u, m, label, out, src, stokes, lsts, cells):
    """the body of the 10-minute group unit (run holds the unit lock)"""
    try:
        omission = omission_of(getattr(args, 'group_omission', None))
        u.add_input(omission.path)
        if src:
            for n in ('master_catalog.csv', 'pertie_detections.csv'):
                u.add_input(os.path.join(src, n))
            mc, pt = filter_input(src, out, omission)
            u.meta['input'] = src
        else:
            ld = {}
            for x in getattr(args, 'group_lst_dir', None) or []:
                if '=' not in x:
                    raise StageRefusal(f'--group-lst-dir {x!r}: LST=DIR - REFUSING')
                k, v = x.split('=', 1)
                ld[cells.parse_lst(k)] = os.path.abspath(v)
            nd = []
            for x in getattr(args, 'group_night_dir', None) or []:
                if '=' not in x:
                    raise StageRefusal(f'--group-night-dir {x!r}: LST=DIR - REFUSING')
                k, v = x.split('=', 1)
                nd.append((cells.parse_lst(k), os.path.abspath(v)))
            mc, pt, runs = assemble_tenmin_input(lsts, out, lst_dirs=ld, night_dirs=nd, omission=omission)
            u.meta['input'] = {L: R for L, R in runs.items()}
            if nd:
                u.meta['input_nights'] = [[L, D] for L, D in nd]
            for R in list(runs.values()) + [D for _, D in nd]:
                u.add_input(os.path.join(R, 'master_catalog.csv'))
                u.add_input(_pertie_of(R))
    except StageRefusal as e:
        u.refuse(str(e))
    # what every build of this grouping drops (recorded_omission): the decision table, read once above
    u.meta['omission'] = dict(path=omission.path, sha256=omission.sha256, decisions=omission.decisions,
                              omit_keys=len(omission.keys), lsts=omission.lsts, dropped=omission.dropped,
                              record=os.path.join(out, 'grouping_input', OMISSION_DROPPED))
    u.note(f'deep omission: {omission.path} (sha256 {omission.sha256[:16]}; {len(omission.keys)} OMIT keys of '
           f'{omission.decisions} decisions): dropped ' + (', '.join(f'{n} {k}' for k, n in sorted(omission.dropped.items()))
                                                          or 'no row'))
    for p in (mc, pt):
        if not os.path.exists(p):
            u.refuse(f'grouping input {p} is missing')
        u.add_input(p)
    grouped = {}
    for S in stokes:
        if S not in ('I', 'V'):
            u.refuse(f'--group-stokes {S}: I and/or V')
    order, reserved = naming_order(stokes)
    for S in order:
        from exopipe.phase3.lib.catalog import tests_grouping as TG
        try:
            det = grouped[S] = load_det('10min', S, mc, pt)
            paths, summ, _res = run_group(det, '10min', S, out, out, f'{label}.{STAGE}', dict(master_catalog=mc, pertie=pt),
                                          group_mode=getattr(args, 'group_mode', None),
                                          new_state=bool(getattr(args, 'group_new_state', False)),
                                          check=lambda pp, S=S: TG.failures(pp, tier='10min', stokes=S, input=mc),
                                          reserved_names=reserved.get(S))
        except (StageRefusal, G.GroupingRefused) as e:
            u.refuse(f'{S}: {e}')
        u.meta.setdefault('summary', {})[S] = summ['counts']
        u.add_output(paths['detections'], 'grouping_table', tier='10min', stokes=S, input=mc)
        u.add_output(paths['state'], 'grouping_state', tier='10min', stokes=S, table=paths['detections'])
        # what the LST builds check (recorded_table): this run's table and its sha1, recorded only on success
        u.meta.setdefault('tables', {})[S] = dict(path=paths['detections'], sha1=paths['detections_sha1'])
        u.note(f"{S}: {summ['counts']['detections']} detections -> {summ['counts']['identities']} identities, "
               f"{summ['counts']['multiples']} multiples ({summ['counts']['identities_in_multiples']} identities); "
               f"table sha1 {paths['detections_sha1'][:12]}")
    link_unit(u, out, '10min', grouped)


def link_unit(u, out, tier, have=None):
    """the cross-Stokes link of the groupings under out (lib/catalog/stokes_link.py; R-GREGG-20261124-CROSS-STOKES-LINK-TEST-
    APPROVED), after any grouping run: both Stokes' present tables (the one grouped now and the other as it stands) -> out/
    stokes_link/; a refusal refuses the unit with gate S-G27 (the readers merge on the link: a stale link must not stand
    beside a new table; a shared display name, a broken merge chain, a missing state or table). have: {S: det} the unit just
    grouped (the very detections; the other Stokes' are loaded from its grouping summary's inputs). With one Stokes grouped
    there is nothing to link: (None, None), noted (the readers ask for a link only when both are)."""
    from exopipe.phase3.lib.catalog import stokes_link as SL
    one = SL.grouped_stokes(out) | set(have or ())
    if one != {'I', 'V'}:                       # one Stokes grouped so far: nothing to link (the readers need no link then)
        u.meta['stokes_link'] = None
        u.note(f'cross-Stokes link: only Stokes {sorted(one)} grouped under {out} - nothing to link until the other is grouped')
        return None, None
    try:
        paths, counts = SL.run_link(SL.dets_of_tables(out, tier, have), out, tier)
    except SL.LinkRefused as e:
        u.refuse(f'S-G27 cross-Stokes link: {e}')
    u.meta['stokes_link'] = counts
    u.add_output(paths['links'], 'stokes_link', tier=tier)
    u.note(f"cross-Stokes link: {counts['linked_pairs']} linked I/V pairs in {counts['components']} components "
           f"({counts['ambiguous_components']} ambiguous: one-to-many) of {counts['identities_I']} I and "
           f"{counts['identities_V']} V identities")
    return paths, counts
