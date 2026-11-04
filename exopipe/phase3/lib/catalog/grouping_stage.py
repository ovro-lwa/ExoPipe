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
  10-minute  `exopipe3 catalog --tier sublst --stages group [--lsts ...]` (one unit '<label>.group'): every detection of the
             published LST runs (their master_catalog.csv + pertie_detections.csv, concatenated: what every LST's build sees -
             BEFORE the cumulative merge's deep omission, which the merge applies as production does), per Stokes, into
             catalog_grouping.SUBLST_DIR (or --out; the state beside it); --group-input DIR groups DIR/master_catalog.csv +
             DIR/pertie_detections.csv instead (e.g. a frozen cumulative). Every LST's build stage then reads the grouping and
             refuses a selected row it does not hold (S-G18): an LST re-catalogued since the grouping is grouped again first.
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
import json
import os
import re
import shutil
import time

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
CARRY_COLS = ('new_identity', 'new_identity_id', 'n_detections', 'old_identities', 'outcome', 'verdicts')
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
    tmp = f'{path}.tmp.{os.getpid()}'
    with open(tmp, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(cols))
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)
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


def write_products(det, res, tier, stokes, out_dir, inputs, mode_):
    """Write grouping_<S>/ under out_dir (moved aside by the caller when it existed): the three tables, summary.json and this
    run's events. Returns {name: path}."""
    d = group_dir(out_dir, stokes)
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
    with open(p + '.tmp', 'w') as f:
        json.dump(summ, f, indent=1, sort_keys=True, default=str)
    os.replace(p + '.tmp', p)
    paths['summary'] = p
    return paths, summ


def state_path(state_dir, tier, stokes):
    return G.GroupingState(tier, stokes).path(state_dir)


def history_path(state_dir, tier, stokes):
    return os.path.join(state_dir, f'grouping_{tier}_{stokes}_events.jsonl')


def run_group(det, tier, stokes, out_dir, state_dir, run_label, inputs, group_mode=None, new_state=False, check=None):
    """Group det (one tier and Stokes) against the saved state, write the products, run ``check`` (a callable (paths, state
    copy) -> [failures]: the content tests) and only then save the state (atomic) and append this run's events to the state's
    history. Returns (paths, summary, result). A gate refuses before anything is saved."""
    m = mode()
    if not m.design_b:
        raise StageRefusal('grouping stage under catalog_grouping.GROUPING = production - nothing to do (the caller ends no-input)')
    try:
        st = G.GroupingState.load(state_dir, tier, stokes, new=new_state)
    except G.GroupingRefused as e:
        raise StageRefusal(str(e)) from None
    try:
        res = G.group_catalog(det, tier, stokes, state=st, run_label=run_label, mode=group_mode)
    except G.GroupingRefused as e:
        raise StageRefusal(str(e)) from None
    paths, summ = write_products(det, res, tier, stokes, out_dir, inputs, m)
    # the state this run hands on, written BESIDE the products first (the content test reads it), then saved
    tmpd = os.path.join(group_dir(out_dir, stokes), 'state_next')
    os.makedirs(tmpd, exist_ok=True)
    paths['state_next'] = res.state.save(tmpd)
    prev = state_path(state_dir, tier, stokes)
    if os.path.exists(prev):
        os.makedirs(os.path.join(group_dir(out_dir, stokes), 'state_previous'), exist_ok=True)
        paths['state_previous'] = shutil.copy2(prev, os.path.join(group_dir(out_dir, stokes), 'state_previous'))
    if check is not None:
        bad = check(paths)
        if bad:
            raise StageRefusal('grouping products failed their content tests: ' + '; '.join(bad[:6]) + ' - the state is NOT '
                               'advanced - REFUSING')
    paths['state'] = res.state.save(state_dir)
    os.makedirs(state_dir, exist_ok=True)
    hp = history_path(state_dir, tier, stokes)
    G.append_events(hp, res)
    paths['history'] = hp
    return paths, summ, res


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


def builder_env(table, tier, mode_=None):
    """The environment of the runner: the grouping table of the key's Stokes, the tier, the repeat rungs."""
    m = mode_ or mode()
    return {'EXOPIPE3_GROUPING_MODE': 'design-b', 'EXOPIPE3_GROUPING_TABLE': os.path.abspath(table),
            'EXOPIPE3_GROUPING_TIER': TIER_OF.get(tier, tier), 'EXOPIPE3_GROUPING_RUNGS': m.rungs}


def builder_line(text):
    """The runner's machine line ('GROUPING-BUILDER ...') from a log tail, or ''."""
    for ln in reversed((text or '').splitlines() if isinstance(text, str) else list(text or ())):
        if ln.startswith('GROUPING-BUILDER '):
            return ln
    return ''


def covers(table_path, selected_csv):
    """[problems]: rows of the builder's input (a selected_<K>.csv) the grouping table does not hold, or holds with another
    fingerprint (band, LST, date, catalogue ra / dec / S/N) - a stale grouping (the builder runner refuses the same)."""
    T = read_detections(table_path)
    bad = []
    with open(selected_csv, newline='') as f:
        for r in csv.DictReader(f):
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

def read_journal(path):
    """{source_id: the latest 'classify' record of scope 'source'} of a judgements.jsonl (a 'retract' removes it) - the
    journal's semantics (the review server's)."""
    J = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get('action') == 'retract':
                J.pop(r.get('source_id'), None)
            elif r.get('scope', 'source') == 'source' and r.get('action') == 'classify':
                J[r['source_id']] = r
    return J


def members_of_catalogs(cat_dirs, stokes=None):
    """{old page id (source_id): [cand_id, ...]} from the *_source_members.csv files of the given catalogue directories (the
    previous build the judgements were made on)."""
    out = collections.defaultdict(list)
    for d in cat_dirs:
        for p in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            if p.endswith('_source_members.csv'):
                with open(os.path.join(d, p), newline='') as f:
                    for r in csv.DictReader(f):
                        if r['cand_id'] not in out[r['source_id']]:
                            out[r['source_id']].append(r['cand_id'])
    return dict(out)


def carry_judgements(journal, old_members, table_path, out_dir, run_id, queue_name='', carried_from_run='', on_page=None):
    """Carry the judgements of ``journal`` (judgements.jsonl, keyed by the OLD page ids) to the identities of the grouping
    ``table_path`` by member detections: ``old_members`` = {old page id: [cand_id]} (the membership the judgements were made
    on). A new identity reached by judged old pages that all agree on (judge_confidence, judge_fp_reason, judge_fp_reason_2)
    takes the judgement UNCHANGED (source_id = the identity's display name; carried_* fields name the old pages and the shared
    detections); a disagreement is a CONFLICT - reported, not carried; an old judgement whose detections reach no identity is
    reported. Writes out_dir/judgements.jsonl (REFUSES to write over a non-empty journal: the carry only seeds) and
    out_dir/carry_report.csv. Returns the counts."""
    T = read_detections(table_path)
    name_of = {c: (r['identity_name'], r['identity_id']) for c, r in T.items()}
    J = read_journal(journal)
    owner, unmatched = collections.defaultdict(set), []
    for s0 in sorted(J):
        hit = 0
        for c in old_members.get(s0, []):
            if c in name_of:
                owner[name_of[c]].add(s0)
                hit += 1
        if not hit:
            unmatched.append(s0)
    ndet = collections.Counter(name_of.values())
    recs, rows = [], []
    now = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    for (name, iid), olds in sorted(owner.items()):
        share = {s0: sum(1 for c in old_members[s0] if name_of.get(c) == (name, iid)) for s0 in olds}
        shared = ';'.join('%s:%d/%d' % (s0, k, len(old_members[s0])) for s0, k in sorted(share.items()))
        verdicts = {(J[s].get('judge_confidence'), J[s].get('judge_fp_reason', ''), J[s].get('judge_fp_reason_2', ''))
                    for s in olds}
        base = dict(new_identity=name, new_identity_id=iid, n_detections=ndet[(name, iid)], old_identities=shared)
        if len(verdicts) > 1:
            rows.append(dict(base, outcome='CONFLICT - not carried',
                             verdicts='|'.join('/'.join(str(x) for x in v) for v in sorted(verdicts, key=str))))
            continue
        s0 = max(sorted(olds), key=lambda s: share[s])
        r = dict(J[s0])
        r.update(source_id=name, run_id=run_id, ts=now, carried_from_run=carried_from_run,
                 carried_from_source=';'.join(sorted(olds)), shared_detections=shared,
                 new_source_detections=ndet[(name, iid)], group_identity_id=iid,
                 judge_note=('CARRIED from %s (%s shared detections; grouping identity %s with %d detections)'
                             % (','.join(sorted(olds)), shared, iid, ndet[(name, iid)])
                             + (' | original note: %s' % J[s0]['judge_note'] if J[s0].get('judge_note') else '')))
        if queue_name:
            r['queue_name'] = queue_name
        for k in ('src_ra', 'src_dec'):
            r.pop(k, None)
        recs.append(r)
        rows.append(dict(base, outcome='carried', verdicts='/'.join(str(x) for x in next(iter(verdicts)))))
    for s0 in unmatched:
        rows.append(dict(new_identity='', new_identity_id='', n_detections=0, old_identities=s0,
                         outcome='old judgement reached no detection', verdicts=J[s0].get('judge_confidence', '')))
    os.makedirs(out_dir, exist_ok=True)
    jp = os.path.join(out_dir, 'judgements.jsonl')
    if os.path.exists(jp) and os.path.getsize(jp) > 0:
        raise StageRefusal(f'{jp} exists and is not empty - the carry only seeds an empty journal - REFUSING')
    with open(jp + '.tmp', 'w') as f:
        for r in recs:
            f.write(json.dumps(r) + '\n')
    os.replace(jp + '.tmp', jp)
    _write_csv(os.path.join(out_dir, 'carry_report.csv'), CARRY_COLS, rows)
    return dict(judged_old=len(J), carried=len(recs), conflicts=sum(1 for r in rows if r['outcome'].startswith('CONFLICT')),
                unmatched=len(unmatched), new_identities_touched=len(owner),
                carried_on_page=(sum(1 for r in recs if r['source_id'] in on_page) if on_page is not None else None))


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
    p.add_argument('--group-stokes', default='I,V', metavar='I,V', help='Stokes to group (default both)')


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


def assemble_tenmin_input(lsts, out_dir, cfg=None):
    """The 10-minute grouping input: the master_catalog.csv and pertie_detections.csv rows of every published LST run of
    ``lsts`` (header union, LST order, the run's row order) - every detection the LSTs' builds see, BEFORE the cumulative's deep
    omission (the merge applies it, as production; a build refuses a detection the grouping does not hold). Writes
    out_dir/grouping_input/{master_catalog,pertie_detections}.csv; returns (their paths, {lst: run dir}). A run without
    either file, or a cand_id in two runs, REFUSES."""
    d = os.path.join(out_dir, 'grouping_input')
    os.makedirs(d, exist_ok=True)
    runs = {L: lst_run(L, cfg) for L in lsts}
    out = []
    for name in ('master_catalog.csv', 'pertie_detections.csv'):
        hdr, rows, seen = [], [], {}
        for L, R in runs.items():
            p = os.path.join(R, name)
            if not os.path.exists(p):
                raise StageRefusal(f'{L}: no {p} - publish the LST before grouping the cumulative')
            with open(p, newline='') as f:
                rd = csv.DictReader(f)
                hdr += [k for k in (rd.fieldnames or []) if k not in hdr]
                for r in rd:
                    c = r.get('cand_id')
                    if c in seen:
                        if seen[c] != (L, r):
                            raise StageRefusal(f'{name}: cand_id {c} in {seen[c][0]} and {L} - REFUSING')
                        continue
                    seen[c] = (L, r)
                    rows.append(r)
        out.append(_write_csv(os.path.join(d, name), hdr, rows))
    return out[0], out[1], runs


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
              + f'; out {out} (state in the same directory); Stokes {",".join(stokes)}')
        return 0
    from exopipe.phase3.core.record import NoInput
    with run.unit(f'{label}.{STAGE}', tier='sublst', stage=STAGE) as u:
        u.meta['mode'] = m.as_dict()
        if not m.design_b:
            raise NoInput("catalog_grouping.GROUPING = 'production': production groups per LST in the builder and joins LSTs "
                          "by J-name (the parity file) - no grouping stage")
        try:
            if src:
                mc, pt = os.path.join(src, 'master_catalog.csv'), os.path.join(src, 'pertie_detections.csv')
                u.meta['input'] = src
            else:
                mc, pt, runs = assemble_tenmin_input(lsts, out)
                u.meta['input'] = {L: R for L, R in runs.items()}
                for R in runs.values():
                    for name in ('master_catalog.csv', 'pertie_detections.csv'):
                        u.add_input(os.path.join(R, name))
        except StageRefusal as e:
            u.refuse(str(e))
        for p in (mc, pt):
            if not os.path.exists(p):
                u.refuse(f'grouping input {p} is missing')
            u.add_input(p)
        for S in stokes:
            if S not in ('I', 'V'):
                u.refuse(f'--group-stokes {S}: I and/or V')
            d = group_dir(out, S)
            if os.path.exists(d):
                os.rename(d, f'{d}.superseded_{time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())}')
            from exopipe.phase3.lib.catalog import tests_grouping as TG
            try:
                det = load_det('10min', S, mc, pt)
                paths, summ, _res = run_group(det, '10min', S, out, out, f'{label}.{STAGE}', dict(master_catalog=mc, pertie=pt),
                                              group_mode=getattr(args, 'group_mode', None),
                                              new_state=bool(getattr(args, 'group_new_state', False)),
                                              check=lambda pp, S=S: TG.failures(pp, tier='10min', stokes=S, input=mc))
            except (StageRefusal, G.GroupingRefused) as e:
                u.refuse(f'{S}: {e}')
            u.meta.setdefault('summary', {})[S] = summ['counts']
            u.add_output(paths['detections'], 'grouping_table', tier='10min', stokes=S, input=mc)
            u.add_output(paths['state'], 'grouping_state', tier='10min', stokes=S, table=paths['detections'])
            u.note(f"{S}: {summ['counts']['detections']} detections -> {summ['counts']['identities']} identities, "
                   f"{summ['counts']['multiples']} multiples ({summ['counts']['identities_in_multiples']} identities)")
    return None
