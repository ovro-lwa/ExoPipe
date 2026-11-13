"""The frozen populations of the deep search's beam and V sanity gates - recorded by a whole-cell search, read by an append.

R-GREGG-20261112-PER-HOUR-STEPS-SIX-RULINGS (2), Gregg: "Frozen statistics from earlier nights judge a new night, everywhere"
- "Approved."; the deep tier's frozen state lives with the master's records and the published attempt
(R-GREGG-20261113-PER-HOUR-STATE-LOCATION-AND-REBUILD-TRIGGER: build_stats.json, admissions.csv, the published attempt's sigstack/
and frame_noise.csv).

WHERE. The beam gate (deep_search.run_deep_cell, "(1c) BEAM GATE") keys on the median fitted-beam area of the nights the gates
before it leave (data_decisions, night-image, epoch, the 10x sanity gate, input provenance), admitted nights excluded. That set
exists only at search time, so its record is written with the search's attempt - FILE in transients/Run_<label>/, next to
frame_noise.csv, carried by the copy an append starts from and by the publish - not in the master's build_stats.json (the
build's populations, whose content test fixes its keys; build_stats.KEYS).

WHAT. One row per (cell, gate) per whole-cell search: the population's nights and count, its median (the engine's upper-middle
element, repr - a FITS-free exact round trip), the threshold factor (P3_BEAM_GATE_RATIO) and the minimum count
(BEAM_GATE_MIN_NIGHTS) it ran with, and the outcome: 'applied'; 'below-minimum' (fewer nights than the minimum: the gate was
skipped and the skip counted - R-BEAM-GATE-FIVE-NIGHT-FLOOR-WAIVED-THIN-CELLS-SEARCHED, Gregg 2026-09-11 "Waive the floor. I want a
complete Masters set."); 'off' (P3_BEAM_GATE_OFF=1). A thin cell is therefore a recorded value, never an absence: a reader tells
"below the minimum, no gate" from "no record". basis 'search' = written by the search; 'backfill' = reconstructed for an attempt
published before this record existed (backfill_attempt), from its own refusal records and the cell's product headers, and
accepted only when the reconstruction reproduces the attempt's recorded beam-gate refusals exactly.

READ. An append (deep_search incremental mode: per hour, --nights) judges every night with the base attempt's row (frozen_beam):
no row, two different rows, another factor or minimum, a base searched with the gate off, or a population holding a night
data_decisions excludes now refuses the cell (a gate refuses; it never falls back to recomputing). Whole-cell searches read
nothing (byte-identical gate; the row is an addition).

THE V SANITY GATE ('v_sanity'). A whole-cell search that runs a V search records the gate's population too: the nights with a
positive finite V frame rms (admitted nights excluded), its median (the engine's _vf[len // 2]) when the gate applied, the factor
(SANITY_I_X) and the minimum (SANITY_MIN_NIGHTS). Under the B-02 rules an append judges by the median its base recorded in
frame_noise.csv (removal_rules.base_vsanmed); when the base recorded none, this row tells why (frozen_v_sanity): 'below-minimum'
(the base skipped the gate - the append skips it, counted, fallbacks.SEARCH_GATE_SKIP_PERHOUR_FALLBACK) from no row or an 'applied' row
without a median, which refuse. Under the production rules an append recomputes the V median from its own frames, as before.
"""
import csv
import glob
import os
import re

FILE = 'gate_population.csv'
COLS = ['lst', 'scope', 'gate', 'n', 'median', 'factor', 'min_nights', 'outcome', 'nights', 'basis', 'attempt']
GATES = ('beam', 'v_sanity')
OUTCOMES = ('applied', 'below-minimum', 'off')
BASES = ('search', 'backfill')
BACKFILL_STAGE = 'gate-backfill'
_KEY = ('n', 'median', 'factor', 'min_nights', 'outcome', 'nights')     # a row's content (the attempt tag aside)


def row(lst, scope, gate, nights, median, factor, min_nights, outcome, basis, attempt):
    """One FILE row (COLS)."""
    if gate not in GATES or outcome not in OUTCOMES or basis not in BASES:
        raise ValueError(f'gate_population row: gate {gate!r} / outcome {outcome!r} / basis {basis!r}')
    nights = sorted(nights)
    return [lst, scope, gate, len(nights), '' if median is None else repr(float(median)), repr(float(factor)), int(min_nights),
            outcome, ';'.join(nights), basis, attempt]


def append(run_dir, rows):
    """Append ``rows`` to ``run_dir``/FILE under an exclusive lock, the header written first (deep_search._append_locked's
    rule: rows are appended only under their own header - another header REFUSES)."""
    import fcntl
    p = os.path.join(run_dir, FILE)
    with open(p + '.lock', 'w') as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        new = not os.path.exists(p)
        if not new:
            with open(p, newline='') as fh:
                have = next(csv.reader(fh), None)
            if have is not None and have != COLS:
                raise RuntimeError(f'{p}: header {have} != {COLS} - REFUSING to append')
        with open(p, 'a', newline='') as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(COLS)
            w.writerows(rows)


def read(run_dir, lst, scope, gate):
    """The cell's row of ``gate`` in ``run_dir``/FILE as a dict (n, min_nights int; median float or None; factor float; nights
    list), or None when the file or the row is absent. RuntimeError (REFUSING) when the file's header is not COLS or the cell has
    rows of different content (an attempt holds one population per gate)."""
    p = os.path.join(run_dir, FILE)
    if not os.path.exists(p):
        return None
    with open(p, newline='') as fh:
        rd = csv.DictReader(fh)
        if rd.fieldnames != COLS:
            raise RuntimeError(f'{p}: header {rd.fieldnames} != {COLS} - REFUSING')
        rows = [r for r in rd if (r['lst'], r['scope'], r['gate']) == (lst, scope, gate)]
    if not rows:
        return None
    if len({tuple(r[k] for k in _KEY) for r in rows}) > 1:
        raise RuntimeError(f'{p}: {lst} {scope} {gate}: {len(rows)} rows of different content - one population per attempt; '
                           f'REFUSING')
    r = rows[0]
    return dict(n=int(r['n']), median=float(r['median']) if r['median'] else None, factor=float(r['factor']),
                min_nights=int(r['min_nights']), outcome=r['outcome'], nights=[x for x in r['nights'].split(';') if x],
                basis=r['basis'], attempt=r['attempt'], path=p)


def frozen_beam(run_dir, lst, scope, factor, min_nights, excluded):
    """The base attempt's beam-gate population an append is judged by (module docstring, READ), or RuntimeError naming why it
    cannot be: missing, another factor / minimum, a base searched with the gate off, a population night ``excluded(date)`` now."""
    fz = read(run_dir, lst, scope, 'beam')
    if fz is None:
        raise RuntimeError(f'{lst} {scope}: BEAM GATE of an append - the base attempt holds no frozen beam-gate population '
                           f'({os.path.join(run_dir, FILE)}): a new night is judged by the population the cell was searched '
                           f'with (R-GREGG-20261112-PER-HOUR-STEPS-SIX-RULINGS (2)), never a recomputed one - backfill the '
                           f'published attempt (search --tier deep --stages {BACKFILL_STAGE}) or re-search the cell; REFUSING')
    if fz['outcome'] == 'off':
        raise RuntimeError(f'{lst} {scope}: BEAM GATE of an append - the base attempt was searched with the beam gate off '
                           f'(P3_BEAM_GATE_OFF=1); an append with it on would mix the rules in one attempt; REFUSING')
    if fz['factor'] != float(factor) or fz['min_nights'] != int(min_nights):
        raise RuntimeError(f'{lst} {scope}: BEAM GATE of an append - the base attempt ran it with factor {fz["factor"]:g} / '
                           f'minimum {fz["min_nights"]}, this run with {float(factor):g} / {int(min_nights)}; REFUSING')
    if fz['n'] != len(fz['nights']) or (fz['outcome'] == 'applied') != (fz['n'] >= fz['min_nights']) \
            or (fz['outcome'] == 'applied') != (fz['median'] is not None and fz['median'] > 0):
        raise RuntimeError(f'{fz["path"]}: {lst} {scope} beam row inconsistent (n {fz["n"]}, {len(fz["nights"])} nights, outcome '
                           f'{fz["outcome"]}, median {fz["median"]}); REFUSING')
    ex = sorted(d for d in fz['nights'] if excluded(d))
    if ex:
        raise RuntimeError(f'{lst} {scope}: BEAM GATE of an append - the frozen population holds night(s) {ex} that '
                           f'data_decisions.csv excludes now: a population holding them is not one to judge a new night by '
                           f'(exclusions are absolute; re-search the cell); REFUSING')
    return fz


def frozen_v_sanity(run_dir, lst, scope, factor, min_nights):
    """The V sanity row of the base attempt of an append whose base carries no V noise-rule median in frame_noise.csv (module
    docstring, THE V SANITY GATE): returned when it records 'below-minimum' (the base skipped the gate; the append skips it,
    counted); RuntimeError naming why otherwise - no row (thin cannot be told from missing), an 'applied' row (the base applied
    the gate, so its frame_noise.csv should carry the median), another factor or minimum, an inconsistent row."""
    fz = read(run_dir, lst, scope, 'v_sanity')
    if fz is None:
        raise RuntimeError(f'{lst} {scope}: V SANITY GATE of an append - the base attempt recorded no V noise-rule median '
                           f'(frame_noise.csv) and no V sanity population ({os.path.join(run_dir, FILE)}): a base below the '
                           f'minimum cannot be told from a missing value (R-GREGG-20261112-PER-HOUR-STEPS-SIX-RULINGS (2)); '
                           f're-search the cell; REFUSING')
    if fz['factor'] != float(factor) or fz['min_nights'] != int(min_nights):
        raise RuntimeError(f'{lst} {scope}: V SANITY GATE of an append - the base attempt ran it with factor {fz["factor"]:g} / '
                           f'minimum {fz["min_nights"]}, this run with {float(factor):g} / {int(min_nights)}; REFUSING')
    if fz['outcome'] != 'below-minimum' or fz['n'] != len(fz['nights']) or fz['n'] >= fz['min_nights'] \
            or fz['median'] is not None:
        raise RuntimeError(f'{fz["path"]}: {lst} {scope}: V SANITY GATE of an append - the base recorded the gate '
                           f'{fz["outcome"]} (n {fz["n"]}, {len(fz["nights"])} nights, minimum {fz["min_nights"]}, median '
                           f'{fz["median"]}) but its frame_noise.csv carries no V median - inconsistent records; REFUSING')
    return fz


def judge(fz, areas, dates):
    """The nights of ``dates`` the frozen population refuses: area > factor x median ('applied'); none otherwise. ``areas`` =
    {date: BMAJ x BMIN} as the engine reads them. The engine's own comparison, the median taken from the record."""
    if fz['outcome'] != 'applied':
        return []
    return [d for d in dates if areas.get(d, 0) > fz['factor'] * fz['median']]


# ---------------------------------------------------------------------------------------------------------------------
# backfill: the row of an attempt published before FILE existed
# ---------------------------------------------------------------------------------------------------------------------
_PRE = ('datadecisions', 'nightimage', 'epochgate', 'sanity', 'stale_input')   # remove a night before the beam gate
_KEPT = ('sanity_I', 'searchtier', 'wbmember', 'sanity_V')                      # recorded, the night stays in the gate's set
_TAG = re.compile(r'^\d{8}T\d{6}$')


def _refusal_rows(rd, cell, gate):
    p = os.path.join(rd, 'refusals', f'{cell}_{gate}_excluded.csv')
    if not os.path.exists(p):
        return []
    with open(p, newline='') as fh:
        return [r for r in csv.reader(fh) if r and len(r) >= 4 and r[0] + '_' + r[1] == cell]


def _by_tag(rows, admitted):
    """{attempt tag: set of nights} of refusal rows (the tag is the last column), admitted nights left out."""
    out = {}
    for r in rows:
        out.setdefault(r[-1], set())
        if r[2] not in admitted:
            out[r[-1]].add(r[2])
    return out


def _utc(tag):
    import calendar
    import time
    return calendar.timegm(time.strptime(tag, '%Y%m%dT%H%M%S'))


def backfill_attempt(att, cell, stokes, cell_dir, factor, min_nights, beam_area, admitted, excluded):
    """(row, report) for the published attempt ``att`` of (cell, Stokes): its beam-gate population reconstructed from its own
    records and the cell's products, or (None, report) with report['refused'] naming why. ``beam_area(date)`` = the engine's
    reader (wideband_admission.beam_area); ``admitted`` = the cell's admitted nights; ``excluded(date)`` = data_decisions now.
    The population = the nights the attempt searched in its Stokes (v_deep_images.csv) or refused at or after the beam gate
    while they stayed in its set (beam, the 3x I / V sanity, search tier, wideband member), admitted nights left out. Accepted
    only when (a) every night with an I product in the cell now is placed - in the population, refused before the gate by the
    attempt (data_decisions, night-image, epoch, 10x sanity, input provenance), admitted, or excluded now; (b) every gate file's
    nights agree between the attempt's runs (a base and its appends re-apply the gates to the whole cell); (c) no population
    night's I product is newer than the attempt's first run; (d) the median of the population's areas reproduces the attempt's
    beam refusals exactly - the same nights, each ratio to two decimals and the factor."""
    lst, scope = cell.split('_', 1)
    runs = sorted(glob.glob(os.path.join(att, 'transients', 'Run_*')))
    rep = dict(attempt=att, cell=cell, stokes=stokes)
    if len(runs) != 1:
        rep['refused'] = f'{len(runs)} transients/Run_* directories'
        return None, rep
    rd = runs[0]
    adm = set(admitted)
    tags, sets = set(), {}
    for g in _PRE + ('beam',) + _KEPT:
        rows = _refusal_rows(rd, cell, g)
        bt = _by_tag(rows, adm)
        tags |= set(bt)
        if len({frozenset(v) for v in bt.values()}) > 1:
            rep['refused'] = f'{g} refusals differ between the attempt\'s runs {sorted(bt)}'
            return None, rep
        sets[g] = set().union(*bt.values()) if bt else set()
        if g == 'beam':
            beam_rows = [r for r in rows if r[2] not in adm]
    bad_tags = sorted(t for t in tags if not _TAG.match(t))
    if bad_tags:
        rep['refused'] = f'refusal rows without an attempt tag: {bad_tags[:3]}'
        return None, rep
    m = re.search(r'_(\d{8}T\d{6})$', os.path.basename(att.rstrip('/')))
    first = min(list(tags) + ([m.group(1)] if m else []), default=None)
    if first is None:
        rep['refused'] = 'no attempt tag (directory name or refusal rows)'
        return None, rep
    img = os.path.join(rd, 'v_deep_images.csv')
    searched = set()
    if os.path.exists(img):
        with open(img, newline='') as fh:
            searched = {r['date'] for r in csv.DictReader(fh) if r.get('stokes') == stokes and r['lst'] + '_' + r['scope'] == cell}
    pop = (searched | sets['beam'] | set().union(*(sets[g] for g in _KEPT))) - adm
    have = set()
    for c in glob.glob(os.path.join(cell_dir, 'subtracted', f'*_I_sub_{scope}.fits')):
        mm = re.match(r'(\d{4}-\d\d-\d\d)_I_sub_', os.path.basename(c))
        if mm:
            have.add(mm.group(1))
    pre = set().union(*(sets[g] for g in _PRE))
    unplaced = sorted(d for d in have - adm - pop - pre if not excluded(d))
    gone = sorted(pop - have)
    rep.update(n=len(pop), searched=len(searched), refused_before=len(pre), admitted=len(adm & have),
               excluded_now_in_population=sorted(d for d in pop if excluded(d)), first_run=first)
    if unplaced or gone:
        rep['refused'] = ((f'night(s) {unplaced[:4]} with an I product the attempt\'s records do not place' if unplaced else '')
                          + ('; ' if unplaced and gone else '')
                          + (f'population night(s) {gone[:4]} without an I product now' if gone else ''))
        return None, rep
    newer = sorted(d for d in pop if os.path.getmtime(os.path.join(cell_dir, 'subtracted', f'{d}_I_sub_{scope}.fits'))
                   > _utc(first))
    if newer:
        rep['refused'] = f'I product(s) of {newer[:4]} newer than the attempt\'s first run {first}'
        return None, rep
    areas = {d: beam_area(d) for d in sorted(pop)}
    nob = sorted(d for d, a in areas.items() if a is None)
    if nob:
        rep['refused'] = f'no readable beam for {nob[:4]}'
        return None, rep
    n = len(pop)
    if n >= min_nights:
        fin = sorted(areas.values())
        med = fin[n // 2]
        want = {d for d in pop if areas[d] > factor * med}
        got = {r[2]: r for r in beam_rows}
        diff = sorted(want ^ set(got))
        badr = sorted(d for d, r in got.items() if d in want and (r[3] != f'{areas[d] / med:.2f}' or float(r[4]) != float(factor)))
        rep.update(median=med, refused_now=sorted(want))
        if diff or badr:
            rep['refused'] = (f'the reconstructed population does not reproduce the attempt\'s beam refusals (differ: {diff[:4]}; '
                              f'ratio or factor: {badr[:4]})')
            return None, rep
        return row(lst, scope, 'beam', pop, med, factor, min_nights, 'applied', 'backfill', first), rep
    if beam_rows:
        rep['refused'] = f'{n} population nights (below the minimum {min_nights}) but {len(beam_rows)} beam refusal(s) recorded'
        return None, rep
    return row(lst, scope, 'beam', pop, None, factor, min_nights, 'below-minimum', 'backfill', first), rep
