"""D-ADMIT (a), slice 1 (Gregg 2026-10-13: "gates against the build's recorded statistics"): the population numbers of the
cell-median gates of the build and of the search, recorded in <cell>/coadd_final/build_stats.json. A night admitted later is
judged against the population it did not join and is never added to it. The record holds the POPULATION (count, median,
MAD, areas) and the settings it was computed with; a gate applies its own settings, in force when it runs, to the recorded
population. Each number is derived from the cell's own records, in the cell directory, exactly as its gate computes it:
  stage2       per Stokes, over the finite provisional residual rms values of the build record (membership.stage2_record's
               source: coadd_final/stage2_verdict.csv, else residuals_provisional.csv): median, MAD and the thresholds at
               the recorded OUTLIER_SIGMA / LOW_OUTLIER_SIGMA (coadd.detect_outliers_mad; fewer than 5 values: no gate).
               The record's own i_thresh_mjy / v_thresh_mjy - the thresholds the build applied - are copied and must equal
               the re-derived high thresholds exactly.
  beam_gate    every night of the record (residuals_provisional.csv): the Science dir the build used (weights.csv sci_dir),
               the image the co-add beam gate reads there (coadd.beam_gate_image) and its fitted-beam area; the median as
               that gate takes it (the upper middle element; fewer than BEAM_GATE_MIN_AREAS: no gate), the nights above
               BEAM_GATE_RATIO x it, and the nights the build itself flagged (residuals_provisional.csv beam_outlier) - for a
               cell built before the gate's image rule of 2026-10-13 (D1) the two can differ: 'agree' says whether they do.
  sanity       the Stokes I sanity gate of the deep search (deep_search, ~1268): the cell median qa3 rms of weights.csv (the
               upper middle element of the finite positive values; fewer than SANITY_MIN_NIGHTS: no gate) - the 10x discard
               (both Stokes) and the SANITY_X cut (Stokes I). Not recorded: the search's Stokes V sanity population (the V
               frame rms of its searched frames) and its own beam gate - both come from the search frames, so the search
               computes them (slice 6).
  search_tier  per Stokes, over the stage-1 rms of residuals_provisional.csv: numpy median, 1.4826 x MAD and the threshold at
               the recorded SEARCH_TIER_SIGMA (deep_search; fewer than SEARCH_TIER_MIN: no gate).
basis: 'build' - written by the masters build when masters_build.WRITE_BUILD_STATS is True (ruled; production False writes
nothing); 'stage' - written later by `masters --tier deep --stages build-stats` from a built cell's records. Sources are
relative to the cell directory, so the record stays valid where install copies it. Content test (class build_stats): the
cell, sources and nights its folder gives, and every number re-derived with the recorded settings (recheck). What the test
cannot see: a setting is checked only through the numbers derived with it (OUTLIER_SIGMA through the threshold the build
applied, the other sigmas through their thresholds); a setting no recorded number depends on (X_DISCARD, SANITY_X), a
sigma edited together with its threshold where the build applied none (LOW_OUTLIER_SIGMA), and 'basis' are informational.
No gate reads them: every gate applies its own settings to the recorded population (2026-10-15 review).
Below a gate's minimum count the population is recorded all the same (n, median, MAD): a later ruling may lower a minimum.
A cell whose records are not one generation (residuals_provisional.csv dates other than weights.csv's - the search's own
check) or whose stage-2 record is another metric (membership.stage2_record's rule) has no record. A record made after the
build (basis 'stage') carries the build's own OUTLIER_SIGMA, read back from the threshold it applied, not today's.
"""
import csv
import json
import math
import os

FILE = 'build_stats.json'
VERSION = 2
KEYS = ('version', 'cell', 'basis', 'sources', 'settings', 'nights', 'stage2', 'stage2_recorded_thr', 'beam_gate',
        'sanity', 'search_tier')
X_DISCARD = 10.0               # deep_search's qa3 discard, '_wr.get(_kd(d), 0) > 10.0 * _med' (drift test)
STAGE2_MIN = 5                 # coadd.detect_outliers_mad: 'if len(valid) < 5'
BUILD_NIGHTS = 'residuals_provisional.csv'
WEIGHTS = os.path.join('coadd_final', 'weights.csv')
SETTINGS = ('OUTLIER_SIGMA', 'LOW_OUTLIER_SIGMA', 'BEAM_GATE_RATIO', 'BEAM_GATE_MIN_AREAS', 'SANITY_X', 'SANITY_MIN_NIGHTS',
            'X_DISCARD', 'SEARCH_TIER_SIGMA', 'SEARCH_TIER_MIN')


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float('nan')
    return v


def _rows(p):
    with open(p, newline='') as f:
        return list(csv.DictReader(f))


def _median(vals):
    import numpy as np
    return float(np.median(np.asarray(vals, dtype=float)))


def stage2_source(cell_dir):
    """The record membership.stage2_record reads: (path, rows); rows [] when it is unusable - another stage-2 metric
    (membership's guard) or missing (stage2_problem says why)."""
    from exopipe.phase3.lib.masters import membership as MB
    fd = os.path.join(cell_dir, 'coadd_final')
    v = os.path.join(fd, MB.VERDICT_FILE)
    if os.path.exists(v):
        rows = _rows(v)
        if rows and {r.get('metric') for r in rows} == {MB.STAGE2_METRIC}:
            return v, rows
    p = os.path.join(cell_dir, MB.BUILD_RECORD)
    rows = _rows(p) if os.path.exists(p) else []
    if {r.get('stage2_metric') for r in rows} - {None, '', MB.STAGE2_METRIC}:
        return p, []
    return p, rows


def stage2_problem(cell_dir):
    """Why the cell has no usable stage-2 record (membership.stage2_record's own words), else None."""
    from exopipe.phase3.lib.masters import membership as MB
    return MB.stage2_record(cell_dir)['problem']


def generation_problem(cell_dir):
    """The search's generation check (deep_search ~1327): residuals_provisional.csv must hold weights.csv's nights, else it
    is a stale derivative of another build. None when consistent."""
    a = os.path.join(cell_dir, BUILD_NIGHTS)
    b = os.path.join(cell_dir, WEIGHTS)
    if not (os.path.exists(a) and os.path.exists(b)):
        return None
    pa, pb = {r['date'] for r in _rows(a) if r.get('date')}, {r['date'] for r in _rows(b)}
    if pa != pb:
        return (f'{BUILD_NIGHTS} is not this generation\'s: its nights differ from weights.csv '
                f'({sorted(pa ^ pb)[:4]}) - a stale derivative')
    return None


def cell_sources(cell_dir):
    """{stage2, residuals_provisional, weights}: the records of the cell, relative to its directory."""
    return dict(stage2=os.path.relpath(stage2_source(cell_dir)[0], cell_dir), residuals_provisional=BUILD_NIGHTS,
                weights=WEIGHTS)


def stage2_stats(rows, outlier_sigma, low_sigma):
    """{'I': {...}, 'V': {...}} as coadd.detect_outliers_mad computes them (median and MAD of the finite values)."""
    out = {}
    for st, col in (('I', 'i_rms_mjy'), ('V', 'v_rms_mjy')):
        vals = [x for x in (_f(r.get(col)) for r in rows) if math.isfinite(x)]
        if not vals:
            out[st] = dict(n=0, gate=False)
            continue
        med = _median(vals)
        mad = _median([abs(x - med) for x in vals])
        sig = mad * 1.4826
        out[st] = dict(n=len(vals), gate=len(vals) >= STAGE2_MIN, median=med, mad=mad)
        if out[st]['gate']:
            out[st].update(thr_hi=med + outlier_sigma * sig, thr_lo=med - low_sigma * sig)
    return out


def recorded_thresholds(rows):
    """The high thresholds the build applied, per Stokes, as its record holds them (each distinct finite value)."""
    return {st: sorted({x for x in (_f(r.get(col)) for r in rows) if math.isfinite(x)})
            for st, col in (('I', 'i_thresh_mjy'), ('V', 'v_thresh_mjy'))}


def beam_stats(nights_areas, ratio, min_areas):
    """The co-add beam gate's median as it takes it (the upper middle element of the sorted areas) and the nights above
    ratio x it."""
    areas = sorted(a for a in nights_areas.values() if a is not None and math.isfinite(a))
    if not areas:
        return dict(n=0, gate=False, outliers=[])
    med = areas[len(areas) // 2]
    gate = len(areas) >= min_areas
    return dict(n=len(areas), gate=gate, median_area=med,
                outliers=sorted(d for d, a in nights_areas.items() if a is not None and a > ratio * med) if gate else [])


def beam_images(cell_dir, lst, scope, co):
    """The image the co-add beam gate reads for each night of the record, in the Science dir the build used (as _derive
    resolves them, no area read): with the records (sources) the inputs of a record of this cell directory."""
    prov = os.path.join(cell_dir, BUILD_NIGHTS)
    wp = os.path.join(cell_dir, WEIGHTS)
    nights = sorted({r['date'] for r in _rows(prov)}) if os.path.exists(prov) else []
    sci = {r['date']: r.get('sci_dir') for r in _rows(wp)} if os.path.exists(wp) else {}
    out = []
    for d in nights:
        img = co.beam_gate_image(lst, scope, d, sci=sci[d]) if sci.get(d) else None
        if img:
            out.append(img)
    return out


def beam_nights(lst, scope, nights, sci_dirs, co):
    """{date: {sci_dir, image, area}}: for each night the image the co-add beam gate reads in the Science dir the build
    used, and its fitted-beam area (None when no image or no readable beam). No side effects (coadd.beam_gate_image)."""
    out = {}
    for d in nights:
        sd = sci_dirs.get(d) or None
        img = co.beam_gate_image(lst, scope, d, sci=sd) if sd else None
        out[d] = dict(sci_dir=sd, image=img, area=co.beam_area(img) if img else None)
    return out


def sanity_stats(weights_rows, min_nights):
    """The search's Stokes I sanity population: the cell median qa3 rms of weights.csv (the upper middle element)."""
    vals = sorted(x for x in (_f(r.get('qa3_rms_mjy')) for r in weights_rows if r.get('qa3_rms_mjy') not in (None, '', 'nan'))
                  if math.isfinite(x) and x > 0)
    if not vals:
        return dict(n=0, gate=False)
    return dict(n=len(vals), gate=len(vals) >= min_nights, qa3_median=vals[len(vals) // 2])


def search_tier_stats(prov_rows, sigma, min_n):
    out = {}
    for st, col in (('I', 'i_rms_mjy'), ('V', 'v_rms_mjy')):
        vals = sorted(x for x in (_f(r.get(col)) for r in prov_rows if r.get(col) not in (None, '', 'nan'))
                      if math.isfinite(x) and x > 0)
        if not vals:
            out[st] = dict(n=0, gate=False)
            continue
        med = _median(vals)
        mad = _median([abs(x - med) for x in vals]) * 1.4826
        out[st] = dict(n=len(vals), gate=len(vals) >= min_n, median=med, mad=mad)
        if out[st]['gate']:
            out[st]['thr'] = med + sigma * mad
    return out


def _flag(x):
    return str(x).strip() in ('True', '1', 'true')


def _derive(cell_dir, lst, scope, co, S):
    """Every number of the record from the cell's records with the settings S - the one computation compute() and
    recheck() share."""
    _s2, s2_rows = stage2_source(cell_dir)
    prov = os.path.join(cell_dir, BUILD_NIGHTS)
    prov_rows = _rows(prov) if os.path.exists(prov) else []
    wp = os.path.join(cell_dir, WEIGHTS)
    w_rows = _rows(wp) if os.path.exists(wp) else []
    nights = sorted({r['date'] for r in prov_rows})
    sci = {r['date']: r.get('sci_dir') for r in w_rows}
    bn = beam_nights(lst, scope, nights, sci, co)
    bs = beam_stats({d: x['area'] for d, x in bn.items()}, S['BEAM_GATE_RATIO'], S['BEAM_GATE_MIN_AREAS'])
    rec_out = sorted(r['date'] for r in prov_rows if _flag(r.get('beam_outlier')))
    return dict(nights=nights,
                stage2=stage2_stats(s2_rows, S['OUTLIER_SIGMA'], S['LOW_OUTLIER_SIGMA']),
                stage2_recorded_thr=recorded_thresholds(s2_rows),
                beam_gate=dict(bs, nights=bn, recorded_outliers=rec_out, agree=(bs['outliers'] == rec_out)),
                sanity=sanity_stats(w_rows, S['SANITY_MIN_NIGHTS']),
                search_tier=search_tier_stats(prov_rows, S['SEARCH_TIER_SIGMA'], S['SEARCH_TIER_MIN']))


def settings_in_force(co, cfg):
    """The settings the numbers are computed with: the co-add's (in place) and the search's, in force now."""
    b, d = cfg.masters_build, cfg.search_detect
    return dict(OUTLIER_SIGMA=float(co.OUTLIER_SIGMA), LOW_OUTLIER_SIGMA=float(co.LOW_OUTLIER_SIGMA),
                BEAM_GATE_RATIO=float(b.BEAM_GATE_RATIO), BEAM_GATE_MIN_AREAS=int(b.BEAM_GATE_MIN_AREAS),
                SANITY_X=float(d.SANITY_X), SANITY_MIN_NIGHTS=int(d.SANITY_MIN_NIGHTS), X_DISCARD=X_DISCARD,
                SEARCH_TIER_SIGMA=float(d.SEARCH_TIER_SIGMA), SEARCH_TIER_MIN=int(d.SEARCH_TIER_MIN))


def build_outlier_sigma(rows, in_force):
    """The OUTLIER_SIGMA the build applied, read back from its record: (thr - median) / (1.4826 MAD) of Stokes I, rounded to
    6 decimals (the settings are short decimals); the value in force when the record holds no applied threshold."""
    st = stage2_stats(rows, 1.0, 1.0)['I']
    thr = recorded_thresholds(rows)['I']
    if not (st.get('gate') and thr and len(thr) == 1 and st['mad'] > 0):
        return in_force
    return round((thr[0] - st['median']) / (1.4826 * st['mad']), 6)


class NoRecord(ValueError):
    """The cell's records cannot give a population record (the reason)."""


def compute(cell_dir, lst, scope, co, cfg, basis):
    """The record of one built cell (co: the in-place co-add module - its thresholds and the beam gate's image rule).
    Raises NoRecord when the cell's stage-2 record is unusable or its records are not one generation."""
    if basis not in ('build', 'stage'):
        raise ValueError(f'basis {basis!r}')
    why = stage2_problem(cell_dir) or generation_problem(cell_dir)
    if why:
        raise NoRecord(why)
    S = settings_in_force(co, cfg)
    if basis == 'stage':                        # the build's sigma, not today's (a later ruling must not orphan old cells)
        S['OUTLIER_SIGMA'] = build_outlier_sigma(stage2_source(cell_dir)[1], S['OUTLIER_SIGMA'])
    return dict(version=VERSION, cell=os.path.basename(os.path.normpath(cell_dir)), basis=basis,
                sources=cell_sources(cell_dir), settings=S, **_derive(cell_dir, lst, scope, co, S))


def write(cell_dir, stats):
    p = os.path.join(cell_dir, 'coadd_final', FILE)
    with open(p + '.tmp', 'w') as f:
        json.dump(stats, f, indent=1, sort_keys=True)
    os.replace(p + '.tmp', p)
    return p


def read(cell_dir):
    p = os.path.join(cell_dir, 'coadd_final', FILE)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def recheck(path, co):
    """[problems] of <cell>/coadd_final/build_stats.json: the version, the cell, the sources and the nights its folder gives,
    the settings complete, and every number re-derived from the cell's records with the RECORDED settings (the settings in
    force today may differ: a gate applies its own to the population); the build's applied thresholds equal to their
    re-derived values exactly; the beam images re-resolved in the recorded Science dirs and their areas re-read."""
    with open(path) as f:
        s = json.load(f)
    cell_dir = os.path.dirname(os.path.dirname(os.path.abspath(path)))
    cell = os.path.basename(cell_dir)
    probs = []
    if s.get('version') != VERSION:
        return [f'version {s.get("version")!r}, not {VERSION}']
    if set(s) != set(KEYS):
        probs.append(f'keys {sorted(set(s) ^ set(KEYS))} not those of a version-{VERSION} record')
    why = stage2_problem(cell_dir) or generation_problem(cell_dir)
    if why:
        return probs + [why]
    if s.get('cell') != cell:
        probs.append(f'cell {s.get("cell")!r}, but the record is in {cell}/')
    if s.get('basis') not in ('build', 'stage'):
        probs.append(f'basis {s.get("basis")!r}')
    if s.get('sources') != cell_sources(cell_dir):
        probs.append(f'sources {s.get("sources")} are not the cell\'s own records {cell_sources(cell_dir)}')
    S = s.get('settings') or {}
    bad = [k for k in SETTINGS if not isinstance(S.get(k), (int, float)) or isinstance(S.get(k), bool)]
    if bad or set(S) != set(SETTINGS):
        return probs + [f'settings {sorted(S)}: missing, extra or not numbers ({bad})']
    lst, scope = cell.split('_', 1)
    d = _derive(cell_dir, lst, scope, co, S)

    def close(a, b_, what):
        if isinstance(a, dict) or isinstance(b_, dict):
            if not (isinstance(a, dict) and isinstance(b_, dict)):
                probs.append(f'{what}: recorded {b_!r}, re-derived {a!r}')
                return
            for k in sorted(set(a) | set(b_)):
                close(a.get(k), b_.get(k), f'{what}.{k}')
        elif isinstance(a, bool) or isinstance(b_, bool):
            if type(a) is not type(b_) or a != b_:
                probs.append(f'{what}: recorded {b_!r}, re-derived {a!r}')
        elif isinstance(a, float) or isinstance(b_, float):
            if not (isinstance(a, (int, float)) and isinstance(b_, (int, float))
                    and ((math.isnan(a) and math.isnan(b_)) or abs(a - b_) <= 1e-9 * max(1.0, abs(a), abs(b_)))):
                probs.append(f'{what}: recorded {b_!r}, re-derived {a!r}')
        elif a != b_ or type(a) is not type(b_):
            probs.append(f'{what}: recorded {b_!r}, re-derived {a!r}')
    for k in ('nights', 'stage2', 'stage2_recorded_thr', 'beam_gate', 'sanity', 'search_tier'):
        close(d[k], s.get(k), k)
    for st in ('I', 'V'):                                   # the thresholds the build applied: exactly the re-derived ones
        g, thr = d['stage2'][st], d['stage2_recorded_thr'][st]
        if g.get('gate') and thr and thr != [g['thr_hi']]:
            probs.append(f'stage2.{st}: the build applied {thr}, the re-derived high threshold is {g["thr_hi"]!r}')
    return probs
