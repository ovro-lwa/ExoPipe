"""What the hour's email adds to Phase 2's QA section (Gregg 2026-11-14: "Add all the Phase 3 QA percentiles to the QA
section. The ionospheric QA is poorly calibrated. Even the best night i have ever seen is listed as 'Moderate
(Structured)'").

Phase 3 QA percentiles. The plan of record's Phase 3 quality metric is the number of transient candidates detected in each
epoch image, as a percentile against the other epoch images of the same band and cadence tier. Here it is read from the
per-detection table of each listed page's run (<run>/master_catalog.csv), per Stokes:
  Stokes V   the search's recorded count of the image (column n_v_epoch);
  Stokes I   the table's Stokes I rows of the image (the Stokes I search records no count).
An epoch image is one band, LST hour and night (deep) or one 10-minute window of them (10-minute). An image without a row
in the table has no count on record and is in no population. The percentile is the mid-rank: of the band's and tier's
images, the share with a smaller count plus half the share with the same count.
The scintillation index of the hour and band and its percentile in the band are read from the run's
scintillation_index.csv where the run has one.

Ionosphere. Phase 2 words its two numbers (the disturbance RMS of the 73 MHz deep anchor, the mean variability across the
10-minute intervals) by fixed thresholds. The email gives instead each number's percentile (mid-rank) among the processed
hours. The hours' numbers are kept in one small file under the step's output; an hour is read from Phase 2 once.
"""
import glob
import json
import os

from exopipe.phase3.lib.notify import sources
from exopipe.phase3.lib.notify.block import esc, num, table

_COUNTS = {}
WIDEBAND = ('Red', 'Green', 'Blue', 'Full')


def midrank(value, population):
    """Percentile (0-100) of a value in a list: the share below it plus half the share equal to it; None without both."""
    if value is None or not population:
        return None
    below = sum(1 for x in population if x < value)
    same = sum(1 for x in population if x == value)
    return 100.0 * (below + 0.5 * same) / len(population)


def image_counts(run):
    """{Stokes: {band: {(LST, night or window): count}}} of a run's per-detection table; one pass per run and process."""
    if run not in _COUNTS:
        out = {'V': {}, 'I': {}}
        for r in sources._rows(os.path.join(run, 'master_catalog.csv')):
            st, key = r.get('stokes'), (r.get('lst'), r.get('date'))
            if st == 'V':
                n = sources._f(r.get('n_v_epoch'))
                if n is not None:
                    b = out['V'].setdefault(r.get('band'), {})
                    b[key] = max(n, b.get(key, 0))
            elif st == 'I':
                b = out['I'].setdefault(r.get('band'), {})
                b[key] = b.get(key, 0) + 1
        _COUNTS[run] = out
    return _COUNTS[run]


def _band_order(b):
    return (1, WIDEBAND.index(b)) if b in WIDEBAND else (0, sources._f(str(b).replace('MHz', '')) or 0.0) if 'MHz' in str(b) \
        else (2, 0)


def phase3_percentiles(item, CFG):
    """(HTML of the Phase 3 QA percentile table of the hour, limits)."""
    lst, date = item.split('_', 1)
    limits, cols, cells, bands = [], [], {}, set()
    try:
        reg = sources._registry(CFG)
    except (OSError, ValueError, KeyError) as e:
        return '', [f'Phase 3 QA percentiles: the page registry cannot be read ({type(e).__name__}) - not shown']
    scint = {}
    for (tier, stokes), pid in CFG.notify.PAGES.items():
        run = (reg.get(pid) or {}).get('run')
        name = f'{tier}, Stokes {stokes}'
        if not run or not os.path.exists(os.path.join(run, 'master_catalog.csv')):
            limits.append(f'Phase 3 QA percentiles, {name}: no per-detection table for page {pid} - not shown')
            continue
        try:
            by_band = image_counts(run)[stokes]
        except (OSError, ValueError) as e:
            limits.append(f'Phase 3 QA percentiles, {name}: the per-detection table of page {pid} cannot be read '
                          f'({type(e).__name__}) - not shown')
            continue
        cols.append(name)
        n_pop = 0
        for band, imgs in by_band.items():
            pop = list(imgs.values())
            n_pop += len(pop)
            own = [n for (l, d), n in imgs.items() if l == lst and str(d)[:10] == date]
            if own:
                bands.add(band)
                worst = max(own)
                cells[(name, band)] = (f'{worst:g} ({num(midrank(worst, pop), ".0f")})'
                                       + (f'; {len(own)} windows' if tier != 'deep' else ''))
        cells[(name, None)] = n_pop
        sp = os.path.join(run, 'scintillation_index.csv')
        if tier not in scint and os.path.exists(sp):
            try:
                scint[tier] = {r['band']: r for r in sources._rows(sp, f'{lst},{date}')
                               if r.get('lst') == lst and r.get('date', '')[:10] == date}
                bands.update(scint[tier])
            except (OSError, ValueError, KeyError) as e:
                limits.append(f'scintillation index, {tier}: {sp} cannot be read ({type(e).__name__}) - not shown')
    if not cols:
        return '', limits
    columns = [('Band', lambda b: esc(b))]
    for name in cols:
        columns.append((f'{name}: count (percentile)', lambda b, name=name: cells.get((name, b), '&ndash;')))
    for tier, rows in scint.items():
        columns.append((f'Scintillation index, {tier} (percentile in its band)',
                        lambda b, rows=rows: (f'{num(rows[b].get("scint_index"), ".3f")} '
                                              f'({num(rows[b].get("pct_in_band"), ".0f")})') if b in rows else '&ndash;'))
    sizes = '; '.join(f'{name}: {cells[(name, None)]:,} images' for name in cols)
    html = ('<h4>Phase 3 QA percentiles</h4>'
            + table(columns, sorted(bands, key=_band_order))
            + '<p><small>Count: transient candidates detected in the epoch image of this hour (Stokes V: the count the '
              'search recorded for the image; Stokes I: the Stokes I detections of the image in the run\'s per-detection '
              'table). For the 10-minute tier the largest count among the hour\'s windows that have a detection, and the '
              'number of those windows. Percentile: against the epoch images of the same band and tier in the served run '
              '(the share with a smaller count plus half the share with the same count; 0 = the fewest candidates). A dash: '
              f'no detection of that band, tier and Stokes in this hour, so no count is on record. Populations: {esc(sizes)}. '
              'Scintillation index: as the served run records it for the hour and band, with its percentile among the '
              'band\'s hours.</small></p>')
    return html, limits


def _read_hour(sci_dir):
    """{'rms': disturbance RMS in arcmin, 'var': mean variability} of a Phase 2 run directory (keys absent when the file
    is), or None when the run has no dewarp diagnostics yet."""
    d = os.path.join(sci_dir, 'Dewarp_Diagnostics')
    try:
        names = sorted(os.listdir(d))
    except OSError:
        return None
    out = {}
    for n in names:
        if not n.endswith('.json'):
            continue
        try:
            if 'warp_stats_' in n and '73MHz' in n:
                with open(os.path.join(d, n)) as f:
                    out['rms'] = json.load(f).get('disturbance_rms_arcmin')
            elif 'ionosphere_variability_' in n:
                with open(os.path.join(d, n)) as f:
                    out['var'] = json.load(f).get('mean_variability')
        except (OSError, ValueError):
            pass
    return out


def ionosphere_rank(sci_dir, item, CFG, out_root):
    """({'disturbance': words, 'variability': words} for the mailer, limits): the hour's two ionosphere numbers as
    percentiles among the processed hours. The population file is brought up to date first: every hour under
    paths.PHASE2_ROOT that it does not hold yet is read once (its latest Science directory)."""
    path = os.path.join(out_root, 'notify', CFG.notify.IONOSPHERE_POPULATION)
    try:
        with open(path) as f:
            pop = json.load(f)
    except (OSError, ValueError):
        pop = {}
    n0 = len(pop)
    for ld in sorted(glob.glob(os.path.join(CFG.paths.PHASE2_ROOT, '[0-2][0-9]h'))):
        try:
            dates = sorted(os.listdir(ld))
        except OSError:
            continue
        for d in dates:
            key = f'{os.path.basename(ld)}_{d}'
            if key in pop or len(d) != 10:
                continue
            sd = sorted(glob.glob(os.path.join(ld, d, 'Science_*')))
            got = _read_hour(sd[-1]) if sd else None
            if got is not None:
                pop[key] = got
    own = _read_hour(sci_dir)
    if own is not None:
        pop[item] = own
    if len(pop) != n0 or own is not None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f'{path}.tmp{os.getpid()}'
        with open(tmp, 'w') as f:
            json.dump(pop, f, sort_keys=True)
        os.replace(tmp, path)
    out, limits = {}, []
    for key, label, unit, word in (('rms', 'disturbance', "'", 'RMS'), ('var', 'variability', '', 'mean')):
        v = (own or {}).get(key)
        vals = [h[key] for h in pop.values() if isinstance(h.get(key), (int, float))]
        p = midrank(v, vals) if isinstance(v, (int, float)) else None
        if p is None:
            limits.append(f'ionosphere {label}: no value for this hour or no processed hours to compare with - Phase 2\'s '
                          f'own wording is shown')
            continue
        out[label] = (f'{word} {v:.2f}{unit}, percentile {p:.0f} of the {len(vals):,} processed hours with this measurement '
                      f'(0 = the calmest hour on record, 100 = the most disturbed)')
    return out, limits
