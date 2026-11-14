"""What the hour's email adds to Phase 2's QA section (Gregg 2026-11-14: "Add all the Phase 3 QA percentiles to the QA
section. The ionospheric QA is poorly calibrated. Even the best night i have ever seen is listed as 'Moderate
(Structured)'").

Phase 3 QA percentiles. The plan of record's Phase 3 quality metric: the number of transient candidates detected in each
searched image, as a percentile against the other searched images of the same band, cadence tier and Stokes. The rows
come from lib/qa/percentiles.py (the run's qa_percentiles.csv when it is current, else computed from the run's per-image
table v_deep_images.csv; this step never writes into a run): Stokes V the count after the master-structure gate,
Stokes I the count after the Stokes I search's morphology gate, images without a candidate included.
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
from exopipe.phase3.lib.qa import percentiles

_COUNTS = {}
WIDEBAND = ('Red', 'Green', 'Blue', 'Full')


def midrank(value, population):
    """percentiles.midrank of a value in an unsorted list; None without both."""
    if value is None or not population:
        return None
    return percentiles.midrank(value, sorted(population))


def image_rows(run, tier):
    """The QA percentile rows of a run (lib/qa/percentiles.read); one read per run and process."""
    if (run, tier) not in _COUNTS:
        _COUNTS[(run, tier)] = percentiles.read(run, tier)[0]
    return _COUNTS[(run, tier)]


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
        if not run or not os.path.exists(os.path.join(run, percentiles.IMAGES)):
            limits.append(f'Phase 3 QA percentiles, {name}: no per-image table of the search for page {pid} - not shown')
            continue
        try:
            rows = [r for r in image_rows(run, tier) if r['stokes'] == stokes]
        except (OSError, ValueError) as e:
            limits.append(f'Phase 3 QA percentiles, {name}: the per-image table of page {pid} cannot be read '
                          f'({type(e).__name__}) - not shown')
            continue
        cols.append(name)
        own = {}
        for r in rows:
            if r['lst'] == lst and str(r['date'])[:10] == date:
                own.setdefault(r['band'], []).append(r)
        for band, rr in own.items():
            bands.add(band)
            worst = max(rr, key=lambda r: r['count'])
            cells[(name, band)] = (f'{worst["count"]} ({num(worst["percentile"], ".0f")})'
                                   + (f'; {len(rr)} windows' if tier != 'deep' else ''))
        cells[(name, None)] = len(rows)
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
            + '<p><small>Count: transient candidates detected in the searched image of this hour, as the search recorded '
              'it (Stokes V: after the master-structure gate; Stokes I: after the Stokes I search\'s morphology gate, which '
              'is the nearest count that search records). For the 10-minute tier the largest count among the hour\'s '
              'searched windows, and the number of those windows. Percentile: against the searched images of the same '
              'band, tier and Stokes in the served run, images without a candidate included (the share with a smaller '
              'count plus half the share with the same count; 0 = the fewest candidates). A dash: no searched image of '
              f'that band, tier and Stokes in this hour. Populations: {esc(sizes)}. '
              'Scintillation index: as the served run records it for the hour and band, with its percentile among the '
              'band\'s hours.</small></p>')
    return html, limits


def _sources(sci_dir):
    """(the two diagnostics files of a Phase 2 run directory as {'rms' | 'var': file name}, their stamp: the run
    directory's name and each file's name, size and mtime), or None when the run has no dewarp diagnostics yet."""
    d = os.path.join(sci_dir, 'Dewarp_Diagnostics')
    try:
        names = sorted(os.listdir(d))
    except OSError:
        return None
    files = {}
    for n in names:
        if n.endswith('.json') and 'warp_stats_' in n and '73MHz' in n:
            files['rms'] = n
        elif n.endswith('.json') and 'ionosphere_variability_' in n:
            files['var'] = n
    stamp = [os.path.basename(sci_dir)]
    for k in sorted(files):
        try:
            st = os.stat(os.path.join(d, files[k]))
            stamp.append([files[k], st.st_size, st.st_mtime_ns])
        except OSError:
            stamp.append([files[k], None, None])
    return files, stamp


def _read_hour(sci_dir, known=None):
    """{'rms': disturbance RMS in arcmin, 'var': mean variability, 'stamp': what they were read from} of a Phase 2 run
    directory (a value is absent while its file is), or None when the run has no dewarp diagnostics yet. known: the
    hour's stored entry, returned as it is while its stamp still holds (same run directory, same files, sizes and
    mtimes); an hour that gains a file, is rewritten or is reprocessed into a new run directory is read again."""
    got = _sources(sci_dir)
    if got is None:
        return None
    files, stamp = got
    if known is not None and known.get('stamp') == stamp:
        return known
    d = os.path.join(sci_dir, 'Dewarp_Diagnostics')
    out = {'stamp': stamp}
    for key, field in (('rms', 'disturbance_rms_arcmin'), ('var', 'mean_variability')):
        if key in files:
            try:
                with open(os.path.join(d, files[key])) as f:
                    out[key] = json.load(f).get(field)
            except (OSError, ValueError):
                out['stamp'] = None                  # unreadable now (being written?): read again next time
    return out


def ionosphere_rank(sci_dir, item, CFG, out_root):
    """({'disturbance': words, 'variability': words} for the mailer, limits): the hour's two ionosphere numbers as
    percentiles among the processed hours. The population file is brought up to date first: every hour under
    paths.PHASE2_ROOT (its latest Science directory) is read when the file does not hold it or its stored stamp no longer
    matches its diagnostics files (_read_hour); an hour whose diagnostics are gone leaves the population."""
    from exopipe.phase3.core import inputs
    path = os.path.join(out_root, 'notify', CFG.notify.IONOSPHERE_POPULATION)
    try:
        with open(path) as f:
            old = json.load(f)
    except (OSError, ValueError):
        old = {}
    pop = {}
    for ld in sorted(glob.glob(os.path.join(CFG.paths.PHASE2_ROOT, '[0-2][0-9]h'))):
        try:
            dates = sorted(os.listdir(ld))
        except OSError:
            continue
        for d in dates:
            if len(d) != 10:
                continue
            key = f'{os.path.basename(ld)}_{d}'
            sd = inputs.latest_science_dir(os.path.basename(ld), d, CFG.paths.PHASE2_ROOT)
            got = _read_hour(sd, old.get(key)) if sd else None
            if got is not None:
                pop[key] = got
    own = _read_hour(sci_dir, pop.get(item))
    if own is not None:
        pop[item] = own
    if pop != old:
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
