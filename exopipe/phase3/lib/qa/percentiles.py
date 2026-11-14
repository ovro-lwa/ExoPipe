"""The Phase 3 QA percentile table of a search run (PHASE3_ARCHITECTURE.md section 3, "QA doctrine - descriptive
percentiles"; Gregg 2026-11-14 approved making the per band and tier table a product).

The metric: the number of transient candidates detected in each searched image, as a percentile against the other
searched images of the same band, cadence tier and Stokes. It is descriptive: it never excludes anything.

Input: the run's per-image table of the search, <run>/v_deep_images.csv (one row per searched image and Stokes, images
without a candidate included; columns lst, scope = the band, stokes, date = the night, or the night with a 10-minute
window tag, n_after_morph, n_after_mstruct, n_raw_floor).
  count      Stokes V: n_after_mstruct, the candidates left after the master-structure gate (the document's count: it
             does not move when later vetoes are tuned).
             Stokes I: n_after_morph, the candidates left after the Stokes I search's morphology gate. The Stokes I search
             records no master-structure count (its n_after_mstruct is blank), so this is the nearest recorded count;
             count_kind says which one a row carries.
             A blank count with n_after_morph 0 is 0: nothing reached the later gate (the Stokes V search records no
             master-structure count for an image with no candidate left after the morphology gate; Run_SLSTB: 319 of its
             31,078 V images). Any other blank count is not recorded: the image is left out and is in no population.
  count_raw  n_raw_floor, the raw count at or above the search floor, kept alongside for reference (the document: a poor
             quality predictor).
  percentile the mid-rank among the population, 0 to 100: the share of the population's images with a smaller count plus
             half the share with the same count. 0 = the fewest candidates.
  population the number of images of the band, tier and Stokes in the run.
  population_version  the first 16 hex digits of the sha256 of the per-image table the percentiles were computed from. A
             table whose version is not the per-image table's is stale: read() does not use it.

One run holds one cadence tier; the caller names it. write() puts the table beside the per-image table as
<run>/qa_percentiles.csv; a catalog chain calls it after it has assembled the run's per-image table. read() gives the
rows of a current table file, else computes them (a reader, such as the hour's email, never writes into a run).
"""
import bisect
import csv
import hashlib
import os

IMAGES = 'v_deep_images.csv'
TABLE = 'qa_percentiles.csv'
COLUMNS = ('lst', 'date', 'band', 'stokes', 'tier', 'count', 'count_kind', 'percentile', 'count_raw', 'percentile_raw',
           'population', 'population_version')
COUNT_OF = {'V': ('n_after_mstruct', 'after_mstruct'), 'I': ('n_after_morph', 'after_morph')}
# the tier a catalog chain's table carries: the names the hour's email reads the tables by (config notify.PAGES)
DEEP, TENMIN = 'deep', '10-minute'


def midrank(value, ordered):
    """Percentile (0-100) of a value in a sorted list: the share below it plus half the share equal to it."""
    lo, hi = bisect.bisect_left(ordered, value), bisect.bisect_right(ordered, value)
    return 100.0 * (lo + 0.5 * (hi - lo)) / len(ordered)


def version(run):
    """The population version of a run: the first 16 hex digits of the sha256 of its per-image table."""
    h = hashlib.sha256()
    with open(os.path.join(run, IMAGES), 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()[:16]


def _int(x):
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return None


def table(run, tier):
    """The table's rows (dicts with COLUMNS; count, count_raw and population integers, percentiles floats rounded to two
    decimals, a missing count_raw and its percentile ''), in the per-image table's order. A blank count with
    n_after_morph 0 is 0; a row whose count is otherwise not recorded is left out and is in no population. Raises
    OSError when the run has no per-image table and ValueError when the table lacks a column."""
    ver = version(run)
    with open(os.path.join(run, IMAGES), newline='') as f:
        rd = csv.DictReader(f)
        need = {'lst', 'scope', 'stokes', 'date', 'n_after_morph', 'n_after_mstruct'}   # n_raw_floor: reference only,
                                                                                    # optional (older tables lack it)
        if not need <= set(rd.fieldnames or ()):
            raise ValueError(f'{os.path.join(run, IMAGES)} lacks {sorted(need - set(rd.fieldnames or ()))}')
        rows = []
        for r in rd:
            col, kind = COUNT_OF.get(r['stokes'], (None, None))
            n = _int(r.get(col)) if col else None
            if n is None and col and _int(r.get('n_after_morph')) == 0:
                n = 0                   # nothing left after the morphology gate: nothing reached the later one
            if n is None:
                continue
            rows.append(dict(lst=r['lst'], date=r['date'], band=r['scope'], stokes=r['stokes'], tier=tier, count=n,
                             count_kind=kind, count_raw=_int(r.get('n_raw_floor'))))
    pops, raws = {}, {}
    for r in rows:
        k = (r['band'], r['stokes'])
        pops.setdefault(k, []).append(r['count'])
        if r['count_raw'] is not None:
            raws.setdefault(k, []).append(r['count_raw'])
    for v in list(pops.values()) + list(raws.values()):
        v.sort()
    for r in rows:
        k = (r['band'], r['stokes'])
        r['percentile'] = round(midrank(r['count'], pops[k]), 2)
        r['percentile_raw'] = '' if r['count_raw'] is None else round(midrank(r['count_raw'], raws[k]), 2)
        if r['count_raw'] is None:
            r['count_raw'] = ''
        r['population'] = len(pops[k])
        r['population_version'] = ver
    return rows


def write(run, tier, path=None):
    """Write the run's table (default <run>/qa_percentiles.csv), whole and atomically. Returns (path, rows written)."""
    rows = table(run, tier)
    path = path or os.path.join(run, TABLE)
    tmp = f'{path}.tmp{os.getpid()}'
    try:
        with open(tmp, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator='\n')
            w.writeheader()
            w.writerows(rows)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):                 # a failed write leaves nothing beside the run's products
            os.remove(tmp)
    return path, len(rows)


def write_for_chain(run, tier):
    """A catalog chain's call, right after its merge wrote <run>/v_deep_images.csv: write the table beside it and return
    its path. A per-image table the table cannot be made from, or one with no image whose count is recorded, REFUSES the
    chain's unit (core.record.Refusal, with the reason; the unit is not done and a later merge writes both again)."""
    from exopipe.phase3.core.record import Refusal
    try:
        path, n = write(run, tier)
    except (OSError, ValueError) as e:
        raise Refusal(f'QA percentile table of {run}: {e}')
    if not n:
        os.remove(path)
        raise Refusal(f'QA percentile table of {run}: {IMAGES} has no searched image with a recorded count')
    return path


def read(run, tier):
    """(rows, source): the rows of the run's table file when it is current (its population_version is the per-image
    table's and its tier the one asked for), source 'table'; else the rows computed now, source 'computed'."""
    path = os.path.join(run, TABLE)
    if os.path.exists(path):
        ver = version(run)
        with open(path, newline='') as f:
            rows = list(csv.DictReader(f))
        if rows and all(r.get('population_version') == ver and r.get('tier') == tier for r in rows):
            for r in rows:
                r['count'], r['population'], r['percentile'] = int(r['count']), int(r['population']), float(r['percentile'])
                if r['count_raw'] != '':
                    r['count_raw'], r['percentile_raw'] = int(r['count_raw']), float(r['percentile_raw'])
            return rows, 'table'
    return table(run, tier), 'computed'
