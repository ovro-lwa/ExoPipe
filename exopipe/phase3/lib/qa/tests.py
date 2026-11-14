"""Content test of the Phase 3 QA percentile table (class qa_percentiles; core/validate discovers this module)."""
import csv
import os

from exopipe.phase3.core.validate import content_test
from exopipe.phase3.lib.qa import percentiles


@content_test('qa_percentiles')
def qa_percentiles_current(path, tier=None, **ctx):
    """<run>/qa_percentiles.csv is the table of the per-image table beside it NOW: same population version, and every
    row (count, percentile, raw count, its percentile, population) equal to the rows computed again from that table;
    percentiles within 0 to 100. The tier is the table's own (one run, one tier) unless ctx names it."""
    run = os.path.dirname(os.path.abspath(path))
    with open(path, newline='') as f:
        rd = csv.DictReader(f)
        cols, rows = tuple(rd.fieldnames or ()), list(rd)
    if cols != percentiles.COLUMNS:
        return False, 0, f'columns {cols} are not {percentiles.COLUMNS}'
    if not rows:
        return False, 0, 'no row'
    tiers = {r['tier'] for r in rows}
    if len(tiers) != 1 or (tier and tiers != {tier}):
        return False, len(rows), f'tiers in the table: {sorted(tiers)}' + (f'; asked for {tier}' if tier else '')
    want = percentiles.table(run, rows[0]['tier'])
    got = [{k: str(v) for k, v in r.items()} for r in rows]
    exp = [{k: str(r[k]) for k in percentiles.COLUMNS} for r in want]
    if got != exp:
        bad = next((i for i, (a, b) in enumerate(zip(got, exp)) if a != b), min(len(got), len(exp)))
        return False, len(rows), (f'{len(got)} rows, {len(exp)} computed now from {percentiles.IMAGES}; first difference at '
                                  f'row {bad + 1} (stale table, or another population version)')
    if any(not 0.0 <= float(r['percentile']) <= 100.0 for r in rows):
        return False, len(rows), 'a percentile outside 0 to 100'
    return True, len(rows), (f'{len(rows)} images, {len({(r["band"], r["stokes"]) for r in rows})} band and Stokes '
                             f'populations, version {rows[0]["population_version"]}')
