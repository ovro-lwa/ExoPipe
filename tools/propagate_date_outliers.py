#!/usr/bin/env python3
"""Date-level outlier propagation. A date that the per-cell 2-stage QA rejects in
a strong majority of cells is a GLOBAL bad date (e.g. an ionospheric vortex whose
whole-field rms still clears the loose threshold at a noisy plane LST like 19h).
Propagate it: add data_decisions exclude rows (all 15 subbands) for the cells
where it was KEPT, so is_excluded() drops it from those co-adds + searches too.

Usage: propagate_date_outliers.py [min_frac] [min_cells]   (default 0.6, 5)
"""
import sys, csv, glob, os, shutil, collections
_A = sys.argv[1:]
MIN_FRAC = float(_A[0]) if _A else 0.60
MIN_CELLS = int(_A[1]) if len(_A) > 1 else 5
DD = '/lustre/gh/main/clv15/data_decisions.csv'
SUBBANDS = [18, 23, 27, 32, 36, 41, 46, 50, 55, 59, 64, 69, 73, 78, 82]

# scan every cell's 2-stage verdict per date
verdict = collections.defaultdict(lambda: collections.defaultdict(dict))  # date -> lst -> {cell:outlier}
for f in glob.glob('/fast/gh/main/phase3/coadd/*_*/residuals_provisional.csv'):
    cell = os.path.basename(os.path.dirname(f))
    lst = cell.split('_')[0]
    try:
        for r in csv.DictReader(open(f)):
            verdict[r['date']][lst][cell] = (str(r['outlier']) == 'True')
    except Exception:
        pass

# date-level: fraction of ALL cells (any LST) that rejected this date
prop = {}   # date -> (n_rej, n_tot, set(leak_lsts))
for date, by_lst in verdict.items():
    cells = [(lst, c, o) for lst, cc in by_lst.items() for c, o in cc.items()]
    n_tot = len(cells); n_rej = sum(1 for *_ , o in cells if o)
    if n_tot >= MIN_CELLS and n_rej / n_tot >= MIN_FRAC and n_rej < n_tot:
        # LSTs where it leaked (some cell KEPT it)
        leak = {lst for lst, cc in by_lst.items() if any(not o for o in cc.values())}
        if leak:
            prop[date] = (n_rej, n_tot, leak)

print(f"date-level outliers (rejected in >= {MIN_FRAC:.0%} of >= {MIN_CELLS} cells, with leaks):")
existing = set()
for r in csv.DictReader(open(DD)):
    existing.add((r['lst'], r['subband'], r['date']))
add = []
for date, (nr, nt, leak) in sorted(prop.items()):
    print(f"  {date}: rejected {nr}/{nt} cells; LEAKED at {sorted(leak)} -> propagating exclude")
    for lst in sorted(leak):
        for s in SUBBANDS:
            if (lst, f'{s}MHz', date) not in existing:
                add.append(dict(lst=lst, subband=f'{s}MHz', date=date, verdict='exclude',
                                reason=f'date-level outlier propagation (rejected {nr}/{nt} cells)',
                                score_thermal_noise='', score_flux_scale='', score_ionosphere=''))
if not add:
    print("nothing to add."); sys.exit(0)
shutil.copy(DD, DD + '.bak_pre_propagate')
rows = list(csv.DictReader(open(DD))); cols = rows[0].keys()
with open(DD, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(cols)); w.writeheader()
    w.writerows(rows); w.writerows(add)
print(f"added {len(add)} exclude rows (backup {DD}.bak_pre_propagate)")
