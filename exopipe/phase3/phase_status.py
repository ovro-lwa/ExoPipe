#!/usr/bin/env python3
"""Fast Phase 1/2/3 status snapshot for the ExoPipe pipeline.
Reads the cached processing ledger + counts Phase-3 products + local running
jobs. Designed to run at SessionStart (must be FAST: no tree walks, no ssh).
Refresh the ledger with _inventory_p1p2.py -> _build_ledger.py."""
import os, csv, glob, subprocess, time
from collections import Counter

LEDGER = '/fast/gh/main/phase3/qa_assembly_2/processing_ledger.csv'
DECISIONS = '/lustre/gh/main/clv15/data_decisions.csv'
DECISIONS_FP = '/fast/gh/main/phase3/qa_assembly_2/.data_decisions.fingerprint'
COADD = '/fast/gh/main/phase3/coadd'

def age(path):
    try:
        return (time.time() - os.path.getmtime(path)) / 3600.0
    except OSError:
        return None

print("===== ExoPipe Phase 1/2/3 STATUS (session start) =====")

# ---- Phase 1/2 from the ledger ----
if os.path.exists(LEDGER):
    rows = list(csv.DictReader(open(LEDGER)))
    p1 = Counter(r['p1'] for r in rows)
    p2 = Counter(r['p2'] for r in rows)
    peel = Counter(r['p1_peel'] for r in rows)
    qa = Counter(r['qa_verdict'] for r in rows)
    await_p2 = [f"{r['lst']} {r['date']}" for r in rows if r['p2'] == 'none']
    a = age(LEDGER)
    print(f"[ledger] {len(rows)} (LST,date) cells | refreshed {a:.0f}h ago"
          + ("  *** STALE >24h: rerun _inventory_p1p2.py+_build_ledger.py ***" if a and a > 24 else ""))
    print(f"  P1: {dict(p1)}   P2: {dict(p2)}")
    print(f"  P1 peel-fix: {dict(peel)}   qa_verdict: {dict(qa)}")
    if await_p2:
        print(f"  AWAITING Phase 2 ({len(await_p2)}): {', '.join(await_p2[:20])}")
else:
    print("[ledger] MISSING — run _inventory_p1p2.py + _build_ledger.py")

# ---- data_decisions.csv (CRITICAL exclusion list — review every session) ----
if os.path.exists(DECISIONS):
    drows = list(csv.DictReader(open(DECISIONS)))
    dcells = {(r['lst'], r['date']) for r in drows}
    ndates = len({d for _, d in dcells})
    da = age(DECISIONS)
    # fingerprint = mtime; flag if it changed since last recorded session
    cur = str(os.path.getmtime(DECISIONS))
    prev = open(DECISIONS_FP).read().strip() if os.path.exists(DECISIONS_FP) else None
    changed = (prev is not None and prev != cur)
    try:
        open(DECISIONS_FP, 'w').write(cur)
    except OSError:
        pass
    flag = "  *** CHANGED since last session — reconcile with Gregg ***" if changed else ""
    print(f"[data_decisions] {len(drows)} exclude-rows | {len(dcells)} (lst,date) cells | "
          f"{ndates} dates | edited {da:.0f}h ago{flag}")
    print("  ACTION: confirm with Gregg whether any exclusions changed before co-add/search work")
else:
    print("[data_decisions] MISSING — /lustre/gh/main/clv15/data_decisions.csv not found")

# ---- Phase 3 product inventory ----
if os.path.isdir(COADD):
    cells = [d for d in os.listdir(COADD) if len(d) > 3 and d[2] == 'h' and '_' in d]
    wb = [c for c in cells if c.split('_', 1)[1] in ('Red', 'Green', 'Blue', 'Full')]
    sb = [c for c in cells if c.split('_', 1)[1].endswith('MHz')]
    masters = len(glob.glob(f'{COADD}/*/coadd_final/.done'))
    print(f"[P3 coadd] {len(cells)} cells (wideband {len(wb)}, subband {len(sb)}); "
          f"deep masters built (coadd_final/.done): {masters}")
    # newest activity
    recent = sorted(glob.glob(f'{COADD}/*/coadd_final/.done'), key=os.path.getmtime)
    if recent:
        print(f"  newest deep master: {age(recent[-1]):.0f}h ago")

# ---- running jobs on THIS node ----
try:
    out = subprocess.run(['pgrep', '-af', 'phase2_controller|phase3|tenmin|save_residual|coadd_full'],
                         capture_output=True, text=True, timeout=5).stdout
    procs = [l for l in out.splitlines() if 'pgrep' not in l]
    print(f"[running here] {len(procs)} pipeline procs"
          + (": " + "; ".join(p.split('python')[-1][:60] for p in procs[:6]) if procs else ""))
except Exception:
    pass
print("  (note: batch jobs run on lwacalim02; ssh there to check those)")
print("=====================================================")
