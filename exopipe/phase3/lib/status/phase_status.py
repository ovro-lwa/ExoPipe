#!/usr/bin/env python3
"""Fast Phase 1/2/3 status snapshot for the ExoPipe pipeline.
Reads the cached processing ledger + counts Phase-3 products + local running
jobs. Designed to run at SessionStart (must be FAST: no tree walks, no ssh).
Refresh the ledger with _inventory_p1p2.py -> _build_ledger.py."""
import os, csv, subprocess, time
from collections import Counter

LEDGER = '/lustre/gh/main/clv15/processing_ledger.csv'
DECISIONS = '/lustre/gh/main/clv15/data_decisions.csv'
DECISIONS_FP = '/fast/gh/main/phase3/qa_assembly_2/.data_decisions.fingerprint'
# Current-generation co-add roots, in the same precedence the consumers use
# (exopipe_ops/resolve_master_root.sh). /fast/gh/main/phase3/coadd is the
# SUPERSEDED tree and is deliberately absent from this list.
COADD_ROOTS = ['/fast/gh/main/phase3/rebuild_w2',
               '/fast/gh/main/phase3/rebuild_v2',
               '/fast/gh/main/phase3/rebuild_13h',
               '/data07/gh/phase3/coadd_sb']
RETIRED_COADD = '/fast/gh/main/phase3/coadd'
# The installed Stokes-I master of a cell (<root>/<cell>/coadd_final/). FIXED 2026-10-14 (refactor review, masters
# finding): a master is counted from the FILE - this name carrying a COADDOP card (the co-add operator every build
# writes, e.g. 'ivw-mean-v3') - and "newest deep master" is that file's mtime. The coadd_final/.done marker this
# report counted before is never evidence (CLAUDE.md: no marker files; 0 markers exist in the deep tree on 2026-10-14
# against 456 masters).
MASTER = 'I_deep_Taper_R075_{scope}.fits'

def age(path):
    try:
        return (time.time() - os.path.getmtime(path)) / 3600.0
    except OSError:
        return None

def header_card(path, key, max_blocks=100):
    """The value text of ``key`` in the primary FITS header of ``path`` (stdlib only: no astropy at SessionStart);
    None when the card is absent or the file unreadable."""
    try:
        with open(path, 'rb') as f:
            for _ in range(max_blocks):
                blk = f.read(2880)
                if len(blk) < 2880:
                    return None
                for i in range(0, 2880, 80):
                    card = blk[i:i + 80].decode('ascii', 'replace')
                    k = card[:8].strip()
                    if k == 'END':
                        return None
                    if k == key and card[8:10] == '= ':
                        v = card[10:].strip()
                        if v.startswith("'"):
                            return v[1:].split("'", 1)[0].strip()
                        return v.split('/', 1)[0].strip()
    except OSError:
        return None
    return None

def master_inventory(root, cells):
    """The masters of ``cells`` under ``root``: {'masters': n carrying COADDOP, 'coaddop': {value: n},
    'without_coaddop': n files without the card, 'newest': (mtime, path) of the newest counted master or None}."""
    ops, n_bad, newest = Counter(), 0, None
    for c in cells:
        p = os.path.join(root, c, 'coadd_final', MASTER.format(scope=c.split('_', 1)[1]))
        if not os.path.exists(p):
            continue
        op = header_card(p, 'COADDOP')
        if not op:
            n_bad += 1
            continue
        ops[op] += 1
        try:
            m = os.path.getmtime(p)
        except OSError:
            continue
        if newest is None or m > newest[0]:
            newest = (m, p)
    return {'masters': sum(ops.values()), 'coaddop': dict(ops), 'without_coaddop': n_bad, 'newest': newest}

def main():
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
        # FIXED 2026-09-02 (found by the follow-up session): this printed
        # len(drows) as "exclude-rows", counting every row in the file.
        # 425 of them are verdict=keep (the READMITTED-2026-08-30 set), so
        # the exclusion count was overstated by ~24% in every session-start
        # report. Count verdicts explicitly, and count a cell as excluded
        # only if it carries at least one exclude row.
        n_excl = sum(1 for r in drows if r.get('verdict') == 'exclude')
        n_keep = sum(1 for r in drows if r.get('verdict') == 'keep')
        n_other = len(drows) - n_excl - n_keep
        excl_cells = {(r['lst'], r['date']) for r in drows
                      if r.get('verdict') == 'exclude'}
        ndates = len({d for _, d in excl_cells})
        print(f"[data_decisions] {n_excl} exclude-rows"
              + (f" (+{n_keep} keep" + (f", {n_other} other" if n_other else "")
                 + f"; {len(drows)} rows total)" if n_keep or n_other else "")
              + f" | {len(excl_cells)} excluded (lst,date) cells | "
              f"{ndates} dates | edited {da:.0f}h ago{flag}")
        print("  ACTION: confirm with Gregg whether any exclusions changed before co-add/search work")
    else:
        print("[data_decisions] MISSING — /lustre/gh/main/clv15/data_decisions.csv not found")

    # ---- Phase 3 product inventory ----
    # FIXED 2026-09-02 (found by the follow-up session): this counted only
    # /fast/gh/main/phase3/coadd — the SUPERSEDED sigma-clipped tree — so
    # every session-start report described a retired generation as the
    # Phase-3 inventory. Same stale-root class as the ledger row
    # FIX-CONSUMER-MASTER-ROOT-STALE-COADD-TREE. Count across the
    # current-generation roots instead, and say so if the retired tree is
    # still on disk.
    # Each cell is counted in the first root (consumer precedence) that holds it; its master is that root's file.
    seen = set()
    tot_cells = tot_masters = n_no_op = 0
    ops = Counter()
    newest = None
    per_root = []
    for root in COADD_ROOTS:
        if not os.path.isdir(root):
            continue
        cells = [d for d in os.listdir(root)
                 if len(d) > 3 and d[2] == 'h' and '_' in d]
        new = [c for c in cells if c not in seen]
        seen.update(cells)
        inv = master_inventory(root, new)
        tot_cells += len(new)
        tot_masters += inv['masters']
        n_no_op += inv['without_coaddop']
        ops.update(inv['coaddop'])
        if inv['newest'] and (newest is None or inv['newest'][0] > newest[0]):
            newest = inv['newest']
        per_root.append(f"{os.path.basename(root)} {len(new)}")
    if per_root:
        wb = [c for c in seen if c.split('_', 1)[1] in ('Red', 'Green', 'Blue', 'Full')]
        sb = [c for c in seen if c.split('_', 1)[1].endswith('MHz')]
        print(f"[P3 coadd] {len(seen)} distinct cells "
              f"(wideband {len(wb)}, subband {len(sb)}) across "
              + ", ".join(per_root)
              + f"; masters ({MASTER.format(scope='<scope>')} carrying COADDOP): {tot_masters} {dict(ops)}"
              + (f"; {n_no_op} without COADDOP" if n_no_op else ""))
        print("  NOTE: COADDOP names the co-add operator only — "
              "run exopipe_ops/verify_master_generation.py for the generation + loo/")
        if newest:
            print(f"  newest deep master: {age(newest[1]):.0f}h ago "
                  f"({os.path.basename(os.path.dirname(os.path.dirname(newest[1])))})")
    if os.path.isdir(RETIRED_COADD):
        print(f"  *** RETIRED TREE STILL ON DISK: {RETIRED_COADD} — "
              f"superseded sigma-clipped generation; no consumer may select it ***")

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


if __name__ == '__main__':
    main()
