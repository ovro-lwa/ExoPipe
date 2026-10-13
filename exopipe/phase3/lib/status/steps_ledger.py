"""`status ledger` - the Phase 1/2 inventory and the processing ledger: ops/_inventory_p1p2.py then ops/_build_ledger.py
run IN PLACE in this process (module-level scripts, runpy) with their literal paths redirected (spec 7.2 'remap'):
/tmp/_inventory_p1p2.pkl (node-local hand-off, edge case E22) -> <out>/status/_inventory_p1p2.pkl and the ledger
/lustre/gh/main/clv15/processing_ledger.csv -> <out>/clv15/processing_ledger.csv. Whole survey only (S-G10).
`--stale ledger` runs the same into --out and lists the (LST, night)s whose row differs from the production ledger.
"""
import os

from exopipe.phase3.lib.status import steps_common as K

KEY_COLUMNS = ('p1', 'p1_nsub', 'p1_peel', 'p2', 'p2_nsub', 'p2_nwide', 'p2_state', 'dd_excl_subbands', 'dd_status',
               'qa_verdict')


def run_ledger(out, unit, log, allowed):
    from exopipe.phase3.lib.status import steps_inplace as IP
    C = K.cfg()
    L = C.status_ledger
    pkl = K.check_write(K.place(out, 'status', os.path.basename(L.LEGACY_HANDOFF)))
    led = K.check_write(K.place(out, 'clv15', os.path.basename(C.status.LEDGER)))
    for p in (pkl, led):
        os.makedirs(os.path.dirname(p), exist_ok=True)
    mapping = {L.LEGACY_HANDOFF: pkl, C.status.LEDGER: led}
    for p in (C.paths.PHASE1_ROOT, C.paths.PHASE2_ROOT, L.BATCH_PROGRESS_DIR):
        unit.meta.setdefault('roots_walked', []).append(p)
    rc1, t1 = IP.run_path(IP.legacy_script('_inventory_p1p2.py'), [], mapping=mapping, allowed_roots=allowed, log=log,
                          unit=unit)
    if rc1 != 0:
        return rc1, t1, led
    rc2, t2 = IP.run_path(IP.legacy_script('_build_ledger.py'), [], mapping=mapping, allowed_roots=allowed, log=log,
                          unit=unit)
    return rc2, t1[-2000:] + '\n' + t2, led


def ledger_diff(fresh, production):
    """Rows (item = <LST>_<date>) whose key columns differ between the fresh ledger and the production one, plus nights
    present in one only."""
    a = {(r['lst'], r['date']): r for r in K.read_csv(production)} if os.path.exists(production) else {}
    b = {(r['lst'], r['date']): r for r in K.read_csv(fresh)}
    rows = []
    for k in sorted(set(a) | set(b)):
        if k not in a:
            why = 'night missing from the production ledger'
        elif k not in b:
            why = 'night in the production ledger only'
        else:
            ch = [c for c in KEY_COLUMNS if str(a[k].get(c, '')) != str(b[k].get(c, ''))]
            if not ch:
                continue
            why = 'changed: ' + ', '.join(f'{c} {a[k].get(c, "")}->{b[k].get(c, "")}' for c in ch[:4])
        rows.append(dict(item=f'{k[0]}_{k[1]}', kind='epoch', status='stale', reason=why, step='status',
                         root=production, detail=''))
    return rows
