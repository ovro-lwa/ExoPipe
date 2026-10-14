"""`status --served-diff A B` - what changes for the served list between two runs of one tier (port of
ops/served_change_20261006.py:21-75, generalised to any two runs; its 2026-10-06 elevation-gate arm (:61-64, :76-81) is
specific to that day and not ported).

A = the run the hub registry serves for the tier (default) or a label / directory; B = a label / directory (required).
Served list of a run = the sources of status.SERVED_CATALOG_DIRS[tier] (*_sources_survivors.csv), keyed per Stokes by
identity_id (else source_id) as served_change_20261006.py:44-48 keys them; members from the matching
*_source_members.csv; detections from master_catalog_asrun.csv (else master_catalog.csv).
Fate of each member detection of a served source of A:
  night_rejected  its night now fails the cell's night-image gate or epoch gate (current cell records; served_change :23-32)
  kept            a member of some served source of B (same Stokes)
  research        its cell was searched anew for B (the cell's detection set differs between A's and B's catalogues) and
                  the detection is not in B: replaced by the fresh search (served_change 'research', :57)
  removed         present in B's catalogue but in no served source of B (B's gates removed it)
  lost            in neither B's catalogue nor a re-searched cell (the cell's detections are unchanged): counted leaving
Source status (served_change :66-70): gone = every member leaves (night_rejected, removed, lost); reduced = some leave;
're-searched cell' = none leaves and some are research; unchanged otherwise; 'new' = a source of B with no member of A.
Rows: one per source (A's sources with their status; B's new sources), with judged labels (judgements_<S>/judgements.jsonl,
action 'classify') where Gregg judged the source. served_change wrote rows only for judged sources (:72-75); here every
source has a row and the 'label' column says which are judged.
"""
import collections
import glob
import json
import os

from exopipe.phase3.lib.status import steps_common as K

COLUMNS = ('stokes', 'source', 'side', 'status', 'label', 'detections', 'kept', 'night_rejected', 'research', 'removed',
           'lost')
LEAVING = ('night_rejected', 'removed', 'lost')


def catalogue(run_dir):
    for name in ('master_catalog_asrun.csv', 'master_catalog.csv'):
        p = os.path.join(run_dir, name)
        if os.path.exists(p):
            return p
    return None


def served_sources(run_dir, tier):
    """{(S, source key): [cand_id]}, files read."""
    C = K.cfg()
    out, files = collections.defaultdict(list), []
    for S in ('I', 'V'):
        for dpat in C.status.SERVED_CATALOG_DIRS.get(tier, ('source_catalogs_{S}',)):
            for p in sorted(glob.glob(os.path.join(run_dir, dpat.format(S=S), '*_sources_survivors.csv'))):
                mp = p.replace('_sources_survivors.csv', '_source_members.csv')
                if not os.path.exists(mp):
                    continue
                files += [p, mp]
                ids = {r['source_id']: (r.get('identity_id') or r['source_id']) for r in K.read_csv(p)}
                for r in K.read_csv(mp):
                    if r.get('source_id') in ids and r.get('cand_id'):
                        out[(S, ids[r['source_id']])].append(r['cand_id'])
    return out, files


def judgements(run_dir):
    out = {}
    for S in ('I', 'V'):
        jp = os.path.join(run_dir, f'judgements_{S}', 'judgements.jsonl')
        if not os.path.exists(jp):
            continue
        for ln in open(jp):
            try:
                j = json.loads(ln)
            except ValueError:
                continue
            if j.get('action') == 'classify':
                out[(S, j.get('source_id'))] = j.get('label') or j.get('judge_confidence') or 'judged'
    return out


def cell_of_cand(cid):
    p = cid.split('_')
    return (p[0], p[1], p[2]) if len(p) >= 3 and p[0].endswith('h') else (None, None, None)


def served_diff(run_a, run_b, tier):
    """(rows, summary dict, inputs) of the served diff A -> B."""
    from exopipe.phase3.core.record import NoInput, Refusal
    ta, tb = K.run_tier(run_a), K.run_tier(run_b)
    if ta != tb:
        raise Refusal(f'S-G7: --served-diff runs of different tiers ({os.path.basename(run_a)} = {ta}, '
                      f'{os.path.basename(run_b)} = {tb})')
    A, fa = served_sources(run_a, tier)
    B, fb = served_sources(run_b, tier)
    for run, s in ((run_a, A), (run_b, B)):
        if not s:
            raise NoInput(f'S-G7: {run} has no served catalogue ({", ".join(K.cfg().status.SERVED_CATALOG_DIRS.get(tier, ()))}'
                          f'/*_sources_survivors.csv with members)')
    ca, cb = catalogue(run_a), catalogue(run_b)
    raw_a = {r['cand_id'] for r in K.read_csv(ca)} if ca else set()
    raw_b = {r['cand_id'] for r in K.read_csv(cb)} if cb else set()
    cells_a, cells_b = collections.defaultdict(set), collections.defaultdict(set)
    for c in raw_a:
        l, d, b = cell_of_cand(c)
        if l:
            cells_a[f'{l}_{b}'].add(c)
    for c in raw_b:
        l, d, b = cell_of_cand(c)
        if l:
            cells_b[f'{l}_{b}'].add(c)
    researched = {cell for cell in set(cells_a) | set(cells_b) if cells_a.get(cell) != cells_b.get(cell)}
    in_b = collections.defaultdict(set)
    for (S, _), cids in B.items():
        in_b[S].update(cids)
    root = K.cfg().paths.COADD_SB_ROOT
    fails, gate_unread = {}, []

    def rejected(cell, night):
        if cell not in fails:
            f = set()
            fin = os.path.join(root, cell, 'coadd_final')
            from exopipe.phase3.lib.masters import night_image_gate as NG, epoch_gate as EG
            for gname, fn in (('night_image_gate', lambda: NG.failed_dates(fin)),
                              ('epoch_gate', lambda: EG.failed_dates(fin, cell.split('_', 1)[1]))):
                try:
                    f |= set(fn())
                except Exception as e:                # noqa: BLE001 - served_change :27-30 skips it; counted (S-G16)
                    gate_unread.append(f'{cell} {gname}: {type(e).__name__}: {e}'[:200])
            fails[cell] = f
        return night[:10] in fails[cell]
    jud = judgements(run_a)
    jud.update({k: v for k, v in judgements(run_b).items() if k not in jud})
    rows, summ = [], collections.Counter()
    for (S, sid), cids in sorted(A.items()):
        fs = collections.Counter()
        for c in cids:
            l, d, b = cell_of_cand(c)
            cell = f'{l}_{b}' if l else ''
            if l and rejected(cell, d):
                fs['night_rejected'] += 1
            elif c in in_b[S]:
                fs['kept'] += 1
            elif cell in researched and c not in raw_b:
                fs['research'] += 1
            elif c in raw_b:
                fs['removed'] += 1
            else:
                fs['lost'] += 1
        leaving = sum(fs[k] for k in LEAVING)
        status = ('gone' if leaving == len(cids) else ('reduced' if leaving else
                                                        ('re-searched cell' if fs['research'] else 'unchanged')))
        summ[(S, status)] += 1
        rows.append(dict(stokes=S, source=sid, side='A', status=status, label=jud.get((S, sid), ''), detections=len(cids),
                         **{k: fs.get(k, 0) for k in ('kept', 'night_rejected', 'research', 'removed', 'lost')}))
    in_a = collections.defaultdict(set)
    for (S, _), cids in A.items():
        in_a[S].update(cids)
    for (S, sid), cids in sorted(B.items()):
        if not any(c in in_a[S] for c in cids):
            summ[(S, 'new')] += 1
            rows.append(dict(stokes=S, source=sid, side='B', status='new', label=jud.get((S, sid), ''),
                             detections=len(cids), kept=0, night_rejected=0, research=0, removed=0, lost=0))
    summary = {'A': run_a, 'B': run_b, 'tier': tier, 'researched_cells': len(researched),
               'gate_records_unread': len(gate_unread), 'gate_records_unread_examples': gate_unread[:20],
               'counts': {f'{S} {st}': n for (S, st), n in sorted(summ.items())},
               'served_A': {S: sum(1 for (s, _) in A if s == S) for S in ('I', 'V')},
               'served_B': {S: sum(1 for (s, _) in B if s == S) for S in ('I', 'V')}}
    return rows, summary, fa + fb + [p for p in (ca, cb) if p]


def summary_text(summary):
    L = [f"served diff {os.path.basename(summary['A'])} -> {os.path.basename(summary['B'])} (tier {summary['tier']}; "
         f"{summary['researched_cells']} cells searched anew; {summary.get('gate_records_unread', 0)} cell gate records "
         f"unreadable - their nights not judged night_rejected)"]
    c = summary['counts']
    for S in ('I', 'V'):
        L.append(f"Stokes {S}: {summary['served_A'][S]} served sources in A - unchanged {c.get(f'{S} unchanged', 0)}, "
                 f"reduced {c.get(f'{S} reduced', 0)}, gone {c.get(f'{S} gone', 0)}, in re-searched cells "
                 f"{c.get(f'{S} re-searched cell', 0)}; {summary['served_B'][S]} in B, new {c.get(f'{S} new', 0)}")
    return '\n'.join(L) + '\n'
