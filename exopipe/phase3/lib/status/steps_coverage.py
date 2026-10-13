"""`status --coverage` / `--gaps` per tier (port of ops/completeness_scan_20260907.py:14-35 and the census classes of
ops/completeness_census_20260821.py:49-60 on the CURRENT roots; exclusions from core/exclusions - ledger 816/817).

Deep: one row per (LST, scope, night) with a Phase 2 deep product (a non-empty <p2>/<LST>/<date>/Science_*/<scope>/I/deep
in any Science run - completeness_scan_20260907.py:22-35). A row is a GAP when it is not excluded and has no Phase 3
residual in the working tree (<COADD_SB_ROOT>/<cell>/subtracted/<date>_I_sub_<scope>.fits): neither co-added nor searched
by omission rather than by a ruling (the scan's definition).
Census class (precedence of completeness_census_20260821.py:49-60, current roots):
  excluded         core.exclusions.is_excluded(lst, scope, date)
  ok               archived epoch image (I) and archived search frames (I and V) of paths.ARCHIVE_RUN
  never-attempted  no installed master I_deep_Taper_R075_<scope>.fits in the working tree
  recover-local    residual in the working tree (the archive lacks a product)
  backfill-stageb  master exists, nothing local
  '+partial-exclusion' is appended (not to 'excluded') when some other scope of the (LST, night) has an exclude row
  (the census used the ledger's dd_status 'partial-excluded', :58-59; here core/exclusions rows, read now).
Sub-LST: one row per (LST, scope): 10-minute windows, search state per Stokes, the LST's published run and its
generation (pos_kind + pertie_detections.csv: sublst_cumulative.py:78-86), served in the hub registry.
Snapshot: one row per night cell of the v9 worklist and the shipped cells: shipped, in a served snapshot run.
"""
import collections
import json
import os
import re

from exopipe.phase3.lib.status import steps_common as K

DATE_RE = re.compile(r'^\d{4}-\d{2}-\d{2}$')

DEEP_COLUMNS = ('lst', 'scope', 'date', 'cell', 'p2_runs', 'excluded', 'partial_exclusion', 'residual',
                'in_build_record', 'in_master', 'master_exists', 'epoch_archived_I', 'frame_archived_I',
                'frame_archived_V', 'searched_V', 'searched_I', 'p1_newer_than_p2', 'census_class', 'gap')
SUBLST_COLUMNS = ('lst', 'scope', 'windows', 'no_windows', 'search_V', 'search_I', 'lst_run', 'lst_run_generation_ok',
                  'served', 'gap', 'reason')
SNAPSHOT_COLUMNS = ('cell', 'lst', 'date', 'in_worklist', 'shipped', 'ship_dir', 'served_run', 'gap', 'reason')


def census_class(r):
    """The census precedence over a coverage row's own columns (used by the writer AND the content test)."""
    if str(r['excluded']) == '1':
        return 'excluded'
    if str(r['epoch_archived_I']) == '1' and str(r['frame_archived_I']) == '1' and str(r['frame_archived_V']) == '1':
        c = 'ok'
    elif str(r['master_exists']) != '1':
        c = 'never-attempted'
    elif str(r['residual']) == '1':
        c = 'recover-local'
    else:
        c = 'backfill-stageb'
    if str(r['partial_exclusion']) == '1':
        c += '+partial-exclusion'
    return c


def gap_of(r):
    return int(str(r['excluded']) != '1' and str(r['residual']) != '1' and int(r['p2_runs'] or 0) > 0)


# ---------------------------------------------------------------------------------------------------------------------
# deep
# ---------------------------------------------------------------------------------------------------------------------

def deep_universe(cells_spec):
    from exopipe.phase3.core import cells
    allc = list(cells.all_cells())
    return K.glob_cells(allc, cells_spec) if cells_spec else allc


def deep_inputs():
    C = K.cfg()
    return [C.paths.PHASE2_ROOT, C.paths.COADD_SB_ROOT]


def _p2_night(args):
    """{scope: [Science runs with a non-empty I/deep]} for one (LST, date)."""
    lst, date, scopes, p2 = args
    C = K.cfg()
    out = collections.defaultdict(list)
    dd = os.path.join(p2, lst, date)
    for run in K.listdir(dd):
        if not run.startswith(C.status_coverage.PHASE2_RUN_PREFIX):
            continue
        rd = os.path.join(dd, run)
        for sc in K.listdir(rd):
            if sc in scopes:
                d = os.path.join(rd, sc, 'I', 'deep')
                if K.listdir(d):
                    out[sc].append(run)
    return (lst, date), dict(out)


def _archive_night(args):
    """{(scope, 'epoch_I'|'frame_I'|'frame_V'): 1} for one (LST, date) of the current archive run (build_qa_table.py:172-181)."""
    lst, date, scopes = args
    C = K.cfg()
    rd = os.path.join(C.paths.PHASE3_ARCHIVE_ROOT, lst, date, C.paths.ARCHIVE_RUN)
    out = {}
    for sc in K.listdir(rd):
        if sc not in scopes:
            continue
        for st in ('I', 'V'):
            for b in K.listdir(os.path.join(rd, sc, st, 'deep')):
                if not b.endswith('.fits'):
                    continue
                if '_msub' in b and '_pbcorr' not in b:
                    out[(sc, f'frame_{st}')] = 1
                elif not any(t in b for t in ('_msub', '_pbcorr', '_loo')):
                    out[(sc, f'epoch_{st}')] = 1
    return (lst, date), out


def _newest_suffix(names, prefix):
    s = [n[len(prefix):] for n in names if n.startswith(prefix)]
    return max(s) if s else ''


def searched_dates(label, cells_wanted):
    """{(cell, Stokes): set(dates)} from the published attempts of the deep run label (p3s3_sources.published_attempts
    and each attempt's v_deep_images.csv, as build_qa_table.py:156-162 reads them). In place; P3S_LABEL set from config
    for the import (p3s3_sources reads it at import)."""
    from exopipe.phase3.lib.status import steps_inplace as IP
    S = IP.import_quiet('p3s3_sources', env={'P3S_LABEL': label})
    atts = [a for a in S.published_attempts(['V', 'I']) if a['harvested'] and a['cell'] in cells_wanted]

    def one(a):
        p = f"{a['dir']}/transients/Run_{S.LABEL}/v_deep_images.csv"
        try:
            return (a['cell'], a['stokes']), {r['date'][:10] for r in K.read_csv(p)}
        except OSError:
            return (a['cell'], a['stokes']), set()
    return dict(K.thread_map(one, atts)), len(atts), S.STATE


def deep_coverage(cells_spec, ex, workers=None, archive=None, searched=None, note=print):
    """Rows of the deep coverage table for the selected cells."""
    C = K.cfg()
    CC = C.status_coverage
    archive = CC.ARCHIVE_COLUMNS if archive is None else archive
    searched = CC.SEARCHED_COLUMNS if searched is None else searched
    cells = deep_universe(cells_spec)
    by_lst = collections.defaultdict(set)
    for c in cells:
        l, s = c.split('_', 1)
        by_lst[l].add(s)
    p2root = C.paths.PHASE2_ROOT
    jobs = [(l, d, by_lst[l], p2root) for l in sorted(by_lst) for d in K.listdir(os.path.join(p2root, l))
            if DATE_RE.match(d)]
    p2 = dict(K.thread_map(_p2_night, jobs, workers))
    note(f'coverage: Phase 2 walked {len(jobs)} (LST, night)s for {len(cells)} cells')
    root = C.paths.COADD_SB_ROOT
    resid, build, inm, master = {}, {}, {}, {}

    def cell_state(c):
        l, s = c.split('_', 1)
        sd = os.path.join(root, c, 'subtracted')
        suf = f'_I_sub_{s}.fits'
        res = {f.split('_')[0] for f in K.listdir(sd) if f.endswith(suf)}
        w = os.path.join(root, c, 'coadd_final', C.status_rebuild.BUILD_RECORD)
        rows = K.read_csv(w) if os.path.exists(w) else []
        m = os.path.exists(K.expand(CC.MASTER, lst=l, scope=s))
        return c, res, {r['date'] for r in rows}, {r['date'] for r in rows if str(r.get('in_master')) == '1'}, m
    for c, res, b, im, m in K.thread_map(cell_state, cells, workers):
        resid[c], build[c], inm[c], master[c] = res, b, im, m
    arch = {}
    if archive:
        ajobs = [(l, d, by_lst[l]) for (l, d), v in p2.items() if v]
        arch = dict(K.thread_map(_archive_night, ajobs, workers))
        note(f'coverage: archive {C.paths.ARCHIVE_RUN} walked for {len(ajobs)} (LST, night)s')
    srch = {}
    if searched:
        srch, n_att, state = searched_dates(C.status.SEARCH_LABEL['deep'], set(cells))
        note(f'coverage: searched columns from {n_att} published attempts ({state})')
    p1new = {}
    if CC.P1_NEWER_THAN_P2:
        def p1n(key):
            l, d = key
            r1 = _newest_suffix(K.listdir(os.path.join(C.paths.PHASE1_ROOT, l, d)), 'Run_')
            r2 = _newest_suffix(K.listdir(os.path.join(p2root, l, d)), CC.PHASE2_RUN_PREFIX)
            return key, int(bool(r1 and r2 and r1 > r2))
        p1new = dict(K.thread_map(p1n, [k for k, v in p2.items() if v], workers))
    partial_nights = collections.defaultdict(set)
    for (l, s, d) in ex.rows:
        partial_nights[(l, d)].add(s)
    rows = []
    for (l, d), scopes in sorted(p2.items()):
        for s, runs in sorted(scopes.items()):
            c = f'{l}_{s}'
            if c not in resid:
                continue
            exc = int(ex.is_excluded(l, s, d))
            a = arch.get((l, d), {})
            r = dict(lst=l, scope=s, date=d, cell=c, p2_runs=len(runs), excluded=exc,
                     partial_exclusion=int(not exc and bool(partial_nights.get((l, d)))),
                     residual=int(d in resid[c]), in_build_record=int(d in build[c]), in_master=int(d in inm[c]),
                     master_exists=int(master[c]),
                     epoch_archived_I=a.get((s, 'epoch_I'), 0) if archive else '',
                     frame_archived_I=a.get((s, 'frame_I'), 0) if archive else '',
                     frame_archived_V=a.get((s, 'frame_V'), 0) if archive else '',
                     searched_V=int(d in srch.get((c, 'V'), ())) if searched else '',
                     searched_I=int(d in srch.get((c, 'I'), ())) if searched else '',
                     p1_newer_than_p2=p1new.get((l, d), '') if CC.P1_NEWER_THAN_P2 else '')
            r['census_class'] = census_class(r)
            r['gap'] = gap_of(r)
            rows.append(r)
    return rows, cells


# ---------------------------------------------------------------------------------------------------------------------
# sub-LST
# ---------------------------------------------------------------------------------------------------------------------

def sublst_inputs():
    CC = K.cfg().status_coverage
    return [K.expand(CC.SUBLST_WINDOWS), K.expand(CC.SUBLST_SEARCH_STATE)]


def sublst_generation(run_dir):
    """(ok, reason) of a published sub-LST LST run: pertie_detections.csv present and master_catalog.csv has pos_kind
    (sublst_cumulative.py:78-86, exopipe_ops 444656a) - a header read, the script is not imported (it exits)."""
    import csv
    CC = K.cfg().status_coverage
    if not os.path.exists(os.path.join(run_dir, CC.SUBLST_REQUIRED_FILE)):
        return False, f'no {CC.SUBLST_REQUIRED_FILE} (catalogued before the window tie)'
    mc = os.path.join(run_dir, 'master_catalog.csv')
    try:
        with open(mc, newline='') as fh:
            hdr = next(csv.reader(fh), [])
    except OSError:
        return False, 'no master_catalog.csv'
    if CC.SUBLST_REQUIRED_COLUMN not in hdr:
        return False, f'master_catalog.csv has no {CC.SUBLST_REQUIRED_COLUMN} column (catalogued before fitted positions)'
    return True, ''


def served_sublst_lsts():
    """LSTs the hub registry says the served sub-LST run covers (its generation text 'LSTs: 00h 01h ...')."""
    try:
        _, reg = K.registry()
    except (OSError, ValueError):
        return set(), ''
    keys = K.cfg().status.REGISTRY_KEYS.get('sublst', ())
    for S in ('V', 'I'):
        for k in keys:
            g = ((reg.get(S) or {}).get(k) or {}).get('generation', '')
            if 'LSTs:' in g:
                return set(re.findall(r'\b\d\dh\b', g.split('LSTs:', 1)[1])), g
    return set(), ''


def sublst_coverage(lsts_spec):
    from exopipe.phase3.core import cells
    C = K.cfg()
    CC = C.status_coverage
    win = K.expand(CC.SUBLST_WINDOWS)
    state = K.expand(CC.SUBLST_SEARCH_STATE)
    lsts = K.glob_cells(list(cells.LSTS), lsts_spec) if lsts_spec else list(cells.LSTS)
    st = {}
    for kind in ('done', 'claims', 'failed'):
        for n in K.listdir(os.path.join(state, kind)):
            st.setdefault(n, set()).add(kind)
    served, _ = served_sublst_lsts()
    serve = getattr(C, 'serve', None)
    pat = getattr(serve, 'SUBLST_LST_RUN', 'Run_SLST{hh}B')
    rows = []
    for l in lsts:
        rd = os.path.join(K.runs_root('sublst'), pat.format(hh=l[:2]))
        has_run = os.path.isdir(rd)
        gen_ok, why = sublst_generation(rd) if has_run else (False, 'no published LST run')
        for s in C.instrument.ALL_SUBBANDS:
            wd = os.path.join(win, f'{l}_{s}')
            r = dict(lst=l, scope=s, windows=int(os.path.exists(os.path.join(wd, 'frames.csv'))),
                     no_windows=int(os.path.exists(os.path.join(wd, 'NO_WINDOWS.txt'))),
                     lst_run=os.path.basename(rd) if has_run else '', lst_run_generation_ok=int(gen_ok),
                     served=int(l in served))
            for S in ('V', 'I'):
                k = st.get(f'{l}_{s}:{S}', set())
                r[f'search_{S}'] = 'done' if 'done' in k else ('failed' if 'failed' in k else ('claimed' if 'claims' in k
                                                                                                else ''))
            reasons = []
            if r['windows'] and not (r['search_V'] == 'done' and r['search_I'] == 'done'):
                reasons.append('search not done')
            if r['windows'] and not has_run:
                reasons.append('LST not published')
            elif r['windows'] and not gen_ok:
                reasons.append('LST run of an older generation: ' + why)
            if r['windows'] and has_run and not r['served']:
                reasons.append('LST not in the served cumulative run')
            r['gap'] = int(bool(reasons))
            r['reason'] = '; '.join(reasons)
            rows.append(r)
    return rows, lsts


# ---------------------------------------------------------------------------------------------------------------------
# snapshot
# ---------------------------------------------------------------------------------------------------------------------

def snapshot_inputs():
    C = K.cfg()
    return [C.paths.SNAPSHOT_FLEET_OUT, K.expand(C.status_snapshot.FLEET_CLAIMS)]


def _cells_of(path):
    out = []
    try:
        for l in open(path):
            p = l.split()
            if not p or p[0].startswith('#'):
                continue
            out.append(p[0] if len(p) == 1 else f'{p[0]}_{p[1]}')
    except OSError:
        pass
    return out


def snapshot_coverage(cells_spec):
    C = K.cfg()
    wl = K.expand(C.status_coverage.SNAPSHOT_WORKLIST)
    claims = K.expand(C.status_snapshot.FLEET_CLAIMS)
    ship = C.paths.SNAPSHOT_FLEET_OUT
    work = set(_cells_of(wl))
    shipped_dirs = {c for c in K.listdir(ship) if re.match(r'^\d\dh_\d{4}-\d\d-\d\d$', c)}
    universe = sorted(work | shipped_dirs | {c for c in K.listdir(claims) if re.match(r'^\d\dh_\d{4}-\d\d-\d\d$', c)})
    if cells_spec:
        universe = K.glob_cells(universe, cells_spec)
    served = {}
    for run in served_snapshot_runs():
        p = os.path.join(K.runs_root('snapshot'), run, 'chain_summary.json')
        try:
            for c in json.load(open(p)).get('cells', []):
                served.setdefault(c, []).append(run)
        except (OSError, ValueError):
            pass
    rows = []
    for c in universe:
        l, d = c.split('_', 1)
        r = dict(cell=c, lst=l, date=d, in_worklist=int(c in work),
                 shipped=int(os.path.exists(os.path.join(claims, c, 'done'))), ship_dir=int(c in shipped_dirs),
                 served_run=';'.join(sorted(served.get(c, []))))
        reasons = []
        if r['in_worklist'] and not r['shipped']:
            reasons.append('in the worklist, not shipped')
        r['gap'] = int(bool(reasons))
        r['reason'] = '; '.join(reasons)
        rows.append(r)
    return rows, universe


def served_snapshot_runs():
    try:
        return sorted(K.served_runs('snapshot'))
    except (OSError, ValueError):
        return []
