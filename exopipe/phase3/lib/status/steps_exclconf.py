"""`status --exclusion-conformance` - products and served rows on excluded (LST, night, scope), per tier
(CLAUDE.md 2: exclusion is absolute and total; GATE-BUILT-COADD-EXCLUSION-CONFORMANCE (ledger 1535);
IMPLEMENTED-SNAPSHOT-SERVE-EXCLUSION-GATE-20261013 (f1996ee)).

Three parts (deep; sub-LST and snapshot have the served part only):
  master-archive   ops/coadd_exclusion_conformance.py IN PLACE (child, core.legacy.run) on the archive generation
                   status_conformance.MASTERS_ARCHIVE_ROOT, CONF_OUT -> <out>/QA/coadd_exclusion_conformance.csv
                   (production table format). Its exit 1 = breaches found: recorded, exit 2 only with --strict.
  master-working   NEW: the installed working-tree masters (<COADD_SB_ROOT>/<cell>/coadd_final/WORKING_MASTERS): every
                   'COADD constituent' HISTORY line judged with core.exclusions now; BREACH when the newest
                   data_decisions backup at or before the master's mtime already excluded it (judged with core.exclusions
                   on that backup), STALE when it did not, UNKNOWN-EPOCH when no backup is that old (the snapshot choice
                   of coadd_exclusion_conformance.py:47-80). The backup is judged with the co-add rule (a band master
                   is excluded by its member subbands), as the production script does since exopipe_ops c1e4a5d (its
                   excluded_at looked the BAND name up among the backup's subband rows, so a band-master constituent
                   excluded before the build read STALE, never BREACH - refactor finding, fixed 2026-10-14).
  served-run       every CSV of each run the hub registry serves for the tier (judgements/ skipped): each row keyed by
                   snapshot_serve_exclusion_gate.keys_of (in place; its main() is NOT called - it writes into the run dir),
                   plus lst/date/scope rows (search image tables); each key judged with core.exclusions now.
"""
import collections
import glob
import os
import re
import sys
import time

from exopipe.phase3.lib.status import steps_common as K

COLUMNS = ('part', 'run', 'file', 'row_id', 'lst', 'date', 'band', 'verdict', 'note')
ARCHIVE_COLUMNS = ('lst', 'band', 'kind', 'product', 'date', 'verdict', 'note')   # coadd_exclusion_conformance.py:84-85


# ---------------------------------------------------------------------------------------------------------------------
# DD snapshots (BREACH vs STALE)
# ---------------------------------------------------------------------------------------------------------------------

_SNAPS = None


def dd_snapshots():
    """[(YYYYMMDD, Exclusions)] of the dated data_decisions backups (status_conformance.DD_HISTORY_GLOB), oldest first,
    each read with core.exclusions (the one rule)."""
    global _SNAPS
    if _SNAPS is None:
        from exopipe.phase3.core.exclusions import Exclusions
        out = []
        for b in sorted(glob.glob(K.cfg().status_conformance.DD_HISTORY_GLOB)):
            m = re.search(r'bak_(\d{8})', b)
            if not m:
                continue
            try:
                out.append((m.group(1), Exclusions(b)))
            except Exception:                         # noqa: BLE001 - an unreadable backup is not a snapshot
                continue
        _SNAPS = sorted(out, key=lambda t: t[0])
    return _SNAPS


def verdict_at(lst, scope, date, when_yyyymmdd):
    prev = [e for d, e in dd_snapshots() if d <= when_yyyymmdd]
    if not prev:
        return 'UNKNOWN-EPOCH'
    return 'BREACH' if prev[-1].is_excluded(lst, scope, date) else 'STALE'


# ---------------------------------------------------------------------------------------------------------------------
# masters
# ---------------------------------------------------------------------------------------------------------------------

def working_tree_audit(cells, ex, workers=None):
    """Rows of the working-tree master audit for the selected deep cells."""
    C = K.cfg()
    root = C.paths.COADD_SB_ROOT

    def one(cell):
        lst, scope = cell.split('_', 1)
        out = []
        for pat in C.status_conformance.WORKING_MASTERS:
            m = os.path.join(root, cell, 'coadd_final', pat.format(scope=scope))
            if not os.path.exists(m):
                continue
            try:
                con = K.coadd_constituents(m)
            except OSError as e:
                out.append(dict(part='master-working', run='working-tree', file=m, row_id=os.path.basename(m), lst=lst,
                                date='', band=scope, verdict='UNREADABLE', note=str(e)[:80]))
                continue
            when = time.strftime('%Y%m%d', time.gmtime(os.path.getmtime(m)))
            for d in con:
                if ex.is_excluded(lst, scope, d):
                    out.append(dict(part='master-working', run='working-tree', file=m, row_id=os.path.basename(m),
                                    lst=lst, date=d, band=scope, verdict=verdict_at(lst, scope, d, when),
                                    note=f'product written {when}'))
        return out
    rows = []
    for r in K.thread_map(one, cells, workers):
        rows.extend(r)
    return rows


def coverage_gaps(cells, ex, workers=None):
    """Rows of the masters' coverage - the other direction of working_tree_audit (B02_DESIGN.md "Step 3c, first review"):
    per deep cell, every night with Phase 2 data for it (the build's own finder, coadd.discover_dates, under
    paths.PHASE2_ROOT) that data_decisions admits now (``ex``) and that neither a record of the installed build names
    (admit.seen_nights: the build's seven night records) nor its admissions hold (admit.admitted_rows). The remedy is the
    admission rule's own (admit.nights_to_admit): a night Phase 2 first had after the build fixed its night set can be
    admitted, or the cell rebuilt; one Phase 2 had before (a lifted exclusion; an admission refused) only a rebuild takes in.
    A cell whose records cannot be read is reported unjudged. [(cell, date, cell dir, remedy, detail)]; remedy 'admit or
    rebuild' | 'rebuild' | 'unjudged' (date '')."""
    import contextlib
    import io
    C = K.cfg()
    from exopipe.phase3.lib.masters import admit as A, coadd as co
    root = C.paths.COADD_SB_ROOT
    keep = co.PHASE2_ROOT
    co.PHASE2_ROOT = C.paths.PHASE2_ROOT

    def one(cell):
        lst, scope = cell.split('_', 1)
        cd = os.path.join(root, cell)
        try:
            A.build_time(cd)                        # no installed build: no coverage to judge
            seen = set(A.seen_nights(cd)) | set(A.admitted_rows(cd))
            entries = co.discover_dates(lst, scope.replace('MHz', ''))
            gaps = sorted(d for d, _sci, _img in entries if not ex.is_excluded(lst, scope, d) and d not in seen)
            if not gaps:
                return []
            can, why = A.nights_to_admit(lst, scope, cd, entries, ex, co)
            can = {d for d, _sci, _img in can}
        except Exception as e:                      # a record the rules cannot read (Refusal), an unreadable file
            return [(cell, '', cd, 'unjudged', f'{type(e).__name__}: {str(e)[:200]}')]
        return [(cell, d, cd, 'admit or rebuild' if d in can else 'rebuild', str(why.get(d, ''))[:200]) for d in gaps]
    try:
        rows = []
        with contextlib.redirect_stdout(io.StringIO()):   # the finders' failsafe notes (admit.science_dirs_before does the same)
            for r in K.thread_map(one, cells, workers):
                rows.extend(r)
    finally:
        co.PHASE2_ROOT = keep
    return rows


def archive_audit(out_table, archive_root, unit=None, log=None):
    """coadd_exclusion_conformance.py in place (child). Returns (rc, rows); writes the empty table with its header
    when the script found nothing (it writes no file then, :82-85)."""
    from exopipe.phase3.core import legacy
    from exopipe.phase3.lib.status import steps_inplace as IP
    script = IP.legacy_script('coadd_exclusion_conformance.py')
    K.check_write(out_table)
    os.makedirs(os.path.dirname(out_table), exist_ok=True)
    if os.path.exists(out_table):
        os.remove(out_table)
    res = legacy.run(script, [archive_root], python='default', env={'CONF_OUT': out_table}, log=log, unit=unit,
                     refusal_codes=(), refusal_text=())
    if res.rc not in (0, 1):
        from exopipe.phase3.core.record import Refusal
        raise Refusal(f'coadd_exclusion_conformance.py exit {res.rc}: {res.tail[-400:]}')
    if not os.path.exists(out_table):
        K.write_csv(out_table, ARCHIVE_COLUMNS, [])
    rows = K.read_csv(out_table)
    return res.rc, rows, res


# ---------------------------------------------------------------------------------------------------------------------
# served runs
# ---------------------------------------------------------------------------------------------------------------------

_KEYS_OF = None


def keys_of_fn():
    """snapshot_serve_exclusion_gate.keys_of (in place; the module inserts the production package root into sys.path at
    import - restored)."""
    global _KEYS_OF
    if _KEYS_OF is None:
        from exopipe.phase3.lib.status import steps_inplace as IP
        _KEYS_OF = IP.import_quiet('snapshot_serve_exclusion_gate').keys_of
    return _KEYS_OF


def row_keys(row):
    ks = keys_of_fn()(row)
    if ks:
        return ks
    lst, date, scope = (row.get('lst') or '').strip(), (row.get('date') or '').strip()[:10], (row.get('scope') or '').strip()
    if lst and date and scope:
        return [(lst, date, scope)]
    return []


def scan_csv(args):
    """(hits, n_rows, n_keyed, error) of one CSV of a served run."""
    import csv
    path, run, rel = args
    from exopipe.phase3.core import exclusions as E
    ex = E.load()
    hits, n, nk, cache = [], 0, 0, {}
    try:
        with open(path, newline='', encoding='utf-8') as fh:     # undecodable = unreadable (the gate's rule, :67-85)
            rd = csv.DictReader(fh)
            if not rd.fieldnames:
                return hits, 0, 0, ''
            for row in rd:
                n += 1
                ks = row_keys(row)
                if ks:
                    nk += 1
                for k in ks:
                    if k not in cache:
                        try:
                            cache[k] = ex.is_excluded(k[0], k[2], k[1])
                        except Exception:              # noqa: BLE001 - an unparseable key is not a scope
                            cache[k] = False
                    if cache[k]:
                        hits.append(dict(part='served-run', run=run, file=rel,
                                         row_id=row.get('cand_id') or row.get('cell') or row.get('source_id') or '',
                                         lst=k[0], date=k[1], band=k[2], verdict='EXCLUDED-ROW',
                                         note='excluded now (core.exclusions)'))
    except (OSError, csv.Error, UnicodeDecodeError) as e:
        return hits, n, nk, f'{rel}: {e}'
    return hits, n, nk, ''


def served_run_files(run_dir):
    C = K.cfg().status_conformance
    out = []
    for p in sorted(glob.glob(os.path.join(run_dir, '**', '*.csv'), recursive=True)):
        rel = os.path.relpath(p, run_dir)
        if any(rel.startswith(d + os.sep) for d in C.EXCLCONF_SKIP_DIRS):   # top level only, as the gate (:65)
            continue
        if os.path.basename(p) in C.EXCLCONF_SKIP_FILES:
            continue
        out.append((p, rel))
    return out


def served_runs_audit(tier, run_dirs, workers=None, note=print):
    """(rows, counts) of the served-run part for the given run directories."""
    from concurrent.futures import ProcessPoolExecutor
    keys_of_fn()                          # import once in the parent (children inherit it under fork)
    rows, counts, errors = [], collections.Counter(), []
    for rd in run_dirs:
        run = os.path.basename(rd)
        files = served_run_files(rd)
        jobs = [(p, run, rel) for p, rel in files]
        n = max(1, min(workers or K.cfg().status.WORKERS, len(jobs) or 1))
        if n > 1 and len(jobs) > 1:
            with ProcessPoolExecutor(max_workers=n) as ex:
                res = list(ex.map(scan_csv, jobs, chunksize=8))
        else:
            res = [scan_csv(j) for j in jobs]
        for hits, nr, nk, err in res:
            rows.extend(hits)
            counts[(run, 'rows')] += nr
            counts[(run, 'keyed')] += nk
            if err:
                errors.append(err)
        counts[(run, 'files')] += len(files)
        note(f'exclusion-conformance: {run}: {len(files)} CSV files, {counts[(run, "rows")]} rows, '
             f'{counts[(run, "keyed")]} keyed, {sum(1 for r in rows if r["run"] == run)} on excluded cell-bands')
    return rows, counts, errors
