"""`status --stale [CLASS]` - stale items per class (spec section 2, 7.2; status.STALE_CLASSES).

record            run records (DESIGN 5, 11.5; S-G15): a done record is STALE when it was written from a dirty tree, an
                  input fingerprint changed, a config section it hashed changed, a module it imported (exopipe or legacy,
                  incl. the legacy child fingerprint) changed, an output is missing, or - for a cell / epoch unit with an
                  admitted night set - a night it admitted is excluded now (deep requirement (d)). A record not done is
                  listed with its status (not stale: unfinished). Roots: status.REPORT_ROOT, status.RECORD_ROOTS, --records,
                  and the --out of every job in runner.JOBS_ROOT.
search_attempt    published deep search attempts that searched a night data_decisions now excludes
                  (p3s3_stale_exclusions.stale_attempts in place; report only - never refuse_and_unmark, S-G13), with the
                  attempt's age against data_decisions.csv (aad9c3c).
master            installed (working tree) masters holding a constituent excluded AFTER the master was written (STALE rows
                  of the master exclusion audit, steps_exclconf.working_tree_audit).
frame_time        shipped snapshot cells whose 1-minute / wideband products are pre-fix (snapshot_frametime_worklist.needs).
prefit            snapshot cells whose PREFIT_DONE marker is older than their newest runinfo (eligibility PREFIT-STALE).
sublst_generation published sub-LST LST runs without pertie_detections.csv or without pos_kind (B-73).
qa_night_pages    ledger (LST, night)s without a hub night page (content test P1 of build_qa_pages.check).
ledger            ledger rows that differ from a fresh inventory (steps_ledger: the ledger stage run into --out).
"""
import glob
import hashlib
import importlib.util
import json
import os
import re
import sys

from exopipe.phase3.lib.status import steps_common as K

COLUMNS = ('class', 'item', 'kind', 'status', 'reason', 'step', 'root', 'detail')


# ---------------------------------------------------------------------------------------------------------------------
# record
# ---------------------------------------------------------------------------------------------------------------------

def record_roots(extra=()):
    """Roots holding _records/: status.REPORT_ROOT, status.RECORD_ROOTS, --records, and every job's --out."""
    C = K.cfg()
    roots = [C.status.REPORT_ROOT] + list(C.status.RECORD_ROOTS) + list(extra or ())
    jobs = C.runner.JOBS_ROOT
    for jd in K.listdir(jobs):
        jp = os.path.join(jobs, jd, 'job.json')
        if not os.path.isfile(jp):
            continue
        try:
            j = json.load(open(jp))
        except (OSError, ValueError):
            continue
        argv = list(j.get('step_argv') or []) + list(j.get('forward_argv') or [])
        for i, a in enumerate(argv):
            if a == '--out' and i + 1 < len(argv):
                roots.append(argv[i + 1])
            elif a.startswith('--out='):
                roots.append(a.split('=', 1)[1])
    seen, out = set(), []
    for r in roots:
        r = os.path.abspath(r)
        if r not in seen and os.path.isdir(os.path.join(r, '_records')):
            seen.add(r)
            out.append(r)
    return out


def _file_sha12(path):
    try:
        with open(path, 'rb') as f:
            return hashlib.sha1(f.read()).hexdigest()[:12]
    except OSError:
        return None


def _module_file(name):
    m = sys.modules.get(name)
    f = getattr(m, '__file__', None) if m else None
    if f:
        return f
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError, AttributeError):
        return None
    return getattr(spec, 'origin', None) if spec else None


def record_staleness(rec, ex):
    """(stale reasons, not-done status or '') of one unit record."""
    from exopipe.phase3.core import record as R
    st = rec.get('status')
    if st != 'done':
        return [], st or 'unknown'
    why = []
    if str(rec.get('commit') or '').endswith('-dirty'):
        why.append(f"written from a dirty tree ({rec.get('commit')})")
    for p, fp in (rec.get('inputs') or {}).items():
        if not R.same_fingerprint(fp, R.fingerprint(p)):
            why.append(f'input changed: {p}')
            break
    secs = rec.get('config_sections')
    try:
        cur = K.cfg().hash(secs) if secs is not None else K.cfg().hash()
        if rec.get('config_hash') and cur != rec.get('config_hash'):
            why.append('config changed (sections ' + (','.join(secs) if secs else 'all') + ')')
    except Exception as e:                                   # noqa: BLE001 - a section no longer defined
        why.append(f'config sections not computable now ({e})')
    for mod, h in (rec.get('code') or {}).items():
        f = _module_file(mod)
        if f and f.endswith('.py') and os.path.exists(f) and _file_sha12(f) != h:
            why.append(f'code changed: {mod}')
            break
    for f, h in ((rec.get('meta') or {}).get('legacy_code') or {}).items():
        if os.path.exists(f) and _file_sha12(f) not in (None, str(h)[:12]):
            why.append(f'legacy code changed: {f}')
            break
    for o in rec.get('outputs') or ():
        if not os.path.exists(o['path']):
            why.append(f"output missing: {o['path']}")
            break
    adm = rec.get('admitted') or []
    unit = rec.get('unit') or ''
    if adm and ex is not None:
        m = re.match(r'^(\d\dh)_(\d+MHz|Red|Green|Blue|Full)(?:_|$)', unit)
        if m:
            bad = ex.excluded_constituents(m.group(1), m.group(2), adm)
            if bad:
                why.append(f"admitted night(s) excluded now: {','.join(bad[:5])}")
        else:
            m = re.match(r'^(\d\dh)_(\d{4}-\d\d-\d\d)', unit)
            if m and ex.is_whole_night_excluded(m.group(1), m.group(2)):
                why.append(f'night {m.group(2)} wholly excluded now')
    return why, ''


def stale_records(roots, ex, step_filter=None):
    rows = []
    for root in roots:
        for p in sorted(glob.glob(os.path.join(root, '_records', '*', '*.json'))):
            step = os.path.basename(os.path.dirname(p))
            if step_filter and step not in step_filter:
                continue
            try:
                rec = json.load(open(p))
            except (OSError, ValueError) as e:
                rows.append(dict(item=os.path.basename(p)[:-5], kind='unit', status='unreadable', reason=str(e),
                                 step=step, root=root, detail=p))
                continue
            why, st = record_staleness(rec, ex)
            if why:
                rows.append(dict(item=rec.get('unit'), kind='unit', status='stale', reason=why[0],
                                 step=step, root=root, detail='; '.join(why[1:4])))
            elif st:
                rows.append(dict(item=rec.get('unit'), kind='unit', status=st, reason=f'record status {st}',
                                 step=step, root=root, detail=rec.get('reason', '')[:200]))
    return rows


# ---------------------------------------------------------------------------------------------------------------------
# search_attempt
# ---------------------------------------------------------------------------------------------------------------------

def stale_search_attempts(label):
    from exopipe.phase3.lib.status import steps_inplace as IP
    env = {'P3S_LABEL': label}
    S = IP.import_quiet('p3s3_sources', env=env)
    SE = IP.import_quiet('p3s3_stale_exclusions', env=env)
    dd = K.cfg().paths.DATA_DECISIONS
    dd_m = os.path.getmtime(dd)
    old = os.environ.get('P3S_LABEL')
    os.environ['P3S_LABEL'] = label                  # stale_attempts reads it at call time (:22)
    sys_path = list(sys.path)
    try:
        st = SE.stale_attempts(S.published_attempts())
    finally:
        sys.path[:] = sys_path
        if old is None:
            os.environ.pop('P3S_LABEL', None)
        else:
            os.environ['P3S_LABEL'] = old
    rows = []
    for a, bad in st:
        try:
            age = 'older than data_decisions.csv' if os.path.getmtime(a['dir']) < dd_m else 'newer than data_decisions.csv'
        except OSError:
            age = 'attempt dir missing'
        rows.append(dict(item=a['cell'], kind='cell', status='stale', reason=f"searched excluded night(s) {'|'.join(bad)}",
                         step='search', root=a['dir'], detail=f"Stokes {a['stokes']}; node {a['node']}; attempt {a['att']}; {age}"))
    return rows


# ---------------------------------------------------------------------------------------------------------------------
# snapshot: frame_time, prefit
# ---------------------------------------------------------------------------------------------------------------------

def stale_frame_time(cells_spec=None):
    from exopipe.phase3.lib.status import steps_inplace as IP
    C = K.cfg()
    FW = IP.import_quiet('snapshot_frametime_worklist')
    ship = C.paths.SNAPSHOT_FLEET_OUT
    cells = sorted(c for c in K.listdir(ship) if re.match(r'^\d\dh_\d{4}-\d\d-\d\d$', c))
    if cells_spec:
        cells = K.glob_cells(cells, cells_spec)
    res = K.thread_map(lambda c: FW.needs(os.path.join(ship, c)), cells, C.status_snapshot.FRAMETIME_THREADS)
    rows = []
    for c, (why, sbs) in zip(cells, res):
        if why:
            rows.append(dict(item=c, kind='night_cell', status='stale', reason='+'.join(sorted(why)), step='frames',
                             root=os.path.join(ship, c), detail=f'{len(sbs)} subbands searched in both Stokes'))
    return rows, FW.__file__


def stale_prefit(eligibility_csv):
    rows = []
    for r in K.read_csv(eligibility_csv):
        if r.get('status') == 'PREFIT-STALE':
            rows.append(dict(item=r['cell'], kind='night_cell', status='stale',
                             reason='PREFIT_DONE older than the newest runinfo', step='catalog', root=eligibility_csv,
                             detail=''))
    return rows


# ---------------------------------------------------------------------------------------------------------------------
# sublst_generation, qa_night_pages
# ---------------------------------------------------------------------------------------------------------------------

def stale_sublst_generation():
    from exopipe.phase3.lib.status import steps_coverage as COV
    C = K.cfg()
    serve = getattr(C, 'serve', None)
    pat = getattr(serve, 'SUBLST_LST_RUN_GLOB', 'Run_SLST[0-9][0-9]B')
    rows = []
    for rd in sorted(glob.glob(os.path.join(K.runs_root('sublst'), pat))):
        ok, why = COV.sublst_generation(rd)
        if not ok:
            lst = re.search(r'SLST(\d\d)B', os.path.basename(rd)).group(1) + 'h'
            rows.append(dict(item=lst, kind='lst', status='stale', reason=why, step='catalog', root=rd, detail=''))
    return rows


def stale_night_pages(ledger_path=None, hub=None):
    C = K.cfg()
    ledger_path = ledger_path or C.status.LEDGER
    hub = hub or K.hub_dir()
    rows = []
    for r in K.read_csv(ledger_path):
        p = os.path.join(hub, C.status_pages.NIGHT_PAGE.format(lst=r['lst'], date=r['date']))
        if not os.path.exists(p):
            rows.append(dict(item=f"{r['lst']}_{r['date']}", kind='epoch', status='stale',
                             reason='ledger night without a hub night page (P1)', step='status', root=hub, detail=p))
    return rows
