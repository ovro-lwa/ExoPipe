"""Content tests of the product classes `exopipe3 status` writes (registered with core/validate; `exopipe3 verify <class>
<file>` runs any of them alone). Each test reads the file it tests (rows, links, pixels) and re-derives what it can from
the file's own columns or from the inputs it names; none tests presence alone. Spec section 6.

Classes: item_list, stale_table, coverage_table, rebuild_due, exclusion_conformance, served_diff, cell_load_review,
status_report, readjudication_list, date_outlier_proposals, qa_table, solar_audit, beam_audit, noise_plot,
bandpass_phase_qa, provenance_audit, membership_conformance, qa_night_pages, status_page, processing_ledger,
ratified_register, snapshot_eligibility, snapshot_batch_list.
"""
import ast
import collections
import json
import os
import re

from exopipe.phase3.core.validate import content_test
from exopipe.phase3.lib.status import steps_common as K


def _rows(path):
    return K.read_csv(path)


def _cols(path):
    import csv
    with open(path, newline='', encoding='utf-8', errors='replace') as f:
        return next(csv.reader(f), [])


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _missing(path, want):
    have = _cols(path)
    return [c for c in want if c not in have]


# ---------------------------------------------------------------------------------------------------------------------
# item lists and stale tables
# ---------------------------------------------------------------------------------------------------------------------

def _qualifier(text):
    """'col=a|b' -> (col, {a, b})."""
    if '=' not in (text or ''):
        return None, None
    c, v = text.split('=', 1)
    return c.strip(), {x.strip() for x in v.split('|')}


def _item_from_row(kind, r):
    if 'kind' in r and 'class' in r and r['kind'] != kind:
        return None                                   # a stale table lists the rows of the list's kind only
    return K.row_item(kind, r)


@content_test('item_list')
def items_parse(path, **ctx):
    head, items = K.read_items(path)
    kind = head.get('kind')
    if kind not in K.ITEM_KINDS:
        return False, len(items), f'header kind {kind!r} not one of {K.ITEM_KINDS}'
    bad = []
    for i in items:
        try:
            K.parse_item(kind, i)
        except ValueError as e:
            bad.append(str(e))
    if bad:
        return False, len(bad), bad[0]
    if items != sorted(set(items)):
        return False, len(items), 'items not sorted and unique'
    return True, len(items), f'{len(items)} {kind} items parse'


@content_test('item_list')
def items_qualify(path, **ctx):
    """Every item is exactly the set of qualifying rows of the producing table (none missing, none extra); a record
    list's units are still stale or unfinished now."""
    head, items = K.read_items(path)
    kind, table = head.get('kind'), head.get('table')
    col, vals = _qualifier(head.get('qualifies'))
    if not table or not os.path.exists(table) or col is None:
        return False, None, f'header names no readable table / qualifier ({table!r}, {head.get("qualifies")!r})'
    want = set()
    for r in _rows(table):
        if str(r.get(col, '')) in vals:
            it = _item_from_row(kind, r)
            if it:
                want.add(it)
    have = set(items)
    if want != have:
        return False, len(have ^ want), (f'{len(want - have)} qualifying rows missing from the list (e.g. '
                                         f'{sorted(want - have)[:2]}), {len(have - want)} listed without a qualifying '
                                         f'row (e.g. {sorted(have - want)[:2]})')
    if head.get('stale-class') == 'record':
        from exopipe.phase3.lib.status import steps_stale as ST
        ex = K.exclusions()
        for r in _rows(table):
            if r.get('item') in have and r.get('status') == 'stale':
                p = os.path.join(r['root'], '_records', r['step'], f"{r['item']}.json")
                if os.path.exists(p):
                    why, st = ST.record_staleness(json.load(open(p)), ex)
                    if not why and not st:
                        return False, r['item'], f'record {p} is no longer stale'
    return True, len(have), f'{len(have)} items = the qualifying rows ({col} in {sorted(vals)}) of {os.path.basename(table)}'


@content_test('stale_table')
def stale_rows(path, **ctx):
    from exopipe.phase3.lib.status import steps_stale as ST
    miss = _missing(path, ST.COLUMNS)
    if miss:
        return False, None, f'missing columns {miss}'
    rows = _rows(path)
    seen = set()
    for r in rows:
        if not r['reason']:
            return False, r['item'], 'row without a reason'
        k = (r['class'], r['item'], r['step'], r['root'])
        if k in seen:
            return False, r['item'], 'duplicate row'
        seen.add(k)
        try:
            K.parse_item(r['kind'], r['item'])
        except ValueError as e:
            return False, r['item'], str(e)
    return True, len(rows), f'{len(rows)} stale/unfinished items with reasons'


# ---------------------------------------------------------------------------------------------------------------------
# coverage, rebuild due
# ---------------------------------------------------------------------------------------------------------------------

@content_test('coverage_table')
def coverage_rows(path, **ctx):
    from exopipe.phase3.lib.status import steps_coverage as COV
    cols = _cols(path)
    rows = _rows(path)
    if 'census_class' in cols:                         # deep
        miss = [c for c in COV.DEEP_COLUMNS if c not in cols]
        if miss:
            return False, None, f'missing columns {miss}'
        ex = K.exclusions()
        seen = set()
        for r in rows:
            k = (r['lst'], r['scope'], r['date'])
            if k in seen:
                return False, k, 'duplicate (lst, scope, date)'
            seen.add(k)
            if int(r['p2_runs'] or 0) < 1:
                return False, k, 'row without a Phase 2 product'
            if int(r['excluded']) != int(ex.is_excluded(r['lst'], r['scope'], r['date'])):
                return False, k, f"excluded={r['excluded']} but core.exclusions says {ex.is_excluded(*k)} now"
            if int(r['gap']) != COV.gap_of(r):
                return False, k, 'gap column disagrees with (excluded, residual)'
            if r['census_class'] != COV.census_class(r):
                return False, k, f"census_class {r['census_class']} != {COV.census_class(r)} (census precedence)"
        return True, len(rows), f'{len(rows)} rows; {sum(int(r["gap"]) for r in rows)} gaps; exclusions agree now'
    if 'windows' in cols or 'in_worklist' in cols:      # sub-LST / snapshot
        sub = 'windows' in cols
        need = COV.SUBLST_COLUMNS if sub else COV.SNAPSHOT_COLUMNS
        miss = [c for c in need if c not in cols]
        if miss:
            return False, None, f'missing columns {miss}'
        ship = K.cfg().paths.SNAPSHOT_FLEET_OUT
        for r in rows:
            k = (r['lst'], r['scope']) if sub else r['cell']
            if int(r['gap']) != int(bool(r['reason'])):
                return False, k, 'gap without reason or reason without gap'
            want = COV.sublst_reasons(r) if sub else COV.snapshot_reasons(r)
            got = [x.split(':')[0] for x in r['reason'].split('; ') if x]
            if got != [x.split(':')[0] for x in want]:
                return False, k, f'reasons {got} != re-derived from the row {want}'
            if sub:
                for S in ('V', 'I'):                  # 'done' only from the search record, re-read now
                    if (r[f'search_{S}'] == 'done') != COV.sublst_search_record(f"{r['lst']}_{r['scope']}", S)[0]:
                        return False, k, f'search_{S}={r[f"search_{S}"]} but the search record says otherwise now'
            elif int(r['shipped']) != int(COV.snapshot_search_records(os.path.join(ship, r['cell']))[0]):
                return False, k, f"shipped={r['shipped']} but the cell's runinfo / night-refusal records say otherwise now"
        return True, len(rows), (f'{len(rows)} rows; {sum(int(r["gap"]) for r in rows)} gaps; reasons and search records '
                                 f're-derived')
    return False, None, 'not a coverage table (no census_class / windows / in_worklist column)'


@content_test('patched_attempts')
def patched_from_own_record(path, label=None, state=None, cells=None, **ctx):
    """B-01 (b): one row per published (cell, Stokes); each row's patched / patched_from re-derived from the attempt
    folder's own PATCH_RECORD.json (lib/catalog/deep_ported.patched_from); and - given the register the table was made from
    (label, state, cells_spec of its unit) - the rows are exactly that register's published attempts of the selected cells,
    none missing, none extra."""
    from exopipe.phase3.lib.catalog import deep_ported as DP
    need = ('cell', 'lst', 'scope', 'stokes', 'attempt', 'dir', 'patched', 'patched_from')
    miss = _missing(path, need)
    if miss:
        return False, None, f'missing columns {miss}'
    rows = _rows(path)
    bad, seen = [], set()
    if label and state:
        from exopipe.phase3.lib.status import steps_inplace as IPL
        want = set(cells) if cells is not None else None              # the expanded selection the unit recorded
        exp = {(a['cell'], a['stokes'], a['att'], a['dir']) for a in IPL.search_sources(label, state=state).published_attempts(['V', 'I'])
               if want is None or a['cell'] in want}
        got = {(r['cell'], r['stokes'], r['attempt'], r['dir']) for r in rows}
        if got != exp:
            bad.append(f'rows are not the register\'s published attempts: {len(exp - got)} missing (e.g. {sorted(exp - got)[:2]}), '
                       f'{len(got - exp)} extra (e.g. {sorted(got - exp)[:2]})')
    for r in rows:
        k = (r['cell'], r['stokes'])
        if k in seen:
            bad.append(f'{k} twice')
        seen.add(k)
        if r['cell'] != f"{r['lst']}_{r['scope']}":
            bad.append(f"{k}: cell is not lst_scope")
        src = DP.patched_from(r['dir'])
        if (r['patched'] == '1') != (src is not None) or (r['patched_from'] or None) != src:
            bad.append(f"{k}: patched={r['patched']} from {r['patched_from']!r}, the attempt's record says {src!r}")
    n = sum(1 for r in rows if r['patched'] == '1')
    return not bad, n, (f'{len(rows)} published attempts, {n} patched' + (f'; {bad[:4]}' if bad else ' - every row as its record says'))


@content_test('rebuild_due')
def rebuild_due_rule(path, **ctx):
    """Each row's decision re-derived from its own columns (due iff new_s >= due_fraction x build_s); its new nights are
    not excluded now and not measured build nights of the cell's weights.csv; n_build = the weights.csv rows."""
    from exopipe.phase3.lib.status import steps_rebuild as RB
    miss = _missing(path, RB.COLUMNS)
    if miss:
        return False, None, f'missing columns {miss}'
    ex = K.exclusions()
    root = K.cfg().paths.COADD_SB_ROOT
    rows = _rows(path)
    cells = set()
    for r in rows:
        c = r['cell']
        if c in cells:
            return False, c, 'duplicate cell'
        cells.add(c)
        if r['status'] not in RB.STATUSES:
            return False, c, f"status {r['status']!r}"
        new = [d for d in r['new_nights'].split(';') if d]
        if len(new) != int(r['n_new']):
            return False, c, 'n_new != len(new_nights)'
        if r['status'] in ('due', 'not-due'):
            b, n, fr = _f(r['build_s']), _f(r['new_s']), _f(r['due_fraction'])
            if not b or b <= 0:
                return False, c, 'decided row without build time'
            if abs(_f(r['frac_new']) - n / b) > 1e-5:
                return False, c, f"frac_new {r['frac_new']} != new_s/build_s {n / b:.6f}"
            if (r['status'] == 'due') != (n >= fr * b) or int(r['due']) != int(n >= fr * b):
                return False, c, 'due flag disagrees with new_s >= due_fraction x build_s'
        lst, scope = c.split('_', 1)
        for d in new:
            if ex.is_excluded(lst, scope, d):
                return False, c, f'new night {d} is excluded now'
        if r['new_source'] not in RB.NEW_SOURCES:
            return False, c, f"new_source {r['new_source']!r}"
        if r['new_source'] == 'admission-record':     # D-ADMIT hook: every new night is a row of the cell's record
            rp = os.path.join(root, c, 'coadd_final', str(K.cfg().status_rebuild.ADMISSION_RECORD))
            adm = {x['date'] for x in _rows(rp)} if os.path.exists(rp) else set()
            if set(new) - adm:
                return False, c, f'new night(s) {sorted(set(new) - adm)[:3]} not in the admission record {rp}'
        w = os.path.join(root, c, 'coadd_final', K.cfg().status_rebuild.BUILD_RECORD)
        if os.path.exists(w):
            wr = _rows(w)
            if len({x['date'] for x in wr}) != int(r['n_build']):
                return False, c, f"n_build {r['n_build']} != {len(wr)} rows of {w}"
            noimg = set(d for d in r['build_without_image'].split(';') if d)
            overlap = (set(new) & {x['date'] for x in wr}) - noimg
            if overlap:
                return False, c, f'new night(s) {sorted(overlap)[:3]} are measured build nights'
    return True, len(rows), (f'{len(rows)} cells; due {sum(r["status"] == "due" for r in rows)}, never-built '
                             f'{sum(r["status"] == "never-built" for r in rows)}, unmeasured '
                             f'{sum(r["status"] == "unmeasured" for r in rows)}')


# ---------------------------------------------------------------------------------------------------------------------
# exclusion conformance, served diff, cell load
# ---------------------------------------------------------------------------------------------------------------------

@content_test('exclusion_conformance')
def rows_are_excluded(path, **ctx):
    """Every row's (LST, date, band) is excluded now (core.exclusions); a master row's BREACH / STALE / UNKNOWN-EPOCH is
    re-derived from the data_decisions backups at the product's write date (coadd_exclusion_conformance.py:47-80)."""
    from exopipe.phase3.lib.status import steps_exclconf as XC
    cols = _cols(path)
    rows = _rows(path)
    ex = K.exclusions()
    archive_fmt = 'kind' in cols and 'product' in cols
    need = XC.ARCHIVE_COLUMNS if archive_fmt else XC.COLUMNS
    miss = [c for c in need if c not in cols]
    if miss:
        return False, None, f'missing columns {miss}'
    n_v = collections.Counter()
    for r in rows:
        v = r['verdict']
        n_v[v] += 1
        if v == 'UNREADABLE':
            continue
        if not ex.is_excluded(r['lst'], r['band'], r['date']):
            return False, (r['lst'], r['band'], r['date']), f'row {v} but not excluded now'
        if v in ('BREACH', 'STALE', 'UNKNOWN-EPOCH') and not archive_fmt:
            m = re.search(r'written (\d{8})', r.get('note', ''))
            if m and XC.verdict_at(r['lst'], r['band'], r['date'], m.group(1)) != v:
                return False, (r['lst'], r['band'], r['date']), f'verdict {v} != re-derived ' \
                    f"{XC.verdict_at(r['lst'], r['band'], r['date'], m.group(1))}"
        elif not archive_fmt and v != 'EXCLUDED-ROW':
            return False, r.get('row_id'), f'unknown verdict {v!r}'
    return True, len(rows), ', '.join(f'{k} {n}' for k, n in sorted(n_v.items())) or '0 rows (conformant)'


@content_test('served_diff')
def served_diff_sums(path, **ctx):
    from exopipe.phase3.lib.status import steps_serveddiff as SD
    miss = _missing(path, SD.COLUMNS)
    if miss:
        return False, None, f'missing columns {miss}'
    sj = path[:-len('_rows.csv')] + '_summary.json' if path.endswith('_rows.csv') else None
    summ = json.load(open(sj)) if sj and os.path.exists(sj) else None
    rows = _rows(path)
    nA = collections.Counter()
    for r in rows:
        fates = {k: int(r[k]) for k in ('kept', 'night_rejected', 'research', 'removed', 'lost')}
        if r['side'] == 'A':
            nA[r['stokes']] += 1
            if sum(fates.values()) != int(r['detections']):
                return False, r['source'], 'member fates do not add up to the detections'
            leaving = sum(fates[k] for k in SD.LEAVING)
            st = ('gone' if leaving == int(r['detections']) else ('reduced' if leaving else
                                                                  ('re-searched cell' if fates['research'] else 'unchanged')))
            if st != r['status']:
                return False, r['source'], f"status {r['status']} != re-derived {st}"
        elif r['status'] != 'new':
            return False, r['source'], 'a B-side row that is not new'
    if summ:
        for S in ('I', 'V'):
            if nA[S] != summ['served_A'][S]:
                return False, S, f"{nA[S]} A rows != {summ['served_A'][S]} served sources of A"
    return True, len(rows), f"A sources {dict(nA)}; new {sum(r['status'] == 'new' for r in rows)}" + \
        ('' if summ else ' (no summary JSON beside it: source totals not checked)')


def _cell_load_run(path):
    b = os.path.basename(path)
    if b == 'cell_load_review.csv':
        return os.path.dirname(path)
    m = re.match(r'cell_load_review_(.+)\.csv$', b)
    return K.resolve_run(m.group(1), 'deep') if m else None


@content_test('cell_load_review')
def cell_load_sums(path, **ctx):
    from exopipe.phase3.lib.status import steps_report as RP
    cols = _cols(path)
    want = list(K.cfg().status.CELL_LOAD_COLUMNS)
    if cols != want:
        return False, None, f'columns {cols} != {want}'
    rows = _rows(path)
    dets = [int(r['detections_5sig']) for r in rows]
    if dets != sorted(dets, reverse=True):
        return False, None, f'rows not sorted by detections ({RP.CELL_LOAD_SORT})'
    run = ctx.get('run_dir') or _cell_load_run(path)
    if run and os.path.isdir(run):
        ref, _ = RP.cell_load(run)
        if sum(dets) != sum(r['detections_5sig'] for r in ref) or \
                sum(int(r['served_sources']) for r in rows) != sum(r['served_sources'] for r in ref):
            return False, None, f'sums differ from {run}/selected_all.csv and pages_input_*.csv now'
    return True, len(rows), f'{sum(dets)} detections, {sum(int(r["served_sources"]) for r in rows)} served-source rows'


# ---------------------------------------------------------------------------------------------------------------------
# report, re-adjudication, outliers
# ---------------------------------------------------------------------------------------------------------------------

@content_test('status_report')
def report_counts(path, **ctx):
    """The JSON's ledger counters equal a Counter over the ledger it names (when the ledger is unchanged); its
    data_decisions sha1 / exclude-key count equal core.exclusions when the file is unchanged; every CLAUDE.md 1
    section is present (empty is allowed, absent is not)."""
    from exopipe.phase3.lib.status import steps_report as RP
    if not path.endswith('.json'):
        return _report_text(path)
    j = json.load(open(path))
    miss = [s for s in RP.REPORT_SECTIONS if s not in j]
    if miss:
        return False, None, f'sections absent: {miss}'
    led = j['phase_completion']
    if led.get('ledger') and os.path.exists(led['ledger']) and led.get('sha1') == _sha1(led['ledger']):
        if RP.ledger_counts(_rows(led['ledger'])) != led.get('counts'):
            return False, None, 'ledger counters differ from the ledger they name'
    dd = j['data_decisions']
    fp = K.exclusions().fingerprint()
    if dd.get('sha1') == fp['sha1'] and dd.get('n_exclude_keys') != fp['n_exclude_rows']:
        return False, None, 'exclude-key count differs from core.exclusions on the same file'
    bad = _masters_rederived(j)
    if bad:
        return False, None, bad
    return True, len(j), 'every section present; counters agree (ledger, data_decisions, masters re-read from the files)'


def _masters_rederived(j):
    """The masters section re-derived from the master FILES now (phase_status.master_inventory: COADDOP card), for every
    root the report read, when no master of that root was written after the report (else the state moved on: skipped)."""
    from exopipe.phase3.lib.status import phase_status as PS
    import calendar
    import time as _t
    try:
        written = calendar.timegm(_t.strptime(j.get('written', ''), '%Y-%m-%dT%H:%M:%SZ'))
    except ValueError:
        return f"report 'written' time unreadable: {j.get('written')!r}"
    for r, v in (j.get('masters') or {}).items():
        if not isinstance(v, dict):
            continue
        if not os.path.isdir(r):
            return f'masters root {r} read by the report is gone'
        cells = [d for d in K.listdir(r) if len(d) > 3 and d[2] == 'h' and '_' in d]
        inv = PS.master_inventory(r, cells)
        if inv['newest'] and inv['newest'][0] > written + 1:
            continue
        if (inv['masters'], inv['coaddop'], len(cells)) != (v.get('masters_coaddop'), v.get('coaddop'), v.get('cells')):
            return (f"masters of {r}: report {v.get('masters_coaddop')} {v.get('coaddop')} in {v.get('cells')} cells, "
                    f"the files now {inv['masters']} {inv['coaddop']} in {len(cells)} cells")
    return ''


def _report_text(path):
    """The text twin of a report: every CLAUDE.md 1 section line is present and its counts equal the JSON beside it."""
    from exopipe.phase3.lib.status import steps_report as RP
    txt = open(path).read()
    jp = path[:-4] + '.json'
    if not os.path.exists(jp):
        return False, None, f'no JSON twin {os.path.basename(jp)} beside the text'
    j = json.load(open(jp))
    lines = txt.splitlines()
    if RP.TEXT_HEADER not in lines:
        return False, None, f'no line {RP.TEXT_HEADER!r}'
    lines = lines[lines.index(RP.TEXT_HEADER) + 1:]           # the new sections (phase_status.py's own text precedes)
    miss = [t for t in RP.TEXT_TAGS if not any(ln.startswith(t) for ln in lines)]
    if miss:
        return False, miss, f'section lines absent: {miss}'
    led = next(ln for ln in lines if ln.startswith('[ledger]'))
    if not led.startswith(f"[ledger] {j['phase_completion']['rows']} "):
        return False, led[:80], f"ledger row count differs from the JSON ({j['phase_completion']['rows']})"
    dd = next(ln for ln in lines if ln.startswith('[data_decisions]'))
    if not dd.startswith(f"[data_decisions] {j['data_decisions']['n_exclude_keys']} exclude keys") or \
            j['data_decisions']['sha1'][:12] not in dd:
        return False, dd[:80], 'data_decisions line differs from the JSON'
    rj = next(ln for ln in lines if ln.startswith('[re-adjudication]'))
    if not rj.startswith(f"[re-adjudication] {j['readjudication']['rows']} "):
        return False, rj[:80], 're-adjudication count differs from the JSON'
    ms = next(ln for ln in lines if ln.startswith('[masters]'))
    if ms != '[masters] ' + RP.masters_text(j['masters']):
        return False, ms[:120], 'masters line differs from the JSON'
    return True, len(lines), 'every section line present; counts equal the JSON twin'


def _sha1(p):
    import hashlib
    return hashlib.sha1(open(p, 'rb').read()).hexdigest()


@content_test('readjudication_list')
def readjudication_rows(path, ledger=None, ledger_sha1=None, previous=None, **ctx):
    """Columns; every row an all- or partial-excluded night with an exclude row now; go_no_go text. With the run's context
    (the ledger it read and the previous state it compared with): the rows are EXACTLY the excluded nights whose
    (p1_peel, p2) differs from the previous state - none missing, none extra (CLAUDE.md 1.3)."""
    R = K.cfg().status_report
    cols = _cols(path)
    if cols != list(R.READJUDICATION_COLUMNS):
        return False, None, f'columns {cols} != {list(R.READJUDICATION_COLUMNS)}'
    ex = K.exclusions()
    excl_nights = {(l, d) for (l, s, d) in ex.rows}
    rows = _rows(path)
    for r in rows:
        if r['dd_status'] not in ('all-excluded', 'partial-excluded'):
            return False, (r['lst'], r['date']), f"dd_status {r['dd_status']}"
        if (r['lst'], r['date']) not in excl_nights:
            return False, (r['lst'], r['date']), 'night has no exclude row now'
        if r['go_no_go'] != R.READJUDICATION_GO:
            return False, (r['lst'], r['date']), 'go_no_go text'
    if previous is not None and ledger and os.path.exists(ledger) and ledger_sha1 and _sha1(ledger) == ledger_sha1:
        from exopipe.phase3.lib.status import steps_report as RP
        want = {(r['lst'], r['date']) for r in RP.readjudication_rows(_rows(ledger), previous)}
        got = {(r['lst'], r['date']) for r in rows}
        if want != got:
            return False, len(want ^ got), (f'{len(want - got)} advanced excluded nights missing (e.g. {sorted(want - got)[:2]}), '
                                            f'{len(got - want)} listed without an advance')
        return True, len(rows), f'{len(rows)} excluded nights advanced since the previous state (re-derived from the ledger)'
    return True, len(rows), f'{len(rows)} excluded nights proposed for re-adjudication (empty allowed)'


@content_test('date_outlier_proposals')
def proposals_rule(path, **ctx):
    """The proposals equal the selection rule re-run now over the working tree's residuals_provisional.csv, none is an
    exclusion already (core.exclusions), and data_decisions.csv is never the file tested."""
    from exopipe.phase3.lib.status import steps_qa as QA
    if os.path.realpath(path) == os.path.realpath(K.cfg().paths.DATA_DECISIONS):
        return False, None, 'proposals written over data_decisions.csv'
    miss = _missing(path, QA.OUTLIER_COLUMNS)
    if miss:
        return False, None, f'missing columns {miss}'
    ex = K.exclusions()
    rows = _rows(path)
    for r in rows:
        if ex.is_excluded(r['lst'], r['subband'], r['date']):
            return False, (r['lst'], r['subband'], r['date']), 'proposed row already excluded'
    if ctx.get('recompute', True):
        want, _ = QA.date_outliers(ex)
        got = {(r['lst'], r['subband'], r['date']) for r in rows}
        exp = {(r['lst'], r['subband'], r['date']) for r in want}
        if got != exp:
            return False, len(got ^ exp), 'proposals differ from the rule re-run now'
    return True, len(rows), f'{len(rows)} proposals for Gregg; none already excluded'


# ---------------------------------------------------------------------------------------------------------------------
# QA tables (in-place products)
# ---------------------------------------------------------------------------------------------------------------------

def _list_concat(node):
    if isinstance(node, ast.List):
        return [ast.literal_eval(e) for e in node.elts]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _list_concat(node.left) + _list_concat(node.right)
    raise ValueError('not a list concatenation')


def qa_table_columns():
    """COLS of ops/build_qa_table.py read from its source (ast; the script is never executed here)."""
    from exopipe.phase3.lib.status import steps_inplace as IP
    tree = ast.parse(open(IP.legacy_script('build_qa_table.py')).read())
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'COLS' for t in n.targets):
            v = n.value.elts[0] if isinstance(n.value, ast.Tuple) else n.value
            return _list_concat(v)
    raise ValueError('COLS not found in build_qa_table.py')


def p3_status_of(row):
    """build_qa_table.py:215-229 (rule as oracle)."""
    if (row.get('verdict') or '').lower() == 'exclude':
        return 'excluded (data_decisions)'
    if row['p3g_in_record'] == '':
        return ''
    if row['p3g_nightgate'] == 'FAIL':
        return 'refused: night-image gate'
    if row['p3g_epochgate'] == 'FAIL':
        return 'refused: epoch gate'
    sv = [st for st in ('V', 'I') if row[f'p3s_searched_{st}'] == '1']
    return (('in master; ' if row['p3g_in_master'] == '1' else 'not in master; ')
            + (f"searched {'+'.join(sv)}" if sv else 'not searched')
            + ''.join(f"; {st} refused: {row[f'p3s_refused_{st}']}" for st in ('V', 'I')
                      if row[f'p3s_refused_{st}'] and row[f'p3s_searched_{st}'] != '1'))


@content_test('qa_table')
def qa_columns(path, **ctx):
    want = qa_table_columns()
    cols = _cols(path)
    if cols != want:
        return False, None, f'header differs from build_qa_table.py COLS ({len(cols)} vs {len(want)} columns)'
    rows = _rows(path)
    seen = set()
    for r in rows:
        k = (r['lst'], r['date'], r['subband'])
        if k in seen:
            return False, k, 'duplicate (lst, date, subband)'
        seen.add(k)
        if r['subband'] in ('all', ''):
            return False, k, "row with subband 'all' or ''"
        if p3_status_of(r) != r['p3_status']:
            return False, k, f"p3_status {r['p3_status']!r} != re-derived {p3_status_of(r)!r}"
    return True, len(rows), f'{len(rows)} rows; header and p3_status re-derived'


def _solar_literals():
    from exopipe.phase3.lib.status import steps_inplace as IP
    tree = ast.parse(open(IP.legacy_script('solar_contamination_audit.py')).read())
    lit = {}
    for n in tree.body:
        if isinstance(n, ast.Assign):
            for t in n.targets:
                names = [t.id] if isinstance(t, ast.Name) else ([e.id for e in t.elts] if isinstance(t, ast.Tuple) else [])
                try:
                    v = ast.literal_eval(n.value)
                except ValueError:
                    continue
                if len(names) == 1:
                    lit[names[0]] = v
                elif isinstance(v, tuple) and len(v) == len(names):
                    lit.update(dict(zip(names, v)))
    return lit


@content_test('solar_audit')
def verdict_rule(path, **ctx):
    """Verdict and cause re-derived from (R_f, R_46, S_f, sun_el_deg) by solar_contamination_audit.py:63-89, with the
    constants of status_qa (refused when they differ from the script's literals)."""
    Q = K.cfg().status_qa
    lit = _solar_literals()
    if (lit.get('S_WARN'), lit.get('S_BAD'), tuple(lit.get('LOW', ())), lit.get('REF')) != \
            (Q.SOLAR_S_WARN, Q.SOLAR_S_BAD, tuple(Q.SOLAR_SUBBANDS_MHZ), Q.SOLAR_REF_MHZ):
        return False, None, 'status_qa.SOLAR_* constants differ from the in-place script literals'
    lo, hi = Q.SOLAR_REF_WINDOW
    rows = _rows(path)
    seen = set()
    for r in rows:
        k = (r['lst'], r['date'], r['subband'])
        if k in seen:
            return False, k, 'duplicate row'
        seen.add(k)
        if int(r['subband'].replace('MHz', '')) not in Q.SOLAR_SUBBANDS_MHZ:
            return False, k, 'subband not judged by the audit'
        rf, r46, sf, el = _f(r['R_f']), _f(r['R_46']), _f(r['S_f']), _f(r['sun_el_deg'])
        if r46 and abs(rf / r46 - sf) > 0.0015 + 1e-3 * abs(sf):
            return False, k, f'S_f {sf} != R_f/R_46 {rf / r46:.3f}'
        ref_ok = lo <= r46 <= hi
        if rf >= Q.SOLAR_S_BAD:
            v = 'QUARANTINE'
        elif not ref_ok:
            v = 'reference-band-anomalous' if rf > Q.SOLAR_S_WARN else 'ok (46 MHz anomalous)'
        elif sf >= Q.SOLAR_S_BAD:
            v = 'QUARANTINE'
        elif sf >= Q.SOLAR_S_WARN or rf >= Q.SOLAR_S_WARN:
            v = 'suspect'
        else:
            v = 'ok'
        if v != r['verdict']:
            # rounding at the boundaries (values stored to 3 decimals): accept a disagreement only within rounding
            near = min(abs(x - t) for x in (rf, sf, r46) for t in (Q.SOLAR_S_WARN, Q.SOLAR_S_BAD, lo, hi))
            if near > 0.0011:
                return False, k, f"verdict {r['verdict']!r} != re-derived {v!r}"
        c1, c2 = Q.SOLAR_CAUSE_DEG
        cause = 'solar' if el > c1 else ('twilight' if el > c2 else 'not-solar')
        if cause != r['cause'] and min(abs(el - c1), abs(el - c2)) > 0.051:
            return False, k, f"cause {r['cause']} != {cause}"
    return True, len(rows), f'{len(rows)} blocks; verdict and cause re-derived'


@content_test('beam_audit')
def ratio_rule(path, **ctx):
    Q = K.cfg().status_qa
    rows = _rows(path)
    for r in rows:
        a, b, m, q = _f(r['bmaj_as']), _f(r['bmin_as']), _f(r['cell_median_area']), _f(r['area_ratio'])
        if None in (a, b, m, q) or m <= 0:
            continue
        if abs(a * b / m - q) > 0.002 + 2e-4 * q:
            return False, r.get('path'), f'area_ratio {q} != bmaj x bmin / cell median {a * b / m:.4f}'
    fl = os.path.join(os.path.dirname(path), 'beam_audit_flagged.csv')
    if os.path.exists(fl):
        # beam_audit.py:82 flags on the unrounded ratio, the file stores it to 3 decimals (:79): a stored ratio within
        # half a unit of the threshold is undecidable from the file and left out of the comparison
        amb = {r['path'] for r in rows if _f(r['area_ratio']) is not None and
               abs(_f(r['area_ratio']) - Q.BEAM_RATIO_FLAG) <= 0.0005}
        want = {r['path'] for r in rows if _f(r['area_ratio']) is not None and _f(r['area_ratio']) >= Q.BEAM_RATIO_FLAG} - amb
        got = {r['path'] for r in _rows(fl)} - amb
        if want != got:
            return False, len(want ^ got), f'flagged file differs from rows with area_ratio >= {Q.BEAM_RATIO_FLAG}'
    return True, len(rows), f'{len(rows)} epochs; ratios and flags re-derived'


@content_test('noise_plot')
def png_decodes(path, **ctx):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.image as mpimg
    img = mpimg.imread(path)
    h, w = img.shape[:2]
    want = tuple(K.cfg().status_qa.NOISE_PX)
    if (w, h) != want:
        return False, (w, h), f'size {w}x{h} != {want[0]}x{want[1]}'
    if float(np.ptp(img[..., :3])) == 0.0:
        return False, None, 'a single colour'
    return True, (w, h), 'decodes; size as configured; not blank'


def _bandpass_decimals():
    """The rounding of bandpass_phase_warp.py (round(v, N) at :112), read from its source (never executed)."""
    from exopipe.phase3.lib.status import steps_inplace as IP
    m = re.search(r'round\(v,\s*(\d+)\)', open(IP.legacy_script(K.cfg().status_qa.BANDPASS_SCRIPT)).read())
    return int(m.group(1)) if m else None


@content_test('bandpass_phase_qa')
def bandpass_rows(path, tables_list=None, **ctx):
    """One row per table (and, given the run's shard list ``tables_list``, exactly the listed tables); error rows carry no
    measures; ratio = res_hi / res_in within the rounding of the three stored values (round(v, BANDPASS_DECIMALS),
    bandpass_phase_warp.py:86-87, :112); rows sorted by (cal_date, cal_lst) (run_bandpass_phase_qa.sh:19)."""
    nd = K.cfg().status_qa.BANDPASS_DECIMALS
    if _bandpass_decimals() != nd:
        return False, None, f'status_qa.BANDPASS_DECIMALS {nd} differs from the in-place script ({_bandpass_decimals()})'
    h = 0.5 * 10.0 ** -nd                             # half a unit of the last stored decimal
    rows = _rows(path)
    seen = set()
    for r in rows:
        if r['table'] in seen:
            return False, r['table'], 'duplicate table row'
        seen.add(r['table'])
        if r.get('error'):
            if any(r.get(c) for c in ('res_in', 'res_hi', 'ratio')):
                return False, r['table'], 'error row with measures'
            continue
        ri, rh, ra = _f(r['res_in']), _f(r['res_hi']), _f(r['ratio'])
        if None not in (ri, rh, ra) and ri > h:
            bound = h * (1.0 + 1.0 / (ri - h) + (rh + h) / (ri - h) ** 2) * 1.001
            if abs(rh / ri - ra) > bound:
                return False, r['table'], f'ratio {ra} != res_hi/res_in {rh / ri:.5f} (rounding bound {bound:.5f})'
    keys = [(r['cal_date'], r['cal_lst']) for r in rows]
    if keys != sorted(keys):
        return False, None, 'rows not sorted by (cal_date, cal_lst)'
    if tables_list:
        if not os.path.exists(tables_list):
            return False, tables_list, 'the shard table list of the run is gone (cannot check one row per table)'
        want = {ln.strip() for ln in open(tables_list) if ln.strip()}
        if want != seen:
            return False, len(want ^ seen), (f'{len(want - seen)} listed tables without a row (e.g. {sorted(want - seen)[:1]}), '
                                             f'{len(seen - want)} rows not in the list')
    return True, len(rows), f'{len(rows)} tables, {sum(1 for r in rows if r.get("error"))} failed' + \
        (' ; one row per listed table' if tables_list else '')


@content_test('provenance_audit')
def provenance_rows(path, **ctx):
    cols = _cols(path)
    rows = _rows(path)
    ex = K.exclusions()
    if 'excluded_data' in cols:                       # candidate_provenance_audit table
        for r in rows:
            exc = int(ex.is_excluded(r['lst'], r['band'], r['date']))
            if int(r['excluded_data']) != exc:
                return False, r['cand_id'], f"excluded_data {r['excluded_data']} but core.exclusions says {exc} now"
            if not exc and r['solar'] not in ('SUN-UP', 'TWILIGHT-FLAGGED-BAND'):
                return False, r['cand_id'], f"row neither excluded nor solar ({r['solar']})"
        return True, len(rows), f'{len(rows)} flagged detections; exclusions re-checked'
    if 'sun_min' in cols:                             # perframe_solar_check table
        for r in rows:
            lo, hi, v = _f(r['sun_min']), _f(r['sun_max']), r['verdict']
            if v == 'NO-FRAME' or lo is None:
                continue
            exp = ('SUN-UP' if lo > 0 else 'SUN-UP-PARTIAL') if hi > 0 else None
            if exp and v != exp:
                return False, r['cand_id'], f'verdict {v} != {exp}'
            if exp is None and v.startswith('SUN-UP'):
                return False, r['cand_id'], f'verdict {v} with the Sun below the horizon'
            if hi is not None and hi <= -18.0 and v != 'night':
                return False, r['cand_id'], f'verdict {v} with the Sun below -18 deg'
        return True, len(rows), f'{len(rows)} detections; per-frame verdicts re-derived where determinate'
    return False, None, 'not a provenance table'


@content_test('membership_conformance')
def membership_rows(path, **ctx):
    C = K.cfg()
    cols = _cols(path)
    if cols != list(C.status_conformance.COLUMNS):
        return False, None, f'columns {cols}'
    rows = _rows(path)
    cells = [r['cell'] for r in rows]
    if len(cells) != len(set(cells)):
        return False, None, 'duplicate cell'
    for r in rows:
        if (r['status'] == 'PASS') != (int(r['n_violations']) == 0):
            return False, r['cell'], 'status disagrees with n_violations'
        if not os.path.exists(os.path.join(C.paths.COADD_SB_ROOT, r['cell'], 'coadd_final', 'weights.csv')):
            return False, r['cell'], 'cell without coadd_final/weights.csv'
    hist = os.path.join(os.path.dirname(path), os.path.basename(C.status_conformance.HISTORY))
    if os.path.exists(hist):
        h = _rows(hist)
        if h:
            last = h[-1]
            npass = sum(r['status'] == 'PASS' for r in rows)
            if (int(last['cells']), int(last['pass'])) != (len(rows), npass) and \
                    last['checked_utc'] == max(r['checked_utc'] for r in rows):
                return False, None, 'last history row differs from the table'
    return True, len(rows), f'{sum(r["status"] == "PASS" for r in rows)} of {len(rows)} PASS'


@content_test('qa_night_pages')
def pages_check(path, **ctx):
    """P1-P4: build_qa_pages.check() IN PLACE on the hub folder ``path`` (HUB set to it; ledger, data_decisions and the
    working tree from config); P5: the stage-3 window drawn equals membership.qa3_window for all 19 bands."""
    from exopipe.phase3.lib.status import steps_inplace as IP
    from exopipe.phase3.lib.masters import membership as M
    C = K.cfg()
    hub = path if os.path.isdir(path) else os.path.dirname(os.path.dirname(path))
    attrs = dict(HUB=hub.rstrip('/'), LEDGER=ctx.get('ledger', C.status.LEDGER), DD=C.paths.DATA_DECISIONS,
                 ROOT=C.paths.COADD_SB_ROOT, QA=C.status.QA_DIR)
    IP.search_sources(C.status.SEARCH_LABEL['deep'])      # the register the page imports, pinned (2026-10-15 review)
    rc, text, mod = IP.call_attr('build_qa_pages', attrs, 'check', allowed_roots=('/tmp',),
                                 env=IP.search_env(C.status.SEARCH_LABEL['deep']))
    line = next((ln for ln in text.splitlines() if ln.startswith('CHECK:')), text.strip()[-300:])
    if mod is None or 'CHECK PASS' not in text:
        return False, rc, line[:400]
    bands = list(C.instrument.ALL_SUBBANDS) + ['Red', 'Green', 'Blue', 'Full']
    bad = [b for b in bands if tuple(mod.qa3_window(b)) != tuple(M.qa3_window(b))]
    if bad:
        return False, bad[0], f'stage-3 window drawn differs from membership.qa3_window for {bad}'
    return True, None, line[:300] + '; P5 stage-3 windows equal'


@content_test('status_page')
def status_page(path, **ctx):
    C = K.cfg()
    s = open(path).read()
    if '</html>' not in s:
        return False, None, 'page not complete (no </html>)'
    d = os.path.dirname(path)
    b = os.path.basename(path)
    if b == os.path.basename(C.status_conformance.PAGE):
        lat = os.path.join(os.path.dirname(d), 'QA', os.path.basename(C.status_conformance.LATEST))
        lat = lat if os.path.exists(lat) else K.expand(C.status_conformance.LATEST)
        rows = _rows(lat)
        m = re.search(r'(\d+) of (\d+) cells PASS', s)
        npass = sum(r['status'] == 'PASS' for r in rows)
        if not m or (int(m.group(1)), int(m.group(2))) != (npass, len(rows)):
            return False, None, f'page summary differs from {lat} ({npass} of {len(rows)})'
        return True, npass, f'{npass} of {len(rows)} PASS as the table'
    if b == C.status_pages.REGISTER_PAGE:
        # the page's count table (build_ratified_register.py:77-78: one <th> per status of REGISTER_ORDER present, then
        # 'total'; one <td> per count) equals Counter(status) of the register beside it (this run's, else production's)
        reg = os.path.join(os.path.dirname(d), 'clv15', os.path.basename(C.status_report.REGISTER))
        reg = reg if os.path.exists(reg) else C.status_report.REGISTER
        rr = _rows(reg)
        cnt = collections.Counter(r['status'] for r in rr)
        m = re.search(r'<table><tr>((?:<th>[^<]*</th>)+)</tr><tr>((?:<td[^>]*>[^<]*</td>)+)</tr></table>', s)
        if not m:
            return False, None, 'no status count table on the page'
        page = dict(zip(re.findall(r'<th>([^<]*)</th>', m.group(1)), re.findall(r'<td[^>]*>([^<]*)</td>', m.group(2))))
        want = {st: str(cnt[st]) for st in C.status_pages.REGISTER_ORDER if cnt.get(st)}
        want['total'] = str(len(rr))
        if page != want:
            diff = sorted(k for k in set(page) | set(want) if page.get(k) != want.get(k))
            return False, diff[:3], f'page counts differ from {reg}: ' + ', '.join(
                f'{k} page {page.get(k)} register {want.get(k)}' for k in diff[:3])
        return True, len(rr), f'status counts on the page = {os.path.basename(reg)} ({len(rr)} rows)'
    links = set(re.findall(r'href="((?:nights|dates)/[^"]+\.html)"', s))
    missing = [u for u in links if not os.path.exists(os.path.join(d, u))]
    if missing:
        return False, len(missing), f'{len(missing)} night/date links missing, e.g. {missing[0]}'
    return True, len(links), f'{len(links)} night/date links resolve'


@content_test('processing_ledger')
def ledger_rows_consistent(path, **ctx):
    """Columns; one row per (lst, date); p1/p2 'complete' iff the subband (and wideband) counts; dd_excl_subbands = the
    distinct excluded scopes of the night in core.exclusions now; qa_verdict / qa_basis re-derived by the rule of
    _build_ledger.derive (:16-32; the script runs at import, so its rule is the oracle here)."""
    C = K.cfg()
    L = C.status_ledger
    cols = _cols(path)
    if cols != list(L.COLUMNS):
        return False, None, f'columns {cols}'
    ex = K.exclusions()
    n_ex = collections.Counter()
    for (l, s, d) in ex.rows:
        n_ex[(l, d)] += 1
    rows = _rows(path)
    seen = set()
    for r in rows:
        k = (r['lst'], r['date'])
        if k in seen:
            return False, k, 'duplicate (lst, date)'
        seen.add(k)
        if (r['p1'] == 'complete') != (r['p1'] != 'none' and int(r['p1_nsub'] or 0) == L.P1_COMPLETE_NSUB):
            return False, k, 'p1 status disagrees with p1_nsub'
        if (r['p2'] == 'complete') != (r['p2'] != 'none' and (int(r['p2_nsub'] or 0), int(r['p2_nwide'] or 0)) ==
                                       tuple(L.P2_COMPLETE)):
            return False, k, 'p2 status disagrees with p2_nsub / p2_nwide'
        if int(r['dd_excl_subbands'] or 0) != n_ex.get(k, 0):
            return False, k, f"dd_excl_subbands {r['dd_excl_subbands']} != {n_ex.get(k, 0)} excluded scopes now"
        ne = int(r['dd_excl_subbands'] or 0)
        dd_status = 'all-excluded' if ne == L.ALL_EXCLUDED_N else ('partial-excluded' if ne else 'none')
        if r['dd_status'] != dd_status:
            return False, k, f"dd_status {r['dd_status']} != {dd_status}"
        qv, qb = _derive(r, dd_status)
        if (qv, qb) != (r['qa_verdict'], r['qa_basis']) and not (qv == 'exclude' and r['qa_verdict'] == 'exclude'):
            return False, k, f"qa_verdict/basis ({r['qa_verdict']}, {r['qa_basis']}) != ({qv}, {qb})"
    return True, len(rows), f'{len(rows)} (LST, night) rows consistent'


def _derive(r, dd_status):
    """_build_ledger.derive (:16-32) as oracle; the reason-specific exclude basis is checked only for its verdict."""
    if r['p2'] == 'none':
        return 'incomplete', 'awaiting Phase 2'
    rl = (r.get('dd_reason') or '').lower()
    if 'peeling divergence' in rl and int(r['p1_npeel'] or 0) >= K.cfg().status_ledger.PEEL_FULL_MIN:
        return 'needs-QA', 'peel-reprocessed; stale exclusion, re-evaluate'
    if dd_status == 'all-excluded':
        if 'peeling divergence' in rl:
            return 'exclude', 'peeling divergence; NOT yet peel-reprocessed'
        return 'exclude', (r.get('dd_reason') or '')[:50]
    if dd_status == 'partial-excluded':
        return 'needs-QA', 'partial subband exclusion'
    return 'needs-QA', 'unverified (no exclusion, no positive QA)'


@content_test('ratified_register')
def register_rows(path, **ctx):
    P = K.cfg().status_pages
    cols = _cols(path)
    if cols != list(P.REGISTER_COLUMNS):
        return False, None, f'columns {cols}'
    rows = _rows(path)
    order = list(P.REGISTER_ORDER)
    for r in rows:
        if r['status'] != r['status'].upper():
            return False, r['ruling_id'], 'status not uppercase'
    keys = [(order.index(r['status']) if r['status'] in order else 99, r['date'], r['ruling_id']) for r in rows]
    if keys != sorted(keys):
        return False, None, 'rows not sorted by (status order, date, ruling_id)'
    return True, len(rows), f'{len(rows)} rulings; {sum(1 for r in rows if r["status"] not in order)} with an unknown status'


def eligibility_status(r):
    """snapshot_batch_audit.py:109-125 precedence over the row's own columns (oracle)."""
    if not int(r['shipped']):
        return 'NOT-SHIPPED'
    if int(r['dirty']):
        return 'HOLD-DIRTY'
    if int(r['n_flow_unregistered']) or int(r['n_tier1min_paused']) or int(r['n_missing_1min']):
        return 'WAIT-FRAMETIME-JOB' if int(r['frametime_job']) else 'HOLD-CHAIN-REFUSAL'
    if int(r['frametime_job']):
        return 'WAIT-FRAMETIME-JOB'
    if r['lightcurves_missing'].strip():
        return 'WAIT-LIGHTCURVES'
    if int(r['old_stage_active']) or int(r['prefit_active']):
        return 'WRITER-ACTIVE'
    if int(r['prefit_done']):
        return 'PREFIT-STALE' if int(r['prefit_stale']) else 'PREFIT-DONE'
    return 'TO-PREFIT'


@content_test('snapshot_eligibility')
def status_precedence(path, **ctx):
    rows = _rows(path)
    for r in rows:
        try:
            want = eligibility_status(r)
        except (KeyError, ValueError) as e:
            return False, r.get('cell'), f'column missing or not an integer: {e}'
        if want != r['status']:
            return False, r['cell'], f"status {r['status']} != re-derived {want}"
    return True, len(rows), f'{len(rows)} cells; ' + ', '.join(f'{k} {n}' for k, n in
                                                                 sorted(collections.Counter(r['status'] for r in rows).items()))


@content_test('snapshot_batch_list')
def list_rule(path, all=None, max_new=None, base=None, **ctx):
    """list = base + new (no duplicate); every new cell is a candidate whose audit status is in LIST_OK (LIST_OK_ALL with
    --all); new cells in candidate order; at most max_new new cells unless --all; every candidate not listed is in
    left_out.txt (snapshot_batch_lists.py:86-115). ``all``/``max_new``/``base`` come from the run (record context); without
    them only the status set LIST_OK_ALL is checked."""
    C = K.cfg().status_snapshot
    cells = [l.split()[0] for l in open(path) if l.strip()]
    if len(cells) != len(set(cells)):
        return False, None, 'duplicate cell in the list'
    aud = path + '.audit.csv'
    left = path + '.left_out.txt'
    cand = path + '.candidates.txt'
    for p in (aud, left, cand):
        if not os.path.exists(p):
            return False, p, 'sidecar missing'
    st = {r['cell']: r['status'] for r in _rows(aud)}
    cands = [l.strip() for l in open(cand) if l.strip()]
    lefts = {l.split()[0] for l in open(left) if l.strip()}
    new = [c for c in cands if c in cells]
    ok = C.LIST_OK_ALL if (all is None or all) else C.LIST_OK
    for c in new:
        if st.get(c) not in ok:
            return False, c, f'new cell with audit status {st.get(c)} (allowed {list(ok)})'
    missing = [c for c in cands if c not in cells and c not in lefts]
    if missing:
        return False, missing[0], 'candidate neither listed nor in left_out'
    if base is not None:
        if not os.path.exists(base):
            return False, base, 'base list of the run is gone'
        bl = [l.split()[0] if len(l.split()) == 1 else '_'.join(l.split()[:2]) for l in open(base)
              if l.split() and not l.split()[0].startswith('#')]
        if cells != bl + new:
            return False, None, f'list != base ({len(bl)}) + new candidates in candidate order ({len(new)})'
    if all is False and max_new is not None and len(new) > int(max_new):
        return False, len(new), f'{len(new)} new cells > max_new {max_new}'
    return True, len(cells), f'{len(cells)} cells ({len(new)} new); every candidate listed or left out with its status'
