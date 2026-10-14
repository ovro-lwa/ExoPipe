"""`status report` stages that are not coverage / rebuild / stale / exclusion / served-diff: the session summary
(CLAUDE.md section 1), the re-adjudication list, the cell-load review (contract step 50) and the membership violations.

summary          pkg/phase3/phase_status.py IN PLACE (branch lib/status/phase_status.py: f824776 pure rename, then one edit
                 2026-10-14 - masters counted from I_deep_Taper_R075_<scope>.fits carrying COADDOP, newest master by that
                 file, never coadd_final/.done; production still counts the markers, 0 of them today against 456 masters):
                 its globals set from config (LEDGER, DECISIONS, DECISIONS_FP, COADD_ROOTS, RETIRED_COADD) and main()'s
                 text captured; with status_report.FINGERPRINT_MODE 'read' the production fingerprint file is COPIED into
                 <out>/status and phase_status writes the copy (the SessionStart hook's "changed" flag is never consumed).
                 Then the sections CLAUDE.md 1 asks for: node allocation (site NODE_POOLS), phase completion (ledger),
                 data_decisions (core/exclusions fingerprint), served runs (hub registry), re-adjudication list, masters
                 conformance (latest membership table), rulings not enforced (the register), canon check output.
readjudication   every all- or partial-excluded (LST, night) of the ledger whose (p1_peel, p2) differs from the previous
                 summary record's state (first run: the newest production QA/readjudication_list_*.csv); columns and
                 go_no_go text of QA/readjudication_list_20261007.csv. Proposals for Gregg; data_decisions is never written.
cell-load        per (Stokes, LST, night, band): detections in <run>/selected_all.csv and served sources in
                 <run>/source_catalogs/pages_input_*.csv, sorted by detections (port of ops/p3s3_pass.sh:210-222;
                 R-CELL-LOAD-REVIEW-AFTER-FIRST-ITERATION, ledger 836: report only, no cut is made from it).
violations       deep cells FAIL in the latest membership conformance table (this --out's, else production's).
"""
import collections
import glob
import json
import os
import shutil
import subprocess
import time

from exopipe.phase3.lib.status import steps_common as K

REPORT_SECTIONS = ('node_allocation', 'phase_completion', 'data_decisions', 'masters', 'served_runs', 'readjudication',
                   'conformance', 'rulings_not_enforced', 'canon_check')
LEDGER_COUNTERS = ('p1', 'p2', 'p1_peel', 'qa_verdict')
# the text report: the new sections follow this header line (after phase_status.py's own text), one line per tag
TEXT_HEADER = '===== exopipe3 status: sections of CLAUDE.md section 1 ====='
TEXT_TAGS = ('[nodes]', '[ledger]', '[data_decisions]', '[masters]', '[served]', '[re-adjudication]', '[conformance]',
             '[rulings not enforced]', '[canon check]')
CELL_LOAD_SORT = 'detections_5sig descending (p3s3_pass.sh:218)'


def ledger_counts(rows):
    return {k: dict(collections.Counter(r.get(k, '') for r in rows)) for k in LEDGER_COUNTERS}


def legacy_summary_text(out, allowed, unit=None, log=None):
    """phase_status.main() in place (attr). Returns (rc, text)."""
    from exopipe.phase3.core import env as E
    from exopipe.phase3.lib.status import steps_inplace as IP
    C = K.cfg()
    R = C.status_report
    fp_copy = K.place(out, 'status', '.data_decisions.fingerprint')
    # the production fingerprint lives on a node-local disk: on another host the same path is a different file
    here = E.readable_here(R.DECISIONS_FINGERPRINT) and os.path.exists(R.DECISIONS_FINGERPRINT)
    if R.FINGERPRINT_MODE == 'write':
        fp = K.check_write(R.DECISIONS_FINGERPRINT)
    else:
        os.makedirs(os.path.dirname(fp_copy), exist_ok=True)
        if R.FINGERPRINT_MODE == 'read' and here:
            shutil.copyfile(R.DECISIONS_FINGERPRINT, fp_copy)
        fp = fp_copy
    if unit is not None and here:
        unit.add_input(R.DECISIONS_FINGERPRINT)
    elif unit is not None:
        unit.note(f'{R.DECISIONS_FINGERPRINT} not read on {E.host()} (node-local to '
                  f'{E.node_local_holder(R.DECISIONS_FINGERPRINT) or "no host"} or absent): no CHANGED flag compared')
    attrs = dict(LEDGER=C.status.LEDGER, DECISIONS=C.paths.DATA_DECISIONS, DECISIONS_FP=fp,
                 COADD_ROOTS=list(R.COADD_ROOTS), RETIRED_COADD=R.RETIRED_COADD)
    rc, text, _ = IP.call_attr('exopipe.phase3.lib.status.phase_status', attrs, 'main', allowed_roots=allowed, log=log,
                               unit=unit)
    return rc, text


def canon_check():
    R = K.cfg().status_report
    try:
        p = subprocess.run(R.CANON_CHECK, shell=True, capture_output=True, text=True, timeout=R.CANON_TIMEOUT_S)
        return p.stdout + p.stderr
    except Exception as e:                            # noqa: BLE001 - the report states it, never hides it
        return f'canon check could not run: {e}'


def masters_section():
    """Per co-add root: cells, cells with a build record (coadd_final/weights.csv), masters = I_deep_Taper_R075_<scope>
    .fits carrying COADDOP (phase_status.master_inventory: the file, never a .done marker) per COADDOP value, and the
    newest master's write time and age from that file. Every cell of each root is counted (no precedence here)."""
    from exopipe.phase3.core import env as E
    from exopipe.phase3.lib.status import phase_status as PS
    C = K.cfg()
    out = {}
    me = E.host()
    for r in C.status_report.COADD_ROOTS:
        if not E.readable_here(r):           # node-local to another host: the same path here is a different disk
            out[r] = f'node-local to {E.node_local_holder(r)}, not read on {me}'
        elif os.path.isdir(r):
            cells = [d for d in K.listdir(r) if len(d) > 3 and d[2] == 'h' and '_' in d]
            inv = PS.master_inventory(r, cells)
            nw = inv['newest']
            out[r] = {'cells': len(cells), 'with_weights_record': sum(
                1 for c in cells if os.path.exists(os.path.join(r, c, 'coadd_final', 'weights.csv'))),
                'masters_coaddop': inv['masters'], 'coaddop': inv['coaddop'], 'masters_without_coaddop':
                inv['without_coaddop'], 'master_file': PS.MASTER,
                'newest_master': (os.path.relpath(nw[1], r) if nw else None),
                'newest_master_utc': (time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(nw[0])) if nw else None),
                'newest_master_age_h': (round((time.time() - nw[0]) / 3600, 1) if nw else None)}
        else:
            out[r] = f'absent on {me}'
    return out


def masters_text(masters):
    """The [masters] line of the text report from the JSON masters section (the content test compares the two; roots
    and COADDOP values in sorted order, as the JSON is written with sorted keys)."""
    parts = []
    for r, v in sorted(masters.items()):
        if isinstance(v, dict):
            parts.append(f"{r}: {v['cells']} cells, {v['masters_coaddop']} masters carrying COADDOP "
                         f"{dict(sorted(v['coaddop'].items()))}"
                         + (f", {v['masters_without_coaddop']} without" if v.get('masters_without_coaddop') else '')
                         + (f", newest {v['newest_master']} {v['newest_master_age_h']} h ago" if v.get('newest_master')
                            else ''))
        else:
            parts.append(f'{r}: {v}')
    return '; '.join(parts)


def conformance_section(out_root):
    C = K.cfg()
    for p in (K.place(out_root, 'QA', os.path.basename(C.status_conformance.LATEST)), K.expand(C.status_conformance.LATEST)):
        if os.path.exists(p):
            rows = K.read_csv(p)
            return {'table': p, 'cells': len(rows), 'pass': sum(r.get('status') == 'PASS' for r in rows),
                    'fail': sum(r.get('status') == 'FAIL' for r in rows),
                    'checked_utc': max((r.get('checked_utc', '') for r in rows), default='')}
    return {'table': None}


def rulings_section():
    R = K.cfg().status_report
    if not os.path.exists(R.REGISTER):
        return {'register': R.REGISTER, 'missing': True}
    c = collections.Counter(r.get('status', '').strip().upper() for r in K.read_csv(R.REGISTER))
    return {'register': R.REGISTER, 'problem_counts': {k: c.get(k, 0) for k in R.REGISTER_PROBLEM},
            'all_counts': dict(c)}


def served_section():
    try:
        p, reg = K.registry()
    except (OSError, ValueError) as e:
        return {'registry': None, 'error': str(e)}
    rows = []
    for S in ('V', 'I'):
        for k, v in (reg.get(S) or {}).items():
            if isinstance(v, dict):
                rows.append({'stokes': S, 'page': k, 'port': v.get('port'), 'run': v.get('run'),
                             'served': v.get('served'), 'live': v.get('live')})
    return {'registry': p, 'pages': rows}


# ---------------------------------------------------------------------------------------------------------------------
# re-adjudication
# ---------------------------------------------------------------------------------------------------------------------

def readjudication_state(ledger_rows):
    return {f"{r['lst']}_{r['date']}": [r.get('p1_peel', ''), r.get('p2', '')] for r in ledger_rows
            if r.get('dd_status') in ('all-excluded', 'partial-excluded')}


def previous_state(out_root, tier):
    """(state dict, source) - the previous summary record's state, else the newest production list's rows."""
    from exopipe.phase3.core import record as R
    rec = R.load_record(out_root, 'status', K.unit_name('report', 'summary', tier))
    if rec and (rec.get('meta') or {}).get('readjudication_state') is not None:
        return rec['meta']['readjudication_state'], f"record {R.record_path(out_root, 'status', rec['unit'])}"
    lists = sorted(glob.glob(K.expand(K.cfg().status_report.READJUDICATION_GLOB)))
    if lists:
        return ({f"{r['lst']}_{r['date']}": [r.get('p1_peel', ''), r.get('p2', '')] for r in K.read_csv(lists[-1])},
                f'production list {lists[-1]} (baseline: cells not on it are not judged this run)')
    return {}, 'none (first run: baseline established, nothing flagged)'


def readjudication_rows(ledger_rows, prev):
    R = K.cfg().status_report
    out = []
    for r in ledger_rows:
        if r.get('dd_status') not in ('all-excluded', 'partial-excluded'):
            continue
        k = f"{r['lst']}_{r['date']}"
        if k in prev and list(prev[k]) != [r.get('p1_peel', ''), r.get('p2', '')]:
            out.append(dict(lst=r['lst'], date=r['date'], dd_status=r['dd_status'],
                            excluded_subbands=r.get('dd_excl_subbands', ''), reason=r.get('dd_reason', ''),
                            p1_peel=r.get('p1_peel', ''), p2=r.get('p2', ''), go_no_go=R.READJUDICATION_GO))
    return out


# ---------------------------------------------------------------------------------------------------------------------
# cell-load, violations
# ---------------------------------------------------------------------------------------------------------------------

def cell_load(run_dir):
    """Rows of the cell-load review (p3s3_pass.sh:213-221) and the input files read."""
    sel = os.path.join(run_dir, 'selected_all.csv')
    if not os.path.exists(sel):
        from exopipe.phase3.core.record import NoInput
        raise NoInput(f'{sel} not found (the run has no selection)')
    det, src = collections.Counter(), collections.Counter()
    for r in K.read_csv(sel):
        det[(r['stokes'], r['lst'], r['date'], r['band'])] += 1
    pin = sorted(glob.glob(os.path.join(run_dir, 'source_catalogs', 'pages_input_*.csv')))
    for p in pin:
        for r in K.read_csv(p):
            src[(r.get('stokes', ''), r['lst'], r['date'], r['band'])] += 1
    # p3s3_pass.sh:218 sorts by detections only, over a set: ties come out in hash order (differs per process); the key
    # breaks ties deterministically (the same rows, an order production can also produce)
    keys = sorted(set(det) | set(src), key=lambda k: (-(det.get(k, 0)), k))
    rows = [dict(stokes=k[0], lst=k[1], date=k[2], band=k[3], detections_5sig=det.get(k, 0), served_sources=src.get(k, 0))
            for k in keys]
    return rows, [sel] + pin


def membership_table(out_root):
    C = K.cfg()
    for p in (K.place(out_root, 'QA', os.path.basename(C.status_conformance.LATEST)), K.expand(C.status_conformance.LATEST)):
        if os.path.exists(p):
            return p
    return None


def violations(out_root, cells=None):
    p = membership_table(out_root)
    if not p:
        from exopipe.phase3.core.record import NoInput
        raise NoInput('no membership conformance table (run `exopipe3 status conformance` first)')
    rows = [r for r in K.read_csv(p) if r.get('status') == 'FAIL' and (not cells or r.get('cell') in cells)]
    return rows, p


def report_json(path, obj):
    K.write_text(path, json.dumps(obj, indent=1, sort_keys=True, default=str) + '\n')


def now_utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
