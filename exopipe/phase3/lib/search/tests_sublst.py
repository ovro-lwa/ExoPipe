"""Content tests of the sub-LST `exopipe3 search` product classes that the deep classes do not cover (core/validate
discovers this module). The catalogue, ledgers, refusal records, stamps, FP-gate verdicts and fit sidecar of a sub-LST unit
are the deep classes (lib/search/tests.py: detections, gate_ledger - including the sub-LST 'no in-search table removal'
rule of sublst_search_pass.sh:49-50 - image_ledger, refusal_record, candidate_stamps, fp_gate_verdicts,
detection_positions), the parent catalogue is masters' master_catalog, frames and sigma stacks are frames' search_frame and
noise_map; tier_sublst passes them tier='sublst'.

Classes registered here:
  sublst_search_log      <cell>.log (or <cell>_<night>.log of a per-hour append): the success criteria of
                         sublst_search_pass.sh:51-52 on the log's CONTENT - the engine's summary line '<lst> <band>: N epochs
                         -> M candidates', the completion sentinel 'SUBLST-SEARCH-DONE <cell> <S>', no Python traceback and
                         no REFUSING line - plus M = the catalogue rows of the searched windows when the catalogue is beside it.
  sublst_verify_report   verify_report.json written by a --replay-attempt or no-publish run: parses, names its root and unit,
                         every test entry has name/passed, every path it names exists.
"""
import csv
import json
import os
import re

from exopipe.phase3.core.validate import content_test

_SUMMARY = re.compile(r'^\s*(\d\dh) (\S+): (\d+) epochs -> (\d+) candidates', re.M)
_DONE = re.compile(r'^SUBLST-SEARCH-DONE (\S+) ([IV])\s*$', re.M)
_LOG = re.compile(r'^(?P<lst>\d\dh)_(?P<band>\d+MHz|Red|Green|Blue)(?:_(?P<night>\d{4}-\d\d-\d\d))?\.log$')


def _read(path, limit=50 * 2 ** 20):
    with open(path, 'rb') as f:
        f.seek(0, 2)
        n = f.tell()
        f.seek(max(0, n - limit))
        return f.read().decode('utf-8', 'replace')


@content_test('sublst_search_log')
def completed(path, **ctx):
    """The search log shows a completed search (SP:51-52): the summary line of this cell, the DONE sentinel of this cell,
    no traceback, no REFUSING line; the summary's candidate count = the catalogue rows of the searched windows when the
    attempt's transients/Run_*/master_catalog.csv is beside the log (published layout)."""
    m = _LOG.match(os.path.basename(path))
    if not m:
        return False, None, f'not a <lst>_<band>[_<night>].log name: {os.path.basename(path)}'
    txt = _read(path)
    probs = []
    summ = [s for s in _SUMMARY.findall(txt) if s[0] == m['lst'] and s[1] == m['band']]
    done = [d for d in _DONE.findall(txt) if d[0] == f"{m['lst']}_{m['band']}"]
    if not summ:
        probs.append('no engine summary line "<lst> <band>: N epochs -> M candidates" of this cell')
    if not done:
        probs.append('no SUBLST-SEARCH-DONE sentinel of this cell')
    if re.search(r'^Traceback \(most recent call last\)', txt, re.M):
        probs.append('a Python traceback')
    if re.search(r'REFUSING', txt):
        probs.append('a REFUSING line')
    n_cand = int(summ[-1][3]) if summ else None
    if summ and done:
        st = done[-1][1]
        d = os.path.dirname(os.path.abspath(path))
        cats = [os.path.join(d, 'transients', x, 'master_catalog.csv') for x in sorted(os.listdir(os.path.join(d, 'transients')))
                ] if os.path.isdir(os.path.join(d, 'transients')) else []
        cats = [c for c in cats if os.path.exists(c)]
        if cats:
            with open(cats[0], newline='') as f:
                rows = [r for r in csv.DictReader(f) if r.get('stokes') == st]
            if m['night']:
                rows = [r for r in rows if r.get('date', '')[:10] == m['night']]
            else:                           # the pass's log: rows of nights appended later (their own logs) excluded
                later = {x[len(m['lst']) + len(m['band']) + 2:-4] for x in os.listdir(d)
                         if re.match(re.escape(f"{m['lst']}_{m['band']}_") + r'\d{4}-\d\d-\d\d\.log$', x)}
                rows = [r for r in rows if r.get('date', '')[:10] not in later]
            if len(rows) != n_cand:
                probs.append(f'summary says {n_cand} candidates, the catalogue holds {len(rows)} rows of Stokes {st}'
                             + (f' on {m["night"]}' if m['night'] else ''))
    if probs:
        return False, n_cand, '; '.join(probs)
    return True, n_cand, f'{summ[-1][2]} epochs -> {n_cand} candidates; DONE sentinel; no traceback'


@content_test('sublst_verify_report')
def well_formed(path, **ctx):
    """verify_report.json: root and unit named, the root exists, a non-empty test list of {name, passed}; every path a test
    names exists. The report RECORDS failures (the unit's own checks refuse it); the file is tested for being a complete
    record, and its value is the number of failed tests it lists."""
    with open(path) as f:
        rep = json.load(f)
    probs = []
    for k in ('root', 'unit', 'tests', 'written'):
        if k not in rep:
            probs.append(f'no {k}')
    if rep.get('root') and not os.path.isdir(rep['root']):
        probs.append(f"root {rep['root']} does not exist")
    tests = rep.get('tests') or []
    if not tests:
        probs.append('no test entries')
    bad = [t for t in tests if 'name' not in t or 'passed' not in t]
    if bad:
        probs.append(f'{len(bad)} entries without name/passed')
    gone = sorted({t['path'] for t in tests if t.get('path') and not os.path.exists(t['path'])})
    if gone:
        probs.append(f'{len(gone)} tested paths no longer exist, e.g. {gone[:2]}')
    nfail = sum(1 for t in tests if t.get('passed') is False)
    if probs:
        return False, nfail, '; '.join(probs)
    return True, nfail, f'{len(tests)} tests recorded ({nfail} failed) on {rep.get("root")}'
