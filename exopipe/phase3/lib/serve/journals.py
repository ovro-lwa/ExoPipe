"""Judgement journals: carry to a newly served run, the HOLD rule, and the sub-LST copy / swap / append.

  deep_carry(old, U)          deep_carry_judgements.py OLD NEW, in place (match by night and position within half a beam)
  snapshot_carry(old, DST)    snapshot_carry_judgements.py OLD NEW, in place (match by detection key); must print 'CARRY:'
                              (snapshot_rechain_batch.sh:374-375: 'carry produced no report' holds the serve)
  hold_check(old, DST)        the HOLD rule of snapshot_rechain_batch.sh:376-407, IN PROCESS (the ssh heredoc printed
                              nothing once: DEFECT-BATCH-DRIVER-SSH-N-HEREDOC, ledger 1797): an old judged source that
                              reached no new source with a verdict outside HOLD_FP_VERDICTS holds the serve (G19); the
                              missed list <DST>/judgements/carry_missed.csv is written either way
  sublst_copy_journals        sublst_cumulative.sh:98-100 (cp -p of each journal into the build)
  sublst_swap                 sublst_cumulative.sh:101-105 (U -> U.prev_<utc>, build -> U)
  append_between              sublst_cumulative.sh:108-120 (lines journalled on the old servers between copy and swap)
"""
import csv
import json
import os
import shutil

from exopipe.phase3.core.record import Refusal
from exopipe.phase3.lib.serve import procs, scripts


def _S():
    from exopipe.phase3.config import CFG
    return CFG.serve


def latest_carry(journal):
    """Latest source-level verdict per source as the carry tools read the journal (snapshot_carry_judgements.py:57-69):
    a retract clears; only action 'classify' with scope 'source' counts."""
    last = {}
    with open(journal) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            key = r['source_id']
            if r.get('action') == 'retract':
                last.pop(key, None)
            elif r.get('scope') == 'source' and r.get('action') == 'classify':
                last[key] = r
    return last


def latest_driver(journal):
    """The batch driver's reader (snapshot_rechain_batch.sh:381-384): last record per source_id of scope 'source'
    (default 'source'), whatever its action."""
    J = {}
    for l in open(journal):
        r = json.loads(l)
        if r.get('scope', 'source') == 'source' and r.get('source_id'):
            J[r['source_id']] = r
    return J


def served_run_of_port(port):
    """The --run of the review server listening on ``port`` (exact pid; snapshot_rechain_batch.sh:369), or None."""
    for pid in procs.listeners(port):
        info = procs.proc_info(pid)
        if procs.is_review_server(info):
            return procs.argv_value(info['argv'], '--run')
    return None


def journal_nonempty(path):
    return os.path.exists(path) and os.path.getsize(path) > 0


def deep_carry(old, U, log_path=None, unit=None):
    """Run deep_carry_judgements.py OLD U. Raises Refusal (G18) on its refusals; returns its CARRY lines."""
    for S_ in ('V', 'I'):
        nj = os.path.join(U, f'judgements_{S_}', 'judgements.jsonl')
        if journal_nonempty(nj):
            raise Refusal(f'G18 carry preconditions: {nj} exists and is not empty - carry-over only seeds a run that has '
                          f'no journal (deep_carry_judgements.py:66-67)')
    if not os.path.isdir(old):
        raise Refusal(f'G18 carry preconditions: no served run to carry from at {old}')
    res = scripts.run('deep_carry', [old, U], log=log_path, unit=unit)
    if not res.ok:
        raise Refusal(f'G18 deep carry refused (rc {res.rc}): {scripts.last_line(res.tail)}')
    lines = [l for l in res.tail.splitlines() if l.startswith('CARRY') or 'nothing to carry' in l]
    if not lines:
        raise Refusal('G18 deep carry printed no report - servers untouched')
    return lines


def snapshot_carry(old, DST, log_path=None, unit=None):
    """Run snapshot_carry_judgements.py OLD DST (the new journal must not exist). Returns the 'CARRY:' line."""
    nj = os.path.join(DST, 'judgements', 'judgements.jsonl')
    if journal_nonempty(nj):
        raise Refusal(f'G18 carry preconditions: {DST} already has a journal - not overwritten (snapshot_rechain_batch.sh:373)')
    res = scripts.run('snapshot_carry', [old, DST], log=log_path, unit=unit)
    cr = [l for l in res.tail.splitlines() if l.startswith('CARRY:')]
    if res.rc != 0 or not cr:
        raise Refusal(f'G18 snapshot carry: rc {res.rc}, ' + ('no CARRY report' if not cr else '') + f' - '
                      f'{scripts.last_line(res.tail)} - servers untouched')
    return cr[-1]


def hold_check(old, new, reader=None):
    """Write <new>/judgements/carry_missed.csv and return (not-FP list ['source (verdict)'], n missed rows)."""
    S = _S()
    reader = reader or S.HOLD_JOURNAL_READER
    oj = os.path.join(old, 'judgements', 'judgements.jsonl')
    J = latest_carry(oj) if reader == 'carry' else latest_driver(oj)
    FPV = set(S.HOLD_FP_VERDICTS)
    reached = set()
    for r in csv.DictReader(open(os.path.join(new, 'judgements', 'carry_report.csv'))):
        for x in r['old_sources'].split(';'):
            if x:
                reached.add(x.split(':')[0])
    notfp, n = [], 0
    with open(os.path.join(new, 'judgements', 'carry_missed.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['old_source', 'judge_confidence', 'judge_fp_reason', 'class', 'judge_note'])
        for s, r in sorted(J.items()):
            if s in reached:
                continue
            c = r.get('judge_confidence', '')
            cl = 'FP verdict' if c in FPV else 'NOT an FP verdict'
            if c not in FPV:
                notfp.append('%s (%s)' % (s, c))
            w.writerow([s, c, r.get('judge_fp_reason', ''), cl, (r.get('judge_note') or '')[:S.HOLD_NOTE_CHARS]])
            n += 1
    return notfp, n


def sublst_copy_journals(src_run, B):
    """mkdir -p B/J; cp -p src/J/judgements.jsonl B/J/ for every journal (sublst_cumulative.sh:98-100). Returns the copied
    paths [(source, copy)]."""
    out = []
    for J in _S().SUBLST_JOURNALS:
        os.makedirs(os.path.join(B, J), exist_ok=True)
        a = os.path.join(src_run, J, 'judgements.jsonl')
        if os.path.isfile(a):
            b = os.path.join(B, J, 'judgements.jsonl')
            shutil.copy2(a, b)
            out.append((a, b))
    return out


def sublst_swap(U, B, prev):
    """mv U prev (when U exists); mv B U (sublst_cumulative.sh:102-104). Returns prev or None."""
    moved = None
    if os.path.isdir(U):
        os.rename(U, prev)
        moved = prev
    os.rename(B, U)
    return moved


def append_between(P, U):
    """Append to U's journals the lines of P's journals they lack (exact lines, journal order; sublst_cumulative.sh:110-119).
    Returns {journal: n appended}."""
    out = {}
    for J in _S().SUBLST_JOURNALS:
        a, b = os.path.join(P, J, 'judgements.jsonl'), os.path.join(U, J, 'judgements.jsonl')
        if not os.path.exists(a):
            continue
        have = set(open(b).read().splitlines()) if os.path.exists(b) else set()
        add = [l for l in open(a).read().splitlines() if l.strip() and l not in have]
        if add:
            os.makedirs(os.path.dirname(b), exist_ok=True)
            with open(b, 'a') as f:
                f.write('\n'.join(add) + '\n')
        out[J] = len(add)
    return out


def resolve_carry_from(plan, spec):
    """The run to carry judgements from. A path is taken as given. 'served' = the run the tier's first page serves: in a
    production plan the last served_page record of that page, else the --run of the review server on its port (exact pid,
    snapshot_rechain_batch.sh:369); in a parity plan the production server on the page's PRODUCTION port (the parity
    records describe parity runs). Refuses (G18) when nothing is served or the source is the run itself."""
    if spec != 'served':
        old = os.path.abspath(spec)
    else:
        first = next(pg for pg in plan.all_pages if pg.family == plan.family)
        old = None
        if not plan.parity and os.path.exists(first.record_path()):
            with open(first.record_path()) as f:
                old = json.load(f).get('run')
        if not old:
            old = served_run_of_port(first.port_cfg)
        if not old:
            raise Refusal(f'G18 carry preconditions: no run served on {first.port_cfg} to carry judgements from - servers '
                          f'untouched (snapshot_rechain_batch.sh:370)')
    if not os.path.isdir(old):
        raise Refusal(f'G18 carry preconditions: the carry source {old} is not a directory')
    if plan.U and os.path.realpath(old) == os.path.realpath(plan.U):
        raise Refusal(f'G18 carry preconditions: the carry source is the run being served ({old})')
    return old
