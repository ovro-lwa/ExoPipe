"""`exopipe3 jupiter flag-backfill` - the Phase-1 Jupiter check of Stokes-V-flagged integrations on archived nights
(exopipe/phase1/jupiter_flag_backfill.py with jupiter_flag_qa.py, production package, run as today's subprocess).

Ported from exopipe_ops/jupiter_flag_backfill_launch.sh (logic items 1-11 of spec 7.1):
  window  `TOOL window --window-csv STATE/window.csv` unless window.csv exists (kept, as the launcher); --cells writes the
          explicit list instead (the old JFQ_CELLS: lst,date and a blank jup_el_mid).
  run     ONE shard per invocation - the shard of --shard NODE (the runner item) or of this host: window row i goes to node
          i mod n (the launcher's awk 'NR==1 || (NR-2) % m == k'), written to STATE/shard_<node>.csv; per-shard copy budget =
          int(LIMIT_GB_TOTAL // n) GB (bash integer division); `TOOL run --window-csv STATE/shard_<node>.csv --dest-root DEST
          --out STATE/summary_<node>.csv --nproc NPROC --max-copy-gb B` with OMP/OPENBLAS/MKL = 1, PYTHONPATH = the production
          exopipe, cwd STATE, log STATE/run_<node>.log (the previous log kept with a UTC suffix; 'Exit status: <rc>' appended so
          the old launcher's --status/--launch gates still read it). Many nodes: `exopipe3 run submit --nodes n1:1,n2:1,... --
          jupiter flag-backfill --stages run --shard {item} --items n1,n2,...`.
  verify  `TOOL verify --dest-root DEST --nproc NPROC` (OMP 1), warnings filtered, written to STATE/VERIFY_<UTC>.txt; exit != 0
          refuses (the backfill's "Complete when", ledger 1993).
Gates: J30 the tool and QA module are the tested versions (sha256 pins); J31 STATE/APPROVED present and non-empty for a copying
run (its text goes into the record; a --no-copy run copies nothing and needs none); J32 a previous shard log without 'Exit
status' refuses (a shard may still run); J33 the tool's copy-budget refusal (exit 2) refuses that shard; J34 MAX_HOURS.
STATE = jupiter_flag.STATE_DIR for a production --out, <out>/jupiter_flag_backfill_state otherwise; DEST = <out>/phase1_flagged.
"""
import csv
import os
import re
import sys
import tempfile
import time

from exopipe.phase3.core.record import NoInput, Refusal
from exopipe.phase3.lib.jupiter import common as K
from exopipe.phase3.lib.jupiter import mirrors


def _C():
    from exopipe.phase3.config import CFG
    return CFG


def code_gate():
    """J30: the tool and its QA module are the tested versions."""
    F = _C().jupiter_flag
    bad = [p for p, h in ((F.TOOL, F.TOOL_SHA256), (F.QA_MODULE, F.QA_SHA256)) if not os.path.exists(p) or K.sha256(p) != h]
    if bad:
        raise Refusal(f'J30 tested code only: {", ".join(bad)} is not the tested version (sha256 pin, '
                      f'APPROVED-PREAMBLES-1-4-20261007)')


def nodes(args):
    n = [x.strip() for x in (args.nodes or '').split(',') if x.strip()] or list(getattr(_C().site, 'JUPITER_FLAG_NODES', []) or [])
    if not n:
        raise Refusal('no backfill nodes: pass --nodes n1,n2,... or set JUPITER_FLAG_NODES in the site file '
                      '(the launcher default was lwacalim01, 02, 04, 05)')
    return [x.split(':')[0] for x in n]


def shard_rows(window_csv, node_list, node):
    with open(window_csv, newline='') as f:
        rows = list(csv.reader(f))
    head, body = rows[0], rows[1:]
    k = node_list.index(node)
    return head, [r for i, r in enumerate(body) if i % len(node_list) == k]


def budget_gb(args, n_nodes):
    total = args.budget_gb if args.budget_gb is not None else _C().jupiter_flag.LIMIT_GB_TOTAL
    return int(total) // n_nodes


def _env(threads=True):
    F = _C().jupiter_flag
    env = dict(_C().jupiter.ENV)
    if threads:
        env.update(F.THREAD_ENV)
        env['PYTHONPATH'] = _C().jupiter.PRODUCTION_EXOPIPE_ROOT
    else:
        env['OMP_NUM_THREADS'] = F.THREAD_ENV['OMP_NUM_THREADS']
    return env


def run(args, run_, roots, stages):
    F = _C().jupiter_flag
    code_gate()
    mirrors.check_literals(mirrors.ACTION_MODULES['flag-backfill'])
    state, dest = roots.flag_state, roots.flagged
    verify_root = os.path.abspath(args.flagged_root) if args.flagged_root else dest
    wcsv = os.path.join(state, 'window.csv')
    if run_.dry_run:
        return dry_run(args, run_, roots, stages, state, dest, verify_root, wcsv)
    os.makedirs(state, exist_ok=True)
    if 'window' in stages:
        stage_window(args, run_, roots, state, wcsv)
    if 'run' in stages:
        stage_run(args, run_, roots, state, dest, wcsv)
    if 'verify' in stages:
        stage_verify(args, run_, roots, state, verify_root)
    return None


def _cells_list(args):
    from exopipe.phase3.core import cells
    items = cells.expand(args.cells, kind='Jupiter cell')
    for c in items:
        cells.parse_epoch(c)
    return items


def stage_window(args, run_, roots, state, wcsv):
    from exopipe.phase3.core import validate
    if args.cells:
        with run_.unit('flag.window', action='flag-backfill', stage='window') as u:
            items = _cells_list(args)
            tmp = wcsv + '.part'
            with open(tmp, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['lst', 'date', 'jup_el_mid'])
                for c in items:
                    lst, date = c.split('_', 1)
                    w.writerow([lst, date, ''])
            os.replace(tmp, wcsv)
            u.note(f'explicit cell list ({len(items)} cells, the old JFQ_CELLS)')
            u.add_output(wcsv, 'jupiter_window')
        return
    if os.path.exists(wcsv) and os.path.getsize(wcsv) > 0:
        res = validate.run('jupiter_window', wcsv)
        u = run_.unit('flag.window', action='flag-backfill', stage='window')
        ok = all(r['passed'] for r in res)
        u.status = 'skipped-valid' if ok else 'refused'
        u.reason = (f'{wcsv} exists (kept, as the launcher) and passes its content test' if ok
                    else f'{wcsv} exists but fails {[r["detail"] for r in res if not r["passed"]]}')
        u.write()
        if not ok:
            raise Refusal(u.reason)
        return
    with run_.unit('flag.window', action='flag-backfill', stage='window') as u:
        from exopipe.phase3.core import legacy
        res = legacy.run(_C().jupiter_flag.TOOL, ['window', '--window-csv', wcsv], python=K.interpreter_key('flag'),
                         env=dict(_C().jupiter.ENV, PYTHONPATH=_C().jupiter.PRODUCTION_EXOPIPE_ROOT),
                         cwd=state, log=None,
                         parity_root=os.pathsep.join(roots.guard) if roots.guard else None, unit=u)
        if not res.ok:
            raise RuntimeError(f'window build failed: {res.reason}')
        u.add_input(_C().paths.DATA_DECISIONS)
        u.add_output(wcsv, 'jupiter_window')


def _rotate_logs(state, node):
    """J32: the previous launch's log must show 'Exit status' (the shard finished); it is kept with a UTC suffix."""
    p = os.path.join(state, f'run_{node}.log')
    if not os.path.exists(p):
        return None
    if 'Exit status' not in open(p, errors='replace').read():
        raise Refusal(f'J32 shard log {p} has no exit line - a shard may still run (check, then move the log)')
    new = p[:-4] + '.' + time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '.log'
    os.replace(p, new)
    return new


def stage_run(args, run_, roots, state, dest, wcsv):
    F = _C().jupiter_flag
    if not os.path.exists(wcsv):
        raise NoInput(f'no {wcsv}: run --stages window first')
    nl = nodes(args)
    node = args.shard or K.host()
    if node not in nl:
        raise Refusal(f'shard {node!r} is not one of the backfill nodes {nl}: pass --shard NODE (run submit item) on that node')
    if args.shard and args.shard != K.host() and not args.no_copy:
        raise Refusal(f'--shard {args.shard} runs here on {K.host()}: a copying shard runs on its own node (submit it with '
                      f'exopipe3 run submit --nodes {args.shard}:1 ... --shard {{item}})')
    with run_.unit(f'flag.run.{node}', action='flag-backfill', stage='run', shard=node, host=K.host()) as u:
        appr = os.path.join(state, F.APPROVAL_FILE)
        if not args.no_copy:
            if not (os.path.exists(appr) and os.path.getsize(appr) > 0):
                u.refuse(f'J31 approval: {appr} missing or empty (the session that received Gregg\'s approval writes his '
                         f'words and the time; APPROVED-PREAMBLES-1-4-20261007)')
            u.meta['approval'] = open(appr).read()[:2000]
            u.add_input(appr)
        else:
            u.note('--no-copy: records only, nothing copied (no approval file needed)')
        old = _rotate_logs(state, node)
        if old:
            u.note(f'previous log kept as {old}')
        head, rows = shard_rows(wcsv, nl, node)
        sp = os.path.join(state, f'shard_{node}.csv')
        with open(sp + '.part', 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(head)
            w.writerows(rows)
        os.replace(sp + '.part', sp)
        if not rows:
            raise NoInput(f'shard {node}: no cells ({len(nl)} nodes)')
        u.add_input(wcsv)
        u.add_input(_C().paths.DATA_DECISIONS)
        gb = budget_gb(args, len(nl))
        hours = args.max_hours if args.max_hours is not None else F.MAX_HOURS
        summ = os.path.join(state, f'summary_{node}.csv')
        argv = ['run', '--window-csv', sp, '--dest-root', dest, '--out', summ, '--nproc', str(F.NPROC), '--max-copy-gb', str(gb)]
        if args.no_copy:
            argv.append('--no-copy')
        u.meta.update(shard_cells=len(rows), budget_gb=gb, max_hours=hours, nodes=nl)
        from exopipe.phase3.core import legacy
        K.pin_free_cores(args)
        log = os.path.join(state, f'run_{node}.log')
        res = legacy.run(F.TOOL, argv, python=K.interpreter_key('flag'), env=_env(), cwd=state, log=log,
                         parity_root=os.pathsep.join(roots.guard) if roots.guard else None, timeout=hours * 3600.0, unit=u,
                         refusal_codes=(2,), refusal_text=('REFUSED',))
        with open(log, 'a') as f:
            f.write(f'\tExit status: {res.rc}\n')
        if res.rc == -9:
            raise RuntimeError(f'J34 shard {node} exceeded {hours:g} h (stopped)')
        if res.rc == 2:
            u.refuse(f'J33 copy budget: shard {node} stopped before the subband-night that would pass {gb} GB ({res.reason})')
        if res.rc != 0:
            raise RuntimeError(f'shard {node}: {res.reason}')
        if os.path.exists(summ):
            u.add_output(summ, 'csv_table', required_columns=('lst', 'date', 'subband'), min_rows=1)
        else:
            raise NoInput(f'shard {node} wrote no summary (no subband-night processed)')
    print(f'jupiter flag-backfill run {node}: {u.status}' + (f' ({u.reason})' if u.reason else ''), file=sys.stderr)


def stage_verify(args, run_, roots, state, verify_root):
    F = _C().jupiter_flag
    with run_.unit('flag.verify', action='flag-backfill', stage='verify', root=verify_root) as u:
        if not os.path.isdir(verify_root):
            raise NoInput(f'no copy tree {verify_root}')
        from exopipe.phase3.core import legacy
        K.pin_free_cores(args)
        fd, raw = tempfile.mkstemp(prefix='jupiter_flag_verify_', suffix='.log')
        os.close(fd)
        try:
            res = legacy.run(F.TOOL, ['verify', '--dest-root', verify_root, '--nproc', str(F.NPROC)], python=K.interpreter_key('flag'),
                             env=_env(threads=False), cwd=state, log=raw,
                             parity_root=os.pathsep.join(roots.guard) if roots.guard else None, unit=u)
            drop = re.compile('|'.join(re.escape(x) for x in F.VERIFY_DROP))
            lines = [l for l in open(raw, errors='replace').read().splitlines()[1:] if not drop.search(l)]
        finally:
            os.remove(raw)
        out = os.path.join(state, 'VERIFY_' + time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '.txt')
        with open(out + '.part', 'w') as f:
            f.write('\n'.join(lines) + '\n')
        os.replace(out + '.part', out)
        last = next((l for l in reversed(lines) if l.startswith('VERIFY ')), '')
        from exopipe.phase3.lib.jupiter.tests import VERIFY_RE
        m = VERIFY_RE.match(last)
        exp = None
        if m:
            exp = dict(dirs=int(m.group(1)), flagged_rows=int(m.group(2)), jupiter_rows=int(m.group(3)), copies=int(m.group(4)),
                       gb=float(m.group(5)))
        u.meta['verify_rc'] = res.rc
        u.add_output(out, 'jupiter_flag_verify', expected=exp)
        if res.rc != 0:
            u.refuse(f'J35 verify exit {res.rc}: {last[:200]} (record {out})')
    print(f'jupiter flag-backfill verify: {u.status}' + (f' ({u.reason})' if u.reason else ''), file=sys.stderr)


def dry_run(args, run_, roots, stages, state, dest, verify_root, wcsv):
    F = _C().jupiter_flag
    print(f'jupiter flag-backfill: stages {",".join(stages)}; {roots.describe()}')
    print(f'  J30 tested code: {F.TOOL} and {F.QA_MODULE} match their sha256 pins')
    print(f'  state {state}; copies to {dest}')
    if 'window' in stages:
        if args.cells:
            print(f'  window: explicit list of {len(_cells_list(args))} cells -> {wcsv}')
        elif os.path.exists(wcsv):
            from exopipe.phase3.core import validate
            r = validate.run('jupiter_window', wcsv)
            print(f'  window: {wcsv} exists - kept ({r[0]["detail"]})')
        else:
            print(f'  window: {F.TOOL} window --window-csv {wcsv}')
    refuse = False
    if 'run' in stages:
        try:
            nl = nodes(args)
        except Refusal as e:
            print(f'  run: REFUSE - {e}')
            nl, refuse = [], True
        if nl:
            n_cells = sum(1 for _ in open(wcsv)) - 1 if os.path.exists(wcsv) else None
            gb = budget_gb(args, len(nl))
            print(f'  run: {n_cells if n_cells is not None else "?"} cells over {len(nl)} nodes ({", ".join(nl)}); per shard --nproc '
                  f'{F.NPROC}, --max-copy-gb {gb}, timeout {(args.max_hours or F.MAX_HOURS):g} h' + (' --no-copy' if args.no_copy else ''))
            if os.path.exists(wcsv):
                for n in nl:
                    _, rows = shard_rows(wcsv, nl, n)
                    log = os.path.join(state, f'run_{n}.log')
                    lstate = ('no previous log' if not os.path.exists(log) else
                              'previous log finished' if 'Exit status' in open(log, errors='replace').read() else
                              'previous log WITHOUT exit line - J32 refuses')
                    refuse |= 'J32' in lstate
                    print(f'    shard {n}: {len(rows)} cells -> {os.path.join(state, "shard_" + n + ".csv")}; {lstate}')
            appr = os.path.join(state, F.APPROVAL_FILE)
            ok = args.no_copy or (os.path.exists(appr) and os.path.getsize(appr) > 0)
            refuse |= not ok
            print(f'  J31 approval {appr}: {"present" if os.path.exists(appr) else "MISSING"}' +
                  (' (not needed: --no-copy)' if args.no_copy else '' if ok else ' - a copying run refuses'))
            node = args.shard or K.host()
            print(f'  this invocation runs the shard of {node}' + ('' if node in nl else ' - NOT a backfill node: refuses'))
            print(f'  many nodes: exopipe3 run submit --nodes {",".join(n + ":1" for n in nl)} --items {",".join(nl)} -- '
                  f'jupiter flag-backfill --stages run --shard {{item}}')
    if 'verify' in stages:
        print(f'  verify: {F.TOOL} verify --dest-root {verify_root} --nproc {F.NPROC} -> {state}/VERIFY_<UTC>.txt '
              f'[jupiter_flag_verify]')
    return 2 if refuse else 0
