"""`exopipe3 run update` (PR-8; spec plan/specs/hybrid.md 3.7, 2.16 item 9): a partial update of the hybrid route as ONE plan.

Input:  trigger cells <LST>_<scope> (the masters that change: a night newly excluded, a master rebuilt) and the standing work
        runs of the last full build (--hybrid-out, --v3-out, --v5-out: every step runs with --resume there, so a unit whose
        recorded inputs, admitted nights, code and config are unchanged is skipped, never rebuilt; D5).
Plan:   runner.UPDATE_ORDER (config) entries in order, each a step command or a runner chain over its items:
          cells       the trigger cells and, for a subband cell, the cells of the bands holding it (its wideband, Full),
                      subbands first, then widebands, then Full; cells:sub / cells:wb / cells:full one class only;
          lsts        the cells' LSTs;
          v3 / v5     the cells' scopes that are mosaic_v3 / mosaic_v5 scopes, in config order;
          tiles       every LST of each v5 scope (<LST>_<scope>: every tile's fill comes from the new mosaic);
          deep        one item, 'deep' (a run-wide unit: tables, a push; the command takes it as --tier {item}).
        {hybrid_out} {v3_out} {v5_out} are the work runs; {lst} {scope} the parts of a <LST>_<scope> item.
        For each push entry the plan lists the installed files the push may replace (predicted from the archive now):
          push_hybrid     per tile unit (LST, scope, weighting): its filled plain and pbcorr masters; for a unit the hybrid
                          entries rebuild (source 'work'): also its CLEAN residual, NoTaper reconstructions (not Full), the
                          cell's diffuse PB map (hybrid_full.PBDIFFUSE_WEIGHTING unit) and its weighting's deep PB map;
          push_mosaic_v5  every installed file of each v5 scope in the hybrid mosaic run, and the tables its tables unit may
                          rewrite (tables_touched: run-wide tables, the per-scope tables of the updated scopes);
          push_mosaic_v3  the same in the FEATHER3 run (its carried tables aside).
        A planned file the update leaves byte-identical is kept at the push (keep = byte-identical only).
Output: plan.json and plan.csv (one row per entry, item and command; one row per predicted push file) under the plan
        directory; nothing else is written. `run update` without --plan submits the entries as runner jobs, each --after the
        one before; the pushes stay plan-only unless --apply (and the class is in the cutover allowance).
"""
import csv
import json
import os
import re

CELL_RE = re.compile(r'^(?P<lst>\d\dh)_(?P<scope>\S+)$')
KINDS = ('cells', 'cells:sub', 'cells:wb', 'cells:full', 'lsts', 'v3', 'v5', 'tiles', 'deep')
OUTS = ('hybrid_out', 'v3_out', 'v5_out')
W_TOKEN = {'R075': 'Robust-0.75', 'R0': 'Robust+0.0'}
# every predicted file is installed now: the push replaces it (the old file to the push round's quarantine, D2) or keeps it when
# the new one is byte-identical; which of the two is known only after the steps ran (PR-8b: publish names from the dry runs)
ACTION = 'replace or keep'


def _cfg():
    from exopipe.phase3.config import CFG
    return CFG


def scope_class(scope):
    """'sub', 'wb' or 'full' of a hybrid scope."""
    if scope == 'Full':
        return 'full'
    return 'wb' if scope in _cfg().hybrid.WIDEBAND_MEMBERS else 'sub'


def closure(trigger):
    """[(LST, scope)]: the trigger cells and, for a subband cell, the cells of its wideband and Full; ordered subbands, widebands,
    Full (each class's units read the class before it), then by LST and the config scope order. ValueError on a bad cell."""
    from exopipe.phase3.core.cells import LSTS
    C = _cfg().hybrid
    out, bad = set(), []
    for c in trigger:
        m = CELL_RE.match(c)
        if not m or m['lst'] not in LSTS or m['scope'] not in C.SCOPES:
            bad.append(c)
            continue
        lst, scope = m['lst'], m['scope']
        out.add((lst, scope))
        if scope_class(scope) == 'sub':
            out |= {(lst, wb) for wb, mem in C.WIDEBAND_MEMBERS.items() if scope in mem} | {(lst, 'Full')}
    if bad or not trigger:
        raise ValueError(f'--trigger: {bad or "no cell"} - cells are <LST>_<scope> with a hybrid scope ({", ".join(C.SCOPES)})')
    order = {'sub': 0, 'wb': 1, 'full': 2}
    return sorted(out, key=lambda c: (order[scope_class(c[1])], LSTS.index(c[0]), C.SCOPES.index(c[1])))


def items_for(kind, cells):
    """The items of an UPDATE_ORDER entry of item kind `kind` over the closure cells."""
    from exopipe.phase3.core.cells import LSTS
    C = _cfg()
    scopes = {s for _l, s in cells}
    if kind == 'cells':
        return [f'{lst}_{s}' for lst, s in cells]
    if kind.startswith('cells:'):
        return [f'{lst}_{s}' for lst, s in cells if scope_class(s) == kind[6:]]
    if kind == 'lsts':
        return sorted({lst for lst, _s in cells}, key=LSTS.index)
    if kind == 'v3':
        return [s for s in C.mosaic_v3.SCOPES if s in scopes]
    if kind in ('v5', 'tiles'):
        v5 = [s for s in C.mosaic_v5.SCOPES if s in scopes]
        return v5 if kind == 'v5' else [f'{lst}_{s}' for s in v5 for lst in LSTS]
    if kind == 'deep':
        return ['deep']
    raise ValueError(f'runner.UPDATE_ORDER: unknown item kind {kind!r} ({", ".join(KINDS)})')


def command(cmd, outs, apply=False):
    """(argv or None, chain or None, forward) of an UPDATE_ORDER command: 'chain:NAME' or a step command with the work runs
    filled in; every step gets --resume (a chain: forwarded into each of its steps), a push --apply only when asked."""
    if cmd.startswith('chain:'):
        return None, cmd[6:], ['--resume']
    argv = cmd.split()
    for k in OUTS:
        argv = [a.replace('{' + k + '}', outs[k]) for a in argv]
    left = sorted({m for a in argv for m in re.findall(r'\{(\w+)\}', a)} - {'item', 'lst', 'scope'})
    if left:
        raise ValueError(f'runner.UPDATE_ORDER {cmd!r}: unknown placeholder(s) {left}')
    if argv[:2] == ['archive', 'push']:
        return argv + (['--apply'] if apply else []), None, []
    return argv + ['--resume'], None, []


def build(trigger, outs, apply=False):
    """The plan dict: trigger, closure cells, entries [{name, kind, items, argv | chain, forward}] (an entry without items is
    listed with skipped=reason). ValueError on a bad trigger, a missing work run or a bad UPDATE_ORDER entry."""
    miss = [k for k in OUTS if not outs.get(k)]
    if miss:
        raise ValueError(f'run update needs the standing work runs: --{", --".join(k.replace("_", "-") for k in miss)}')
    cells = closure(trigger)
    entries = []
    for name, kind, cmd in _cfg().runner.UPDATE_ORDER:
        its = items_for(kind, cells)
        argv, chain, fwd = command(cmd, outs, apply)
        e = dict(name=name, kind=kind, items=its, argv=argv, chain=chain, forward=fwd)
        if not its:
            e['skipped'] = 'no item'
        entries.append(e)
    return dict(trigger=list(trigger), cells=[f'{lst}_{s}' for lst, s in cells], outs=dict(outs), apply=bool(apply), entries=entries)


def _files(d):
    return sorted(os.path.join(d, n) for n in os.listdir(d) if os.path.isfile(os.path.join(d, n))) if os.path.isdir(d) else []


def _walk(d):
    out = []
    for dp, dn, fn in os.walk(d):
        dn.sort()
        out += [os.path.join(dp, f) for f in sorted(fn)]
    return out


def hybrid_unit_files(cell_dir, scope, w, rebuilt):
    """[(kind, path)] of the installed files of a hybrid cell directory a push of unit (scope, w) may replace (module docstring)."""
    from exopipe.phase3.lib.masters import pb_apply as PBA
    tok = f'{scope}_I_deep_Taper_{W_TOKEN[w]}_'
    ntok = f'{scope}_I_deep_NoTaper_{W_TOKEN[w]}_'
    out = []
    for p in _files(cell_dir):
        n = os.path.basename(p)
        if n.startswith(tok) and '_lsfill' in n and 'cleanres' not in n:
            out.append(('pbcorr' if '_pbcorr_' in n else 'plain', p))
        elif not rebuilt:
            continue
        elif n.startswith(tok) and '_cleanres' in n:
            out.append(('cleanres', p))
        elif scope != 'Full' and n.startswith(ntok) and '_lsrecon' in n:
            out.append(('recon_pbcorr' if '_pbcorr_' in n else 'recon', p))
        elif n == f'PBdiffuse_deep_{scope}.fits' and w == _cfg().hybrid_full.PBDIFFUSE_WEIGHTING:
            out.append(('pbdiffuse', p))
        elif n == PBA.deep_pb_name(scope, w):
            out.append(('pb', p))
    return out


def tables_touched(d, cls, scopes):
    """The installed tables of a mosaic run its tables unit may rewrite for an update of `scopes`: every run-wide table, and a
    per-scope table (a scope token in its name: caveats_<V>_<scope>_lsfill.csv) only of an updated scope; never a carried table
    (mosaic_v3.TABLES_CARRIED: copied unchanged)."""
    C = _cfg()
    every = set(C.mosaic_v5.SCOPES) | set(C.mosaic_v3.SCOPES)
    carried = set(C.mosaic_v3.TABLES_CARRIED) if cls == 'mosaic_v3' else set()
    out = []
    for p in _files(d):
        n = os.path.basename(p)
        sc = set(re.split(r'[_.]', n)) & every
        if n not in carried and (not sc or sc & set(scopes)):
            out.append(p)
    return out


def predicted(plan, archive_root=None):
    """[{entry, unit, kind, dst}]: the installed files each push entry of the plan may replace (module docstring)."""
    C = _cfg()
    root = archive_root or C.paths.PHASE3_ARCHIVE_ROOT

    def under(run):
        return os.path.join(root, os.path.relpath(run, C.paths.PHASE3_ARCHIVE_ROOT))
    hyb = {tuple(c.split('_', 1)) for c in plan['cells']}
    rows = []
    for e in plan['entries']:
        if e.get('skipped') or not e['argv'] or e['argv'][:2] != ['archive', 'push']:
            continue
        cls = e['argv'][e['argv'].index('--class') + 1]
        if cls == 'hybrid':
            tiles = next(x['items'] for x in plan['entries'] if x['kind'] == 'tiles')
            for it in tiles:
                lst, scope = it.split('_', 1)
                for w in C.hybrid.WEIGHTINGS:
                    for kind, p in hybrid_unit_files(os.path.join(under(C.hybrid.ARCHIVE_HYBRID), lst, scope), scope, w,
                                                     (lst, scope) in hyb):
                        rows.append(dict(entry=e['name'], unit=f'{lst}_{scope}_{w}', kind=kind, dst=p, action=ACTION))
        elif cls in ('mosaic_v5', 'mosaic_v3'):
            run = under(C.hybrid_tilefill.FILLED_MOSAIC_RUN if cls == 'mosaic_v5' else C.mosaic_v3.ARCHIVE_RUN)
            scopes = next(x['items'] for x in plan['entries'] if x['kind'] == cls[-2:])
            for s in scopes:
                rows += [dict(entry=e['name'], unit=s, kind='product', dst=p, action=ACTION) for p in _walk(os.path.join(run, s))]
            rows += [dict(entry=e['name'], unit='tables', kind='table', dst=p, action=ACTION)
                     for p in tables_touched(os.path.join(run, 'tables'), cls, scopes)]
        else:
            raise ValueError(f'runner.UPDATE_ORDER {e["name"]}: no file prediction for push class {cls!r}')
    return rows


def write(plan, rows, d):
    """plan.json and plan.csv under d (made if missing): the entries, their items and commands, and the predicted push files."""
    import shlex
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'plan.json'), 'w') as f:
        json.dump(dict(plan, predicted=rows), f, indent=1)
    with open(os.path.join(d, 'plan.csv'), 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(('order', 'entry', 'item', 'command', 'kind', 'action', 'dst'))
        for i, e in enumerate(plan['entries']):
            cmd = f'chain:{e["chain"]} {" ".join(e["forward"])}' if e['chain'] else shlex.join(e['argv'])
            for it in e['items'] or ['']:
                w.writerow((i, e['name'], it, e.get('skipped') or cmd, 'run', '', ''))
        for r in rows:
            w.writerow(('', r['entry'], r['unit'], '', r['kind'], r['action'], r['dst']))
    return os.path.join(d, 'plan.json')


def submit(plan, nodes, name_prefix):
    """Submit the plan's entries as runner jobs in order, each --after the one before: [(entry, job id)]."""
    from exopipe.phase3.core import runner
    jobs, prev = [], None
    for e in plan['entries']:
        if e.get('skipped'):
            continue
        job = runner.submit(f'{name_prefix}_{e["name"]}', e['argv'] or [], e['items'], nodes, after=prev, chain=e['chain'],
                            forward=e['forward'])
        jobs.append((e['name'], job))
        prev = job
    return jobs
