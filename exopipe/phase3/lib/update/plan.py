"""`exopipe3 run update` (PR-8; spec plan/specs/hybrid.md 3.7, 2.16 item 9): a partial update of the hybrid route as ONE plan.

Input:  trigger cells <LST>_<scope> (the masters that change: a night newly excluded, a master rebuilt), given (--trigger) or
        found (--from-exclusions, from_exclusions: every installed master holding a night data_decisions.csv excludes now), and
        the standing work runs of the last full build (--hybrid-out, --v3-out, --v5-out: every step runs with --resume there, so a unit whose
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
        A planned file the update leaves byte-identical is kept at the push (keep = byte-identical only). Each such row names
        its quarantine destination (the push class's quarantine Run of the plan's round: every push entry runs with
        --round <the plan's round>, so the destination planned is the one the push uses). A planned hybrid tile unit with no
        installed filled master of a kind (plain, pbcorr) gets a 'publish' row, named by the tilefill step's dry-run plan
        over the work run when it can name it (else the cell directory and the dry run's reason).
Output: plan.json and plan.csv (one row per entry, item and command; one row per predicted push file; with --from-exclusions
        trigger.csv: each trigger cell's master, night and audit verdict) under the plan
        directory; nothing else is written. `run update` without --plan creates every job record first (a refusal starts
        none), each --after the one before with after_ok (a failed or refused item upstream refuses every later job), all
        from a read-only worktree of one commit (--code, default the clean HEAD), then starts them. Without --apply it stops
        before install, the first entry that writes the installed tree or the archive (every later entry reads what install
        publishes); with --apply every entry runs, each push class still subject to the cutover allowance.
"""
import csv
import json
import os
import re

from exopipe.phase3.core.runner import ITEM_CELL_RE as CELL_RE
KINDS = ('cells', 'cells:sub', 'cells:wb', 'cells:full', 'lsts', 'v3', 'v5', 'tiles', 'deep')
OUTS = ('hybrid_out', 'v3_out', 'v5_out')
PER_SCOPE_TABLE = re.compile(r'^caveats_.+_(?P<scope>[^_]+)_lsfill\.csv$')
W_TOKEN = {'R075': 'Robust-0.75', 'R0': 'Robust+0.0'}
# an installed file: the push replaces it (the old file to the push round's quarantine, D2) or keeps it when the new one is
# byte-identical; which of the two is known only after the steps ran. A kind a planned unit makes with no installed file: publish.
ACTION = 'replace or keep'
PUBLISH = 'publish'
QUARANTINE = ('hybrid', 'mosaic_v5', 'mosaic_v3')   # push classes with a quarantine_dir in lib/archive/<class>.py


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


def command(cmd, outs, apply=False, rnd=None):
    """(argv or None, chain or None, forward) of an UPDATE_ORDER command: 'chain:NAME' or a step command with the work runs
    filled in; every step gets --resume (a chain: forwarded into each of its steps), a push --round rnd (the plan's round; a
    template with its own --round refuses, so the planned quarantine destinations are the push's) and --apply only when asked."""
    if cmd.startswith('chain:'):
        return None, cmd[6:], ['--resume']
    argv = cmd.split()
    for k in OUTS:
        argv = [a.replace('{' + k + '}', outs[k]) for a in argv]
    left = sorted({m for a in argv for m in re.findall(r'\{(\w+)\}', a)} - {'item', 'item_lst', 'item_scope'})
    if left:
        raise ValueError(f'runner.UPDATE_ORDER {cmd!r}: unknown placeholder(s) {left}')
    if argv[:2] == ['archive', 'push']:
        if '--round' in argv:
            raise ValueError(f'runner.UPDATE_ORDER {cmd!r}: a push takes the plan\'s round (run update --round), never its own')
        return argv + (['--round', rnd] if rnd else []) + (['--apply'] if apply else []), None, []
    return argv + ['--resume'], None, []


def writes_archive(argv):
    """True for a step command that writes the installed tree or the archive: install, archive push."""
    return bool(argv) and (argv[0] == 'install' or argv[:2] == ['archive', 'push'])


def check_outs(outs):
    """ValueError unless every standing work run (OUTS) is given."""
    miss = [k for k in OUTS if not outs.get(k)]
    if miss:
        raise ValueError(f'run update needs the standing work runs: --{", --".join(k.replace("_", "-") for k in miss)}')


def build(trigger, outs, apply=False, rnd=None):
    """The plan dict: trigger, closure cells, round, entries [{name, kind, items, argv | chain, forward}] (an entry without items
    is listed with skipped=reason). rnd: the quarantine round of every push (default the UTC date now). ValueError on a bad
    trigger, a missing work run or a bad UPDATE_ORDER entry."""
    import time
    check_outs(outs)
    cells = closure(trigger)
    rnd = rnd or time.strftime('%Y%m%d', time.gmtime())
    try:
        ok = bool(re.fullmatch(r'\d{8}', rnd)) and bool(time.strptime(rnd, '%Y%m%d'))
    except ValueError:
        ok = False
    if not ok:
        raise ValueError(f'the quarantine round must be a date YYYYMMDD, got {rnd!r}')
    entries, gated = [], False
    for name, kind, cmd in _cfg().runner.UPDATE_ORDER:
        its = items_for(kind, cells)
        argv, chain, fwd = command(cmd, outs, apply, rnd)
        gated = gated or writes_archive(argv)
        e = dict(name=name, kind=kind, items=its, argv=argv, chain=chain, forward=fwd, needs_apply=gated)
        if not its:
            e['skipped'] = 'no item'
        elif gated and not apply:
            e['skipped'] = 'needs --apply (this entry or one before it writes the installed tree or the archive)'
        entries.append(e)
    return dict(trigger=list(trigger), cells=[f'{lst}_{s}' for lst, s in cells], outs=dict(outs), apply=bool(apply), round=rnd,
                entries=entries)


def from_exclusions(workers=None):
    """--from-exclusions (spec 3.7): (trigger cells, evidence rows, masters read, exclusions fingerprint). A trigger cell is a
    deep cell whose installed working-tree master (status_conformance.WORKING_MASTERS: Stokes I and V) holds a 'COADD
    constituent' night that data_decisions.csv read now excludes for the cell (core.exclusions, the band rule): the
    master-working part of `status --exclusion-conformance` (lib/status/steps_exclconf.working_tree_audit), every verdict that
    names a night (STALE: excluded after the master was written; BREACH: already excluded then; UNKNOWN-EPOCH: no
    data_decisions backup that old; `status --stale master` lists the STALE ones only). A master the audit cannot read (not
    FITS, a header without END, no constituent line) refuses (ValueError): the trigger never under-selects. The masters root
    is node-local (paths.COADD_SB_ROOT on lwacalim07; need_inputs refuses elsewhere): 912 headers in ~2 s on 2026-11-19.
    The plan rebuilds these masters first (UPDATE_ORDER 'masters'). Only this direction triggers: a night admitted now that
    no build record names (a lifted exclusion, a new night: `status --stale master_coverage`) is growth, taken in by the
    admission rule (D-ADMIT) or by a rebuild `status --rebuild-due` calls due, never by an update."""
    from exopipe.phase3.core.record import Refusal
    from exopipe.phase3.lib.status import steps_common as SK, steps_coverage as COV, steps_exclconf as XC
    C = _cfg()
    try:
        SK.need_inputs([C.paths.COADD_SB_ROOT], 'run update --from-exclusions')
        ex = SK.exclusions()
    except Refusal as e:
        raise ValueError(str(e))
    cells = COV.deep_universe(None)
    n = sum(os.path.exists(os.path.join(C.paths.COADD_SB_ROOT, c, 'coadd_final', pat.format(scope=c.split('_', 1)[1])))
            for c in cells for pat in C.status_conformance.WORKING_MASTERS)
    rows = XC.working_tree_audit(cells, ex, workers=workers)
    bad = [r for r in rows if r['verdict'] == 'UNREADABLE']
    if bad:
        raise ValueError(f'--from-exclusions: {len(bad)} installed master(s) unreadable, e.g. {bad[0]["file"]} ({bad[0]["note"]}) - '
                         f'a trigger that cannot read every master could under-select; REFUSING')
    ev = [dict(cell=f'{r["lst"]}_{r["band"]}', master=r['file'], date=r['date'], verdict=r['verdict'], note=r['note'])
          for r in rows]
    return sorted({e['cell'] for e in ev}), ev, n, ex.fingerprint()


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
    per-scope table (PER_SCOPE_TABLE: caveats_<V>_<scope>_lsfill.csv, the only per-scope tables either run writes) only of an
    updated scope; never a carried table (mosaic_v3.TABLES_CARRIED: copied unchanged). A name the pattern does not match counts
    as run-wide (over-predicting is safe; the push keeps a byte-identical file)."""
    C = _cfg()
    carried = set(C.mosaic_v3.TABLES_CARRIED) if cls == 'mosaic_v3' else set()
    out = []
    for p in _files(d):
        n = os.path.basename(p)
        m = PER_SCOPE_TABLE.match(n)
        if n not in carried and (not m or m['scope'] in scopes):
            out.append(p)
    return out


def quarantine_dst(cls, root, rnd, path, run):
    """Where push --class cls of round rnd moves the installed file `path` when it replaces it: the class's quarantine Run (its
    lib/archive module's quarantine_dir) + the path relative to the class's archive run `run` (lib/archive/hybrid.apply_unit's
    qpath: the hybrid run's <LST>/<scope>/<name>, a mosaic run's <scope>/I/<sub>/<name> or tables/<name>)."""
    import importlib
    import types
    mod = importlib.import_module(f'exopipe.phase3.lib.archive.{cls}')
    return os.path.join(mod.quarantine_dir(types.SimpleNamespace(archive_root=root), rnd), os.path.relpath(path, run))


def publish_names(hybrid_out, lst, scope, w, kinds, rebuilt):
    """({kind: product name}, note): the names the tilefill step's dry-run plan (steps.hybrid.tilefill_source, source 'work',
    plan_only) gives the unit's filled masters of `kinds` in the work run now. A unit the plan rebuilds is not named: its
    name (night span, N) comes from the rebuild, and the work run holds the record from before it ({}, note). The dry run's
    own refusals (and a missing manifest or record) become the note; any other error is a bug and propagates."""
    from exopipe.phase3.core.record import NoInput, Refusal
    from exopipe.phase3.lib.hybrid.tilefill import TilefillRefusal
    from exopipe.phase3.steps import hybrid as SH
    if rebuilt:
        return {}, 'named by the step after the rebuild (the plan rebuilds this unit: its night span and N may change)'
    try:
        s = SH.tilefill_source(hybrid_out, lst, scope, w, SH._manifest_rows(_cfg().hybrid_tilefill), 'work', plan_only=True,
                               need_pb=False)
    except (TilefillRefusal, Refusal, NoInput, OSError) as e:
        return {}, f'named by the step (dry run: {type(e).__name__}: {str(e)[:160]})'
    return {k: s['src'][k][1] for k in kinds}, ''


def predicted(plan, archive_root=None):
    """[{entry, unit, kind, dst, action, quarantine, note}]: the installed files each push entry of the plan may replace, each
    with its quarantine destination, and the filled masters a planned hybrid unit publishes (module docstring)."""
    C = _cfg()
    root = archive_root or C.paths.PHASE3_ARCHIVE_ROOT
    rnd = plan['round']

    def under(run):
        return os.path.join(root, os.path.relpath(run, C.paths.PHASE3_ARCHIVE_ROOT))
    hyb = {tuple(c.split('_', 1)) for c in plan['cells']}
    rows = []

    def items_of(kind):
        e = next((x for x in plan['entries'] if x['kind'] == kind), None)
        if e is None:
            raise ValueError(f'runner.UPDATE_ORDER has no entry of item kind {kind!r}, which the push predictions need')
        return e['items']
    for e in plan['entries']:
        if e.get('skipped') or not e['argv'] or e['argv'][:2] != ['archive', 'push']:
            continue
        cls = e['argv'][e['argv'].index('--class') + 1]
        if cls not in QUARANTINE:
            raise ValueError(f'runner.UPDATE_ORDER {e["name"]}: no file prediction for push class {cls!r}')

        def row(unit, kind, p, run, action=ACTION, note=''):
            q = quarantine_dst(cls, root, rnd, p, run) if action == ACTION else ''
            return dict(entry=e['name'], unit=unit, kind=kind, dst=p, action=action, quarantine=q, note=note)
        if cls == 'hybrid':
            for it in items_of('tiles'):
                lst, scope = it.split('_', 1)
                hrun = under(C.hybrid.ARCHIVE_HYBRID)
                cd = os.path.join(hrun, lst, scope)
                for w in C.hybrid.WEIGHTINGS:
                    got = hybrid_unit_files(cd, scope, w, (lst, scope) in hyb)
                    rows += [row(f'{lst}_{scope}_{w}', kind, p, hrun) for kind, p in got]
                    miss = [k for k in ('plain', 'pbcorr') if k not in {kind for kind, _p in got}]
                    if miss:
                        names, why = publish_names(plan['outs']['hybrid_out'], lst, scope, w, miss, (lst, scope) in hyb)
                        rows += [row(f'{lst}_{scope}_{w}', k, os.path.join(cd, names[k]) if k in names else cd, hrun,
                                     action=PUBLISH, note=why) for k in miss]
        else:
            run = under(C.hybrid_tilefill.FILLED_MOSAIC_RUN if cls == 'mosaic_v5' else C.mosaic_v3.ARCHIVE_RUN)
            scopes = items_of({'mosaic_v5': 'v5', 'mosaic_v3': 'v3'}[cls])
            for s in scopes:
                rows += [row(s, 'product', p, run) for p in _walk(os.path.join(run, s))]
            rows += [row('tables', 'table', p, run) for p in tables_touched(os.path.join(run, 'tables'), cls, scopes)]
    return rows


def write(plan, rows, d):
    """plan.json and plan.csv under d (made if missing): the entries, their items and commands, and the predicted push files."""
    import shlex
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'plan.json'), 'w') as f:
        json.dump(dict(plan, predicted=rows), f, indent=1)
    with open(os.path.join(d, 'plan.csv'), 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(('order', 'entry', 'item', 'command', 'kind', 'action', 'dst', 'quarantine', 'note'))
        for i, e in enumerate(plan['entries']):
            cmd = f'chain:{e["chain"]} {" ".join(e["forward"])}' if e['chain'] else shlex.join(e['argv'])
            for it in e['items'] or ['']:
                w.writerow((i, e['name'], it, e.get('skipped') or cmd, 'run', '', '', '', ''))
        for r in rows:
            w.writerow(('', r['entry'], r['unit'], '', r['kind'], r['action'], r['dst'], r['quarantine'], r['note']))
    if plan.get('trigger_evidence'):          # --from-exclusions: why each trigger cell is one
        with open(os.path.join(d, 'trigger.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, ('cell', 'master', 'date', 'verdict', 'note'), lineterminator='\n')
            w.writeheader()
            w.writerows(plan['trigger_evidence'])
    return os.path.join(d, 'plan.json')


def submit(plan, nodes, name_prefix, code):
    """Create the plan's job records in order, each --after the one before with after_ok (a job whose upstream job ended
    with a failed or refused item refuses every item, so a failure stops the rest of the plan), all from the read-only
    worktree of `code`; nothing starts here (start(): after every record exists and the plan is written).
    [(entry, job id)]."""
    from exopipe.phase3.core import runner
    jobs, prev = [], None
    for e in plan['entries']:
        if e.get('skipped'):
            continue
        job = runner.submit(f'{name_prefix}_{e["name"]}', e['argv'] or [], e['items'], nodes, after=prev, chain=e['chain'],
                            forward=e['forward'], code=code, start=False, after_ok=prev is not None)
        jobs.append((e['name'], job))
        prev = job
    return jobs


def start(jobs):
    """Start the supervisors of the created jobs (each waits for its --after job)."""
    from exopipe.phase3.core import runner
    for _name, job in jobs:
        runner.start_supervisor(job)
