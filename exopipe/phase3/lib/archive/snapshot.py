"""Snapshot-tier archive actions, both running the production scripts IN PLACE (spec 7.2 items 3-4).

supersede        ops/snapshot_frametime_ship.sh CELL WORKDIR SHIP_ROOT PHOT_ROOT QUAR_ROOT (snapshot_frametime_rerun.sh step 6,
                 :236-238): every superseded file of the cell (1-minute files, runinfo, wideband files and logs) and its
                 wideband photometry cutouts are renamed into <QUAR_ROOT>/<cell>/<UTC>/ with their sha256 in MANIFEST.csv
                 BEFORE the move; each new file is copied beside the live one, sha256-checked, renamed in; then every
                 quarantined copy and every shipped file is verified; one SHIPPED line; exit status = zero problems. Not
                 separable in the window (one call does manifest, transfer, quarantine and audit). The PLAN (no --apply)
                 is the script's own file listing (:18-31), ported here: which files would be shipped, replaced and
                 quarantined. Gate G14: the approval marker (production roots only; a mirror root is the script's
                 FT_TEST_ROOT case, snapshot_frametime_rerun.sh:76-79).
quarantine-rows  ops/snapshot_quarantine_wideband_fits.py QDIR STORE... [--dry] --quiet-min 10 (snapshot_rechain_batch.sh:
                 176-183): rows of wideband fits without the 'identity screen' mark (and their template rows) leave the fit
                 stores, with copies, the rows on their own, MANIFEST.csv and MANIFEST.sha256 in QDIR. The plan runs the
                 script with --dry, once per store (its dry run prints only totals, so per-store counts need one call
                 each; nothing is written either way). Run once as a migration (0 unmarked rows since 2026-10-08,
                 ops-snapshot-rest.md E10); catalog's store stage refuses unmarked rows instead (crosscheck group 11).
"""
import glob
import os
import re

from exopipe.phase3.core.record import NoInput, Refusal
from exopipe.phase3.lib.archive import common as K

_EXEC_BASH = 'import os, sys; os.execvp("bash", ["bash"] + sys.argv[1:])'
_SHIPPED = re.compile(r'^SHIPPED (\S+) files=(\d+) cutouts=(\d+) quarantined=(\d+) bad=(\d+) quarantine=(\S+)', re.M)
_ROWS = re.compile(r'^(DRY|QUARANTINED): (\d+) stores, (\d+) rows, (\d+) wideband rows without the identity-screen mark '
                   r'\(\+(\d+) of (\d+) template rows\)', re.M)


def _sn():
    return K.sec('archive_snapshot')


def _braces(pattern):
    """Expand one level of shell braces: 'a_{x,y}_b' -> ['a_x_b', 'a_y_b'] (the ship script's ls patterns)."""
    m = re.search(r'\{([^{}]*)\}', pattern)
    if not m:
        return [pattern]
    out = []
    for alt in m.group(1).split(','):
        out += _braces(pattern[:m.start()] + alt + pattern[m.end():])
    return out


def _ls(d, patterns):
    if not os.path.isdir(d):
        return []
    hits = set()
    for p in patterns:
        for q in _braces(p):
            hits.update(os.path.basename(x) for x in glob.glob(os.path.join(d, q)))
    return sorted(hits)


def supersede_plan(cell, workdir, ship, phot):
    """The ship script's listing (:18-31): new fleet files, old fleet files, new cutouts, old cutouts."""
    lst = cell.split('_', 1)[0]
    new_f = _ls(os.path.join(workdir, 'out'), _sn().SHIP_NEW_GLOBS)
    old_f = _ls(os.path.join(ship, cell), _sn().SHIP_OLD_GLOBS)
    cdir = os.path.join(workdir, 'cutouts')
    new_c = sorted(os.path.relpath(os.path.join(dp, f), cdir) for dp, _, fs in os.walk(cdir) for f in fs
                   if f.endswith('.fits')) if os.path.isdir(cdir) else []
    old_c = set()
    for g in ('Red', 'Green', 'Blue', 'Full'):
        for st in ('I', 'V'):
            n = os.path.join(ship, cell, f'noise_{g}_{st}_median.csv')
            try:
                with open(n) as f:
                    f.readline()
                    t1 = f.readline().split(',')[1].strip()
            except (OSError, IndexError):
                continue
            if t1:
                for x in glob.glob(os.path.join(phot, '*', g, '*', f'*_{lst}_{t1[:8]}T{t1[9:15]}_{st}.fits')):
                    old_c.add(os.path.relpath(x, phot))
    return {'new_fleet': new_f, 'old_fleet': old_f, 'new_cutouts': new_c, 'old_cutouts': sorted(old_c),
            'replaced_fleet': sorted(set(new_f) & set(old_f)), 'superseded_no_successor': sorted(set(old_f) - set(new_f)),
            'replaced_cutouts': sorted(set(new_c) & set(old_c)), 'cutouts_no_successor': sorted(old_c - set(new_c))}


def run_supersede(ctx, args, run, unit_fn):
    from exopipe.phase3.core import cells as CC
    if not args.cells:
        raise ValueError('supersede needs --cells <LST>_<YYYY-MM-DD> (snapshot night cells)')
    if not args.workdir:
        raise ValueError('supersede needs --workdir W (the frame-time work directory: W/out, W/cutouts)')
    C = K.cfg()
    ship = os.path.abspath(args.ship_root or C.paths.SNAPSHOT_FLEET_OUT)
    phot = os.path.abspath(args.phot_root or _sn().PHOT_ROOT)
    quar = os.path.abspath(args.quar_root or _sn().FRAMETIME_QUARANTINE_ROOT)
    for cell in CC.expand(args.cells, kind='night cell'):
        lst, date, sb = CC.parse_night_cell(cell)
        if sb:
            raise ValueError(f'supersede works on whole cells ({cell}: drop the subband)')
        wd = args.workdir.replace('{cell}', cell)

        def body(u, cell=cell, wd=wd):
            u.meta.update(cell=cell, workdir=wd, ship_root=ship, phot_root=phot, quar_root=quar)
            if not os.path.isdir(os.path.join(wd, 'out')):
                raise NoInput(f'{cell}: no work directory output {wd}/out')
            plan = supersede_plan(cell, wd, ship, phot)
            u.meta['plan_counts'] = {k: len(v) for k, v in plan.items()}
            if ctx.dry:
                return plan
            if not ctx.apply:
                p = os.path.join(ctx.manifest_dir('supersede'), f'{cell}_plan_{ctx.stamp}.json')
                pre = preship_problems(wd)
                K.write_json(p, {'action': 'supersede', 'cell': cell, 'workdir': wd, 'roots': [ship, phot, quar],
                                 'plan': plan, 'inputs': _fingerprints([os.path.join(wd, 'out', f) for f in plan['new_fleet']]),
                                 'preship_problems': pre, 'script': K.expand(_sn().SHIP_SCRIPT), 'written_utc': K.utc()})
                u.add_output(p, 'archive_plan')
                if pre:
                    u.note(f'the apply would refuse (G14 pre-ship checks): {"; ".join(pre[:3])}')
                return plan
            ctx.guard_writes([ship, phot, quar], f'supersede {cell}')
            production = any(K.production_like(r) for r in (ship, phot, quar))
            marker = K.expand(_sn().SUPERSEDE_APPROVAL_MARKER)
            if production and not os.path.exists(marker):
                raise Refusal(f'G14: no {marker} (Gregg approves the activity preamble first, CLAUDE.md 4b; '
                              f'snapshot_frametime_rerun.sh:80-81)')
            bad = preship_problems(wd)
            u.check('archive.G14_preship_content_tests', not bad, len(bad),
                    '; '.join(bad[:3]) if bad else 'kept 10-s inputs unchanged; content tests ' +
                    ','.join(_sn().SUPERSEDE_PRESHIP_TESTS) + ' PASS in the work dir')
            if bad:
                raise Refusal(f'G14: {cell}: the work dir did not pass the pre-ship checks (snapshot_frametime_rerun.sh:223-'
                              f'232 ships only after them) - nothing shipped: ' + '; '.join(bad[:3]))
            from exopipe.phase3.core import legacy
            log = os.path.join(ctx.out, 'logs', 'archive', f'supersede_{cell}.log')
            os.makedirs(os.path.dirname(log), exist_ok=True)
            res = legacy.run('-c', [_EXEC_BASH, K.expand(_sn().SHIP_SCRIPT), cell, wd, ship, phot, quar], python='default',
                             log=log, unit=u, refusal_codes=(1,), refusal_text=('REFUSED', 'FAILED'))
            m = _SHIPPED.search(res.tail)
            if not m:
                raise Refusal(f'G14: the ship script printed no SHIPPED line (exit {res.rc}): {res.tail[-300:]}')
            q = m.group(6)
            u.meta.update(shipped=dict(files=int(m.group(2)), cutouts=int(m.group(3)), quarantined=int(m.group(4)),
                                       bad=int(m.group(5)), quarantine=q))
            if not res.ok or int(m.group(5)):
                raise Refusal(f'G14: ship of {cell} not verified: bad={m.group(5)} (see {q}/verify.log)')
            u.add_output(os.path.join(q, 'MANIFEST.csv'), 'quarantine_manifest', algo='sha256')
            return plan
        unit_fn(f'supersede_{lst}_{date}', body, describe=lambda p: ', '.join(f'{k} {len(v)}' for k, v in p.items()
                                                                                 if k.startswith(('replaced', 'super', 'new'))))


_TEST_LINE = re.compile(r'^TEST (\S+) .*-> (PASS|FAIL)\s*$', re.M)


def preship_problems(wd):
    """The checks snapshot_frametime_rerun.sh runs on a work dir BEFORE its ship step (:223-232): the kept 10-s inputs
    unchanged (W/kept_10s_check.log written and empty: `sha256sum -c --quiet` prints nothing on success) and every
    content test of archive_snapshot.SUPERSEDE_PRESHIP_TESTS passed (W/test_<K>.log holds 'TEST <K> ... -> PASS'; the
    test exits 0 exactly when it prints PASS, snapshot_frametime_content_test.py:809). [] = ship allowed."""
    bad = []
    kept = os.path.join(wd, 'kept_10s_check.log')
    if not os.path.exists(kept):
        bad.append(f'no {kept} (kept 10-s inputs never checked)')
    elif os.path.getsize(kept):
        bad.append(f'kept 10-s inputs changed (see {kept})')
    for k in _sn().SUPERSEDE_PRESHIP_TESTS:
        p = os.path.join(wd, f'test_{k}.log')
        try:
            hits = [m for m in _TEST_LINE.finditer(open(p, errors='replace').read()) if m.group(1) == k]
        except OSError:
            bad.append(f'no {p} (content test {k} never ran)')
            continue
        if not hits or hits[-1].group(2) != 'PASS':
            bad.append(f'content test {k} did not pass ({p})')
    return bad


def _fingerprints(paths):
    from exopipe.phase3.core.record import fingerprint
    return {p: fingerprint(p) for p in paths}


def run_rows(ctx, args, run, unit_fn):
    from exopipe.phase3.core import cells as CC
    from exopipe.phase3.core import legacy
    stores = list(args.stores or [])
    if not stores and args.cells:
        broot = K.expand(_sn().BATCH_ROOT)
        stores = [os.path.join(broot, 'fits', f'fits_{c}.csv') for c in CC.expand(args.cells, kind='night cell')]
        stores = [s for s in stores if os.path.exists(s)]          # snapshot_rechain_batch.sh:176 ([ -f ] && echo)
    if not stores:
        raise NoInput('quarantine-rows: no store (give --stores FILE... or --cells of the batch)')
    name = args.batch_name or ctx.stamp
    qdir = os.path.abspath(args.qdir or K.expand(_sn().ROW_QUARANTINE_QDIR, name=name))
    quiet = str(args.quiet_min if args.quiet_min is not None else _sn().ROW_QUARANTINE_QUIET_MIN)
    script = K.expand(_sn().ROW_SCRIPT)

    def body(u):
        u.meta.update(stores=stores, qdir=qdir, quiet_min=float(quiet))
        for s in stores:
            u.add_input(s)
        logp = None if ctx.dry else os.path.join(ctx.out, 'logs', 'archive', f'rows_{name}.log')
        if logp:
            os.makedirs(os.path.dirname(logp), exist_ok=True)
        if not ctx.apply or ctx.dry:
            per = {}
            for s in stores:                                    # --dry: writes nothing (the script makes QDIR only without --dry)
                res = legacy.run(script, [qdir, s, '--dry', '--quiet-min', quiet], python=_sn().ROW_PYTHON, log=logp,
                                 unit=u, refusal_codes=(1,), refusal_text=('REFUSED',))
                if not res.ok:
                    raise Refusal(f'G12/G13: {res.reason or res.tail[-200:]}')
                m = _ROWS.search(res.tail)
                if not m:
                    raise Refusal(f'quarantine-rows: no DRY line from the script for {s}: {res.tail[-200:]}')
                per[s] = dict(rows=int(m.group(3)), quarantined=int(m.group(4)), template_quarantined=int(m.group(5)),
                              template_rows=int(m.group(6)))
            u.meta['per_store'] = per
            tot = sum(v['quarantined'] for v in per.values())
            u.note(f'DRY: {len(stores)} stores, {tot} wideband rows without the identity-screen mark')
            if ctx.dry:
                return per
            p = os.path.join(ctx.manifest_dir('quarantine_rows'), f'rows_{name}_plan_{ctx.stamp}.json')
            K.write_json(p, {'action': 'quarantine-rows', 'qdir': qdir, 'stores': stores, 'per_store': per,
                             'inputs': _fingerprints(stores), 'script': script, 'written_utc': K.utc()})
            u.add_output(p, 'archive_plan')
            return per
        ctx.guard_writes([qdir] + stores, 'quarantine-rows apply')
        if os.path.exists(os.path.join(qdir, 'MANIFEST.csv')):
            u.note(f'{qdir}/MANIFEST.csv exists: the script appends to it (as production)')
        res = legacy.run(script, [qdir] + stores + ['--quiet-min', quiet], python=_sn().ROW_PYTHON, log=logp, unit=u,
                         refusal_codes=(1,), refusal_text=('REFUSED',))
        if not res.ok:
            raise Refusal(f'quarantine-rows: {res.reason or res.tail[-200:]}')
        m = _ROWS.search(res.tail)
        u.meta['result'] = m.group(0) if m else res.tail[-200:]
        u.add_output(os.path.join(qdir, 'MANIFEST.csv'), 'row_quarantine_manifest', fresh=True)
        return {}
    unit_fn(f'rows_{name}', body, describe=lambda per: f'{len(stores)} store(s), '
            f'{sum(v.get("quarantined", 0) for v in per.values()) if per else "?"} unmarked wideband row(s)')
