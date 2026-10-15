"""Pre-search quarantine of a deep (cell, Stokes)'s earlier archive search frames (`exopipe3 archive quarantine`; the search
step's `quarantine` stage calls ``quarantine_for_search``; crosscheck group 10: archive owns this function).

Production: ops/p3s5_quarantine_frames.py <cell> <S> <launch epoch>, run by the deep search fleet before every FRESH attempt
(p3s5_search_fleet.sh:194-197, never when a completed attempt is adopted), from /lustre/gh/main/exopipe, with P3S5_QROUND
from the launch (p3s5_launch.sh:22-23) and P3_COADD_ROOT unset (:197-198). It
  1. checks the membership rulings of the cell's master (membership.verify + verify_products) and moves NOTHING when they fail
     or when no cell root holds coadd_final/weights.csv (:34-49; gate G7);
  2. selects every archive file of <root>/<LST>/*/<run>/<scope>/<S>/deep/ whose name has '_msub' and whose mtime is before the
     launch epoch (:50-51; archive_quarantine.SELECT_RULE - deep defect D2 kept);
  3. renames each to <root>/Run_SEARCH_QUARANTINE_<round>/<same relative path> (G4 destination free, G5 inode + size), one
     manifest row each in p3s5_state/frame_quarantine/<cell>_<S>.csv (path, dest, bytes, md5, cell, stokes, moved_at) (:52-67).

Here (refactor window, DESIGN 11.1): the PLAN (selection, destinations, G4, G3, md5) is evaluated in this process with the
helper's literal rule; the membership precheck runs production's code in a child (core.legacy.run, '-c' with sys.path[0] =
the production repo exactly as :37-39 - an in-process import would resolve the worktree's moved membership module); the
APPLY on the production archive runs the helper itself in place (only when paths.PRODUCTION_WRITES is True); an apply on a
mirror archive root (--archive-root) uses the same rename primitive (common.quarantine_move, the port of :58-67).
"""
import glob
import os

from exopipe.phase3.core.record import NoInput, Refusal
from exopipe.phase3.lib.archive import common as K

_PRECHECK = r'''
import os, sys
sys.path.insert(0, %(repo)r)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
from exopipe.phase3 import membership as _MB
cell = sys.argv[1]
roots = %(roots)r
marker = %(marker)r
cands = ["%%s/%%s" %% (os.environ.get('P3_COADD_ROOT', roots[0]), cell)] + ["%%s/%%s" %% (r, cell) for r in roots[1:]]
base = next((b for b in cands if os.path.exists("%%s/%%s" %% (b, marker))), None)
if base is None:
    print("PRECHECK-NO-INPUTS %%s: no cell inputs to judge the membership rulings - NOTHING quarantined - STOPPED" %% cell)
    sys.exit(4)
_vm = _MB.verify(base)
_vp = _MB.verify_products(base)
v = _vm + _vp
if len(sys.argv) > 2:          # D-ADMIT: the violations, for the parent's reading of the cell's admitted nights
    import json
    with open(sys.argv[2], 'w') as _f:
        json.dump({'base': base, 'membership': _vm, 'products': _vp}, _f)
if v:
    print("PRECHECK-FAIL %%s: master fails the membership rulings (%%s%%s) - the search would refuse it; NOTHING quarantined - STOPPED"
          %% (cell, v[0][:160], " ..." if len(v) > 1 else ""))
    sys.exit(5)
print("PRECHECK-OK %%s %%s" %% (cell, base))
'''


def _q():
    return K.sec('archive_quarantine')


def launch_epoch(value=None):
    """The launch epoch (UTC s): --launch-epoch, else the content of archive_quarantine.LAUNCH_EPOCH_FILE. Required: the
    fleet refuses an unset launch epoch (p3s5_search_fleet.sh:37)."""
    if value not in (None, ''):
        return float(value)
    p = K.expand(_q().LAUNCH_EPOCH_FILE)
    try:
        return float(open(p).read().strip())
    except (OSError, ValueError) as e:
        raise Refusal(f'launch epoch unset: no --launch-epoch and {p} unreadable ({e}); the fleet refuses an unset launch '
                      f'epoch (p3s5_search_fleet.sh:37)')


def quarantine_root(archive_root, round_name):
    return os.path.join(archive_root, f'{_q().RUN_PREFIX}{round_name}')


def select_frames(archive_root, run_name, cell, stokes, launch):
    """p3s5_quarantine_frames.py:50-51: archive '_msub' files of (cell, Stokes) last modified before the launch epoch."""
    if _q().SELECT_RULE != 'mtime-before-launch':
        raise Refusal(f'archive_quarantine.SELECT_RULE = {_q().SELECT_RULE!r}: only the production rule '
                      f"'mtime-before-launch' is implemented (deep defect D2 waits for the deep chat)")
    lst, scope = cell.split('_', 1)
    tok = K.sec().SEARCH_FRAME_TOKEN
    return [p for p in sorted(glob.glob(K.deep_frames_glob(archive_root, run_name, lst, scope, stokes)))
            if tok in os.path.basename(p) and os.path.getmtime(p) < launch]


def plan_frame_quarantine(cell, stokes, launch, round_name, archive_root=None, run_name=None, checksum=True):
    """The rows production would write, without moving anything: [{path, dest, bytes, md5, cell, stokes, moved_at: ''}].
    G4 (a destination exists: the helper would stop) and G3 (different filesystem) refuse here, before any move."""
    C = K.cfg()
    archive_root = archive_root or C.paths.PHASE3_ARCHIVE_ROOT
    run_name = run_name or C.paths.ARCHIVE_RUN
    qroot = quarantine_root(archive_root, round_name)
    rows = []
    for p in select_frames(archive_root, run_name, cell, stokes, launch):
        q = os.path.join(qroot, os.path.relpath(p, archive_root))
        if os.path.lexists(q):
            raise Refusal(f'G4: {cell} {stokes}: quarantine destination exists: {q} - the helper would STOP there '
                          f'(choose another --round)')
        if K.sec().SAME_FILESYSTEM_ONLY and not K.same_filesystem(p, q):
            raise Refusal(f'G3: {p} and {q} are on different filesystems')
        rows.append({'path': p, 'dest': q, 'bytes': os.path.getsize(p), 'md5': K.checksum(p, 'md5') if checksum else '',
                     'cell': cell, 'stokes': stokes, 'moved_at': ''})
    return rows


def membership_precheck(cell, unit=None, log=None):
    """G7: production's membership verdict on the cell (child process, production code), read with the cell's admitted
    nights (D-ADMIT, masters_admit.ADMITTED_NIGHTS): the child also writes its violations to a file and a failure that is
    only nights admitted into the master ('admitted, rebuild pending': lib/masters/admit.membership) passes. Returns
    (ok, message)."""
    import json
    import shutil
    import tempfile
    from exopipe.phase3.core import legacy
    from exopipe.phase3.lib.masters import admit as AD
    q = _q()
    code = _PRECHECK % {'repo': K.sec().LEGACY_CWD, 'roots': tuple(q.PRECHECK_CELL_ROOTS), 'marker': q.PRECHECK_CELL_MARKER}
    td = tempfile.mkdtemp(prefix='exopipe3_precheck_')
    try:
        jp = os.path.join(td, 'violations.json')
        res = legacy.run('-c', [code, cell, jp], python='default', cwd=K.sec().LEGACY_CWD, log=log, unit=unit,
                         refusal_codes=(4, 5), refusal_text=('STOPPED',))
        line = next((l for l in reversed(res.tail.splitlines()) if l.startswith('PRECHECK-')), res.tail[-300:])
        if not res.ok and line.startswith('PRECHECK-FAIL') and os.path.exists(jp):
            with open(jp) as f:
                got = json.load(f)
            viol, rep = AD.membership(got['base'], got['membership'])
            if not viol and not got['products']:
                return True, f'PRECHECK-OK {cell} {got["base"]} (' + '; '.join(rep) + ')'
        return res.ok, line
    finally:
        shutil.rmtree(td, ignore_errors=True)


def quarantine_for_search(u, cell, stokes, launch, round_name, apply=False, archive_root=None, run_name=None,
                          out=None, precheck=True, checksum=True, check_writer=True, log=None, approved_manifest=None):
    """The whole pre-search quarantine of one (cell, Stokes), recorded in unit ``u`` (an archive or a search unit).

    Plan (apply False): G7 precheck, plan rows, written to <out>/manifests/frame_quarantine/<cell>_<S>_plan_<UTC>.csv
    (class quarantine_manifest; moved_at empty) when ``out`` is given. Apply: G2, G12 (concurrent writer), then the
    production helper in place (production archive) or the rename primitive (mirror archive), rows appended to the
    manifest of record (production: p3s5_state/frame_quarantine/<cell>_<S>.csv). Returns the rows."""
    C = K.cfg()
    archive_root = os.path.abspath(archive_root or C.paths.PHASE3_ARCHIVE_ROOT)
    run_name = run_name or C.paths.ARCHIVE_RUN
    production = os.path.realpath(archive_root) == os.path.realpath(C.paths.PHASE3_ARCHIVE_ROOT)
    u.meta.update(cell=cell, stokes=stokes, launch_epoch=launch, round=round_name, archive_root=archive_root)
    if precheck:
        ok, msg = membership_precheck(cell, unit=u, log=log)
        u.check('archive.G7_membership_precheck', ok, detail=msg)
        u.note(f'G7 (production membership code): {msg}')
        if not ok:
            raise Refusal(f'G7: {msg}')
    else:
        u.note('G7 membership precheck skipped (--no-membership-precheck with a ruling)')
    rows = plan_frame_quarantine(cell, stokes, launch, round_name, archive_root, run_name, checksum=checksum)
    u.meta['n_planned'] = len(rows)
    u.meta['bytes_planned'] = sum(int(r['bytes']) for r in rows)
    if approved_manifest:
        _, man = K.read_csv(approved_manifest)
        man = [r for r in man if r.get('cell', cell) == cell and r.get('stokes', stokes) == stokes]
        diffs = K.compare_to_manifest({r['path']: r['bytes'] for r in rows}, man, 'path', 'bytes',
                                      'md5' if checksum else None, sums={r['path']: r['md5'] for r in rows})
        u.check('archive.G11_manifest_equal', not diffs, len(diffs), '; '.join(diffs[:3]) or 'plan = approved manifest')
        if diffs:
            raise Refusal('G11: the plan re-derived now differs from --manifest: ' + '; '.join(diffs[:3]))
    if not apply:
        if out:
            p = os.path.join(out, 'manifests', 'frame_quarantine', f'{cell}_{stokes}_plan_{K.stamp()}.csv')
            K.write_csv(p, K.sec().QUARANTINE_COLUMNS, rows)
            u.add_output(p, 'quarantine_manifest')
        u.note(f'{cell} {stokes}: {len(rows)} earlier search frames would be quarantined '
               f'({u.meta["bytes_planned"] / 1e9:.2f} GB) to {quarantine_root(archive_root, round_name)}')
        return rows
    # ---- apply
    if production:
        mdir = K.expand(K.sec().FRAME_QUARANTINE_MANIFEST_DIR)
    else:
        mdir = os.path.join(out or '.', 'manifests', 'frame_quarantine')
    manifest = os.path.join(mdir, f'{cell}_{stokes}.csv')
    if not C.paths.PRODUCTION_WRITES:
        for p in (archive_root, manifest):
            if K.production_like(p):
                raise Refusal(f'G2: quarantine apply would write under the production tree {K.production_like(p)} ({p}) '
                              f'and paths.PRODUCTION_WRITES is False (refactor window: plan only, or a mirror --archive-root)')
    if check_writer and production:              # the fleet's claims name production cells; a mirror has no such writer
        live, owner = K.writer_claim(cell, stokes)
        if live:
            raise Refusal(f'G12: the search fleet holds a live claim of {cell} {stokes} ({owner}) - a concurrent writer')
    t0 = K.utc()
    if production:
        _apply_helper(u, cell, stokes, launch, round_name, rows, manifest, t0, log)
    else:
        with K.CsvAppender(manifest, K.sec().QUARANTINE_COLUMNS) as w:
            for r in rows:
                b, m, t = K.quarantine_move(r['path'], r['dest'], 'md5', what=f'{cell} {stokes}: quarantine')
                if r['md5'] and m != r['md5']:
                    u.note(f"{r['path']}: md5 changed between plan and move ({r['md5']} -> {m}); recorded value is the move's")
                w.write(dict(r, bytes=b, md5=m, moved_at=t))
    u.add_output(manifest, 'quarantine_manifest', since=t0)
    u.note(f'{cell} {stokes}: {len(rows)} earlier search frames quarantined')
    return rows


def _apply_helper(u, cell, stokes, launch, round_name, rows, manifest, t0, log):
    """The production helper in place (p3s5_search_fleet.sh:197 with p3s5_launch.sh:23); refused on any non-zero exit
    ('frame quarantine FAILED - not searched'), and the rows it appended must be the planned paths."""
    from exopipe.phase3.core import legacy
    helper = K.expand(_q().HELPER)
    res = legacy.run(helper, [cell, stokes, repr(launch) if not float(launch).is_integer() else str(int(launch))],
                     python='default', env={'P3S5_QROUND': round_name}, cwd=K.sec().LEGACY_CWD, log=log, unit=u,
                     refusal_codes=(1,), refusal_text=('STOPPED',))
    if not res.ok:
        raise Refusal(f'frame quarantine FAILED - not searched: {res.reason or res.tail[-200:]}')
    got = []
    if os.path.exists(manifest):
        _, man = K.read_csv(manifest)
        got = [r for r in man if (r.get('moved_at') or '') >= t0]
    want, have = {r['path'] for r in rows}, {r['path'] for r in got}
    u.check('archive.quarantine_rows_equal_plan', want == have, len(have),
            f'{len(have)} rows appended, {len(want)} planned' + ('' if want == have else
                                                               f'; differ by {len(want ^ have)}'))


def run_action(ctx, items, stokes, args, unit_fn):
    """`archive quarantine --tier deep --cells C --stokes S`: one unit per (cell, Stokes)."""
    launch = launch_epoch(getattr(args, 'launch_epoch', None))
    round_name = args.round or _q().ROUND_DEFAULT
    precheck = _q().MEMBERSHIP_PRECHECK and not args.no_membership_precheck
    if args.no_membership_precheck:
        K.require_ruling(args.apply_ruling, '--no-membership-precheck')
    for cell in items:
        for st in stokes:
            def body(u, cell=cell, st=st):
                return quarantine_for_search(u, cell, st, launch, round_name, apply=ctx.apply and not ctx.dry,
                                             archive_root=ctx.archive_root, run_name=ctx.gen_run,
                                             out=None if ctx.dry else ctx.out, precheck=precheck, checksum=not ctx.dry,
                                             approved_manifest=args.manifest, log=ctx_log(ctx, f'quarantine_{cell}_{st}'))
            unit_fn(f'quarantine_{cell}_{st}', body, describe=lambda rows: f'{len(rows)} frame(s) to '
                    f'{quarantine_root(ctx.archive_root, round_name)}')


def ctx_log(ctx, unit):
    if ctx.dry:
        return None
    p = os.path.join(ctx.out, 'logs', 'archive', f'{unit}.log')
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


__all__ = ['launch_epoch', 'select_frames', 'plan_frame_quarantine', 'membership_precheck', 'quarantine_for_search',
           'quarantine_root', 'NoInput']
