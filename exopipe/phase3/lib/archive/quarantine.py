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
the production repo exactly as :37-39 - an in-process import would resolve the worktree's moved membership module); every
APPLY - on the production archive (paths.PRODUCTION_WRITES) as on a mirror archive root (--archive-root) - moves with the
port of the helper's move loop (common.quarantine_move, :58-67). The helper itself is never run: it writes its manifest
under paths.OPS_ROOT (p3s5_state/frame_quarantine, :18), which v2 never writes. A NEW round's manifest of record is
<archive.SEARCH_QUARANTINE_RECORDS_DIR>/<cell>_<S>.csv, one directory per round under archive.RECORDS_ROOT, with the round's
record (its launch epoch); a round production used, or a round of another launch epoch, refuses (round_gate). The old
p3s5_state manifests stay where they are, read only (the refactor chat's decision of 2026-11-27, cutover row 64 step 2).
"""
import glob
import json
import os
import re
import time

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


# the narrow allowance key of the v2 deep search's archive writes (paths.PRODUCTION_WRITES_ALLOWED): its quarantine round, its
# archive frames and the launch file begin_launch writes
ALLOW_KEY = 'archive:search_deep'


def _q():
    return K.sec('archive_quarantine')


def launch_epoch(value=None):
    """The launch epoch (UTC s): --launch-epoch, else the content of archive_quarantine.LAUNCH_EPOCH_FILE (written by
    begin_launch, `exopipe3 archive quarantine --begin-launch`). Required: the fleet refuses an unset launch epoch
    (p3s5_search_fleet.sh:37)."""
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


def round_for(launch, configured):
    """The quarantine round: 'launch-date' = the UTC date of the launch epoch (YYYYMMDD; production named its rounds so,
    P3S5_QROUND 20261004 ... 20261113), else the literal round. A round that is not a plain name refuses."""
    rnd = time.strftime('%Y%m%d', time.gmtime(float(launch))) if configured == 'launch-date' else str(configured or '')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', rnd):
        raise Refusal(f'quarantine round {rnd!r} (search_run.QUARANTINE_ROUND {configured!r}): a plain name, e.g. launch-date')
    return rnd


def round_records_dir(archive_root, round_name, out=None):
    """Where a round's manifest of record and round record live: archive.SEARCH_QUARANTINE_RECORDS_DIR (under
    archive.RECORDS_ROOT) for the production archive, <out>/manifests/frame_quarantine/<run> for a mirror archive root."""
    run = f'{_q().RUN_PREFIX}{round_name}'
    C = K.cfg()
    if os.path.realpath(os.path.abspath(archive_root)) == os.path.realpath(C.paths.PHASE3_ARCHIVE_ROOT):
        return K.expand(K.sec().SEARCH_QUARANTINE_RECORDS_DIR, run=run)
    return os.path.join(out or '.', 'manifests', 'frame_quarantine', run)


def under_ops_root(path):
    """paths.OPS_ROOT when ``path`` is under it (production's state and manifests, never written by v2), else None."""
    ops = os.path.realpath(K.cfg().paths.OPS_ROOT).rstrip('/')
    real = os.path.realpath(os.path.abspath(path))
    return ops if real == ops or real.startswith(ops + '/') else None


def round_gate(archive_root, round_name, launch, rdir, create=False):
    """A NEW round (the refactor chat's decision of 2026-11-27): the round's record in ``rdir`` names this launch epoch, or
    there is no record and no quarantine Run of that name in the archive (a Run without a v2 record is a round production
    used - 20261004, ..., 20261113 - whose manifests stay under p3s5_state, read only). ``create``: write the record (before
    any move; a concurrent unit's record is read and compared). Refuses: an earlier round, another launch's round, a record
    directory under paths.OPS_ROOT. Returns the record."""
    if under_ops_root(rdir):
        raise Refusal(f'quarantine round {round_name}: its records directory {rdir} is under paths.OPS_ROOT '
                      f'({under_ops_root(rdir)}), which v2 never writes - REFUSED')
    rec_path = os.path.join(rdir, _q().ROUND_RECORD)
    want = {'round': round_name, 'run': f'{_q().RUN_PREFIX}{round_name}', 'launch_epoch': float(launch)}

    def compare(rec):
        if float(rec.get('launch_epoch', 'nan')) != float(launch):
            raise Refusal(f'quarantine round {round_name}: its record {rec_path} names launch epoch {rec.get("launch_epoch")}, '
                          f'this run {float(launch)} - one round per launch; a new launch needs a new round (launch date)')
        return rec
    if os.path.exists(rec_path):
        return compare(K.read_json(rec_path))
    qroot = quarantine_root(archive_root, round_name)
    if os.path.lexists(qroot):
        raise Refusal(f'quarantine round {round_name}: {qroot} exists with no v2 round record ({rec_path}) - an earlier '
                      f'round (production\'s manifests: p3s5_state/frame_quarantine, read only); a NEW round is required '
                      f'(search_run.QUARANTINE_ROUND launch-date with a new launch epoch, or --round)')
    if not create:
        return dict(want, created='')
    os.makedirs(rdir, exist_ok=True)
    rec = dict(want, created=K.utc(), host=os.uname()[1])
    try:
        fd = os.open(rec_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:                       # a concurrent unit of the same launch wrote it first
        return compare(K.read_json(rec_path))
    with os.fdopen(fd, 'w') as fh:
        json.dump(rec, fh, indent=1)
        fh.flush()
        os.fsync(fh.fileno())
    return rec


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
    nights (D-ADMIT, masters_admit.ADMITTED_NIGHTS) and the 18 MHz exception as the search judges the master: the child also
    writes its violations to a file, and a failure that is only nights admitted into the master ('admitted, rebuild pending':
    lib/masters/admit.membership) or, under search_detect.MASTER_HELD_18MHZ 'keep', nights held only for their excluded
    18 MHz (admit.keep_held_18; production's p3s5_quarantine_frames.keep_held_18 since 2026-11-08) passes - the quarantine
    never refuses a cell the search accepts (search/admitted.verify_master). Returns (ok, message)."""
    import json
    import shutil
    import tempfile
    from exopipe.phase3.core import legacy
    from exopipe.phase3.lib.masters import admit as AD
    from exopipe.phase3.lib.search import admitted as SA
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
            mv, rep18 = got['membership'], []
            if SA.step_held_18(K.cfg().search_detect.MASTER_HELD_18MHZ) == 'keep':
                mv, rep18 = AD.keep_held_18(mv)
            viol, rep = AD.membership(got['base'], mv)
            if not viol and not got['products']:
                return True, f'PRECHECK-OK {cell} {got["base"]} (' + '; '.join(rep18 + rep) + ')'
        return res.ok, line
    finally:
        shutil.rmtree(td, ignore_errors=True)


def quarantine_for_search(u, cell, stokes, launch, round_name, apply=False, archive_root=None, run_name=None,
                          out=None, precheck=True, checksum=True, check_writer=True, log=None, approved_manifest=None):
    """The whole pre-search quarantine of one (cell, Stokes), recorded in unit ``u`` (an archive or a search unit).

    Plan (apply False): G7 precheck, the round gate (read only), plan rows, written to
    <out>/manifests/frame_quarantine/<cell>_<S>_plan_<UTC>.csv (class quarantine_manifest; moved_at empty) when ``out`` is
    given. Apply: G2, G12 (concurrent writer), the round gate (the round's record written), then the rename primitive, rows
    appended to the round's manifest of record (production archive: archive.SEARCH_QUARANTINE_RECORDS_DIR/<cell>_<S>.csv
    under archive.RECORDS_ROOT; mirror: <out>/manifests/frame_quarantine/<run>/<cell>_<S>.csv). Never under paths.OPS_ROOT.
    Returns the rows."""
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
    rdir = round_records_dir(archive_root, round_name, out)
    round_gate(archive_root, round_name, launch, rdir, create=False)      # an earlier round refuses at the plan already
    u.meta.update(round_records=rdir)
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
    manifest = os.path.join(rdir, f'{cell}_{stokes}.csv')
    if under_ops_root(manifest):
        raise Refusal(f'G2: the quarantine manifest of record {manifest} is under paths.OPS_ROOT, which v2 never writes '
                      f'(production\'s p3s5_state manifests are read only) - REFUSED')
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
    rec = round_gate(archive_root, round_name, launch, rdir, create=True)
    u.note(f'quarantine round {rec["run"]} (launch epoch {rec["launch_epoch"]}); manifest of record {manifest}')
    # the port of the helper's move loop (p3s5_quarantine_frames.py:52-67), for the production archive as for a mirror:
    # the helper writes its manifest under paths.OPS_ROOT. Its membership check (:34-49) is G7 above (production's code in a
    # child, read with the cell's admitted nights and the 18 MHz exception as the search judges the master)
    with K.CsvAppender(manifest, K.sec().QUARANTINE_COLUMNS) as w:
        for r in rows:
            b, m, t = K.quarantine_move(r['path'], r['dest'], 'md5', what=f'{cell} {stokes}: quarantine')
            if r['md5'] and m != r['md5']:
                u.note(f"{r['path']}: md5 changed between plan and move ({r['md5']} -> {m}); recorded value is the move's")
            w.write(dict(r, bytes=b, md5=m, moved_at=t))
    u.add_output(manifest, 'quarantine_manifest', since=t0)
    u.note(f'{cell} {stokes}: {len(rows)} earlier search frames quarantined')
    return rows


def begin_launch(u, epoch=None, retire=False, apply=False):
    """`exopipe3 archive quarantine --begin-launch [--launch-epoch UTC_S] [--retire-launch] --apply`: the operator step that
    starts a v2 deep-search launch (the refactor chat's decision (1) of 2026-11-27) - archive_quarantine.LAUNCH_EPOCH_FILE
    written with the launch epoch (UTC s; default now), which launch_epoch and the search's quarantine read.

    The new launch's round (ROUND_DEFAULT 'launch-date': Run_SEARCH_QUARANTINE_<its UTC date>) must be free (round_gate, read
    only). An existing launch file refuses, named, unless ``retire``: then it is first moved into its own round's records
    directory, beside that round's ROUND.json (round_records_dir of its launch date; a link that never replaces, then the
    unlink) - refused when that round is the new launch's (two launches on one UTC date), when the round's record names
    another launch epoch, or when a retired file is already there. The new file: a temporary file opened O_EXCL, written and
    synced, then linked into place (a link never replaces: a concurrent writer's file refuses), the temporary file removed;
    read back. Apply only (otherwise the plan): refused under paths.OPS_ROOT, for an injected run, and under a production
    tree while paths.PRODUCTION_WRITES is False unless the narrow allowance opens it to ALLOW_KEY. Returns the plan."""
    from exopipe.phase3.core import injected
    C = K.cfg()
    q = _q()
    p = K.expand(q.LAUNCH_EPOCH_FILE)
    new = float(epoch) if epoch not in (None, '') else round(time.time(), 3)
    arch = C.paths.PHASE3_ARCHIVE_ROOT
    rnd = round_for(new, q.ROUND_DEFAULT)
    round_gate(arch, rnd, new, round_records_dir(arch, rnd), create=False)     # the new launch's round is free
    plan = {'launch_file': p, 'launch_epoch': new, 'round': rnd, 'retired_to': None,
            'launch_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(new))}
    old = None
    if os.path.lexists(p):
        try:
            old = float(open(p).read().strip())
        except (OSError, ValueError) as e:
            raise Refusal(f'begin-launch: {p} exists and is unreadable ({e}) - REFUSED (nothing written)')
        if not retire:
            raise Refusal(f'begin-launch: {p} exists (launch epoch {old}, '
                          f'{time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(old))}) - a launch is under way or was not '
                          f'retired; a new launch first moves it into its round\'s directory (--retire-launch)')
        ornd = round_for(old, q.ROUND_DEFAULT)
        if ornd == rnd:
            raise Refusal(f'begin-launch: the launch in {p} ({old}) has round {rnd}, the new launch\'s - two launches on one '
                          f'UTC date would share a quarantine round; REFUSED')
        odir = round_records_dir(arch, ornd)
        orec = os.path.join(odir, q.ROUND_RECORD)
        if os.path.exists(orec) and float(K.read_json(orec).get('launch_epoch', 'nan')) != old:
            raise Refusal(f'begin-launch: the round record {orec} names launch epoch '
                          f'{K.read_json(orec).get("launch_epoch")}, the launch file {old} - REFUSED')
        plan['retired_to'] = os.path.join(odir, os.path.basename(p))
        if os.path.lexists(plan['retired_to']):
            raise Refusal(f'begin-launch: {plan["retired_to"]} exists - the launch of round {ornd} was retired already; '
                          f'REFUSED')
    u.meta.update(begin_launch=plan)
    if not apply:
        u.note(f'begin-launch (plan): {p} <- {new} ({plan["launch_utc"]}, round Run_SEARCH_QUARANTINE_{rnd})'
               + (f'; the launch {old} moved first to {plan["retired_to"]}' if old is not None else ''))
        return plan
    targets = [p] + ([plan['retired_to']] if old is not None else [])
    injected.check_all(targets, 'archive quarantine --begin-launch')
    for t in targets:
        if under_ops_root(t):
            raise Refusal(f'G2: begin-launch would write {t} under paths.OPS_ROOT ({under_ops_root(t)}), which v2 never writes '
                          f'- REFUSED')
        if K.production_like(t) and not C.writes_allowed(ALLOW_KEY, t):
            raise Refusal(f'G2: begin-launch would write {t} under the production tree {K.production_like(t)} and '
                          f'paths.PRODUCTION_WRITES is False and paths.PRODUCTION_WRITES_ALLOWED does not open it to '
                          f'{ALLOW_KEY!r} - REFUSED')
    if old is not None:
        os.makedirs(os.path.dirname(plan['retired_to']), exist_ok=True)
        try:
            os.link(p, plan['retired_to'])
        except FileExistsError:
            raise Refusal(f'begin-launch: {plan["retired_to"]} appeared meanwhile - REFUSED (nothing moved)')
        os.unlink(p)
        u.note(f'begin-launch: the launch {old} retired to {plan["retired_to"]}')
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = f'{p}.{os.uname()[1]}.{os.getpid()}.tmp'
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(fd, 'w') as fh:
            fh.write(f'{new!r}\n')
            fh.flush()
            os.fsync(fh.fileno())
        try:
            os.link(tmp, p)
        except FileExistsError:
            raise Refusal(f'begin-launch: {p} appeared meanwhile (a concurrent begin-launch) - REFUSED')
    finally:
        os.unlink(tmp)
    if launch_epoch() != new:
        raise Refusal(f'begin-launch: {p} reads back {launch_epoch()}, written {new}')
    u.note(f'begin-launch: {p} = {new} ({plan["launch_utc"]}); round Run_SEARCH_QUARANTINE_{rnd}')
    return plan


def run_begin_launch(ctx, args, unit_fn):
    """`archive quarantine --begin-launch`: one unit, begin_launch (no --cells: the launch is the fleet's)."""
    unit_fn('begin_launch', lambda u: begin_launch(u, getattr(args, 'launch_epoch', None),
                                                   retire=bool(getattr(args, 'retire_launch', False)),
                                                   apply=ctx.apply and not ctx.dry),
            describe=lambda plan: f'{plan["launch_file"]} <- {plan["launch_epoch"]} (round {plan["round"]})'
            + (f'; the earlier launch to {plan["retired_to"]}' if plan['retired_to'] else ''))


def run_action(ctx, items, stokes, args, unit_fn):
    """`archive quarantine --tier deep --cells C --stokes S`: one unit per (cell, Stokes)."""
    launch = launch_epoch(getattr(args, 'launch_epoch', None))
    round_name = round_for(launch, args.round or _q().ROUND_DEFAULT)
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


__all__ = ['ALLOW_KEY', 'launch_epoch', 'begin_launch', 'select_frames', 'plan_frame_quarantine', 'membership_precheck', 'quarantine_for_search',
           'quarantine_root', 'round_for', 'round_records_dir', 'round_gate', 'under_ops_root', 'NoInput']
