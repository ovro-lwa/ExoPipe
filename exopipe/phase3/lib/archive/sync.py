"""`archive sync` and `archive retire-refused` (deep): make the archive search frames of a published (cell, Stokes) attempt
equal the attempt's frames.

Production: ops/p3s5_archive_sync.py (written 2026-10-04 under Gregg's 'yes to both - patch path approved', R-INSTALL-GATE-
AND-SEARCH-PATCH-PATH ledger 1789; NEVER APPLIED - deep defect D6 / medium defect M10). Its plan per (cell, Stokes):
  keep     archive file equal to the attempt frame in pixels (same_data: same blanks, 1e-6 relative)
  replace  archive file whose pixels differ -> quarantined to Run_SEARCH_QUARANTINE_20261004, the attempt frame copied in
  publish  attempt frame with no archive file -> copied in (md5 verified)
  retire   archive '_msub' file of the cell and Stokes that no attempt frame names (a refused or excluded night) -> quarantined
The module is imported IN PLACE (spec 7.2 item 2: it imports only csv, numpy, astropy and, in main, p3s3_sources) and its
``plan`` and ``same_data`` are called as production would; its module globals ARCH / RUN / QROOT are pointed at a mirror
root for a mirror run (the values are restored afterwards). Its ``apply`` is NOT called: it hard-codes the archive root
and the ops manifests (gate G2) and moves before it copies; the apply here (mirror roots in the window, the production
archive after cutover) is the spec's stage order: transfer ('.part' copies, md5 verified) -> audit (each '.part' equals
its attempt frame in pixels, by the module's same_data) -> quarantine (superseded / retired files renamed out with manifest
rows, then each '.part' renamed in) -> closing re-plan, which must be all 'keep' (G17). Retire rows need Gregg's go
(archive.RULING_REQUIRED 'sync:retire', M10 deferred with the excluded-night quarantine).

retire-refused = the 'retire' class alone, split by core.exclusions into gate-refused nights (moved by this action, with a
ruling) and excluded nights (left to retire-excluded and only listed here) - the two counts the refactor triage (16) and
the deep chat (52 incl. excluded nights, ledger 2309) gave for M10.
"""
import contextlib
import importlib
import os
import sys

from exopipe.phase3.core.record import NoInput, Refusal
from exopipe.phase3.lib.archive import common as K


def _s():
    return K.sec('archive_sync')


def legacy_module():
    """Import ops/p3s5_archive_sync.py in place (P3S5_STATE and P3S_LABEL set first, as p3s5_archive_sync.py:26, :120) and
    check that its literals still equal the config MIRROR values (else the step's description of what it runs is untrue)."""
    from exopipe.phase3.core import env
    C = K.cfg()
    os.environ['P3S5_STATE'] = K.expand(K.sec().STATE_DIR)
    os.environ['P3S_LABEL'] = _s().RUN_LABEL
    root = env.legacy_root('exopipe_ops')
    if root not in sys.path:
        sys.path.append(root)
    S = importlib.import_module(_s().MODULE)
    import inspect
    problems = []
    if os.path.realpath(S.ARCH) != os.path.realpath(C.paths.PHASE3_ARCHIVE_ROOT) and not getattr(S, '_exopipe3_mirror', False):
        problems.append(f'ARCH {S.ARCH} != paths.PHASE3_ARCHIVE_ROOT')
    if S.RUN != C.paths.ARCHIVE_RUN and not getattr(S, '_exopipe3_mirror', False):
        problems.append(f'RUN {S.RUN} != paths.ARCHIVE_RUN')
    if os.path.basename(S.QROOT) != _s().QUARANTINE_RUN:
        problems.append(f'QROOT {S.QROOT} != archive_sync.QUARANTINE_RUN')
    src = inspect.getsource(S.same_data)
    for lit in (repr(_s().SAME_DATA_RTOL).replace('1e-06', '1e-6'), repr(_s().SAME_DATA_ATOL)):
        if lit not in src:
            problems.append(f'same_data no longer uses {lit}')
    psrc = inspect.getsource(S.plan)
    if f"'{_s().ARCHIVE_NAME_CARD}'" not in psrc or f'[:{_s().ATTEMPT_DATE_CHARS}]' not in psrc:
        problems.append('plan() naming rule (PRODNAME, date chars) differs from config')
    if problems:
        raise Refusal('archive_sync MIRROR values differ from the in-place module ' + _s().MODULE + ': ' + '; '.join(problems))
    return S


@contextlib.contextmanager
def roots(S, archive_root, run_name):
    """Point the module's ARCH / RUN / QROOT at ``archive_root`` / ``run_name`` (a mirror) for the duration."""
    old = (S.ARCH, S.RUN, S.QROOT)
    S.ARCH, S.RUN, S.QROOT = archive_root, run_name, os.path.join(archive_root, _s().QUARANTINE_RUN)
    S._exopipe3_mirror = True
    try:
        yield S
    finally:
        S.ARCH, S.RUN, S.QROOT = old
        S._exopipe3_mirror = False


@contextlib.contextmanager
def names_only(S):
    """Plan without reading pixels: same_data answers True, so every named archive file is 'keep'. The 'retire' and
    'publish' rows - all that retire-refused needs - do not depend on same_data (p3s5_archive_sync.py:65-69)."""
    old = S.same_data
    S.same_data = lambda a, b: True
    try:
        yield S
    finally:
        S.same_data = old


def attempts(args):
    """[(attempt dir, cell, stokes)]: --attempt dirs (basename <cell>_<S>_<att>, p3s5_archive_sync.py:114-118) or the
    published, harvested attempts of the register (p3s3_sources.published_attempts, :119-122)."""
    out = []
    if args.attempt:
        for d in args.attempt:
            b = os.path.basename(d.rstrip('/'))
            parts = b.rsplit('_', 2)
            if len(parts) < 3:
                raise ValueError(f'--attempt {d}: basename must be <cell>_<S>_<attempt>')
            out.append((d.rstrip('/'), parts[0], parts[1]))
        return out
    from exopipe.phase3.core import env
    os.environ['P3S_LABEL'] = _s().RUN_LABEL
    root = env.legacy_root('exopipe_ops')
    if root not in sys.path:
        sys.path.append(root)
    if 'p3s3_sources' in sys.modules and sys.modules['p3s3_sources'].LABEL != _s().RUN_LABEL:
        raise Refusal(f"p3s3_sources was imported with label {sys.modules['p3s3_sources'].LABEL}, not {_s().RUN_LABEL}")
    P = importlib.import_module('p3s3_sources')
    for x in P.published_attempts(['V', 'I']):
        if x['harvested'] or _s().ATTEMPTS != 'published-harvested':
            out.append((x['dir'], x['cell'], x['stokes']))
    return out


def select(args, cells, stokes):
    atts = attempts(args)
    if cells:
        atts = [a for a in atts if a[1] in cells]
    return [a for a in atts if a[2] in stokes]


def _date_of(path, archive_root):
    rel = os.path.relpath(path, archive_root).split(os.sep)
    return rel[1] if len(rel) > 1 else ''


def plan(S, ctx, att, cell, st, pixels=True):
    """The module's plan for (cell, Stokes) against ctx's roots: [(action, archive path, attempt path|None)]."""
    with roots(S, ctx.archive_root, ctx.gen_run):
        if pixels:
            return S.plan(att, cell, st)
        with names_only(S):
            return S.plan(att, cell, st)


def counts(acts):
    return {k: sum(1 for x in acts if x[0] == k) for k in ('keep', 'replace', 'publish', 'retire')}


def class_of(name):
    """Product class of an archived per-epoch file (archive_push.EPOCH_NAME_CLASSES, first match)."""
    b = os.path.basename(name)
    for tok, tok2, cls in K.sec('archive_push').EPOCH_NAME_CLASSES:
        if tok in b and (tok2 is None or tok2 in b):
            return cls
    return 'epoch_image'


# ---------------------------------------------------------------------------------------------------------------------
# sync
# ---------------------------------------------------------------------------------------------------------------------

def sync_unit(u, ctx, S, att, cell, st, args):
    u.meta.update(cell=cell, stokes=st, attempt=att, archive_root=ctx.archive_root, run=ctx.gen_run)
    if not os.path.isdir(att):
        raise NoInput(f'attempt directory {att} does not exist')
    if ctx.dry:
        lst, sc = cell.split('_', 1)
        import glob
        fr = sorted(glob.glob(f"{att}/Run_{_s().RUN_LABEL}/{_s().ATTEMPT_FRAME_DIR[st]}/{cell}/*.fits"))
        arch = [p for p in glob.glob(K.deep_frames_glob(ctx.archive_root, ctx.gen_run, lst, sc, st))
                if K.sec().SEARCH_FRAME_TOKEN in os.path.basename(p)]
        return {'attempt_frames': len(fr), 'archive_msub': len(arch), 'note': 'dry run: no headers or pixels read'}
    try:
        acts = plan(S, ctx, att, cell, st)
    except RuntimeError as e:          # G8: no PRODNAME / PRODNAME not a search frame (p3s5_archive_sync.py:56, :64)
        raise Refusal(f'G8: {e}')
    c = counts(acts)
    ex = _exclusions()
    u.add_input(ex.path, extra=ex.fingerprint())
    retire_excl = [a for x, a, p in acts if x == 'retire' and _excluded(ex, a, ctx.archive_root)]
    # G20 (exclusion absolute, CLAUDE.md 2; never waivable): a publish / replace row on an excluded (LST, scope, date)
    # would put that night's frame back into the generation Run - the attempt searched a night excluded since (an
    # attempt older than data_decisions.csv is never adopted: FIX-FLEET-ADOPTED-PRE-EXCLUSION-ATTEMPTS-20261012). The
    # production module plans these rows as any other (p3s5_archive_sync.py:62-68); the step refuses them.
    readmit = [[x, a] for x, a, p in acts if x in ('publish', 'replace') and _excluded(ex, a, ctx.archive_root)]
    keep_excl = [a for x, a, p in acts if x == 'keep' and _excluded(ex, a, ctx.archive_root)]
    rec = {'kind': 'plan', 'cell': cell, 'stokes': st, 'attempt': att, 'archive_root': ctx.archive_root, 'run': ctx.gen_run,
           'actions': [list(x) for x in acts], 'counts': c, 'retire_excluded': retire_excl, 'readmit_excluded': readmit,
           'keep_excluded': keep_excl, 'written_utc': K.utc(), 'exclusions': ex.fingerprint()}
    u.meta['counts'] = c
    u.note(f'{cell} {st}: ' + ', '.join(f'{k} {v}' for k, v in c.items()) + ('' if ctx.apply else ' (plan)'))
    if keep_excl:
        u.note(f'{len(keep_excl)} kept archive frame(s) are of excluded (LST, scope, date) (the attempt predates the '
               f'exclusion): sync leaves them; `archive retire-excluded` moves them (Gregg\'s go, ledger 2278)')
    u.check('archive.G20_exclusion_no_readmit', not readmit, len(readmit),
            (f'{len(readmit)} publish/replace row(s) on excluded (LST, scope, date) - the attempt searched an excluded '
             f'night; re-search the cell, never sync it: e.g. {readmit[0]}') if readmit else
            'no publish/replace row on an excluded (LST, scope, date)')
    if not ctx.apply:
        p = os.path.join(ctx.manifest_dir('sync'), f'{cell}_{st}_plan_{ctx.stamp}.json')
        K.write_json(p, rec)
        u.add_output(p, 'archive_sync_record')
        return c
    if readmit:
        raise Refusal(f'G20: sync of {cell} {st} would put {len(readmit)} frame(s) of excluded (LST, scope, date) back '
                      f'into the archive (exclusion absolute), e.g. {readmit[0][1]} - nothing moved')
    # ---- apply (mirror in the window; production after cutover)
    if args.manifest:
        old = K.read_json(args.manifest)
        if [list(x) for x in old.get('actions', [])] != rec['actions']:
            raise Refusal(f'G11: the plan re-derived now differs from --manifest {args.manifest}')
    if c['retire'] and 'sync:retire' in K.sec().RULING_REQUIRED:
        K.require_ruling(args.apply_ruling, f'sync of {cell} {st} with {c["retire"]} retire row(s)')
    qdir = ctx.applied_manifest_dir('frame_quarantine', K.sec().FRAME_QUARANTINE_MANIFEST_DIR)
    pdir = ctx.applied_manifest_dir('frame_publish', K.sec().FRAME_PUBLISH_MANIFEST_DIR)
    qman, pman = os.path.join(qdir, f'{cell}_{st}.csv'), os.path.join(pdir, f'{cell}_{st}.csv')
    ctx.guard_writes([ctx.archive_root, qman, pman], f'sync apply of {cell} {st}')
    live, owner = K.writer_claim(cell, st)
    if live and ctx.production_archive:
        raise Refusal(f'G12: the search fleet holds a live claim of {cell} {st} ({owner})')
    todo = [x for x in acts if x[0] != 'keep']
    if not todo:
        # production applies only when something is to do (p3s5_archive_sync.py:129): no manifest file is created
        u.note('nothing to do: every archive frame equals its attempt frame')
        rec.update(kind='closing', applied_actions=rec['actions'], closing_not_keep=[], written_utc=K.utc())
        p = os.path.join(ctx.manifest_dir('sync'), f'{cell}_{st}_closing_{ctx.stamp}.json')
        K.write_json(p, rec)
        u.add_output(p, 'archive_sync_record')
        return c
    need = sum(os.path.getsize(p) for x, a, p in todo if x in ('replace', 'publish'))
    ok, free = K.free_space_ok(ctx.archive_root, K.sec().FREE_SPACE_FLOOR_GB)
    if not ok:
        raise Refusal(f'G19: {free} GB free on the archive filesystem < archive.FREE_SPACE_FLOOR_GB')
    t0 = K.utc()
    parts = {}
    try:
        for x, a, p in todo:                                    # transfer
            if x in ('replace', 'publish'):
                parts[a] = K.verified_copy(p, a, 'md5', preserve=True)
        for x, a, p in todo:                                    # audit of the copies, before anything live moves
            if a in parts and not S.same_data(parts[a][0], p):
                raise Refusal(f'audit: the copy {parts[a][0]} does not equal {p} in pixels')
    except Refusal:
        for part, _ in parts.values():
            if os.path.exists(part):
                os.remove(part)                                 # only this run's own '.part' files
        raise
    u.meta['bytes_transferred'] = need
    with K.CsvAppender(qman, K.sec().QUARANTINE_COLUMNS) as qw, K.CsvAppender(pman, K.sec().PUBLISH_COLUMNS) as pw:
        for x, a, p in todo:                                    # quarantine, then rename into place
            if x in ('replace', 'retire'):
                q = os.path.join(ctx.archive_root, _s().QUARANTINE_RUN, os.path.relpath(a, ctx.archive_root))
                b, m, t = K.quarantine_move(a, q, 'md5', what=f'sync {cell} {st}')
                qw.write(dict(path=a, dest=q, bytes=b, md5=m, cell=cell, stokes=st, moved_at=t))
            if x in ('replace', 'publish'):
                part, m = parts[a]
                K.commit_part(part, a, replace=True)
                pw.write(dict(source=p, dest=a, bytes=os.path.getsize(a), md5=m, cell=cell, stokes=st, action=x,
                              written_at=K.utc()))
    closing = plan(S, ctx, att, cell, st)                         # G17: the re-plan must be all 'keep'
    bad = [(x, a) for x, a, p in closing if x != 'keep']
    rec.update(kind='closing', actions=[list(x) for x in closing], applied_actions=rec['actions'], closing_not_keep=bad,
               counts=counts(closing), written_utc=K.utc(), applied_since=t0)
    p = os.path.join(ctx.manifest_dir('sync'), f'{cell}_{st}_closing_{ctx.stamp}.json')
    K.write_json(p, rec)
    if bad:
        raise Refusal(f'G17: ARCHIVE-SYNC-AUDIT-FAIL {cell} {st}: {bad[:3]}')
    u.add_output(qman, 'quarantine_manifest', since=t0)
    u.add_output(pman, 'archive_manifest', since=t0)
    for x, a, _ in todo:
        if x in ('replace', 'publish'):
            u.add_output(a, class_of(a))
    u.add_output(p, 'archive_sync_record')
    u.note(f'ARCHIVE-SYNC-AUDIT-OK {cell} {st}: archive frames equal the run frames in pixels')
    return c


# ---------------------------------------------------------------------------------------------------------------------
# retire-refused
# ---------------------------------------------------------------------------------------------------------------------

def _exclusions():
    from exopipe.phase3.core import exclusions
    return exclusions.load()


def _excluded(ex, path, archive_root):
    """<LST>/<date>/<run>/<scope>/<S>/<cadence>/<name> of an excluded (LST, scope, date) (common.excluded_key)."""
    return K.excluded_key(ex, path, archive_root) is not None


def refused_rows(S, ctx, att, cell, st, checksum=True):
    """(rows to move, excluded paths listed only): the plan's 'retire' rows, split by core.exclusions."""
    acts = plan(S, ctx, att, cell, st, pixels=False)
    ex = _exclusions()
    qroot = os.path.join(ctx.archive_root, K.sec('archive_retire').REFUSED_QUARANTINE_RUN)
    rows, excl = [], []
    for x, a, p in acts:
        if x != 'retire':
            continue
        if _excluded(ex, a, ctx.archive_root):
            excl.append(a)
            continue
        rows.append({'path': a, 'dest': os.path.join(qroot, os.path.relpath(a, ctx.archive_root)),
                     'bytes': os.path.getsize(a), 'md5': K.checksum(a, 'md5') if checksum else '', 'cell': cell,
                     'stokes': st, 'moved_at': ''})
    return rows, excl


def retire_refused_unit(u, ctx, S, att, cell, st, args):
    u.meta.update(cell=cell, stokes=st, attempt=att, archive_root=ctx.archive_root, run=ctx.gen_run)
    if not os.path.isdir(att):
        raise NoInput(f'attempt directory {att} does not exist')
    ex = _exclusions()
    u.add_input(ex.path, extra=ex.fingerprint())
    try:
        rows, excl = refused_rows(S, ctx, att, cell, st, checksum=not ctx.dry)
    except RuntimeError as e:
        raise Refusal(f'G8: {e}')
    u.meta.update(n_refused_night_frames=len(rows), n_excluded_night_frames=len(excl))
    u.note(f'{cell} {st}: {len(rows)} archive frame(s) of gate-refused nights not named by the attempt; {len(excl)} more '
           f'of excluded nights (left to retire-excluded)')
    for r in rows:
        if os.path.lexists(r['dest']):
            raise Refusal(f"G4: quarantine destination exists: {r['dest']}")
    if ctx.dry:
        return {'refused_night_frames': len(rows), 'excluded_night_frames': len(excl)}
    audit = {'kind': 'refused', 'cell': cell, 'stokes': st, 'attempt': att, 'archive_root': ctx.archive_root,
             'run': ctx.gen_run, 'rows': [r['path'] for r in rows], 'excluded_listed': excl, 'exclusions': ex.fingerprint()}
    if not ctx.apply:
        p = os.path.join(ctx.manifest_dir('retire_refused'), f'{cell}_{st}_plan_{ctx.stamp}.csv')
        K.write_csv(p, K.sec().QUARANTINE_COLUMNS, rows)
        u.add_output(p, 'quarantine_manifest')
        a = os.path.join(ctx.manifest_dir('retire_refused'), f'{cell}_{st}_audit_plan_{ctx.stamp}.json')
        K.write_json(a, dict(audit, state='plan', manifest=p, written_utc=K.utc()))
        u.add_output(a, 'retirement_audit')
        return len(rows)
    K.require_ruling(args.apply_ruling, f'retire-refused of {cell} {st}')
    if not args.manifest:
        raise Refusal('G11: retire-refused --apply needs --manifest <the approved plan>')
    _, man = K.read_csv(args.manifest)
    man = [r for r in man if r.get('cell') == cell and r.get('stokes') == st]
    diffs = K.compare_to_manifest({r['path']: r['bytes'] for r in rows}, man, 'path', 'bytes', 'md5',
                                  sums={r['path']: r['md5'] for r in rows})
    if diffs:
        raise Refusal('G11: plan differs from --manifest: ' + '; '.join(diffs[:3]))
    qman = os.path.join(ctx.applied_manifest_dir('frame_quarantine', K.sec().FRAME_QUARANTINE_MANIFEST_DIR), f'{cell}_{st}.csv')
    ctx.guard_writes([ctx.archive_root, qman], f'retire-refused apply of {cell} {st}')
    K.refuse_live_writers(ctx, [(cell, st)], f'retire-refused of {cell} {st}')
    t0 = K.utc()
    with K.CsvAppender(qman, K.sec().QUARANTINE_COLUMNS) as w:
        for r in rows:
            b, m, t = K.quarantine_move(r['path'], r['dest'], 'md5', what='retire-refused')
            w.write(dict(r, bytes=b, md5=m, moved_at=t))
    u.add_output(qman, 'quarantine_manifest', since=t0)
    a = os.path.join(ctx.manifest_dir('retire_refused'), f'{cell}_{st}_audit_applied_{ctx.stamp}.json')
    K.write_json(a, dict(audit, state='applied', manifest=qman, applied_since=t0, written_utc=K.utc()))
    u.add_output(a, 'retirement_audit')
    return len(rows)
