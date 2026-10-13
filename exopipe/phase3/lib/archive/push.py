"""`archive push` (manifest -> checksum transfer -> content audit; CLAUDE.md 7; RULING-LUSTRE-PUSH-GO-recover-local-class-
approved-asap, ledger 335; R-PUSH-WHEN-READY, 489) and `archive audit` of an earlier push.

push --class mosaic   the mosaic v2 Taper push, procedure of ops/push_mosaic_v2_20260929.py (its manifest/push/audit modes
                      write into ops/ with hard-coded paths, so they are ported here, not run - spec 7.2 item 5):
                      manifest: every staged file -> <root>/mosaic/Run_FEATHER2_<d0>-<d1>/<band>/I/{healpix,mollweide,radec}/
                      <name> and tables/; REFUSES a non-Taper name, no staged file, an intermediate file, any destination that
                      exists (never overwrite). transfer: source unchanged since the manifest (size + md5), then each file to
                      '<dst>.part', md5 verified, renamed in (production renames first and checks after; the end state of a
                      successful push is the same, a failed copy here leaves nothing at <dst>). audit: the content audit of
                      push_mosaic_v2:75-114 on every destination (class archive_mosaic_copy).
push --class epoch    per hour (DESIGN 12.1): one EPOCH item <LST>_<date> = every file of that hour under the staging tree
                      <staging>/<LST>/<date>/<run>/<scope>/<S>/<cadence>/ (a tree mirroring the archive layout - in the window
                      the search step's <out>/archive_mirror) -> <root>/<same relative path>. Per file: keep (equal md5, or
                      equal pixels by p3s5_archive_sync.same_data), publish (absent), replace (different: the old file is
                      renamed to <root>/Run_SEARCH_QUARANTINE_<round>/<rel> with a manifest row, then the new one renamed in -
                      standing-approved, ledger 1677). A staged file of an excluded (LST, scope, date) or an intermediate name
                      is a refused item, never pushed. Without a staging tree (archive_push.EPOCH_STAGING_ROOT None and no
                      --staging-root) the unit audits the hour's archive files (keep rows, each content-tested by its class).
                      Bulk (intermittent, DESIGN 12.2): --epochs @list, or --lsts/--nights over the staging tree.
push --class qa | run-tables   disabled (archive_push.CLASSES[...]['enabled'] False): refused with the reason.
"""
import glob
import os

from exopipe.phase3.core.record import NoInput, Refusal
from exopipe.phase3.lib.archive import common as K


def _p():
    return K.sec('archive_push')


def class_cfg(name):
    c = _p().CLASSES.get(name)
    if c is None:
        raise ValueError(f'push --class {name}: unknown (classes: {", ".join(_p().CLASSES)})')
    if not c.get('enabled'):
        raise Refusal(f'push --class {name} is disabled (archive_push.CLASSES[{name!r}]["enabled"] False): ' +
                      {'qa': 'ARCHIVED-PHASE3-QA-TO-LUSTRE-AND-INDEXED-THE-PER-NIGHT-PHASE2-QA (1494) made one manual copy; '
                             'no standing QA push was asked',
                       'run-tables': 'nothing pushes run catalogues / pages / verdict journals today and no destination is '
                                     'ruled (B17 gap, archive.md 12 item 5)'}.get(name, 'not enabled'))
    return c


# ---------------------------------------------------------------------------------------------------------------------
# mosaic
# ---------------------------------------------------------------------------------------------------------------------

def audit_mosaic(dst):
    """Problems of one archived mosaic file (push_mosaic_v2_20260929.py:79-107, verbatim rules): name parses and round-trips;
    PBCORR agrees with the 'pbcorr' token; BUNIT agrees with the domain token; NSTACK = the name's N; DATE-OBS / DATE-END
    present; at least one finite value at stride 97. The data HDU: an image in HDU 0, else the HEALPix table in HDU 1
    (the first audit read the wrong HDU, ledger 1620). A .csv table: md5 is checked by the manifest test instead."""
    import numpy as np
    from astropy.io import fits
    from exopipe.phase3.core.products import build_name, parse_name
    unit = _p().AUDIT_BUNIT
    stride = int(_p().AUDIT_PIXEL_STRIDE)
    p = []
    try:
        n = os.path.basename(dst)
        pn = parse_name(n)
        if build_name(**pn) != n:
            p.append('name does not round-trip')
        proc = pn.get('proc') or []
        with fits.open(dst, memmap=True) as h:
            dh = h[0] if h[0].data is not None else h[1]
            hdr_all = {**dict(h[0].header), **dict(dh.header)}
            if bool(hdr_all.get('PBCORR')) != ('pbcorr' in proc):
                p.append(f"PBCORR {hdr_all.get('PBCORR')} vs name")
            dom = next((t for t in ('psfmap', 'pbmap', 'jybeam', 'jysr') if t in proc), None)
            if dom and str(hdr_all.get('BUNIT', '')).strip() != unit[dom]:
                p.append(f"BUNIT {hdr_all.get('BUNIT')!r} vs {dom}")
            if int(hdr_all.get('NSTACK', -1)) != int(pn.get('n_epochs') or -2):
                p.append(f"NSTACK {hdr_all.get('NSTACK')} vs N{pn.get('n_epochs')}")
            for k in ('DATE-OBS', 'DATE-END'):
                if not hdr_all.get(k):
                    p.append(f'no {k}')
            arr = dh.data if dh is h[0] else dh.data['SIGNAL']
            if not np.isfinite(np.asarray(arr[::stride])).any():
                p.append('no finite pixel')
    except Exception as e:
        p.append(f'unreadable {str(e)[:60]}')
    return p


def mosaic_manifest(ctx, c, checksum=True):
    """Rows (src, dst, bytes, md5) of the mosaic push (push_mosaic_v2:42-58), with the refusals of G9."""
    from exopipe.phase3.core.products import parse_name
    fs = sorted([f for g in c['src_globs'] for f in glob.glob(g)])
    bad = [f for f in fs if '_Taper_' not in os.path.basename(f)] if c.get('taper_only') else []
    if bad or not fs:
        raise Refusal(f'G9: REFUSED: {len(fs)} staged files, {len(bad)} not Taper {bad[:3]}')
    inter = [f for f in fs + list(c.get('tables') or ()) if K.intermediate(f)]
    if inter:
        raise Refusal(f'G9: {len(inter)} intermediate file(s) staged for push, e.g. {inter[0]}')
    d0 = min(parse_name(os.path.basename(f))['date_start'] for f in fs)
    d1 = max(parse_name(os.path.basename(f))['date_end'] for f in fs)
    R = c['run_dir'].format(root=ctx.archive_root, d0=d0, d1=d1)
    rows = []
    for f in fs:
        n = os.path.basename(f)
        band = n.split('_')[0]
        sub = 'healpix' if '/wideband/' in f and '/renders/' not in f else os.path.basename(os.path.dirname(f))
        rows.append({'src': f, 'dst': f'{R}/{band}/I/{sub}/{n}', 'bytes': os.path.getsize(f)})
    for t in c.get('tables') or ():
        rows.append({'src': t, 'dst': f'{R}/tables/{os.path.basename(t)}', 'bytes': os.path.getsize(t)})
    if c.get('existing', 'refuse') == 'refuse':
        ex = [r['dst'] for r in rows if os.path.lexists(r['dst'])]
        if ex:
            raise Refusal(f'G9: REFUSED: {len(ex)} destinations already exist, e.g. {ex[0]}')
    for r in rows:
        r['md5'] = K.checksum(r['src'], 'md5') if checksum else ''
    return R, rows


def run_mosaic(ctx, args, run, unit_fn):
    c = class_cfg('mosaic')
    stages = args._stages

    def body(u):
        u.meta.update(push_class='mosaic', stages=stages)
        if args.manifest:
            _, rows = K.read_csv(args.manifest)
            man = args.manifest
            R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(rows[0]['dst'])))) if rows else ''
        else:
            if 'manifest' not in stages:
                raise Refusal('push --class mosaic: transfer/audit without the manifest stage need --manifest FILE')
            R, rows = mosaic_manifest(ctx, c, checksum=not ctx.dry)
            man = None
        u.meta.update(run_dir=R, n_files=len(rows), bytes=sum(int(r['bytes']) for r in rows))
        if ctx.dry:
            return rows
        if man is None:
            man = os.path.join(ctx.manifest_dir('push'), f'MANIFEST_MOSAIC_{os.path.basename(R)}_{ctx.stamp}.csv')
            K.write_csv(man, ('src', 'dst', 'bytes', 'md5'), rows)
        if 'transfer' in stages:
            ctx.guard_writes([R] + [r['dst'] for r in rows[:1]], 'push --class mosaic')
            if _p().REFUSE_CHANGED_SOURCE:
                for r in rows:
                    if os.path.getsize(r['src']) != int(r['bytes']) or K.checksum(r['src'], 'md5') != r['md5']:
                        raise Refusal(f"push: REFUSED: {r['src']} changed since the manifest")
            ok, free = K.free_space_ok(R, K.sec().FREE_SPACE_FLOOR_GB)
            if not ok:
                raise Refusal(f'G19: {free} GB free < archive.FREE_SPACE_FLOOR_GB')
            for i, r in enumerate(rows, 1):
                if os.path.lexists(r['dst']):
                    raise Refusal(f"push: REFUSED: {r['dst']} exists")
                part, _ = K.verified_copy(r['src'], r['dst'], 'md5', want=r['md5'], preserve=False)
                K.commit_part(part, r['dst'], replace=False)
            u.note(f'PUSHED {len(rows)} files, all md5-verified')
        if 'audit' in stages:
            audit_rows(u, ctx, man, rows, 'mosaic')
        else:
            u.add_output(man, 'archive_manifest')
        return rows
    name = 'push_mosaic' + (f'_{os.path.splitext(os.path.basename(args.manifest))[0]}' if args.manifest else '')
    unit_fn(name, body, describe=lambda rows: f'{len(rows)} file(s), {sum(int(r["bytes"]) for r in rows) / 1e9:.2f} GB')


def audit_rows(u, ctx, man, rows, push_class):
    """The content audit of a push manifest: the manifest test (rows_transferred) and each destination's own class test;
    writes <out>/manifests/audit/<manifest>_audit.csv (dst, problems) as push_mosaic_v2:108-110 does."""
    from exopipe.phase3.lib.archive.sync import class_of
    out = []
    for r in rows:
        d = r.get('dst') or r.get('dest')
        if not d or not os.path.exists(d):
            out.append({'dst': d, 'problems': 'missing'})
            continue
        if d.endswith('.csv'):
            m = r.get('md5', '')
            out.append({'dst': d, 'problems': '' if (not m or K.checksum(d, 'md5') == m) else 'md5'})
            continue
        cls = 'archive_mosaic_copy' if push_class == 'mosaic' else class_of(d)
        ent = u.add_output(d, cls)
        bad = [t for t in u.tests if t.get('path') == ent['path'] and not t['passed']]
        out.append({'dst': d, 'problems': '; '.join(f"{t['name']}: {t.get('detail')}" for t in bad)})
    a = os.path.join(ctx.manifest_dir('audit'), os.path.splitext(os.path.basename(man))[0] + f'_audit_{ctx.stamp}.csv')
    K.write_csv(a, ('dst', 'problems'), out)
    u.add_output(a, 'csv_table', required_columns=('dst', 'problems'))
    u.add_output(man, 'archive_manifest')
    nbad = sum(1 for o in out if o['problems'])
    u.note(f'AUDIT: {len(out)} files, {nbad} with problems -> {a}')
    return nbad


# ---------------------------------------------------------------------------------------------------------------------
# epoch (per hour) and bulk epoch pushes
# ---------------------------------------------------------------------------------------------------------------------

def epoch_items(args, staging):
    from exopipe.phase3.core import cells as CC
    if getattr(args, 'epochs', None):
        return CC.expand_epochs(args.epochs)
    if not staging:
        raise ValueError('push --class epoch: give --epochs <LST>_<YYYY-MM-DD> (per hour), or --staging-root with '
                         '--lsts / --nights (bulk)')
    lsts = CC.expand(args.lsts, universe=CC.LSTS) if getattr(args, 'lsts', None) else CC.LSTS
    nights = set(CC.expand(args.nights)) if getattr(args, 'nights', None) else None
    items = []
    for lst in lsts:
        d = os.path.join(staging, lst)
        if os.path.isdir(d):
            items += [f'{lst}_{x}' for x in sorted(os.listdir(d)) if (nights is None or x in nights)
                      and os.path.isdir(os.path.join(d, x))]
    return items


def _walk(base):
    out = []
    for dp, dn, fn in os.walk(base):
        dn.sort()
        out += [os.path.join(dp, f) for f in sorted(fn)]
    return out


def epoch_plan(ctx, item, staging, checksum=True):
    """(rows, refused [(path, reason, gate)], archived_excluded [paths]) of one epoch item."""
    from exopipe.phase3.core import cells as CC
    from exopipe.phase3.core import exclusions
    lst, date = CC.parse_epoch(item)
    ex = exclusions.load()
    rows, refused, arch_excl = [], [], []
    same_data = None
    if staging:
        base = os.path.join(staging, lst, date, ctx.gen_run)
        for f in _walk(base) if os.path.isdir(base) else []:
            rel = os.path.relpath(f, staging)
            parts = rel.split(os.sep)
            if K.intermediate(f):
                refused.append((f, 'intermediate file name (archive_push.INTERMEDIATE_PATTERN): never pushed', 'G9'))
                continue
            if len(parts) != 7:
                refused.append((f, f'not in the archive layout <LST>/<date>/<run>/<scope>/<S>/<cadence>/<name>', 'G9'))
                continue
            scope, st = parts[3], parts[4]
            if ex.is_excluded(lst, scope, date):
                refused.append((f, f'excluded ({lst}, {scope}, {date}) in data_decisions.csv: never pushed', 'exclusion'))
                continue
            dst = os.path.join(ctx.archive_root, rel)
            m = K.checksum(f, ctx.algo) if checksum else ''
            if not os.path.lexists(dst):
                act = 'publish'
            elif not checksum:
                act = 'keep-or-replace'
            elif K.checksum(dst, ctx.algo) == m:
                act = 'keep'
            else:
                act = 'replace'
                if f.endswith('.fits'):
                    if same_data is None:
                        from exopipe.phase3.lib.archive.sync import legacy_module
                        same_data = legacy_module().same_data
                    if same_data(dst, f):
                        act = 'keep'
            rows.append({'source': f, 'dest': dst, 'bytes': os.path.getsize(f), 'md5': m, 'cell': f'{lst}_{scope}',
                         'stokes': st, 'action': act, 'written_at': ''})
    else:
        base = os.path.join(ctx.archive_root, lst, date, ctx.gen_run)
        for f in _walk(base) if os.path.isdir(base) else []:
            parts = os.path.relpath(f, ctx.archive_root).split(os.sep)
            if len(parts) != 7 or K.intermediate(f):
                continue
            scope, st = parts[3], parts[4]
            if ex.is_excluded(lst, scope, date):
                arch_excl.append(f)
                continue
            rows.append({'source': '', 'dest': f, 'bytes': os.path.getsize(f),
                         'md5': K.checksum(f, ctx.algo) if checksum else '', 'cell': f'{lst}_{scope}', 'stokes': st,
                         'action': 'keep', 'written_at': ''})
    return rows, refused, arch_excl


def run_epoch(ctx, args, run, unit_fn):
    c = class_cfg('epoch')
    if ctx.tier not in c.get('tiers', ('deep',)):
        raise NoInput(f'push --class epoch --tier {ctx.tier}: the {ctx.tier} tier archives no per-epoch products today '
                      f'(sub-LST searches with P3_ARCHIVE_WRITE=0, sublst_search_pass.sh:30; snapshot products are shipped '
                      f'by the fleet / `archive supersede`); nothing to push')
    staging = args.staging_root or _p().EPOCH_STAGING_ROOT
    rnd = args.round or K.sec('archive_quarantine').ROUND_DEFAULT
    qrun = f'{_p().EPOCH_REPLACE_QUARANTINE_PREFIX}{rnd}'
    stages = args._stages
    for item in epoch_items(args, staging):
        def body(u, item=item):
            u.meta.update(push_class='epoch', epoch=item, staging_root=staging, archive_root=ctx.archive_root,
                          mode='push' if staging else 'audit-only')
            rows, refused, arch_excl = epoch_plan(ctx, item, staging, checksum=not ctx.dry)
            for f, why, gate in refused:
                u.refuse_item(f, why, gate=gate)
            if arch_excl:
                u.note(f'{len(arch_excl)} archive file(s) of excluded keys of this hour are in the archive (left to '
                       f'`archive retire-excluded`, which needs Gregg\'s go), e.g. {arch_excl[0]}')
                u.meta['archived_excluded'] = arch_excl
            if not rows and not refused:
                raise NoInput(f'{item}: no per-epoch product under {staging or ctx.archive_root}')
            cnt = {k: sum(1 for r in rows if r['action'] == k) for k in ('keep', 'publish', 'replace', 'keep-or-replace')}
            u.meta['counts'] = cnt
            if ctx.dry:
                return rows
            man = os.path.join(ctx.manifest_dir('push_epoch'), f'{item}_{ctx.stamp}.csv')
            if not staging or not ctx.apply or 'transfer' not in stages:
                K.write_csv(man, K.sec().PUBLISH_COLUMNS, rows)
                if rows:
                    if staging:
                        u.add_output(man, 'archive_manifest')
                    else:
                        audit_rows(u, ctx, man, rows, 'epoch')
                return rows
            # ---- apply
            todo = [r for r in rows if r['action'] in ('publish', 'replace')]
            qdir = ctx.applied_manifest_dir('frame_quarantine', K.sec().FRAME_QUARANTINE_MANIFEST_DIR)
            pdir = ctx.applied_manifest_dir('frame_publish', K.sec().FRAME_PUBLISH_MANIFEST_DIR)
            ctx.guard_writes([ctx.archive_root, qdir, pdir, man], f'push --class epoch {item}')
            for r in todo:
                q = os.path.join(ctx.archive_root, qrun, os.path.relpath(r['dest'], ctx.archive_root))
                if r['action'] == 'replace' and os.path.lexists(q):
                    raise Refusal(f'G4: quarantine destination exists: {q} (choose another --round)')
            ok, free = K.free_space_ok(ctx.archive_root, K.sec().FREE_SPACE_FLOOR_GB)
            if not ok:
                raise Refusal(f'G19: {free} GB free < archive.FREE_SPACE_FLOOR_GB')
            parts = {}
            try:
                for r in todo:
                    parts[r['dest']] = K.verified_copy(r['source'], r['dest'], ctx.algo, want=r['md5'], preserve=True)
            except Refusal:
                for part, _ in parts.values():
                    if os.path.exists(part):
                        os.remove(part)
                raise
            t0 = K.utc()
            touched = set()
            for r in todo:
                key = f"{r['cell']}_{r['stokes']}.csv"
                if r['action'] == 'replace':
                    q = os.path.join(ctx.archive_root, qrun, os.path.relpath(r['dest'], ctx.archive_root))
                    with K.CsvAppender(os.path.join(qdir, key), K.sec().QUARANTINE_COLUMNS) as w:
                        b, m, t = K.quarantine_move(r['dest'], q, ctx.algo, what=f'push {item} replace')
                        w.write(dict(path=r['dest'], dest=q, bytes=b, md5=m, cell=r['cell'], stokes=r['stokes'], moved_at=t))
                    touched.add(('q', key))
                part, m = parts[r['dest']]
                K.commit_part(part, r['dest'], replace=True)
                r['written_at'] = K.utc()
                with K.CsvAppender(os.path.join(pdir, key), K.sec().PUBLISH_COLUMNS) as w:
                    w.write(r)
                touched.add(('p', key))
            K.write_csv(man, K.sec().PUBLISH_COLUMNS, rows)
            for kind, key in sorted(touched):
                if kind == 'q':
                    u.add_output(os.path.join(qdir, key), 'quarantine_manifest', since=t0)
            audit_rows(u, ctx, man, [dict(r, dst=r['dest']) for r in rows if r['action'] in ('publish', 'replace')], 'epoch')
            return rows
        unit_fn(f'push_epoch_{item}', body, describe=lambda rows: ', '.join(
            f'{k} {sum(1 for r in rows if r["action"] == k)}' for k in ('keep', 'publish', 'replace', 'keep-or-replace')
            if any(r['action'] == k for r in rows)) or '0 files')


def run_action(ctx, args, run, unit_fn):
    cls = args.push_class
    if not cls:
        cls = 'epoch' if getattr(args, 'epochs', None) else None
    if not cls:
        raise ValueError('push needs --class (mosaic, epoch, qa, run-tables); --epochs alone means --class epoch')
    class_cfg(cls)
    if cls == 'mosaic':
        return run_mosaic(ctx, args, run, unit_fn)
    if cls == 'epoch':
        return run_epoch(ctx, args, run, unit_fn)
    raise Refusal(f'push --class {cls}: enabled in config but not implemented')


# ---------------------------------------------------------------------------------------------------------------------
# audit of earlier manifests
# ---------------------------------------------------------------------------------------------------------------------

AUDIT_CLASSES = {'mosaic': 'archive_manifest', 'epoch': 'archive_manifest', 'quarantine': 'quarantine_manifest',
                 'sync': 'archive_sync_record', 'retire': None, 'rollback': 'restore_manifest',
                 'rows': 'row_quarantine_manifest', 'supersede': 'quarantine_manifest'}


def run_audit(ctx, args, run, unit_fn):
    cls = args.push_class
    if cls not in AUDIT_CLASSES:
        raise ValueError(f'audit --class {cls}: one of {", ".join(AUDIT_CLASSES)}')
    mans = []
    for m in args.manifest_list or []:
        mans += [x.strip() for x in open(m[1:])] if m.startswith('@') else [m]
    if not mans:
        raise ValueError('audit needs --manifest FILE (repeatable, or @list)')
    for man in mans:
        def body(u, man=man):
            if not os.path.exists(man):
                raise NoInput(f'manifest {man} does not exist')
            u.add_input(man)
            u.meta.update(audit_class=cls, manifest=man)
            if ctx.dry:
                return []
            if cls in ('mosaic', 'epoch'):
                _, rows = K.read_csv(man)
                return [None] * audit_rows(u, ctx, man, rows, cls)
            mc = AUDIT_CLASSES[cls] or ('retirement_audit' if man.endswith('.json') else 'quarantine_manifest')
            u.add_output(man, mc)
            return []
        unit_fn(f'audit_{cls}_{os.path.splitext(os.path.basename(man))[0]}', body,
                describe=lambda r: f'{len(r)} file(s) with problems' if r else 'audited')
