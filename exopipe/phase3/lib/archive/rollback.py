"""`archive rollback`: put quarantined files back from their quarantine manifest (ops/membership_rebuild_20261007/
restore_refused_frames.py, R-RESTORE-REFUSED-CELL-FRAMES-APPROVED ledger 2093, generalised from its hard-coded nine
(cell, Stokes) sets and its one occupant rule).

Per (cell, Stokes): every row of the manifest (default archive.FRAME_QUARANTINE_MANIFEST_DIR/<cell>_<S>.csv) moved at or
after --since (restore_refused_frames.py:33-35):
  - the quarantined copy must exist and its md5 must equal the manifest's (:45-49);
  - the original path must be empty, OR (--occupant-quarantine RUN given) hold a file written at or after --since, which is
    first renamed to <root>/<RUN>/<same relative path> (the 4 Stokes V cells of 2026-10-08: :37-44, :61-65); the product must
    not be in its directory under another name either (common.occupant: the conformant-names rename of 2026-10-16 renames a
    product after its quarantine manifest recorded the old path) - never restored beside it;
  - any problem refuses the whole unit before anything moves (gate G15; :52-58 'REFUSED: problems found - nothing moved');
  - then the copy is renamed back (same filesystem) and inode + size re-checked (:66-68).
The plan (no --apply) checks everything and writes the rows it would restore (action 'would restore' / 'would swap'). The
apply (archive.RULING_REQUIRED) writes one record row per move as it goes (restore_refused_frames.py wrote its record only
at the end: review item 23 - a pure fix, same rows).
"""
import os
import time

from exopipe.phase3.core.record import NoInput, Refusal
from exopipe.phase3.lib.archive import common as K


def _since(text):
    """--since: 'YYYY-MM-DDTHH:MM:SS[Z]' or an epoch -> (iso without Z as production compares, epoch s)."""
    t = str(text).strip()
    if t.replace('.', '', 1).isdigit():
        e = float(t)
        return time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(e)), e
    t = t.rstrip('Z')
    import calendar
    e = calendar.timegm(time.strptime(t, '%Y-%m-%dT%H:%M:%S'))
    return t, float(e)


def plan(ctx, manifest, cell, st, since, occupant_q, checksum=True, ex=None):
    """(rows, problems): the restore rows (restore_refused_frames.py:31-51). A row whose original path is a file of an
    excluded (LST, scope, date) is a problem (G20: exclusion is absolute - a rollback never puts an excluded night's file
    back into the generation Run; restore_refused_frames.py had no such case: its nine sets held no excluded night)."""
    since_iso, since_s = _since(since)
    _, man = K.read_csv(manifest)
    rows, bad = [], []
    for r in man:
        if (r.get('cell') or cell) != cell or (r.get('stokes') or st) != st:
            continue
        if (r.get('moved_at') or '') < since_iso:
            continue
        p, q = r['path'], r['dest']
        key = K.excluded_key(ex, p, ctx.archive_root) if ex is not None else None
        if key:
            bad.append(f'{p}: excluded {key} in data_decisions.csv - never restored (G20, exclusion absolute)')
            continue
        occ = ''
        holder = K.occupant(p)                        # the product at p - under p's name or a later (conformant) one
        if holder and holder != p:
            bad.append(f'{p}: its product is in the archive under another name ({os.path.basename(holder)}; renamed after '
                       f'the quarantine) - never restored beside it')
            continue
        if holder:
            if occupant_q and os.path.getmtime(p) >= since_s:
                occ = os.path.join(ctx.archive_root, occupant_q, os.path.relpath(p, ctx.archive_root))
                if os.path.lexists(occ):
                    bad.append(f'{occ}: retired destination exists')
                    continue
            else:
                bad.append(f'{p}: original path is occupied' + ('' if occupant_q else ' (no --occupant-quarantine)'))
                continue
        if not os.path.exists(q):
            bad.append(f'{q}: quarantined copy missing')
            continue
        m = K.checksum(q, 'md5') if checksum else r['md5']
        if m != r['md5']:
            bad.append(f'{q}: md5 {m} != manifest {r["md5"]}')
            continue
        rows.append(dict(cell=cell, stokes=st, path=p, dest=q, md5=m, bytes=r['bytes'], retired_to=occ,
                         action=('swap' if occ else 'restore') if ctx.apply else ('would swap' if occ else 'would restore'),
                         restored_at='', retired_md5=''))
    return rows, bad


def run_action(ctx, args, run, unit_fn, items, stokes):
    if not args.since:
        raise ValueError('rollback needs --since <UTC YYYY-MM-DDTHH:MM:SSZ or epoch>: manifest rows moved at or after it')
    cols = K.sec('archive_rollback').RESTORE_COLUMNS
    for cell in items:
        for st in stokes:
            def body(u, cell=cell, st=st):
                man = args.manifest or os.path.join(K.expand(K.sec().FRAME_QUARANTINE_MANIFEST_DIR), f'{cell}_{st}.csv')
                if '{cell}' in man or '{stokes}' in man:
                    man = man.format(cell=cell, stokes=st)
                u.meta.update(cell=cell, stokes=st, manifest=man, since=args.since,
                              occupant_quarantine=args.occupant_quarantine)
                if not os.path.exists(man):
                    raise NoInput(f'no quarantine manifest {man}')
                u.add_input(man)
                from exopipe.phase3.core import exclusions
                ex = exclusions.load()                  # G1: missing / unreadable data_decisions.csv refuses (C-09)
                u.add_input(ex.path, extra=ex.fingerprint())
                rows, bad = plan(ctx, man, cell, st, args.since, args.occupant_quarantine, checksum=not ctx.dry, ex=ex)
                u.meta.update(n_rows=len(rows), n_problems=len(bad), problems=bad[:20])
                excl = [b for b in bad if '(G20' in b]
                if excl:
                    raise Refusal(f'G20: REFUSED: {len(excl)} row(s) would put a file of an excluded (LST, scope, date) back '
                                  f'into the generation Run - nothing moved; e.g. {excl[0]}')
                if bad:
                    raise Refusal(f'G15: REFUSED: {len(bad)} problem(s) found - nothing moved; e.g. {bad[0]}')
                if not rows:
                    raise NoInput(f'{cell} {st}: no manifest row moved at or after {args.since}')
                if ctx.dry:
                    return rows
                out = os.path.join(ctx.manifest_dir('rollback'), f'restore_{cell}_{st}_{"applied" if ctx.apply else "plan"}_'
                                   f'{ctx.stamp}.csv')
                if not ctx.apply:
                    K.write_csv(out, cols, rows)
                    u.add_output(out, 'restore_manifest')
                    return rows
                ids = K.require_ruling(args.apply_ruling, f'rollback of {cell} {st}')
                u.meta['ruling_ids'] = ids
                ctx.guard_writes([ctx.archive_root] + [r['path'] for r in rows[:1]], f'rollback of {cell} {st}')
                K.refuse_live_writers(ctx, [(cell, st)], f'rollback of {cell} {st}')
                with K.CsvAppender(out, cols) as w:
                    for o in rows:
                        try:
                            if o['retired_to']:
                                b, m, t = K.quarantine_move(o['path'], o['retired_to'], 'md5', what='rollback occupant')
                                o['retired_md5'] = m
                            s0 = os.stat(o['dest'])
                            if os.path.lexists(o['path']):
                                raise Refusal(f"G15: {o['path']} became occupied during the rollback - STOPPED")
                            os.rename(o['dest'], o['path'])
                            s1 = os.stat(o['path'])
                            if (s1.st_ino, s1.st_size) != (s0.st_ino, s0.st_size):
                                os.rename(o['path'], o['dest'])
                                raise Refusal(f"G5: {o['path']} not verified after the rename; moved back")
                        except BaseException:
                            if o['retired_md5']:          # the occupant moved but the restore did not: record that move
                                w.write(dict(o, action='occupant moved, restore failed'))
                            raise
                        o['restored_at'] = K.utc()
                        w.write(o)
                u.add_output(out, 'restore_manifest')
                return rows
            unit_fn(f'rollback_{cell}_{st}', body, describe=lambda rows: f'{len(rows)} file(s) to restore '
                    f'({sum(1 for r in rows if r["retired_to"])} swaps)')
