"""`archive retire-excluded`: no product of an excluded (LST, date, scope) left in the current generation Run (deep
requirement (c), DESIGN.md 8; FINDING-EXCLUDED-NIGHTS-ARCHIVE-PRODUCTS-NOT-RETIRED-20261012, ledger 2278: 1,211 files /
29.9 GB in 44 keys on 2026-10-12, "dry run - awaiting Gregg go"). No production code does this today; the 2026-10-12 dry
run (ops/excluded_quarantine_20261012/DRYRUN_excluded_archive_products.csv) was a refactor-chat measurement, and its row
format is kept here so a new plan compares to it row by row.

Deep tier
  scan      every <root>/<LST>/<date>/<run>/<scope>/<S>/<cadence>/ (cadences archive_retire.CADENCES = deep, 10min) of an
            excluded (LST, scope, date) under core.exclusions (the only reader of data_decisions.csv; same rule as
            coadd_full_pipeline.is_excluded) - is_excluded is asked per (LST, date, scope) directory before listing it.
  plan      (default) per excluded key a unit with that key's rows; a summary unit writes the combined plan
            DRYRUN_excluded_archive_products_<UTC>.csv (columns archive_retire.MANIFEST_COLUMNS) and the retirement audit.
  apply     --apply --apply-ruling "<Gregg's words, UTC, ledger ID>" --manifest <the approved plan>: the plan re-derived now
            must equal the approved file (G11, path set and sizes, checksums where it has them), then each file is renamed to
            <root>/Run_EXCLUDED_QUARANTINE_20261012/<same relative path> (common.quarantine_move), one APPLIED row per move
            (+ md5, moved_at); then the retirement audit re-scans: zero files of the keys left (G16, never waivable).
Sub-LST and snapshot tiers: report only (no ruling covers their archive retirement): one unit per night cell lists the
files of excluded (LST, scope, date) under archive_retire.REPORT_TREES; nothing moves.
"""
import os
import re

from exopipe.phase3.core.record import NoInput, Refusal
from exopipe.phase3.lib.archive import common as K

_DATE = re.compile(r'^\d{4}-\d{2}-\d{2}$')
_LSTS = tuple(f'{h:02d}h' for h in range(24))


def _r():
    return K.sec('archive_retire')


def exclusions():
    from exopipe.phase3.core import exclusions as E
    return E.load()


def kind_of(name):
    if _r().KIND_RULE != 'msub-token':
        raise Refusal(f'archive_retire.KIND_RULE {_r().KIND_RULE!r} not implemented')
    return 'search_frame' if K.sec().SEARCH_FRAME_TOKEN in os.path.basename(name) else 'epoch_image'


def key_name(key):
    lst, date, scope = key
    return f'{lst}_{scope}_{date}'


def scan_excluded(archive_root, run_name, cadences, ex, lsts=None, keep=None):
    """{(lst, date, scope): [file paths]} of every excluded key under the deep generation Run. ``lsts`` limits the LSTs;
    ``keep(lst, date, scope)`` (optional) filters keys."""
    out = {}
    for lst in (lsts or _LSTS):
        d_l = os.path.join(archive_root, lst)
        if not os.path.isdir(d_l):
            continue
        for date in sorted(os.listdir(d_l)):
            if not _DATE.match(date):
                continue
            d_r = os.path.join(d_l, date, run_name)
            if not os.path.isdir(d_r):
                continue
            for scope in sorted(os.listdir(d_r)):
                if keep is not None and not keep(lst, date, scope):
                    continue
                if not ex.is_excluded(lst, scope, date):
                    continue
                files = []
                d_s = os.path.join(d_r, scope)
                if not os.path.isdir(d_s):
                    continue
                for st in sorted(os.listdir(d_s)):
                    for cad in cadences:
                        d = os.path.join(d_s, st, cad)
                        if os.path.isdir(d):
                            files += sorted(e.path for e in os.scandir(d) if e.is_file(follow_symlinks=False))
                if files:
                    out[(lst, date, scope)] = files
    return out


def rows_for(key, files, archive_root, listed=None):
    lst, date, scope = key
    qroot = os.path.join(archive_root, _r().EXCLUDED_QUARANTINE_RUN)
    listed = listed or K.utc()
    return [{'lst': lst, 'date': date, 'scope': scope, 'kind': kind_of(p), 'size_bytes': os.path.getsize(p), 'path': p,
             'quarantine_path': os.path.join(qroot, os.path.relpath(p, archive_root)), 'listed_utc': listed}
            for p in files]


def universe_filter(args):
    """keep(lst, date, scope) from --cells (deep cells / patterns), --nights, --exclude-cells; and the LST list (--lsts,
    or the LSTs of --cells)."""
    from exopipe.phase3.core import cells as CC
    cells = set(CC.expand_cells(args.cells)) if getattr(args, 'cells', None) else None
    excl = set(CC.expand_cells(args.exclude_cells)) if getattr(args, 'exclude_cells', None) else set()
    nights = set(CC.expand(args.nights)) if getattr(args, 'nights', None) else None
    lsts = list(CC.expand(args.lsts, universe=CC.LSTS)) if getattr(args, 'lsts', None) else None
    if cells and not lsts:
        lsts = sorted({c.split('_', 1)[0] for c in cells})

    def keep(lst, date, scope):
        c = f'{lst}_{scope}'
        return ((cells is None or c in cells) and c not in excl and (nights is None or date in nights))
    return keep, lsts, {'cells': sorted(cells) if cells else None, 'nights': sorted(nights) if nights else None,
                        'lsts': lsts, 'exclude_cells': sorted(excl) or None}


# ---------------------------------------------------------------------------------------------------------------------
# deep: plan / apply / audit
# ---------------------------------------------------------------------------------------------------------------------

def run_deep(ctx, args, run, unit_fn):
    ex = exclusions()                               # G1: a missing / unreadable data_decisions.csv refuses (C-09)
    keep, lsts, filt = universe_filter(args)
    cad = tuple(_r().CADENCES if not args.cadences else [c.strip() for c in args.cadences.split(',') if c.strip()])
    found = scan_excluded(ctx.archive_root, ctx.gen_run, cad, ex, lsts=lsts, keep=keep)
    listed = K.utc()
    plan = {k: rows_for(k, v, ctx.archive_root, listed) for k, v in sorted(found.items())}
    n = sum(len(v) for v in plan.values())
    gb = sum(int(r['size_bytes']) for v in plan.values() for r in v) / 1e9
    print(f'retire-excluded {ctx.tier}: {n} files, {gb:.2f} GB in {len(plan)} excluded (LST, date, scope) keys under '
          f'{ctx.archive_root}/<LST>/<date>/{ctx.gen_run} (cadences {",".join(cad)}; exclusions sha1 '
          f'{ex.fingerprint()["sha1"][:12]}, {ex.fingerprint()["n_exclude_rows"]} exclude rows)')
    approved = None
    if ctx.apply and not ctx.dry:
        ruling = K.require_ruling(args.apply_ruling, 'retire-excluded')
        if not args.manifest:
            raise Refusal('G11: retire-excluded --apply needs --manifest <the approved plan> (the dry run Gregg approved)')
        _, man_rows = K.read_csv(args.manifest)
        planned = {r['path']: r['size_bytes'] for v in plan.values() for r in v}
        man = [r for r in man_rows if keep(r['lst'], r['date'], r['scope']) and (not lsts or r['lst'] in lsts)]
        diffs = K.compare_to_manifest(planned, man, 'path', 'size_bytes', 'md5' if 'md5' in (man[0] if man else {}) else None)
        if diffs:
            raise Refusal('G11: the plan re-derived now differs from the approved manifest ' + args.manifest + ': '
                          + '; '.join(diffs[:4]) + ' - a fresh plan must be approved')
        ctx.guard_writes([ctx.archive_root, ctx.manifest_dir('retire_excluded')], 'retire-excluded apply')
        approved = ruling
    kdir = os.path.join(ctx.manifest_dir('retire_excluded'), ctx.stamp)
    applied = os.path.join(kdir, f'APPLIED_excluded_archive_products_{ctx.stamp}.csv')
    cols_applied = tuple(_r().MANIFEST_COLUMNS) + tuple(_r().APPLIED_EXTRA_COLUMNS)
    if not plan:
        print('retire-excluded: no file of an excluded key under the scanned trees')
    for key, rows in plan.items():
        def body(u, key=key, rows=rows):
            u.meta.update(key=key_name(key), n_files=len(rows), bytes=sum(int(r['size_bytes']) for r in rows))
            if approved:
                u.meta['ruling_ids'] = approved
            u.add_input(ex.path, extra=ex.fingerprint())
            if ctx.dry:
                return rows
            for r in rows:
                if os.path.lexists(r['quarantine_path']):
                    raise Refusal(f"G4: quarantine destination exists: {r['quarantine_path']}")
            if not ctx.apply:
                p = os.path.join(kdir, 'keys', f'{key_name(key)}.csv')
                K.write_csv(p, _r().MANIFEST_COLUMNS, rows)
                u.add_output(p, 'quarantine_manifest')
                return rows
            t0 = K.utc()
            paths = set()
            with K.CsvAppender(applied, cols_applied) as w:
                for r in rows:
                    b, m, t = K.quarantine_move(r['path'], r['quarantine_path'], ctx.algo, what='retire-excluded')
                    w.write(dict(r, size_bytes=b, md5=m, moved_at=t))
                    paths.add(r['path'])
            left = scan_excluded(ctx.archive_root, ctx.gen_run, cad, ex, lsts=[key[0]],
                                 keep=lambda l, d, s, key=key: (l, d, s) == key)
            u.check('retirement_audit.no_excluded_product_left', not left, sum(len(v) for v in left.values()),
                    f'{sum(len(v) for v in left.values())} file(s) of {key_name(key)} left after the moves')
            u.add_output(applied, 'quarantine_manifest', paths=sorted(paths))
            return rows
        unit_fn(f'retire-excluded_{key_name(key)}', body, describe=lambda rows: f'{len(rows)} file(s), '
                f'{sum(int(r["size_bytes"]) for r in rows) / 1e9:.2f} GB')

    def summary(u):
        u.meta.update(n_keys=len(plan), n_files=n, gb=round(gb, 3), filter=filt, cadences=cad)
        u.add_input(ex.path, extra=ex.fingerprint())
        if ctx.dry:
            return plan
        rows = [r for v in plan.values() for r in v]
        state = 'applied' if ctx.apply else 'plan'
        man = os.path.join(kdir, f'DRYRUN_excluded_archive_products_{ctx.stamp}.csv')
        if not ctx.apply:
            K.write_csv(man, _r().MANIFEST_COLUMNS, rows)
            u.add_output(man, 'quarantine_manifest')
        rec = {'kind': 'excluded', 'state': state, 'tier': ctx.tier, 'archive_root': ctx.archive_root, 'run': ctx.gen_run,
               'cadences': list(cad), 'filter': filt, 'keys': [list(k) for k in plan], 'paths': [r['path'] for r in rows],
               'manifest': applied if ctx.apply else man, 'approved_manifest': args.manifest if ctx.apply else None,
               'exclusions': ex.fingerprint(), 'n_files': n, 'bytes': sum(int(r['size_bytes']) for r in rows),
               'written_utc': K.utc()}
        a = os.path.join(kdir, f'retirement_audit_{state}_{ctx.stamp}.json')
        K.write_json(a, rec)
        u.add_output(a, 'retirement_audit')
        if not plan:
            u.note('no file of an excluded key under the scanned trees: the generation Run is clean')
        return plan
    unit_fn(f'retire-excluded_{ctx.tier}', summary, describe=lambda p: f'{n} file(s) in {len(p)} key(s), {gb:.2f} GB')


# ---------------------------------------------------------------------------------------------------------------------
# sub-LST / snapshot: report only
# ---------------------------------------------------------------------------------------------------------------------

_SCOPE_TOK = re.compile(r'(?:^|[_/.-])(\d{2}MHz|Red|Green|Blue|Full)(?=$|[_/.-])')
_CELL_LIKE = re.compile(r'\d\dh_\d{4}-\d{2}-\d{2}')
_LST_LIKE = re.compile(r'^\d\dh(?:$|_)')


def report_scan(roots, cell, max_depth=3):
    """[(path, scope or None)] of every file of the night cell <LST>_<date> under the report roots: a directory whose name
    holds the cell token is taken whole; other directories are entered only to depth ``max_depth`` and only when they are
    not another LST's or another night's."""
    lst = cell.split('_', 1)[0]
    out = []
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        stack = [(root, 0, False)]
        while stack:
            d, depth, whole = stack.pop()
            try:
                ents = list(os.scandir(d))
            except OSError:
                continue
            for e in ents:
                name = e.name
                if e.is_dir(follow_symlinks=False):
                    if whole or cell in name:
                        stack.append((e.path, depth + 1, True))
                    elif depth + 1 < max_depth and not _CELL_LIKE.search(name) and \
                            (not _LST_LIKE.match(name) or name.startswith(lst)):
                        stack.append((e.path, depth + 1, False))
                elif e.is_file(follow_symlinks=False) and (whole or cell in name):
                    rel = os.path.relpath(e.path, root)
                    m = _SCOPE_TOK.search(rel.replace(cell, ''))
                    out.append((e.path, m.group(1) if m else None))
    return sorted(out)


def report_rows(tier, cell, ex):
    lst, date = cell.split('_', 1)
    roots = [K.resolve_root(x) for x in _r().REPORT_TREES.get(tier, ())]
    rows = []
    listed = K.utc()
    for p, scope in report_scan(roots, f'{lst}_{date}'):
        bad = ex.is_excluded(lst, scope, date) if scope else ex.is_whole_night_excluded(lst, date)
        if bad:
            rows.append({'lst': lst, 'date': date, 'scope': scope or '(night)', 'kind': kind_of(p),
                         'size_bytes': os.path.getsize(p), 'path': p, 'quarantine_path': '', 'listed_utc': listed})
    return roots, rows


def run_report(ctx, args, run, unit_fn):
    from exopipe.phase3.core import cells as CC
    if not args.cells and not getattr(args, 'epochs', None):
        raise ValueError(f'retire-excluded --tier {ctx.tier} (report only) needs --cells <LST>_<YYYY-MM-DD> night cells')
    items = CC.expand(args.cells or args.epochs, kind='night cell')
    ex = exclusions()
    for cell in items:
        lst, date, sb = CC.parse_night_cell(cell)

        def body(u, cell=f'{lst}_{date}'):
            u.add_input(ex.path, extra=ex.fingerprint())
            roots, rows = report_rows(ctx.tier, cell, ex)
            u.meta.update(cell=cell, roots=roots, n_files=len(rows), bytes=sum(int(r['size_bytes']) for r in rows))
            u.note(f'REPORT ONLY (no ruling covers {ctx.tier} archive retirement): {len(rows)} file(s) of excluded '
                   f'(LST, scope, date) under {", ".join(roots)}')
            if ctx.dry:
                return rows
            p = os.path.join(ctx.manifest_dir('retire_excluded_report'), f'{ctx.tier}_{cell}_{ctx.stamp}.csv')
            K.write_csv(p, _r().MANIFEST_COLUMNS, rows)
            u.add_output(p, 'csv_table')
            a = os.path.join(ctx.manifest_dir('retire_excluded_report'), f'{ctx.tier}_{cell}_{ctx.stamp}.json')
            K.write_json(a, {'kind': 'report', 'state': 'report', 'tier': ctx.tier, 'cell': cell, 'roots': roots,
                             'paths': [r['path'] for r in rows], 'manifest': p, 'exclusions': ex.fingerprint(),
                             'written_utc': K.utc()})
            u.add_output(a, 'retirement_audit')
            return rows
        unit_fn(f'retire-excluded_{ctx.tier}_{lst}_{date}', body, describe=lambda rows: f'{len(rows)} file(s) (report only)')


def run_action(ctx, args, run, unit_fn):
    if ctx.tier == 'deep':
        return run_deep(ctx, args, run, unit_fn)
    if ctx.apply:
        raise Refusal(f'retire-excluded --tier {ctx.tier}: report only - no ruling covers {ctx.tier} archive retirement '
                      f'(archive.md section 9 / 12 item 6); --apply refused')
    return run_report(ctx, args, run, unit_fn)


__all__ = ['scan_excluded', 'rows_for', 'report_scan', 'report_rows', 'run_action', 'NoInput']
