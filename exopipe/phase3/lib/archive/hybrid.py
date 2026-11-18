"""`archive push --class hybrid` (PR-5b; spec plan/specs/hybrid.md 2.7, D2, D6, D7, HG12): the hybrid masters of a hybrid run's
--out into the hybrid archive run.

Units: every <staging>/<LST>_<scope>/<w>_tilefill.json whose tilefill unit record (<staging>/_records/hybrid/tilefill.<LST>_<scope>_<w>
.json) is done (--lsts narrows). Files of a unit, under their PRODNAME names into <root>/<hybrid.ARCHIVE_HYBRID relative to the archive
root>/<LST>/<scope>/ (production notaper_largescale.archive_hybrid_unit 1637, then the 11-11 / 11-13 tile swaps):
  plain, pbcorr, cleanres   the filled master, its beam-corrected sibling and the renamed CLEAN residual (the tilefill outputs);
  recon, recon_pbcorr       a unit filled from its own rebuild (source work), not Full: the NoTaper reconstruction and its split
                            beam-corrected sibling (<w>_lsrecon(_wb).json);
  pbdiffuse, pb             source work: the cell's diffuse PB map (the tilefill stage wrote it; Full's the full stage) and the
                            weighting's full PB map of the masters archive (hybrid.ARCHIVE_MASTERS).
A pinned unit (filled from the installed masters' own unfilled sources) pushes only the tilefill outputs.

Per file: keep (the installed file of the kind has the same name and the same checksum: byte-identical; a file with the same pixels
and another header is replaced - its beam, noise and stack cards are part of the product, and the 11-13 lost-sky swap replaced five
such files whose restoring beam and NSTACK had changed), publish (none installed), replace (D2: the installed file of the kind - same name or another night
span - is renamed into <root>/<archive_push.HYBRID_QUARANTINE_PREFIX><round>/<LST>/<scope>/ with a row in that Run's
MANIFEST_HYBRID_QUARANTINE.csv, then the new file is renamed in, back to back), retire (source work: an installed product kind the
rebuild no longer makes is quarantined the same way, without a successor).

Gates (a unit refuses whole; nothing of it is copied): the tilefill unit done and its record PASS; its admitted nights = the record's;
no night of the unit now excluded (steps.hybrid.excluded_in_unit, per member for wideband and Full); every source's checksum = the one
its stage recorded; a product's PRODNAME = its file name and a phase3 name; HG12 / D6 every PB map's header conformant
(pb_apply.pb_map_header_problems; a diffuse map also pbdiffuse_header_problems); D7 a replacement holds every card key of the file it
replaces except archive_push.HYBRID_RETIRED_KEYS[kind]; at most one installed file per kind; a destination that exists is the
installed file of its kind; G4 quarantine destinations free; G19 free space; G2 production writes. All copies are verified (.part,
checksum) before the first move. Content audit after the swap: class hybrid_archive_copy on every file written.
"""
import glob
import json
import os
import time

from exopipe.phase3.core.record import NoInput, Refusal
from exopipe.phase3.lib.archive import common as K

PRODUCT_KINDS = ('plain', 'pbcorr', 'cleanres', 'recon', 'recon_pbcorr')
PB_KINDS = ('pbdiffuse', 'pb')
QUARANTINE_MANIFEST = 'MANIFEST_HYBRID_QUARANTINE.csv'


def _p():
    return K.sec('archive_push')


def _json(path):
    with open(path) as f:
        return json.load(f)


def header(path):
    from astropy.io import fits
    return fits.getheader(path, memmap=False)


def kind_of(path):
    """A file's kind from its name alone (the content tests without a context)."""
    b = os.path.basename(path)
    if b.startswith('PBdiffuse_deep_'):
        return 'pbdiffuse'
    if b.startswith('PB_deep_'):
        return 'pb'
    return 'product'


def header_problems(h, name, kind):
    """Why a file's header may not enter the hybrid archive under name (empty = conformant): a product's PRODNAME is its file name
    and a phase3 name; a PB map: pb_map_header_problems (HG12), a diffuse map also pbdiffuse_header_problems (D6)."""
    from exopipe.phase3.core.products import parse_name
    from exopipe.phase3.lib.masters import pb_apply as PBA
    if kind == 'pbdiffuse':
        return PBA.pbdiffuse_header_problems(h)
    if kind == 'pb':
        return PBA.pb_map_header_problems(h)
    pn = h.get('PRODNAME')
    if pn != name:
        return [f'PRODNAME {pn!r} is not the file name {name}']
    try:
        ok = (parse_name(pn) or {}).get('grammar') == 'phase3'
    except ValueError:
        ok = False
    return [] if ok else [f'PRODNAME {pn!r} is not a phase3 name']


def missing_keys(new_h, old_h, kind):
    """D7: card keys of the replaced file's header (commentary aside) absent from the new one, less the keys retired for the kind."""
    retired = set(_p().HYBRID_RETIRED_KEYS.get(kind, ()))
    have = set(new_h.keys())
    return sorted(k for k in set(old_h.keys()) if k not in ('HISTORY', 'COMMENT', '') and k not in have and k not in retired)


def archive_dir(ctx, lst=None, scope=None):
    """<archive root>/<hybrid.ARCHIVE_HYBRID relative to paths.PHASE3_ARCHIVE_ROOT>[/<LST>/<scope>]."""
    C = K.cfg()
    rel = os.path.relpath(C.hybrid.ARCHIVE_HYBRID, C.paths.PHASE3_ARCHIVE_ROOT)
    if rel.startswith('..'):
        raise Refusal(f'hybrid.ARCHIVE_HYBRID {C.hybrid.ARCHIVE_HYBRID} is not under paths.PHASE3_ARCHIVE_ROOT')
    d = os.path.join(ctx.archive_root, rel)
    return os.path.join(d, lst, scope) if lst else d


def quarantine_dir(ctx, rnd):
    return os.path.join(ctx.archive_root, f'{_p().HYBRID_QUARANTINE_PREFIX}{rnd}')


def hybrid_units(args, staging):
    """[(LST, scope, weighting)] of the tilefill records under the staging root (--lsts narrows)."""
    from exopipe.phase3.core import cells as CC
    if not staging or not os.path.isdir(staging):
        raise ValueError(f'push --class hybrid: --staging-root must be the hybrid step\'s --out (the tilefill records are read '
                         f'there); got {staging!r}')
    lsts = CC.expand(args.lsts, universe=CC.LSTS) if getattr(args, 'lsts', None) else CC.LSTS
    units = []
    for lst in lsts:
        for d in sorted(glob.glob(os.path.join(staging, f'{lst}_*'))):
            if not os.path.isdir(d):
                continue
            scope = os.path.basename(d)[len(lst) + 1:]
            for p in sorted(glob.glob(os.path.join(d, '*_tilefill.json'))):
                units.append((lst, scope, os.path.basename(p)[:-len('_tilefill.json')]))
    return units


def unit_items(staging, lst, scope, w):
    """(items [{kind, src, name, md5 recorded or None}], record, reads) of a done tilefill unit; Refusal otherwise."""
    from exopipe.phase3.config import CFG
    from exopipe.phase3.lib.masters import pb_apply as PBA
    wd = os.path.join(staging, f'{lst}_{scope}')
    tp = os.path.join(wd, f'{w}_tilefill.json')
    up = os.path.join(staging, '_records', 'hybrid', f'tilefill.{lst}_{scope}_{w}.json')
    if not os.path.exists(up):
        raise Refusal(f'{up}: no tilefill unit record - nothing of {lst} {scope} {w} is pushed')
    ur = _json(up)
    if ur.get('status') != 'done':
        raise Refusal(f'{up}: the tilefill unit is {ur.get("status")!r}, not done - nothing of it is pushed')
    tf = _json(tp)
    if tf.get('result') != 'PASS':
        raise Refusal(f'{tp}: result {tf.get("result")!r} - nothing of it is pushed')
    if 'nights' not in tf or 'nights_from' not in tf:
        raise Refusal(f'{tp}: made before the tilefill record held its nights (PR-5b); run --stages tilefill again')
    if sorted(tf['nights']) != sorted(ur.get('admitted') or ()):
        raise Refusal(f'{tp}: nights differ from the unit record\'s admitted nights ({up}) - REFUSING')
    reads = [tp, up]
    items = [dict(kind=k, src=tf[k]['out'], name=os.path.basename(tf[k]['out']), md5=tf[k]['out_md5']) for k in ('plain', 'pbcorr')]
    if tf.get('cleanres'):
        items.append(dict(kind='cleanres', src=tf['cleanres']['out'], name=tf['cleanres']['name'], md5=tf['cleanres']['out_md5']))
    if tf.get('source') == 'work':
        if scope != 'Full':
            rp = os.path.join(wd, f'{w}_lsrecon_wb.json' if scope in CFG.hybrid.WIDEBAND_MEMBERS else f'{w}_lsrecon.json')
            rr = _json(rp)
            reads.append(rp)
            for k, key in (('recon', 'product'), ('recon_pbcorr', 'product_pbcorr')):
                items.append(dict(kind=k, src=rr[key]['path'], name=rr[key]['prodname'], md5=rr[key].get('md5')))
        items.append(dict(kind='pbdiffuse', src=os.path.join(wd, f'PBdiffuse_deep_{scope}.fits'), name=f'PBdiffuse_deep_{scope}.fits',
                          md5=None))
        pbf = os.path.join(CFG.hybrid.ARCHIVE_MASTERS, lst, scope, PBA.deep_pb_name(scope, w))
        items.append(dict(kind='pb', src=pbf, name=os.path.basename(pbf), md5=None))
    elif tf.get('source') != 'pinned':
        raise Refusal(f'{tp}: source {tf.get("source")!r} is neither work nor pinned - REFUSING')
    return items, tf, reads


def hybrid_plan(ctx, staging, lst, scope, w, checksum=True):
    """(rows, reads, notes) of one unit: a row per file (unit, kind, src, dst, bytes, md5, action, replaces, written_at) plus a
    'retire' row per installed product kind a work-source unit no longer makes. Refusal on any gate (module docstring)."""
    from exopipe.phase3.core import exclusions
    from exopipe.phase3.lib.hybrid.tilefill import TilefillRefusal
    from exopipe.phase3.steps import hybrid as SH
    unit = f'{lst}_{scope}_{w}'
    items, tf, reads = unit_items(staging, lst, scope, w)
    excl = exclusions.for_tier('deep', fresh=True)
    try:
        bad, r2 = SH.excluded_in_unit(excl, os.path.dirname(os.path.dirname(tf['nights_from'])), lst, scope, w, tf['nights'])
    except TilefillRefusal as e:
        raise Refusal(str(e))
    reads += r2 + ([excl.path] if getattr(excl, 'path', None) else [])
    if bad:
        raise Refusal(f'{unit}: night(s) now excluded in data_decisions.csv {bad} - never pushed; rebuild the unit')
    dd = archive_dir(ctx, lst, scope)
    inst = SH._installed(lst, scope, w, root=archive_dir(ctx))
    many = {k: v for k, v in inst.items() if len(v) > 1}
    if many:
        raise Refusal(f'{unit}: more than one installed file of a kind in {dd}: {many} - which one a new file replaces is ambiguous')
    rows, notes = [], []
    for it in items:
        kind, src, name = it['kind'], it['src'], it['name']
        if not os.path.exists(src):
            raise Refusal(f'{unit} {kind}: {src} does not exist')
        if K.intermediate(name):
            raise Refusal(f'{unit} {kind}: {name} is an intermediate name (archive_push.INTERMEDIATE_PATTERN)')
        h = header(src)
        hp = header_problems(h, name, kind)
        if hp:
            raise Refusal(f'{"HG12" if kind in PB_KINDS else "name"}: {src}: {"; ".join(hp)} - REFUSING the unit')
        m = K.checksum(src, ctx.algo) if checksum else (it['md5'] or '')
        if checksum and it['md5'] and m != it['md5']:
            raise Refusal(f'{unit} {kind}: {src} changed since its stage recorded it ({ctx.algo} {m}, recorded {it["md5"]})')
        dst = os.path.join(dd, name)
        old = (inst[kind][0] if inst[kind] else '') if kind in PRODUCT_KINDS else (dst if os.path.lexists(dst) else '')
        if os.path.lexists(dst) and dst != old:
            raise Refusal(f'{unit} {kind}: {dst} exists and is not the installed {kind} file ({old or "none"}) - REFUSING')
        if not old:
            act = 'publish'
        elif old != dst:
            act = 'replace'                                              # another night span: the new file supersedes it
        elif not checksum:
            act = 'keep-or-replace'
        elif K.checksum(dst, ctx.algo) == m:
            act = 'keep'                                                 # byte-identical
        else:                                                            # same name, other bytes (pixels or header)
            act = 'replace'
        if act in ('replace', 'keep-or-replace'):
            miss = missing_keys(h, header(old), kind)
            if miss:
                raise Refusal(f'D7: {src} lacks card key(s) {miss[:10]} of the file it replaces {old} - REFUSING the unit')
        keep = act == 'keep'
        rows.append(dict(unit=unit, kind=kind, src=src, dst=dst, bytes=os.path.getsize(dst if keep else src),
                         md5=(K.checksum(dst, ctx.algo) if keep else m), action=act, replaces=old if not keep else '', written_at=''))
    if tf.get('source') == 'work':
        made = {it['kind'] for it in items}
        for k in PRODUCT_KINDS:
            if k not in made and inst[k]:
                rows.append(dict(unit=unit, kind=k, src='', dst='', bytes=os.path.getsize(inst[k][0]), md5='', action='retire',
                                 replaces=inst[k][0], written_at=''))
                notes.append(f'{k} {os.path.basename(inst[k][0])}: the rebuild makes no {k} - quarantined without a successor')
    return rows, reads, notes


def _qpath(ctx, rnd, old):
    return os.path.join(quarantine_dir(ctx, rnd), os.path.relpath(old, archive_dir(ctx)))


def apply_unit(ctx, u, rows, rnd):
    """D2 swap of one planned unit: verified copies first, then per file the installed one into the round quarantine (manifest
    row) and the new one renamed in, back to back; the applied push manifest; the content audit."""
    unit = rows[0]['unit']
    todo = [r for r in rows if r['action'] in ('publish', 'replace', 'retire')]
    qdir = quarantine_dir(ctx, rnd)
    qman = os.path.join(qdir, QUARANTINE_MANIFEST)
    pman = os.path.join(archive_dir(ctx), f'MANIFEST_{ctx.stamp}.csv')
    ctx.guard_writes([archive_dir(ctx), qdir, qman, pman], f'push --class hybrid {unit}')
    for r in todo:
        if r['action'] in ('replace', 'retire') and os.path.lexists(_qpath(ctx, rnd, r['replaces'])):
            raise Refusal(f'G4: quarantine destination exists: {_qpath(ctx, rnd, r["replaces"])} (choose another --round)')
    ok, free = K.free_space_ok(ctx.archive_root, K.sec().FREE_SPACE_FLOOR_GB)
    if not ok:
        raise Refusal(f'G19: {free} GB free < archive.FREE_SPACE_FLOOR_GB')
    parts = {}
    try:
        for r in todo:
            if r['action'] in ('publish', 'replace'):
                parts[r['dst']] = K.verified_copy(r['src'], r['dst'], ctx.algo, want=r['md5'], preserve=True)
    except Refusal:
        for part, _m in parts.values():
            if os.path.exists(part):
                os.remove(part)
        raise
    t0 = K.utc()
    moved, written = [], []
    for r in todo:
        if r['action'] in ('replace', 'retire'):
            q = _qpath(ctx, rnd, r['replaces'])
            with K.CsvAppender(qman, _p().HYBRID_QUARANTINE_COLUMNS) as wq:
                b, m, t = K.quarantine_move(r['replaces'], q, ctx.algo, what=f'push hybrid {unit} {r["action"]}')
                wq.write(dict(unit=unit, kind=r['kind'], path=r['replaces'], dest=q, bytes=b, md5=m,
                              replaced_by=r['dst'] if r['action'] == 'replace' else '', moved_at=t))
            moved.append((r, q))
        if r['action'] in ('publish', 'replace'):
            K.commit_part(parts[r['dst']][0], r['dst'], replace=False)     # the installed file left first: nothing is overwritten
            r['written_at'] = K.utc()
            written.append(r)
    with K.CsvAppender(pman, _p().HYBRID_PUSH_COLUMNS) as wp:
        for r in rows:
            if r['action'] != 'retire':
                wp.write(r)
    if moved:
        u.add_output(qman, 'quarantine_manifest', since=t0, paths=[r['replaces'] for r, _q in moved])
    u.add_output(pman, 'archive_manifest', paths=[r['dst'] for r in rows if r['action'] != 'retire'])
    qof = {r['dst']: q for r, q in moved if r['action'] == 'replace'}
    for r in written:
        u.add_output(r['dst'], 'hybrid_archive_copy', kind=r['kind'], md5=r['md5'], replaces=qof.get(r['dst']), algo=ctx.algo)
    return moved, written


def run_hybrid(ctx, args, run, unit_fn):
    from exopipe.phase3.lib.archive.push import class_cfg
    c = class_cfg('hybrid')
    if ctx.tier not in c.get('tiers', ('deep',)):
        raise NoInput(f'push --class hybrid --tier {ctx.tier}: hybrid masters are deep-tier products; nothing to push')
    staging = args.staging_root
    rnd = args.round or time.strftime('%Y%m%d', time.gmtime())
    stages = args._stages
    for lst, scope, w in hybrid_units(args, staging):
        def body(u, lst=lst, scope=scope, w=w):
            u.meta.update(push_class='hybrid', lst=lst, scope=scope, weighting=w, staging_root=staging,
                          archive_root=ctx.archive_root, archive_dir=archive_dir(ctx, lst, scope), round=rnd)
            rows, reads, notes = hybrid_plan(ctx, staging, lst, scope, w, checksum=not ctx.dry)
            for p in reads:
                u.add_input(p)
            for n in notes:
                u.note(n)
            cnt = {k: sum(1 for r in rows if r['action'] == k) for k in ('keep', 'publish', 'replace', 'retire', 'keep-or-replace')}
            u.meta['counts'] = cnt
            if ctx.dry:
                return rows
            if not ctx.apply or 'transfer' not in stages:
                man = os.path.join(ctx.manifest_dir('push_hybrid'), f'{lst}_{scope}_{w}_{ctx.stamp}.csv')
                K.write_csv(man, _p().HYBRID_PUSH_COLUMNS, rows)
                u.add_output(man, 'csv_table', required_columns=('unit', 'kind', 'src', 'dst', 'action', 'replaces'))
                return rows
            if any(r['action'] in ('publish', 'replace', 'retire') for r in rows):
                apply_unit(ctx, u, rows, rnd)
            return rows
        unit_fn(f'push_hybrid_{lst}_{scope}_{w}', body, describe=lambda rows: ', '.join(
            f'{k} {sum(1 for r in rows if r["action"] == k)}' for k in ('keep', 'publish', 'replace', 'retire', 'keep-or-replace')
            if any(r['action'] == k for r in rows)) or '0 files')
