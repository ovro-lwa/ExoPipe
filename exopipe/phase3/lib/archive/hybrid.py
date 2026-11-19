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

Units the work run did not finish (refused_unit; steps.hybrid.work_states / unit_state over every unit record of the staging root,
--lsts narrows): a unit whose stage records include a refused or no-input one (refused; this takes precedence over a done tilefill
record), or one neither done nor refused (error, running, unreadable), or none done through tilefill (unfinished); a unit whose only
record is a stack staged from production is a member input and is not listed. Each is a push unit of its own, so the installed products
of a cell the rebuild did not remake never stay or go silently: unfinished refuses (finish the run or narrow --lsts); refused with
nothing installed refuses as a gap; refused with installed products refuses unless a ruling set hybrid_unit.REFUSED_INSTALLED for it:
keep (a 'keep' row per installed product with its checksum, recorded as the unit's input, noted as kept by ruling) or
quarantine (a 'retire' row per installed product, moved into the round quarantine as below). The PB maps stay: they describe the
beam, not the refused product.

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
    from exopipe.phase3.lib.hybrid.tests_tilefill import missing_card_keys
    return missing_card_keys(new_h, old_h, _p().HYBRID_RETIRED_KEYS.get(kind, ()))


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


def push_lsts(args):
    from exopipe.phase3.core import cells as CC
    return CC.expand(args.lsts, universe=CC.LSTS) if getattr(args, 'lsts', None) else CC.LSTS


def hybrid_units(args, staging):
    """[(LST, scope, weighting)] of the tilefill records under the staging root (--lsts narrows)."""
    if not staging or not os.path.isdir(staging):
        raise ValueError(f'push --class hybrid: --staging-root must be the hybrid step\'s --out (the tilefill records are read '
                         f'there); got {staging!r}')
    lsts = push_lsts(args)
    units = []
    for lst in lsts:
        for d in sorted(glob.glob(os.path.join(staging, f'{lst}_*'))):
            if not os.path.isdir(d):
                continue
            scope = os.path.basename(d)[len(lst) + 1:]
            for p in sorted(glob.glob(os.path.join(d, '*_tilefill.json'))):
                units.append((lst, scope, os.path.basename(p)[:-len('_tilefill.json')]))
    return units


def refused_rows(ctx, lst, scope, w, state, checksum=True):
    """(rows, notes) of a unit the work run did not finish (module docstring); Refusal unless hybrid_unit.REFUSED_INSTALLED rules it.
    state: steps.hybrid.unit_state's (state, stage, status, reason). A kept file's row carries its checksum (checksum=False: the
    dry run, none computed), recorded as the unit's input: the ruling covers that file and no other."""
    from exopipe.phase3.config import CFG
    from exopipe.phase3.steps import hybrid as SH
    unit = f'{lst}_{scope}_{w}'
    st, stage, status, reason = state
    why = f'{st} in the work run at stage {stage} ({status}: {reason[:240]})'
    if st == 'unfinished':
        raise Refusal(f'{unit}: {why} - finish the run, or narrow --lsts, before pushing it - REFUSED')
    inst = {k: v for k, v in SH._installed(lst, scope, w, root=archive_dir(ctx)).items() if v}
    if not inst:
        raise Refusal(f'{unit}: {why}; the archive holds no product of it - a gap, nothing to push or keep - REFUSED')
    many = {k: v for k, v in inst.items() if len(v) > 1}
    if many:
        raise Refusal(f'{unit}: {why}; more than one installed file of a kind {many} - REFUSED')
    disp = CFG.hybrid_unit.REFUSED_INSTALLED.get((lst, scope, w))
    names = ', '.join(os.path.basename(v[0]) for v in inst.values())
    if disp not in ('keep', 'quarantine'):
        raise Refusal(f'{unit}: {why}; the archive holds its products from an earlier build ({names}), which would stay silently. '
                      f'A ruling sets hybrid_unit.REFUSED_INSTALLED[{(lst, scope, w)!r}] = keep | quarantine '
                      f'(got {disp!r}) - REFUSED')
    keep = disp == 'keep'
    rows = [dict(unit=unit, kind=k, src='', dst=v[0] if keep else '', bytes=os.path.getsize(v[0]),
                 md5=K.checksum(v[0], ctx.algo) if keep and checksum else '', action='keep' if keep else 'retire',
                 replaces='' if keep else v[0], written_at='') for k, v in inst.items()]
    return rows, [f'{disp} by ruling (hybrid_unit.REFUSED_INSTALLED): {why}; installed {names}']


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
        if w == CFG.hybrid_full.PBDIFFUSE_WEIGHTING:     # the cell's diffuse map: written and pushed by this weighting's unit only
            pd = tf.get('pbdiffuse_map')
            if not pd:
                raise Refusal(f'{tp}: no diffuse PB map recorded (made before the map was written by the '
                              f'hybrid_full.PBDIFFUSE_WEIGHTING unit only); run --stages tilefill again')
            items.append(dict(kind='pbdiffuse', src=pd['path'], name=f'PBdiffuse_deep_{scope}.fits', md5=pd.get('md5')))
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
    kinds = {it['kind'] for it in items} | (set(PRODUCT_KINDS) if tf.get('source') == 'work' else set())   # pushed or retired here
    many = {k: v for k, v in inst.items() if len(v) > 1 and k in kinds}
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
        else:
            dm = K.checksum(dst, ctx.algo)
            act = 'keep' if dm == m else 'replace'                       # keep only byte-identical; other bytes (pixels or header)
        if act in ('replace', 'keep-or-replace'):
            miss = missing_keys(h, header(old), kind)
            if miss:
                raise Refusal(f'D7: {src} lacks card key(s) {miss[:10]} of the file it replaces {old} - REFUSING the unit')
        keep = act == 'keep'
        rows.append(dict(unit=unit, kind=kind, src=src, dst=dst, bytes=os.path.getsize(dst if keep else src),
                         md5=(dm if keep else m), action=act, replaces=old if not keep else '', written_at=''))
    if tf.get('source') == 'work':
        made = {it['kind'] for it in items}
        for k in PRODUCT_KINDS:
            if k not in made and inst[k]:
                rows.append(dict(unit=unit, kind=k, src='', dst='', bytes=os.path.getsize(inst[k][0]), md5='', action='retire',
                                 replaces=inst[k][0], written_at=''))
                notes.append(f'{k} {os.path.basename(inst[k][0])}: the rebuild makes no {k} - quarantined without a successor')
    return rows, reads, notes


def apply_unit(ctx, u, rows, rnd, cls='hybrid', adir=None, qdir=None, qman_name=QUARANTINE_MANIFEST,
               copy_class='hybrid_archive_copy'):
    """D2 swap of one planned unit: verified copies first, then per file the installed one into the round quarantine (manifest
    row) and the new one renamed in, back to back; the applied push manifest; the content audit. The class hybrid's places by
    default; push --class mosaic_v5 passes its run (adir), its round quarantine (qdir, qman_name) and its copy class. An
    installed file's quarantine path is qdir + its path under adir."""
    unit = rows[0]['unit']
    todo = [r for r in rows if r['action'] in ('publish', 'replace', 'retire')]
    adir = adir or archive_dir(ctx)
    qdir = qdir or quarantine_dir(ctx, rnd)
    qpath = lambda old: os.path.join(qdir, os.path.relpath(old, adir))     # noqa: E731  (the installed file's place in the round)
    qman = os.path.join(qdir, qman_name)
    pman = os.path.join(adir, f'MANIFEST_{ctx.stamp}.csv')
    ctx.guard_writes([adir, qdir, qman, pman], f'push --class {cls} {unit}', allow=f'archive:{cls}')
    for r in todo:
        if r['action'] in ('replace', 'retire') and os.path.lexists(qpath(r['replaces'])):
            raise Refusal(f'G4: quarantine destination exists: {qpath(r["replaces"])} (choose another --round)')
    ok, free = K.free_space_ok(ctx.archive_root, K.sec().FREE_SPACE_FLOOR_GB)
    if not ok:
        raise Refusal(f'G19: {free} GB free < archive.FREE_SPACE_FLOOR_GB')
    parts, cur = {}, None
    try:
        for r in todo:
            if r['action'] in ('publish', 'replace'):
                # this copy's own .part (removed on any failure); a .part that existed before is G6's to report, never removed here
                cur = None if os.path.lexists(r['dst'] + '.part') else r['dst'] + '.part'
                parts[r['dst']] = K.verified_copy(r['src'], r['dst'], ctx.algo, want=r['md5'], preserve=True)
                cur = None
    except BaseException:
        for part in [p for p, _m in parts.values()] + ([cur] if cur else []):
            if os.path.lexists(part):
                os.remove(part)
        raise
    t0 = K.utc()
    moved, written = [], []
    try:
        for r in todo:
            if r['action'] in ('replace', 'retire'):
                q = qpath(r['replaces'])
                with K.CsvAppender(qman, _p().HYBRID_QUARANTINE_COLUMNS) as wq:
                    b, m, t = K.quarantine_move(r['replaces'], q, ctx.algo, what=f'push {cls} {unit} {r["action"]}')
                    wq.write(dict(unit=unit, kind=r['kind'], path=r['replaces'], dest=q, bytes=b, md5=m,
                                  replaced_by=r['dst'] if r['action'] == 'replace' else '', moved_at=t))
                moved.append((r, q))
            if r['action'] in ('publish', 'replace'):
                K.commit_part(parts[r['dst']][0], r['dst'], replace=False)     # the installed file left first: nothing is overwritten
                r['written_at'] = K.utc()
                written.append(r)
    except BaseException as e:
        # a swap stopped part way: the files moved so far are in the quarantine manifest (a row per move), the files written so far
        # go to the push manifest, and every unswapped .part is removed - a rerun then keeps the written files and publishes a kind
        # left without an installed file (its old file stays in quarantine, recorded)
        done = {r['dst'] for r in written}
        left = [p for d_, (p, _m) in parts.items() if d_ not in done and os.path.lexists(p)]
        for p in left:
            os.remove(p)
        with K.CsvAppender(pman, _p().HYBRID_PUSH_COLUMNS) as wp:
            for r in written:
                wp.write(r)
        if not isinstance(e, Exception):
            raise
        raise Refusal(f'{unit}: the swap stopped ({type(e).__name__}: {e}) after {len(moved)} file(s) quarantined (rows in {qman}) and '
                      f'{len(written)} written (rows in {pman}); {len(left)} unswapped .part removed - a rerun keeps the written files '
                      f'and publishes any kind left without an installed file') from e
    listed = [r for r in rows if r['action'] != 'retire']
    if listed:                                  # a retire-only unit (a ruled quarantine) writes no push manifest
        with K.CsvAppender(pman, _p().HYBRID_PUSH_COLUMNS) as wp:
            for r in listed:
                wp.write(r)
    if moved:
        u.add_output(qman, 'quarantine_manifest', since=t0, paths=[r['replaces'] for r, _q in moved])
    if listed:
        u.add_output(pman, 'archive_manifest', paths=[r['dst'] for r in listed])
    qof = {r['dst']: q for r, q in moved if r['action'] == 'replace'}
    for r in written:
        u.add_output(r['dst'], copy_class, kind=r['kind'], md5=r['md5'], replaces=qof.get(r['dst']), algo=ctx.algo)
    return moved, written


def run_hybrid(ctx, args, run, unit_fn):
    from exopipe.phase3.lib.archive.push import class_cfg
    c = class_cfg('hybrid')
    if ctx.tier not in c.get('tiers', ('deep',)):
        raise NoInput(f'push --class hybrid --tier {ctx.tier}: hybrid masters are deep-tier products; nothing to push')
    staging = args.staging_root
    rnd = args.round or time.strftime('%Y%m%d', time.gmtime())
    stages = args._stages
    from exopipe.phase3.core.record import record_path
    from exopipe.phase3.steps import hybrid as SH
    done = hybrid_units(args, staging)
    states = {k: SH.unit_state(v) for k, v in SH.work_states(staging, lsts=set(push_lsts(args))).items()}
    notdone = {k: s for k, s in states.items() if s[0] in ('refused', 'unfinished')}   # before a done tilefill record
    for lst, scope, w in sorted(set(done) | set(notdone)):
        def body(u, lst=lst, scope=scope, w=w):
            u.meta.update(push_class='hybrid', lst=lst, scope=scope, weighting=w, staging_root=staging,
                          archive_root=ctx.archive_root, archive_dir=archive_dir(ctx, lst, scope), round=rnd)
            if (lst, scope, w) in notdone:
                u.meta['work_state'] = list(notdone[(lst, scope, w)])
                rows, notes = refused_rows(ctx, lst, scope, w, notdone[(lst, scope, w)], checksum=not ctx.dry)
                reads = [r for r in (record_path(staging, 'hybrid', f'{x}.{lst}_{scope}_{w}') for x in SH.STAGES) if os.path.exists(r)]
                for r in rows:
                    if r['action'] == 'keep':
                        u.add_input(r['dst'], extra={'md5': r['md5']} if r['md5'] else None)
                u.meta['kept'] = [dict(path=r['dst'], md5=r['md5']) for r in rows if r['action'] == 'keep']
            else:
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
