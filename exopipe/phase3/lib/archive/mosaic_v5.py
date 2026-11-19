"""`archive push --class mosaic_v5` (PR-7d; spec plan/specs/hybrid.md 2.15, 3.1, D2, D7): the conformed hybrid mosaics and the V5
tables of an `exopipe3 mosaic --generation v5` run's --out into the hybrid mosaic run (hybrid_tilefill.FILLED_MOSAIC_RUN).

Units: push_mosaic_v5_<scope>_<w> for every conform record of --staging-root (<staging>/_records/mosaic/v5-conform.<scope>_<w>.json),
then push_mosaic_v5_tables when the run holds a v5-tables record. Files of a scope unit: the products of its conform
record (Jy/sr and Jy/beam maps and their renders, PB_eff, PSF, PKCORR and the caveat map) into <run>/<scope>/I/<subdirectory>/<name>;
the unfilled Jy/sr and Jy/beam twins conform stages beside them are NOT pushed (spec Q2, R-GREGG-20261118-HYBRID-Q1-Q3-GO-WITH-
RECOMMENDATIONS: the archive receives filled products only; the twins stay in --out, the input of the tilefill band field). Files of
the tables unit: the tables of the v5-tables record into <run>/tables/<name>.

Per file: keep (the installed file of the kind has the same name and the same checksum), publish (none installed), replace (D2: the
installed file of the kind - the same name, or the same name with another night span and N - is renamed into
<root>/<archive_push.MOSAIC_V5_QUARANTINE_PREFIX><round>/<its path relative to the run> with a row in that Run's
MANIFEST_MOSAIC_V5_QUARANTINE.csv, then the new file is renamed in, back to back). The copy, swap and manifests are the class hybrid's
(lib/archive/hybrid.apply_unit).

Gates (a unit refuses whole; nothing of it is copied):
  a scope unit: the conform record done with every test passed; its tile list (stage/tiles.json) at the record's md5; no night of a
  tile now excluded (steps.hybrid.excluded_in_unit, per member for wideband and Full); every product at the md5 its record holds, its
  PRODNAME its file name and a phase3 name, the archive audit clean (push.audit_mosaic: name, PBCORR, BUNIT, NSTACK, DATE-OBS / END,
  finite pixels); HG20 the masters first: every PROVENANCE FILE (the record's prov_files) is installed, and the filled master of a
  hybrid tile holds TPSRCMD5 = the tile's md5, TPMOSAIC = the Jy/sr product's name and TPMOSMD5 = its md5 (the masters were filled
  from this mosaic: `exopipe3 hybrid --stages tilefill --mosaic-out` then `archive push --class hybrid` come first); the run's
  v5-tables record done, listing the unit and made from its conform (the archive tables must describe the products installed: the
  tables stage runs before the push); D7 a replacement holds every card key of the file it replaces (commentary aside) less
  archive_push.MOSAIC_V5_RETIRED_KEYS; at most one installed file per kind; the header gates run before any file is hashed, and a
  file is hashed once per process (gen_common._md5, keyed by inode, size and mtime; a file written is seeded at its verified md5);
  the tables unit: the v5-tables record done with every test passed, each table at its md5; every conform unit whose rows the tables
  hold is installed (each pushed product of its record in the run at the record's md5);
  the push: a --staging-root without any conform or tables record refuses (NoInput), never a clean exit with nothing done;
  both: G4 quarantine destinations free; G19 free space; G2 production writes. All copies are verified (.part, checksum) before the
  first move. Content audit after the swap: class mosaic_v5_archive_copy on every file written.
The run's README.txt is not written here (spec PR-8: README notes as templated files).
"""
import glob
import json
import os
import re
import time

from exopipe.phase3.core.record import NoInput, Refusal
from exopipe.phase3.lib.archive import common as K

QUARANTINE_MANIFEST = 'MANIFEST_MOSAIC_V5_QUARANTINE.csv'
TABLES_UNIT = 'tables'
TABLE_KEYS = ('content', 'curve', 'caveat_table')      # a conform record's meta.tables files the tables stage reads
_SPAN = re.compile(r'_\d{8}-\d{8}_N\d+\.fits$')


def _p():
    return K.sec('archive_push')


def run_dir(ctx):
    """<archive root>/<hybrid_tilefill.FILLED_MOSAIC_RUN relative to paths.PHASE3_ARCHIVE_ROOT>."""
    return archived(ctx, K.cfg().hybrid_tilefill.FILLED_MOSAIC_RUN)


def archived(ctx, path):
    """The --archive-root copy of a path under paths.PHASE3_ARCHIVE_ROOT (the mosaic run, a PROVENANCE FILE)."""
    rel = os.path.relpath(path, K.cfg().paths.PHASE3_ARCHIVE_ROOT)
    if rel.startswith('..'):
        raise Refusal(f'{path} is not under paths.PHASE3_ARCHIVE_ROOT')
    return os.path.join(ctx.archive_root, rel)


def quarantine_dir(ctx, rnd):
    return os.path.join(ctx.archive_root, f'{_p().MOSAIC_V5_QUARANTINE_PREFIX}{rnd}')


def kind_key(name):
    """A product's name less its night span and N: the files of one kind share it."""
    return _SPAN.sub('', name)


def installed(dd, name):
    """The installed files in dd of name's kind (kind_key equal; a table: the same name)."""
    if not name.endswith('.fits'):
        p = os.path.join(dd, name)
        return [p] if os.path.lexists(p) else []
    key = kind_key(name)
    return sorted(p for p in glob.glob(os.path.join(glob.escape(dd), glob.escape(key) + '_*.fits'))
                  if kind_key(os.path.basename(p)) == key)


def header(path, ext=0):
    from astropy.io import fits
    return fits.getheader(path, ext, memmap=False)


def header_problems(path, name):
    """Why a product may not enter the run under name (empty = conformant): PRODNAME (HDU 0, and HDU 1 of a HEALPix map) is name and a
    phase3 name; push.audit_mosaic."""
    from exopipe.phase3.core.fitsio import has_data
    from exopipe.phase3.core.products import parse_name
    from exopipe.phase3.lib.archive.push import audit_mosaic
    from astropy.io import fits
    bad = []
    with fits.open(path, memmap=False) as h:
        pns = [h[0].header.get('PRODNAME')] + ([h[1].header.get('PRODNAME')] if not has_data(h[0]) and len(h) > 1 else [])
    for pn in pns:
        if pn != name:
            bad.append(f'PRODNAME {pn!r} is not the file name {name}')
    try:
        if (parse_name(name) or {}).get('grammar') != 'phase3':
            bad.append(f'{name} is not a phase3 name')
    except ValueError:
        bad.append(f'{name} is not a phase3 name')
    return bad + list(audit_mosaic(path))


def missing_keys(new, old, retired=None):
    """D7 per HDU: card keys of the replaced file (commentary aside) absent from the new one, less retired (default
    MOSAIC_V5_RETIRED_KEYS; the class mosaic_v3 passes its own)."""
    from astropy.io import fits
    from exopipe.phase3.lib.hybrid.tests_tilefill import missing_card_keys
    retired = tuple(_p().MOSAIC_V5_RETIRED_KEYS) if retired is None else tuple(retired)
    out = []
    with fits.open(new, memmap=False) as a, fits.open(old, memmap=False) as b:
        for i in range(len(b)):
            if i >= len(a):
                out.append(f'HDU {i} ({b[i].name})')
                continue
            out += [k if i == 0 else f'{i}:{k}' for k in missing_card_keys(a[i].header, b[i].header, retired)]
    return out


def _rec_path(staging, unit):
    from exopipe.phase3.core.record import record_path
    from exopipe.phase3.lib.mosaic import v5
    return record_path(staging, v5.STEP, unit)


def v5_units(staging):
    """[(scope, weighting)] of the conform records under the staging root, in mosaic_v5.SCOPES x WEIGHTINGS order."""
    from exopipe.phase3.lib.mosaic import v5
    if not staging or not os.path.isdir(staging):
        raise ValueError(f'push --class mosaic_v5: --staging-root must be the v5 mosaic run\'s --out (the conform records are read '
                         f'there); got {staging!r}')
    V5 = K.cfg().mosaic_v5
    return [(sc, w) for sc in V5.SCOPES for w in V5.WEIGHTINGS if os.path.exists(_rec_path(staging, v5.unit_name('conform', sc, w)))]


def _done(rec, path):
    if rec is None:
        raise NoInput(f'{path}: no record')
    if rec.get('status') != 'done':
        raise Refusal(f'{path}: the unit is {rec.get("status")!r}, not done - nothing of it is pushed')
    bad = [t.get('name') for t in rec.get('tests') or [] if not t.get('passed')]
    if bad:
        raise Refusal(f'{path}: failed content test(s) {bad[:4]} (a waived failure is never pushed) - nothing of it is pushed')


def _row(ctx, unit, kind, src, dst, md5, checksum, d7=True, retired=None):
    """The plan row of one file (keep / publish / replace / keep-or-replace) and the installed file it supersedes; Refusal on
    an ambiguous kind, an occupied destination or D7 (missing_keys with retired)."""
    inst = installed(os.path.dirname(dst), os.path.basename(dst))
    if len(inst) > 1:
        raise Refusal(f'{unit} {kind}: more than one installed file of the kind: {inst} - which one the new file replaces is ambiguous')
    old = inst[0] if inst else ''
    if os.path.lexists(dst) and dst != old:
        raise Refusal(f'{unit} {kind}: {dst} exists and is not the installed {kind} file ({old or "none"}) - REFUSING')
    if not old:
        act = 'publish'
    elif old != dst:
        act = 'replace'
    elif not checksum:
        act = 'keep-or-replace'
    else:
        act = 'keep' if md5_of(ctx, dst) == md5 else 'replace'
    if d7 and act in ('replace', 'keep-or-replace'):
        miss = missing_keys(src, old, retired)
        if miss:
            raise Refusal(f'D7: {src} lacks card key(s) {miss[:10]} of the file it replaces {old} - REFUSING the unit')
    keep = act == 'keep'
    return dict(unit=unit, kind=kind, src=src, dst=dst, bytes=os.path.getsize(dst if keep else src), md5=md5, action=act,
                replaces='' if keep else old, written_at='')


def md5_of(ctx, p):
    """The checksum (ctx.algo) of a file; md5 through the process cache gen_common._md5 (a file hashed, or written and seeded, in this
    process is not read again while its inode, size and mtime hold)."""
    if ctx.algo == 'md5':
        from exopipe.phase3.lib.mosaic.gen_common import _md5
        return _md5(p)
    return K.checksum(p, ctx.algo)


def made_from(trec, crec, keys):
    """[key] of the conform record crec's table inputs (meta.tables[key] = [path, md5]) that the tables record trec was NOT made from
    (its input of that path holds another md5)."""
    ins = trec.get('inputs') or {}
    ct = (crec.get('meta') or {}).get('tables') or {}
    return [f'{k} {ct[k][0]}' for k in keys if ct.get(k) and (ins.get(os.path.abspath(ct[k][0])) or {}).get('md5') != ct[k][1]]


def tables_current(staging, step, tables_unit, label, listed_key, crec, keys):
    """[problem] (empty = current): the run's tables record is done, lists the conform unit label under meta[listed_key] and was made
    from this conform record's files. A scope unit is pushed only then: the archive tables must describe the products installed."""
    from exopipe.phase3.core.record import load_record
    trec = load_record(staging, step, tables_unit)
    if trec is None or trec.get('status') != 'done':
        return [f'the run holds no done tables record ({tables_unit})']
    if label not in ((trec.get('meta') or {}).get(listed_key) or []):
        return [f'the tables record does not list {label}']
    st = made_from(trec, crec, keys)
    return [f'the tables record was made from an earlier conform of {label} ({st[:2]})'] if st else []


def installed_problems(rd, label, products, conform_dir, checksum, skip=None):
    """[problem] of a conform unit's products (kinds starting with skip aside) against the run rd: each installed at the md5 its record
    holds (a dry run: existence only); md5 through the process cache."""
    from exopipe.phase3.lib.mosaic.gen_common import _md5
    bad = []
    for kind, e in sorted(products.items()):
        if skip and kind.startswith(skip):
            continue
        dst = os.path.join(rd, os.path.relpath(e['path'], conform_dir))
        if not os.path.exists(dst) or (checksum and _md5(dst) != e['md5']):
            bad.append(f'{label} {kind}: {dst} is not installed at the record\'s md5')
    return bad


def tables_rows(ctx, rec, rp, rd, arch_run, checksum):
    """The plan rows of a done tables record's outputs into <rd>/tables: each at the md5 the record holds; an installed table is the
    one the tables stage merged from (the record's input from <arch_run>/tables) or this one (no lost update between pushes)."""
    ins = rec.get('inputs') or {}
    md5s = (rec.get('meta') or {}).get('md5') or {}
    arch_t = os.path.realpath(os.path.join(arch_run, 'tables'))
    merged = {os.path.basename(p): (fp or {}).get('md5') for p, fp in ins.items() if os.path.dirname(os.path.realpath(p)) == arch_t}
    rows = []
    for o in rec.get('outputs') or []:
        src = o['path']
        name = os.path.basename(src)
        if name not in md5s:
            raise Refusal(f'{rp}: no md5 recorded for {name} (made before the tables record held them); run --stages tables again')
        if not os.path.exists(src):
            raise Refusal(f'tables: {src} does not exist - REFUSING')
        m = _checked(ctx, src, md5s[name], checksum, f'tables {name}', 'the tables stage')
        dst = os.path.join(rd, 'tables', name)
        if os.path.exists(dst) and md5_of(ctx, dst) not in (merged.get(name), md5s[name]):   # no lost update between pushes
            raise Refusal(f'tables {name}: the installed {dst} is not the table the tables stage merged from '
                          f'(md5 {merged.get(name)}) - another push changed it; run --stages tables again')
        rows.append(_row(ctx, TABLES_UNIT, 'table', src, dst, m, checksum, d7=False))
    if not rows:
        raise Refusal(f'{rp}: no table recorded - REFUSING')
    return rows


def run_push(ctx, args, unit_fn, cls, what, items_fn, rd, qd, qman_name):
    """The units of a mosaic push class (mosaic_v5, mosaic_v3): items_fn() -> [(suffix, meta, plan)], plan(checksum) -> (rows, reads);
    unit push_<cls>_<suffix>. No item refuses (NoInput: a push that would do nothing is an error, never a clean exit). Without --apply
    (or the transfer stage) each unit writes its plan manifest; with it, apply_unit (the class hybrid's D2 swap), and every file written
    is seeded into the process md5 cache (its copy was verified), so the tables unit's installed check reads it no second time."""
    from exopipe.phase3.lib.archive import hybrid as AH
    from exopipe.phase3.lib.archive.push import class_cfg
    from exopipe.phase3.lib.mosaic.gen_common import _seed
    c = class_cfg(cls)
    if ctx.tier not in c.get('tiers', ('deep',)):
        raise NoInput(f'push --class {cls} --tier {ctx.tier}: {what} are deep-tier products; nothing to push')
    items = items_fn()
    if not items:
        raise NoInput(f'push --class {cls}: no conform or tables record under --staging-root {args.staging_root} - nothing to push')
    rnd = args.round or time.strftime('%Y%m%d', time.gmtime())
    stages = args._stages
    pre = f'push_{cls}_'
    for suffix, meta, plan in items:
        name = pre + suffix

        def body(u, meta=meta, plan=plan, name=name):
            u.meta.update(push_class=cls, staging_root=args.staging_root, archive_root=ctx.archive_root, run_dir=rd, round=rnd, **meta)
            rows, reads = plan(not ctx.dry)
            for p in reads:
                u.add_input(p)
            u.meta['counts'] = {k: sum(1 for r in rows if r['action'] == k) for k in ('keep', 'publish', 'replace', 'keep-or-replace')}
            if ctx.dry:
                return rows
            if not ctx.apply or 'transfer' not in stages:
                man = os.path.join(ctx.manifest_dir(pre[:-1]), f'{name[len(pre):]}_{ctx.stamp}.csv')
                K.write_csv(man, _p().HYBRID_PUSH_COLUMNS, rows)
                u.add_output(man, 'csv_table', required_columns=('unit', 'kind', 'src', 'dst', 'action', 'replaces'))
                return rows
            if any(r['action'] in ('publish', 'replace') for r in rows):
                AH.apply_unit(ctx, u, rows, rnd, cls=cls, adir=rd, qdir=qd, qman_name=qman_name, copy_class=f'{cls}_archive_copy')
                if ctx.algo == 'md5':
                    for r in rows:
                        if r['action'] in ('publish', 'replace') and os.path.exists(r['dst']):
                            _seed(r['dst'], r['md5'])
            return rows
        unit_fn(name, body, describe=lambda rows: ', '.join(
            f'{k} {sum(1 for r in rows if r["action"] == k)}' for k in ('keep', 'publish', 'replace', 'keep-or-replace')
            if any(r['action'] == k for r in rows)) or '0 files')


def masters_problems(ctx, tiles, prov_files, jysr_name, jysr_md5):
    """HG20: [problem] of the installed masters against the mosaic: each PROVENANCE FILE installed; a hybrid tile's master holds
    TPSRCMD5 = the tile's md5 and TPMOSAIC / TPMOSMD5 = the Jy/sr product's name / md5."""
    if len(tiles) != len(prov_files):
        return [f'{len(tiles)} tiles but {len(prov_files)} PROVENANCE files']
    bad = []
    for t, f in zip(tiles, prov_files):
        a = archived(ctx, f)
        if not os.path.exists(a):
            bad.append(f'{t["lst"]}: {a} is not installed')
            continue
        if t.get('kind', 'hybrid') != 'hybrid':
            continue
        h = header(a)
        for k, want in (('TPSRCMD5', t['md5']), ('TPMOSAIC', jysr_name), ('TPMOSMD5', jysr_md5)):
            if str(h.get(k)) != str(want):
                bad.append(f'{t["lst"]}: {os.path.basename(a)} {k} {h.get(k)!r}, not {want!r}')
    return bad


def scope_plan(ctx, staging, scope, w, checksum=True):
    """(rows, reads) of one scope unit; Refusal on any gate (module docstring)."""
    from exopipe.phase3.core import exclusions
    from exopipe.phase3.core.record import load_record
    from exopipe.phase3.lib.hybrid.tilefill import TilefillRefusal
    from exopipe.phase3.lib.mosaic import v5
    from exopipe.phase3.steps import hybrid as SH
    unit = f'{scope}_{w}'
    rp = _rec_path(staging, v5.unit_name('conform', scope, w))
    rec = load_record(staging, v5.STEP, v5.unit_name('conform', scope, w))
    _done(rec, rp)
    meta = rec.get('meta') or {}
    P = v5.paths(staging, scope, w)
    if not os.path.exists(P['tiles']) or K.checksum(P['tiles'], 'md5') != meta.get('tiles_md5'):
        raise Refusal(f'{unit}: {P["tiles"]} is not the tile list conform used (md5 {meta.get("tiles_md5")}) - REFUSING')
    with open(P['tiles']) as f:
        tiles = json.load(f)['tiles']
    reads = [rp, P['tiles']]
    excl = exclusions.for_tier('deep', fresh=True)
    bad = {}
    for t in tiles:
        if not t.get('nights_from'):
            raise Refusal(f'{unit} {t["lst"]}: the tile records no nights (nights_from) - its exclusions cannot be checked')
        try:
            b, r2 = SH.excluded_in_unit(excl, os.path.dirname(os.path.dirname(t['nights_from'])), t['lst'], scope, w, t['nights'])
        except TilefillRefusal as e:
            raise Refusal(str(e))
        reads += r2
        if b:
            bad[t['lst']] = b
    if bad:
        raise Refusal(f'{unit}: night(s) now excluded in data_decisions.csv {bad} - never pushed; rebuild the mosaic')
    if getattr(excl, 'path', None):
        reads.append(excl.path)
    prods = {k: e for k, e in (meta.get('products') or {}).items() if not k.startswith('unfilled_')}
    if 'jysr' not in prods:
        raise Refusal(f'{rp}: no Jy/sr product recorded - REFUSING')
    mp = masters_problems(ctx, tiles, meta.get('prov_files') or [], os.path.basename(prods['jysr']['path']), prods['jysr']['md5'])
    if mp:                                                    # header reads only: before any product is hashed
        raise Refusal(f'HG20: the installed masters were not filled from this mosaic ({len(mp)}): {mp[:4]} - push the masters filled '
                      f'from it first (hybrid --stages tilefill --mosaic-out {staging}, then archive push --class hybrid)')
    tp = tables_current(staging, v5.STEP, v5.TABLES_UNIT, unit, 'units', rec, TABLE_KEYS)
    if tp:
        raise Refusal(f'{unit}: {tp[0]} - the archive tables must describe the products installed: run mosaic --stages tables, then '
                      f'push - REFUSING')
    rd = run_dir(ctx)
    rows = []
    for kind, e in sorted(prods.items()):
        src, name = e['path'], os.path.basename(e['path'])
        if not os.path.exists(src):
            raise Refusal(f'{unit} {kind}: {src} does not exist')
        if K.intermediate(name):
            raise Refusal(f'{unit} {kind}: {name} is an intermediate name (archive_push.INTERMEDIATE_PATTERN)')
        m = _checked(ctx, src, e['md5'], checksum, f'{unit} {kind}', 'conform')
        hp = header_problems(src, name)
        if hp:
            raise Refusal(f'{unit} {kind}: {src}: {"; ".join(hp[:4])} - REFUSING the unit')
        rel = os.path.relpath(src, P['conform'])
        if rel.startswith('..') or rel.split(os.sep)[0] != scope:
            raise Refusal(f'{unit} {kind}: {src} is not under {P["conform"]}/{scope}')
        rows.append(_row(ctx, unit, kind, src, os.path.join(rd, rel), m, checksum))
    return rows, reads


def _checked(ctx, src, md5, checksum, what, stage):
    """The checksum (ctx.algo) of src for the manifest after its md5 is found equal to the one its stage recorded (Refusal otherwise);
    without checksum (a dry run) the recorded md5."""
    if not checksum:
        return md5
    m = K.checksum(src, 'md5')
    if m != md5:
        raise Refusal(f'{what}: {src} changed since {stage} recorded it (md5 {m}, recorded {md5}) - REFUSING')
    return m if ctx.algo == 'md5' else K.checksum(src, ctx.algo)


def tables_plan(ctx, staging, checksum=True):
    """(rows, reads) of the tables unit; Refusal on any gate (module docstring)."""
    from exopipe.phase3.core.record import load_record
    from exopipe.phase3.lib.mosaic import v5
    rp = _rec_path(staging, v5.TABLES_UNIT)
    rec = load_record(staging, v5.STEP, v5.TABLES_UNIT)
    _done(rec, rp)
    meta = rec.get('meta') or {}
    rd = run_dir(ctx)
    bad, stale = [], []
    for sw in meta.get('units') or []:
        scope, w = sw.rsplit('_', 1)
        cr = load_record(staging, v5.STEP, v5.unit_name('conform', scope, w))
        if cr is None or cr.get('status') != 'done':
            bad.append(f'{sw}: {"no conform record" if cr is None else cr.get("status")}')
            continue
        stale += [f'{sw} {x}' for x in made_from(rec, cr, TABLE_KEYS)]                # made from THIS conform's files
        bad += installed_problems(rd, sw, (cr.get('meta') or {}).get('products') or {}, v5.paths(staging, scope, w)['conform'],
                                  checksum, skip='unfilled_')
    if stale:
        raise Refusal(f'the tables were made from an earlier conform ({len(stale)}): {stale[:4]} - run --stages tables again')
    if not meta.get('units'):
        bad.append('the tables record lists no conform unit')
    if bad:
        raise Refusal(f'the tables describe products not installed ({len(bad)}): {bad[:4]} - push the scopes first')
    return tables_rows(ctx, rec, rp, rd, K.cfg().hybrid_tilefill.FILLED_MOSAIC_RUN, checksum), [rp]


def run_mosaic_v5(ctx, args, run, unit_fn):
    from exopipe.phase3.lib.mosaic import v5
    staging = args.staging_root

    def items():
        it = [(f'{scope}_{w}', dict(scope=scope, weighting=w),
               lambda checksum, scope=scope, w=w: scope_plan(ctx, staging, scope, w, checksum=checksum)) for scope, w in v5_units(staging)]
        if os.path.exists(_rec_path(staging, v5.TABLES_UNIT)):
            it.append((TABLES_UNIT, dict(scope=TABLES_UNIT, weighting=None), lambda checksum: tables_plan(ctx, staging, checksum=checksum)))
        return it
    rnd = args.round or time.strftime('%Y%m%d', time.gmtime())
    run_push(ctx, args, unit_fn, 'mosaic_v5', 'the hybrid mosaics', items, run_dir(ctx), quarantine_dir(ctx, rnd), QUARANTINE_MANIFEST)
