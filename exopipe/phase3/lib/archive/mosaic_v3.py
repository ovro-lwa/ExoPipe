"""`archive push --class mosaic_v3` (PR-6e; spec plan/specs/hybrid.md 2.15, D2, D7): the conformed FEATHER3 mosaics (Stokes I and V)
and the FEATHER3 tables of an `exopipe3 mosaic --generation v3` run's --out into the FEATHER3 mosaic run (mosaic_v3.ARCHIVE_RUN;
production push_mosaic_v3_20261022.py and the Release 1 / 11-09 / 11-12 / 11-13 in-place updates, L3365).

Units: push_mosaic_v3_<scope>_<job> for every conform record of --staging-root (<staging>/_records/mosaic/v3-conform.<scope>_<job>.json),
then push_mosaic_v3_tables when the run holds a v3-tables record. Files of a scope unit: the products of its conform record (Jy/sr and
Jy/beam maps and their renders, PB_eff, PSF and, Stokes I, PKCORR) into <run>/<scope>/<I|V>/<subdirectory>/<name>, production's layout.
Files of the tables unit: the tables of the v3-tables record (the merged summary, summary_vclean and pkcorr_curves tables and the
carried selection and striation tables) into <run>/tables/<name>.

Per file: keep, publish or replace (D2) exactly as the class mosaic_v5 (lib/archive/mosaic_v5.py; an installed file of the kind is
renamed into <root>/<archive_push.MOSAIC_V3_QUARANTINE_PREFIX><round>/<its path relative to the run> with a row in that Run's
MANIFEST_MOSAIC_V3_QUARANTINE.csv, then the new file is renamed in, back to back). The copy, swap and manifests are the class hybrid's
(lib/archive/hybrid.apply_unit).

Gates (a unit refuses whole; nothing of it is copied):
  a scope unit: the conform record done with every test passed; its tile list (stage/tiles.json) at the record's md5; no night of a
  tile now excluded (data_decisions.csv read fresh); HG13 at the push: every tile master still at the md5 the stage recorded (the
  mosaic describes the installed masters; a dry run: existence only); every product at the md5 its record holds, its PRODNAME its
  file name and a phase3 name, the archive audit clean (push.audit_mosaic: name, PBCORR, BUNIT, NSTACK, DATE-OBS / END, finite
  pixels); D7 a replacement holds every card key of the file it replaces (commentary aside) less archive_push.MOSAIC_V3_RETIRED_KEYS;
  at most one installed file per kind;
  the tables unit: the v3-tables record done with every test passed, each table at its md5; the tables made from the current conform
  records; every conform unit whose rows the tables hold is installed (each product of its record in the run at the record's md5);
  an installed table is the one the tables stage merged from (no lost update between pushes);
  both: G4 quarantine destinations free; G19 free space; G2 production writes ('archive:mosaic_v3' is not in the narrow cutover
  allowance: a production apply refuses until a ruling adds it). All copies are verified (.part, checksum) before the first move.
  Content audit after the swap: class mosaic_v3_archive_copy on every file written.
The run's README.txt is not written here (spec PR-8: README notes as templated files).
"""
import json
import os
import time

from exopipe.phase3.core.record import NoInput, Refusal
from exopipe.phase3.lib.archive import common as K
from exopipe.phase3.lib.archive.mosaic_v5 import _checked, _done, _row, archived, header_problems

QUARANTINE_MANIFEST = 'MANIFEST_MOSAIC_V3_QUARANTINE.csv'
TABLES_UNIT = 'tables'


def _p():
    return K.sec('archive_push')


def run_dir(ctx):
    """<archive root>/<mosaic_v3.ARCHIVE_RUN relative to paths.PHASE3_ARCHIVE_ROOT>."""
    return archived(ctx, K.cfg().mosaic_v3.ARCHIVE_RUN)


def quarantine_dir(ctx, rnd):
    return os.path.join(ctx.archive_root, f'{_p().MOSAIC_V3_QUARANTINE_PREFIX}{rnd}')


def _rec_path(staging, unit):
    from exopipe.phase3.core.record import record_path
    from exopipe.phase3.lib.mosaic import v3
    return record_path(staging, v3.STEP, unit)


def all_jobs():
    """{scope: [Job]}: every job a v3 run can make for each of mosaic_v3.SCOPES (the plain, then the striation-corrected Stokes I job
    of each variant, then Vclean: jobs_for's order without the striation selection); a push takes those with a conform record."""
    from exopipe.phase3.lib.mosaic import v3
    V3 = K.cfg().mosaic_v3
    js = [v3.Job('I', v) for v in V3.VARIANTS] + [v3.Job('I', v, ssub=True) for v in V3.VARIANTS] + [v3.Job('Vclean', V3.VCLEAN_VARIANT)]
    return {sc: list(js) for sc in V3.SCOPES}


def v3_units(staging):
    """[(scope, Job)] of the conform records under the staging root, in mosaic_v3.SCOPES x jobs_for order."""
    from exopipe.phase3.lib.mosaic import v3
    if not staging or not os.path.isdir(staging):
        raise ValueError(f'push --class mosaic_v3: --staging-root must be the v3 mosaic run\'s --out (the conform records are read '
                         f'there); got {staging!r}')
    return [(sc, j) for sc, js in all_jobs().items() for j in js if os.path.exists(_rec_path(staging, v3.unit_name('conform', sc, j)))]


def _retired():
    return tuple(_p().MOSAIC_V3_RETIRED_KEYS)


def scope_plan(ctx, staging, scope, job, checksum=True):
    """(rows, reads) of one scope unit; Refusal on any gate (module docstring)."""
    from exopipe.phase3.core import exclusions
    from exopipe.phase3.core.record import load_record
    from exopipe.phase3.lib.mosaic import v3
    from exopipe.phase3.lib.mosaic.conform_v3 import stokes_of
    unit = f'{scope}_{job.label}'
    name = v3.unit_name('conform', scope, job)
    rp = _rec_path(staging, name)
    rec = load_record(staging, v3.STEP, name)
    _done(rec, rp)
    meta = rec.get('meta') or {}
    P = v3.paths(staging, scope, job)
    if not os.path.exists(P['tiles']) or K.checksum(P['tiles'], 'md5') != meta.get('tiles_md5'):
        raise Refusal(f'{unit}: {P["tiles"]} is not the tile list conform used (md5 {meta.get("tiles_md5")}) - REFUSING')
    with open(P['tiles']) as f:
        tiles = json.load(f)['tiles']
    reads = [rp, P['tiles']]
    excl = exclusions.for_tier('deep', fresh=True)
    bad = {t['lst']: b for t in tiles for b in [excl.excluded_constituents(t['lst'], scope, t['nights'])] if b}
    if bad:
        raise Refusal(f'{unit}: night(s) now excluded in data_decisions.csv {bad} - never pushed; rebuild the mosaic')
    if getattr(excl, 'path', None):
        reads.append(excl.path)
    gone = [t['lst'] for t in tiles if not os.path.exists(t['path'])
            or (checksum and K.checksum(t['path'], 'md5') != t['md5'])]
    if gone:
        raise Refusal(f'HG13: tile master(s) missing or replaced since the stage at LST {gone[:6]} - the mosaic no longer describes the '
                      f'installed masters; rebuild it from --stages stage')
    reads += [t['path'] for t in tiles]
    prods = meta.get('products') or {}
    if 'jysr' not in prods:
        raise Refusal(f'{rp}: no Jy/sr product recorded - REFUSING')
    rd = run_dir(ctx)
    st = stokes_of(job)
    rows = []
    for kind, e in sorted(prods.items()):
        src, fname = e['path'], os.path.basename(e['path'])
        if not os.path.exists(src):
            raise Refusal(f'{unit} {kind}: {src} does not exist')
        if K.intermediate(fname):
            raise Refusal(f'{unit} {kind}: {fname} is an intermediate name (archive_push.INTERMEDIATE_PATTERN)')
        m = _checked(ctx, src, e['md5'], checksum, f'{unit} {kind}', 'conform')
        hp = header_problems(src, fname)
        if hp:
            raise Refusal(f'{unit} {kind}: {src}: {"; ".join(hp[:4])} - REFUSING the unit')
        rel = os.path.relpath(src, P['conform'])
        if rel.startswith('..') or rel.split(os.sep)[:2] != [scope, st]:
            raise Refusal(f'{unit} {kind}: {src} is not under {P["conform"]}/{scope}/{st}')
        rows.append(_row(ctx, unit, kind, src, os.path.join(rd, rel), m, checksum, retired=_retired()))
    return rows, reads


def tables_plan(ctx, staging, checksum=True):
    """(rows, reads) of the tables unit; Refusal on any gate (module docstring)."""
    from exopipe.phase3.core.record import load_record
    from exopipe.phase3.lib.mosaic import v3
    rp = _rec_path(staging, v3.TABLES_UNIT)
    rec = load_record(staging, v3.STEP, v3.TABLES_UNIT)
    _done(rec, rp)
    meta = rec.get('meta') or {}
    ins = rec.get('inputs') or {}
    rd = run_dir(ctx)
    jobs = {(sc, j.label): j for sc, js in all_jobs().items() for j in js}
    bad, stale = [], []
    for sl in meta.get('conformed') or []:
        sc, label = sl.split(' ', 1)
        j = jobs.get((sc, label))
        if j is None:
            bad.append(f'{sl}: not a mosaic_v3 job')
            continue
        cr = load_record(staging, v3.STEP, v3.unit_name('conform', sc, j))
        if cr is None or cr.get('status') != 'done':
            bad.append(f'{sl}: {"no conform record" if cr is None else cr.get("status")}')
            continue
        ct = (cr.get('meta') or {}).get('tables') or {}
        for k in ('content', 'curve'):                                          # made from THIS conform's files
            if ct.get(k) and (ins.get(os.path.abspath(ct[k][0])) or {}).get('md5') != ct[k][1]:
                stale.append(f'{sl} {k} {ct[k][0]}')
        P = v3.paths(staging, sc, j)
        for kind, e in ((cr.get('meta') or {}).get('products') or {}).items():
            dst = os.path.join(rd, os.path.relpath(e['path'], P['conform']))
            if not os.path.exists(dst) or (checksum and K.checksum(dst, 'md5') != e['md5']):   # a dry run: existence only
                bad.append(f'{sl} {kind}: {dst} is not installed at the record\'s md5')
    if stale:
        raise Refusal(f'the tables were made from an earlier conform ({len(stale)}): {stale[:4]} - run --stages tables again')
    if not meta.get('conformed'):
        bad.append('the tables record lists no conform unit')
    if bad:
        raise Refusal(f'the tables describe products not installed ({len(bad)}): {bad[:4]} - push the scopes first')
    rows = []
    md5s = meta.get('md5') or {}
    arch_t = os.path.realpath(os.path.join(K.cfg().mosaic_v3.ARCHIVE_RUN, 'tables'))
    merged = {os.path.basename(p): (fp or {}).get('md5') for p, fp in ins.items() if os.path.dirname(os.path.realpath(p)) == arch_t}
    for o in rec.get('outputs') or []:
        src = o['path']
        name = os.path.basename(src)
        if name not in md5s:
            raise Refusal(f'{rp}: no md5 recorded for {name} (made before the tables record held them); run --stages tables again')
        if not os.path.exists(src):
            raise Refusal(f'tables: {src} does not exist - REFUSING')
        m = _checked(ctx, src, md5s[name], checksum, f'tables {name}', 'the tables stage')
        dst = os.path.join(rd, 'tables', name)
        if os.path.exists(dst) and K.checksum(dst, 'md5') not in (merged.get(name), md5s[name]):   # no lost update between pushes
            raise Refusal(f'tables {name}: the installed {dst} is not the table the tables stage merged from '
                          f'(md5 {merged.get(name)}) - another push changed it; run --stages tables again')
        rows.append(_row(ctx, TABLES_UNIT, 'table', src, dst, m, checksum, d7=False))
    if not rows:
        raise Refusal(f'{rp}: no table recorded - REFUSING')
    return rows, [rp]


def run_mosaic_v3(ctx, args, run, unit_fn):
    from exopipe.phase3.lib.archive import hybrid as AH
    from exopipe.phase3.lib.archive.push import class_cfg
    from exopipe.phase3.lib.mosaic import v3
    c = class_cfg('mosaic_v3')
    if ctx.tier not in c.get('tiers', ('deep',)):
        raise NoInput(f'push --class mosaic_v3 --tier {ctx.tier}: the FEATHER3 mosaics are deep-tier products; nothing to push')
    staging = args.staging_root
    rnd = args.round or time.strftime('%Y%m%d', time.gmtime())
    stages = args._stages
    items = v3_units(staging)
    if os.path.exists(_rec_path(staging, v3.TABLES_UNIT)):
        items.append((TABLES_UNIT, None))
    rd = run_dir(ctx)
    qd = quarantine_dir(ctx, rnd)

    for scope, job in items:
        name = 'push_mosaic_v3_tables' if scope == TABLES_UNIT else f'push_mosaic_v3_{scope}_{job.label}'

        def body(u, scope=scope, job=job, name=name):
            u.meta.update(push_class='mosaic_v3', scope=scope, job=job.label if job else None, staging_root=staging,
                          archive_root=ctx.archive_root, run_dir=rd, round=rnd)
            if scope == TABLES_UNIT:
                rows, reads = tables_plan(ctx, staging, checksum=not ctx.dry)
            else:
                rows, reads = scope_plan(ctx, staging, scope, job, checksum=not ctx.dry)
            for p in reads:
                u.add_input(p)
            u.meta['counts'] = {k: sum(1 for r in rows if r['action'] == k) for k in ('keep', 'publish', 'replace', 'keep-or-replace')}
            if ctx.dry:
                return rows
            if not ctx.apply or 'transfer' not in stages:
                man = os.path.join(ctx.manifest_dir('push_mosaic_v3'), f'{name[len("push_mosaic_v3_"):]}_{ctx.stamp}.csv')
                K.write_csv(man, _p().HYBRID_PUSH_COLUMNS, rows)
                u.add_output(man, 'csv_table', required_columns=('unit', 'kind', 'src', 'dst', 'action', 'replaces'))
                return rows
            if any(r['action'] in ('publish', 'replace') for r in rows):
                AH.apply_unit(ctx, u, rows, rnd, cls='mosaic_v3', adir=rd, qdir=qd, qman_name=QUARANTINE_MANIFEST,
                              copy_class='mosaic_v3_archive_copy')
            return rows
        unit_fn(name, body, describe=lambda rows: ', '.join(
            f'{k} {sum(1 for r in rows if r["action"] == k)}' for k in ('keep', 'publish', 'replace', 'keep-or-replace')
            if any(r['action'] == k for r in rows)) or '0 files')
