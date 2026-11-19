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

Gates (a unit refuses whole; nothing of it is copied), the cheap ones before any file is hashed:
  a scope unit: the conform record done with every test passed; its tile list (stage/tiles.json) at the record's md5; no night of a
  tile's own master now excluded (data_decisions.csv read fresh); HG13 at the push, conform's own check (v3.resolve_now: the masters
  the stage would choose now, the plain masters behind the weighting maps and their nights included, and the substituted LSTs the
  selection names now); a Stokes I unit's THERMNOI that of the scope's done Stokes V conform record (one value per scope); the run's
  tables record done, listing the unit and made from its conform (the archive tables must describe the products installed: the
  tables stage runs before the push); every product present, not an intermediate name, its PRODNAME its file name and a phase3 name,
  the archive audit clean (push.audit_mosaic: name, PBCORR, BUNIT, NSTACK, DATE-OBS / END, finite pixels), under <scope>/<stokes>;
  then the hashes: every master the tile list names (its own and, for a substituted row, the plain master) at the md5 the stage
  recorded (a dry run: existence only), the reference master of each weighting map of another variant (pbfor) the file the stage
  read (record fingerprint: DATASUM and processing cards, a header read), and every product at the md5 its record holds; D7 a
  replacement holds every card key of the file it replaces (commentary aside) less archive_push.MOSAIC_V3_RETIRED_KEYS; at most one
  installed file per kind;
  the tables unit: the v3-tables record done with every test passed, each table at its md5; the tables made from the current conform
  records; every conform unit whose rows the tables hold is installed (each product of its record in the run at the record's md5);
  an installed table is the one the tables stage merged from (no lost update between pushes);
  the push: a --staging-root without any conform or tables record refuses (NoInput), never a clean exit with nothing done;
  both: G4 quarantine destinations free; G19 free space; G2 production writes ('archive:mosaic_v3' is not in the narrow cutover
  allowance: a production apply refuses until a ruling adds it). All copies are verified (.part, checksum) before the first move.
  Content audit after the swap: class mosaic_v3_archive_copy on every file written.
The run's README.txt is not written here (spec PR-8: README notes as templated files).
"""

import json
import os
import time

from exopipe.phase3.core.record import Refusal
from exopipe.phase3.lib.archive import common as K
from exopipe.phase3.lib.archive.mosaic_v5 import (_checked, _done, _row, archived, header_problems, installed_problems, made_from, md5_of,
                                                   run_push, tables_current, tables_rows)

QUARANTINE_MANIFEST = 'MANIFEST_MOSAIC_V3_QUARANTINE.csv'
TABLES_UNIT = 'tables'
TABLE_KEYS = ('content', 'curve')                      # a conform record's meta.tables files the tables stage reads


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
    """[(scope, Job)] of the conform records under the staging root, in mosaic_v3.SCOPES x all_jobs order."""
    from exopipe.phase3.lib.mosaic import v3
    if not staging or not os.path.isdir(staging):
        raise ValueError(f'push --class mosaic_v3: --staging-root must be the v3 mosaic run\'s --out (the conform records are read '
                         f'there); got {staging!r}')
    return [(sc, j) for sc, js in all_jobs().items() for j in js if os.path.exists(_rec_path(staging, v3.unit_name('conform', sc, j)))]


def scope_plan(ctx, staging, scope, job, checksum=True, sel=None):
    """(rows, reads) of one scope unit; Refusal on any gate (module docstring). sel: the striation selection (default read now)."""
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
        t = json.load(f)
    tiles = t['tiles']
    reads = [rp, P['tiles']]
    excl = exclusions.for_tier('deep', fresh=True)
    bad = {r['lst']: b for r in tiles for b in [excl.excluded_constituents(r['lst'], scope, r['nights'])] if b}
    if bad:
        raise Refusal(f'{unit}: night(s) now excluded in data_decisions.csv {bad} - never pushed; rebuild the mosaic')
    if getattr(excl, 'path', None):
        reads.append(excl.path)
    probs = v3.resolve_now(scope, job, t, v3.striation_selection() if sel is None else sel)
    if probs:
        raise Refusal(f'HG13 {unit}: {"; ".join(probs[:3])} - the mosaic no longer describes the installed masters - REFUSING')
    if job.product == 'I':                                        # THERMNOI: one value per scope, the Stokes V conform's
        vj = v3.Job('Vclean', K.cfg().mosaic_v3.VCLEAN_VARIANT)
        vp = _rec_path(staging, v3.unit_name('conform', scope, vj))
        vr = load_record(staging, v3.STEP, v3.unit_name('conform', scope, vj))
        if vr is None or vr.get('status') != 'done' or (vr.get('meta') or {}).get('thermnoi') != meta.get('thermnoi'):
            raise Refusal(f'{unit}: its THERMNOI {meta.get("thermnoi")} is not that of the done Stokes V conform record {vp} '
                          f'(the Stokes V conform ran again since, or is not done): rerun --stages conform of {scope} - REFUSING')
        reads.append(vp)
    tp = tables_current(staging, v3.STEP, v3.TABLES_UNIT, f'{scope} {job.label}', 'conformed', rec, TABLE_KEYS)
    if tp:
        raise Refusal(f'{unit}: {tp[0]} - the archive tables must describe the products installed: run mosaic --stages tables, then '
                      f'push - REFUSING')
    prods = meta.get('products') or {}
    if 'jysr' not in prods:
        raise Refusal(f'{rp}: no Jy/sr product recorded - REFUSING')
    st = stokes_of(job)
    for kind, e in sorted(prods.items()):                         # header reads only: before any file is hashed
        src, fname = e['path'], os.path.basename(e['path'])
        if not os.path.exists(src):
            raise Refusal(f'{unit} {kind}: {src} does not exist')
        if K.intermediate(fname):
            raise Refusal(f'{unit} {kind}: {fname} is an intermediate name (archive_push.INTERMEDIATE_PATTERN)')
        hp = header_problems(src, fname)
        if hp:
            raise Refusal(f'{unit} {kind}: {src}: {"; ".join(hp[:4])} - REFUSING the unit')
        rel = os.path.relpath(src, P['conform'])
        if rel.startswith('..') or rel.split(os.sep)[:2] != [scope, st]:
            raise Refusal(f'{unit} {kind}: {src} is not under {P["conform"]}/{scope}/{st}')
    named = sorted({(r['lst'], r['path'], r['md5']) for r in tiles}
                   | {(r['lst'], r['plain_path'], r['plain_md5']) for r in tiles if r.get('plain_path')})
    gone = sorted({lst for lst, pth, m in named if not os.path.exists(pth) or (checksum and md5_of(ctx, pth) != m)})
    if gone:
        raise Refusal(f'HG13 {unit}: tile master(s) missing or replaced since the stage at LST {gone[:6]} - the mosaic no longer '
                      f'describes the installed masters; rebuild it from --stages stage')
    reads += sorted({pth for _l, pth, _m in named})
    refs = sorted({(r['lst'], r['pbfor']) for r in tiles if r.get('pbfor') and r['pbfor'] not in (r['path'], r.get('plain_path'))})
    if refs:                                                      # the weighting maps' reference masters of another variant
        from exopipe.phase3.core.record import fingerprint, same_fingerprint
        pj = job.pb_job()
        sr = load_record(staging, v3.STEP, v3.unit_name('stage', scope, pj)) or {}
        sin = sr.get('inputs') or {}
        moved = [lst for lst, ref in refs if not same_fingerprint(sin.get(os.path.abspath(ref)) or {'missing': True}, fingerprint(ref))]
        if moved:
            raise Refusal(f'HG13 {unit}: the reference master of the weighting map at LST {moved[:6]} is missing or not the file the '
                          f'stage of {scope} {pj.label} read (DATASUM and processing cards) - rebuild the mosaic from --stages stage')
        reads += sorted({ref for _l, ref in refs})
    rd = run_dir(ctx)
    rows = []
    for kind, e in sorted(prods.items()):
        m = _checked(ctx, e['path'], e['md5'], checksum, f'{unit} {kind}', 'conform')
        rows.append(_row(ctx, unit, kind, e['path'], os.path.join(rd, os.path.relpath(e['path'], P['conform'])), m, checksum,
                         retired=tuple(_p().MOSAIC_V3_RETIRED_KEYS)))
    return rows, reads


def tables_plan(ctx, staging, checksum=True):
    """(rows, reads) of the tables unit; Refusal on any gate (module docstring)."""
    from exopipe.phase3.core.record import load_record
    from exopipe.phase3.lib.mosaic import v3
    rp = _rec_path(staging, v3.TABLES_UNIT)
    rec = load_record(staging, v3.STEP, v3.TABLES_UNIT)
    _done(rec, rp)
    meta = rec.get('meta') or {}
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
        stale += [f'{sl} {x}' for x in made_from(rec, cr, TABLE_KEYS)]                 # made from THIS conform's files
        bad += installed_problems(rd, sl, (cr.get('meta') or {}).get('products') or {}, v3.paths(staging, sc, j)['conform'], checksum)
    if stale:
        raise Refusal(f'the tables were made from an earlier conform ({len(stale)}): {stale[:4]} - run --stages tables again')
    if not meta.get('conformed'):
        bad.append('the tables record lists no conform unit')
    if bad:
        raise Refusal(f'the tables describe products not installed ({len(bad)}): {bad[:4]} - push the scopes first')
    return tables_rows(ctx, rec, rp, rd, K.cfg().mosaic_v3.ARCHIVE_RUN, checksum), [rp]


def run_mosaic_v3(ctx, args, run, unit_fn):
    from exopipe.phase3.lib.mosaic import v3
    staging = args.staging_root

    def items():
        us = v3_units(staging)
        sel = v3.striation_selection() if us else None             # read once for the run (HG13 of every scope unit)
        it = [(f'{scope}_{job.label}', dict(scope=scope, job=job.label),
               lambda checksum, scope=scope, job=job: scope_plan(ctx, staging, scope, job, checksum=checksum, sel=sel)) for scope, job in us]
        if os.path.exists(_rec_path(staging, v3.TABLES_UNIT)):
            it.append((TABLES_UNIT, dict(scope=TABLES_UNIT, job=None), lambda checksum: tables_plan(ctx, staging, checksum=checksum)))
        return it
    rnd = args.round or time.strftime('%Y%m%d', time.gmtime())
    run_push(ctx, args, unit_fn, 'mosaic_v3', 'the FEATHER3 mosaics', items, run_dir(ctx), quarantine_dir(ctx, rnd), QUARANTINE_MANIFEST)
