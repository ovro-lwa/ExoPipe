"""`exopipe3 masters --tier deep`: the deep tier (INTERMITTENT, DESIGN 12.2; never per hour) - stages stage-in, build, pbmaps,
vclean, fluxscale, verify, and after `dewarp` + `install`: sources; optional variants. Unit of a run record: one cell per
stage, ``<stage>_<cell>`` (e.g. build_23h_23MHz); item kind: deep cell (core.cells.parse_cell). Interface of a masters tier
module: add_arguments(p) (tier-unique flags, prefixed --deep-...), default_out(args) and run(args, run).

Chain 'deep-build-cell' (runner.CHAINS, refactor decision 8): `masters --stages stage-in` -> `flow --tier deep` (stage-in of
the stored flows, prep, solve on the GPU, check) -> `masters` (default stages build,pbmaps,vclean,fluxscale,verify). The
flow step reads <out>/_records/masters/stage-in_<cell>.json; build reads <out>/_records/flow/<cell>.json.

Each stage reads only what the previous stage's run record lists as done (a stage refuses without a done record of the
stage before it, unless that stage ran earlier in the same invocation), calls the in-place production code
(lib/masters/*.py moved by pure git mv; exopipe_ops/p3s5_parents.py) exactly as the production launchers do
(rebuild_cell.sh, install_cell.sh:26), maps the in-place outcome (return value None / status string / exception) to
done / refused / no-input, and registers every product it wrote with its product class so its content tests run now.
Products keep production's names and layout (DESIGN 11.1): <out>/<cell>/{coadd_final,coadd_final_ref,coadd_provisional,
subtracted,flow,gen_shflux}/..., <out>/shflux_table.csv, <out>/parent_catalogs/.
"""
import csv
import glob
import json
import os
import shutil
import sys
import traceback

from exopipe.phase3.core import record
from exopipe.phase3.core.record import NoInput, Refusal

STAGE_ORDER = ('stage-in', 'build', 'pbmaps', 'vclean', 'fluxscale', 'verify', 'sources', 'variants')
# the stage whose done record a stage requires (spec 1.2); sources requires the install step's record instead
REQUIRES = {'build': 'stage-in', 'pbmaps': 'build', 'vclean': 'pbmaps', 'fluxscale': 'vclean', 'verify': 'fluxscale',
            'variants': 'build'}
# stages whose products derive from the build's master: they refuse when a night the build admitted is excluded NOW
# (DESIGN 5 / deep requirement (d): an exclusion change invalidates every result built on the old admitted set)
BUILD_DERIVED = ('pbmaps', 'vclean', 'fluxscale', 'verify', 'variants')
# what a cell directory may hold when the build starts: stage-in's reference + carried epoch-gate record, the flow step's
# flow/ and its prep-stage caches (coadd.process_lst_freq). Anything a build writes means an earlier build ran here.
BUILD_PRODUCT_PATHS = ('coadd_final/weights.csv', 'coadd_final/.done', 'coadd_final/loo', 'coadd_final/stage2_verdict.csv',
                       'coadd_final/night_image_gate.csv', 'coadd_provisional', 'residuals_provisional.csv', 'subtracted',
                       'prep_final')
WAIVE_REFERENCE = 'masters.reference'                 # --waive masters.reference --ruling "..." for --reference path:/night:
WAIVE_INSTALL_RECORD = 'masters.sources.install_record'   # window: installed cells have no install-step record yet


def _cfg():
    from exopipe.phase3.config import CFG
    return CFG


def add_arguments(p):
    g = p.add_argument_group('deep tier (--tier deep)')
    g.add_argument('--deep-variants', dest='deep_variants', metavar='LIST',
                   help='variants stage: which I variants rebuild_variants_from_record rebuilds (default '
                        'masters_build.VARIANTS_ONLY = Taper_R0,NoTaper_R0)')
    g.add_argument('--deep-installed-root', dest='deep_installed_root', metavar='DIR',
                   help='sources: the installed working tree holding <cell>/coadd_final/ (default paths.COADD_SB_ROOT)')
    g.add_argument('--deep-install-records', dest='deep_install_records', metavar='OUT',
                   help="sources: product root whose _records/install/ holds the cell's install record (default --out)")


def default_out(args):
    """Deep product root: --out, else the site file's MASTERS_GEN_ROOT for this host, else masters.DEEP_GEN_ROOT (production
    scratch generation root - writes refused while paths.PRODUCTION_WRITES is False)."""
    from exopipe.phase3.core import env
    c = _cfg()
    by_host = getattr(c.site, 'MASTERS_GEN_ROOT', None) or {}
    return by_host.get(env.host()) or c.masters.DEEP_GEN_ROOT


# =====================================================================================================================
# selection, records
# =====================================================================================================================

def select_cells(args):
    from exopipe.phase3.core import cells as C
    if getattr(args, 'epochs', None):
        raise Refusal('--epochs is refused for --tier deep: masters are rebuilt intermittently (DESIGN 12.2); the per-hour '
                      'chain reads the installed masters (frames, search, ...)')
    if getattr(args, 'nights', None):
        raise Refusal('--nights is refused for --tier deep: a deep master is built from every admitted night of the cell '
                      '(exclusions are data_decisions.csv rows, never a command-line subset)')
    if args.cells:
        cells = C.expand_cells(args.cells)
    elif getattr(args, 'lsts', None) or getattr(args, 'scopes', None):
        lsts = C.expand(args.lsts, universe=C.LSTS, kind='LST') if args.lsts else list(C.LSTS)
        scs = C.expand(args.scopes, universe=C.scopes(), kind='scope') if args.scopes else list(C.scopes())
        cells = [f'{l}_{s}' for l in lsts for s in scs]
    else:
        raise SystemExit('masters --tier deep: give --cells (e.g. 23h_23MHz, 04h_*, @list) or --lsts/--scopes')
    if getattr(args, 'exclude_cells', None):
        drop = set(C.expand_cells(args.exclude_cells))
        cells = [x for x in cells if x not in drop]
    for x in cells:
        C.parse_cell(x)
    return cells


def done_record(out, stage, cell):
    rec = record.load_record(out, 'masters', f'{stage}_{cell}')
    return rec if rec and rec.get('status') == 'done' else None


def exclusion_stale(out, cell):
    """Nights the cell's done build record admitted that data_decisions.csv excludes NOW ([] = still current, or no done
    build record). Exclusions are absolute: a product derived from such a build is never extended or resumed."""
    rec = done_record(out, 'build', cell)
    if not rec or not rec.get('admitted'):
        return []
    from exopipe.phase3.core import exclusions
    lst, sc = cell.split('_', 1)
    ex = exclusions.load()
    return [d for d in rec['admitted'] if ex.is_excluded(lst, sc, d)]


def require_stage(out, stage, cell, ran_now):
    """Refuse unless the stage this one depends on is done for the cell (its record), or ran done in this invocation.
    A record from an earlier invocation must still be current (DESIGN 11.5: the chain of records): every output it lists
    still exists (a --fresh --supersede stage-in moves the cell away), and - for the stages derived from the build - no
    night the build admitted is excluded now."""
    need = REQUIRES.get(stage)
    if not need:
        return None
    if ran_now.get((need, cell)) == 'done':
        rec = record.load_record(out, 'masters', f'{need}_{cell}')
    else:
        rec = done_record(out, need, cell)
        if rec is None:
            got = record.load_record(out, 'masters', f'{need}_{cell}')
            raise Refusal(f'{stage}: the {need} stage of {cell} is not done under {out} (record status '
                          f'{got.get("status") if got else "missing"}) - run --stages {need} first (spec 1.2)')
        gone = [o['path'] for o in rec.get('outputs') or [] if not os.path.exists(o['path'])]
        if gone:
            raise Refusal(f'{stage}: {len(gone)} output(s) of the done {need} record of {cell} are gone (e.g. {gone[0]}) - '
                          f'the cell was superseded or removed; re-run --stages {need} (DESIGN 11.5)')
    if stage in BUILD_DERIVED and ran_now.get(('build', cell)) != 'done':
        stale = exclusion_stale(out, cell)
        if stale:
            raise Refusal(f'{stage}: the build of {cell} admitted night(s) {stale} that data_decisions.csv excludes now - '
                          f'rebuild from stage-in (--fresh --supersede TAG); exclusions are absolute (G-M01, DESIGN 5 (d))')
    return rec


def _epoch_gate_carry():
    v = _cfg().masters_stagein.EPOCH_GATE_CARRY
    if v not in ('none', 'carry'):
        raise Refusal(f'masters_stagein.EPOCH_GATE_CARRY {v!r}: none (B-05 a) or carry (production)')
    return v


def _epoch_gate_source():
    v = _cfg().masters_build.EPOCH_GATE_SOURCE
    if v not in ('own-master', 'carried-record'):
        raise Refusal(f'masters_build.EPOCH_GATE_SOURCE {v!r}: own-master (B-05 a) or carried-record (production)')
    return v


def build_clean_start(cell_dir, sc):
    """[reasons] why the cell directory is not as stage-in (+ the flow step) left it: a product of an earlier build, or a
    stage-in reference that no longer equals its coadd_final_ref/ copy. The in-place co-add takes coadd_final/
    I_deep_Taper_R075_<scope>.fits as its alignment and flux reference (coadd._select_reference): after a build that file is
    the NEW master, so a second build in the same directory aligns to it (or moves the flows aside when its CONSTDIG
    differs). Production never meets this: a rebuild always starts from a clean scratch cell (rebuild_cell.sh:34) and a
    content-tested cell is never rebuilt in place (chain_cell.sh:23)."""
    from exopipe.phase3.lib.masters import deep_inplace as IP
    why = []
    for p in BUILD_PRODUCT_PATHS:              # an empty directory (a stage that refused before writing) is not a product
        q = os.path.join(cell_dir, p)
        if os.path.isdir(q) and not os.path.islink(q):
            if any(True for _ in os.scandir(q)):
                why.append(p + '/')
        elif os.path.lexists(q):
            why.append(p)
    for var in ('I_deep_Taper_R075', 'V_deep_Taper_R0'):
        ref = os.path.join(cell_dir, 'coadd_final_ref', f'{var}_{sc}.fits')
        cur = os.path.join(cell_dir, 'coadd_final', f'{var}_{sc}.fits')
        if os.path.exists(ref) and (not os.path.exists(cur) or IP.sha1_file(cur) != IP.sha1_file(ref)):
            why.append(f'coadd_final/{var}_{sc}.fits != the stage-in reference coadd_final_ref/{var}_{sc}.fits')
    return why


def not_here(path, what, stage):
    """Refuse when ``path`` is node-local to ANOTHER host (site NODE_LOCAL_ROOTS): the same path here is another disk or
    absent - chain_cell.sh:17 runs only on the host holding the working tree; a site check, never a host name in code."""
    from exopipe.phase3.core import env
    if not env.readable_here(path):
        raise Refusal(f'{stage}: {what} {path} is node-local to {env.node_local_holder(path)}, not this host '
                      f'({env.host()}) - run the {stage} there (chain_cell.sh:17)')


def free_space_gate(out):
    floor = (_cfg().masters.FREE_FLOOR_GB or {}).get('deep')
    if not floor:
        return None
    p = os.path.abspath(out)
    while not os.path.exists(p):
        p = os.path.dirname(p)
    free = shutil.disk_usage(p).free / 2 ** 30
    if free < floor:
        raise Refusal(f'{p} has {free:.0f} GB free (< {floor}) - not started (chain_cell.sh:19-20; G-M27)')
    return free


def parse_reference(text):
    t = (text or 'previous-master').strip()
    if t == 'previous-master':
        return ('previous-master', None)
    if t.startswith('path:'):
        return ('path', t[5:])
    if t.startswith('night:'):
        from exopipe.phase3.core import cells as C
        return ('night', C.parse_date(t[6:]))
    if t.startswith('constdig:'):
        raise SystemExit('--reference constdig:D is not implemented for the deep tier (spec 2: previous-master | path:P | '
                         'night:YYYY-MM-DD)')
    raise SystemExit(f'--reference {t!r}: expected previous-master, path:P or night:YYYY-MM-DD')


def _freq(scope):
    return scope[:-3] if scope.endswith('MHz') else scope


def _log(out, cell, name):
    """Log of an in-place call: outside the cell layout (install takes the cell directories as they are)."""
    return os.path.join(out, '_records', 'masters', 'logs', f'{name}_{cell}.log')


# =====================================================================================================================
# in-place setup (once per process)
# =====================================================================================================================

_SETUP = {}


def inplace(out):
    """Import the in-place masters modules once, with config applied (env-import keys before pb_apply's first import)."""
    from exopipe.phase3.lib.masters import deep_inplace as IP
    if 'mods' not in _SETUP:
        _SETUP['cleared_p3'] = IP.clear_inherited_p3()
        IP.set_import_env()
        from exopipe.phase3.lib.masters import coadd, membership, night_image_gate, epoch_gate   # noqa: F401
        _SETUP['mods'] = dict(coadd=coadd, membership=membership, night_image_gate=night_image_gate, epoch_gate=epoch_gate)
    co = _SETUP['mods']['coadd']
    co._update_workdir(out)
    co.DECISIONS_CSV = _cfg().paths.DATA_DECISIONS          # protected path (C-09, U-1c), set explicitly
    applied = IP.apply_attrs('masters_build') + IP.apply_attrs('masters_verify') + IP.apply_attrs('masters_fluxscale',
                                                                                                    only_modules=('membership',))
    IP.check_mirrors(modules=('coadd',))
    return _SETUP['mods'], applied


def pin_build_tables():
    from exopipe.phase3.lib.masters import deep_inplace as IP
    b = _cfg().masters_build
    return {'solar audit': IP.sha1_pin(b.SOLAR_AUDIT, b.SOLAR_AUDIT_SHA1, 'solar contamination audit (G-M11)'),
            'named out of master': IP.sha1_pin(b.NAMED_OUT_OF_MASTER, b.NAMED_OUT_OF_MASTER_SHA1,
                                               'named_out_of_master.csv (G-M12)'),
            'variant band medians': IP.sha1_pin(b.VARIANT_BAND_MEDIANS, b.VARIANT_BAND_MEDIANS_SHA1,
                                                'variant_band_medians.json (G-M17)')}


# =====================================================================================================================
# stage: stage-in
# =====================================================================================================================

def stage_in(cell, args, run, out, ran_now):
    from exopipe.phase3.lib.masters import deep_inplace as IP, deep_stagein as DS
    lst, sc = cell.split('_', 1)
    cell_dir = os.path.join(out, cell)
    mode, val = parse_reference(args.reference)
    with run.unit(f'stage-in_{cell}', tier='deep', stage='stage-in', cell=cell,
                  reference=f'{mode}:{val}' if val else mode) as u:
        free_space_gate(out)
        if os.path.exists(cell_dir):
            if args.fresh and getattr(args, 'supersede', None):
                u.note(f'earlier cell directory moved aside: {DS.supersede(cell_dir, out, args.supersede)}')
            else:
                u.refuse(f'{cell_dir} exists - a rebuild always starts from a clean cell directory (rebuild_cell.sh:34; '
                         f'G-M25); pass --fresh --supersede TAG to move it aside with a sha1 manifest')
        ov = DS.working_root_table()
        tables = DS.reference_tables()
        for t in (_cfg().masters_stagein.REF_OVERRIDE_TABLE, _cfg().masters_stagein.REF_OVERRIDE_I_TABLE,
                  _cfg().masters_stagein.REF_PREVGEN_TABLE):
            if t and os.path.exists(t):
                u.add_input(t)
        C = DS.working_cell(cell, ov)
        u.note(f'working cell {C}' + (' (ref_override.csv)' if cell in ov else ''))
        not_here(C, 'the working cell', 'stage-in')
        if mode == 'previous-master':
            entries = DS.resolve_reference(cell, C, tables=tables)
        else:
            ruling = run.waived(WAIVE_REFERENCE)
            if not ruling:
                u.refuse(f'--reference {mode}:{val} needs Gregg\'s ruling: --waive {WAIVE_REFERENCE} --ruling "<words, UTC, '
                         f'ledger ID>" (B-12, B-61)')
            u.check(WAIVE_REFERENCE, False, value=f'{mode}:{val}', detail=f'reference chosen by argument, not the '
                                                                          f'previous master (ruling recorded)')
            entries = DS.reference_from_path(val) if mode == 'path' else []
            if mode == 'night':
                u.note(f'--reference night:{val}: no reference master written; the build takes the fixed-table night {val}')
        for e in entries:
            if e['source']:
                u.add_input(e['source'])
        for st, var, _ in DS.REF_VARS:
            p = f'{C}/coadd_final/{var}_{sc}.fits'
            if os.path.exists(p):
                u.add_input(p)
        u.note('reference: ' + DS.describe(entries))
        carry = _epoch_gate_carry()
        egc = DS.epoch_gate_source(cell, C) if carry == 'carry' else None
        if carry == 'none':
            u.note('no epoch-gate record carried in (B-05 a: masters_stagein.EPOCH_GATE_CARRY none) - the build tests every '
                   'night against its own master')
        os.makedirs(os.path.join(cell_dir, 'coadd_final'), exist_ok=True)
        if egc:
            u.add_input(egc)
            dst = os.path.join(cell_dir, 'coadd_final', 'epoch_gate.csv')
            shutil.copy(egc, dst)                                    # cp (rebuild_cell.sh:42)
            fails = [r['date'] for r in csv.DictReader(open(dst)) if r.get('verdict') == 'FAIL']
            u.check('epoch_gate_carried', IP.sha1_file(dst) == IP.sha1_file(egc), value=len(fails),
                    detail=f'epoch-gate record carried in from {egc}: FAIL {fails}')
        elif carry == 'carry':
            u.note('no epoch-gate record to carry (rebuild_cell.sh:42 skips)')
        u.note(f'next: exopipe3 flow --tier deep --cells {cell} --out {out} (copies the stored flows of {C}/flow and '
               f'computes the stale ones against this reference), then masters --stages build,...')
        for e in entries:
            if e['data'] is None and e['none_reason']:
                u.note(e['none_reason'])
        written = DS.write_reference(cell_dir, sc, entries)
        for (st, p, q, hist), e in zip(written, [e for e in entries if e['data'] is not None]):
            u.note(f'reference {e["var"]}: {e["src"]}; CONSTDIG {e["constdig"]} NCOADD {e["ncoadd"]}')
            prev = f'{C}/coadd_final/{e["var"]}_{sc}.fits' if mode == 'previous-master' else None
            u.add_output(q, 'reference_master', source=e['source'], factor=e['factor'], previous=prev,
                         generation=e.get('generation'))
        if not written and mode != 'night':
            u.refuse('no reference master written (G-M04)')
        if egc:          # the carried record (the co-add final stage reads it: G-M10)
            u.add_output(os.path.join(cell_dir, 'coadd_final', 'epoch_gate.csv'), 'csv_table',
                         required_columns=('date', 'i_ratio', 'verdict', 'master_md5'), min_rows=1)
        elif mode == 'night' and carry == 'carry':
            raise NoInput('--reference night: nothing written for the cell (no reference, no epoch-gate record)')
        elif mode == 'night':
            u.check('stage-in.nothing_to_stage', True, value=0,
                    detail='--reference night: no reference master (the build takes the fixed-table night) and no carried '
                           'epoch-gate record (B-05 a) - the cell directory is ready for the flow step')


# =====================================================================================================================
# stage: build (provisional + final co-add)
# =====================================================================================================================

def _refuse_or_noinput(u, tee, stage, noinput=('No images for',), refuse=('REFUSING', 'Only', 'Too few', 'Reference',
                                                                            'No flow/', 'No provisional', 'REJECTED')):
    """The in-place co-add returned None: choose NoInput (no Phase-2 image) or Refusal and copy its printed reason. The
    verdict came from the return value; the captured lines only name the reason."""
    line = tee.find(*noinput)
    if line:
        raise NoInput(f'{stage}: {line}')
    line = tee.find(*refuse)
    raise Refusal(f'{stage} returned no product: {line or "(no reason printed)"}')


def _build_outputs(cell_dir, sc):
    out = []
    fd = os.path.join(cell_dir, 'coadd_final')
    for v in ('I_deep_Taper_R075', 'V_deep_Taper_R0', 'I_deep_Taper_R0', 'I_deep_NoTaper_R0', 'I_deep_NoTaper_R075'):
        p = os.path.join(fd, f'{v}_{sc}.fits')
        if os.path.exists(p):
            out.append((p, 'master'))
    for p in sorted(glob.glob(os.path.join(fd, 'loo', f'*_loo_*_{sc}.fits'))):
        out.append((p, 'loo_master'))
    for p in (os.path.join(fd, 'weights.csv'), os.path.join(cell_dir, 'residuals_provisional.csv')):
        if os.path.exists(p):
            out.append((p, 'weights_table'))
    if os.path.exists(os.path.join(fd, 'stage2_verdict.csv')):
        out.append((os.path.join(fd, 'stage2_verdict.csv'), 'membership_record'))
    for p in sorted(glob.glob(os.path.join(cell_dir, 'subtracted', f'*_{sc}.fits'))):
        out.append((p, 'coadd_epoch'))
    return out


def build_nights(cell, out):
    """(coadd module, Phase-2 dates, core Exclusions, co-add exclusion state, admitted nights) of a deep cell: the co-add's
    own finder and BOTH exclusion readers (G-M01). Used by the build and by its --resume decision (admitted set now)."""
    from exopipe.phase3.lib.masters import deep_inplace as IP
    lst, sc = cell.split('_', 1)
    mods, _ = inplace(out)
    co = mods['coadd']
    dates = co.discover_dates(lst, _freq(sc))
    nights = [d for d, _, _ in dates]
    ex, excl = IP.exclusion_readers_agree(lst, sc, nights, co) if nights else (None, None)
    return co, dates, ex, excl, (ex.admitted(lst, sc, nights) if nights else [])


def build(cell, args, run, out, ran_now):
    from exopipe.phase3.lib.masters import deep_inplace as IP
    lst, sc = cell.split('_', 1)
    freq = _freq(sc)
    cell_dir = os.path.join(out, cell)
    with run.unit(f'build_{cell}', tier='deep', stage='build', cell=cell) as u:
        prev = require_stage(out, 'build', cell, ran_now)
        flow_note = flow_gate(out, cell, u)          # cheap gates first: G-M05 at record level
        dirty = build_clean_start(cell_dir, sc)
        if dirty:
            u.refuse(f'{cell_dir} is not as stage-in left it ({"; ".join(dirty[:4])}): a build runs once on a clean cell '
                     f'(rebuild_cell.sh:34; G-M25) - its reference would be the earlier build\'s master. Re-run '
                     f'`masters --stages stage-in --fresh --supersede TAG`, the flow step, then the build')
        own = _epoch_gate_source() == 'own-master'
        carried = os.path.join(cell_dir, 'coadd_final', 'epoch_gate.csv')
        if own and os.path.exists(carried):
            u.refuse(f'{carried} exists: masters_build.EPOCH_GATE_SOURCE own-master builds without a carried record (B-05 a) - '
                     f'the co-add would drop its FAIL nights before the gate. Re-run `masters --stages stage-in --fresh '
                     f'--supersede TAG` with masters_stagein.EPOCH_GATE_CARRY none, the flow step, then the build')
        free_space_gate(out)
        mods, applied = inplace(out)
        co = mods['coadd']
        pins = pin_build_tables()
        if _SETUP.get('cleared_p3'):
            u.note(f'inherited P3_* variables removed before the in-place import: {sorted(_SETUP["cleared_p3"])}')
        u.note(f'in-place attributes: {", ".join(f"{k}={v!r}" for k, v in applied)[:2000]}')
        u.note(f'pinned tables: {pins}')
        for p in (_cfg().masters_build.SOLAR_AUDIT, _cfg().masters_build.NAMED_OUT_OF_MASTER,
                  _cfg().masters_build.VARIANT_BAND_MEDIANS, _cfg().masters_build.VLSSR_PEAKS, _cfg().paths.DATA_DECISIONS):
            if not os.path.exists(p):
                u.refuse(f'input table {p} missing')
            if p != _cfg().paths.DATA_DECISIONS:   # the exclusion state is the ADMITTED set (set_admitted), not the file
                u.add_input(p)
        u.note(f'data_decisions.csv sha1 {IP.sha1_file(_cfg().paths.DATA_DECISIONS)} (the admitted set below is its effect '
               f'on this cell; --resume compares the admitted set)')
        reference = (prev or {}).get('meta', {}).get('reference', 'previous-master')
        _co, dates, ex, excl, admitted = build_nights(cell, out)
        if not dates:
            raise NoInput(f'No Phase-2 images for {lst} {sc} under {co.PHASE2_ROOT}')
        nights = [d for d, _, _ in dates]
        u.set_admitted(admitted)
        u.note(f'{len(nights)} Phase-2 nights, {len(admitted)} admitted; excluded {sorted(set(nights) - set(admitted))}')
        if not admitted:
            raise NoInput(f'every Phase-2 night of {lst} {sc} is excluded in data_decisions.csv')
        for d, _sd, img in dates:
            if d in admitted:
                u.add_input(img)
        # identity of what stage-in and the flow step left: the flows, the reference COPIES (coadd_final/{I,V} are the same
        # bytes - build_clean_start - but the build overwrites them with the new masters) and the carried epoch-gate record
        for p in (sorted(glob.glob(os.path.join(cell_dir, 'flow', '*'))) + sorted(glob.glob(os.path.join(cell_dir, 'coadd_final_ref', '*')))
                  + [os.path.join(cell_dir, 'coadd_final', 'epoch_gate.csv')]):
            if os.path.isfile(p):
                u.add_input(p)
        u.note(f'flow step: {flow_note}')
        factor = None
        if own:                 # the gate evaluates in the frame of the baked generation (verify T6, install)
            factor, what = bake_factor(lst, sc)
            for p in (_cfg().masters_fluxscale.SCALE_TABLE, _cfg().masters_fluxscale.FLUXCORR_TABLE):
                u.add_input(p)
            u.note(f'B-05 epoch gate frame: x{factor:.5f} ({what})' if factor else
                   f'B-05 epoch gate frame: unbaked products - {what} (the fluxscale stage refuses the cell)')
        extra = {}
        if reference.startswith('night:'):
            co.REFERENCE_DATES = dict(co.REFERENCE_DATES, **{lst: reference.split(':', 1)[1]})
            u.note(f'reference night from stage-in: {reference}')
        with IP.call_env(['masters', 'masters_build'], extra=extra) as env, \
                IP.capture_stdout(_log(out, cell, 'build')) as tee:
            u.note(f'call environment: {env}')
            try:
                prov = co.run_provisional_lst_freq(lst, freq, excl)
                if prov is None:
                    _refuse_or_noinput(u, tee, 'co-add provisional')
                fin = co.run_final_lst_freq(lst, freq, excl)
                if fin is None:
                    _refuse_or_noinput(u, tee, 'co-add final')
                if fin.get('status') == 'skipped':
                    u.refuse('co-add skipped the build (.done + current COADDOP) - FORCE_REBUILD not applied')
                if own:                                      # B-05 (a): gate on this build's master, bounded passes
                    from exopipe.phase3.lib.masters import epoch_gate_passes as EGP

                    def _rerun():
                        r = co.run_final_lst_freq(lst, freq, excl)
                        if r is None:
                            _refuse_or_noinput(u, tee, 'co-add final (B-05 rebuild pass)')
                        if r.get('status') == 'skipped':
                            u.refuse('co-add skipped a B-05 rebuild pass (.done + current COADDOP)')
                        return r
                    fin = EGP.run_passes(u, _rerun, cell_dir, sc, out, fin, run.run_id,
                                         lambda: mods['membership'].verify(cell_dir, excl), factor=factor)
            except RuntimeError as e:                 # every in-place gate that raises (G-M02/04/05/09/16/19)
                u.refuse(str(e)[:1500])
        u.note(f'co-add final: {fin.get("n_good")}/{fin.get("n_total")} nights in the master')
        grid = os.path.join(cell_dir, 'refusals_grid.csv')
        if os.path.exists(grid):
            u.note('grid refusals (G-M03): ' + ' | '.join(l.strip() for l in open(grid))[:1000])
        for p, cls in _build_outputs(cell_dir, sc):
            u.add_output(p, cls, cell=cell)


def flow_record_path(out, cell):
    """The flow step's record of a deep cell (one unit per cell, cumulative over its stages; lib/flow/steps_deep.py)."""
    return os.path.join(out, '_records', 'flow', f'{cell}.json')


def flow_gate(out, cell, u=None):
    """G-M05 at record level: the cell's flow/ directory must exist under --out (the flow step's stage-in wrote it) and the
    flow step's record, when present, must be done. Without a record (window: flows staged by other means) the in-place
    freshness gate of the provisional stage decides (missing or stale flow -> the co-add refuses the cell)."""
    fdir = os.path.join(out, cell, 'flow')
    if not os.path.isdir(fdir):
        raise Refusal(f'no {fdir}: run `exopipe3 flow --tier deep --cells {cell} --out {out}` after masters stage-in '
                      f'(the flows are the flow step\'s; G-M05)')
    p = flow_record_path(out, cell)
    if not os.path.exists(p):
        return f'no record {p} - the in-place freshness gate decides (window)'
    try:
        rec = json.load(open(p))
    except (OSError, ValueError) as e:
        raise Refusal(f'flow record {p} unreadable ({e}) - G-M05')
    if rec.get('status') != 'done':
        raise Refusal(f'flow record {p} status {rec.get("status")!r} ({rec.get("reason", "")[:200]}) - G-M05')
    if u is not None:
        u.add_input(p)
    return f'record done ({p}; stages {rec.get("meta", {}).get("stages")})'


# =====================================================================================================================
# stage: pbmaps
# =====================================================================================================================

def siblings_expected(fd, sc):
    """pbcorr siblings pbmaps must leave: the five I/V variants (masters_pbmaps.SIBLINGS_AFTER_PBMAPS), + the cleaned V when
    pbmaps runs again after vclean (do_cell regenerates its sibling too: V_deep_Taper_R0_clean is in PB_VARIANTS)."""
    return (_cfg().masters_pbmaps.SIBLINGS_AFTER_PBMAPS
            + int(os.path.exists(os.path.join(fd, f'V_deep_Taper_R0_clean_{sc}.fits'))))


def pb_qa_missing(fd, sc, variants):
    """[(variant, nights with a Phase-2 noise file, nights without)] of the pbcorr siblings of ``variants`` in ``fd``, from the
    cards pb_apply.apply_pbcorr stamps (PBNDATE, PBMISS): a night without its QA thermal-noise file is left out of the PB
    weights (pb_apply.subband_weights)."""
    from astropy.io import fits
    out = []
    for v in variants:
        p = os.path.join(fd, f'{v}_pbcorr_{sc}.fits')
        if os.path.exists(p):
            h = fits.getheader(p)
            out.append((v, int(h.get('PBNDATE', -1)), int(h.get('PBMISS', 0) or 0)))
    return out


def pb_qa_inputs(fd, lst, sc):
    """[(beam master, nights with a Phase-2 noise file, nights without)] of the I masters the PB maps are made from
    (pb_apply.wideband_pb: subband_weights over constituent_dates of I_deep_Taper_<robust>), read BEFORE pbmaps writes anything
    (review 2026-10-14 m4: the refusal came after do_cell had overwritten the maps). The siblings' PBNDATE / PBMISS cards
    (pb_qa_missing) are the same split, compared after do_cell."""
    from exopipe.phase3.lib.masters import pb_apply as PB
    out = []
    for rob in ('R075', 'R0'):
        b = os.path.join(fd, f'I_deep_Taper_{rob}_{sc}.fits')
        if os.path.exists(b):
            _w, used, missing = PB.subband_weights(lst, PB.constituent_dates(b))
            out.append((f'I_deep_Taper_{rob}', len(used), len(missing)))
    return out


def pb_qa_gate(u, cell, sc, rows, max_frac):
    """B-30 (Gregg 2026-10-14): every night missing from the PB weights is COUNTED (fallbacks.PB_QA_MISSING_FALLBACK, the
    survey variant's count once per cell); a WIDEBAND scope - its member subbands mixed by those weights; a single subband's
    weights cancel - refuses when a sibling misses more than ``max_frac`` of its nights (masters_pbmaps.PB_QA_MISSING_MAX_FRAC,
    PROPOSED 0.25; production 1.0 = no limit)."""
    from exopipe.phase3.core import fallback
    u.meta['pb_qa_nights'] = {v: {'with_noise_file': nd, 'missing': nm} for v, nd, nm in rows}
    survey = next(((v, nd, nm) for v, nd, nm in rows if v == 'I_deep_Taper_R075'), rows[0] if rows else None)
    if survey and survey[2]:
        fallback.site('fallbacks.PB_QA_MISSING_FALLBACK', f'{cell}: {survey[2]} of {survey[1] + survey[2]} nights lack their '
                      f'Phase-2 noise file - left out of the PB weights', n=survey[2])
    if sc.endswith('MHz'):
        return
    over = [(v, nd, nm) for v, nd, nm in rows if nm and nm / max(1, nd + nm) > float(max_frac)]
    if over:
        v, nd, nm = over[0]
        u.refuse(f'PB weights: {v} misses the Phase-2 noise file of {nm} of {nd + nm} nights ({nm / (nd + nm):.0%}), above '
                 f'masters_pbmaps.PB_QA_MISSING_MAX_FRAC {float(max_frac):.0%} - a wideband PB mixes its subbands by those '
                 f'weights')


def pbmaps(cell, args, run, out, ran_now):
    from astropy.io import fits
    from exopipe.phase3.lib.masters import deep_inplace as IP
    lst, sc = cell.split('_', 1)
    fd = os.path.join(out, cell, 'coadd_final')
    s = _cfg().masters_pbmaps
    with run.unit(f'pbmaps_{cell}', tier='deep', stage='pbmaps', cell=cell) as u:
        require_stage(out, 'pbmaps', cell, ran_now)
        inplace(out)
        from exopipe.phase3.lib.masters import pbcorr_masters, pb_apply
        IP.check_import_env()
        IP.apply_attrs('masters_pbmaps')
        IP.check_mirrors(modules=('pb_apply',))
        bsrc = os.path.join(fd, f'I_deep_Taper_R075_{sc}.fits')
        if os.path.exists(bsrc) and fits.getheader(bsrc).get('MDWARP'):
            u.refuse('master-dewarped cell: its PB maps carry the dewarp screen and are made by the dewarp install, not '
                     'here (G-M21; B-65 default)')
        plains = [v for v in s.PB_VARIANTS if os.path.exists(os.path.join(fd, f'{v}_{sc}.fits'))]
        if not plains:
            raise NoInput(f'no master variant in {fd}')
        for v in plains:
            u.add_input(os.path.join(fd, f'{v}_{sc}.fits'))
        for p in [s.PB_RESIDUAL_V2_TABLE] + sorted(glob.glob(os.path.join(s.PB_LIBRARY, 'PB_*_deep.fits'))):
            u.add_input(p)
        for name in s.PB_MAP_NAMES.values():
            u.plan(os.path.join(fd, name.format(scope=sc)), 'pb_map')
        for v in plains:
            u.plan(os.path.join(fd, f'{v}_pbcorr_{sc}.fits'), 'pbcorr_sibling')
        pre = pb_qa_inputs(fd, lst, sc)                                   # WAVE2 B-30: before anything is written
        pb_qa_gate(u, cell, sc, pre, s.PB_QA_MISSING_MAX_FRAC)
        with IP.capture_stdout(_log(out, cell, 'pbmaps')):
            n_ok, n_skip, errs = pbcorr_masters.do_cell(out, cell)
            print('do_cell', (n_ok, n_skip, errs))
        u.note(f'do_cell ({n_ok}, {n_skip}, {errs})')
        if errs or n_skip:
            u.refuse(f'pbcorr_masters.do_cell: {n_skip} corrupt variant(s) skipped, errors {errs[:4]} (G-M21)')
        for name in s.PB_MAP_NAMES.values():
            u.add_output(os.path.join(fd, name.format(scope=sc)), 'pb_map', cell=cell)
        for v in plains:
            u.add_output(os.path.join(fd, f'{v}_pbcorr_{sc}.fits'), 'pbcorr_sibling', cell=cell)
        cards = dict((v, (nd, nm)) for v, nd, nm in pb_qa_missing(fd, sc, plains))      # WAVE2 B-30: the cards cross-check
        for v, nd, nm in pre:
            if v in cards and cards[v] != (nd, nm):
                u.note(f'PB weights: {v} sibling cards PBNDATE/PBMISS {cards[v]} differ from the pre-check {(nd, nm)} '
                       f'(the Phase-2 QA tree changed during the stage?)')
        nsib = len(glob.glob(os.path.join(fd, f'*_pbcorr_{sc}.fits')))
        want = siblings_expected(fd, sc)
        u.check('masters_pbmaps.siblings_after_pbmaps', nsib == want, value=nsib,
                detail=f'{nsib} siblings after pbmaps ({want} expected: the five I/V variants'
                       f'{" + the cleaned V" if want > s.SIBLINGS_AFTER_PBMAPS else ""}; check_rebuild.py T4 expects 6 '
                       f'once the cleaned V exists)')


# =====================================================================================================================
# stage: vclean
# =====================================================================================================================

def vclean(cell, args, run, out, ran_now):
    from exopipe.phase3.core import fitsio
    from exopipe.phase3.lib.masters import deep_inplace as IP
    lst, sc = cell.split('_', 1)
    fd = os.path.join(out, cell, 'coadd_final')
    with run.unit(f'vclean_{cell}', tier='deep', stage='vclean', cell=cell) as u:
        require_stage(out, 'vclean', cell, ran_now)
        inplace(out)
        from exopipe.phase3.lib.masters import vclean_masters
        IP.check_import_env()
        IP.apply_attrs('masters_vclean')
        for v in ('V_deep_Taper_R0', 'I_deep_Taper_R075', 'I_deep_Taper_R0', 'PB_deep'):
            p = os.path.join(fd, f'{v}_{sc}.fits')
            if os.path.exists(p):
                u.add_input(p)
        u.add_input(vclean_masters.SV2_SCRIPT)
        tmp = fitsio.scratch_dir(f'vclean_{cell}', args.work)
        u.plan(os.path.join(fd, f'V_deep_Taper_R0_clean_{sc}.fits'), 'vclean_master')
        u.plan(os.path.join(fd, f'V_deep_Taper_R0_clean_pbcorr_{sc}.fits'), 'pbcorr_sibling')
        try:
            with IP.capture_stdout(_log(out, cell, 'vclean')):
                r = vclean_masters.build_vclean_cell(out, cell, tmp_root=tmp, skip_existing=False)
                print(r)
        finally:
            fitsio.cleanup(tmp)
        u.note(f'build_vclean_cell -> {r}')
        if r[1] != 'OK':
            u.refuse(f'vclean: {r[1]} (G-M22)')
        u.add_output(os.path.join(fd, f'V_deep_Taper_R0_clean_{sc}.fits'), 'vclean_master', cell=cell)
        u.add_output(os.path.join(fd, f'V_deep_Taper_R0_clean_pbcorr_{sc}.fits'), 'pbcorr_sibling', cell=cell)


# =====================================================================================================================
# stage: fluxscale
# =====================================================================================================================

SHFLUX_COLUMNS = ['lst', 'scope', 'status', 'scale', 'err', 'nsrc', 'srcs', 'freq_hz', 'freq_src', 'rms_jy', 'n_written',
                  'detail', 'measured_at']     # phase3_master_shflux.py main() cols


def scale_row(lst, scope):
    """The shflux_table row production builds from the scale table (phase3_master_shflux.py main(), scale-table branch)."""
    s = _cfg().masters_fluxscale
    tab = {r['lst']: r for r in csv.DictReader(open(s.SCALE_TABLE))}
    t = tab.get(lst)
    return dict(lst=lst, scope=scope, status='OK' if t else 'REFUSED no factor for this LST in the scale table',
                scale=(t['scale'] if t else ''), err=(t['err'] if t else ''), nsrc=(t['nsrc'] if t else 0),
                srcs=(f"VLSSr {t['window']}" if t else ''), freq_hz=s.SCALE_MODEL_FREQ_HZ,
                freq_src=(t.get('ref', 'VLSSr 73 MHz subband, one factor per LST') if t else ''), rms_jy='',
                detail=f'scale table {os.path.basename(s.SCALE_TABLE)}', measured_at=(t['measured_at'] if t else ''))


def bake_factor(lst, scope):
    """(factor, what): SHSCALE[lst] x FLUXCORR[scope] as the fluxscale stage will bake it into the pixels (shflux.apply:
    float32(scale x corr)), from the same pinned tables (refuses on a pin mismatch, as the fluxscale stage does); (None, why)
    when the fluxscale stage would refuse the cell."""
    from exopipe.phase3.lib.masters import deep_inplace as IP, flux_correction as FC
    s = _cfg().masters_fluxscale
    IP.sha1_pin(s.SCALE_TABLE, s.SCALE_TABLE_SHA1, 'scale table v4 (G-M23)')
    IP.sha1_pin(s.FLUXCORR_TABLE, s.FLUXCORR_TABLE_SHA1, 'flux correction table v2 (G-M23)')
    row = scale_row(lst, scope)
    corr_t = FC.load_table(s.FLUXCORR_TABLE)
    if row['status'] != 'OK':
        return None, row['status']
    if scope not in corr_t:
        return None, f'no flux-correction row for {scope} in {s.FLUXCORR_TABLE}'
    return float(row['scale']) * float(corr_t[scope]['corr']), f"SHSCALE {row['scale']} x FLUXCORR {corr_t[scope]['corr']}"


def append_shflux_row(table, row):
    new = not os.path.exists(table)
    with open(table, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=SHFLUX_COLUMNS, extrasaction='ignore')
        if new:
            w.writeheader()
        w.writerow(row)


def already_scaled(cell_dir):
    """Defect D4 (pure): a product of coadd_final/ or subtracted/ that already carries SHSCALE or FLUXBAKE would be scaled
    twice by shflux.apply - list them (PB maps excepted)."""
    from exopipe.phase3.core.record import fits_primary_cards
    s = _cfg().masters_fluxscale
    bad = []
    for sub in s.SCALED_DIRS:
        for p in glob.glob(os.path.join(cell_dir, sub, '**', '*.fits'), recursive=True):
            if os.path.basename(p).startswith(tuple(s.UNSCALED_PREFIX)):
                continue
            cards = fits_primary_cards(p)
            if cards.get('SHSCALE') is not None or cards.get('FLUXBAKE'):
                bad.append(os.path.relpath(p, cell_dir))
    return bad


def fluxscale(cell, args, run, out, ran_now):
    from exopipe.phase3.lib.masters import deep_inplace as IP
    lst, sc = cell.split('_', 1)
    cell_dir = os.path.join(out, cell)
    s = _cfg().masters_fluxscale
    with run.unit(f'fluxscale_{cell}', tier='deep', stage='fluxscale', cell=cell) as u:
        require_stage(out, 'fluxscale', cell, ran_now)
        inplace(out)
        from exopipe.phase3.lib.masters import shflux, flux_correction as FC
        IP.apply_attrs('masters_fluxscale')
        IP.sha1_pin(s.SCALE_TABLE, s.SCALE_TABLE_SHA1, 'scale table v4 (G-M23)')
        IP.sha1_pin(s.FLUXCORR_TABLE, s.FLUXCORR_TABLE_SHA1, 'flux correction table v2 (G-M23)')
        u.add_input(s.SCALE_TABLE)
        u.add_input(s.FLUXCORR_TABLE)
        corr_t = FC.load_table(s.FLUXCORR_TABLE)
        if sc not in corr_t:
            u.refuse(f'no flux-correction row for {sc} in {s.FLUXCORR_TABLE} (G-M23)')
        twice = already_scaled(cell_dir)
        if twice:
            u.refuse(f'{len(twice)} source product(s) already carry SHSCALE/FLUXBAKE (would be scaled twice; defect D4): {twice[:4]}')
        for sub in s.SCALED_DIRS:
            for p in sorted(glob.glob(os.path.join(cell_dir, sub, '**', '*.fits'), recursive=True)):
                u.add_input(p)
        row = scale_row(lst, sc)
        row['n_written'] = 0
        table = os.path.join(out, 'shflux_table.csv')
        with IP.capture_stdout(_log(out, cell, 'fluxscale')):
            if row['status'] == 'OK':
                row['n_written'] = shflux.apply(lst, sc, out, row, s.OUTGEN, corr_t)
            append_shflux_row(table, row)
            print(f"{lst} {sc}: {row['status']} scale {row['scale']} +- {row['err']} (N {row['nsrc']} {row['srcs']}; "
                  f"{row['freq_src']}; written {row['n_written']}) | {row['detail'][:200]}", flush=True)
        u.note(f"shflux row: {row['status']} scale {row['scale']} written {row['n_written']} -> {table}")
        if row['status'] != 'OK':
            u.refuse(f"flux apply: {row['status']} (G-M23)")
        if not row['n_written']:
            u.refuse('flux apply wrote 0 files')
        u.add_output(os.path.join(cell_dir, s.OUTGEN), 'master_generation', cell=cell)


# =====================================================================================================================
# stage: verify (T1-T8 alone; replaces REBUILD_CONTENT_OK)
# =====================================================================================================================

def verify(cell, args, run, out, ran_now):
    from exopipe.phase3.lib.masters import deep_inplace as IP
    lst, sc = cell.split('_', 1)
    cell_dir = os.path.join(out, cell)
    gen = os.path.join(cell_dir, _cfg().masters_fluxscale.OUTGEN)
    v = _cfg().masters_verify
    with run.unit(f'verify_{cell}', tier='deep', stage='verify', cell=cell) as u:
        require_stage(out, 'verify', cell, ran_now)
        inplace(out)
        got = IP.sha1_file(v.DEWARP_SCREEN_RECORD) if os.path.exists(v.DEWARP_SCREEN_RECORD) else None
        u.add_input(v.DEWARP_SCREEN_RECORD)
        u.note(f'screen record {v.DEWARP_SCREEN_RECORD} sha1 {got} (pin {v.DEWARP_SCREEN_RECORD_SHA1}'
               + ('' if got == v.DEWARP_SCREEN_RECORD_SHA1 else '; CHANGED since the pin - recorded, the record moves at '
                                                                 'every dewarp install') + ')')
        for p in glob.glob(os.path.join(gen, '**', '*.fits'), recursive=True):
            u.add_input(p)
        for name in ('weights.csv', 'night_image_gate.csv'):
            u.add_input(os.path.join(cell_dir, 'coadd_final', name))
        u.add_input(os.path.join(cell_dir, 'residuals_provisional.csv'))
        u.add_output(gen, 'master_generation', cell=cell)
        from exopipe.phase3.lib.masters.tests import _integrity
        for p in sorted(glob.glob(os.path.join(gen, 'coadd_final', '*_deep_*_shflux_*.fits'))):
            if '_pbcorr_' in os.path.basename(p):
                continue
            ok, val, det = _integrity(p)
            u.check('masters_verify.integrity', ok, value=val, detail=f'{os.path.basename(p)}: {det}', path=p)


# =====================================================================================================================
# stage: sources (after dewarp + install)
# =====================================================================================================================

def _install_records(root, cell):
    """The install step's record of the cell: <root>/_records/install/<cell>.json (one unit per cell for its stages
    plan..verify; install spec section 1). [] when absent or unreadable."""
    p = os.path.join(root, '_records', 'install', f'{cell}.json')
    try:
        return [json.load(open(p))]
    except (OSError, ValueError):
        return []


def p3s5_parents():
    """exopipe_ops/p3s5_parents.py imported IN PROCESS (spec 7.2; crosscheck group 8: masters `sources` is the only builder).
    Not a core/legacy.run child: the script hard-codes ROOT (the installed tree) and TMP (/fast/gh/main/phase3/..., a
    production root on lwacalim07) and takes neither from argv or the environment - in process the step sets P.ROOT,
    P.COLLECT and P.TMP. Its old-name imports (exopipe.phase3.phase3_deep, phase3_tenmin) resolve to the branch's modules
    through core/_shim's finder; the module's sha1 enters the record's code fingerprint (it lives under a legacy root)."""
    from exopipe.phase3.core.env import legacy_root
    from exopipe.phase3.lib.masters import deep_inplace as IP
    want = os.path.join(legacy_root('exopipe_ops'), 'p3s5_parents.py')
    with IP.saved_sys_path():
        import p3s5_parents as P
    if os.path.realpath(P.__file__) != os.path.realpath(want):
        raise Refusal(f'p3s5_parents imported from {P.__file__}, not the legacy root {want}')
    return P


def installed_cell_violations(installed, cell):
    """membership.verify + membership.verify_products of the INSTALLED cell - production's `python -m exopipe.phase3.membership
    verify /data07/.../<cell>` (install_cell.sh:70-71), which stops the chain before the parent catalogue is built. Run here
    on every sources unit (also when the install record is waived during the window, and after an exclusion added since the
    install: a catalogue is never built on a master holding an excluded constituent)."""
    from exopipe.phase3.lib.masters import membership as MB
    cd = os.path.join(installed, cell)
    return list(MB.verify(cd)) + list(MB.verify_products(cd))


def sources(cell, args, run, out, ran_now):
    from astropy.io import fits
    from exopipe.phase3.core import fitsio
    from exopipe.phase3.lib.masters import deep_inplace as IP, membership as MB
    lst, sc = cell.split('_', 1)
    s = _cfg().masters_sources
    installed = args.deep_installed_root or _cfg().paths.COADD_SB_ROOT
    dest = args.out or s.DEEP_COLLECT
    with run.unit(f'sources_{cell}', tier='deep', stage='sources', cell=cell) as u:
        IP.refuse_production_write(os.path.join(dest, 'parent_catalogs'), 'sources')
        recs = _install_records(args.deep_install_records or out, cell)
        done = [r for r in recs if r.get('status') == 'done']
        if not done:
            if not run.waived(WAIVE_INSTALL_RECORD):
                u.refuse(f'no done install record for {cell} under {args.deep_install_records or out}/_records/install '
                         f'(G-M26: the master catalogue is built from the INSTALLED, dewarped master). During the window '
                         f'cells installed by production have no install-step record: --waive {WAIVE_INSTALL_RECORD} '
                         f'--ruling "<words>" to accept the installed tree as it is')
            u.check(WAIVE_INSTALL_RECORD, False, detail=f'{len(recs)} install record(s), none done; the installed tree '
                                                        f'{installed} accepted by ruling')
        master = os.path.join(installed, cell, 'coadd_final', f'I_deep_Taper_R075_{sc}.fits')
        not_here(master, 'the installed master', 'sources')
        if not os.path.exists(master):
            u.refuse(f'installed master {master} missing')
        u.add_input(master)
        h = fits.getheader(master)
        inplace(out)                       # the co-add's exclusion reader and membership's config values for the check below
        viol = installed_cell_violations(installed, cell)
        u.check('masters_sources.installed_cell_membership', not viol, value=len(viol),
                detail=('; '.join(viol[:3]) if viol else f'membership.verify + verify_products of {installed}/{cell}: PASS'))
        if viol:                           # a gate, not a content test: never waivable (an excluded constituent)
            u.refuse(f'the installed cell fails membership.verify / verify_products ({len(viol)}: {viol[0][:300]}) - no '
                     f'parent catalogue on it (install_cell.sh:70-71 stops the chain before p3s5_parents; G-M26)')
        if s.REQUIRE_DEWARP_SCREEN:
            rec = _cfg().masters_verify.DEWARP_SCREEN_RECORD
            rows = MB._rcsv(rec)
            if not rows:
                u.refuse(f'screen record {rec} unreadable')
            u.add_input(rec)
            applied = (lst, sc) in {(r['lst'], r['scope']) for r in rows if r.get('applied', '1') == '1'}
            if applied and not h.get('MDWARP'):
                u.refuse(f'the screen record says the RBF master dewarp was applied to {cell} but the installed master '
                         f'carries no MDWARP (G-M26; DESIGN 11.2: sources after dewarp + install)')
            u.note(f'screen record: applied={applied}; master MDWARP {h.get("MDWARP")!r}')
        tmp = fitsio.scratch_dir(f'parents_{cell}', args.work)
        out_csv = os.path.join(dest, 'parent_catalogs', f'master_R075_{cell}.gaul.csv')
        u.plan(out_csv, 'master_catalog')
        try:
            P = p3s5_parents()
            P.ROOT, P.COLLECT, P.TMP = installed, dest, tmp
            u.add_input(P.__file__)
            with IP.call_env(['masters_sources']), IP.capture_stdout(_log(out, cell, 'sources')):
                try:
                    msg = P.one(cell)
                    print(msg, flush=True)
                except Exception as e:                       # p3s5_parents prints FAILED and exits 0 (install_cell.sh:26)
                    u.refuse(f'P3S5-PARENT-FAILED {cell}: {type(e).__name__}: {e} (G-M26)')
            u.note(msg)
        finally:
            fitsio.cleanup(tmp)
        u.add_output(out_csv, 'master_catalog', master=master, installed_root=installed)


# =====================================================================================================================
# stage: variants (optional)
# =====================================================================================================================

def variants(cell, args, run, out, ran_now):
    from exopipe.phase3.lib.masters import deep_inplace as IP
    lst, sc = cell.split('_', 1)
    freq = _freq(sc)
    vks = tuple(v for v in (args.deep_variants.split(',') if args.deep_variants else _cfg().masters_build.VARIANTS_ONLY) if v)
    with run.unit(f'variants_{cell}', tier='deep', stage='variants', cell=cell, variants=list(vks)) as u:
        require_stage(out, 'variants', cell, ran_now)
        mods, _ = inplace(out)
        co = mods['coadd']
        pin_build_tables()
        fd = os.path.join(out, cell, 'coadd_final')
        for p in (os.path.join(fd, f'I_deep_Taper_R075_{sc}.fits'), os.path.join(fd, 'weights.csv')):
            u.add_input(p)
        with IP.call_env(['masters', 'masters_build'], extra={'P3_VARIANTS_ONLY': ','.join(vks)}), \
                IP.capture_stdout(_log(out, cell, 'variants')):
            try:
                res = co.rebuild_variants_from_record(lst, freq, variants=vks)
            except RuntimeError as e:
                u.refuse(str(e)[:1500])
        for vk, (p, n, refused) in sorted(res.items()):
            if p:
                u.add_output(p, 'master', cell=cell)
            else:
                u.note(f'{vk}: NOT WRITTEN ({n} constituents < MIN_DATES_FOR_VARIANT; {len(refused)} refused)')


STAGES = {'stage-in': stage_in, 'build': build, 'pbmaps': pbmaps, 'vclean': vclean, 'fluxscale': fluxscale,
          'verify': verify, 'sources': sources, 'variants': variants}


# =====================================================================================================================
# dry run: the plan, per cell and stage (writes nothing)
# =====================================================================================================================

def _plan_stage_in(cell, args, out, planned):
    from exopipe.phase3.lib.masters import deep_stagein as DS
    lst, sc = cell.split('_', 1)
    lines, gates = [], []
    cell_dir = os.path.join(out, cell)
    if os.path.exists(cell_dir) and not (args.fresh and getattr(args, 'supersede', None)):
        gates.append(f'{cell_dir} exists (G-M25: --fresh --supersede TAG)')
    try:
        ov = DS.working_root_table()
    except Refusal as e:
        return lines, [str(e)]
    C = DS.working_cell(cell, ov)
    lines.append(f'working cell {C}' + (' (ref_override.csv)' if cell in ov else ''))
    try:
        not_here(C, 'the working cell', 'stage-in')
    except Refusal as e:
        return lines, [str(e)]
    mode, val = parse_reference(args.reference)
    try:
        if mode == 'previous-master':
            ents = DS.resolve_reference(cell, C, read_data=False)
        elif mode == 'path':
            ents = DS.reference_from_path(val, read_data=False)
        else:
            ents = []
        for e in ents:
            lines.append(f"reference {e['var']}: {e.get('generation')}, branch {e['branch']}; source {e['source']}; factor {e['factor']}; "
                         f"CONSTDIG {e['constdig']} NCOADD {e['ncoadd']}" + (f"; {e['none_reason']}" if e['none_reason'] else ''))
    except Refusal as e:
        gates.append(f'reference: {e}')
    if mode != 'previous-master' and WAIVE_REFERENCE not in (getattr(args, 'waive', None) or []):
        gates.append(f'--reference {mode}:{val} needs --waive {WAIVE_REFERENCE} --ruling "..."')
    egc = DS.epoch_gate_source(cell, C) if _epoch_gate_carry() == 'carry' else None
    lines.append(f'epoch-gate record: {egc or "none to carry"}' if _epoch_gate_carry() == 'carry' else
                 'epoch-gate record: none carried (B-05 a: the build gates every night on its own master)')
    fsrc = os.path.join(C, 'flow')
    nf = sum(len(fs) for _, _, fs in os.walk(fsrc)) if os.path.isdir(fsrc) else 0
    lines.append(f'outputs: {cell_dir}/coadd_final/{{I_deep_Taper_R075,V_deep_Taper_R0}}_{sc}.fits, coadd_final_ref/ copies '
                 f'[reference_master]' + (', coadd_final/epoch_gate.csv [csv_table]' if egc else ''))
    lines.append(f'next: flow --tier deep (copies the {nf} stored flow files of {fsrc}, computes the stale ones), then '
                 f'masters build')
    return lines, gates


def _plan_build(cell, args, out, planned):
    from exopipe.phase3.lib.masters import deep_inplace as IP
    lst, sc = cell.split('_', 1)
    lines, gates = [], []
    mods, _ = inplace(out)
    co = mods['coadd']
    try:
        pins = pin_build_tables()
        lines.append(f'pinned tables ok: {", ".join(pins)}')
    except Refusal as e:
        gates.append(str(e))
    with IP.capture_stdout(None, echo=False) as tee:       # the finder's [failsafe] lines: counted, not echoed
        dates = co.discover_dates(lst, _freq(sc))
    nraw = sum(1 for l in tee.lines if '[failsafe]' in l)
    if nraw:
        lines.append(f'{nraw} night(s) without a dewarped Phase-2 image: the RAW image is used and the flow dewarps it '
                     f'(coadd failsafe; RAW_INPUT_FLOW_RULE {_cfg().masters_build.RAW_INPUT_FLOW_RULE!r}, B-03)')
    nights = [d for d, _, _ in dates]
    try:
        ex, _ = IP.exclusion_readers_agree(lst, sc, nights, co)
        adm = ex.admitted(lst, sc, nights)
        lines.append(f'{len(nights)} Phase-2 nights; {len(adm)} admitted (exclusion readers agree); excluded '
                     f'{sorted(set(nights) - set(adm))}')
        if not adm:
            gates.append('no admitted night (NO-INPUT)')
    except Refusal as e:
        gates.append(str(e))
    if 'stage-in' not in planned and not done_record(out, 'stage-in', cell):
        gates.append('no done stage-in record (run --stages stage-in first)')
    try:
        lines.append('flow step: ' + flow_gate(out, cell))
    except Refusal as e:
        gates.append(str(e))
    dirty = build_clean_start(os.path.join(out, cell), sc) if 'stage-in' not in planned else []
    if dirty:
        gates.append(f'cell not as stage-in left it ({"; ".join(dirty[:3])}): stage-in --fresh --supersede TAG first (G-M25)')
    try:
        own = _epoch_gate_source() == 'own-master'
        lines.append('epoch gate: ' + ('on this build\'s own master after the final co-add, up to '
                                       f'{_cfg().masters_build.EPOCH_GATE_MAX_PASSES} final pass(es) without in-master FAIL '
                                       'nights (B-05 a); membership verified with the record; passes kept under '
                                       f'{out}/_epoch_gate_passes/{cell}/<run id>/' if own else
                                       'the carried record of stage-in (production; the install evaluates the new master)'))
        if own and os.path.exists(os.path.join(out, cell, 'coadd_final', 'epoch_gate.csv')) and 'stage-in' not in planned:
            gates.append('coadd_final/epoch_gate.csv exists (carried): own-master builds without one - stage-in --fresh '
                         '--supersede TAG with masters_stagein.EPOCH_GATE_CARRY none first (B-05 a)')
        if own:
            f, what = bake_factor(lst, sc)
            lines.append(f'epoch gate frame: x{f:.5f} ({what})' if f else f'epoch gate frame: unbaked - {what}')
    except Refusal as e:
        gates.append(str(e))
    lines.append(f'in place: coadd.run_provisional_lst_freq({lst!r}, {_freq(sc)!r}) then run_final_lst_freq; env '
                 f'{IP.call_env_values(["masters", "masters_build"])}')
    lines.append(f'outputs: {out}/{cell}/coadd_final/ 5 variants [master], loo/ [loo_master], weights.csv + '
                 f'residuals_provisional.csv [weights_table], stage2_verdict.csv [membership_record], subtracted/ per '
                 f'admitted night x I,V x epoch,sub,clean [coadd_epoch]')
    return lines, gates


def _plan_simple(stage):
    def f(cell, args, out, planned):
        lst, sc = cell.split('_', 1)
        lines, gates = [], []
        need = REQUIRES.get(stage)
        if need and need not in planned and not done_record(out, need, cell):
            gates.append(f'no done {need} record (run --stages {need} first)')
        if stage in BUILD_DERIVED and 'build' not in planned:
            stale = exclusion_stale(out, cell)
            if stale:
                gates.append(f'the build admitted night(s) {stale} excluded now (rebuild from stage-in; G-M01)')
        s = _cfg()
        if stage == 'pbmaps':
            lines.append(f'in place: pbcorr_masters.do_cell({out!r}, {cell!r}); PB library {s.masters_pbmaps.PB_LIBRARY}; '
                         f'outputs PB_deep_{sc}, PB_deep_R075_{sc} [pb_map] + a sibling per variant [pbcorr_sibling]')
        elif stage == 'vclean':
            try:
                from exopipe.phase3.core import fitsio
                lines.append(f'scratch root {fitsio.scratch_root(args.work)}')
            except Refusal as e:
                gates.append(str(e))
            lines.append(f'in place: vclean_masters.build_vclean_cell(skip_existing=False), timeout {s.masters_vclean.TIMEOUT_S} s; '
                         f'outputs V_deep_Taper_R0_clean_{sc} [vclean_master] + its sibling [pbcorr_sibling]')
        elif stage == 'fluxscale':
            row = scale_row(lst, sc)
            from exopipe.phase3.lib.masters import flux_correction as FC
            corr = FC.load_table(s.masters_fluxscale.FLUXCORR_TABLE)
            lines.append(f"scale row {row['status']} scale {row['scale']}; FLUXCORR[{sc}] "
                         f"{corr.get(sc, {}).get('corr')}; outputs {out}/{cell}/{s.masters_fluxscale.OUTGEN} "
                         f"[master_generation: T1-T8], a row in {out}/shflux_table.csv")
            if row['status'] != 'OK' or sc not in corr:
                gates.append('no scale/correction row (G-M23)')
        elif stage == 'verify':
            lines.append(f'content tests master_generation.T1-T8 on {out}/{cell}/{s.masters_fluxscale.OUTGEN}')
        elif stage == 'sources':
            installed = args.deep_installed_root or s.paths.COADD_SB_ROOT
            m = os.path.join(installed, cell, 'coadd_final', f'I_deep_Taper_R075_{sc}.fits')
            dest = args.out or s.masters_sources.DEEP_COLLECT
            lines.append(f'installed master {m} ({"present" if os.path.exists(m) else "MISSING"}); output '
                         f'{dest}/parent_catalogs/master_R075_{cell}.gaul.csv [master_catalog]')
            if not os.path.exists(m):
                gates.append('installed master missing')
            else:
                inplace(out)
                viol = installed_cell_violations(installed, cell)
                if viol:
                    gates.append(f'installed cell fails membership.verify / verify_products ({len(viol)}): {viol[0][:200]}')
                else:
                    lines.append('installed cell: membership.verify + verify_products PASS (install_cell.sh:70-71)')
            if not [r for r in _install_records(args.deep_install_records or out, cell) if r.get('status') == 'done']:
                if WAIVE_INSTALL_RECORD in (getattr(args, 'waive', None) or []):
                    lines.append(f'no done install record: accepted by --waive {WAIVE_INSTALL_RECORD} (ruling recorded)')
                else:
                    gates.append(f'no done install record (window: --waive {WAIVE_INSTALL_RECORD} --ruling "...")')
            from exopipe.phase3.lib.masters import deep_inplace as IP
            r = IP.is_production_path(os.path.join(dest, 'parent_catalogs'))
            if r and not s.paths.PRODUCTION_WRITES:
                gates.append(f'{dest} is under the production root {r} (pass --out)')
        elif stage == 'variants':
            lines.append(f'in place: coadd.rebuild_variants_from_record(variants={args.deep_variants or s.masters_build.VARIANTS_ONLY})')
        return lines, gates
    return f


PLANS = {'stage-in': _plan_stage_in, 'build': _plan_build}


def dry_run(cells, stages, args, out):
    refused = 0
    for cell in cells:
        planned = set()
        for st in stages:
            fn = PLANS.get(st) or _plan_simple(st)
            try:
                lines, gates = fn(cell, args, out, planned)
            except (Refusal, NoInput) as e:
                lines, gates = [], [str(e)]
            print(f'== {st}_{cell}')
            for l in lines:
                print(f'   {l}')
            for g in gates:
                print(f'   GATE would refuse: {g}')
            if gates:
                refused += 1
            planned.add(st)
    print(f'plan: {len(cells)} cell(s) x stages {",".join(stages)}; {refused} unit(s) would be refused; nothing written')
    return record.EXIT_REFUSED if refused else record.EXIT_DONE


# =====================================================================================================================
# entry
# =====================================================================================================================

def _resume_ok(run, name, st, cell, out):
    """--resume: (skip?, why). The record must pass record.resume_check; the build's with the admitted night set computed
    NOW (data_decisions.csv is not a build input: its effect on the cell is the admitted set), and every stage derived from
    the build only while no night that build admitted is excluded now."""
    admitted = None
    if st == 'build':
        if not done_record(out, 'build', cell):
            return False, 'no done record'
        from exopipe.phase3.lib.masters import deep_inplace as IP
        try:
            with IP.capture_stdout(None, echo=False):       # the finder's [failsafe] lines belong to the build's log
                admitted = build_nights(cell, out)[4]
        except (Refusal, NoInput) as e:
            return False, f'admitted set not computable now ({e})'
    ok, why = run.resume_check(name, inputs=None, admitted=admitted)
    if ok and st in BUILD_DERIVED:
        stale = exclusion_stale(out, cell)
        if stale:
            return False, f'the build admitted night(s) {stale} excluded now'
    return ok, why


def run(args, run):
    from exopipe.phase3.lib.masters import deep_inplace as IP
    IP.refuse_forbidden_overrides()
    out = args.out or default_out(args)
    stages = args.stages_resolved
    cells = select_cells(args)
    parse_reference(args.reference)             # a malformed --reference stops before any unit
    if args.dry_run:
        return dry_run(cells, stages, args, out)
    if not _cfg().paths.PRODUCTION_WRITES:
        IP.refuse_production_write(out, 'masters --tier deep')
    ran_now = {}
    for cell in cells:
        for st in stages:
            name = f'{st}_{cell}'
            if args.resume and not args.fresh:
                ok, why = _resume_ok(run, name, st, cell, out)
                if ok:
                    u = run.unit(name, tier='deep', stage=st, cell=cell)
                    u.status, u.reason = 'skipped-valid', why
                    ran_now[(st, cell)] = 'done'
                    print(f'{name}: skipped ({why})', file=sys.stderr)
                    continue
            n0 = len(run.units)
            try:
                STAGES[st](cell, args, run, out, ran_now)
            except Exception as e:             # the unit's record says error (UnitRecord.__exit__); the next cell goes on
                if len(run.units) == n0:       # raised before its unit existed: record the outcome as that unit's
                    u = run.unit(name, tier='deep', stage=st, cell=cell)
                    u.status = ('refused' if isinstance(e, Refusal) else 'no-input' if isinstance(e, NoInput) else 'error')
                    u.reason = str(e) if isinstance(e, (Refusal, NoInput)) else f'{type(e).__name__}: {e}'
                    u.write()
                if not isinstance(e, (Refusal, NoInput)):
                    traceback.print_exc()
            u = run.units[-1]
            ran_now[(st, cell)] = u.status
            print(f'{name}: {u.status}' + (f' - {u.reason}' if u.reason else ''), file=sys.stderr)
            if u.status != 'done':
                rest = stages[stages.index(st) + 1:]
                if rest:
                    print(f'{cell}: stages {",".join(rest)} not attempted ({st} {u.status})', file=sys.stderr)
                break
    return None
