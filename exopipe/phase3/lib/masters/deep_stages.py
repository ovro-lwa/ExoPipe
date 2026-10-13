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
import socket
import sys

from exopipe.phase3.core import record
from exopipe.phase3.core.record import NoInput, Refusal

STAGE_ORDER = ('stage-in', 'build', 'pbmaps', 'vclean', 'fluxscale', 'verify', 'sources', 'variants')
# the stage whose done record a stage requires (spec 1.2); sources requires the install step's record instead
REQUIRES = {'build': 'stage-in', 'pbmaps': 'build', 'vclean': 'pbmaps', 'fluxscale': 'vclean', 'verify': 'fluxscale',
            'variants': 'build'}
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
    c = _cfg()
    by_host = getattr(c.site, 'MASTERS_GEN_ROOT', None) or {}
    return by_host.get(socket.gethostname()) or c.masters.DEEP_GEN_ROOT


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


def require_stage(out, stage, cell, ran_now):
    """Refuse unless the stage this one depends on is done for the cell (its record), or ran done in this invocation."""
    need = REQUIRES.get(stage)
    if not need:
        return None
    if ran_now.get((need, cell)) == 'done':
        return record.load_record(out, 'masters', f'{need}_{cell}')
    rec = done_record(out, need, cell)
    if rec is None:
        got = record.load_record(out, 'masters', f'{need}_{cell}')
        raise Refusal(f'{stage}: the {need} stage of {cell} is not done under {out} (record status '
                      f'{got.get("status") if got else "missing"}) - run --stages {need} first (spec 1.2)')
    return rec


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
        ov, ovi = DS.override_tables()
        for t in (_cfg().masters_stagein.REF_OVERRIDE_TABLE, _cfg().masters_stagein.REF_OVERRIDE_I_TABLE):
            if os.path.exists(t):
                u.add_input(t)
        C = DS.working_cell(cell, ov)
        u.note(f'working cell {C}' + (' (ref_override.csv)' if cell in ov else ''))
        if mode == 'previous-master':
            entries = DS.native_reference(cell, C, ovi)
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
        egc = DS.epoch_gate_source(cell, C)
        os.makedirs(os.path.join(cell_dir, 'coadd_final'), exist_ok=True)
        if egc:
            u.add_input(egc)
            dst = os.path.join(cell_dir, 'coadd_final', 'epoch_gate.csv')
            shutil.copy(egc, dst)                                    # cp (rebuild_cell.sh:42)
            fails = [r['date'] for r in csv.DictReader(open(dst)) if r.get('verdict') == 'FAIL']
            u.check('epoch_gate_carried', IP.sha1_file(dst) == IP.sha1_file(egc), value=len(fails),
                    detail=f'epoch-gate record carried in from {egc}: FAIL {fails}')
        else:
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
            u.add_output(q, 'reference_master', source=e['source'], factor=e['factor'], previous=prev)
        if not written and mode != 'night':
            u.refuse('no reference master written (G-M04)')
        if egc:          # the carried record (the co-add final stage reads it: G-M10)
            u.add_output(os.path.join(cell_dir, 'coadd_final', 'epoch_gate.csv'), 'csv_table',
                         required_columns=('date', 'i_ratio', 'verdict', 'master_md5'), min_rows=1)
        elif mode == 'night':
            raise NoInput('--reference night: nothing written for the cell (no reference, no epoch-gate record)')


# =====================================================================================================================
# stage: build (provisional + final co-add)
# =====================================================================================================================

def _refuse_or_noinput(u, tee, stage, noinput=('No images for',), refuse=('REFUSING', 'Only', 'Reference', 'No flow/',
                                                                            'No provisional', 'REJECTED')):
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


def build(cell, args, run, out, ran_now):
    from exopipe.phase3.lib.masters import deep_inplace as IP
    lst, sc = cell.split('_', 1)
    freq = _freq(sc)
    cell_dir = os.path.join(out, cell)
    with run.unit(f'build_{cell}', tier='deep', stage='build', cell=cell) as u:
        prev = require_stage(out, 'build', cell, ran_now)
        flow_note = flow_gate(out, cell, u)          # cheap gates first: G-M05 at record level
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
            u.add_input(p)
        reference = (prev or {}).get('meta', {}).get('reference', 'previous-master')
        dates = co.discover_dates(lst, freq)
        if not dates:
            raise NoInput(f'No Phase-2 images for {lst} {sc} under {co.PHASE2_ROOT}')
        nights = [d for d, _, _ in dates]
        ex, excl = IP.exclusion_readers_agree(lst, sc, nights, co)
        admitted = ex.admitted(lst, sc, nights)
        u.set_admitted(admitted)
        u.note(f'{len(nights)} Phase-2 nights, {len(admitted)} admitted; excluded {sorted(set(nights) - set(admitted))}')
        if not admitted:
            raise NoInput(f'every Phase-2 night of {lst} {sc} is excluded in data_decisions.csv')
        for d, _sd, img in dates:
            if d in admitted:
                u.add_input(img)
        for p in sorted(glob.glob(os.path.join(cell_dir, 'flow', '*'))) + sorted(glob.glob(os.path.join(cell_dir, 'coadd_final', '*'))):
            if os.path.isfile(p):
                u.add_input(p)
        u.note(f'flow step: {flow_note}')
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
        nsib = len(glob.glob(os.path.join(fd, f'*_pbcorr_{sc}.fits')))
        u.check('masters_pbmaps.siblings_after_pbmaps', nsib == s.SIBLINGS_AFTER_PBMAPS, value=nsib,
                detail=f'{nsib} siblings after pbmaps ({s.SIBLINGS_AFTER_PBMAPS} expected: the five I/V variants; '
                       f'check_rebuild.py T4 expects 6 once the cleaned V exists)')


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
        if not os.path.exists(master):
            u.refuse(f'installed master {master} missing')
        u.add_input(master)
        h = fits.getheader(master)
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
        ov, ovi = DS.override_tables()
    except Refusal as e:
        return lines, [str(e)]
    C = DS.working_cell(cell, ov)
    lines.append(f'working cell {C}' + (' (ref_override.csv)' if cell in ov else ''))
    mode, val = parse_reference(args.reference)
    try:
        if mode == 'previous-master':
            ents = DS.native_reference(cell, C, ovi, read_data=False)
        elif mode == 'path':
            ents = DS.reference_from_path(val, read_data=False)
        else:
            ents = []
        for e in ents:
            lines.append(f"reference {e['var']}: branch {e['branch']}; source {e['source']}; factor {e['factor']}; "
                         f"CONSTDIG {e['constdig']} NCOADD {e['ncoadd']}" + (f"; {e['none_reason']}" if e['none_reason'] else ''))
    except Refusal as e:
        gates.append(f'reference: {e}')
    if mode != 'previous-master' and WAIVE_REFERENCE not in (getattr(args, 'waive', None) or []):
        gates.append(f'--reference {mode}:{val} needs --waive {WAIVE_REFERENCE} --ruling "..."')
    egc = DS.epoch_gate_source(cell, C)
    lines.append(f'epoch-gate record: {egc or "none to carry"}')
    fsrc = os.path.join(C, 'flow')
    nf = sum(len(fs) for _, _, fs in os.walk(fsrc)) if os.path.isdir(fsrc) else 0
    lines.append(f'outputs: {cell_dir}/coadd_final/{{I_deep_Taper_R075,V_deep_Taper_R0}}_{sc}.fits, coadd_final_ref/ copies '
                 f'[reference_master], coadd_final/epoch_gate.csv [csv_table]')
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
                ok, why = run.resume_check(name, inputs=None, admitted=None)
                if ok:
                    u = run.unit(name, tier='deep', stage=st, cell=cell)
                    u.status, u.reason = 'skipped-valid', why
                    ran_now[(st, cell)] = 'done'
                    print(f'{name}: skipped ({why})', file=sys.stderr)
                    continue
            STAGES[st](cell, args, run, out, ran_now)
            u = run.units[-1]
            ran_now[(st, cell)] = u.status
            print(f'{name}: {u.status}' + (f' - {u.reason}' if u.reason else ''), file=sys.stderr)
            if u.status != 'done':
                rest = stages[stages.index(st) + 1:]
                if rest:
                    print(f'{cell}: stages {",".join(rest)} not attempted ({st} {u.status})', file=sys.stderr)
                break
    return None
