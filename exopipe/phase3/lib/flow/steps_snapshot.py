"""`exopipe3 flow --tier snapshot`: per-frame flow of every staged 10-s frame to its subband's multi-night Master (PER HOUR,
DESIGN 12.1; chains 'epoch' and 'snapshot-cell': frames stage,regrid -> flow -> frames cube,movies -> search -> curves).

Item: night cell <LST>_<date>[_<subband>] (--cells; --epochs <LST>_<date> is the same cell, all subbands; --lsts = the cells
staged on this node). Units:
  flow_<cell>_<sb>   clv15/phase3_snapshot_flow.py IN PLACE through core.legacy.run under site PYTHON['flow'], exactly the
                     fleet's command line (snapshot_fleet_node_v9.sh:182-190): --stage <stage>/<sb> --reference <Master>
                     --subband <sb> --out <work>/<cell>/flow, env CUDA_VISIBLE_DEVICES, XLA_PYTHON_CLIENT_PREALLOCATE=false,
                     XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 (+ JAX_COMPILATION_CACHE_DIR). Subbands round-robin over the GPUs,
                     one child per GPU at a time (v9:143-144). After the child, G-S6 compares SETS per subband by name:
                     expected = the stage record's frames (I and V) minus the frames whose own header records WSCNVIS = 0;
                     flowed = <frame>_flowed.fits present (snapshot defect D12; cf72d51: never a count over the cell).
  ship_<cell>        copy the fields to <out>/snapshot_flow/<cell>/ and flow_<sb>.csv / flow_<sb>.log to
                     <out>/snapshot_fleet_out/<cell>/flow/ (production paths.SNAPSHOT_FLOW_ROOT / SNAPSHOT_FLEET_OUT), each
                     copy verified by sha1; a destination with different bytes refuses (G-S9; today `cp -n` keeps the old
                     file silently, v9:336-341). The flow movies and series are frames' `movies` stage (_CROSSCHECK group 4).
Cell gates (whole cell refused, every subband unit records the reason): G-S1 no GPU id, or no site GPU lock for a GPU;
G-S2 no done frames stage record, or one made on another node; G-S3 a broken Master link / G-S4 a subband without a Master
(over the launcher's subband selection: every selected subband in the search chain, v9:146-181; the staged ones in the
cube chain, cube:66-92); G-S5 a staged subband off its own Master's grid (cube:78-81).
"""
import glob
import json
import os
import re
import shutil
import threading

from exopipe.phase3.core.record import NoInput, Refusal
from exopipe.phase3.lib.flow import steps_common as C

STAGES = ('flow', 'ship')
FLOW_SCRIPT = ('clv15', 'phase3_snapshot_flow.py')     # (legacy root, file); tests replace it with a stand-in
_COMPRESSED = re.compile(r'\.f[sz]$')


def add_arguments(p):
    g = p.add_argument_group('tier snapshot (per-frame flows to the multi-night Master; per hour)')
    g.add_argument('--snap-stage-root', dest='snap_stage_root', metavar='DIR',
                   help='staging root holding <cell>/<sb>/ (default: the frames stage record, else <work>/'
                        'flow_snapshot.STAGE_DIR)')
    g.add_argument('--snap-stage-record', dest='snap_stage_record', metavar='PATH',
                   help="the frames stage record of the cell (default: <out>/_records/frames/<cell>_stage.json, else frames' "
                        "own default root)")
    g.add_argument('--snap-frames', dest='snap_frames', metavar='FILE',
                   help='flow only the Stokes I frames listed (paths or names; the module\'s --frames): single-frame checks, '
                        'freeze flow checks; the unit is recorded partial (meta.partial) and never shipped')
    g.add_argument('--snap-against-shipped', dest='snap_against_shipped', action='store_true',
                   help='also compare every new field with the field already shipped for that frame (test F, 0.001 px)')


def default_out(args):
    return C.cfg().paths.PHASE3_ARCHIVE_ROOT


# ---------------------------------------------------------------------------------------------------------------------
# items, roots, the stage record
# ---------------------------------------------------------------------------------------------------------------------

def _work_root(args):
    from exopipe.phase3.core import fitsio
    return fitsio.scratch_root(args.work)


def items(args):
    """[(cell, [subbands], explicit)] in order; explicit = the subbands were named (a _<sb> suffix or --scopes), else every
    subband the staging holds is flowed (the fleet's SUBBANDS default, v9:47)."""
    from exopipe.phase3.core import cells
    allsb = list(C.cfg().instrument.ALL_SUBBANDS)
    want, explicit = {}, set()
    toks = cells.expand(args.cells, kind='night cell') if args.cells else []
    toks += [f'{l}_{d}' for l, d in (cells.parse_epoch(e) for e in cells.expand_epochs(args.epochs))] if args.epochs else []
    for t in toks:
        lst, date, sb = C.parse_snapshot_item(t)
        want.setdefault(f'{lst}_{date}', set()).update([sb] if sb else allsb)
        if sb:
            explicit.add(f'{lst}_{date}')
    if args.lsts:
        lsts = set(cells.expand(args.lsts, universe=cells.LSTS, kind='LST'))
        root = args.snap_stage_root or os.path.join(_work_root(args), os.path.dirname(C.cfg().flow_snapshot.STAGE_DIR))
        for d in sorted(glob.glob(os.path.join(root, '*'))):
            b = os.path.basename(d)
            try:
                lst, date, _ = C.parse_snapshot_item(b)
            except ValueError:
                continue
            if lst in lsts:
                want.setdefault(b, set()).update(allsb)
    if args.nights:
        nights = set(cells.expand(args.nights, kind='night'))
        want = {c: s for c, s in want.items() if c.split('_', 1)[1] in nights}
    if args.scopes:
        keep = set(cells.expand(args.scopes, universe=allsb, kind='subband'))
        want = {c: s & keep for c, s in want.items()}
        explicit |= set(want)
    if args.exclude_cells:
        drop = set(cells.expand(args.exclude_cells, kind='night cell'))
        want = {c: s for c, s in want.items() if c not in drop}
    return [(c, [s for s in allsb if s in sbs], c in explicit) for c, sbs in want.items() if sbs]


def product_dirs(args, run, cell):
    """(fields dir, QA dir): production paths.SNAPSHOT_FLOW_ROOT/<cell>, paths.SNAPSHOT_FLEET_OUT/<cell>/flow; with --out D
    D/snapshot_flow/<cell>, D/snapshot_fleet_out/<cell>/flow (spec Q4)."""
    p = C.cfg().paths
    if args.out:
        return os.path.join(run.out_root, 'snapshot_flow', cell), os.path.join(run.out_root, 'snapshot_fleet_out', cell, 'flow')
    return os.path.join(p.SNAPSHOT_FLOW_ROOT, cell), os.path.join(p.SNAPSHOT_FLEET_OUT, cell, 'flow')


def stage_record(args, run, cell):
    """(record dict or None, path or None) of `frames --tier snapshot --stages stage` for the cell."""
    import importlib
    cands = [args.snap_stage_record] if args.snap_stage_record else []
    cands.append(os.path.join(run.out_root, '_records', 'frames', f'{cell}_stage.json'))
    try:
        fr = importlib.import_module('exopipe.phase3.steps.frames')
        ns = type('A', (), dict(vars(args), tier='snapshot'))()
        cands.append(os.path.join(fr.default_out(ns), '_records', 'frames', f'{cell}_stage.json'))
    except Exception:
        pass
    for p in cands:
        if p and os.path.exists(p):
            with open(p) as f:
                return json.load(f), p
    return None, None


def stage_dir(args, cell, rec):
    if args.snap_stage_root:
        return os.path.join(args.snap_stage_root, cell)
    meta = (rec or {}).get('meta') or {}
    for k in ('stage_dir', 'stage_root'):
        v = meta.get(k)
        if v:
            return v if os.path.basename(v.rstrip('/')) == cell else os.path.join(v, cell)
    return os.path.join(_work_root(args), C.cfg().flow_snapshot.STAGE_DIR.format(cell=cell))


def listed_frames(sdir, sb, rec):
    """(staged frame paths of the subband per the stage record, source). The record's meta 'frames'[sb] (or
    meta['subbands'][sb]['frames']) when present, else the staging's .filelist (the discovery list the stage wrote, v9:106-111;
    compressed suffix stripped), else None."""
    meta = (rec or {}).get('meta') or {}
    lst = None
    if isinstance(meta.get('frames'), dict) and sb in meta['frames']:
        lst, src = meta['frames'][sb], 'stage record meta.frames'
    elif isinstance(meta.get('subbands'), dict) and isinstance(meta['subbands'].get(sb), dict) \
            and 'frames' in meta['subbands'][sb]:
        lst, src = meta['subbands'][sb]['frames'], 'stage record meta.subbands'
    else:
        fl = os.path.join(sdir, sb, '.filelist')
        if os.path.exists(fl):
            lst, src = [l.strip() for l in open(fl) if l.strip()], '.filelist'
    if lst is None:
        return None, None
    names = sorted({_COMPRESSED.sub('', os.path.basename(x)) for x in lst})
    return [os.path.join(sdir, sb, n) for n in names], src


def staged_on_disk(sdir, sb):
    d = os.path.join(sdir, sb)
    return sorted(os.path.join(d, f) for f in os.listdir(d) if C.is_staged_frame(f)) if os.path.isdir(d) else []


def _record_subband(rec, sb):
    """meta.subbands[sb] of the frames stage record (I, V, names_sha1, wscnvis0, unreadable_header), or {}."""
    subs = ((rec or {}).get('meta') or {}).get('subbands')
    return subs.get(sb) or {} if isinstance(subs, dict) else {}


def staged_names_sha1(sdir, sb):
    """sha1 of the sorted staged *-image.fits names of a subband, as the frames stage record writes it (names_sha1)."""
    import hashlib
    d = os.path.join(sdir, sb)
    names = sorted(f for f in os.listdir(d) if f.endswith('-image.fits')) if os.path.isdir(d) else []
    return hashlib.sha1('\n'.join(names).encode()).hexdigest()


def zero_vis_set(frames, rec, sb):
    """(the staged frames whose own header records WSCNVIS = 0, source): the frames stage record's per-subband list
    (meta.subbands[sb].wscnvis0, written by `frames --tier snapshot --stages stage`; DESIGN 11.5 - read what the previous
    stage recorded), else the headers read here (v9:162-175; an unreadable header = has visibilities, B-23)."""
    meta = (rec or {}).get('meta') or {}
    zv = meta.get('zero_vis')
    rs = _record_subband(rec, sb)
    if isinstance(rs.get('wscnvis0'), list):
        names = {os.path.basename(x) for x in rs['wscnvis0']}
        return {p for p in frames if os.path.basename(p) in names}, 'stage record meta.subbands.wscnvis0'
    if isinstance(zv, dict) and sb in zv:
        names = {os.path.basename(x) for x in zv[sb]}
        return {p for p in frames if os.path.basename(p) in names}, 'stage record'
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(16) as ex:
        flags = list(ex.map(lambda p: C.zero_visibility(p)[0], frames))
    return {p for p, z in zip(frames, flags) if z}, 'frame headers (WSCNVIS)'


def _naxis1(path):
    try:
        return C.card(C.header_cards(path), 'NAXIS1')
    except OSError:
        return None


# ---------------------------------------------------------------------------------------------------------------------
# cell gates (G-S1..G-S4)
# ---------------------------------------------------------------------------------------------------------------------

def cell_gates(args, run, cell, sbs):
    """(info dict, refusal text or None) for the whole cell - the fleet refuses the cell as a whole (v9:146-181; cube:66-92).
    ``sbs`` = the subband selection of the cell (every subband unless named), as the launchers' SUBBANDS."""
    lst = cell.split('_')[0]
    c = C.cfg().flow_snapshot
    info = {'gpus': C.parse_gpus(args.gpus) or C.site_gpu_ids()}
    if not info['gpus']:
        return info, (f'G-S1: no GPU ids for {C.host()} (--gpus, or GPU_IDS in the site file) - the flow is required '
                      f'(R-SNAP-FLOW-ALIGNMENT-AND-FLOW-MOVIE-ARE-PIPELINE-PRODUCTS-FLEET-V4)')
    try:
        info['locks'] = {g: C.required_gpu_locks(g, 'G-S1') for g in info['gpus']}
    except Refusal as e:
        return info, str(e)
    rec, rpath = stage_record(args, run, cell)
    info.update(stage_record=rpath, stage_status=(rec or {}).get('status'))
    if c.REQUIRE_STAGE_RECORD and (not rec or rec.get('status') != 'done'):
        return info, (f'G-S2: no done `frames --tier snapshot --stages stage` record of {cell} '
                      f'({rpath or "none found"}; status {(rec or {}).get("status")}) - flow reads only what the stage '
                      f'record lists (DESIGN 11.5); flow_snapshot.REQUIRE_STAGE_RECORD=False reads the staging instead')
    rhost = ((rec or {}).get('meta') or {}).get('host')
    if rhost and not C.same_host(rhost, C.host()) and not args.snap_stage_root:
        return info, (f'G-S2: the staging of {cell} was made on {rhost} (stage record {rpath}), this is {C.host()} - the '
                      f'staging is node-local; run flow on the node that holds it (the chain keeps the item on one node)')
    try:
        sdir = stage_dir(args, cell, rec)
    except Refusal as e:
        return info, f'staging root: {e}'
    info.update(stage_dir=sdir, rec=rec)
    staged = [sb for sb in sbs if staged_on_disk(sdir, sb)]
    # the subbands the launcher's cell gates look at: the fleet (search chain) every subband of SUBBANDS, staged or not
    # (v9:146-151 broken links, v9:152-155 + 176-181 no Master); the cube pass the staged subband directories (cube:66-92)
    chain = ((rec or {}).get('meta') or {}).get('chain')
    gate_sbs = staged if chain == 'cube' else list(sbs)
    info['gate_subbands'] = gate_sbs
    broken = [sb for sb in gate_sbs if os.path.islink(C.snapshot_master(lst, sb))
              and not os.path.exists(C.snapshot_master(lst, sb))]
    if broken:
        return info, f'G-S3: broken Master link(s) {broken} under {c.MASTERS_ROOT} - the whole cell is refused (v9:146-151)'
    nomaster = [sb for sb in gate_sbs if not os.path.exists(C.snapshot_master(lst, sb))]
    if nomaster:
        return info, (f'G-S4: subband(s) without a Taper R0 Master {nomaster} - the whole cell is refused (the '
                      f'hourly-screen fallback is retired; RESEARCH-24-HOURLY-SCREEN-SUBBANDS-EXECUTING; v9:176-181)')
    # G-S5: every staged subband on its OWN Master's grid (first staged frame vs Master NAXIS1; cube:66-81 refuses the whole
    # cell - pilot frames must have been regridded by `frames --tier snapshot --stages regrid`, v8.1 own-Master rule)
    offgrid = []
    for sb in staged:
        f1 = staged_on_disk(sdir, sb)[0]
        nm, nf = _naxis1(C.snapshot_master(lst, sb)), _naxis1(f1)
        if nm != nf:
            offgrid.append(f'{sb} frames {nf} px, own Master {nm} px')
    if offgrid:
        return info, f'G-S5: frames off their own Master\'s grid: {offgrid} - the whole cell is refused (cube:78-81)'
    info['staged_subbands'] = staged
    return info, None


# ---------------------------------------------------------------------------------------------------------------------
# one subband
# ---------------------------------------------------------------------------------------------------------------------

def _module_args():
    """Module parameters passed only when the config differs from production (production passes none)."""
    c = C.cfg()
    out = []
    for val, dflt, flag in ((c.flow_snapshot.LOWRES, 8, '--lowres'), (c.flow_snapshot.HORIZON_ELEVATION_DEG, 10.0,
                            '--horizon-elevation-deg'), (c.flow_snapshot.BLOCK_S, 0, '--block-s'),
                            (c.flow.PREPROCESS_WEIGHT, 1.5, '--preprocess-weight'), (c.flow.ALPHA, 1.3, '--alpha'),
                            (c.flow.GAMMA, 150.0, '--gamma'), (c.flow.SCALE_FACTOR, 0.7, '--scale-factor')):
        if val != dflt:
            out += [flag, repr(val)]
    return out


def _supersede_work(flow_dir, sb):
    """A re-run never appends to an earlier run's flow_<sb>.csv (the module opens it 'a', phase3_snapshot_flow.py:234):
    move this subband's earlier work files to <flow_dir>/_previous_<UTC>/ (node-local work, never deleted)."""
    olds = [p for p in glob.glob(os.path.join(flow_dir, f'flow_{sb}.*')) + glob.glob(os.path.join(flow_dir, f'flow_{sb}_*.npy'))]
    if not olds:
        return None
    dst = os.path.join(flow_dir, f'_previous_{C.utcstamp()}_{sb}')
    k = 1
    while os.path.lexists(dst):                 # two runs in one second: never rename onto (and lose) earlier files
        dst = os.path.join(flow_dir, f'_previous_{C.utcstamp()}_{sb}.{k}')
        k += 1
    os.makedirs(dst)
    for p in olds:
        os.rename(p, os.path.join(dst, os.path.basename(p)))
    return dst


def flow_subband(u, args, run, cell, sb, gpu, info, partial_list):
    from exopipe.phase3.core import exclusions, legacy
    from exopipe.phase3.lib.flow import tests as T
    c = C.cfg()
    lst, date = cell.split('_')
    sdir, rec = info['stage_dir'], info.get('rec')
    sbdir = os.path.join(sdir, sb)
    work = _work_root(args)
    flow_dir = os.path.join(work, c.flow_snapshot.WORK_FLOW_DIR.format(cell=cell))
    master = C.snapshot_master(lst, sb)
    u.meta.update(cell=cell, subband=sb, gpu=gpu, stage_dir=sdir, flow_dir=flow_dir, master=master,
                  stage_record=info.get('stage_record'), partial=bool(partial_list))
    C.refuse_unreachable(('flow_snapshot.BLANK_FINITE_FRACTION',), 'phase3_snapshot_flow.py (blank rule, :99)')
    if C.subband_mhz(sb) > c.flow_snapshot.FLOW_MAX_MHZ:
        raise NoInput(f'{sb} above flow_snapshot.FLOW_MAX_MHZ {c.flow_snapshot.FLOW_MAX_MHZ}: not flowed (the search does not '
                      f'require flowed frames there)')
    on_disk = staged_on_disk(sdir, sb)
    if not on_disk:
        raise NoInput(f'no staged frame of {sb} in {sbdir}')
    E = exclusions.load()
    u.add_input(E.path, extra={'role': 'data_decisions'})
    if E.is_excluded(lst, sb, date):
        if c.flow_snapshot.EXCLUDED_SUBBAND == 'refuse':
            raise Refusal(f'{lst} {sb} {date} is excluded in data_decisions.csv (flow_snapshot.EXCLUDED_SUBBAND=refuse)')
        u.set_admitted([])
        u.note(f'{lst} {sb} {date} is EXCLUDED in data_decisions.csv: flowed as production does (the search refuses it); '
               f'admitted night set empty')
    else:
        u.set_admitted([date])
    # ---- G-S2 per subband: the staging holds exactly the frames the stage record lists -----------------------------
    listed, src = listed_frames(sdir, sb, rec)
    if listed is None:
        if c.flow_snapshot.REQUIRE_STAGE_RECORD:
            raise Refusal(f'G-S2: the stage record lists no frames of {sb} and {sbdir}/.filelist is absent')
        listed, src = on_disk, 'staging listing (REQUIRE_STAGE_RECORD False)'
    if set(listed) != set(on_disk):
        extra, miss = sorted(set(on_disk) - set(listed)), sorted(set(listed) - set(on_disk))
        raise Refusal(f'G-S2: the staging of {sb} differs from its list ({src}): {len(miss)} listed not staged '
                      f'{[os.path.basename(x) for x in miss[:3]]}, {len(extra)} staged not listed '
                      f'{[os.path.basename(x) for x in extra[:3]]}')
    want_sha1 = _record_subband(rec, sb).get('names_sha1')
    if want_sha1 and staged_names_sha1(sdir, sb) != want_sha1:
        raise Refusal(f'G-S2: the staged frame names of {sb} differ from the frames stage record (names_sha1 '
                      f'{want_sha1[:12]} recorded, {staged_names_sha1(sdir, sb)[:12]} on disk) - the staging changed since '
                      f'the stage unit; stage again')
    u.meta['frame_list_source'] = src + (' (names_sha1 = stage record)' if want_sha1 else '')
    if info.get('stage_record') and os.path.exists(info['stage_record']):
        u.add_input(info['stage_record'], extra={'role': 'frames stage record'})
    # (G-S5, the own-Master grid, is a cell gate: cell_gates)
    rdir = os.path.join(run.out_root, '_records', 'flow', 'inputs')
    C.add_listing_input(u, os.path.join(rdir, f'flow_{cell}_{sb}_frames.json'), listed, 'staged frames')
    u.add_input(master, extra={'role': 'subband Master'})
    zerovis, zsrc = zero_vis_set(listed, rec, sb)
    u.meta['zero_vis'] = {'n': len(zerovis), 'source': zsrc, 'frames': sorted(os.path.basename(p) for p in zerovis)[:50]}
    i_frames = [p for p in listed if (C.frame_info(p) or (0, 0, ''))[2] == 'I']
    todo_i = i_frames
    if partial_list is not None:
        names = {os.path.basename(x) for x in partial_list}
        todo_i = [p for p in i_frames if os.path.basename(p) in names]
        if not todo_i:
            raise NoInput(f'--snap-frames lists no Stokes I frame of {sb}')
    # ---- the child ----------------------------------------------------------------------------------------------------
    os.makedirs(flow_dir, exist_ok=True)
    prev_dir = _supersede_work(flow_dir, sb) if partial_list is None else None
    if prev_dir:
        u.note(f'earlier work files of {sb} moved to {prev_dir} (a re-run never appends to flow_{sb}.csv)')
    argv = ['--stage', sbdir, '--reference', master, '--subband', sb, '--out', flow_dir] + _module_args()
    if partial_list is not None:
        fl = os.path.join(flow_dir, f'_frames_{sb}_{os.getpid()}.txt')
        with open(fl, 'w') as f:
            f.write(''.join(p + '\n' for p in todo_i))
        argv += ['--frames', fl]
    if any(os.path.exists(C.flowed_path(p)) for p in listed):
        argv.append('--overwrite')     # flowed copies without a done record of this unit: a killed run may have left a
        u.note('--overwrite: flowed copies already present (a killed run can leave a truncated one the module would keep)')
    env = {'CUDA_VISIBLE_DEVICES': gpu, 'XLA_PYTHON_CLIENT_PREALLOCATE': c.flow.XLA_PREALLOCATE,
           'JAX_COMPILATION_CACHE_DIR': os.path.expanduser(c.flow.JAX_COMPILATION_CACHE_DIR)}
    if c.flow.XLA_MEM_FRACTION.get('snapshot') is not None:
        env['XLA_PYTHON_CLIENT_MEM_FRACTION'] = c.flow.XLA_MEM_FRACTION['snapshot']
    script = C.legacy_script(*FLOW_SCRIPT)
    u.add_input(script, extra={'role': 'code'})
    log = os.path.join(flow_dir, f'flow_{sb}.log')
    with C.hold_locks(C.required_gpu_locks(gpu, 'G-S1'), c.flow.GPU_LOCK_WAIT_S, u.note) as held:
        res = legacy.run(script, argv, python='flow', env=env, log=log, refusal_codes=(), refusal_text=(), unit=u)
    u.meta['child'] = dict(rc=res.rc, wall_s=res.wall_s, locks=held, log=log)
    if res.rc != 0:
        raise RuntimeError(f'phase3_snapshot_flow.py exit {res.rc} for {cell} {sb}: {(res.tail or "")[-600:]}')
    # ---- G-S6 (sets, by name), G-S7, G-S8 ---------------------------------------------------------------------------
    if partial_list is None:
        expected_frames = listed
    else:
        ts = {(C.frame_info(p) or (0, ''))[1] for p in todo_i}
        expected_frames = [p for p in listed if (C.frame_info(p) or (0, ''))[1] in ts]
    ok, n_exp, missing, setinfo = T.set_complete_flowed(sbdir, frames=expected_frames, zerovis=zerovis)
    u.meta['set'] = setinfo
    u.check('G-S6.flowed_set', ok, value=setinfo['flowed'],
            detail=(f'{len(missing)} of {n_exp} expected frames not flowed: {missing[:5]}' if missing else
                    f'{n_exp} of {n_exp} expected frames flowed; {setinfo}'), cls='snapshot_flowed_frame', path=sbdir)
    if not ok:
        raise Refusal(f'G-S6 FLOW-INCOMPLETE {cell} {sb}: {len(missing)} of {n_exp} expected frames (staged minus '
                      f'WSCNVIS = 0) have no _flowed.fits, e.g. {missing[:3]}')
    stats = os.path.join(flow_dir, f'flow_{sb}.csv')
    failed = []
    if os.path.exists(stats):
        import csv as _csv
        failed = [r['ts'] for r in _csv.DictReader(open(stats)) if str(r.get('state', '')).startswith('FAILED')]
    if failed:
        raise Refusal(f'G-S7: {len(failed)} FAILED rows in flow_{sb}.csv, e.g. {failed[:3]}')
    sok, nfield, sdet = T.set_complete_fields(flow_dir, sb, stage_sb_dir=sbdir) if partial_list is None else \
        (True, len(T._fields_of(flow_dir, sb)), 'partial run: field set not compared with the whole subband')
    u.check('G-S8.field_set', sok, value=nfield, detail=sdet, cls='snapshot_flow_field', path=flow_dir)
    fields = sorted(T._fields_of(flow_dir, sb).values())
    badf = {os.path.basename(p): pr for p, pr in zip(fields, T._pmap(lambda q: T.field_shape_problems(q, master), fields))
            if pr}
    u.check('snapshot_flow_field.field_shape', not badf and bool(fields), value=len(fields),
            detail=(f'{len(badf)} bad, e.g. {next(iter(badf.items()))}' if badf else f'{len(fields)} fields ok'),
            cls='snapshot_flow_field', path=flow_dir)
    if not sok:
        raise Refusal(f'G-S8: {sdet}')
    if args.snap_against_shipped:
        shipped = os.path.join(c.paths.SNAPSHOT_FLOW_ROOT, cell)
        res_f = [T.against_shipped(p, shipped_dir=shipped) for p in fields]
        bad = [d for ok_, _v, d in res_f if not ok_]
        worst = max([v for _o, v, _d in res_f if isinstance(v, float)] or [0.0])
        u.check('snapshot_flow_field.against_shipped', not bad and bool(res_f), value=worst,
                detail='; '.join(bad[:3]) or f'{len(res_f)} fields within 0.001 px of {shipped}', cls='snapshot_flow_field',
                path=flow_dir)
    # ---- outputs (directory-level set tests, then every header) -----------------------------------------------------
    u.add_output(sbdir, 'snapshot_flowed_frame', frames=expected_frames if partial_list is None else None)
    i_ts = [C.frame_info(p)[1] for p in todo_i]
    u.add_output(stats, 'snapshot_flow_stats', i_frames=i_ts, fields_dir=flow_dir)
    u.meta['n_fields'] = len(fields)


# ---------------------------------------------------------------------------------------------------------------------
# ship
# ---------------------------------------------------------------------------------------------------------------------

def _atomic_copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = os.path.join(os.path.dirname(dst), f'.{os.path.basename(dst)}.tmp.{os.getpid()}')
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def ship_cell(u, args, run, cell, sbs, flow_recs):
    from exopipe.phase3.core import record
    c = C.cfg()
    fields_dir, qa_dir = product_dirs(args, run, cell)
    u.meta.update(cell=cell, fields_dir=fields_dir, qa_dir=qa_dir)
    plan = []
    for sb in sbs:
        r = flow_recs.get(sb)
        if not r or r.get('status') != 'done':
            raise Refusal(f'ship: the flow unit flow_{cell}_{sb} is not done ({(r or {}).get("status")}) - nothing of the '
                          f'cell is shipped until every selected subband is flowed')
        if (r.get('meta') or {}).get('partial'):
            raise Refusal(f'ship: flow_{cell}_{sb} was a partial (--snap-frames) run - never shipped')
        # the flow unit record is an input: a re-flow of the subband invalidates a --resume of the ship (DESIGN 11.5)
        u.add_input(record.record_path(run.out_root, 'flow', f'flow_{cell}_{sb}'), extra={'role': 'flow unit record'})
        fd = r['meta']['flow_dir']
        for p in sorted(glob.glob(os.path.join(fd, f'flow_{sb}_*.npy'))):
            plan.append((p, os.path.join(fields_dir, os.path.basename(p))))
        for name in (f'flow_{sb}.csv', f'flow_{sb}.log'):
            if os.path.exists(os.path.join(fd, name)):
                plan.append((os.path.join(fd, name), os.path.join(qa_dir, name)))
    if args.fresh:
        moves = C.supersede([d for _s, d in plan if os.path.exists(d)], run.out_root if args.out else os.path.dirname(fields_dir),
                            run.out_root, args.supersede, f'flow ship {cell} --fresh')
        if moves:
            u.note(f'--fresh: {len(moves)} shipped files moved aside')
    shipped = same = 0
    for s, d in plan:
        if os.path.exists(d):
            if C.same_bytes(s, d):
                same += 1
                continue
            if c.flow_snapshot.SHIP_ON_EXISTING == 'refuse':
                raise Refusal(f'G-S9: {d} exists with different bytes - not overwritten (B-49; today `cp -n` kept the old '
                              f'file silently); --fresh --supersede TAG moves the shipped files aside')
        _atomic_copy(s, d)
        if not C.same_bytes(s, d):
            raise Refusal(f'G-S9: the copy {d} differs from its source {s} (checksum)')
        shipped += 1
    u.meta.update(shipped=shipped, already_identical=same, n_files=len(plan))
    u.note(f'ship {cell}: {shipped} copied, {same} already identical, to {fields_dir} and {qa_dir}')
    u.add_output(fields_dir, 'snapshot_flow_field')
    for sb in sbs:
        q = os.path.join(qa_dir, f'flow_{sb}.csv')
        if os.path.exists(q):
            u.add_output(q, 'snapshot_flow_stats', fields_dir=fields_dir)


# ---------------------------------------------------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------------------------------------------------

def _resumable(run, name, cell=None, sb=None):
    """The resume reason when the unit may be skipped, else None. A flow unit is re-listed (its staged frames now vs the
    recorded listing) and its admitted night recomputed from data_decisions now - never the record's own values."""
    from exopipe.phase3.core import exclusions, record
    prev = record.load_record(run.out_root, 'flow', name)
    if not (run.resume and prev and prev.get('status') == 'done'):
        return None
    admitted = None
    if sb:
        lst, date = cell.split('_')
        sdir = (prev.get('meta') or {}).get('stage_dir')
        ok, why = C.listing_current(prev, 'staged frames', staged_on_disk(sdir, sb) if sdir else [])
        if not ok:
            print(f'flow snapshot {name}: not resumed ({why})')
            return None
        admitted = [] if exclusions.load().is_excluded(lst, sb, date) else [date]
    ok, why = run.resume_check(name, inputs=None, admitted=admitted)
    if not ok:
        print(f'flow snapshot {name}: not resumed ({why})')
    return why if ok else None


def _skip(run, name, why):
    u = run.unit(name, tier='snapshot')
    u.status, u.reason = 'skipped-valid', why
    print(f'flow snapshot {name}: skipped ({why})')


def run(args, run):
    from exopipe.phase3.core import record
    stages = args.flow_stages
    try:
        cells_ = items(args)
    except ValueError as e:
        raise Refusal(str(e))
    if not cells_:
        raise NoInput('no snapshot cell selected (give --cells <LST>_<date>[_<sb>], --epochs <LST>_<date> or --lsts)')
    if run.dry_run:
        return plan(args, run, cells_, stages)
    partial = None
    if args.snap_frames:
        partial = [l.strip() for l in open(args.snap_frames) if l.strip() and not l.startswith('#')]
    for cell, sbs, explicit in cells_:
        selection = list(sbs)         # the launcher's SUBBANDS: the cell gates G-S3/G-S4 look at all of them (v9:146-181)
        if not explicit:              # every subband the staging holds (v9:47; an unstaged subband is not a unit)
            try:
                _rec, _ = stage_record(args, run, cell)
                _sdir = stage_dir(args, cell, _rec)
                staged = [sb for sb in sbs if staged_on_disk(_sdir, sb)]
            except Refusal:
                staged = []
            sbs = staged or sbs
        if 'flow' in stages:
            info, refusal = cell_gates(args, run, cell, selection)
            gpus = info.get('gpus') or []
            if refusal:
                for sb in sbs:
                    with run.unit(f'flow_{cell}_{sb}', tier='snapshot', cell=cell, subband=sb) as u:
                        u.meta.update(info={k: v for k, v in info.items() if k != 'rec'})
                        raise Refusal(refusal)
            else:
                lanes = {g: [] for g in gpus}
                for k, sb in enumerate(sbs):                   # round-robin in instrument order (v9:143-144)
                    lanes[gpus[k % len(gpus)]].append(sb)

                def lane(g, mine):
                    for sb in mine:
                        name = f'flow_{cell}_{sb}'
                        why = _resumable(run, name, cell, sb) if partial is None else None
                        if why:
                            _skip(run, name, why)
                            continue
                        try:
                            with run.unit(name, tier='snapshot', cell=cell, subband=sb) as u:
                                flow_subband(u, args, run, cell, sb, g, info, partial)
                        except Exception as e:      # recorded as 'error' in the unit; the lane goes on with its next subband
                            print(f'flow snapshot {name}: error {type(e).__name__}: {e}')
                threads = [threading.Thread(target=lane, args=(g, mine)) for g, mine in lanes.items() if mine]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
        if 'ship' in stages:
            name = f'ship_{cell}'
            why = _resumable(run, name)
            if why:
                _skip(run, name, why)
                continue
            recs = {sb: record.load_record(run.out_root, 'flow', f'flow_{cell}_{sb}') for sb in sbs}
            with run.unit(name, tier='snapshot', cell=cell) as u:
                ship_cell(u, args, run, cell, sbs, recs)
    return None


def plan(args, run, cells_, stages):
    """--dry-run: the gates, the frame sets and the outputs of each cell; writes nothing."""
    from exopipe.phase3.core import exclusions, inputs
    rc = 0
    E = exclusions.load()
    for cell, sbs, _explicit in cells_:
        lst, date = cell.split('_')
        P = []
        say = P.append
        info, refusal = cell_gates(args, run, cell, sbs)
        fields_dir, qa_dir = product_dirs(args, run, cell)
        say(f'cell {cell} (tier snapshot, stages {",".join(stages)}); subbands {",".join(sbs)}; GPUs {info.get("gpus")}; '
            f'GPU locks {info.get("locks", "(not resolved)")}')
        say(f'stage record: {info.get("stage_record") or "none"} (status {info.get("stage_status")}); staging '
            f'{info.get("stage_dir", "(not resolved)")}')
        for k, sb in enumerate(sbs):
            m = C.snapshot_master(lst, sb)
            mstate = 'present' if os.path.exists(m) else ('BROKEN LINK' if os.path.islink(m) else 'ABSENT')
            sdir = info.get('stage_dir')
            staged = staged_on_disk(sdir, sb) if sdir else []
            try:
                fr = inputs.phase1_frames(lst, date, sb)
                p1 = f'Phase-1 {len(fr["I"])} I + {len(fr["V"])} V frames (one per integration)'
            except Refusal as e:
                p1 = f'Phase-1: {e}'
            excl = ' EXCLUDED in data_decisions (flowed, recorded)' if E.is_excluded(lst, sb, date) else ''
            gpu = info['gpus'][k % len(info['gpus'])] if info.get('gpus') else '-'
            say(f'  {sb}: Master {mstate} ({os.path.realpath(m)}); staged here {len(staged)}; {p1}; GPU {gpu}{excl}')
        say(f'flow: per subband {C.cfg().site.PYTHON.get("flow")} clv15/phase3_snapshot_flow.py --stage <staging>/<sb> '
            f'--reference <Master> --subband <sb> --out <work>/{C.cfg().flow_snapshot.WORK_FLOW_DIR.format(cell=cell)} '
            f'{" ".join(_module_args())}; outputs <staging>/<sb>/*_flowed.fits [snapshot_flowed_frame], flow_<sb>.csv '
            f'[snapshot_flow_stats], flow_<sb>_<ts>.npy')
        if info.get('stage_dir') and not any(staged_on_disk(info['stage_dir'], sb) for sb in sbs):
            say(f'nothing of {cell} is staged in {info["stage_dir"]} on {C.host()}: every flow unit would be NO-INPUT (run '
                f'`frames --tier snapshot --stages stage,regrid` on this node first; the chain keeps the item on one node)')
        if 'ship' in stages:
            say(f'ship: fields -> {fields_dir} [snapshot_flow_field]; flow_<sb>.csv/.log -> {qa_dir}; G-S9 differing bytes refuse')
        say('gates: G-S1 GPU ids + the site GPU lock of each, G-S2 stage record (done, this node) + staging = its list and '
            'names_sha1, G-S3 broken Master link, G-S4 Master missing, G-S5 own-Master grid, G-S6 per-subband SET of '
            'expected (staged minus WSCNVIS = 0) vs flowed, G-S7 FAILED rows, G-S8 field per flowed I frame, G-S9 ship')
        for line in P:
            print(f'[plan] {line}')
        if refusal:
            print(f'[plan] {cell}: WOULD REFUSE - {refusal}')
            rc = 2
    return rc
