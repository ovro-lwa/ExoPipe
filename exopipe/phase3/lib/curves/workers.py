"""Extractor processes of `exopipe3 curves`: clv15/phase3_snapshot_lightcurves.py run IN PLACE, exactly as production runs
it, through the one child runner core.legacy.run (refactor decision 1, _CROSSCHECK.md; DESIGN 11.1).

Production (snapshot_fleet_node_v9.sh:362-371, snapshot_lightcurve_backfill.sh:67-73) runs, 8 at a time over (subband, Stokes):
    taskset -c $CORES $PY -u clv15/phase3_snapshot_lightcurves.py extract --cell C --subband SB --stage $STAGE/SB
        --proto-root <search outputs> --master-catalogs $OPS/snapshot_master_catalogs --out $LCOUT --stokes S
Here: the same script and arguments under the site interpreter 'py36' (the production interpreter, Python 3.6.13); PROCS
subbands at a time, each subband in one thread that runs the Stokes I process and then the Stokes V process (the V pass reads
FLUXSCAL / NREFSCAL from the I table in the same output directory, LC:180-192; P3_LC_WAIT_I_S = WAIT_I_TABLE_S = 0 because
the I table exists before V starts). --run-dir is passed only when curves.CHAIN_RUN_DIR is set (production passes none);
--samples-dir only when paths.CLV15_ROOT differs from the script's own directory (its default, LC:257). The core affinity is
inherited from `exopipe3 --cores`. core.legacy.run clears inherited P3_* variables, applies runner.ENV_DEFAULTS
(NUMPY_MADVISE_HUGEPAGE=0, which the backfill lacked), installs the parity guard in the child under --parity, and records the
sha1 of every legacy module the child imported (res.fingerprint; added to the unit record as meta 'legacy_code').

Outcome of one (subband, Stokes) process (``classify``):
  done      exit 0 and the table <out>/<cell>_<sb>_<S>_lightcurves.fits exists
  refused   exit 0 and no table: the module logged '<cell> <sb> <S>: no frames' (LC:106-108) -> G11;
            exit != 0 with the interpreter's final exception line 'RuntimeError: <message>' naming a gate of the module
            (LC:60 G6, :121 G7, :140 G8, :147 G9, :186 and :240 G10) -> that gate. A V table the module wrote before its
            inline test raised (LC:235 then :240) is NOT published (production's `cp -n` would ship it).
  error     anything else (traceback tail kept).
"""
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor

SCRIPT = 'phase3_snapshot_lightcurves.py'
# RuntimeError messages of the extractor -> the gate they are (spec curves.md section 5)
_GATES = (('no Master catalogue', 'G6'), ('no rows inside the frame', 'G7'), ('WCS differs from the first frame', 'G8'),
          ('no flow field at all', 'G9'), ('no Stokes I light-curve table', 'G10'), ('is not the Stokes I scale', 'G10'))
_EXC_LINE = re.compile(r'^(?P<type>[A-Za-z_][\w.]*(?:Error|Exception|Exit|Interrupt)): ?(?P<msg>.*)$')
_LOCK = threading.Lock()


def gate_of(msg):
    for key, gate in _GATES:
        if key in msg:
            return gate
    return None


def table_path(out_dir, cell, sb, st):
    return os.path.join(out_dir, f'{cell}_{sb}_{st}_lightcurves.fits')


def argv(C, paths_cfg, script, cell, sb, st, stage_sb, out_dir):
    """The production argument list (v9:366-370) for one (subband, Stokes)."""
    a = (['extract', '--cell', cell, '--subband', sb] + (['--stage', stage_sb] if stage_sb else [])   # none: archive frames
         + ['--proto-root', paths_cfg.SNAPSHOT_FLEET_OUT, '--master-catalogs', C.MASTER_CATALOG_DIR, '--out', out_dir,
            '--stokes', st])
    if C.CHAIN_RUN_DIR:
        a += ['--run-dir', C.CHAIN_RUN_DIR]
    if os.path.realpath(paths_cfg.CLV15_ROOT) != os.path.realpath(os.path.dirname(script)):
        a += ['--samples-dir', paths_cfg.CLV15_ROOT]
    return a


def child_env(C, paths_cfg):
    return {'P3_SNAP_FLOWTREE': paths_cfg.SNAPSHOT_FLOW_ROOT,        # read by phase3_snapshot_refastro at import (RA:73)
            'P3_LC_WAIT_I_S': str(C.WAIT_I_TABLE_S)}                 # LC:183


def classify(res, table, cell, sb, st):
    """{'status', 'gate', 'reason', ...} of one extractor process from its exit code, its table and the final exception
    line of its output."""
    tail = res.tail or ''
    out = {'rc': res.rc, 'wall_s': res.wall_s, 'log': res.log}
    if res.rc == 0 and os.path.exists(table):
        out.update(status='done', path=table)
        return out
    if res.rc == 0:
        if f'{cell} {sb} {st}: no frames' in tail:
            out.update(status='refused', gate='G11', reason=f'no frames of Stokes {st} (LC:106-108)')
        else:
            out.update(status='error', reason=f'exit 0 without a table {os.path.basename(table)}')
        return out
    exc = None
    for line in reversed(tail.splitlines()):
        m = _EXC_LINE.match(line.strip())
        if m:
            exc = m
            break
    if exc and exc['type'] == 'RuntimeError' and gate_of(exc['msg']):
        out.update(status='refused', gate=gate_of(exc['msg']), reason=exc['msg'])
        if os.path.exists(table):
            out['unpublished_table'] = table             # written before the inline V test raised (LC:235, :240)
    else:
        out.update(status='error', reason=(f"{exc['type']}: {exc['msg']}" if exc else f'exit {res.rc}'),
                   traceback=tail[-3000:])
    return out


def extract_subband(spec):
    """Run the Stokes processes of one subband in order; {stokes: outcome}."""
    from exopipe.phase3.core import legacy
    out = {}
    for st in spec['stokes']:
        table = table_path(spec['out_dir'], spec['cell'], spec['subband'], st)
        if st == 'V' and spec['skip_v_without_i'] and 'I' in spec['stokes'] and out.get('I', {}).get('status') != 'done':
            out[st] = {'status': 'refused', 'gate': 'G10',
                       'reason': f"no Stokes I table of this pass ({out.get('I', {}).get('status')}) - V carries the "
                                 f"Stokes I scale (LC:180-192)"}
            continue
        try:
            res = legacy.run(spec['script'], spec['argv'][st], python=spec['python'], env=spec['env'],
                             log=os.path.join(spec['out_dir'], f"lightcurves_{spec['subband']}_{st}.log"),
                             unit=None, refusal_codes=(), refusal_text=(), tail_lines=spec['tail_lines'])
        except Exception as e:                           # no interpreter, cannot start the child, disk full ...
            out[st] = {'status': 'error', 'reason': f'{type(e).__name__}: {e}'}
            continue
        o = classify(res, table, spec['cell'], spec['subband'], st)
        o['fingerprint'] = res.fingerprint
        o['argv'] = res.argv
        o['tail'] = (res.tail or '').splitlines()[-spec['tail_lines']:]
        out[st] = o
    return out


def make_spec(C, paths_cfg, script, python, cell, sb, stokes, stage_sb, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    return {'cell': cell, 'subband': sb, 'stokes': list(stokes), 'out_dir': out_dir, 'script': script,
            'python': python, 'env': child_env(C, paths_cfg), 'skip_v_without_i': bool(C.SKIP_V_WITHOUT_I),
            'tail_lines': int(C.LOG_TAIL_LINES),
            'argv': {st: argv(C, paths_cfg, script, cell, sb, st, stage_sb, out_dir) for st in stokes}}


def run_all(specs, procs, unit=None, on_done=None):
    """Extract every subband, ``procs`` subbands at a time; {subband: {stokes: outcome}}. The child calls and their code
    fingerprints are added to the unit record (meta 'legacy_calls', 'legacy_code'), as core.legacy.run(unit=u) does."""
    out = {}
    with ThreadPoolExecutor(max_workers=max(1, int(procs))) as ex:
        futs = {ex.submit(extract_subband, s): s['subband'] for s in specs}
        for fut in futs:
            sb = futs[fut]
            try:
                out[sb] = fut.result()
            except Exception as e:
                out[sb] = {st: {'status': 'error', 'reason': f'{type(e).__name__}: {e}'} for st in ('I', 'V')}
            if unit is not None:
                with _LOCK:
                    for st, o in sorted(out[sb].items()):
                        if 'argv' in o:
                            unit.meta.setdefault('legacy_calls', []).append(
                                {'argv': o['argv'], 'rc': o.get('rc'), 'wall_s': o.get('wall_s'), 'log': o.get('log')})
                        unit.meta.setdefault('legacy_code', {}).update(o.get('fingerprint') or {})
            if on_done:
                on_done(sb, out[sb])
    return out


# ---------------------------------------------------------------------------------------------------------------------
# facts read back from a written table (run record; post-check G4) - the parent reads, the child wrote
# ---------------------------------------------------------------------------------------------------------------------

def staged_frames(stage_sb):
    """{'I': {ts: path}, 'V': {ts: path}} of the decompressed frames staged in one subband directory (Phase-1 names, the
    patterns of core.inputs; *_flowed.fits siblings are not frames)."""
    from exopipe.phase3.core.inputs import _PAT_NEW_I, _PAT_NEW_V, _PAT_OLD
    out = {'I': {}, 'V': {}}
    try:
        names = os.listdir(stage_sb)
    except FileNotFoundError:
        return out
    for fn in names:
        if not fn.endswith('-image.fits'):
            continue
        m = _PAT_OLD.match(fn)
        if m:
            out[m['st']][m['ts']] = os.path.join(stage_sb, fn)
            continue
        m = _PAT_NEW_I.match(fn)
        if m:
            out['I'][m['ts']] = os.path.join(stage_sb, fn)
            continue
        m = _PAT_NEW_V.match(fn)
        if m:
            out['V'][m['ts']] = os.path.join(stage_sb, fn)
    return out


def archive_frames(cell, sb):
    """{'I': {ts: path}, 'V': {...}}: the Phase-1 archive frames the extractor reads when it gets no --stage - its own
    lister, clv15 phase3_snapshot_search.discover_frames(stage_root=None, flow_select=False), called in place."""
    from exopipe.phase3.core import env
    env.legacy()
    from phase3_snapshot_search import discover_frames
    lst, date = cell.split('_')[:2]
    fr = discover_frames(lst, date, sb, stage_root=None, flow_select=False)
    return {st: dict(fr.get(st, [])) for st in ('I', 'V')}


def _frame_cards(p):
    """Header cards of one frame: the primary HDU of a staged (funpacked) frame; HDU 1 of an archive tile-compressed one
    (.fs on the 2024 runs, .fz on the 2025 ones - any frame whose primary HDU has no image), as the extractor's
    SS.read_frame takes it (clv15 phase3_snapshot_search.py:300-310). (Review 2026-10-13: '.fs' was not handled - every
    2024 archive frame gave NAXIS 0 and content test LC3 refused every 2024 table read from the archive.)"""
    from exopipe.phase3.core.record import fits_primary_cards
    cards = fits_primary_cards(p)
    if not int(cards.get('NAXIS', 0) or 0):
        from astropy.io import fits
        with fits.open(p) as h:
            if len(h) > 1:
                cards = dict(h[1].header)
    return cards


def _json_header(cards):
    """The cards of a frame header as JSON values (COMMENT / HISTORY / blank dropped): the first frame's WCS travels in the
    content-test context (light_curve.rows completeness = the regeneration test T2) and in the run record."""
    out = {}
    for k, v in cards.items():
        if not k or k in ('COMMENT', 'HISTORY') or k in out:
            continue
        if isinstance(v, (bool, int, str)) or (isinstance(v, float) and v == v):
            out[k] = v
        else:
            out[k] = str(v)
    return out


def master_file(lst, sb):
    """The subband's own snapshot Master (frames_snapshot.MASTER_FILE under masters_snapshot.LINK_ROOT; v9:135)."""
    from exopipe.phase3.config import CFG
    return CFG.frames_snapshot.MASTER_FILE.format(links=CFG.masters_snapshot.LINK_ROOT, lst=lst, sb=sb)


def archive_grid_problem(cell, sb):
    """Gate G17 (archive frames only, curves.FRAME_SOURCE 'archive'): why the archive frames of (cell, sb) are NOT the
    frames the fleet recipe extracts from, else None. The fleet regrids a subband in the staging when its first frame has
    NAXIS1 != frames_snapshot.ON_GRID_NAXIS (1508; an unreadable header counts as 1508) and the subband has its own Master,
    onto that Master's grid (snapshot_fleet_node_v9.sh:126-139; clv15 phase3_snapshot_regrid.py:23-25 regrids every frame
    whose NAXIS1 differs from the Master's): the pilot-era cells (4096 px, December 2024 before the 27th). The archive
    frames of such a subband are on another grid than the flow fields (computed on the regridded frames) and than the
    staged tables; the regeneration recipe's T4 (exopipe_ops snapshot_lightcurves_regen_test.py:86-98) fails them and
    nothing is installed - here they are refused before extraction: use the fleet recipe (--stage-root of a regridded
    stage, or curves.FRAME_SOURCE=staged). The frame v9 tests is the first staged '*-image.fits' by name (`ls | head -1`,
    v9:128): the archive frame of both Stokes whose name without .fs/.fz sorts first."""
    from exopipe.phase3.config import CFG
    from exopipe.phase3.core.record import fits_primary_cards
    lst = cell.split('_')[0]
    on_grid = int(CFG.frames_snapshot.ON_GRID_NAXIS)
    fr = archive_frames(cell, sb)
    paths = [p for st in ('I', 'V') for p in fr.get(st, {}).values()]
    if not paths:
        return None                                         # no frame: the extractor's own outcome (G11)
    first = min(paths, key=lambda p: re.sub(r'\.f[sz]$', '', os.path.basename(p)))
    try:
        nax = int(_frame_cards(first).get('NAXIS1') or on_grid)
    except (OSError, ValueError, TypeError):
        nax = on_grid                                       # v9:129-130: ${NAX:-1508}
    if nax == on_grid:
        return None
    m = master_file(lst, sb)
    try:
        mnax = int(fits_primary_cards(m).get('NAXIS1') or 0) if os.path.exists(m) else None
    except OSError:
        mnax = None
    if mnax and nax != mnax:                                # no own Master: v9 leaves the frames on their grid (REGRID-SKIPPED)
        return (f'G17: archive frames of {cell} {sb} are {nax} px (first frame {os.path.basename(first)}); the fleet '
                f'regrids them in the staging onto the own Master ({mnax} px; v9:126-139), so the archive frames are not '
                f'the extracted frames - use --stage-root (a regridded stage) or curves.FRAME_SOURCE=staged')
    return None


def table_facts(table, st, stage_sb, admission_csv, samples_dir, samples, cell=None, sb=None):
    """Fallback counts of one written table (spec section 5: kept as today, counted per pair) and the post-check of G4:
    frames_absent_from_admission (admitted by default, LC:156), admission_table_missing, scale_from_all_frames (NADMIT 0,
    LC:166), zero_vis_frames_kept (staged frames with WSCNVIS = 0: ledger 1016/2258 not applied by the extractor),
    missing_sample_files (CH:1643-1645), unstaged_frames (frames of the table without a staged file: the extractor read the
    archive copy; G4), naxis of the first frame (None when unreadable), frame_header (the first frame's header cards as
    JSON: the WCS the extractor selected its rows with, LC:118-127 - content test light_curve.rows completeness),
    mixed_grid_frames (frames whose NAXIS1 differs from the first frame's: reported - where the fleet does not regrid,
    v9:126-139, the staged frames are the archive frames). stage_sb None = archive
    frames (curves.FRAME_SOURCE 'archive'; give cell and sb): every frame is read from the archive by design,
    unstaged_frames is 0."""
    import csv
    from astropy.io import fits
    with fits.open(table, memmap=True) as h:
        ts = [x.decode().strip() if isinstance(x, bytes) else str(x).strip() for x in h['FRAMES'].data['ts']]
        nadmit = int(h[0].header['NADMIT'])
    adm = None
    if os.path.exists(admission_csv):
        with open(admission_csv) as f:
            adm = {r['ts'] for r in csv.DictReader(f)}
    staged = staged_frames(stage_sb)[st] if stage_sb else archive_frames(cell, sb)[st]
    zero, naxis, header, mixed = 0, None, None, 0
    for t in ts:
        p = staged.get(t)
        if not p:
            continue
        try:
            cards = _frame_cards(p)
        except OSError:
            continue
        if naxis is None:
            naxis = (int(cards.get('NAXIS1', 0) or 0), int(cards.get('NAXIS2', 0) or 0))
            if t == ts[0]:                       # the extractor's first frame (LC:118-127): its WCS selects the rows
                header = _json_header(cards)
        elif int(cards.get('NAXIS1', 0) or 0) != naxis[0]:
            mixed += 1
        nv = cards.get('WSCNVIS', 1)
        try:
            if nv is not None and float(nv) == 0:
                zero += 1
        except (TypeError, ValueError):
            pass
    if naxis is not None and (naxis[0] <= 0 or naxis[1] <= 0):
        naxis = None                             # unknown - never a 0 x 0 frame
    return {'frames_absent_from_admission': len(ts) if adm is None else sum(1 for t in ts if t not in adm),
            'admission_table_missing': adm is None, 'scale_from_all_frames': nadmit == 0,
            'zero_vis_frames_kept': zero,
            'missing_sample_files': [fn for _tag, fn in samples if not os.path.exists(os.path.join(samples_dir, fn))],
            'unstaged_frames': sum(1 for t in ts if t not in staged), 'naxis': naxis, 'frame_header': header,
            'mixed_grid_frames': mixed}
