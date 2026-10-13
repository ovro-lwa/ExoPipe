"""exopipe3 search --tier snapshot - the per-frame snapshot search of one LST hour of one night (the per-hour unit).

Called by steps/search.py (the tier dispatcher) through add_arguments(p) and run(args, run).

Unit of work: a snapshot NIGHT CELL <LST>_<night> (core.cells.parse_night_cell; e.g. 21h_2025-08-26 = one epoch item,
DESIGN 12.1), record _records/search/<item>.json. Items: --cells 21h_2025-08-26,@list (patterns such as 21h_2025-08-* match
the Phase-1 archive's <LST>/<date> directories) or the per-hour --epochs <LST>_<date> (the same cells). A night cell with a
subband suffix (21h_2025-08-26_36MHz) searches that subband only (a subset re-search: no group search, as v9 with
P3_FLEET_SUBSET=1); --scopes limits the subbands searched. Every (subband or wideband group, Stokes) outcome is accounted for
inside the unit record: searched (runinfo written), night-refused (nightrefused_<tag>.json, Gregg 2026-10-13 ruling 4),
excluded, no-members, too-few-frames - or the unit is refused / errors.

Stages (search_snapshot.STAGES; the step's --stages/--from/--until pick a subset):
  preflight     the cell's staged + flowed frames (frames/flow --tier snapshot products, node-local:
                <scratch>/snapshot_stage/<cell>/<sb>/, or --snap-stage-root DIR/<cell>/): staged frames = the .filelist
                and no compressed/tmp leftover (v9 STAGE-REFUSED); no broken Master link and a Taper R0 Master for every
                subband (v9 REFUSED, the hourly-screen fallback is retired); the per-subband SET of staged frames equals
                the set of flowed frames, zero-visibility frames (own header WSCNVIS = 0) left out of both (v9 FLOW-INCOMPLETE,
                cf72d51); at least MIN_STAGED_FRAMES staged frames; the solar audit's pinned md5; the engine interpreter.
  subbands      clv15 phase3_snapshot_search.py per subband IN PLACE (core.legacy.run, site PYTHON['py36'], production
                arguments: --stokes VI --mode median --thresh 5.0 --nthreads 8 --stage-root <stage>/<sb>; P3_SNAP_FLOW=1),
                NPAR at a time, into the node-local output <scratch>/snapshot_out/<cell>/; then SUBBAND COMPLETENESS (with
                the completeness stage): every staged subband x Stokes has its runinfo OR its night-refusal record, else
                SEARCH-INCOMPLETE (error, retried; v9:206-229); no runinfo and no record at all = systemic failure (v9
                FLEET-ABORT, error). The photometry cutout cubes the engine writes (class photometry_cutout, owned by
                `photometry`) are recorded on write.
  widebands     clv15 phase3_snapshot_group_search.py per Stokes, V then I (v9:235-239); skipped with --snap-bands subbands,
                a subband item or the 1-minute pause; then WIDEBAND COMPLETENESS: wideband_outcome_<S>.json exists and
                Red/Green/Blue are each searched (with runinfo), night-refused (with record), excluded, no-members or
                too-few-frames (v9:241-269).
  completeness  the two completeness gates above (named stage: dropping it from --stages skips them).
  verify        the content test of every product in the node-local output (snapshot_search, snapshot_1min,
                snapshot_wideband, snapshot_model, night_refusal, snapshot_table).
  ship          copy *.csv *.json *.txt *.log and model_*_median.fits to <out>/<cell>/ (v9:333-337: cp -n); an existing
                DIFFERENT file refuses the unit (no-clobber content test) unless --fresh (then the replaced file is kept under
                <out>/_superseded/); every product is recorded there with its class; the node-local output is removed.
  Optional (never default during the window): fit (window home = catalog's store/prefit, spec K14) and tier1min
  (snapshot_frametime_rerun.sh) are REFUSED here with that reason.

--inject MANIFEST | auto[:N] (the step's flag): an injection search of the cell, recorded, never shipped to the product
tree: auto writes the SNR-drawn manifest from the cell's shipped baseline noise tables (clv15 phase3_snapshot_inject.py
--snr-from-cell, N per Stokes, default 1000 or --snap-inject-n); then the subband search with --inject into
<out>/inject/<tag>/<cell>/ and the recovery table (phase3_snapshot_recover.py --simple). Wideband groups cannot be injected
(FIX-INJECTION-PATH-FOR-THE-1-MINUTE-TIER): the groups part is not run and the record says so. Unit <item>.inject.<tag>.

Inputs:  the staged/flowed frames and their .filelist; the Masters (search_snapshot.MASTERS); data_decisions.csv (read by the
         engines and core.exclusions); the solar audit (pinned); the PB library (engine); the clv15 engine sources.
Outputs (product classes): runinfo_<tag>.json (snapshot_search: with its candidates/events/noise/admission tables),
         noise_<tag>_1min.csv (snapshot_1min), model_<tag>.fits (snapshot_model), nightrefused_<tag>.json (night_refusal),
         wideband_outcome_<S>.json (snapshot_wideband), every other shipped CSV (snapshot_table); the photometry cutout
         cubes under <out>/photometry/ while paths.PRODUCTION_WRITES is False (photometry_cutout); injection runs:
         injected_<sb>_<S>.csv (injection_book). tag = <subband|group>_<S>_median.
Failure modes: exit 2 = a gate refused (staging incomplete, broken/missing Master, flow incomplete, too few frames, pinned md5,
         a staged subband excluded under v9's accounting, no-clobber, a production --out while paths.PRODUCTION_WRITES is
         False, every planned search refused); exit 1 = SEARCH-INCOMPLETE / systemic failure (an engine crash, a full disk:
         logs kept under <out>/_failed/<cell>_<UTC>/), too little free disk; exit 3 = no item.
"""
import fnmatch
import glob
import hashlib
import json
import os
import re
import shutil
import time

TIER = 'snapshot'
CONFIG = ('search_snapshot', 'runner')
_IMG_RE = re.compile(r'-image\.fits$')
_LEFTOVER = ('*.fs', '*.fz', '*.tmp')


def _h():
    from exopipe.phase3.lib.search import tier_sublst as H
    return H


def cfg():
    return _h().cfg()


def add_arguments(p):
    g = p.add_argument_group('tier snapshot (search --tier snapshot)')
    g.add_argument('--snap-bands', choices=('all', 'subbands', 'groups'), default='all',
                   help="band set: all (default); subbands = no wideband group search and no wideband completeness "
                        "(v9 P3_FLEET_SUBSET=1); groups = wideband groups only (reads the cell's shipped subband admission)")
    g.add_argument('--snap-stage-root', metavar='DIR',
                   help='root of the staged + flowed frames, DIR/<cell>/<sb>/ (default <scratch>/'
                        'search_snapshot.STAGE_SUBDIR)')
    g.add_argument('--snap-pause-1min', action='store_true',
                   help='write no 1-minute tier (runinfo tier_1min = paused) and skip the group search '
                        '(= --set search_snapshot.PAUSE_1MIN=True; P3_SNAP_PAUSE_1MIN)')
    g.add_argument('--snap-inject-n', type=int, default=None,
                   help='with --inject auto: injections per Stokes (default 1000, as auto:N)')


def default_out(args):
    return cfg().paths.SNAPSHOT_FLEET_OUT       # SHIP (v9:39): <root>/<cell>/


def engine_dir():
    """The clv15 root the snapshot engines run from in place (core.env.legacy_root; v9:43 CODE)."""
    from exopipe.phase3.core import env
    return env.legacy_root('clv15')


# ---------------------------------------------------------------------------------------------------------------------
# items
# ---------------------------------------------------------------------------------------------------------------------

def _universe():
    root = cfg().paths.PHASE1_ROOT
    out = []
    for d in sorted(glob.glob(os.path.join(root, '[0-2][0-9]h', '????-??-??'))):
        out.append(f'{os.path.basename(os.path.dirname(d))}_{os.path.basename(d)}')
    return out


def expand_cells(args):
    """[(item, cell, subband-or-None)] from --cells (night cells, core.cells.parse_night_cell) and --epochs (the per-hour
    selector, core.cells.expand_epochs). A pattern in --cells matches the Phase-1 archive's <LST>/<date> directories."""
    from exopipe.phase3.core import cells as C
    H = _h()
    toks = []
    spec = getattr(args, 'cells', None)
    if spec:
        raw = [t.strip() for t in str(spec).split(',') if t.strip()]
        uni = _universe() if any(any(ch in t for ch in '*?[') for t in raw) else None
        toks += C.expand(spec, universe=uni, kind='snapshot cell')
    if getattr(args, 'epochs', None):
        toks += C.expand_epochs(args.epochs)
    if not toks:
        raise H.refusal('tier snapshot needs --cells <LST>_<YYYY-MM-DD>[_<subband>] or --epochs <LST>_<YYYY-MM-DD>')
    excl = set(C.expand(getattr(args, 'exclude_cells', None))) if getattr(args, 'exclude_cells', None) else set()
    s = cfg().search_snapshot
    out, seen = [], set()
    for t in toks:
        try:
            lst, date, sb = C.parse_night_cell(t)
        except ValueError:
            raise H.refusal(f'{t!r}: not a snapshot night cell (<LST>_<YYYY-MM-DD>[_<subband>], e.g. 21h_2025-08-26)')
        if sb and sb not in s.SUBBANDS:
            raise H.refusal(f'{t}: {sb} is not a snapshot search subband (the wideband groups are searched with the cell)')
        cell = f'{lst}_{date}'
        if t in excl or cell in excl or t in seen:
            continue
        seen.add(t)
        out.append((t, cell, sb))
    return out


def subbands_of(args, only=None):
    s = cfg().search_snapshot
    sbs = list(s.SUBBANDS) if only is None else [only]
    sel = getattr(args, 'scopes', None)
    if not sel:
        return sbs
    from exopipe.phase3.core import cells as C
    want = set(C.expand(sel))
    bad = [x for x in want if x not in s.SUBBANDS and x not in s.GROUPS]
    if bad:
        raise _h().refusal(f'--scopes {",".join(bad)}: not a snapshot subband ({", ".join(s.SUBBANDS)})')
    return [sb for sb in sbs if sb in want]


def stage_root(args, cell, dry):
    s = cfg().search_snapshot
    if getattr(args, 'snap_stage_root', None):
        return os.path.join(args.snap_stage_root, cell)
    from exopipe.phase3.core import fitsio
    try:
        return os.path.join(fitsio.scratch_root(getattr(args, 'work', None)), s.STAGE_SUBDIR, cell)
    except Exception:
        if dry:
            return os.path.join('<node-local scratch: --work or site SCRATCH_ROOT>', s.STAGE_SUBDIR, cell)
        raise


def work_dir(args, name):
    from exopipe.phase3.core import fitsio
    s = cfg().search_snapshot
    return os.path.join(fitsio.scratch_root(getattr(args, 'work', None)), s.WORK_SUBDIR, name)


# ---------------------------------------------------------------------------------------------------------------------
# preflight (port of v9:92-202 checks on the staged cell; the staging itself is frames --tier snapshot)
# ---------------------------------------------------------------------------------------------------------------------

def _staged(stage, sb):
    d = os.path.join(stage, sb)
    if not os.path.isdir(d):
        return [], [], []
    names = os.listdir(d)
    imgs = sorted(n for n in names if _IMG_RE.search(n))                  # '*-image.fits' (not *_flowed.fits)
    flowed = sorted(n for n in names if n.endswith('-image_flowed.fits'))
    left = sorted(n for n in names if any(fnmatch.fnmatch(n, pat) for pat in _LEFTOVER))
    return imgs, flowed, left


def _wscnvis_zero(path):
    from astropy.io import fits
    try:
        v = fits.getheader(path).get('WSCNVIS', 1)
    except Exception:
        return False                       # an unreadable header is kept (v9:166-169; B-23 default (c))
    try:
        return v is not None and float(v) == 0
    except (TypeError, ValueError):
        return False


def preflight(args, run, cell, only_sb=None):
    H = _h()
    c = cfg()
    s = c.search_snapshot
    lst, date = cell.split('_', 1)
    stage = stage_root(args, cell, run.dry_run)
    sbs = subbands_of(args, only_sb)
    plan = dict(cell=cell, lst=lst, date=date, stage=stage, subbands=sbs, staged={}, gates=[], refuse=None,
                inputs=[], zero_vis={}, excluded={}, excluded_groups={}, n_staged=0)

    def gate(name, ok, detail):
        plan['gates'].append((name, bool(ok), detail))
        return ok

    from exopipe.phase3.core import exclusions
    ex = exclusions.load()
    plan['excluded'] = {sb: True for sb in s.SUBBANDS if ex.is_excluded(lst, sb, date)}
    plan['excluded_groups'] = {g: True for g in s.GROUPS if ex.is_excluded(lst, g, date)}
    if not sbs:
        plan['refuse'] = f'no subband selected for {cell} (--scopes)'
        return plan
    if not gate('stage.present', os.path.isdir(stage), stage):
        plan['refuse'] = (f'no staged frames for {cell} at {stage}: stage and flow the cell first (frames --tier snapshot '
                          f'--stages stage,regrid; flow --tier snapshot), on this node')
        return plan
    n_img = n_list = n_left = 0
    per = {}
    for sb in s.SUBBANDS:
        imgs, flowed, left = _staged(stage, sb)
        fl = os.path.join(stage, sb, '.filelist')
        nl = 0
        if os.path.exists(fl):
            with open(fl) as f:
                nl = sum(1 for line in f if line.strip())
        per[sb] = dict(imgs=imgs, flowed=flowed, left=left, filelist=nl)
        n_img += len(imgs)
        n_list += nl
        n_left += len(left)
    plan['staged'] = {sb: len(v['imgs']) for sb, v in per.items() if v['imgs']}
    plan['n_staged'] = n_img
    if not gate('S14.staging_complete', n_left == 0 and n_img == n_list,
                f'{n_img} decompressed of {n_list} listed frames, {n_left} compressed/tmp leftovers'):
        plan['refuse'] = (f'STAGE-REFUSED: {cell} decompressed {n_img} of {n_list} listed frames, {n_left} compressed/tmp '
                          f'leftovers (v9:116-122)')
        return plan
    broken, nomaster = [], []
    for sb in s.SUBBANDS:
        m = os.path.join(s.MASTERS, f'{lst}_{sb}', 'coadd_final', s.MASTER_FILE.format(sb=sb))
        if os.path.islink(m) and not os.path.exists(m):
            broken.append(sb)
        elif not os.path.exists(m):
            nomaster.append(sb)
        else:
            plan['inputs'].append(os.path.realpath(m))
    if not gate('S3.no_broken_master_link', not broken, ','.join(broken) or 'none'):
        plan['refuse'] = f'REFUSED: {cell} broken Master link(s): {" ".join(broken)} (v9:146-151)'
        return plan
    if not gate('S3.master_every_subband', not nomaster, ','.join(nomaster) or 'all present'):
        plan['refuse'] = (f'REFUSED: {cell} no Taper R0 Master: {" ".join(nomaster)} - every frame must be flow-aligned to '
                          f'its subband\'s Master; the hourly-screen fallback is retired (v9:176-181, Gregg 2026-10-04)')
        return plan
    missing = {}
    for sb, v in per.items():
        if not v['imgs']:
            continue
        zv = {n for n in v['imgs'] if _wscnvis_zero(os.path.join(stage, sb, n))}
        plan['zero_vis'][sb] = len(zv)
        flowed_bases = {n[:-len('_flowed.fits')] + '.fits' for n in v['flowed']}
        miss = sorted(set(v['imgs']) - zv - flowed_bases)
        if miss:
            missing[sb] = len(miss)
    nzv = sum(plan['zero_vis'].values())
    if not gate('S15.flowed_set', not missing,
                (f'missing flowed copies per subband {missing}' if missing else 'every staged frame has its flowed copy')
                + f'; {nzv} zero-visibility frames left out of both sets'):
        plan['refuse'] = (f'FLOW-INCOMPLETE: {cell} staged frames without a flowed copy (zero-visibility frames excepted) per '
                          f'subband: {missing} - the engine would refuse those subbands (v9:191-198; per-subband set, cf72d51)')
        return plan
    if not gate('S16.min_staged_frames', n_img >= s.MIN_STAGED_FRAMES, f'{n_img} staged frames, minimum '
                                                                        f'{s.MIN_STAGED_FRAMES}'):
        plan['refuse'] = f'REFUSED: {cell} only {n_img} decompressed frames (need {s.MIN_STAGED_FRAMES}; v9:199-202)'
        return plan
    try:
        plan['python'] = H.site_python(s.ENGINE_PYTHON)
        gate('site.engine_python', True, f"PYTHON[{s.ENGINE_PYTHON!r}] = {plan['python']}")
    except Exception as e:
        gate('site.engine_python', False, str(e))
        plan['refuse'] = str(e)
        return plan
    for scr in ('phase3_snapshot_search.py', 'phase3_snapshot_group_search.py'):
        p = os.path.join(engine_dir(), scr)
        if not gate(f'engine.{scr}', os.path.exists(p), p):
            plan['refuse'] = f'engine script {p} missing'
            return plan
    if os.path.exists(s.SOLAR_AUDIT):
        m = H.md5_file(s.SOLAR_AUDIT)
        if not gate('G2.solar_audit_md5', m == s.SOLAR_AUDIT_MD5, f'md5 {m}'):
            plan['refuse'] = f'G2: {s.SOLAR_AUDIT} md5 {m} != pinned {s.SOLAR_AUDIT_MD5} (the night rule reads it)'
            return plan
    else:
        gate('G2.solar_audit', False, f'{s.SOLAR_AUDIT} missing')
        plan['refuse'] = f'G2: the solar audit {s.SOLAR_AUDIT} (night rule) is missing'
        return plan
    # inputs: per staged subband a digest of the staged set (names, sizes, mtimes) - frames are never hashed one by one
    for sb, v in per.items():
        if v['imgs'] and sb in sbs:
            plan['inputs'].append((os.path.join(stage, sb), _set_digest(stage, sb, v)))
            fl = os.path.join(stage, sb, '.filelist')
            if os.path.exists(fl):
                plan['inputs'].append(fl)
    for p in (c.paths.DATA_DECISIONS, s.SOLAR_AUDIT,
              os.path.join(engine_dir(), 'phase3_snapshot_search.py'),
              os.path.join(engine_dir(), 'phase3_snapshot_group_search.py'),
              os.path.join(engine_dir(), 'phase3_snapshot_cutouts.py'),
              '/lustre/gh/main/exopipe/exopipe/phase3/coadd_full_pipeline.py'):   # GS:157-158 imports production's
        if os.path.exists(p):
            plan['inputs'].append(p)
    return plan


def _set_digest(stage, sb, v):
    h = hashlib.sha1()
    for n in v['imgs'] + v['flowed']:
        try:
            st = os.stat(os.path.join(stage, sb, n))
            h.update(f'{n}\0{st.st_size}\0{st.st_mtime_ns}\n'.encode())
        except OSError:
            h.update(f'{n}\0missing\n'.encode())
    return {'staged_set_sha1': h.hexdigest(), 'n_frames': len(v['imgs']), 'n_flowed': len(v['flowed'])}


# ---------------------------------------------------------------------------------------------------------------------
# engines in place (core.legacy.run)
# ---------------------------------------------------------------------------------------------------------------------

def cutout_root(run):
    """P3_SNAP_CUTOUT_ROOT: paths.PHASE3_ARCHIVE_ROOT/<CUTOUT_SUBDIR> (production) or <out>/<CUTOUT_SUBDIR> while
    paths.PRODUCTION_WRITES is False."""
    c = cfg()
    return os.path.join(c.paths.PHASE3_ARCHIVE_ROOT if c.paths.PRODUCTION_WRITES else run.out_root,
                        c.search_snapshot.CUTOUT_SUBDIR)


def engine_env(run, args):
    """search.md 7.3 (snapshot rows): the engines' environment from config (core.legacy.run clears inherited P3_*).
    PYTHONUNBUFFERED=1 stands for v9's `python -u`."""
    s = cfg().search_snapshot
    pause = bool(getattr(args, 'snap_pause_1min', False) or s.PAUSE_1MIN)
    return {
        'P3_SNAP_FLOW': '1' if s.FLOW_REQUIRED else None,
        'P3_SNAP_FLOW_MAX_MHZ': f'{s.FLOW_MAX_MHZ:g}',
        'P3_SNAP_PAUSE_1MIN': '1' if pause else None,
        'P3_SNAP_CUTOUTS': '1' if s.CUTOUTS_ON else '0',
        'P3_SNAP_CUTOUT_ROOT': cutout_root(run),
        'P3_SOLAR_AUDIT': s.SOLAR_AUDIT,
        'PYTHONUNBUFFERED': '1',
    }, pause


def search_subbands(u, plan, out, env, inject=None):
    """phase3_snapshot_search.py per subband, NPAR at a time (v9 search_cell). Returns {sb: outcome dict}."""
    H = _h()
    s = cfg().search_snapshot
    os.makedirs(out, exist_ok=True)

    def one(sb):
        args = ['--lst', plan['lst'], '--date', plan['date'], '--subband', sb, '--stokes', 'VI', '--mode', s.MODE,
                '--thresh', f'{s.THRESH_SIGMA:g}', '--out', out, '--stage-root', os.path.join(plan['stage'], sb),
                '--nthreads', str(int(s.NTHREADS))] + (['--inject', inject] if inject else [])
        log = os.path.join(out, f'search_{sb}.log')
        if os.path.exists(log):
            os.remove(log)                  # v9 writes the log with '>' (one log per attempt)
        r = H.legacy_call(os.path.join(engine_dir(), 'phase3_snapshot_search.py'), args, s.ENGINE_PYTHON, env, log, unit=u)
        return sb, r.rc, r.wall_s

    from concurrent.futures import ThreadPoolExecutor
    outcomes = {}
    with ThreadPoolExecutor(max_workers=max(1, int(s.NPAR))) as ex:
        for sb, rc, dt in ex.map(one, plan['subbands']):
            o = dict(rc=rc, seconds=dt, staged=plan['staged'].get(sb, 0), stokes={})
            if rc == 3:                         # SS:1490-1495: the engine's only exit 3 = excluded in data_decisions.csv
                o['outcome'] = 'excluded'
            for st in ('V', 'I'):
                tag = f'{sb}_{st}_{s.MODE}'
                if os.path.exists(os.path.join(out, f'runinfo_{tag}.json')):
                    o['stokes'][st] = 'searched'
                elif os.path.exists(os.path.join(out, f'nightrefused_{tag}.json')):
                    o['stokes'][st] = 'night-refused'
                elif o.get('outcome') == 'excluded':
                    o['stokes'][st] = 'excluded'
                elif not o['staged']:
                    o['stokes'][st] = 'not-staged'
                else:
                    o['stokes'][st] = 'failed: ' + (H.log_tail(os.path.join(out, f'search_{sb}.log'), 2) or f'exit {rc}')
            outcomes[sb] = o
    u.meta['subbands'] = outcomes
    return outcomes


def subband_completeness(u, plan, out, outcomes):
    """v9:206-234 with the accounting rule of IMPLEMENTED-SNAPSHOT-NIGHT-REFUSED-BANDS-ACCOUNTED-20261013."""
    s = cfg().search_snapshot
    n_ri = n_nr = 0
    miss, excl_only = [], []
    for sb in plan['subbands']:
        if not plan['staged'].get(sb):
            continue
        for st in ('V', 'I'):
            v = outcomes.get(sb, {}).get('stokes', {}).get(st, 'failed: not run')
            if v == 'searched':
                n_ri += 1
            elif v == 'night-refused':
                n_nr += 1
            elif v == 'excluded' and s.EXCLUDED_STAGED_SUBBAND == 'accounted':
                pass
            elif v == 'excluded':
                excl_only.append(f'{sb}_{st}')
            else:
                miss.append(f'{sb}_{st} ({v})')
    n_exp = 2 * sum(1 for sb in plan['subbands'] if plan['staged'].get(sb))
    detail = (f'runinfo={n_ri} + night-rule refusals={n_nr} of {n_exp} expected '
              f'({n_exp // 2} staged subbands x 2 Stokes)')
    u.check('completeness:subbands', not miss and not excl_only, value=n_ri + n_nr, detail=detail +
            (f'; unaccounted: {miss + excl_only}' if miss or excl_only else ''))
    if miss:
        raise RuntimeError(f'SEARCH-INCOMPLETE {plan["cell"]} {detail}; unaccounted: {" ".join(miss)} - not shipped '
                           f'(v9:226-229; a crash or a full disk: retried)')
    if excl_only:
        raise _h().refusal(f'SEARCH-INCOMPLETE {plan["cell"]} (gate exclusion): staged subband x Stokes {" ".join(excl_only)} '
                           f'excluded in data_decisions.csv - the engine writes no output and v9 counts it unaccounted '
                           f'(search_snapshot.EXCLUDED_STAGED_SUBBAND = unaccounted, v9 before exopipe_ops a40b088)')
    if n_ri == 0 and n_nr == 0:
        raise RuntimeError(f'FLEET-ABORT {plan["cell"]} produced no runinfo and no refusal record - systemic failure '
                           f'(v9:231-234 stops the node: hold this node in the runner)')
    if n_nr:
        u.note(f'NIGHT-REFUSED {plan["cell"]}: {n_nr} of {n_exp} searches refused by the night rule (recorded)')


def search_widebands(u, plan, out, env, parallel):
    H = _h()
    s = cfg().search_snapshot

    def one(st):
        args = ['--lst', plan['lst'], '--date', plan['date'], '--stokes', st, '--thresh', f'{s.THRESH_SIGMA:g}',
                '--out', out, '--stage-root-fmt', os.path.join(plan['stage'], '%s'), '--nthreads', str(int(s.NTHREADS))]
        log = os.path.join(out, f'group_{st}.log')
        if os.path.exists(log):
            os.remove(log)
        r = H.legacy_call(os.path.join(engine_dir(), 'phase3_snapshot_group_search.py'), args, s.ENGINE_PYTHON, env, log,
                          unit=u)
        return st, r.rc, r.wall_s

    res = {}
    if parallel:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=2) as ex:
            for st, rc, dt in ex.map(one, ('V', 'I')):
                res[st] = dict(rc=rc, seconds=dt)
    else:
        for st in ('V', 'I'):
            st, rc, dt = one(st)
            res[st] = dict(rc=rc, seconds=dt)
    u.meta['widebands'] = res
    return res


def wideband_completeness(u, plan, out):
    """v9:246-264 in Python: each Stokes' outcome record exists and every group is accounted for."""
    s = cfg().search_snapshot
    bad = []
    for st in ('V', 'I'):
        p = os.path.join(out, f'wideband_outcome_{st}.json')
        if not os.path.exists(p):
            bad.append(f'{st}:no-outcome-record')
            continue
        with open(p) as f:
            g = json.load(f).get('groups', {})
        for grp in s.GROUPS:
            v = g.get(grp, 'missing')
            ok = (v in ('excluded', 'no-members', 'too-few-frames')
                  or (v == 'searched' and os.path.exists(os.path.join(out, f'runinfo_{grp}_{st}_{s.MODE}.json')))
                  or (v == 'night-refused' and os.path.exists(os.path.join(out, f'nightrefused_{grp}_{st}_{s.MODE}.json'))))
            if not ok:
                bad.append(f'{grp}_{st}:{v}')
    u.check('completeness:widebands', not bad, value=len(bad), detail=' '.join(bad) or 'every group accounted for')
    if bad:
        raise RuntimeError(f'SEARCH-INCOMPLETE {plan["cell"]} wideband group search unaccounted: {" ".join(bad)} - not '
                           f'shipped (v9:265-267; retried)')


def cutouts_written(run, plan, t0):
    """The photometry cutout cubes the subband searches of this cell wrote (phase3_snapshot_cutouts.write_cutouts:
    <root>/<sample>/<band>/<object>/<object>_<lst>_<utc>_<S>.fits), selected by name and by mtime >= t0."""
    root = cutout_root(run)
    out = []
    for p in glob.glob(os.path.join(root, '*', '*', '*', f'*_{plan["lst"]}_*_[IV].fits')):
        try:
            if os.path.getmtime(p) >= t0 - 1:
                out.append(p)
        except OSError:
            pass
    return sorted(out)


# ---------------------------------------------------------------------------------------------------------------------
# products, verify, ship
# ---------------------------------------------------------------------------------------------------------------------

def product_class(name):
    if name.startswith('runinfo_') and name.endswith('.json'):
        return 'snapshot_search'
    if name.startswith('nightrefused_') and name.endswith('.json'):
        return 'night_refusal'
    if name.startswith('wideband_outcome_') and name.endswith('.json'):
        return 'snapshot_wideband'
    if name.startswith('model_') and name.endswith('.fits'):
        return 'snapshot_model'
    if name.startswith('noise_') and name.endswith('_1min.csv'):
        return 'snapshot_1min'
    if name.startswith('injected_') and name.endswith('.csv'):
        return 'injection_book'
    if name.endswith('.csv'):
        return 'snapshot_table'
    return None                                  # logs, txt markers: shipped, not products


def products(d):
    out = []
    for n in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        cls = product_class(n)
        if cls and os.path.isfile(os.path.join(d, n)):
            out.append((os.path.join(d, n), cls))
    return out


def _same(a, b):
    if os.path.getsize(a) != os.path.getsize(b):
        return False
    with open(a, 'rb') as fa, open(b, 'rb') as fb:
        while True:
            x, y = fa.read(1 << 20), fb.read(1 << 20)
            if x != y:
                return False
            if not x:
                return True


def ship(u, src, dst, fresh, out_root):
    """cp -n of SHIP_PATTERNS (v9:336-337). An existing different PRODUCT file refuses the unit (no-clobber content test)
    unless --fresh (the replaced file is kept under <out>/_superseded/); an existing log or marker (no product class) is
    kept as cp -n keeps it (counted)."""
    s = cfg().search_snapshot
    os.makedirs(dst, exist_ok=True)
    names = sorted({os.path.basename(p) for pat in s.SHIP_PATTERNS for p in glob.glob(os.path.join(src, pat))})
    copied, same, clash, kept = [], [], [], []
    for n in names:
        a, b = os.path.join(src, n), os.path.join(dst, n)
        if os.path.exists(b):
            if _same(a, b):
                same.append(n)
                continue
            if product_class(n) is None and not fresh:
                kept.append(n)
                continue
            if not fresh:
                clash.append(n)
                continue
            keep = os.path.join(out_root, '_superseded', f'{os.path.basename(dst)}_{_h().utc_stamp()}')
            os.makedirs(keep, exist_ok=True)
            shutil.copy2(b, os.path.join(keep, n))
        tmp = f'{b}.tmp.{os.getpid()}'
        shutil.copy2(a, tmp)
        os.replace(tmp, b)
        copied.append(n)
    u.meta['shipped'] = dict(dir=dst, copied=len(copied), identical=len(same), clashes=clash, logs_kept=kept)
    u.check('ship.no_clobber', not clash, value=len(clash),
            detail=(f'existing different product files kept and the unit refused: {clash[:5]}' if clash else
                    f'{len(copied)} copied, {len(same)} identical already present, {len(kept)} earlier logs kept (cp -n)'))
    return not clash


def keep_failed_logs(run, name, out):
    s = cfg().search_snapshot
    if not s.KEEP_FAILED_LOGS or not os.path.isdir(out):
        return None
    dst = os.path.join(run.out_root, '_failed', f'{name}_{_h().utc_stamp()}')
    os.makedirs(dst, exist_ok=True)
    for p in glob.glob(os.path.join(out, '*.log')) + glob.glob(os.path.join(out, 'wideband_outcome_*.json')):
        shutil.copy2(p, dst)
    return dst


def _print_plan(plan, stages, out, ship_dir):
    s = cfg().search_snapshot
    print(f"\n[{plan['unit']}] snapshot cell {plan['cell']} (LST {plan['lst']}, night {plan['date']})"
          + (f" subbands {','.join(plan['subbands'])}" if len(plan['subbands']) < len(s.SUBBANDS) else ''))
    print(f"  stages: {', '.join(stages)}")
    print(f"  staged frames: {plan['stage']}  ({plan['n_staged']} frames; per subband {plan['staged'] or 'none'})")
    print(f"  excluded in data_decisions: subbands {sorted(plan['excluded']) or 'none'}; groups "
          f"{sorted(plan['excluded_groups']) or 'none'}")
    for name, ok, det in plan['gates']:
        print(f"  gate {'PASS' if ok else 'STOP'} {name}: {det}")
    if plan['refuse']:
        print(f"  -> REFUSED: {plan['refuse']}")
        return
    print(f"  subbands: {len(plan['subbands'])} x {engine_dir()}/phase3_snapshot_search.py --stokes VI --mode {s.MODE} "
          f"--thresh {s.THRESH_SIGMA:g} --nthreads {s.NTHREADS} (NPAR {s.NPAR}, site PYTHON[{s.ENGINE_PYTHON!r}]) -> {out}")
    print(f"  completeness: {2 * len([sb for sb in plan['subbands'] if plan['staged'].get(sb)])} subband x Stokes "
          f"searches must have runinfo or a night-refusal record")
    if 'widebands' in stages:
        print(f"  widebands: phase3_snapshot_group_search.py V then I; groups {', '.join(s.GROUPS)} accounted for")
    if 'ship' in stages:
        print(f"  ship -> {ship_dir} (no-clobber)")


def run_unit(args, run, item, cell, only_sb, stages, inject=None):
    H = _h()
    s = cfg().search_snapshot
    tag = None
    if inject:
        if inject.startswith('auto'):
            tag = getattr(args, 'inject_tag', None) or inject.replace(':', '')
        else:
            with open(os.path.abspath(inject), 'rb') as f:
                tag = getattr(args, 'inject_tag', None) or hashlib.sha1(f.read()).hexdigest()[:12]
    unit = item if not inject else f'{item}.inject.{tag}'
    plan = preflight(args, run, cell, only_sb)
    plan['unit'] = unit
    ship_dir = os.path.join(run.out_root, cell) if not inject else os.path.join(run.out_root, 'inject', tag, cell)
    bands = getattr(args, 'snap_bands', 'all') or 'all'
    pause = bool(getattr(args, 'snap_pause_1min', False) or s.PAUSE_1MIN)
    skip_groups = bands == 'subbands' or pause or bool(inject) or only_sb is not None or bool(getattr(args, 'scopes', None))
    if bands == 'groups':
        stages = [x for x in stages if x != 'subbands']
    if skip_groups:
        stages = [x for x in stages if x != 'widebands']
    if run.dry_run:
        try:
            out = work_dir(args, unit)
        except Exception:
            out = f'<node-local scratch>/{s.WORK_SUBDIR}/{unit}'
        _print_plan(plan, stages, out, ship_dir)
        if skip_groups and bands != 'groups':
            print('  widebands: skipped (' + ('--snap-bands subbands' if bands == 'subbands' else '1-minute pause' if pause
                                              else 'injection run: groups cannot be injected' if inject
                                              else 'a subset of subbands: the group search needs the full set') + ')')
        u = run.unit(unit, tier=TIER)
        u.status = 'refused' if plan['refuse'] else 'planned'
        u.reason = plan['refuse'] or ''
        return
    out = work_dir(args, unit)
    inputs = [p if isinstance(p, str) else p[0] for p in plan['inputs']]
    admitted = [plan['date']] if not all(plan['excluded'].get(sb) for sb in plan['subbands']) else []
    if run.resume and not plan['refuse']:
        ok, why = run.resume_check(unit, inputs, admitted)
        if ok:
            u = run.unit(unit, tier=TIER)
            u.status, u.reason = 'skipped-valid', why
            print(f'[{unit}] resume: {why}')
            return
    with run.unit(unit, tier=TIER, cell=cell, subband=only_sb, stage=plan['stage'], bands=bands, pause_1min=pause,
                  inject=inject) as u:
        u.meta.update(stages=stages, gates=plan['gates'], staged=plan['staged'], zero_vis=plan['zero_vis'],
                      excluded_subbands=sorted(plan['excluded']), excluded_groups=sorted(plan['excluded_groups']),
                      code=dict(clv15=H.git_head(engine_dir()), exopipe_main=H.git_head('/lustre/gh/main/exopipe')))
        for name, ok, det in plan['gates']:
            u.check(f'preflight:{name}', ok, detail=det)
        if plan['refuse']:
            u.refuse(plan['refuse'])
        u.set_admitted(admitted)
        for p in plan['inputs']:
            if isinstance(p, str):
                u.add_input(p)
            else:
                u.add_input(p[0], extra=p[1])
        H.wait_free(os.path.dirname(out), s.OUT_MIN_FREE_GB, 0, unit)
        env, pause = engine_env(run, args)
        u.meta['engine_env'] = {k: v for k, v in sorted(env.items()) if v is not None}
        # no lease: v9 runs one cell per node (one worker per node); the runner's lanes (--nodes host:1) bound it
        t0 = time.time()
        try:
            if inject:
                return _inject_run(args, run, u, plan, out, env, inject, tag, ship_dir, stages)
            if 'subbands' in stages:
                if os.path.isdir(out):
                    shutil.rmtree(out)                              # v9:89 'rm -rf $OUT' at the claim
                os.makedirs(out, exist_ok=True)
                print(f'[{unit}] subbands: {len(plan["subbands"])} searches, NPAR {s.NPAR} -> {out}', flush=True)
                outcomes = search_subbands(u, plan, out, env)
                if 'completeness' in stages:
                    subband_completeness(u, plan, out, outcomes)
            elif 'widebands' in stages:
                os.makedirs(out, exist_ok=True)
                adm = sorted(glob.glob(os.path.join(ship_dir, 'admission_*.csv')))
                if not adm and not glob.glob(os.path.join(out, 'admission_*.csv')):
                    u.refuse(f'the group search reads the subband admission of the cell (admission_<sb>.csv); none in '
                             f'{ship_dir} or {out} - search the subbands first')
                for p in adm:
                    if not os.path.exists(os.path.join(out, os.path.basename(p))):
                        shutil.copy2(p, out)
            if 'widebands' in stages:
                print(f'[{unit}] widebands: group search V then I' + (' (parallel)' if s.GROUPS_PARALLEL_STOKES else ''),
                      flush=True)
                search_widebands(u, plan, out, env, s.GROUPS_PARALLEL_STOKES)
                if 'completeness' in stages:
                    wideband_completeness(u, plan, out)
            if 'verify' in stages:
                for p, cls in products(out):
                    H.precheck(u, p, cls, label='verify')
                if u.failed_tests():
                    u.note(f'verify failed: nothing shipped; node-local output kept: {out}')
                    return
            if 'subbands' in stages and s.CUTOUTS_ON:
                n = H.record_outputs(u, cutouts_written(run, plan, t0), 'photometry_cutout')
                u.meta['photometry_cutouts'] = dict(root=cutout_root(run), n=n)
            dest = ship_dir if 'ship' in stages else out
            if 'ship' in stages:
                if not ship(u, out, ship_dir, run.fresh, run.out_root):
                    return
            _record(u, plan, dest, skip_groups or 'widebands' not in stages)
            if 'ship' in stages and not u.failed_tests():
                shutil.rmtree(out, ignore_errors=True)
        except RuntimeError:
            kept = keep_failed_logs(run, unit, out)
            if kept:
                u.note(f'logs of the incomplete cell kept at {kept}')
            shutil.rmtree(out, ignore_errors=True)                  # v9:228, :267 'rm -rf $OUT'
            raise


def _record(u, plan, dest, skip_groups):
    """Planned outputs = one runinfo per staged subband x Stokes (and per group x Stokes when the group search ran);
    each is written (snapshot_search) or refused by its named gate (night rule, exclusion, group outcome)."""
    H = _h()
    s = cfg().search_snapshot
    for p, cls in products(dest):
        H.record_output(u, p, cls)
    for sb in plan['subbands']:
        if not plan['staged'].get(sb):
            continue
        for st in ('V', 'I'):
            t = f'{sb}_{st}_{s.MODE}'
            ri = u.plan(os.path.join(dest, f'runinfo_{t}.json'), 'snapshot_search')
            if os.path.exists(ri):
                continue
            nr = os.path.join(dest, f'nightrefused_{t}.json')
            if os.path.exists(nr):
                u.refuse_item(ri, f'night rule: every frame of {sb} {st} refused (record {os.path.basename(nr)})',
                              gate='night-rule')
            elif sb in plan['excluded'] and s.EXCLUDED_STAGED_SUBBAND == 'accounted':
                u.refuse_item(ri, f'{sb} excluded in data_decisions.csv for {plan["lst"]} {plan["date"]}', gate='exclusion')
    if skip_groups:
        return
    for st in ('V', 'I'):
        p = os.path.join(dest, f'wideband_outcome_{st}.json')
        if not os.path.exists(p):
            continue
        with open(p) as f:
            groups = json.load(f).get('groups', {})
        for g in s.GROUPS:
            t = f'{g}_{st}_{s.MODE}'
            ri = u.plan(os.path.join(dest, f'runinfo_{t}.json'), 'snapshot_search')
            v = groups.get(g, 'missing')
            if v == 'searched' and os.path.exists(ri):
                continue
            if v in ('night-refused', 'excluded', 'no-members', 'too-few-frames'):
                u.refuse_item(ri, f'wideband {g} {st}: {v} (wideband_outcome_{st}.json)', gate=f'wideband-{v}')


def _inject_run(args, run, u, plan, out, env, inject, tag, ship_dir, stages):
    """Injection search of one cell (v9 battery hook, the search part): manifest, injected subband search, recovery."""
    H = _h()
    s = cfg().search_snapshot
    base = os.path.join(run.out_root, plan['cell'])
    os.makedirs(out, exist_ok=True)
    manifest = os.path.join(out, 'injection_manifest.csv')
    if inject.startswith('auto'):
        n = getattr(args, 'snap_inject_n', None) or (int(inject.split(':', 1)[1]) if ':' in inject else 1000)
        if not glob.glob(os.path.join(base, 'noise_*MHz_*_median.csv')):
            u.refuse(f'--inject auto draws injection S/N from the cell\'s baseline noise tables; none in {base} - ship the '
                     f'baseline search of {plan["cell"]} first')
        log = os.path.join(out, 'manifest.log')
        r = H.legacy_call(os.path.join(engine_dir(), 'phase3_snapshot_inject.py'),
                          ['--out', manifest, '--snr-from-cell', base, '--lst', plan['lst'], '--date', plan['date'],
                           '--n-per-stokes', str(n)], s.ENGINE_PYTHON, env, log, unit=u)
        if r.rc != 0 or not os.path.exists(manifest):
            raise RuntimeError(f'injection manifest failed (exit {r.rc}): {H.log_tail(log)}')
    else:
        shutil.copy2(os.path.abspath(inject), manifest)
    u.add_input(manifest)
    inj = os.path.join(out, 'injected')
    outcomes = search_subbands(u, plan, inj, env, inject=manifest)
    if 'completeness' in stages:
        subband_completeness(u, plan, inj, outcomes)
    log = os.path.join(out, 'recover.log')
    r = H.legacy_call(os.path.join(engine_dir(), 'phase3_snapshot_recover.py'),
                      ['--simple', '--inj-dir', inj, '--manifest', manifest, '--out', os.path.join(out, 'recovery_table.csv')],
                      s.ENGINE_PYTHON, env, log, unit=u)
    if r.rc != 0:
        raise RuntimeError(f'recovery table failed (exit {r.rc}): {H.log_tail(log)}')
    u.note('wideband groups are not injected (FIX-INJECTION-PATH-FOR-THE-1-MINUTE-TIER): no group search in this run')
    os.makedirs(ship_dir, exist_ok=True)
    for p in [manifest, os.path.join(out, 'recovery_table.csv')]:
        shutil.copy2(p, ship_dir)
    dst = os.path.join(ship_dir, 'injected')
    os.makedirs(dst, exist_ok=True)
    if not ship(u, inj, dst, True, run.out_root):
        return
    for p, cls in products(dst):
        H.record_output(u, p, cls)
    H.record_output(u, os.path.join(ship_dir, 'recovery_table.csv'), 'snapshot_table')
    shutil.rmtree(out, ignore_errors=True)


def run(args, run):
    """Tier snapshot: one unit per night cell <LST>_<night>[_<subband>] - see the module docstring."""
    H = _h()
    c = cfg()
    H.check_overrides('search_snapshot', TIER)
    s = c.search_snapshot
    H.production_guard(run.out_root, (c.paths.SNAPSHOT_FLEET_OUT,), run.dry_run)
    stages = H.stage_list(args, s.STAGES, optional=s.OPTIONAL_STAGES, tier=TIER)
    for st in stages:
        if st == 'fit':
            raise H.refusal('stage fit (snapshot): during the refactor window the per-cell prefit runs as catalog\'s '
                            'store/prefit stages (one writer of the fit store, spec K14); it becomes a search stage at cutover')
        if st == 'tier1min':
            raise H.refusal('stage tier1min (snapshot): the 1-minute re-run from kept 10-s products '
                            '(snapshot_frametime_rerun.sh) is not ported in this unit yet')
    items = expand_cells(args)
    inject = getattr(args, 'inject', None)
    print(f'search --tier snapshot: {len(items)} cell(s), stages {",".join(stages)}, out {run.out_root}'
          + (f', injection {inject}' if inject else ''), flush=True)
    for item, cell, sb in items:
        try:
            run_unit(args, run, item, cell, sb, stages, inject=inject)
        except Exception as e:                  # recorded as the unit's error (exit 1); the other cells still run
            print(f'[{item}] ERROR {type(e).__name__}: {e}', flush=True)
    if run.dry_run:
        return H.dry_exit(run, 'cell(s)')
    return None
