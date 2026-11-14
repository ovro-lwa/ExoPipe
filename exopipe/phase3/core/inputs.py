"""Inputs - the ONE resolver of Phase 1 snapshot frames (and, as steps move in, Phase 2 images) for every tier.

Phase 1 frames: ``phase1_frames(lst, date, subband)`` -> {'I': [(key, path)], 'V': [(key, path)]}, one frame per 10-s
INTEGRATION. A cell can hold several Phase 1 runs (re-images, patch runs). The pre-refactor discovery
(clv15/phase3_snapshot_search.discover_frames) keyed frames by their FILE-NAME timestamp with 'latest run wins'; two
runs of the same hour carry file-name timestamps 1-9 s apart, so it kept BOTH: 83 cell-bands in 41 cells mixed 2-3 runs
(12h 2025-01-10 55 MHz: 681 Stokes I frames for one hour; ledger IMPLEMENTED-DATA-DECISIONS-VIS50-20261012, FINDING;
handed to the refactor by Gregg 2026-10-13 ~00:35 UTC, "Hand it to refactor").

The rule here: frames are matched by their INTEGRATION (the integration midpoint from the header DATE-OBS, truncated to the
second - the label Phase 1 names frames by since exopipe 192c81c), and the LATEST run wins per integration; an older run
contributes only integrations the later runs lack (a patch run fills its gap and nothing else). Measured on 12h 2025-01-10
55 MHz: the three runs share 357-358 integration labels exactly (DATE-OBS agrees across runs); the union is 358, not 681.
Headers are read only where more than one run holds frames of the subband (elsewhere a file-name timestamp is already one
integration). ``phase1_frames`` refuses a result with more frames than integrations in the hour (content test). A kept
frame's KEY is its own file-name timestamp - the frame identity (R-FRAME-TIME-FROM-INTEGRATION-ALL-STAGES; the refactor chat
2026-10-18, production's port of this rule, clv15 4fe1121 _one_frame_per_integration): two integrations of different runs
under one file-name second keep the later run's frame, and the lost one is counted and listed in the report (ts_collisions,
lost). The keys, paths and order equal production's discover_frames (18h 2025-05-05 41 MHz, pinned and unpinned).

PHASE-1 RUN PIN (Gregg 2026-11-01, 18h 2025-05-05: "Pin to June run (Recommended)"; production clv15 facb3e8, pin file
exopipe_ops c6ddafc): a cell listed in the pin file (paths.PHASE1_RUN_PINS; columns cell, run, ruling, reason) resolves to its
pinned run ONLY - the run its served snapshot search used - while a later Phase-1 run of the hour that the snapshot tier did not
adopt exists (the pipeline account's Run_20261020_205211 of 18h 2025-05-05, made 2026-10-20/21: discovery preferred it and the
re-chain B4K2 refused). A pinned run that is not in the archive refuses; a cell without a pin is unchanged. 18h 2025-05-05
41 MHz Stokes I: 356 frames pinned (Run_20260603_010316, the search's own list), 358 unpinned (the October run).

The frame-time constant of the integration label is search_snapshot.INTEGRATION_S (10.03102041 s, the MS INTERVAL;
R-FRAME-TIME-FROM-INTEGRATION-ALL-STAGES), with production's own arithmetic (phase3_snapshot_search.dateobs_unix /
integration_label). instrument.INTEGRATION_S (Phase 2's LWA_DT, 10.0310 s) is 2.0e-5 s shorter: a midpoint within ~1e-5 s of a
second boundary got another label (FOLD_IN 10-28 finding).

FRAME ROOT (production clv15 phase3_snapshot_search.frame_root, 5c83ddc, 2026-11-10; Gregg 2026-11-08/10): a cell listed in
paths.SNAPSHOT_GPU_FRAME_CELLS (the Dec 18-27 2024 nights without the survey's own CLEANed Stokes I) takes its frames from the
observatory GPU pipeline's tree, paths.SNAPSHOT_GPU_FRAME_ROOT (the same <LST>/<date>/Run_*/<subband>/ layout); every other
cell from paths.PHASE1_ROOT (``frame_root``). Production's run-wide P3_SNAP_PHASE1_ROOT override has no counterpart here
(core/exclusions.refuse_frame_overrides refuses it).

PHASE-1 CODE VERSION (CUTOVER_READINESS 'Phase-1 provenance'; ledger FINDING-PHASE1-MNC-ANTENNA-FLAGS-FROM-WRONG-DATE-BEFORE-
ORCA-ACD5038-20261026: Phase 1 runs on Orca code older than acd5038 (2026-07-24) took the antenna flags of the LATEST MNC health
record, not the observation night's). Every Phase-1 run writes <run>/<subband>/provenance.json; its pipeline_version is the Orca
commit the run used. ``phase1_code`` reads it, and ``phase1_code_of(path)`` names the Phase-1 run(s) behind an input file: a
Phase-1 frame comes from its own run; a frame of the GPU pipeline's tree (FRAME ROOT) from its own run there, whose
provenance.json names the GPU pipeline ('pipeline') and its commit (pipeline_version); a Phase-2 product comes from the run
Phase 2 took each member subband from (the Science directory's log). core/record writes it into the input record of every unit (the input's entry and the unit's per-night
'phase1_code'), so the nights an affected Phase-1 version made can be listed (and re-flagged if Gregg rules so).

Frame file names (Phase 1): old era '<sb>-Pilot-Snapshot-<ts>-<I|V>-image.fits[.fs|.fz]' in snapshots/; new era
'<sb>-Pilot-Snapshot-<ts>-image.fits' = Stokes V in snapshots/, '<sb>-Clean-Snapshot-<ts>-image.fits' = Stokes I in
snapshots_clean/. CLEAN files named by interval index (t0000...) come from CLEAN runs that stopped partway and are not
frames (refactor finding D12): they are counted and reported, never used.
"""
import csv
import datetime
import glob
import hashlib
import io
import json
import math
import os
import re

from exopipe.phase3.core.record import Refusal

_PAT_OLD = re.compile(r'^(?P<sb>\d+MHz)-Pilot-Snapshot-(?P<ts>\d{8}_\d{6})-(?P<st>[IV])-image\.fits(?P<fz>\.f[sz])?$')
_PAT_NEW_V = re.compile(r'^(?P<sb>\d+MHz)-Pilot-Snapshot-(?P<ts>\d{8}_\d{6})-image\.fits(?P<fz>\.f[sz])?$')
_PAT_NEW_I = re.compile(r'^(?P<sb>\d+MHz)-Clean-Snapshot-(?P<ts>\d{8}_\d{6})-image\.fits(?P<fz>\.f[sz])?$')
_PAT_INDEX = re.compile(r'^(?P<sb>\d+MHz)-.*-t\d{4}-image\.fits(\.f[sz])?$')
_EPOCH = datetime.datetime(1970, 1, 1)
_DATEOBS_KEY = b'DATE-OBS='


def _cfg():
    from exopipe.phase3.config import CFG
    return CFG


def frame_root(lst, date):
    """The frame root of ONE cell (module docstring, FRAME ROOT): paths.SNAPSHOT_GPU_FRAME_ROOT for a cell of
    paths.SNAPSHOT_GPU_FRAME_CELLS, paths.PHASE1_ROOT for every other cell."""
    from exopipe.phase3.core import exclusions
    P = _cfg().paths
    return P.SNAPSHOT_GPU_FRAME_ROOT if (lst, date) in exclusions.gpu_frame_cells() else P.PHASE1_ROOT


def read_dateobs(path, nblk=8):
    """DATE-OBS of a FITS file (plain or tile-compressed), read from the first nblk 2880-byte blocks; '' if absent."""
    fd = os.open(path, os.O_RDONLY)
    try:
        buf = os.pread(fd, 2880 * nblk, 0)
    finally:
        os.close(fd)
    for i in range(0, len(buf) - 79, 80):
        card = buf[i:i + 80]
        if card.startswith(_DATEOBS_KEY):
            return card[10:].split(b'/')[0].strip().strip(b"'").strip().decode()
    if len(buf) == 2880 * nblk:
        from astropy.io import fits
        with fits.open(path, memmap=False) as hl:
            for hdu in hl:
                if 'DATE-OBS' in hdu.header:
                    return str(hdu.header['DATE-OBS']).strip()
    return ''


def frame_integration_s():
    """The frame-time constant, s: search_snapshot.INTEGRATION_S (= production phase3_snapshot_search.INTEGRATION_S, the MS
    INTERVAL of one integration; a protected mirror of the engine literal, tests/test_search-tiers.py)."""
    return float(_cfg().search_snapshot.INTEGRATION_S)


def dateobs_unix(date_obs):
    """FITS DATE-OBS 'YYYY-MM-DDTHH:MM:SS[.fff]' (UTC) -> unix seconds; ValueError when it is not one
    (= production phase3_snapshot_search.dateobs_unix, the same arithmetic)."""
    s = str(date_obs).strip()
    base = datetime.datetime.strptime(s[:19], '%Y-%m-%dT%H:%M:%S')
    return (base - _EPOCH).total_seconds() + (float('0' + s[19:]) if len(s) > 19 else 0.0)


def integration_label(date_obs):
    """'YYYYMMDD_HHMMSS' of the integration MIDPOINT (DATE-OBS = the integration start, + half the frame-time constant)
    truncated to the second (= production phase3_snapshot_search.integration_label); ValueError on an unreadable DATE-OBS."""
    mid = dateobs_unix(date_obs) + frame_integration_s() / 2.0
    return (_EPOCH + datetime.timedelta(seconds=int(math.floor(mid)))).strftime('%Y%m%d_%H%M%S')


def max_frames_per_hour():
    """Integrations in one hour + 2 (= production MAX_FRAMES_PER_HOUR): the most frames phase1_frames may return per
    Stokes (+2: frames straddling the hour's boundaries)."""
    return int(3600.0 / frame_integration_s()) + 2


def _dateobs_many(paths, nthreads=16):
    from concurrent.futures import ThreadPoolExecutor
    if not paths:
        return {}
    with ThreadPoolExecutor(max_workers=max(1, min(nthreads, len(paths)))) as ex:
        return dict(zip(paths, ex.map(read_dateobs, paths)))


def _scan_run(rd, subband):
    """{'I': {ts: path}, 'V': {ts: path}} of one run directory, plus the count of index-named CLEAN files."""
    out = {'I': {}, 'V': {}}
    n_index = 0
    for sub, hint in (('snapshots', None), ('snapshots_clean', 'I')):
        d = os.path.join(rd, subband, sub)
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            m = _PAT_OLD.match(fn)
            if m:
                out[m['st']][m['ts']] = os.path.join(d, fn)
                continue
            m = _PAT_NEW_I.match(fn)
            if m:
                out['I'][m['ts']] = os.path.join(d, fn)
                continue
            m = _PAT_NEW_V.match(fn)
            if m and hint != 'I':
                out['V'][m['ts']] = os.path.join(d, fn)
                continue
            if _PAT_INDEX.match(fn):
                n_index += 1
    return out, n_index


# ---- Phase-1 run pins --------------------------------------------------------------------------------------------------
_PIN_CACHE = {}          # pin file -> ((size, mtime_ns), {cell: run}, sha1 of the file)


def phase1_run_pins(path=None):
    """({cell '<LST>_<date>': pinned run}, sha1 of the file) of the Phase-1 run-pin file (default paths.PHASE1_RUN_PINS).
    The file production's phase3_snapshot_search.phase1_run_pin reads: columns cell, run, ruling, reason; a later row of a
    cell wins; an empty run is no pin. No file: ({}, None). Read again whenever the file changes (size, mtime; production
    reads it once per process). A file without the cell and run columns refuses."""
    p = path or _cfg().paths.PHASE1_RUN_PINS
    try:
        st = os.stat(p)
    except FileNotFoundError:
        return {}, None
    key = (st.st_size, st.st_mtime_ns)
    hit = _PIN_CACHE.get(p)
    if hit and hit[0] == key:
        return dict(hit[1]), hit[2]
    with open(p, 'rb') as f:
        raw = f.read()
    rd = csv.DictReader(io.StringIO(raw.decode('utf-8'), newline=''))
    if not {'cell', 'run'} <= set(rd.fieldnames or ()):
        raise Refusal(f'Phase-1 run-pin file {p}: columns {rd.fieldnames} - no cell / run column (production '
                      f'phase3_snapshot_search.phase1_run_pin reads cell, run)')
    pins = {}
    for r in rd:
        pins[(r.get('cell') or '').strip()] = (r.get('run') or '').strip()
    _PIN_CACHE[p] = (key, pins, hashlib.sha1(raw).hexdigest())
    return dict(pins), _PIN_CACHE[p][2]


def phase1_run_pin(lst, date, path=None):
    """The pinned Phase-1 run (directory name) of the cell <lst>_<date>, or None (no pin)."""
    return phase1_run_pins(path)[0].get(f'{lst}_{date}') or None


# ---- Phase-1 code version ---------------------------------------------------------------------------------------------
PROVENANCE_NAME = 'provenance.json'          # Phase 1 writes one per run and subband: <run>/<subband>/provenance.json
PHASE1_CODE_KEYS = ('pipeline_version', 'pipeline_branch', 'package_version', 'started_at')   # pipeline_version = the Orca commit
_CODE_CACHE = {}         # provenance file -> ((size, mtime_ns), code)
_SB_DIR = re.compile(r'^\d+MHz$')


def phase1_code(run_dir, subband):
    """The Phase-1 code of one run's subband: {'run', 'subband', 'pipeline_version' (the Orca commit the run used),
    'pipeline_branch', 'package_version', 'started_at', 'provenance' (the file)} from <run_dir>/<subband>/provenance.json.
    'pipeline' is added when the file names one (the GPU pipeline's runs: 'gpu-subband-imaging', pipeline_version then its
    commit, not Orca's). An absent or unreadable file gives 'pipeline_version' None and 'missing' (why): recorded as such,
    never guessed. Cached per file (size, mtime)."""
    p = os.path.join(run_dir, subband, PROVENANCE_NAME)
    base = {'run': os.path.basename(run_dir.rstrip('/')), 'subband': subband, 'provenance': p}
    try:
        st = os.stat(p)
    except OSError as e:
        return dict(base, pipeline_version=None, missing=f'no {PROVENANCE_NAME} ({e.strerror})')
    key = (st.st_size, st.st_mtime_ns)
    hit = _CODE_CACHE.get(p)
    if hit and hit[0] == key:
        return dict(hit[1])
    try:
        with open(p) as f:
            prov = json.load(f)
        code = dict(base, **{k: prov.get(k) for k in PHASE1_CODE_KEYS})
        if prov.get('pipeline'):
            code['pipeline'] = prov['pipeline']
        if not code['pipeline_version']:
            code['missing'] = f'{PROVENANCE_NAME} names no pipeline_version'
    except (OSError, ValueError, AttributeError) as e:
        code = dict(base, pipeline_version=None, missing=f'{PROVENANCE_NAME} unreadable ({type(e).__name__}: {e})')
    _CODE_CACHE[p] = (key, code)
    return dict(code)


_P2_RUN_LINE = re.compile(r'\s(?P<sb>\d+MHz)\s+(?P<run>Run_\d{8}_\d{6})\s+\(\d+ FITS files\)\s*$')
_P2_RUNS_CACHE = {}      # Science directory -> ((log, size, mtime_ns), {subband: Phase-1 run})


def phase2_phase1_runs(sci_dir):
    """{subband: Phase-1 run} Phase 2 made the Science directory's images from: the '<sb>  Run_* (<n> FITS files)' lines of
    its log phase2_<LST>_<date>_<Science_*>.log (exopipe phase2_controller.discover_subbands - the latest Phase-1 run holding
    the subband when Phase 2 ran; a later line of a subband wins). {} when the directory has no such log."""
    hit = _P2_RUNS_CACHE.get(sci_dir)
    if hit:
        try:
            st = os.stat(hit[0][0])
            if (st.st_size, st.st_mtime_ns) == hit[0][1:]:
                return dict(hit[1])
        except OSError:
            pass
    logs = sorted(glob.glob(os.path.join(sci_dir, f'phase2_*_{os.path.basename(sci_dir.rstrip("/"))}.log')))
    if not logs:
        return {}
    log = logs[-1]
    st = os.stat(log)
    key = (log, st.st_size, st.st_mtime_ns)
    runs = {}
    with open(log, errors='replace') as f:
        for line in f:
            m = _P2_RUN_LINE.search(line)
            if m:
                runs[m['sb']] = m['run']
    _P2_RUNS_CACHE[sci_dir] = (key, runs)
    return dict(runs)


def _scope_subbands(scope):
    C = _cfg()
    if _SB_DIR.match(scope or ''):
        return [scope]
    if scope == 'Full':
        return list(C.instrument.ALL_SUBBANDS)
    return list(C.instrument.SUBBAND_BANDS.get(scope, ()))


def phase1_source(path):
    """('phase1' | 'gpu' | 'phase2', LST, date, run or Science directory name, subband or scope) of a file under
    paths.PHASE1_ROOT or paths.SNAPSHOT_GPU_FRAME_ROOT (<LST>/<date>/Run_*/<subband>/..., the same layout; FRAME ROOT) or
    paths.PHASE2_ROOT (<LST>/<date>/Science_*/<scope>/...), else None. The path is compared as given (os.path.abspath, no
    file-system call), so a staged copy never passes for its source."""
    C = _cfg()
    a = os.path.abspath(str(path))
    for root, kind in ((C.paths.PHASE1_ROOT, 'phase1'), (C.paths.SNAPSHOT_GPU_FRAME_ROOT, 'gpu'),
                       (C.paths.PHASE2_ROOT, 'phase2')):
        r = os.path.abspath(root).rstrip('/') + '/'
        if not a.startswith(r):
            continue
        parts = a[len(r):].split('/')
        if len(parts) < 5:                      # <LST>/<date>/<run>/<scope>/<file ...>
            return None
        lst, date, run, scope = parts[:4]
        if kind in ('phase1', 'gpu') and run.startswith('Run_') and _SB_DIR.match(scope):
            return kind, lst, date, run, scope
        if kind == 'phase2' and run.startswith('Science_') and _scope_subbands(scope):
            return kind, lst, date, run, scope
        return None
    return None


def phase1_code_of(path):
    """[(night '<LST>_<date>', subband, code)] of the Phase-1 run(s) behind the input file ``path`` (code = phase1_code):
    - under paths.PHASE1_ROOT or paths.SNAPSHOT_GPU_FRAME_ROOT, <LST>/<date>/Run_*/<subband>/...: that run's subband;
    - under paths.PHASE2_ROOT, <LST>/<date>/Science_*/<scope>/...: every member subband of the scope (the subband itself, a
      band's subbands, all for Full), each with the Phase-1 run the Science directory's log names for it
      (phase2_phase1_runs); a subband the log does not name: {'run': None, 'pipeline_version': None, 'missing': why};
    - anything else (a staged copy, a Phase 3 product): [] (phase1_source)."""
    src = phase1_source(path)
    if src is None:
        return []
    C = _cfg()
    kind, lst, date, run, scope = src
    night = f'{lst}_{date}'
    if kind in ('phase1', 'gpu'):
        root = C.paths.PHASE1_ROOT if kind == 'phase1' else C.paths.SNAPSHOT_GPU_FRAME_ROOT
        return [(night, scope, phase1_code(os.path.join(root, lst, date, run), scope))]
    try:
        runs, why = phase2_phase1_runs(os.path.join(C.paths.PHASE2_ROOT, lst, date, run)), 'names no Phase-1 run for'
    except OSError as e:               # recorded, never a reason to fail the unit that reads the image
        runs, why = {}, f'could not be read ({e.strerror}) for'
    out = []
    for sb in _scope_subbands(scope):
        if runs.get(sb):
            out.append((night, sb, phase1_code(os.path.join(C.paths.PHASE1_ROOT, lst, date, runs[sb]), sb)))
        else:
            out.append((night, sb, {'run': None, 'subband': sb, 'pipeline_version': None,
                                    'missing': f'the Phase-2 log of {run} {why} {sb}'}))
    return out


def phase1_frames(lst, date, subband, phase1_root=None, report=None, run_pins=None):
    """One frame per integration per Stokes for (LST, date, subband): {'I': [(key, path)], 'V': [(key, path)]}, sorted.

    key = the frame's file-name timestamp (its identity; see the module docstring). A cell listed in the Phase-1 run-pin
    file (``run_pins``, default paths.PHASE1_RUN_PINS) resolves to its pinned run only and refuses when that run is absent.
    ``report`` (a dict, optional) receives: runs (run -> frame counts, of the runs considered), mixed (bool), dropped (frames
    of older runs superseded per integration), integrations, disagree (the runs' integration sets differ), ts_collisions and
    lost (integrations lost to a file-name second another run's frame holds), shadowed, rule, index_named (CLEAN files named
    by interval index, not used), pin ({'run', 'file', 'sha1'} of the cell's pin, or None), phase1_code ({run: phase1_code}
    of every run a returned frame comes from)."""
    root = phase1_root or frame_root(lst, date)
    rundirs = sorted(glob.glob(os.path.join(root, lst, date, 'Run_*')))
    if not rundirs:
        raise Refusal(f'no Phase 1 runs for {lst} {date} under {root}')
    pin_file = run_pins or _cfg().paths.PHASE1_RUN_PINS
    pins, pin_sha1 = phase1_run_pins(pin_file)
    pin = pins.get(f'{lst}_{date}') or None
    if pin:
        rundirs = [r for r in rundirs if os.path.basename(r) == pin]
        if not rundirs:
            raise Refusal(f'{lst} {date} is pinned to Phase-1 run {pin} ({pin_file}), which is not under '
                          f'{os.path.join(root, lst, date)} - REFUSED (the run pin, Gregg 2026-11-01; production clv15 '
                          f'facb3e8)')
    per_run = []
    n_index = 0
    for rd in rundirs:
        fr, ni = _scan_run(rd, subband)
        n_index += ni
        if fr['I'] or fr['V']:
            per_run.append((os.path.basename(rd), fr))
    rep = {'runs': {r: {'I': len(f['I']), 'V': len(f['V'])} for r, f in per_run}, 'index_named': n_index,
           'mixed': False, 'dropped': {'I': 0, 'V': 0}, 'integrations': {'I': 0, 'V': 0},
           'disagree': {'I': False, 'V': False}, 'ts_collisions': {'I': 0, 'V': 0}, 'lost': {'I': [], 'V': []},
           'shadowed': {'I': 0, 'V': 0}, 'rule': 'one frame per integration (header DATE-OBS), latest Phase-1 run wins',
           'pin': {'run': pin, 'file': pin_file, 'sha1': pin_sha1} if pin else None}
    run_of = {p: r for r, f in per_run for st in f for p in f[st].values()}
    out = {}
    for st in ('I', 'V'):
        contributing = [(r, f[st]) for r, f in per_run if f[st]]
        if len(contributing) <= 1:
            out[st] = dict(contributing[0][1]) if contributing else {}
            continue
        rep['mixed'] = True
        dob = _dateobs_many([p for _, f in contributing for p in f.values()])
        chosen, sets = {}, []          # integration label -> (ts, path); per run the set of its integrations
        # later runs first: the latest run wins per integration; older runs only fill integrations still missing. Frames of
        # one run in file-name order, so two files of one integration in one run keep the earlier name.
        for r, frames in reversed(contributing):
            own = set()
            for ts, p in sorted(frames.items()):
                d = dob.get(p, '')
                try:
                    key = integration_label(d)
                except (ValueError, TypeError):
                    raise Refusal(f'{p}: no readable DATE-OBS ({d!r}) - cannot assign the frame to an integration, and '
                                  f'{lst} {date} {subband} holds several Phase-1 runs')
                own.add(key)
                if key in chosen:
                    rep['dropped'][st] += 1
                else:
                    chosen[key] = (ts, p)
            sets.append(own)
        rep['integrations'][st] = len(chosen)
        rep['disagree'][st] = any(x != sets[0] for x in sets[1:])
        kept = {}
        for key, (ts, p) in chosen.items():            # insertion order: the latest run's frames first
            if ts in kept:      # two integrations under one file-name second: the later run's frame is kept, the lost one
                rep['ts_collisions'][st] += 1           # counted and listed (frame identity = the file-name second)
                rep['lost'][st].append({'label': key, 'date_obs': dob.get(p, ''), 'file_ts': ts, 'run': run_of[p]})
                continue
            kept[ts] = p
        # a kept older-run frame whose FILE NAME a later run also uses (for another integration): a staging keyed by file
        # name may hold the later run's file under that name - counted for review
        later = set()
        for r, frames in reversed(contributing):
            for ts, p in frames.items():
                if kept.get(ts) == p and os.path.basename(p) in later:
                    rep['shadowed'][st] += 1
            later.update(os.path.basename(p) for p in frames.values())
        out[st] = kept
    out = {st: sorted(out[st].items()) for st in ('I', 'V')}
    # content test: never more frames than integrations in the hour (+2 for frames straddling the boundaries)
    max_frames = max_frames_per_hour()
    for st in ('I', 'V'):
        if len(out[st]) > max_frames:
            raise Refusal(f'{lst} {date} {subband} Stokes {st}: {len(out[st])} frames for one hour > {max_frames} '
                          f'integrations - frames from several runs are still mixed')
    used = sorted({run_of[p] for st in ('I', 'V') for _, p in out[st]})
    rep['phase1_code'] = {r: phase1_code(os.path.join(root, lst, date, r), subband) for r in used}
    if report is not None:
        report.update(rep)
    return out


# =====================================================================================================================
# Phase 2 deep images of one night (B-03 (a), Gregg 2026-10-13; the same-state reading, 2026-10-14)
# =====================================================================================================================
# The ONE copy of the co-add's five image finders (lib/masters/coadd.py = production coadd_full_pipeline.py):
#   find_subband_alignment_image, find_wideband_alignment_image  -> survey_image   (I Taper Robust-0.75: the flow is measured on it)
#   find_subband_variant_image, find_wideband_deep_images        -> variant_image  (the I variants)
#   find_stokes_v_deep_image                                     -> stokes_v_image (V Taper Robust-0)
# Each chooses as its finder does: the first match in the sorted directory listing, the dewarped file preferred and the raw
# file the fallback (the co-add's "failsafe"). FINDER_PINS are the sha1 of each finder's source text (from its 'def' to the
# next top-level statement) this copy was made from; lib/masters/input_resolver refuses to stand in for a finder whose text
# changed (a fold-in of a production edit needs a re-port here first).
FINDER_PINS = {'find_subband_alignment_image': '538ca2958ddf70966cac8253dea9f34572577d6e',
               'find_wideband_alignment_image': '2f1081e46c0010d2de1d08cc895b2093b1a2ada3',
               'find_subband_variant_image': 'fd19d7ffbe0a6556ac77908cc7004fa8392478f8',
               'find_wideband_deep_images': '2b6f4f885cfcfd1ea2a89ce89c6121f29ab10136',
               'find_stokes_v_deep_image': '2ea538378dddfc8ce4bfb2e0e314925c1a21d31f',
               'is_subband': 'd55ee27938d347484d6c7508dad692d619b86b8e',
               'find_latest_science_dir': 'b5cea611a115359a2a15c1fcc83e78d155c5f0c8'}
VARIANTS = ('Taper_R0', 'Taper_R075', 'NoTaper_R0', 'NoTaper_R075')


def source_block(text, name):
    """The source text of the top-level function ``name`` in a module's text: its 'def' line to the next top-level statement
    (the unit FINDER_PINS hash); None when absent."""
    m = re.search(r'^def ' + re.escape(name) + r'\(.*?(?=^\S)', text, re.S | re.M)
    return m.group(0) if m else None


# every top-level find_* function of the co-add when the copy was made: a new one (a new finder at a new call site) is a
# change the copy does not cover
FINDER_SET = ('find_10min_images', 'find_all_dewarped_variants', 'find_latest_science_dir', 'find_stokes_v_deep_image',
              'find_subband_alignment_image', 'find_subband_variant_image', 'find_wideband_alignment_image',
              'find_wideband_deep_images')


def pin_mismatches(module_text):
    """What changed in ``module_text`` (the co-add module) since the copy was made: each pinned function whose source text
    differs from FINDER_PINS (absent = differs) or that now carries a decorator, and a find_* function that was not there."""
    import hashlib
    out = []
    for name, want in FINDER_PINS.items():
        blk = source_block(module_text, name)
        if blk is None or hashlib.sha1(blk.encode()).hexdigest() != want:
            out.append(name)
        elif re.search(r'^@[^\n]*\n(?=def ' + re.escape(name) + r'\()', module_text, re.M):   # a decorator line right above
            out.append(f'{name} (decorated)')
    now = sorted(set(re.findall(r'^def (find_\w+)\(', module_text, re.M)))
    out += [f'{n} (a new finder)' for n in now if n not in FINDER_SET]
    return out


def is_subband(freq):
    """= coadd.is_subband: a subband given by its MHz number (int or str), one of the instrument's subbands."""
    try:
        return int(freq) in {int(str(s).replace('MHz', '')) for s in _cfg().instrument.ALL_SUBBANDS}
    except (ValueError, TypeError):
        return False


def latest_science_dir(lst, date, phase2_root=None):
    """= coadd.find_latest_science_dir: the night's latest Phase 2 Science_* directory, or None."""
    dirs = sorted(glob.glob(f'{phase2_root or _cfg().paths.PHASE2_ROOT}/{lst}/{date}/Science_*'))
    return dirs[-1] if dirs else None


def _listing(d):
    return sorted(os.listdir(d)) if os.path.isdir(d) else None


def survey_subband(sci_dir, freq_mhz):
    """= coadd.find_subband_alignment_image (without its print): (path or None, raw) of a subband night's I survey image,
    Taper Robust-0.75, not pbcorr; raw: no dewarped file, the raw one is used (the co-add's '[failsafe]')."""
    d = f'{sci_dir}/{freq_mhz}MHz/I/deep'
    files = _listing(d)
    if files is None:
        return None, False

    def base(f):
        return 'Taper' in f and 'NoTaper' not in f and 'Robust-0.75' in f and 'pbcorr' not in f and f.endswith('.fits')
    for f in files:
        if base(f) and f.endswith('_dewarped.fits'):
            return os.path.join(d, f), False
    for f in files:
        if base(f) and 'dewarped' not in f and f.endswith('-image.fits'):
            return os.path.join(d, f), True
    return None, False


def survey_wideband(sci_dir, band):
    """= coadd.find_wideband_alignment_image (without its print): (path or None, raw) of a band night's I survey image."""
    d = f'{sci_dir}/{band}/I/deep'
    files = _listing(d)
    if files is None:
        return None, False

    def base(f):
        return 'Taper' in f and 'NoTaper' not in f and 'Robust-0.75' in f and 'pbcorr' not in f and f.endswith('.fits')
    for f in files:
        if base(f) and 'dewarped' in f:
            return os.path.join(d, f), False
    for f in files:
        if base(f) and 'dewarped' not in f:
            return os.path.join(d, f), True
    return None, False


def variant_subband(sci_dir, freq_mhz, variant):
    """= coadd.find_subband_variant_image: a subband night's I variant image ('Taper_R0', 'NoTaper_R0', 'NoTaper_R075'),
    not pbcorr, dewarped preferred."""
    taper, rob = variant.split('_')
    rtok = 'Robust-0.75' if rob == 'R075' else 'Robust-0'
    d = f'{sci_dir}/{freq_mhz}MHz/I/deep'
    files = _listing(d)
    if files is None:
        return None

    def base(f):
        if 'pbcorr' in f or not f.endswith('.fits'):
            return False
        if taper == 'Taper' and 'NoTaper' in f:
            return False
        if taper == 'NoTaper' and 'NoTaper' not in f:
            return False
        if rtok == 'Robust-0':
            return 'Robust-0-' in f
        return 'Robust-0.75' in f
    for f in files:
        if base(f) and f.endswith('_dewarped.fits'):
            return os.path.join(d, f)
    for f in files:
        if base(f) and 'dewarped' not in f and f.endswith('-image.fits'):
            return os.path.join(d, f)
    return None


def variant_wideband(sci_dir, band, stokes, variant):
    """= coadd.find_wideband_deep_images: a band night's deep image of ``variant`` ('Taper_R0', 'Taper_R075', 'NoTaper_R0',
    'NoTaper_R075') and Stokes, not pbcorr, dewarped preferred."""
    d = f'{sci_dir}/{band}/{stokes}/deep'
    files = _listing(d)
    if files is None:
        return None
    taper_key = 'NoTaper' if 'NoTaper' in variant else 'Taper'
    robust_key = 'Robust-0.75' if 'R075' in variant else 'Robust-0'

    def matches(f):
        if not f.endswith('.fits') or 'pbcorr' in f:
            return False
        if taper_key == 'NoTaper' and 'NoTaper' not in f:
            return False
        if taper_key == 'Taper' and 'NoTaper' in f:
            return False
        if robust_key == 'Robust-0.75' and 'Robust-0.75' not in f:
            return False
        if robust_key == 'Robust-0' and 'Robust-0.75' in f:
            return False
        return robust_key in f and taper_key in f
    for f in files:
        if matches(f) and 'dewarped' in f:
            return os.path.join(d, f)
    for f in files:
        if matches(f) and 'dewarped' not in f:
            return os.path.join(d, f)
    return None


def stokes_v_image(sci_dir, freq_str):
    """= coadd.find_stokes_v_deep_image: the night's V deep image, Taper Robust-0, not pbcorr, dewarped preferred."""
    d = f'{sci_dir}/{freq_str}MHz/V/deep' if is_subband(freq_str) else f'{sci_dir}/{freq_str}/V/deep'
    files = _listing(d)
    if files is None:
        return None

    def base(f):
        return ('Taper' in f and 'NoTaper' not in f and 'Robust-0' in f and 'Robust-0.75' not in f and 'pbcorr' not in f
                and f.endswith('.fits'))
    for f in files:
        if base(f) and 'dewarped' in f:
            return os.path.join(d, f)
    for f in files:
        if base(f) and 'dewarped' not in f:
            return os.path.join(d, f)
    return None


def survey_image(sci_dir, scope):
    """(path or None, raw) of the night's I survey image - the image its co-add flow is measured on, and so the alignment of
    every product of the night. scope: a subband's MHz number (int or str) or a band name, as the co-add passes it."""
    return survey_subband(sci_dir, scope) if is_subband(scope) else survey_wideband(sci_dir, scope)


def product_image(sci_dir, scope, product):
    """The night's image of ``product``: 'V' (V Taper Robust-0) or an I variant ('Taper_R0', 'Taper_R075', 'NoTaper_R0',
    'NoTaper_R075'), as the co-add's finder for it chooses."""
    if product == 'V':
        return stokes_v_image(sci_dir, scope)
    if product not in VARIANTS:
        raise ValueError(f'unknown Phase 2 product {product!r} (V or one of {VARIANTS})')
    if is_subband(scope):
        return variant_subband(sci_dir, scope, product)
    return variant_wideband(sci_dir, scope, 'I', product)


def dewarped(path):
    """The dewarp state of a chosen Phase 2 image: 'dewarped' in its file name - the test every finder applies to choose it."""
    return path is not None and 'dewarped' in os.path.basename(path)


def state_mismatch(sci_dir, scope, path, product):
    """B-03 same-state (Gregg 2026-10-14): None when ``path`` - the night's V or I-variant image - is in the same dewarp state
    as the night's survey image, whose flow aligns every product of the night; else why it is refused. A raw product on a
    dewarped survey image keeps its ionospheric warp (the survey's flow no longer carries it); a dewarped product on a raw
    survey image gets the raw flow's warp correction a second time. Both raw is kept: the flow was measured on that night's
    raw survey image, which carries the same warp."""
    if path is None:
        return None
    surv, _raw = survey_image(sci_dir, scope)
    if surv is None:
        return f'{product}: no survey image of the night to take the flow from'
    if dewarped(path) == dewarped(surv):
        return None
    if dewarped(surv):
        return f'raw {product} on a dewarped survey image (its warp would stay: the flow is measured on the dewarped image)'
    return f'dewarped {product} on a raw survey image (the raw image\'s flow would apply its warp correction again)'
