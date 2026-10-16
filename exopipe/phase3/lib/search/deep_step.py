"""Deep tier of `exopipe3 search` (called by steps/search.py: add_arguments, default_out, run; the --help text lives there).

One unit per (deep cell, Stokes): <cell>_<S> in bulk, <cell>_<S>_<date>[+n] in per-hour / incremental mode, and
<cell>_<S>_replay_<attempt> for a read-only replay of an existing attempt. Stages (config search.STAGES):

  preflight   inputs present and current, gates P1-P9 / G2-G4 of the spec (masters and COADDOP, membership, gate records,
              search-tier table, pinned-input md5s, engine literals = config, flow-cache currency, disk); --dry-run = this only
  quarantine  the cell's earlier archive search frames: archive's job (decision 6, group 10). Production mode runs
              exopipe_ops/p3s5_quarantine_frames.py in place with the fleet's environment (P3_COADD_ROOT unset, P3S5_QROUND);
              while paths.PRODUCTION_WRITES is False it only writes the plan <out>/quarantine_plan/<unit>.csv
  detect      lib/search/deep_search.run_deep_cell IN PLACE in a fresh child (core.legacy.run, module mode: inherited P3_*
              cleared, the engine environment built from config, parity guard, child code fingerprint), cwd = the staged
              attempt, log = <attempt>/<cell>.log (incremental: <cell>.inc.log appended to the base log). The child maps the
              outcome to an exit code: 0 done, 2 refused (RuntimeError carrying search.REFUSAL_MARKERS), 1 error
  fit         core.fitting.detection_position on the SEARCHED frame of every catalogue row (memory-mapped float32 frame, LST at
              MJD-MID; value and rms from core.fitting.forced_value_rms) -> <attempt>/fit_positions.csv + summary with the
              peak-fallback fraction per frame (deep_fit_positions.py computed in search; master_catalog.csv is NOT rewritten,
              search_fit.APPLY_AT = 'catalog-fitpos' in the window)
  verify      content tests of every output on the staged attempt + the stage verifier (completion from the child's result,
              parent catalogue present, refusal ledgers present when the log reports refusals, unaccounted epochs); a failure
              refuses the unit BEFORE anything is published
  publish     copy the attempt to <out>/attempts/<cell>_<S>_<UTC>/, the parent catalogue to <out>/parent_catalogs/, this
              attempt's v110 verdicts to <out>/v110_products/<cell>/, and append "cell,host,attempt" to <out>/published_<S>.csv
              (the layout today's merge reads, p3s3_sources.py); archive copies were written by detect (window:
              <out>/archive_mirror)

Ported launcher logic (p3s5_search_fleet.sh, p3s4_incremental_fleet.sh, p3s5_launch.sh): readiness (the inputs are present and
current - the install record check is the install step's chain), V before I per cell, disk gate (a node condition: error, never
a refusal), flow-cache currency / drop / partial rules by host capability (site GPU_HOSTS), adoption replaced by --resume (record
fingerprints and the admitted night set), quarantine before a FRESH attempt only, outcome from the child's exit code (no log
grep), mandatory parent catalogue before publish, publish layout and register row, hygiene once both Stokes are published.
Dropped by design: input shipping and flow-holder fetches (a chain stays on the node that holds the cell, DESIGN 11.3).
"""
import contextlib
import csv
import fcntl
import glob
import hashlib
import json
import os
import re
import shutil
import sys
import time

TIER = 'deep'
_ATT_RE = re.compile(r'^(?P<cell>\d\dh_(?:\d+MHz|Red|Green|Blue|Full))_(?P<s>[IV])_(?P<att>\d{8}T\d{6})$')
_CHILD_MODULE = 'exopipe.phase3.lib.search.deep_step'
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
# WAVE2 B-30: the values of search_detect / search_sublst.EPHEMERIS_TIME (deep_search._eph_time reads them in the engine)
EPHEMERIS_TIMES = ('frame-mid', 'date-obs')


# The engine child's TMPDIR: PyBDSF's multiprocessing Manager listens on an AF_UNIX socket at
# <TMPDIR>/pymp-XXXXXXXX/listener-XXXXXXXX, and Linux caps a socket path at 107 bytes - a TMPDIR under a deep work root failed
# with 'AF_UNIX path too long' -> EOFError in the engine (P4 parity run, 2026-10-14). Production sets no TMPDIR (/tmp).
_AF_UNIX_MAX = 107
_MP_SOCKET_TAIL = len('/pymp-XXXXXXXX/listener-XXXXXXXX')


def engine_tmpdir(root, tag):
    """A short per-unit TMPDIR for an engine child under ``root`` (search_run / search_sublst.ENGINE_TMP_ROOT, default /tmp
    as production); refuses a root that leaves no room for multiprocessing's sockets. The caller removes it after the child."""
    import tempfile
    from exopipe.phase3.core.record import Refusal
    os.makedirs(root, exist_ok=True)
    d = tempfile.mkdtemp(prefix=f'e3_{tag}_', dir=root)
    if len(d) + _MP_SOCKET_TAIL > _AF_UNIX_MAX:
        shutil.rmtree(d, ignore_errors=True)
        raise Refusal(f'engine TMPDIR {d} under {root} leaves no room for multiprocessing\'s AF_UNIX sockets (a path is at '
                      f'most {_AF_UNIX_MAX} bytes): set a shorter ENGINE_TMP_ROOT')
    return d


def ephemeris_time(value, key):
    """EXOPIPE3_EPHEMERIS_TIME for an engine child: ``value`` of config key ``key``; an unknown value refuses here, before
    any child starts (the engine refuses it too)."""
    if value not in EPHEMERIS_TIMES:
        from exopipe.phase3.core.record import Refusal
        raise Refusal(f'{key} = {value!r} is not one of {EPHEMERIS_TIMES} - REFUSING')
    return value


def _cfg():
    from exopipe.phase3.config import CFG
    return CFG


def _host():
    from exopipe.phase3.core import env as E
    return E.host()


def _utc_tag(t=None):
    return time.strftime('%Y%m%dT%H%M%S', time.gmtime(t))


# =====================================================================================================================
# flags and default root
# =====================================================================================================================

def add_arguments(p):
    g = p.add_argument_group('search --tier deep')
    g.add_argument('--deep-flow-from', metavar='DIR',
                   help="seed a cell's flow cache from DIR/<cell>/ (READ ONLY: ep_*_flow.npy, flow_generation.json, "
                        "master_up2.fits) when this run's flow root holds none; its currency is checked with "
                        "p3s3_cache_current.py and a stale seed refuses. For parity on a node without a GPU")
    g.add_argument('--deep-keep-work', action='store_true',
                   help='keep the staged attempts and flow work dirs (search_run.HYGIENE otherwise removes them once both '
                        'Stokes of a cell are published in this run)')


def default_out(args):
    """Production product root of the deep search (today's collect root). Refused for writes while paths.PRODUCTION_WRITES is
    False (gate G3): pass --out."""
    return _cfg().search_run.COLLECT_ROOT.format(host=_host())


# =====================================================================================================================
# items
# =====================================================================================================================

class Item:
    """One unit: a deep cell and a Stokes, in mode 'bulk' (whole cell), 'incremental' (gates on the whole cell, only `dates`
    built and searched, appended to a copy of the base attempt) or 'replay' (fit/verify of an existing attempt, read only)."""

    def __init__(self, cell, stokes, mode, dates=None, base=None):
        self.cell, self.stokes, self.mode = cell, stokes, mode
        self.lst, self.scope = cell.split('_', 1)
        self.dates = list(dates) if dates else None
        self.base = base
        if mode == 'bulk':
            self.name = f'{cell}_{stokes}'
        elif mode == 'incremental':
            self.name = f'{cell}_{stokes}_{self.dates[0]}' + (f'+{len(self.dates) - 1}' if len(self.dates) > 1 else '')
        else:
            att = _ATT_RE.match(os.path.basename(base.rstrip('/'))) if base and base != 'published' else None
            self.name = f'{cell}_{stokes}_replay_{att["att"] if att else "published"}'

    def __repr__(self):
        return f'Item({self.name}, {self.mode})'


def roots_table():
    """{cell: root} from the fleet's roots.csv (search_inputs.ROOTS_CSV under paths.OPS_ROOT)."""
    c = _cfg()
    p = os.path.join(c.paths.OPS_ROOT, c.search_inputs.ROOTS_CSV)
    out = {}
    with open(p, newline='') as f:
        for row in csv.reader(f):
            if len(row) >= 3 and row[0].strip():
                out[row[0].strip()] = row[2].strip()
    return out


def plan_items(args, stages):
    """The units of this run, in run order (per cell: search.STOKES_ORDER). ValueError on a bad selection (exit 1)."""
    from exopipe.phase3.core import cells as C
    c = _cfg()
    want = [s.strip() for s in (args.stokes or ','.join(c.search.STOKES_ORDER)).split(',') if s.strip()]
    bad = [s for s in want if s not in ('I', 'V')]
    if bad:
        raise ValueError(f'--stokes {args.stokes!r}: I and/or V')
    stokes = [s for s in c.search.STOKES_ORDER if s in want]
    epochs = C.expand_epochs(args.epochs) if getattr(args, 'epochs', None) else []
    nights = [C.parse_date(d) for d in C.expand(args.nights, kind='night')] if getattr(args, 'nights', None) else []
    if epochs and nights:
        raise ValueError('--epochs (per hour) and --nights (incremental on whole cells) are exclusive')
    cellset = C.expand_cells(args.cells) if args.cells else None
    scopes = set(C.expand(args.scopes, universe=C.scopes(), kind='scope')) if getattr(args, 'scopes', None) else None
    excl = set(C.expand_cells(args.exclude_cells)) if getattr(args, 'exclude_cells', None) else set()
    replay = bool(args.replay_attempt) and 'detect' not in stages and not (epochs or nights)

    def keep(cell):
        return cell not in excl and (scopes is None or cell.split('_', 1)[1] in scopes)

    items = []
    if replay and args.replay_attempt != 'published':
        m = _ATT_RE.match(os.path.basename(args.replay_attempt.rstrip('/')))
        if not m:
            raise ValueError(f'--replay-attempt {args.replay_attempt}: the directory name must be <cell>_<S>_<YYYYmmddTHHMMSS>')
        if cellset and cellset != [m['cell']]:
            raise ValueError(f'--replay-attempt is an attempt of {m["cell"]}; --cells {args.cells} names other cells')
        return [Item(m['cell'], m['s'], 'replay', base=os.path.abspath(args.replay_attempt))]
    if epochs:
        for ep in epochs:
            lst, date = C.parse_epoch(ep)
            cands = [x for x in C.all_cells() if x.startswith(lst + '_')]
            if cellset is not None:
                cands = [x for x in cands if x in cellset]
            for cell in cands:
                if keep(cell):
                    items += [Item(cell, s, 'incremental', dates=[date], base=args.replay_attempt or 'published') for s in stokes]
        return items
    cl = list(cellset or [])
    if getattr(args, 'lsts', None):
        for lst in C.expand(args.lsts, universe=C.LSTS, kind='LST'):
            cl += [x for x in C.all_cells() if x.startswith(C.parse_lst(lst) + '_') and x not in cl]
    if not cl:
        raise ValueError('search --tier deep: give --cells, --lsts or --epochs')
    if args.replay_attempt and not (replay or nights):
        raise ValueError('--replay-attempt with the detect stage needs --epochs or --nights (the attempt an incremental search '
                         'is appended to); for a read-only replay give --stages fit,verify')
    for cell in cl:
        if not keep(cell):
            continue
        for s in stokes:
            if nights:
                items.append(Item(cell, s, 'incremental', dates=nights, base=args.replay_attempt or 'published'))
            elif replay:
                items.append(Item(cell, s, 'replay', base='published'))
            else:
                items.append(Item(cell, s, 'bulk'))
    return items


def _last_register_row(path, cell):
    if not os.path.exists(path):
        return None
    hit = None
    with open(path) as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 3 and parts[0] == cell:
                hit = (parts[1], parts[2])
    return hit


def resolve_published(cell, stokes, out):
    """The newest published attempt of (cell, Stokes): this --out's own register first (<out>/published_<S>.csv ->
    <out>/attempts/...), else the production register (paths.OPS_ROOT / search_run.REGISTER) resolved over
    search_run.COLLECT_ROOTS as p3s3_sources.attempt_dir does. None when there is none."""
    c = _cfg()
    sr = c.search_run
    row = _last_register_row(os.path.join(out, f'published_{stokes}.csv'), cell)
    if row:
        d = os.path.join(out, 'attempts', f'{cell}_{stokes}_{row[1]}')
        if os.path.isdir(d):
            return d
    row = _last_register_row(os.path.join(c.paths.OPS_ROOT, sr.REGISTER.format(stokes=stokes)), cell)
    if row:
        node, att = row
        for root in sr.COLLECT_ROOTS:
            d = os.path.join(root, f'{sr.COLLECT_LABEL}_collect_{node}', 'attempts', f'{cell}_{stokes}_{att}')
            if os.path.isdir(d):
                return d
    return None


# =====================================================================================================================
# config truth: engine literals, environment, comparison keys, production trees, pinned inputs
# =====================================================================================================================

def _norm(v):
    if isinstance(v, dict):
        return {str(k): _norm(x) for k, x in sorted(v.items(), key=lambda kv: str(kv[0]))}
    if isinstance(v, (list, tuple)):
        return [_norm(x) for x in v]
    if isinstance(v, (set, frozenset)):
        return sorted(_norm(x) for x in v)
    if isinstance(v, bool) or v is None or isinstance(v, str):
        return v
    if isinstance(v, (int, float)):
        return float(v)
    return repr(v)


def engine_literals():
    """[(config key, config value, engine value, where)] for every config key whose value is hard-coded in the in-place engine
    (the draft's PROTECTED_KEYS engine literals). Imports the engine modules (lib/search/*), which also puts them into this
    process's code fingerprint, so --resume notices an engine edit."""
    from exopipe.phase3.lib.search import (deep_search as DS, search_i as SI, search_v as SV, veto as VT,
                                           identification as ID, candidates as CA, parent_curve_table as PCT, c50 as C50)
    from exopipe.phase3.core import legacy
    c = _cfg()
    sd, st, sf = c.search_detect, c.search_tables, c.search_fpgate
    try:
        v110_py = legacy.interpreter(sf.PYTHON)
    except Exception as e:
        v110_py = f'<{e}>'
    return [
        ('search_detect.I_MASK_FLOOR_DEG', sd.I_MASK_FLOOR_DEG, sd.ELEVATION_FLOOR_DEG,
         'deep_search.py:927 sets the I mask floor = the V floor (P3_DEEP_MIN_ELEVATION)'),
        ('search_detect.DETECTION_FLOOR_SIGMA[I]', sd.DETECTION_FLOOR_SIGMA['I'], SI.CONFIG['SIGMA_THRESHOLD'],
         'search_i.CONFIG[SIGMA_THRESHOLD]'),
        ('search_detect.V_BDSF_THRESH_PIX', sd.V_BDSF_THRESH_PIX, SV.CONFIG['CANDIDATE_SIGMA'], 'search_v.CONFIG[CANDIDATE_SIGMA]'),
        ('search_detect.I_BDSF_THRESH_PIX', sd.I_BDSF_THRESH_PIX, SI.CONFIG['BDSF_THRESH_PIX'], 'search_i.CONFIG[BDSF_THRESH_PIX]'),
        ('search_detect.I_BDSF_THRESH_ISL', sd.I_BDSF_THRESH_ISL, SI.CONFIG['BDSF_THRESH_ISL'], 'search_i.CONFIG[BDSF_THRESH_ISL]'),
        ('search_detect.LOCAL_SNR_ANNULUS_BEAMS', sd.LOCAL_SNR_ANNULUS_BEAMS,
         (SI.CONFIG['LOCAL_SNR_INNER_BEAMS'], SI.CONFIG['LOCAL_SNR_OUTER_BEAMS']), 'search_i.CONFIG[LOCAL_SNR_*_BEAMS]'),
        ('search_detect.LOCAL_SNR_MIN_PIX', sd.LOCAL_SNR_MIN_PIX, SI.CONFIG['LOCAL_SNR_MIN_PIX'], 'search_i.CONFIG[LOCAL_SNR_MIN_PIX]'),
        ('search_detect.MSTRUCT_K', sd.MSTRUCT_K, DS.SIGMA_MSTRUCT, 'deep_search.SIGMA_MSTRUCT'),
        ('search_detect.STRIATION_WINDOW_BEAMS', sd.STRIATION_WINDOW_BEAMS, DS.STRIATION_WINDOW_BEAMS,
         'deep_search.STRIATION_WINDOW_BEAMS'),
        ('search_detect.SIGMA_STACK_BUILD_MIN', sd.SIGMA_STACK_BUILD_MIN, ID.MIN_EPOCHS_SIGMA, 'identification.MIN_EPOCHS_SIGMA'),
        ('search_detect.VETO_SNR_GATE', sd.VETO_SNR_GATE, VT.SNR_GATE, 'veto.SNR_GATE'),
        ('search_detect.VETO_MASK_JY', sd.VETO_MASK_JY, VT.MASK_JY, 'veto.MASK_JY'),
        ('search_detect.VETO_ONSRC_JY', sd.VETO_ONSRC_JY, VT.ONSRC_JY, 'veto.ONSRC_JY'),
        ('search_detect.VETO_MASK_COEF_DEG', sd.VETO_MASK_COEF_DEG, VT.MASK_COEF_DEG, 'veto.MASK_COEF_DEG'),
        ('search_detect.VETO_ATEAM_JY', sd.VETO_ATEAM_JY, VT.ATEAM_JY, 'veto.ATEAM_JY'),
        ('search_detect.VETO_ATEAM_RADIUS_DEG', sd.VETO_ATEAM_RADIUS_DEG, VT.ATEAM_RADIUS_DEG, 'veto.ATEAM_RADIUS_DEG'),
        ('search_detect.VETO_BILOBE', sd.VETO_BILOBE,
         dict(SEP_HI=VT.SEP_HI, LESSER_SIG=VT.LESSER_SIG, LESSER_RATIO=VT.LESSER_RATIO, NEG_SIG=VT.NEG_SIG,
              PA_RAD_BEAM=VT.PA_RAD_BEAM, PA_TOL=VT.PA_TOL), 'veto.SEP_HI ... PA_TOL'),
        ('search_detect.SIZE_FLAG_BEAMS', sd.SIZE_FLAG_BEAMS, DS.SIZE_FLAG_BEAMS, 'deep_search.SIZE_FLAG_BEAMS'),
        ('search_detect.AXIS_FLAG', sd.AXIS_FLAG, DS.AXIS_FLAG, 'deep_search.AXIS_FLAG'),
        ('search_detect.SIZE_SNR_GATE', sd.SIZE_SNR_GATE, DS.SIZE_SNR_GATE, 'deep_search.SIZE_SNR_GATE'),
        ('search_detect.JUPITER_VETO_DEG', sd.JUPITER_VETO_DEG, DS.JUPITER_VETO_DEG, 'deep_search.JUPITER_VETO_DEG'),
        ('search_detect.MOON_VETO_DEG', sd.MOON_VETO_DEG, DS.MOON_VETO_DEG, 'deep_search.MOON_VETO_DEG'),
        ('search_detect.ATEAM_VETO_DEG', sd.ATEAM_VETO_DEG, {'V': DS.ATEAM_VETO_DEG_V, 'I': DS.ATEAM_VETO_DEG_I},
         'deep_search.ATEAM_VETO_DEG_V / _I'),
        ('search_detect.ROWLINE_MIN', sd.ROWLINE_MIN, DS.ROWLINE_MIN, 'deep_search.ROWLINE_MIN'),
        ('search_detect.MAX_V_PER_EPOCH', sd.MAX_V_PER_EPOCH, DS.MAX_V_PER_EPOCH, 'deep_search.MAX_V_PER_EPOCH'),
        ('search_detect.REPEAT_MATCH_ARCMIN', sd.REPEAT_MATCH_ARCMIN, CA.REPEAT_MATCH_ARCMIN, 'candidates.REPEAT_MATCH_ARCMIN'),
        ('search_detect.C50_TABLE', sd.C50_TABLE, C50.MEASURED_FILE, 'c50.MEASURED_FILE'),
        ('search_tables.DEEP_V_FAINT_S', st.DEEP_V_FAINT_S, PCT.DEEP_V_FAINT_S, 'parent_curve_table.DEEP_V_FAINT_S'),
        ('search_tables.XM_EXEMPT_SAMPLES', tuple(st.XM_EXEMPT_SAMPLES), tuple(sorted(PCT.XM_EXEMPT_SAMPLES)),
         'parent_curve_table.XM_EXEMPT_SAMPLES'),
        ('search_tables.XM_EXEMPT_MAX_S', st.XM_EXEMPT_MAX_S, PCT.XM_EXEMPT_MAX_S, 'parent_curve_table.XM_EXEMPT_MAX_S'),
        ('search_tables.XM_SAMPLES_DIR', st.XM_SAMPLES_DIR, PCT._XM_DIR, 'parent_curve_table._XM_DIR'),
        ('search_fpgate.WRAPPER', sf.WRAPPER, DS._V110_WRAP, 'deep_search._V110_WRAP'),
        ('search_fpgate.PYTHON', v110_py, DS._V110_PY, 'deep_search._V110_PY (site PYTHON[search_fpgate.PYTHON])'),
    ] + [(k, a, _src_value(DS, pat, cast), f'deep_search.py source /{pat}/') for k, a, pat, cast in (
        ('search_detect.MSTRUCT_WINDOW_BEAMS', sd.MSTRUCT_WINDOW_BEAMS, r'def _master_structure_map\(vm, bpix, win_beams=([0-9.]+)', float),
        ('search_detect.GARBAGE_PIXEL_JY', sd.GARBAGE_PIXEL_JY, r'a\[np\.abs\(a\) > ([0-9.eE+-]+)\] = 0\.0', float),
        ('search_detect.EMPIRICAL_BEAM_FALLBACK_PX', sd.EMPIRICAL_BEAM_FALLBACK_PX, r'em = ([0-9.]+); en = [0-9.]+', float),
        ('search_detect.DURATION_FALLBACK_S', sd.DURATION_FALLBACK_S, r"600\.0 if cadence == '10min' else ([0-9.]+)\)\)", float),
        ('search_detect.CENTRAL_RMS_HALF_PX', sd.CENTRAL_RMS_HALF_PX, r'_halfI = min\(([0-9]+), _hI // 4', int),
        ('search_detect.PB_FLOOR', sd.PB_FLOOR, r'np\.where\(_pb > ([0-9.]+), _pb, np\.nan\)', float),
        ('search_detect.DETECTION_FLOOR_SIGMA[V]', sd.DETECTION_FLOOR_SIGMA['V'],
         r'run_identification_v\(rV, master_I, hdr, beam, None, sigma_thresh=([0-9.]+)', float),
        ('search_detect.EXTRACTION_FAILURE', sd.EXTRACTION_FAILURE, r"_xrefuse = cadence == '(deep)'",
         lambda x: 'refuse' if x == 'deep' else x),
        ('search_tables.MID_HOUR_LAT_DEG', st.MID_HOUR_LAT_DEG, r'lat = np\.radians\(([0-9.]+)\)', float),
        ('search_tables.MID_HOUR_OFFSET_H', st.MID_HOUR_OFFSET_H, r'int\(lst\[:-1\]\) \+ ([0-9.]+)\)', float))]


def _src_value(mod, pattern, cast):
    """The value an inline literal of `mod`'s source has: the first match of `pattern` (one group), cast; '<absent>' when the
    pattern no longer matches (the engine changed: the config must be re-checked)."""
    import inspect
    m = re.search(pattern, inspect.getsource(mod))
    return cast(m.group(1)) if m else '<absent>'


def literal_mismatches():
    return [(k, a, b, w) for k, a, b, w in engine_literals() if _norm(a) != _norm(b)]


_COMPARISON = (('search_inputs.ALLOW_OLD_MASTERS', False), ('search_detect.BEAM_GATE_ON', True),
               ('search_detect.STRIATION_OFF_BATTERY', False), ('search_detect.SIGMA_STACK_REMOVES', False),
               ('search_detect.GATE_LEDGER_ONLY', False), ('search_detect.STREAMING', True), ('search_fpgate.ON', True),
               ('search_fpgate.REMOVES', ('FP',)), ('search_detect.REFINE_X2', True))


def comparison_deviations():
    c = _cfg()
    return [(k, c.get(k)) for k, prod in _COMPARISON if _norm(c.get(k)) != _norm(prod)]


def engine_env(cell_root_parent, attempt, flow_root, label, archive_root, tmpdir):
    """The engine child's environment (spec 7.3), built from config. core.legacy.run clears every inherited P3_* first; COMPARISON
    keys at their production values set nothing (their variables stay unset, as in the fleet)."""
    from exopipe.phase3.core import legacy
    from exopipe.phase3.core.record import Refusal
    c = _cfg()
    sd, si, st, sf = c.search_detect, c.search_inputs, c.search_tables, c.search_fpgate
    env = {'P3_COADD_ROOT': cell_root_parent, 'P3_FLOW_ROOT': flow_root, 'P3_STAGING_ROOT': attempt,
           'P3_IRESID_ROOT': f'{attempt}/Run_{label}/iresid', 'P3_VRESID_ROOT': f'{attempt}/Run_{label}/vresid',
           'P3_MASTER_GEN': si.MASTER_GEN, 'P3_ARCHIVE_RUN': c.paths.ARCHIVE_RUN, 'P3_ARCHIVE_ROOT': archive_root,
           'P3_TABLE_ELEV': '1' if st.TABLE_ELEV else '0',
           'P3_TABLE_ELEV_BANDS': ','.join(f'{float(lo):g}-{float(hi):g}' for lo, hi in st.ELEV_BANDS_DEG),
           'P3_TABLE_XM_EXEMPT': '1' if st.XM_EXEMPT_ON else '0',
           'P3_DEEP_MIN_ELEVATION': repr(float(sd.ELEVATION_FLOOR_DEG)), 'P3_DEEP_MIN_EPOCHS': str(int(sd.MIN_EPOCHS)),
           'P3_MIN_EPOCHS': str(int(sd.SIGMA_STACK_MIN_EPOCHS)), 'P3_SIZE_FLOOR_BEAMS': repr(float(sd.SIZE_FLOOR_BEAMS)),
           'P3_AXIS_REMOVE': repr(float(sd.AXIS_REMOVE)), 'P3_SANITY_I_X': repr(float(sd.SANITY_X)),
           'P3_SEARCH_TIER_SIGMA': repr(float(sd.SEARCH_TIER_SIGMA)), 'P3_BEAM_GATE_RATIO': repr(float(sd.BEAM_GATE_RATIO)),
           'P3_SANITY_MIN_NIGHTS': str(int(sd.SANITY_MIN_NIGHTS)), 'P3_SEARCH_TIER_MIN': str(int(sd.SEARCH_TIER_MIN)),
           'P3_BEAM_GATE_MIN_NIGHTS': str(int(sd.BEAM_GATE_MIN_NIGHTS)),
           'P3_V110_TIMEOUT': str(int(sf.TIMEOUT_S)), 'DEEP_NPROC_RESID': str(int(sd.NPROC_RESID)),
           'DEEP_XLA_MEMFRAC': repr(float(sd.FLOW_XLA_MEM_FRACTION)), 'P3_FLOW_PYTHON': legacy.interpreter(sd.FLOW_PYTHON),
           'P3_FLOW_LOCK': sd.FLOW_LOCK, 'TMPDIR': tmpdir, 'PYTHONUNBUFFERED': '1',
           'PYTHONPATH': _PKG_ROOT + (os.pathsep + os.environ['PYTHONPATH'] if os.environ.get('PYTHONPATH') else '')}
    env['EXOPIPE3_EPHEMERIS_TIME'] = ephemeris_time(sd.EPHEMERIS_TIME, 'search_detect.EPHEMERIS_TIME')   # WAVE2 B-30
    from exopipe.phase3.lib.masters import admit as _AD
    from exopipe.phase3.lib.search import admitted as _ADS
    env[_ADS.POLICY_ENV] = _AD.admitted_policy(c.masters_admit.ADMITTED_NIGHTS)    # D-ADMIT: how the engine reads admitted nights
    from exopipe.phase3.lib.search import removal_rules as _RR
    try:                                                                           # B-02: the search's removal rules (step 3a)
        env[_RR.RULES_ENV] = _RR.search_rules(sd.REMOVAL_RULES)
    except RuntimeError as e:
        from exopipe.phase3.core.record import Refusal
        raise Refusal(str(e))
    env[_RR.AUDIT_ENV] = c.search_tables.SOLAR_AUDIT                               # B-02: the daytime gate's solar audit (G2 pin)
    if not si.ARCHIVE_WRITE:
        env['P3_ARCHIVE_WRITE'] = '0'
    if not sd.REFINE_X2:
        env['P3_REFINE_OFF'] = '1'
    if st.EXTRA_TABLES:
        env['P3_TABLE_EXTRA'] = st.EXTRA_TABLES
    if si.ALLOW_OLD_MASTERS:
        env['P3_ALLOW_OLD_MASTERS'] = '1'
    if not sd.BEAM_GATE_ON:
        env['P3_BEAM_GATE_OFF'] = '1'
    if sd.STRIATION_OFF_BATTERY:
        env['P3_STRIATION_OFF'] = '1'
    if sd.SIGMA_STACK_REMOVES:
        env['P3_SSTACK_REMOVES'] = '1'
    if sd.GATE_LEDGER_ONLY:
        env['P3_GATE_LEDGER_ONLY'] = '1'
    if not sd.STREAMING:
        env['P3_STREAM_RESID'] = '0'
    if not sf.ON:
        env['P3_V110_OFF'] = '1'
    rem = set(sf.REMOVES)
    if rem == {'FP', 'marginal'}:
        env['P3_V110_READING'] = 'marginal'
    elif rem != {'FP'}:
        raise Refusal(f"search_fpgate.REMOVES = {sf.REMOVES!r}: the engine can remove ('FP',) (production) or ('FP', 'marginal') "
                      f"(P3_V110_READING=marginal) only")
    return env


def production_tree(path):
    """The production root or tree holding `path` (paths.PRODUCTION_ROOTS + search_run.PRODUCTION_TREES), or None."""
    c = _cfg()
    real = os.path.realpath(path)
    for r in tuple(c.paths.PRODUCTION_ROOTS) + tuple(c.search_run.PRODUCTION_TREES):
        rr = os.path.realpath(r)
        if real == rr or real.startswith(rr.rstrip('/') + '/'):
            return r
    return None


def md5_file(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def base_rules_refusal(base_attempt, rules):
    """B-02: one attempt, one set of removal rules, of one version - None when an append under ``rules`` may extend
    ``base_attempt``, else the refusal's words (an attempt without the step's record counts as 'production', an unreadable
    record refuses: removal_rules.attempt_rules)."""
    from exopipe.phase3.lib.search import removal_rules as RR
    try:
        have = RR.attempt_rules(base_attempt)
    except RuntimeError as e:
        return str(e)
    want = RR.rules_record(rules)
    if have == want:
        return None
    # the --set advice only when it can succeed: this code, under the base's rules, writes exactly the base's record
    fits = have['rules'] != want['rules'] and have == RR.rules_record(have['rules'])
    return (f'{os.path.basename(base_attempt.rstrip("/"))} was searched under search_detect.REMOVAL_RULES {have["rules"]!r} '
            f'(version {have["version"]}); an append under {want["rules"]!r} (version {want["version"]}) would mix the rules in '
            f'one attempt - ' + (f'run it with --set search_detect.REMOVAL_RULES={have["rules"]}, or ' if fits else '')
            + 're-search the cell (--fresh)')


def pinned_inputs(stokes):
    """Gate G2 (spec 5.4): [(name, path, ok, detail, refuse)] for the md5-pinned inputs the engine reads."""
    from exopipe.phase3.lib.search import parent_curve_table as PCT
    c = _cfg()
    st, sf = c.search_tables, c.search_fpgate
    out = []

    def pin(name, path, md5, missing_refuses):
        if not os.path.exists(path):
            out.append((name, path, not missing_refuses, 'missing', missing_refuses))
            return
        got = md5_file(path)
        out.append((name, path, got == md5, f'md5 {got}' + ('' if got == md5 else f' != pinned {md5}'), got != md5))
    pin('parent tables (the engine copy)', PCT.PATH, st.PARENT_TABLES_MD5, True)
    pin('pulsar list', st.PULSAR_CSV, st.PULSAR_CSV_MD5, False)
    if c.search_detect.REMOVAL_RULES == 'once-in-search':                       # B-02: the daytime gate's quarantine bands
        pin('solar audit (the daytime gate, B-02)', st.SOLAR_AUDIT, st.SOLAR_AUDIT_MD5, True)
    if stokes == 'V':
        pin('v110 null trials', sf.NULL_TRIALS, sf.NULL_TRIALS_MD5, True)
        pin('v110 PA null profile', sf.PA_NULL, sf.PA_NULL_MD5, bool(sf.PA_NULL_REQUIRED))
    return out


# =====================================================================================================================
# flow cache (p3s5_search_fleet.sh ship_inputs, flow part)
# =====================================================================================================================

def flow_cache_state(cell_work, master_i, n_epochs, log=None):
    """('none' | 'current' | 'unverified' | 'stale' | 'partial' | 'error', n_flows) of a cell's flow cache, with the fleet's
    currency script run in place (exit 0 current, 1 stale, 2 no sidecar)."""
    from exopipe.phase3.core import env as E, legacy
    c = _cfg()
    flows = glob.glob(os.path.join(cell_work, 'ep_*_flow.npy'))
    if not flows:
        return 'none', 0
    side = os.path.exists(os.path.join(cell_work, 'flow_generation.json'))
    script = os.path.join(E.legacy_root('exopipe_ops'), c.search_run.CACHE_CURRENT_SCRIPT)
    res = legacy.run(script, [master_i, cell_work], python='default', refusal_codes=(), refusal_text=(), log=log)
    state = {0: 'current', 1: 'stale', 2: 'unverified'}.get(res.rc, 'error')
    if state in ('current', 'unverified') and not side and len(flows) < n_epochs - c.search_run.FLOW_PARTIAL_SLACK:
        state = 'partial'
    return state, len(flows)


def drop_flow_cache(cell_work):
    """The fleet's drop of a stale cache (p3s5_search_fleet.sh:98,119): flows, sidecar and the up-sampled master."""
    for p in glob.glob(os.path.join(cell_work, 'ep_*_flow.npy')) + [os.path.join(cell_work, 'flow_generation.json'),
                                                                    os.path.join(cell_work, 'master_up2.fits')]:
        if os.path.exists(p):
            os.remove(p)


def seed_flow_cache(src_cell, cell_work):
    """Copy a flow cache read-only from src_cell into cell_work (ep_*_flow.npy, flow_generation.json, master_up2.fits)."""
    os.makedirs(cell_work, exist_ok=True)
    n = 0
    for p in glob.glob(os.path.join(src_cell, 'ep_*_flow.npy')) + [os.path.join(src_cell, 'flow_generation.json'),
                                                                   os.path.join(src_cell, 'master_up2.fits')]:
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(cell_work, os.path.basename(p)))
            n += 1
    return n


# =====================================================================================================================
# preflight
# =====================================================================================================================

def _gpu_host():
    return _host() in set(_cfg().site.GPU_HOSTS or ())


def generation_nights(base, m_i):
    """The nights of the master's generation record as the engine's input-provenance gate reads them (deep_search.py:1264-
    1275): every date of coadd_final/weights.csv (omitted epochs included) and every 'COADD constituent <date>' HISTORY line of
    the I master."""
    from astropy.io import fits
    out = set()
    w = os.path.join(base, 'coadd_final', 'weights.csv')
    if os.path.exists(w):
        with open(w, newline='') as fh:
            out |= {r['date'] for r in csv.DictReader(fh) if r.get('date')}
    hist = fits.getheader(m_i).get('HISTORY', [])
    out |= {str(x).split()[-1] for x in hist if str(x).startswith('COADD constituent')}
    return out


def subtracted_dates(base, scope, stokes):
    out = []
    for p in sorted(glob.glob(os.path.join(base, 'subtracted', f'*_{stokes}_sub_{scope}.fits'))):
        m = re.search(r'(\d{4}-\d\d-\d\d(?:T\d\d)?)_' + stokes + r'_sub_', os.path.basename(p))
        if m:
            out.append(m.group(1))
    return out


def preflight(item, ctx, u=None, fix=False, resume_probe=False):
    """The facts of one unit; raises record.Refusal (a gate) or record.NoInput. With fix=True (a real run, not --dry-run) it may
    act on the run's OWN flow root: drop a stale cache on a GPU host, seed from --deep-flow-from. Read-only otherwise.
    resume_probe=True (the --resume identity check): an incremental unit's base attempt is not re-resolved (after a publish
    the newest register row is the unit's own attempt) - its record's own base inputs are re-fingerprinted instead."""
    from exopipe.phase3.core import exclusions
    from exopipe.phase3.core.record import NoInput, Refusal
    c = _cfg()
    si = c.search_inputs
    f = {'notes': [], 'gates': []}
    root = ctx['roots'].get(item.cell)
    if root is None:
        raise Refusal(f'{item.cell} is not in {c.search_inputs.ROOTS_CSV} (the cell -> root table)')
    if os.path.realpath(root) != os.path.realpath(c.paths.COADD_SB_ROOT):
        f['notes'].append(f'roots.csv root {root} != paths.COADD_SB_ROOT {c.paths.COADD_SB_ROOT} (roots.csv used, as the fleet)')
    base = os.path.join(root, item.cell)
    f.update(root=root, base=base)
    if item.mode == 'replay':
        att = item.base if item.base != 'published' else resolve_published(item.cell, item.stokes, ctx['out'])
        if not att or not os.path.isdir(att):
            raise Refusal(f'{item.name}: no attempt to replay ({item.base}; register rows resolve to nothing)')
        m = _ATT_RE.match(os.path.basename(att.rstrip('/')))
        if not m or (m['cell'], m['s']) != (item.cell, item.stokes):
            raise Refusal(f'{att} is not an attempt of {item.cell} Stokes {item.stokes}')
        runs = sorted(glob.glob(os.path.join(att, 'transients', 'Run_*')))
        if not runs:
            raise Refusal(f'{att}: no transients/Run_<label>/ (not a search attempt)')
        f.update(attempt=att, label=os.path.basename(runs[0])[4:], att_tag=m['att'])
        f['gates'].append(('attempt', True, att))
        return f
    from exopipe.phase3.core import env as E
    if not E.readable_here(base):
        raise Refusal(f'{item.cell}: the cell inputs {base} are node-local to {E.node_local_holder(base)} (site NODE_LOCAL_ROOTS), '
                      f'not readable on {_host()}; run there')
    if not os.path.isdir(base):
        raise Refusal(f'{item.cell}: no cell inputs at {base} on {_host()} (the deep inputs are node-local; run where they are)')
    cf = os.path.join(base, 'coadd_final')
    m_i = os.path.join(cf, f'I_deep_Taper_R075_{item.scope}.fits')
    m_v = os.path.join(cf, f'V_deep_Taper_R0_{item.scope}.fits')
    if not os.path.exists(m_i):
        raise Refusal(f'{item.cell}: no deep I master at {m_i} - inputs missing (gate E1)')
    from exopipe.phase3.core.record import fits_primary_cards
    op_i = fits_primary_cards(m_i).get('COADDOP')
    f['gates'].append(('E2 I master COADDOP', op_i == si.MASTER_GEN, op_i))
    if op_i != si.MASTER_GEN and not si.ALLOW_OLD_MASTERS:
        raise Refusal(f'{item.cell}: I master COADDOP={op_i!r}, required {si.MASTER_GEN!r} - superseded master (gate E2)')
    op_v = fits_primary_cards(m_v).get('COADDOP') if os.path.exists(m_v) else None
    if os.path.exists(m_v) and op_v != si.MASTER_GEN and not si.ALLOW_OLD_MASTERS:
        raise Refusal(f'{item.cell}: V master COADDOP={op_v!r}, required {si.MASTER_GEN!r} - superseded master (gate E2)')
    if item.stokes == 'V' and not os.path.exists(m_v):
        raise Refusal(f'{item.cell}: no V master {m_v} - the V search cannot run (the engine would refuse at E33)')
    for gate, name in (('E14', 'residuals_provisional.csv'), ('E10', 'coadd_final/night_image_gate.csv'),
                       ('E11', 'coadd_final/epoch_gate.csv')):
        p = os.path.join(base, name)
        f['gates'].append((f'{gate} {name}', os.path.exists(p), p))
        if not os.path.exists(p):
            raise Refusal(f'{item.cell}: {name} missing - gate {gate} cannot run; refused (never skipped)')
    if not si.ALLOW_OLD_MASTERS:
        from exopipe.phase3.lib.masters import admit as AD, membership as MB
        viol, rep = AD.verify(base)                    # D-ADMIT: an admitted night is 'admitted, rebuild pending', not absent
        viol = viol + MB.verify_products(base)
        f['gates'].append(('E3 membership rulings', not viol, f'{len(viol)} violation(s)'
                           + (f'; {len(rep)} admitted night(s), rebuild pending' if rep else '')))
        f['notes'] += rep
        if viol:
            raise Refusal(f'{item.cell}: the master violates the membership rulings (gate E3): ' + ' | '.join(viol[:3]))
    dates_i = subtracted_dates(base, item.scope, 'I')
    dates_v = subtracted_dates(base, item.scope, 'V')
    if not dates_i:
        raise Refusal(f'{item.cell}: ZERO epochs (subtracted/*_I_sub_{item.scope}.fits) - gate E5')
    ex = exclusions.load()
    admitted = ex.admitted(item.lst, item.scope, dates_i)
    f.update(m_i=m_i, m_v=m_v if os.path.exists(m_v) else None, coaddop=op_i, dates_i=dates_i, dates_v=dates_v,
             admitted=admitted, excluded_now=ex.excluded_constituents(item.lst, item.scope, dates_i),
             exclusions=ex.fingerprint())
    if item.mode == 'incremental':
        present = [d for d in item.dates if d in dates_i]
        live = [d for d in present if d in admitted]
        f['included'] = item.dates
        if not present:
            raise NoInput(f'{item.cell}: no epoch product for {", ".join(item.dates)} in {base}/subtracted (frames makes it)')
        if not live:
            raise NoInput(f'{item.cell}: {", ".join(present)} excluded by data_decisions.csv for {item.scope}')
        if len(present) < len(item.dates):
            f['notes'].append(f'listed nights without an epoch product: {sorted(set(item.dates) - set(present))}')
        b = item.base if item.base != 'published' else resolve_published(item.cell, item.stokes, ctx['out'])
        if resume_probe:
            b = None
        elif not b or not os.path.isdir(b):
            raise Refusal(f'{item.cell} {item.stokes}: no published attempt to extend (p3s4_incremental_fleet.sh refuses the '
                          f'same; run the bulk search of the cell first)')
        mb = _ATT_RE.match(os.path.basename(b.rstrip('/'))) if b else None
        if b and (not mb or (mb['cell'], mb['s']) != (item.cell, item.stokes)):
            raise Refusal(f'{b} is not an attempt of {item.cell} Stokes {item.stokes}')
        if b:
            if not glob.glob(os.path.join(b, 'transients', f'Run_{ctx["label"]}')):
                raise Refusal(f'{b}: no transients/Run_{ctx["label"]}/ to append to (label mismatch: --run {ctx["label"]})')
            f['base_attempt'] = b
            # A night already searched in the base attempt would be appended twice; a night of the master's generation record
            # (weights.csv / HISTORY constituent: the nights the engine's input-provenance gate E15 admits) that the base did not
            # search is a RE-ADMITTED epoch, which Gregg ruled is searched by a FULL re-search of the cell, never appended
            # (R-SIGMA-STACK-INCLUDES-THE-RE-ADMITTED-EPOCH-BACKFILL-BY-RE-SEARCH, 2026-09-13; spec search.md section 8)
            searched = {r['date'] for r in _image_rows(b, ctx['label'], item.stokes)}
            dup = sorted(set(live) & searched)
            readm = sorted((set(live) & generation_nights(base, m_i)) - searched)
            if dup or readm:
                raise Refusal(f'{item.cell} {item.stokes}: ' + '; '.join(
                    ([f'{", ".join(dup)} already searched in {os.path.basename(b)} (an append would duplicate its rows)']
                     if dup else [])
                    + ([f'{", ".join(readm)} in the master\'s generation record: a re-admitted epoch is searched by a full '
                        f're-search of the cell (search --tier deep --cells {item.cell} --fresh), never appended '
                        f'(R-SIGMA-STACK-INCLUDES-THE-RE-ADMITTED-EPOCH-BACKFILL-BY-RE-SEARCH)'] if readm else [])))
            # B-02: one attempt, one set of removal rules - an append runs under the rules its base was searched under (an
            # attempt without the step's record: 'production'; Gregg 2026-10-14: per-hour updates on Run_P3S5 under
            # REMOVAL_RULES=production until the full re-search)
            from exopipe.phase3.lib.search import removal_rules as RR
            try:
                rules = RR.search_rules(c.search_detect.REMOVAL_RULES)
            except RuntimeError as e:
                raise Refusal(str(e))
            why = base_rules_refusal(b, rules)
            f['gates'].append(('B-02 removal rules of the base attempt', why is None, why or f'{rules}, as the base'))
            if why:
                raise Refusal(f'{item.cell} {item.stokes}: {why}')
    for name, path, ok, detail, refuse in pinned_inputs(item.stokes):
        f['gates'].append((f'G2 {name}', ok, detail))
        if not ok and refuse:
            raise Refusal(f'{name} {path}: {detail} - pinned input (gate G2)')
        if not ok:
            f['notes'].append(f'{name} {path} {detail}: the engine proceeds as production does (recorded)')
    if not os.path.exists(c.search_detect.C50_TABLE):
        f['notes'].append(f'C50 table {c.search_detect.C50_TABLE} not on this node: c50_mjy written blank (B-44, as production)')
    # disk (a node condition, not a cell's fault: error, retried by the runner)
    try:
        free = shutil.disk_usage(ctx['work_fs']).free / 1e9
    except OSError:
        free = None
    need = c.search_run.FAST_MIN_FREE_GB
    f['gates'].append(('disk', free is None or free >= need, f'{free:.0f} GB free, need {need}' if free is not None else '?'))
    if free is not None and free < need and fix:
        raise RuntimeError(f'{ctx["work_fs"]}: {free:.0f} GB free < {need} GB (search_run.FAST_MIN_FREE_GB) - node condition, '
                           f'not an attempt; retry when space is free')
    if resume_probe:                # a done unit's identity does not depend on the flow work dir (hygiene removes it)
        return f
    # flow cache of the cell
    cw = os.path.join(ctx['flow_root'], item.cell)
    state, nf = flow_cache_state(cw, m_i, len(dates_i))
    seed_pending = False
    if state == 'none' and ctx.get('flow_from'):
        src = os.path.join(ctx['flow_from'], item.cell)
        if not glob.glob(os.path.join(src, 'ep_*_flow.npy')):
            f['notes'].append(f'--deep-flow-from: no flows under {src}')
        elif fix:
            seed_flow_cache(src, cw)
            state, nf = flow_cache_state(cw, m_i, len(dates_i))
            f['notes'].append(f'flow cache seeded from {src}: {nf} flows, {state}')
        else:
            seed_pending = True
            f['notes'].append(f'flow cache would be seeded from {src} (currency checked after the copy)')
    gpu = _gpu_host()
    f.update(flow_state=state, n_flows=nf, gpu=gpu, cell_work=cw)
    if state == 'stale' and not gpu:
        raise Refusal(f'{item.cell}: the flow cache in {cw} is of another master generation and {_host()} has no GPU to '
                      f'recompute it (search-only node: not ready)')
    if gpu and state in ('stale', 'unverified', 'partial'):
        # GPU host: a cache the currency script does not call current (stale = exit 1; no sidecar = exit 2, which 'partial'
        # always is) is DROPPED and the search computes every flow (p3s5_search_fleet.sh:96-98: `! p3s3_cache_current.py`
        # drops on any non-zero exit)
        if fix:
            drop_flow_cache(cw)
            f['notes'].append(f'flow cache {state} ({nf} flows) dropped (p3s5_search_fleet.sh:96-98); the GPU computes the '
                              f'flows in the search')
            state, nf = 'none', 0
            f.update(flow_state=state, n_flows=nf)
        else:
            f['notes'].append(f'flow cache {state} ({nf} flows) would be dropped (p3s5_search_fleet.sh:96-98)')
    elif state in ('none', 'partial') and not gpu and not seed_pending:
        raise Refusal(f'{item.cell}: flow cache {state} ({nf} flows for {len(dates_i)} epochs) in {cw} and {_host()} has no GPU '
                      f'(site GPU_HOSTS) - not ready (search-only node); run on a GPU host or seed with --deep-flow-from')
    elif state in ('none', 'partial'):
        f['notes'].append(f'flow cache {state}: the engine computes the missing x2 flows on this GPU host (the documented GPU '
                          f'exception of decision 9)')
    elif state == 'error':
        raise RuntimeError(f'{item.cell}: flow-cache currency check failed ({c.search_run.CACHE_CURRENT_SCRIPT})')
    f['gates'].append(('flow cache', True, f'{state}, {nf} flows, GPU {"yes" if gpu else "no"}'))
    return f


def input_paths(item, f):
    """(small, per-epoch) input paths of a unit: small files fingerprinted by sha1 / FITS identity, per-epoch products and LOO
    masters by size/mtime/cards. The same set is the unit's input identity under --resume (a new epoch product, LOO or V master
    makes a unit not resumable - the fleet's adoption test, p3s5_search_fleet.sh:174-190 incl. D6)."""
    from exopipe.phase3.lib.search import parent_curve_table as PCT
    c = _cfg()
    base = f['base']
    cf = os.path.join(base, 'coadd_final')
    small = [f['m_i'], f.get('m_v'), os.path.join(base, 'residuals_provisional.csv'), os.path.join(cf, 'weights.csv'),
             os.path.join(cf, 'night_image_gate.csv'), os.path.join(cf, 'epoch_gate.csv'),
             os.path.join(cf, 'admissions.csv'),                  # D-ADMIT: which nights the engine accepts as admitted
             PCT.PATH,
             c.search_tables.PULSAR_CSV, os.path.join(c.paths.OPS_ROOT, c.search_inputs.ROOTS_CSV)]
    if c.search_detect.REMOVAL_RULES == 'once-in-search':                       # B-02: the daytime gate reads it
        small.append(c.search_tables.SOLAR_AUDIT)
    small += sorted(glob.glob(os.path.join(cf, f'PB_deep*_{item.scope}.fits')))
    small = [p for p in small if p and os.path.exists(p)]
    if f.get('base_attempt'):
        small += sorted(glob.glob(os.path.join(f['base_attempt'], 'transients', 'Run_*', '*.csv')))
    big = (sorted(glob.glob(os.path.join(base, 'subtracted', f'*_sub_{item.scope}.fits')))
           + sorted(glob.glob(os.path.join(cf, 'loo', f'*_loo_*_{item.scope}.fits'))))
    return small, big


def add_inputs(u, item, f):
    """Fingerprint the unit's inputs (record v2: FITS by DATASUM + processing cards; per-epoch products by size/mtime/cards,
    not by a full sha1 of every epoch frame)."""
    from exopipe.phase3.core import record
    small, big = input_paths(item, f)
    for p in small:
        u.add_input(p)
    for p in big:
        u.inputs[os.path.abspath(p)] = record.fingerprint(p, hash_limit_bytes=0)
    u.set_admitted(f['admitted'])


def print_plan(item, f, stages, ctx):
    lines = [f'== {item.name} ({item.mode}; stages {",".join(stages)})']
    if item.mode == 'replay':
        lines.append(f'   attempt (read only): {f["attempt"]}  label {f["label"]}')
        lines.append(f'   outputs: {os.path.join(ctx["out"], "replay", os.path.basename(f["attempt"]))}/'
                     f'{{fit_positions.csv, fit_positions_summary.json, verify_report.csv}}')
    else:
        lines.append(f'   cell {f["base"]}: I master COADDOP {f["coaddop"]}, V master {"yes" if f.get("m_v") else "no"}; '
                     f'{len(f["dates_i"])} epochs (I), {len(f["dates_v"])} (V); admitted {len(f["admitted"])}; '
                     f'excluded now {f["excluded_now"] or "none"}')
        if f.get('included'):
            lines.append(f'   incremental: nights {",".join(f["included"])} appended to {f["base_attempt"]}')
        lines.append(f'   flow cache: {f["flow_state"]} ({f["n_flows"]} flows) in {f["cell_work"]}; GPU here: {f["gpu"]}')
        lines.append(f'   attempt staged under {ctx["attempts"]}/{item.cell}_{item.stokes}_<UTC>/, published to '
                     f'{ctx["out"]}/attempts/; archive copies -> {ctx["archive_root"]}')
    for g, ok, d in f['gates']:
        lines.append(f'   gate {g}: {"ok" if ok else "NOT OK"} ({d})')
    for n in f['notes']:
        lines.append(f'   note: {n}')
    print('\n'.join(lines), flush=True)


# =====================================================================================================================
# stages
# =====================================================================================================================

def _image_rows(attempt, label, stokes):
    p = os.path.join(attempt, 'transients', f'Run_{label}', 'v_deep_images.csv')
    if not os.path.exists(p):
        return []
    with open(p, newline='') as fh:
        return [r for r in csv.DictReader(fh) if r.get('stokes') == stokes]


def stage_quarantine(u, item, ctx, f):
    """Archive's frame quarantine before a FRESH attempt (decision 6, group 10). Window: plan only."""
    from exopipe.phase3.core.record import Refusal
    c = _cfg()
    sr = c.search_run
    if sr.QUARANTINE_EPOCH == 'launch-file':
        lf = os.path.join(c.paths.OPS_ROOT, sr.LAUNCH_EPOCH_FILE)
        try:
            launch = float(open(lf).read().strip())
        except (OSError, ValueError) as e:
            raise Refusal(f'quarantine: launch epoch unreadable at {lf} ({e}); the fleet refuses an unset launch epoch')
    elif sr.QUARANTINE_EPOCH == 'job-created':
        launch = ctx['t0']
    else:
        raise Refusal(f'search_run.QUARANTINE_EPOCH {sr.QUARANTINE_EPOCH!r}: launch-file or job-created')
    if c.paths.PRODUCTION_WRITES:
        # the archive step's quarantine (lib/archive/quarantine.py): production's helper in place, recorded in this unit
        # with its gates (G2, G7 precheck) and manifest; the writer claim (G12) is the search's own cell lock
        from exopipe.phase3.lib.archive.quarantine import quarantine_for_search
        quarantine_for_search(u, item.cell, item.stokes, launch, sr.QUARANTINE_ROUND, apply=True, check_writer=False,
                              log=os.path.join(ctx['out'], '_records', 'search', 'logs', f'{item.name}.quarantine.log'))
        return
    # window: the frames production WOULD move (p3s5_quarantine_frames.py:50-51), nothing moved
    arch = c.paths.PHASE3_ARCHIVE_ROOT
    qroot = os.path.join(arch, f'Run_SEARCH_QUARANTINE_{sr.QUARANTINE_ROUND}')
    old = [p for p in sorted(glob.glob(os.path.join(arch, item.lst, '*', c.paths.ARCHIVE_RUN, item.scope, item.stokes, 'deep',
                                                    '*.fits')))
           if '_msub' in os.path.basename(p) and os.path.getmtime(p) < launch]
    plan = os.path.join(ctx['out'], 'quarantine_plan', f'{item.name}.csv')
    os.makedirs(os.path.dirname(plan), exist_ok=True)
    with open(plan + '.part', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['path', 'dest', 'bytes', 'mtime_utc', 'cell', 'stokes', 'launch_epoch'])
        for p in old:
            w.writerow([p, os.path.join(qroot, os.path.relpath(p, arch)), os.path.getsize(p),
                        time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(os.path.getmtime(p))), item.cell, item.stokes, launch])
    os.replace(plan + '.part', plan)
    u.add_output(plan, 'csv_table', required_columns=('path', 'dest', 'bytes'))
    u.note(f'quarantine (window, plan only): {len(old)} archive frame(s) older than the launch epoch would be moved '
           f'(a production run moves them with lib.archive.quarantine.quarantine_for_search)')


def stage_detect(u, item, ctx, f):
    """run_deep_cell in place, one child process (core.legacy.run, module mode)."""
    from exopipe.phase3.core import legacy
    from exopipe.phase3.core.record import NoInput, Refusal
    c = _cfg()
    label = ctx['label']
    tag = _utc_tag()
    attempt = os.path.join(ctx['attempts'], f'{item.cell}_{item.stokes}_{tag}')
    if os.path.exists(attempt):
        raise RuntimeError(f'{attempt} exists (two units in one second?)')
    if item.mode == 'incremental':
        shutil.copytree(f['base_attempt'], attempt, symlinks=True,
                        ignore=shutil.ignore_patterns('fit_positions*', 'verify_report*', '*.lock'))
        main_log = os.path.join(attempt, f'{item.cell}.log')
        if os.path.exists(main_log):
            shutil.copy2(main_log, os.path.join(attempt, f'{item.cell}.base.log'))
        log = os.path.join(attempt, f'{item.cell}.inc.log')
    else:
        os.makedirs(attempt)
        log = os.path.join(attempt, f'{item.cell}.log')
    before = len(_image_rows(attempt, label, item.stokes))
    tmpd = os.path.join(ctx['tmp'], item.name + '_' + tag)
    os.makedirs(tmpd, exist_ok=True)
    etmp = engine_tmpdir(c.search_run.ENGINE_TMP_ROOT, f'{item.cell}_{item.stokes}')     # short: AF_UNIX socket paths
    env = engine_env(os.path.dirname(f['base']), attempt, ctx['flow_root'], label, ctx['archive_root'], etmp)
    spec = dict(lst=item.lst, scope=item.scope, cell=item.cell, label=label, stokes=item.stokes, cadence=c.search.CADENCE,
                included_dates=item.dates if item.mode == 'incremental' else None, markers=list(c.search.REFUSAL_MARKERS),
                result=os.path.join(tmpd, 'result.json'))
    spec_path = os.path.join(tmpd, 'spec.json')
    with open(spec_path, 'w') as fh:
        json.dump(spec, fh, indent=1)
    t_detect = time.time()
    u.meta.update(attempt_staged=attempt, engine_env={k: v for k, v in env.items() if k.startswith(('P3_', 'DEEP_'))})
    from exopipe.phase3.lib.search import removal_rules as _RR                    # B-02: the rules this attempt ran (step 3a),
    u.meta['removal_rules'] = _RR.write_attempt_rules(attempt, env[_RR.RULES_ENV])  # in the attempt and the unit record (an
                                                                                   # append reads its base's: preflight)
    try:
        res = legacy.run(_CHILD_MODULE, ['--child', spec_path], python=c.search.ENGINE_PYTHON, module=True, env=env, cwd=attempt,
                         log=log, refusal_codes=(2,), refusal_text=(), unit=u)
    finally:
        shutil.rmtree(etmp, ignore_errors=True)
    try:
        result = json.load(open(spec['result']))
    except (OSError, ValueError):
        result = {'status': 'error', 'reason': f'no result from the engine child (rc {res.rc}): {res.tail[-400:]}'}
    f.update(attempt=attempt, att_tag=tag, label=label, engine=result, t_detect=t_detect, log=log, engine_rc=res.rc)
    u.meta['engine_result'] = result
    if item.mode == 'incremental' and res.rc == 0:
        with open(os.path.join(attempt, f'{item.cell}.log'), 'a') as out_log, open(log) as inc:
            shutil.copyfileobj(inc, out_log)
    if res.rc == 2 or result.get('status') == 'refused':
        raise Refusal(f'engine refused: {result.get("reason", res.reason)[:600]}')
    if res.rc != 0 or result.get('status') != 'done':
        raise RuntimeError(f'engine error rc {res.rc}: {str(result.get("reason") or res.tail[-600:])[:800]}')
    n_thr = open(log, errors='replace').read().count('Thread raised exception')
    if n_thr:          # PyBDSF island-fit exceptions inside its worker pool: published, recorded (p3s5_search_fleet.sh:214-215)
        u.note(f'{n_thr} PyBDSF island-fit exception(s) in the log (published; recorded)')
    after = len(_image_rows(attempt, label, item.stokes))
    f['new_image_rows'] = after - before
    if after == before:
        why = ''
        for line in reversed(open(log, errors='replace').read().splitlines()):
            if 'P3-POLICY-SKIP' in line or 'INCREMENTAL - nothing admitted' in line:
                why = line.strip()
                break
        raise NoInput(f'{item.cell} {item.stokes}: the engine searched nothing ({why or "no image row written"})')


def _fit_frame(job):
    """One searched frame (memory-mapped): LST at MJD-MID, every detection fitted (deep_fit_positions.fit_frame with the core
    fitter)."""
    import math
    import warnings
    warnings.simplefilter('ignore')
    import numpy as np
    import astropy.units as au
    from astropy.io import fits
    from astropy.time import Time
    from astropy.wcs import WCS
    from exopipe.phase3.core import fitting as F
    path, rows, tier, lon = job
    out = []
    with fits.open(path, memmap=True) as hd:
        h = hd[0].header
        d = np.squeeze(hd[0].data)
        w = WCS(h).celestial
        lstd = Time(h['MJD-MID'], format='mjd').sidereal_time('apparent', longitude=lon * au.deg).deg
        for r in rows:
            ra, dec, snr = float(r['ra']), float(r['dec']), float(r['snr'])
            v, s = F.forced_value_rms(d, w, ra, dec)
            o = F.detection_position(d, w, h, ra, dec, v, s, snr, r['band'], r['stokes'], lstd, tier=tier)
            fpx = ''
            if o['pos_kind'] in ('gaussian', 'template'):
                px, py = [float(t) for t in w.all_world2pix(float(o['pos_ra']), float(o['pos_dec']), 0)]
                fpx = f'{math.hypot(px - round(px), py - round(py)):.4f}'
            out.append((r['cand_id'], o, fpx))
    h1 = hashlib.sha1()
    with open(path, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 22), b''):
            h1.update(b)
    return path, h1.hexdigest(), out


def fit_attempt(attempt, label, out_csv, tier='deep', nproc=None, lon=None):
    """The fitted position of every catalogue row of `attempt` on its searched frame -> out_csv (cand_id + DET_COLS + frame,
    frame_sha1, fit_frac_px) and <out_csv dir>/fit_positions_summary.json. Refuses (record.Refusal) a catalogue that already
    carries pos_kind or a row without its searched frame (deep_fit_positions.py gates). Returns the summary dict."""
    from multiprocessing import Pool
    from exopipe.phase3.core.fitting import DET_COLS
    from exopipe.phase3.core.record import Refusal
    from exopipe.phase3.lib.search.tests import frame_path
    c = _cfg()
    nproc = nproc or c.search_fit.NPROC
    lon = c.search_fit.LST_LON_DEG if lon is None else lon
    t0 = time.time()
    cat = os.path.join(attempt, 'transients', f'Run_{label}', 'master_catalog.csv')
    rows, cols = [], []
    if os.path.exists(cat):
        with open(cat, newline='') as fh:
            rd = csv.DictReader(fh)
            cols, rows = list(rd.fieldnames or []), list(rd)
    if 'pos_kind' in cols:
        raise Refusal(f'{cat} already carries pos_kind - the fit stage fits the search record (peak positions) only')
    byf, miss = {}, []
    for r in rows:
        p = frame_path(attempt, label, r['lst'], r['band'], r['date'], r['stokes'])
        if not os.path.exists(p):
            miss.append((r['cand_id'], p))
            continue
        byf.setdefault(p, []).append(r)
    if miss:
        raise Refusal(f'{len(miss)} detection(s) without their searched frame, e.g. {miss[0][0]} {miss[0][1]}')
    jobs = sorted(((p, rs, tier, lon) for p, rs in byf.items()), key=lambda j: -len(j[1]))
    res, shas = {}, {}
    if jobs:
        with Pool(max(1, min(nproc, len(jobs)))) as pool:
            for path, sha, out in pool.imap_unordered(_fit_frame, jobs, chunksize=1):
                shas[path] = sha
                for cid, o, fpx in out:
                    res[cid] = (o, fpx, path)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv + '.part', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['cand_id'] + list(DET_COLS) + ['frame', 'frame_sha1', 'fit_frac_px'])
        for r in rows:
            o, fpx, p = res[r['cand_id']]
            w.writerow([r['cand_id']] + [o.get(k, '') for k in DET_COLS] + [os.path.relpath(p, attempt), shas[p], fpx])
    os.replace(out_csv + '.part', out_csv)
    per_frame = []
    for p, rs in sorted(byf.items()):
        npk = sum(1 for r in rs if res[r['cand_id']][0]['pos_kind'] == 'peak')
        per_frame.append({'frame': os.path.relpath(p, attempt), 'rows': len(rs), 'peak': npk,
                          'peak_frac': round(npk / len(rs), 4)})
    kinds = {}
    for r in rows:
        k = f"{r['stokes']}:{res[r['cand_id']][0]['pos_kind']}"
        kinds[k] = kinds.get(k, 0) + 1
    fp = [float(v[1]) for v in res.values() if v[1] not in ('', None)]
    on_ctr = sum(1 for x in fp if x < c.search_fit.ON_CENTRE_PX)
    offs = sorted(float(v[0]['pos_offset_arcmin']) for v in res.values() if v[0].get('pos_offset_arcmin') not in ('', None))
    npk = sum(x['peak'] for x in per_frame)
    summ = dict(attempt=attempt, out=out_csv, rows=len(rows), frames=len(byf), kinds=kinds, fitted_rows=len(fp),
                fitted_on_pixel_centre=on_ctr, peak_fallback=npk, peak_fallback_frac=round(npk / len(rows), 4) if rows else 0.0,
                median_offset_arcmin=offs[len(offs) // 2] if offs else None, per_frame=per_frame,
                fitter='core.fitting.detection_position', tier=tier, lst_lon_deg=lon,
                template_orientation=c.fitting.TEMPLATE_BEAM_ORIENTATION, seconds=round(time.time() - t0, 1))
    with open(os.path.join(os.path.dirname(out_csv), 'fit_positions_summary.json'), 'w') as fh:
        json.dump(summ, fh, indent=1)
    return summ


def stage_fit(u, item, ctx, f, out_csv):
    c = _cfg()
    if not c.search_fit.ON:
        u.note('fit stage off (search_fit.ON False)')
        return None
    summ = fit_attempt(f['attempt'], f['label'], out_csv, tier=TIER)
    u.check('search.fit_peak_fallback', True, summ['peak_fallback_frac'],
            f"{summ['peak_fallback']} of {summ['rows']} detections kept at the peak pixel; worst frame "
            + (max(summ['per_frame'], key=lambda x: x['peak_frac'])['frame'] if summ['per_frame'] else '-'),
            cls='detection_positions', path=out_csv)
    u.meta['fit_summary'] = {k: summ[k] for k in ('rows', 'frames', 'kinds', 'peak_fallback', 'peak_fallback_frac',
                                                   'fitted_on_pixel_centre', 'fitted_rows', 'seconds')}
    return summ


def unaccounted_epochs(item, f):
    """Admitted epochs with neither an image row nor a refusal row of this Stokes (search.UNACCOUNTED_EPOCHS)."""
    from exopipe.phase3.lib.search.tests import refused_dates
    run_dir = os.path.join(f['attempt'], 'transients', f'Run_{f["label"]}')
    have = set(f['dates_v'] if item.stokes == 'V' else f['dates_i'])
    cand = set(f['admitted']) & have
    if item.mode == 'incremental':
        cand &= set(item.dates)
    img = {r['date'] for r in _image_rows(f['attempt'], f['label'], item.stokes)}
    mine, extraction, _both = refused_dates(run_dir, item.stokes)
    return sorted(cand - img - mine - extraction)


def outputs_of(item, ctx, f, fit_csv=None):
    """[(path, product class, test context)] of a staged / replayed attempt."""
    att, label, cell, S = f['attempt'], f['label'], item.cell, item.stokes
    rd = os.path.join(att, 'transients', f'Run_{label}')
    base_ctx = dict(stokes=S, cell=cell, tier=TIER)
    out = []
    for name, cls in (('master_catalog.csv', 'detections'), ('v_deep_ledger.csv', 'gate_ledger'),
                      ('v_deep_images.csv', 'image_ledger')):
        p = os.path.join(rd, name)
        if os.path.exists(p):
            out.append((p, cls, base_ctx))
    for p in sorted(glob.glob(os.path.join(rd, 'refusals', f'{cell}_*_excluded.csv'))):
        out.append((p, 'refusal_record', {}))
    if os.path.isdir(os.path.join(rd, 'candidates')):
        out.append((os.path.join(rd, 'candidates'), 'candidate_stamps', {}))
    ss = os.path.join(rd, 'sigstack', f'{cell}_{S}_sigstack.fits')
    if os.path.exists(ss):
        out.append((ss, 'noise_map', {}))
    for p in sorted(glob.glob(os.path.join(rd, 'sigstack', f'{cell}_{S}_sigstack_*.fits'))):   # D-ADMIT: an admitted night's
        out.append((p, 'noise_map', {}))                                                    # own stack (eighth review)
    sub = 'iresid' if S == 'I' else 'vresid'
    for p in sorted(glob.glob(os.path.join(att, f'Run_{label}', sub, cell, f'*_{S}_redo_resid_*.fits'))):
        out.append((p, 'search_frame', {'stokes': S, 'cell': cell, 'tier': TIER}))     # plain frames and pbcorr siblings: one
                                                                                      # class, owned by frames (lib/frames/tests.py)
    for p in sorted(glob.glob(os.path.join(att, f'pbcorr_missing_{item.lst}_{item.scope}.csv'))):
        out.append((p, 'frame_refusals', {}))
    if fit_csv and os.path.exists(fit_csv):
        out.append((fit_csv, 'detection_positions', dict(tier=TIER, attempt=att)))
    return out


def side_outputs(item, ctx, f):
    """Products outside the attempt: the cell's parent catalogue and this attempt's v110 verdicts (flow work dir), and the archive
    copies detect wrote (window: the mirror). [(path, class, ctx, publish-as)]"""
    out = []
    cw = f.get('cell_work') or os.path.join(ctx['flow_root'], item.cell)
    g = os.path.join(cw, 'catdir', f'master_R075_{item.cell}.gaul.csv')
    # an incremental attempt publishes its run tree and logs only (p3s4_incremental_fleet.sh:47): never the parent catalogue or
    # the v110 verdicts - the incremental v110 file holds only the appended epochs' survivors and would replace the cell's
    incremental = item.mode == 'incremental'
    if os.path.exists(g) and not incremental:
        out.append((g, 'master_catalog', {}, os.path.join(ctx['out'], 'parent_catalogs', os.path.basename(g))))
    v = os.path.join(cw, 'v110_out', 'V_top100_fp_gate_p3.csv')
    if item.stokes == 'V' and not incremental and os.path.exists(v) and os.path.getmtime(v) >= f.get('t_detect', 0):
        cat = os.path.join(f['attempt'], 'transients', f'Run_{f["label"]}', 'master_catalog.csv')
        out.append((v, 'fp_gate_verdicts', {'catalog': cat}, os.path.join(ctx['out'], 'v110_products', item.cell, os.path.basename(v))))
    # archive copies written by this detect (PRODNAME of each attempt frame; mtime >= detect start)
    if f.get('t_detect'):
        from exopipe.phase3.core.record import fits_primary_cards
        c = _cfg()
        sub = 'iresid' if item.stokes == 'I' else 'vresid'
        for p in sorted(glob.glob(os.path.join(f['attempt'], f'Run_{f["label"]}', sub, item.cell, '*_redo_resid_*.fits'))):
            name = str(fits_primary_cards(p).get('PRODNAME', '') or '')
            d = os.path.basename(p)[:10]
            if not name:
                continue
            dst = os.path.join(ctx['archive_root'], item.lst, d, c.paths.ARCHIVE_RUN, item.scope, item.stokes, 'deep',
                               name if name.endswith('.fits') else name + '.fits')
            if os.path.exists(dst) and os.path.getmtime(dst) >= f['t_detect']:
                out.append((dst, 'search_frame', {'stokes': item.stokes, 'copy_of': p}, dst))
    return out


def _tests_registered(cls):
    from exopipe.phase3.core import validate
    try:
        validate.tests_for(cls)
        return True
    except KeyError:
        return False


def stage_verify(u, item, ctx, f, outs, sides, raise_on_fail=True):
    """Content tests of every output on the staged attempt + the stage verifier. Returns {path: [results]}; refuses before
    publish when anything fails (a --waive'd test does not block); the failing results are recorded first. With
    raise_on_fail=False returns (results, failures) and records nothing of the content tests (the replay records them)."""
    from exopipe.phase3.core import validate
    from exopipe.phase3.core.record import Refusal
    c = _cfg()
    results = {}
    for p, cls, tctx in [(a, b, d) for a, b, d in outs] + [(a, b, d) for a, b, d, _ in sides]:
        if not _tests_registered(cls):
            results[p] = None
            continue
        results[p] = validate.run(cls, p, **tctx)
    failed = []
    for p, rs in results.items():
        for r in rs or ():
            if not r['passed'] and not u.run.waived(r['name']):
                failed.append(f"{r['name']} {os.path.basename(p)}: {str(r['detail'])[:160]}")
    # stage verifier (verify_p3s3_stage.py conditions 1, 4, 5)
    sv = []
    if f.get('engine') is not None:
        ok = f['engine'].get('status') == 'done'
        sv.append(('search.engine_completed', ok, f['engine'].get('n_written'), str(f['engine'].get('reason', ''))[:200]))
    if item.mode != 'replay':
        g = os.path.join(f.get('cell_work') or '', 'catdir', f'master_R075_{item.cell}.gaul.csv')
        sv.append(('search.parent_catalogue_present', os.path.exists(g), None, g))
        lg = f.get('log')
        rdir = os.path.join(f['attempt'], 'transients', f'Run_{f["label"]}', 'refusals')
        if lg and os.path.exists(lg):
            said = 'REFUSED from search' in open(lg, errors='replace').read()
            have = bool(glob.glob(os.path.join(rdir, f'{item.cell}_*excluded.csv')))
            sv.append(('search.refusal_ledgers_present', (not said) or have, int(said), 'log reports refusals' if said else ''))
        un = unaccounted_epochs(item, f)
        mode = c.search.UNACCOUNTED_EPOCHS
        sv.append(('search.unaccounted_epochs', mode == 'record' or not un, len(un),
                   (f'{len(un)} admitted epoch(s) with no image row and no refusal row: {un[:6]}' if un else 'none')
                   + (' (recorded, search.UNACCOUNTED_EPOCHS=record)' if un and mode == 'record' else '')))
    for name, ok, val, det in sv:
        u.check(name, ok, val, det)
        if not ok and not u.run.waived(name):
            failed.append(f'{name}: {det}')
    if not raise_on_fail:
        return results, failed
    if failed:
        types = {p: cls for p, cls, _t in outs}
        types.update({p: cls for p, cls, _t, _f in sides})
        for p, rs in results.items():
            for r in rs or ():
                u.check(r['name'], r['passed'], r['value'], str(r['detail'])[:300], cls=types.get(p), path=p)
        raise Refusal('verify: ' + '; '.join(failed[:6]) + (f' (+{len(failed) - 6} more)' if len(failed) > 6 else ''))
    return results


def _copy_into(src, dst):
    if os.path.isdir(src):
        shutil.copytree(src, dst, symlinks=True)
    else:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst + '.part')
        os.replace(dst + '.part', dst)


def stage_publish(u, item, ctx, f):
    """Copy the attempt and its side products into --out in today's collect layout; append the register row."""
    att, label = f['attempt'], f['label']
    dest = os.path.join(ctx['out'], 'attempts', os.path.basename(att))
    if os.path.exists(dest):
        raise RuntimeError(f'{dest} exists - an attempt is published once')
    os.makedirs(dest + '.part')
    from exopipe.phase3.lib.search import removal_rules as RR
    names = [f'Run_{label}', 'transients', f'{item.cell}.log', f'{item.cell}.base.log', f'{item.cell}.inc.log',
             'fit_positions.csv', 'fit_positions_summary.json', f'pbcorr_missing_{item.lst}_{item.scope}.csv',
             RR.RULES_RECORD]                    # B-02: the rules the attempt was searched under (an append reads its base's)
    for n in names:
        if os.path.exists(os.path.join(att, n)):
            _copy_into(os.path.join(att, n), os.path.join(dest + '.part', n))
    os.replace(dest + '.part', dest)
    return dest


def register_row(out, item, att_tag):
    p = os.path.join(out, f'published_{item.stokes}.csv')
    with open(p + '.lock', 'w') as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        with open(p, 'a') as fh:
            fh.write(f'{item.cell},{_host()},{att_tag}\n')
        fcntl.flock(lk, fcntl.LOCK_UN)
    return p


def record_outputs(u, pairs, verified):
    """pairs = [(staged path, final path, class, ctx)]. A file verified on the staged attempt is recorded at its final path with
    those results (the copy is byte-identical); anything else is tested now."""
    for staged, final, cls, tctx in pairs:
        if not _tests_registered(cls):
            u.add_output(final, cls, test=False)
            u.note(f'{cls}: no content test registered yet ({os.path.basename(final)} recorded untested)')
            continue
        if staged in verified and verified[staged] is not None:
            u.add_output(final, cls, test=False)
            for r in verified[staged]:
                u.check(r['name'], r['passed'], r['value'], r['detail'], cls=cls, path=final)
        else:
            u.add_output(final, cls, **tctx)


def _legacy_code_changed(rec):
    out = []
    for path, h in ((rec or {}).get('meta') or {}).get('legacy_code', {}).items():
        try:
            if hashlib.sha1(open(path, 'rb').read()).hexdigest()[:12] != h:
                out.append(path)
        except OSError:
            out.append(path)
    return out


# =====================================================================================================================
# the unit
# =====================================================================================================================

def run_unit(u, item, stages, ctx):
    from exopipe.phase3.core.record import Refusal
    c = _cfg()
    f = preflight(item, ctx, u=u, fix=not ctx['dry_run'])
    u.meta.update(mode=item.mode, notes_preflight=f['notes'], gates=[list(g) for g in f['gates']])
    for n in f['notes']:
        u.note(n)
    if ctx['dry_run'] or stages == ['preflight']:
        print_plan(item, f, stages, ctx)
        return f
    if item.mode == 'replay':
        bad = [s for s in stages if s not in ('preflight', 'fit', 'verify')]
        if bad:
            raise Refusal(f'replay runs fit and verify only (not {bad})')
        rdir = os.path.join(ctx['out'], 'replay', os.path.basename(f['attempt'].rstrip('/')))
        fit_csv = os.path.join(rdir, 'fit_positions.csv')
        if 'fit' in stages and stage_fit(u, item, ctx, f, fit_csv) is not None:
            u.add_output(fit_csv, 'detection_positions', tier=TIER, attempt=f['attempt'])
        if 'verify' in stages:
            outs = [o for o in outputs_of(item, ctx, f) if o[1] != 'detection_positions']
            g = os.path.join(os.path.dirname(os.path.dirname(f['attempt'].rstrip('/'))), 'parent_catalogs',
                             f'master_R075_{item.cell}.gaul.csv')
            sides = [(g, 'master_catalog', {}, g)] if os.path.exists(g) else []
            res, failed = stage_verify(u, item, ctx, f, outs, sides, raise_on_fail=False)
            rep = os.path.join(rdir, 'verify_report.csv')
            os.makedirs(rdir, exist_ok=True)
            with open(rep + '.part', 'w', newline='') as fh:
                w = csv.writer(fh)
                w.writerow(['test', 'class', 'path', 'passed', 'value', 'detail'])
                for p, cls, _t in outs + [(a, b, d) for a, b, d, _x in sides]:
                    for r in res.get(p) or [{'name': f'{cls}.(no test registered)', 'passed': True, 'value': '', 'detail': ''}]:
                        w.writerow([r['name'], cls, p, int(bool(r['passed'])), r['value'], str(r['detail'])[:300]])
                        if res.get(p):
                            u.check(r['name'], r['passed'], r['value'], str(r['detail'])[:300], cls=cls, path=p)
            os.replace(rep + '.part', rep)
            u.add_output(rep, 'csv_table', required_columns=('test', 'class', 'path', 'passed'))
            if not sides:
                u.note(f'parent catalogue not found beside the attempt ({g}); not tested')
            if failed:
                raise Refusal('replay verify: ' + '; '.join(failed[:6]) + (f' (+{len(failed) - 6} more)' if len(failed) > 6 else ''))
        return f
    add_inputs(u, item, f)
    if 'quarantine' in stages:
        if item.mode == 'incremental':
            u.note('quarantine skipped: an incremental attempt appends to its base (p3s4_incremental_fleet.sh does not quarantine)')
        else:
            stage_quarantine(u, item, ctx, f)
    if 'detect' not in stages:
        raise Refusal('stages without detect operate on an existing attempt: give --replay-attempt DIR|published '
                      '(read-only replay of fit/verify)')
    stage_detect(u, item, ctx, f)
    after_detect(u, item, ctx, f, stages)
    return f


def after_detect(u, item, ctx, f, stages):
    """fit, verify, publish and the output records of a unit whose detect stage produced the staged attempt f['attempt']."""
    published_name = None
    if 'publish' in stages:
        published_name = os.path.join(ctx['out'], 'attempts', os.path.basename(f['attempt']))
        u.plan(os.path.join(published_name, 'transients', f'Run_{f["label"]}', 'v_deep_images.csv'), 'image_ledger')
    fit_csv = os.path.join(f['attempt'], 'fit_positions.csv')
    if 'fit' in stages:
        if stage_fit(u, item, ctx, f, fit_csv) is not None and published_name:      # None: search_fit.ON False (recorded)
            u.plan(os.path.join(published_name, 'fit_positions.csv'), 'detection_positions')
    outs = outputs_of(item, ctx, f, fit_csv=fit_csv if 'fit' in stages else None)
    sides = side_outputs(item, ctx, f)
    verified = stage_verify(u, item, ctx, f, outs, sides) if 'verify' in stages else {}
    if 'publish' in stages:
        dest = stage_publish(u, item, ctx, f)
        pairs = [(p, os.path.join(dest, os.path.relpath(p, f['attempt'])), cls, dict(tctx, attempt=dest)
                  if cls == 'detection_positions' else tctx) for p, cls, tctx in outs]
        for p, cls, tctx, final in sides:
            if final != p:
                _copy_into(p, final)
                if p.endswith('.gaul.csv') and os.path.exists(p + '.gen.json'):
                    _copy_into(p + '.gen.json', final + '.gen.json')
                if cls == 'fp_gate_verdicts':     # the whole v110_out directory, as the fleet (p3s5_search_fleet.sh:235-236)
                    for q in glob.glob(os.path.join(os.path.dirname(p), '*')):
                        if os.path.isfile(q) and q != p:
                            _copy_into(q, os.path.join(os.path.dirname(final), os.path.basename(q)))
                    tctx = dict(tctx, catalog=os.path.join(dest, 'transients', f'Run_{f["label"]}', 'master_catalog.csv'))
            pairs.append((p, final, cls, tctx))
        record_outputs(u, pairs, verified)
        reg = register_row(ctx['out'], item, f['att_tag'])
        u.meta.update(attempt=dest, register=reg)
        ctx.setdefault('published', {}).setdefault(item.cell, set()).add(item.stokes)
    else:
        record_outputs(u, [(p, p, cls, tctx) for p, cls, tctx in outs] + [(p, p, cls, tctx) for p, cls, tctx, _ in sides],
                       verified)
        u.meta['attempt'] = f['attempt']
        u.note('not published (publish not among the stages): outputs recorded in node-local scratch')
    return f


def hygiene(ctx, cell):
    c = _cfg()
    if ctx.get('keep_work') or c.search_run.HYGIENE != 'both-stokes-done':
        return
    if ctx.get('published', {}).get(cell, set()) >= {'I', 'V'}:
        shutil.rmtree(os.path.join(ctx['flow_root'], cell), ignore_errors=True)
        for d in glob.glob(os.path.join(ctx['attempts'], f'{cell}_[IV]_*')):
            shutil.rmtree(d, ignore_errors=True)


def run(args, run):
    """Deep tier entry (steps/search.run dispatches here)."""
    from exopipe.phase3.core import fitsio
    from exopipe.phase3.core.record import NoInput, Refusal
    from exopipe.phase3.steps.search import resolve_stages
    c = _cfg()
    t0 = time.time()
    if getattr(args, 'inject', None):
        raise Refusal('search --tier deep --inject: no production injection-through-search path exists for the deep tier '
                      '(spec Q9; the deep K calibrations inject offline) - refused (gate G4)')
    stages = resolve_stages(args, c.search.STAGES)
    out = os.path.abspath(run.out_root)
    if not c.paths.PRODUCTION_WRITES:
        hit = production_tree(out)
        if hit:
            run.no_records = True              # nothing at all under a production tree (not even the run summary)
            raise Refusal(f'--out {out} is under the production tree {hit} and paths.PRODUCTION_WRITES is False (gate G3); '
                          f'pass --out to your own root')
    dev = comparison_deviations()
    if dev and production_tree(out):
        raise Refusal(f'COMPARISON key(s) off their production value {dev} refused into the production root {out}')
    if c.paths.PRODUCTION_WRITES:
        work = c.search_run.PRODUCTION_WORK_ROOT
        attempts, flow_root, tmp = c.search_run.PRODUCTION_ATTEMPT_ROOT, work, os.path.join(work, '_tmp')
        archive_root = c.paths.PHASE3_ARCHIVE_ROOT
    else:
        wroot = fitsio.scratch_root(args.work)
        work = os.path.join(wroot, c.search_run.WORK_SUBDIR)
        hit = production_tree(work)
        if hit:
            raise Refusal(f'work root {work} is under the production tree {hit} (gate G3)')
        attempts, flow_root, tmp = (os.path.join(work, 'attempts'), os.path.join(work, 'flowroot'), os.path.join(work, 'tmp'))
        archive_root = os.path.join(out, c.search_inputs.ARCHIVE_MIRROR_SUBDIR)
    items = plan_items(args, stages)
    if not items:
        raise NoInput('no deep unit selected')
    mism = literal_mismatches()
    if mism:
        raise Refusal('config is untrue about the in-place engine (engine literals): ' + '; '.join(
            f'{k} = {a!r} but {w} = {b!r}' for k, a, b, w in mism[:5]))
    work_fs = os.path.abspath(work)
    while not os.path.exists(work_fs):              # the filesystem the work root will live on (nearest existing ancestor)
        work_fs = os.path.dirname(work_fs)
    ctx = dict(out=out, work_fs=work_fs,
               attempts=attempts, flow_root=flow_root, tmp=tmp, archive_root=archive_root, dry_run=run.dry_run,
               flow_from=getattr(args, 'deep_flow_from', None), keep_work=getattr(args, 'deep_keep_work', False),
               label=args.run_label or c.search.RUN_LABEL, roots=roots_table(), t0=t0)
    if not run.dry_run:
        for d in (attempts, flow_root, tmp):
            os.makedirs(d, exist_ok=True)
    print(f'search deep: {len(items)} unit(s); stages {",".join(stages)}; out {out}; work {work}; label {ctx["label"]}; '
          f'engine literals {len(engine_literals())} = config', flush=True)
    refused = 0
    for it in items:
        if run.resume and not run.fresh and it.mode != 'replay':
            try:
                pf = preflight(it, ctx, fix=False, resume_probe=True)
                small, big = input_paths(it, pf)
                # bulk: the input SET must be the recorded one (a new epoch product, LOO or V master breaks it: D6); an
                # incremental unit's record also lists its base attempt's tables, re-fingerprinted by resume_check itself
                ok, why = run.resume_check(it.name, inputs=None if it.mode == 'incremental' else small + big,
                                           admitted=pf['admitted'])
                if ok:
                    from exopipe.phase3.core import record
                    ch = _legacy_code_changed(record.load_record(run.out_root, run.step, it.name))
                    if ch:
                        ok, why = False, f'in-place code changed: {ch[:3]}'
            except Exception as e:      # any gate or error: the unit runs and records it
                ok, why = False, f'preflight: {e}'
            print(f'resume {it.name}: {"skipped" if ok else "not skipped"} ({why})', flush=True)
            if ok:
                u = run.unit(it.name, tier=TIER, cell=it.cell, stokes=it.stokes)
                u.status, u.reason = 'skipped-valid', why
                ctx.setdefault('published', {}).setdefault(it.cell, set()).add(it.stokes)
                continue
        with run.unit(it.name, tier=TIER, cell=it.cell, stokes=it.stokes, mode=it.mode) as u:
            with cell_lock(ctx, it):
                run_unit(u, it, stages, ctx)
        if (run.dry_run or stages == ['preflight']) and u.status == 'no-input' and u.reason.startswith('no output written'):
            u.status, u.reason = 'planned', 'preflight passed (plan only)'    # records are not written for a plan
            continue
        if u.status == 'refused':
            refused += 1
            print(f'REFUSED {it.name}: {u.reason}', flush=True)
        elif u.status in ('error', 'no-input'):
            print(f'{u.status.upper()} {it.name}: {u.reason[:300]}', flush=True)
        if not run.dry_run and (it is items[-1] or items[items.index(it) + 1].cell != it.cell):
            hygiene(ctx, it.cell)
    if run.dry_run or stages == ['preflight']:      # plans: 'planned' counts as done; errors and refusals keep their exit
        st = [u.status for u in run.units]
        if 'error' in st:
            return 1
        if refused:
            return 2
        return 3 if st and all(s == 'no-input' for s in st) else 0
    return None


@contextlib.contextmanager
def cell_lock(ctx, item):
    """One search of a cell at a time on this work root: the flow cache of a cell is never dropped, counted or written while
    another search of the cell runs (p3s5_search_fleet.sh:94,169 wait for any run_deep_cell of the cell; p3s4_incremental_
    fleet.sh:58; dc7910b). An exclusive flock on <flow root>/.<cell>.search.lock, held for the whole unit; a second unit waits.
    Plans (--dry-run) and read-only replays take no lock."""
    if ctx['dry_run'] or item.mode == 'replay':
        yield
        return
    p = os.path.join(ctx['flow_root'], f'.{item.cell}.search.lock')
    with open(p, 'a') as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f'{item.name}: another search of {item.cell} holds {p} - waiting for it to finish', flush=True)
            fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


# =====================================================================================================================
# the engine child: python -m exopipe.phase3.lib.search.deep_step --child <spec.json>   (started by core.legacy.run)
# =====================================================================================================================

def _child(spec_path):
    import traceback
    spec = json.load(open(spec_path))
    sys.argv = ['search']                      # production: sys.argv=['p3s3'] before the import (lab imports reset argv)
    out, rc = {'status': 'error', 'reason': 'not started'}, 1
    from exopipe.phase3.core.record import Refusal
    try:
        from exopipe.phase3.core import env as E
        E.apply()
        E.legacy()                             # legacy roots on sys.path + the old-name finder (MOVE_LOG decision 1)
        from exopipe.phase3.lib.search.deep_search import run_deep_cell
        kw = dict(label=spec['label'], stokes=spec['stokes'], cadence=spec['cadence'])
        if spec.get('included_dates'):
            kw['included_dates'] = spec['included_dates']
        n = run_deep_cell(spec['lst'], spec['scope'], **kw)
        print(f"P3S3-CELL-DONE {spec['cell']}", flush=True)     # the sentinel today's readers look for (verify_p3s3_stage.py)
        out, rc = {'status': 'done', 'n_written': int(n or 0)}, 0
    except Refusal as e:                       # a named gate of the engine (B-30 fallback sites included): refused, exit 2
        traceback.print_exc()
        out, rc = {'status': 'refused', 'reason': str(e)}, 2
    except RuntimeError as e:
        msg = str(e)
        traceback.print_exc()
        if any(m in msg for m in spec.get('markers', ())):
            out, rc = {'status': 'refused', 'reason': msg}, 2
        else:
            out, rc = {'status': 'error', 'reason': msg}, 1
    except BaseException as e:                 # noqa: B902 - every other outcome is an error, recorded with its text
        traceback.print_exc()
        out, rc = {'status': 'error', 'reason': f'{type(e).__name__}: {e}'}, 1
    finally:
        tmp = spec['result'] + '.part'
        with open(tmp, 'w') as fh:
            json.dump(out, fh)
        os.replace(tmp, spec['result'])
        sys.stdout.flush()
    return rc


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--child':
        sys.exit(_child(sys.argv[2]))
    sys.exit('usage: python -m exopipe.phase3.lib.search.deep_step --child <spec.json> (started by exopipe3 search)')
