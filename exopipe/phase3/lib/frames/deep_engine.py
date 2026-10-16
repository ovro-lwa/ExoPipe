"""The in-place deep frame engine: one fresh child per unit running run_deep_cell(frames_only=True) (spec 7.2).

Engine: ``exopipe.phase3.lib.search.deep_search.run_deep_cell`` (search's module; moved by pure git mv from
exopipe/phase3/phase3_deep.py). Decision of the refactor chat (2026-10-13, crosscheck group 14): `search` (detect) stays
the deep frame producer in the window; `frames --tier deep` is a frame-only rebuild through the ``frames_only`` keyword,
which the search owner adds as one reviewed commit, byte-neutral for the default call (lib/frames/deep_seam.py holds the
proposed edit and the proof). While the keyword is absent every residuals unit REFUSES with gate 'engine-seam'.

Process model (required, spec 7.2): the engine reads its environment at import (MIN_EPOCHS, NPROC_RESID, _STREAM,
_STRIATION_OFF, _REFINE_OFF, SEARCH_TIER_SIGMA, WORKROOT) and mutates globals and os.environ at call time, so every unit
runs in its own child, started with ``core.legacy.run`` (the one child runner: site interpreter, inherited P3_* cleared,
parity guard, child code fingerprint into the unit record): ``python -u -m exopipe.phase3.lib.frames.deep_child SPEC.json``
with cwd = this worktree and the environment built here from config (``child_env``). The child's output is the
attempt's ``<cell>.log`` (as production). Mapping (spec 7.2): an engine RuntimeError whose text contains 'REFUS' ->
``record.Refusal`` (exit 2); the frames_only policy-skip result -> ``record.NoInput`` (exit 3); anything else -> error.

Every value the engine takes from the step comes from SEARCH's config sections (the effective keys while search produces
the deep frames; crosscheck 4.3) through ``engine_values`` - the one reader.

Products of one unit (attempt directory A; production layout, no renames):
  A/Run_<label>/{iresid,vresid}/<lst>_<scope>/<date>_<S>_redo_resid[_pbcorr]_<scope>.fits     search_frame
  A/transients/Run_<label>/sigstack/<lst>_<scope>_<S>_sigstack.fits                          noise_map
  A/transients/Run_<label>/refusals/<lst>_<scope>_<gate>_excluded.csv                        refusal_record (search's)
  A/pbcorr_missing_<lst>_<scope>.csv                                                         frame_refusals
  <archive>/<LST>/<night>/<ARCHIVE_RUN>/<scope>/<S>/<cadence>/<PRODNAME>                     search_frame (archive copy)
"""
import json
import os
import re

from exopipe.phase3.core.record import NoInput, Refusal

ENGINE_MODULE = 'exopipe.phase3.lib.search.deep_search'
CHILD_MODULE = 'exopipe.phase3.lib.frames.deep_child'

# Where each engine value comes from: (value name, config KEY, kind). Looked up by key name in the sections of
# _SECTIONS[tier] in order (search's sections first: the effective keys while search produces the deep frames;
# crosscheck 4.3). kind 'env' = passed to the engine child as an environment variable; 'step' = used by the step;
# 'mirror' = an engine literal (frames_residuals mirrors the ones search does not carry; equal values required when both do).
VALUE_KEYS = (
    ('master_gen', 'MASTER_GEN', 'env'), ('allow_old_masters', 'ALLOW_OLD_MASTERS', 'env'),
    ('archive_write', 'ARCHIVE_WRITE', 'env'), ('roots_csv', 'ROOTS_CSV', 'step'),
    ('striation_off_battery', 'STRIATION_OFF_BATTERY', 'env'), ('streaming', 'STREAMING', 'env'),
    ('nproc_resid', 'NPROC_RESID', 'env'), ('min_epochs', 'MIN_EPOCHS', 'env'),
    ('sigstack_min_epochs', 'SIGMA_STACK_MIN_EPOCHS', 'env'), ('sanity_i_x', 'SANITY_X', 'env'),
    ('search_tier_sigma', 'SEARCH_TIER_SIGMA', 'env'), ('beam_gate_ratio', 'BEAM_GATE_RATIO', 'env'),
    ('sanity_min_nights', 'SANITY_MIN_NIGHTS', 'env'), ('search_tier_min', 'SEARCH_TIER_MIN', 'env'),
    ('beam_gate_min_nights', 'BEAM_GATE_MIN_NIGHTS', 'env'),
    ('beam_gate_on', 'BEAM_GATE_ON', 'env'), ('flow_xla_mem_fraction', 'FLOW_XLA_MEM_FRACTION', 'env'),
    ('flow_python', 'FLOW_PYTHON', 'env'), ('flow_lock', 'FLOW_LOCK', 'env'),
    ('run_label', 'RUN_LABEL', 'step'), ('stokes_order', 'STOKES_ORDER', 'step'), ('python', 'ENGINE_PYTHON', 'step'),
    ('unaccounted', 'UNACCOUNTED_EPOCHS', 'step'),
    ('striation_window_beams', 'STRIATION_WINDOW_BEAMS', 'mirror'), ('sigstack_build_min', 'SIGMA_STACK_BUILD_MIN', 'mirror'),
    ('pb_floor', 'PB_FLOOR', 'mirror'), ('rms_box_half_px', 'CENTRAL_RMS_HALF_PX', 'mirror'),
    ('sanity_align_x', 'SANITY_ALIGN_X', 'mirror'),
    ('loo_ncoadd_range', 'LOO_NCOADD_RANGE', 'mirror'), ('sigstack_floor', 'SIGSTACK_FLOOR', 'mirror'),
)
_SECTIONS = {'deep': ('search', 'search_inputs', 'search_detect', 'frames_residuals'),
             'sublst': ('search_sublst', 'search', 'search_inputs', 'search_detect', 'frames_residuals')}
# keys whose override cannot reach a frame (engine literals) - refused by check_overrides, in any section
FRAME_FIXED = tuple(k for _, k, kind in VALUE_KEYS if kind == 'mirror')
# keys a frames run uses; an override of any other search key does not change a frame and is recorded as such
FRAME_KEYS = tuple(k for _, k, _kind in VALUE_KEYS)

# refusal files the frame build writes (the gates, phase3_deep.py:1088-1365) and the search's own admission files
FRAME_REFUSAL_KINDS = ('nightimage', 'epochgate', 'sanity', 'sanity_I', 'searchtier', 'stale_input', 'beam',
                       'datadecisions')   # B-02 step 3c: the nights data_decisions excluded, recorded ahead of the gates
SEARCH_REFUSAL_KINDS = ('sanity_V', 'extraction_I', 'extraction_V', 'daytime')   # daytime: B-02 step 3a (the frame is built)
WHOLE_EPOCH_KINDS = ('nightimage', 'epochgate', 'sanity', 'stale_input', 'beam')   # not datadecisions: those nights are out of
                                                                                   # the owed ones, and verify judges the
                                                                                   # admitted nights by today's data_decisions
_FRAME_RE = re.compile(r'^(?P<d>\d{4}-\d\d-\d\d(?:T\d\d)?)_(?P<s>[IV])_redo_resid(?P<pb>_pbcorr)?_(?P<scope>\w+?)\.fits$')


def _cfg():
    from exopipe.phase3.config import CFG
    return CFG


def worktree_root():
    """The repository root this package runs from (the child's cwd and the head of its PYTHONPATH)."""
    here = os.path.abspath(__file__)
    for _ in range(5):                      # lib/frames/deep_engine.py -> frames -> lib -> phase3 -> exopipe -> root
        here = os.path.dirname(here)
    return here


def engine_source():
    import importlib.util
    spec = importlib.util.find_spec(ENGINE_MODULE)
    if spec is None or not spec.origin:
        raise Refusal(f'engine module {ENGINE_MODULE} not found')
    return spec.origin


def engine_has_frames_only(path=None):
    """True when run_deep_cell takes the ``frames_only`` keyword (read from the source; nothing imported)."""
    import ast
    from exopipe.phase3.lib.frames import deep_seam
    with open(path or engine_source()) as f:
        return deep_seam.has_keyword(ast.parse(f.read()))


def _tiered(v, tier, sec, key):
    """A per-tier dict gives its tier's entry; a scalar is the value of the section's own tier (search, search_inputs,
    search_detect are the deep sections; search_sublst the sub-LST one)."""
    if isinstance(v, dict) and tier in v:
        return v[tier]
    if isinstance(v, dict) and set(v) & {'deep', 'sublst', 'snapshot'}:
        raise Refusal(f'config key {sec}.{key} has no entry for tier {tier}')
    return v


def engine_values(tier='deep', cfg=None):
    """Every value a frames run gives the engine or uses itself, from the effective config keys (module VALUE_KEYS). The
    ONE reader: a missing key refuses here with its name; a mirror carried by two sections must agree."""
    cfg = cfg or _cfg()
    secs = cfg.sections()
    order = _SECTIONS[tier]
    out, src = {}, {}
    for name, key, kind in VALUE_KEYS:
        hits = [(sn, getattr(secs[sn], key)) for sn in order if sn in secs and hasattr(secs[sn], key)]
        if tier == 'deep':
            hits = [(sn, v) for sn, v in hits if sn != 'search_sublst']
        if not hits:
            raise Refusal(f'config key {key} not found in any of {", ".join(order)} (frames reads the deep engine values from '
                          f'search\'s sections while search produces the deep frames; crosscheck 4.3)')
        sn, v = hits[0]
        v = _tiered(v, tier, sn, key)
        if kind == 'mirror' and len(hits) > 1:
            for sn2, v2 in hits[1:]:
                if _tiered(v2, tier, sn2, key) != v:
                    raise Refusal(f'engine literal {key}: {sn}.{key}={v!r} but {sn2}.{key}={v2!r} - one literal, two values')
        out[name], src[name] = v, f'{sn}.{key}'
    # roots.csv: relative to paths.OPS_ROOT (search_inputs.ROOTS_CSV) or absolute
    if not os.path.isabs(str(out['roots_csv'])):
        out['roots_csv'] = os.path.join(cfg.paths.OPS_ROOT, out['roots_csv'])
    me = out['min_epochs']
    out['min_epochs'] = {tier: int(me)} if not isinstance(me, dict) else {k: int(x) for k, x in me.items()}
    out['stokes_order'] = tuple(out['stokes_order'])
    out['loo_ncoadd_range'] = tuple(out['loo_ncoadd_range'])
    # deep engine behaviour that no config key sets (the step never sets the variable): x2 refine on (P3_REFINE_OFF
    # unset), striation on unless STRIATION_OFF_BATTERY (P3_STRIATION_OFF), LOO subtraction (phase3_deep.py:1394-1433)
    if tier == 'deep':
        out['refine_x2'], out['striation'], out['loo'] = True, not bool(out['striation_off_battery']), True
    else:
        out['refine_x2'] = bool(getattr(secs.get('search_sublst'), 'REFINE_X2', False))
        out['striation'] = False if out['striation_off_battery'] else bool(getattr(secs.get('search_sublst'), 'STRIATION',
                                                                                    False))
        out['loo'] = False                                      # 10-minute frames: central master (Gregg 2026-09-08)
    out['archive_root'] = cfg.paths.PHASE3_ARCHIVE_ROOT
    out['archive_run'] = cfg.paths.ARCHIVE_RUN
    out['_sources'] = src
    return out


def check_overrides(cfg=None):
    """Refuse an override of an engine literal (FRAME_FIXED, any section: no environment variable reaches it); return
    notes for overrides of search keys a frames run does not use (recorded, not refused)."""
    cfg = cfg or _cfg()
    notes = []
    for o in cfg.overrides:
        key = o[0]
        sec, _, rest = key.partition('.')
        base = re.split(r'\[', rest, 1)[0]
        if base in FRAME_FIXED and (sec.startswith('search') or sec.startswith('frames')):
            raise Refusal(f'{sec}.{base} is a literal of the in-place engine ({ENGINE_MODULE}) during the refactor window; '
                          f'the override {key} cannot reach a frame - refused')
        if sec.startswith('search') and base not in FRAME_KEYS:
            notes.append(f'override {key} does not change a deep frame (a detection / search key): recorded, not used')
    return notes


class UnitSpec:
    """What one engine call needs. ``tier`` 'deep' (cadence 'deep') or 'sublst' (cadence '10min', the optional sub-LST
    searchframes stage); ``attempt`` = the unit's fresh attempt directory; ``work_root`` = the node-local flow root."""

    FIELDS = ('tier', 'cadence', 'lst', 'scope', 'stokes', 'label', 'included_dates', 'attempt', 'coadd_root',
              'archive_root', 'archive_run', 'archive_copy', 'work_root', 'tenmin_root')

    def __init__(self, **kw):
        unknown = set(kw) - set(self.FIELDS)
        if unknown:
            raise TypeError(f'unknown UnitSpec fields {sorted(unknown)}')
        for k in self.FIELDS:
            setattr(self, k, kw.get(k))
        self.cadence = self.cadence or ('10min' if self.tier == 'sublst' else 'deep')
        self.included_dates = list(self.included_dates or [])

    @property
    def cell(self):
        return f'{self.lst}_{self.scope}'

    def to_dict(self):
        return {k: getattr(self, k) for k in self.FIELDS}

    # production layout of the attempt (phase3_deep.py:1012-1018, 1613-1616, 1727-1730; p3s5_search_fleet.sh:196-197)
    def resid_dir(self, stokes):
        return os.path.join(self.attempt, f'Run_{self.label}', 'iresid' if stokes == 'I' else 'vresid', self.cell)

    def run_root(self):
        return os.path.join(self.attempt, 'transients', f'Run_{self.label}')

    def frame_path(self, date, stokes, pbcorr=False):
        return os.path.join(self.resid_dir(stokes),
                            f'{date}_{stokes}_redo_resid{"_pbcorr" if pbcorr else ""}_{self.scope}.fits')

    def sigstack_path(self, stokes):
        return os.path.join(self.run_root(), 'sigstack', f'{self.cell}_{stokes}_sigstack.fits')

    def refusal_path(self, kind):
        return os.path.join(self.run_root(), 'refusals', f'{self.cell}_{kind}_excluded.csv')

    def pbcorr_missing_path(self):
        return os.path.join(self.attempt, f'pbcorr_missing_{self.cell}.csv')      # phase3_deep.py:803 (P3_STAGING_ROOT)

    def log_path(self):
        return os.path.join(self.attempt, f'{self.cell}.log')                     # p3s5_search_fleet.sh:200


def child_env(spec, cfg=None):
    """The variables this step sets for the child (spec 7.2 table; every value from config / the site file). core.legacy
    clears inherited P3_* and adds NUMPY_MADVISE_HUGEPAGE and EXOPIPE3_SITE; the only DEEP_* variables the engine reads
    (DEEP_NPROC_RESID, DEEP_XLA_MEMFRAC) are always set here."""
    cfg = cfg or _cfg()
    v = engine_values(spec.tier, cfg)
    s = {'P3_COADD_ROOT': spec.coadd_root}
    if spec.cadence == '10min':
        if not spec.tenmin_root:
            raise Refusal('a 10-minute unit needs its tree root (UnitSpec.tenmin_root = P3_TENMIN_ROOT)')
        s['P3_TENMIN_ROOT'] = spec.tenmin_root
    s['P3_FLOW_ROOT'] = spec.work_root
    s['P3_STAGING_ROOT'] = spec.attempt
    s['P3_IRESID_ROOT'] = os.path.join(spec.attempt, f'Run_{spec.label}', 'iresid')
    s['P3_VRESID_ROOT'] = os.path.join(spec.attempt, f'Run_{spec.label}', 'vresid')
    s['P3_MASTER_GEN'] = str(v['master_gen'])
    if v['allow_old_masters']:
        s['P3_ALLOW_OLD_MASTERS'] = '1'
    if spec.archive_copy:
        if not spec.archive_run:
            raise Refusal('archive copy on and no archive Run named (paths.ARCHIVE_RUN / --archive-run): the engine refuses '
                          'to archive outside a generation Run (contract 51/66, D-G20)')
        s['P3_ARCHIVE_RUN'] = spec.archive_run
        s['P3_ARCHIVE_ROOT'] = spec.archive_root
    else:
        s['P3_ARCHIVE_WRITE'] = '0'
    if not v['refine_x2']:
        s['P3_REFINE_OFF'] = '1'
    if v['striation_off_battery']:
        s['P3_STRIATION_OFF'] = '1'
    if spec.cadence == '10min' and v['striation']:
        s['P3_STRIATION_10MIN'] = '1'
    s['P3_STREAM_RESID'] = '1' if v['streaming'] else '0'
    s['DEEP_NPROC_RESID'] = str(v['nproc_resid'])
    # the engine reads P3_MIN_EPOCHS twice: the sigma-stack attempt (both cadences) and the 10-minute minimum epochs
    if spec.tier == 'sublst':
        if int(v['min_epochs']['sublst']) != v['sigstack_min_epochs']:
            raise Refusal(f"MIN_EPOCHS[sublst]={v['min_epochs']['sublst']} != SIGMA_STACK_MIN_EPOCHS="
                          f"{v['sigstack_min_epochs']}: the engine reads both from P3_MIN_EPOCHS at 10 minutes")
    else:
        s['P3_DEEP_MIN_EPOCHS'] = str(int(v['min_epochs']['deep']))
    s['P3_MIN_EPOCHS'] = str(v['sigstack_min_epochs'])
    s['P3_BEAM_GATE_RATIO'] = repr(v['beam_gate_ratio'])
    if not v['beam_gate_on']:
        s['P3_BEAM_GATE_OFF'] = '1'
    s['P3_SANITY_I_X'] = repr(v['sanity_i_x'])
    s['P3_SEARCH_TIER_SIGMA'] = repr(v['search_tier_sigma'])
    # the cell-median gates' minimum counts (Gregg 2026-10-14: 5 kept, a skip counted; deep_search.py:82-84), as the search
    # step passes them (deep_step.engine_env, tier_sublst.engine_env)
    s['P3_SANITY_MIN_NIGHTS'] = str(int(v['sanity_min_nights']))
    s['P3_SEARCH_TIER_MIN'] = str(int(v['search_tier_min']))
    s['P3_BEAM_GATE_MIN_NIGHTS'] = str(int(v['beam_gate_min_nights']))
    from exopipe.phase3.lib.masters import admit as _AD                 # D-ADMIT: how the engine reads admitted nights, as the
    from exopipe.phase3.lib.search import admitted as _ADS              # search step passes it (deep_step.engine_env)
    s[_ADS.POLICY_ENV] = _AD.admitted_policy(cfg.masters_admit.ADMITTED_NIGHTS)
    from exopipe.phase3.lib.search import removal_rules as _RR          # B-02: the search's removal rules and the solar audit
    try:                                                                # its daytime gate reads, as the search step passes them
        s[_RR.RULES_ENV] = _RR.search_rules(cfg.search_detect.REMOVAL_RULES)   # (deep_step.engine_env)
    except RuntimeError as e:
        raise Refusal(str(e))
    s[_RR.AUDIT_ENV] = cfg.search_tables.SOLAR_AUDIT
    s[_RR.ELEV_EPOCH_ENV] = cfg.catalog_conform.DEEP_ELEV_EPOCH          # B-02 step 3b (deep_step.engine_env)
    if s[_RR.ELEV_EPOCH_ENV] not in _RR.ELEV_EPOCHS:
        raise Refusal(f'catalog_conform.DEEP_ELEV_EPOCH {s[_RR.ELEV_EPOCH_ENV]!r} is not one of {_RR.ELEV_EPOCHS} - REFUSING')
    s['DEEP_XLA_MEMFRAC'] = repr(v['flow_xla_mem_fraction'])
    if v['flow_lock']:
        s['P3_FLOW_LOCK'] = v['flow_lock']        # node-global lock shared with production searches (one GPU batch per node)
    flow_py = (cfg.site.PYTHON or {}).get(v['flow_python'])
    if not flow_py:
        raise Refusal(f"no interpreter {v['flow_python']!r} in the site file PYTHON (the x2 flow batch, P3_FLOW_PYTHON)")
    s['P3_FLOW_PYTHON'] = flow_py
    parts = [p for p in (os.environ.get('PYTHONPATH') or '').split(os.pathsep) if p]
    root = worktree_root()
    s['PYTHONPATH'] = os.pathsep.join([root] + [p for p in parts if p != root])
    return {k: str(x) for k, x in s.items() if x is not None}


def run_child(spec, scratch, unit=None, cfg=None, timeout=None):
    """Run the engine for one unit in a fresh child (core.legacy.run). Returns the child's result dict (status 'ok').
    Raises Refusal (engine refusal / engine-seam), NoInput (policy skip), RuntimeError (any other failure)."""
    from exopipe.phase3.core import legacy
    cfg = cfg or _cfg()
    v = engine_values(spec.tier, cfg)
    os.makedirs(spec.attempt, exist_ok=True)
    os.makedirs(scratch, exist_ok=True)
    spec_path = os.path.join(scratch, 'engine_spec.json')
    res_path = os.path.join(scratch, 'engine_result.json')
    with open(spec_path, 'w') as f:
        json.dump(dict(spec.to_dict(), result_path=res_path), f, indent=1)
    lr = legacy.run(CHILD_MODULE, [spec_path], python=v['python'], module=True, env=child_env(spec, cfg),
                    cwd=worktree_root(), log=spec.log_path(), timeout=timeout, refusal_codes=(2,), refusal_text=(),
                    unit=unit)
    res = None
    if os.path.exists(res_path):
        with open(res_path) as f:
            res = json.load(f)
    if res is None:
        raise RuntimeError(f'engine child exited rc={lr.rc} after {lr.wall_s:.0f} s without a result (killed, out of '
                           f'memory or crashed); log {spec.log_path()}: {lr.tail[-600:]}')
    res['rc'], res['wall_s'] = lr.rc, lr.wall_s
    st = res.get('status')
    if st == 'refused':
        raise Refusal(f"{res.get('gate') or 'engine'}: {res.get('reason')}")
    if st == 'error':
        raise RuntimeError(f"engine error: {res.get('reason')} (log {spec.log_path()})")
    if st != 'ok':
        raise RuntimeError(f'engine child result has status {st!r} (log {spec.log_path()})')
    ret = res.get('return')
    if not isinstance(ret, dict):
        raise RuntimeError(f'run_deep_cell(frames_only=True) returned {type(ret).__name__}, not the frames result dict')
    if ret.get('policy_skip'):
        raise NoInput(f"policy-skip: {ret['policy_skip']}")
    return res


def archive_path(frame_path, spec):
    """The archive copy of an attempt frame: <archive>/<LST>/<night>/<RUN>/<scope>/<S>/<cadence>/<PRODNAME>
    (phase3_deep.py:850-873; the night is the frame key's date part)."""
    from astropy.io import fits
    m = _FRAME_RE.match(os.path.basename(frame_path))
    name = str(fits.getheader(frame_path).get('PRODNAME', '') or '')
    if not m or not name or name == 'unknown':
        return None
    if not name.endswith('.fits'):
        name += '.fits'
    return os.path.join(spec.archive_root, spec.lst, m['d'][:10], spec.archive_run, spec.scope, m['s'], spec.cadence,
                        name)


def collect(spec):
    """Scan the unit's (fresh) attempt directory: the products the engine wrote for this cell. Returns
    {'frames': {(date, S): path}, 'siblings': {(date, S): path}, 'refusals': {kind: path}, 'search_refusals': {kind:
    path}, 'sigstack': {S: path}, 'pbcorr_missing': path or None, 'extras': [anything else in the run root]}."""
    out = {'frames': {}, 'siblings': {}, 'refusals': {}, 'search_refusals': {}, 'sigstack': {}, 'pbcorr_missing': None,
           'extras': []}
    for st in ('I', 'V'):
        d = spec.resid_dir(st)
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                m = _FRAME_RE.match(f)
                if m and m['scope'] == spec.scope and m['s'] == st:
                    (out['siblings'] if m['pb'] else out['frames'])[(m['d'], st)] = os.path.join(d, f)
        p = spec.sigstack_path(st)
        if os.path.exists(p):
            out['sigstack'][st] = p
    rdir = os.path.join(spec.run_root(), 'refusals')
    if os.path.isdir(rdir):
        for f in sorted(os.listdir(rdir)):
            pre, suf = f'{spec.cell}_', '_excluded.csv'
            if f.startswith(pre) and f.endswith(suf):
                kind = f[len(pre):-len(suf)]
                (out['refusals'] if kind in FRAME_REFUSAL_KINDS else out['search_refusals'])[kind] = os.path.join(rdir, f)
    p = spec.pbcorr_missing_path()
    if os.path.exists(p):
        out['pbcorr_missing'] = p
    rr = spec.run_root()
    if os.path.isdir(rr):
        for f in sorted(os.listdir(rr)):
            if f not in ('refusals', 'sigstack') and not f.endswith('.lock'):
                out['extras'].append(os.path.join(rr, f))
    return out


def refusal_dates(path):
    """{date: reason text} of one headerless refusal file (csv.reader: the provenance gate quotes its reason)."""
    import csv
    out = {}
    with open(path, newline='') as f:
        for r in csv.reader(f):
            if len(r) >= 3:
                out[r[2]] = ','.join(r[3:-1]) if len(r) > 4 else ''
    return out


def pbcorr_missing_dates(path):
    import csv
    out = {}
    if path and os.path.exists(path):
        with open(path, newline='') as f:
            for r in csv.reader(f):
                if len(r) >= 4:
                    out[r[1]] = r[3]
    return out


def reported_set(res, stokes):
    """The frame and sibling paths the frames_only engine reported (deep_seam result contract), or None."""
    ret = res.get('return') if res else None
    if not isinstance(ret, dict) or 'frames' not in ret:
        return None
    fr = ret.get('frames') or {}
    sib = ret.get('siblings') or {}
    return {os.path.abspath(p) for p in list(fr.get(stokes, [])) + list(sib.get(stokes, []))}


def describe_env(s):
    """One line of the variables the step sets for the child (dry-run, record)."""
    return ' '.join(f'{k}={s[k]}' for k in sorted(s) if k != 'PYTHONPATH')
