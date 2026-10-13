"""Config keys of `exopipe3 jupiter` that MIRROR a literal of an in-place module (refactor window, DESIGN 11.1).

During the window the Jupiter engines run unchanged, so each reads its OWN module constant, not config.py. config.py holds
the same value so `exopipe3 config`, the step's gates and its content tests see it. Two checks keep them honest:

- ``check_literals(module_keys)``: at step start every MIRROR key used by the action is compared with the module's literal
  (parsed from the source with ast - nothing imported); a difference refuses the run ("config and the in-place module
  disagree"), so a production edit of a constant cannot drift silently from config.
- ``check_overrides(cfg)``: an override (--set / --config) of a MIRROR key would change nothing in the engine, so it refuses
  - except the REDIRECTABLE roots, which the in-place driver sets as module globals (``redirect_globals``) when the config
  value differs from the literal (parity runs against the pre-bake reserve; a production run sets nothing).
"""
import ast
import os

from exopipe.phase3.core.record import Refusal

# config key -> (module key, module constant name or tuple of names)
MIRRORS = {
    # measure (clv15/phase3_jupiter_measure.py)
    'jupiter.SUBLST_ROOT': ('measure', 'SL'),
    'jupiter.CUBES_ROOT': ('measure', 'CUBES'),
    'jupiter.EPHEM_DIR': ('measure', 'EPH'),
    'jupiter_measure.CROP_ROOT': ('measure', 'SCRATCH_DEFAULT'),
    'jupiter.SUBBANDS': ('measure', 'BANDS'),
    'jupiter.WIDEBANDS': ('measure', 'WIDE'),
    'jupiter.BODIES': ('measure', 'BODIES'),
    'jupiter_measure.FIT_SNR': ('measure', 'FIT_SNR'),
    'jupiter_measure.DET_SNR': ('measure', 'DET_SNR'),
    'jupiter_measure.CUT_DEG': ('measure', 'CUT_DEG'),
    'jupiter_measure.CROP_DEG': ('measure', 'CROP_DEG'),
    'jupiter_measure.ASSIGN_PCT': ('measure', 'PCT'),
    # V2 code tree
    'jupiter.DEEP_WEIGHTS': ('v2:masters/jupv2_masters.py', 'WEIGHTS_CSV'),
    'jupiter_v2.VARIANTS': ('v2:masters/jupv2_masters.py', 'VARIANTS'),
    'jupiter_v2.FAR_BEAMS': ('v2:masters/jupv2_masters.py', 'FAR_BEAMS'),
    'jupiter_v2.V3_BEAMS': ('v2:masters/jupv2_masters.py', 'V3_BEAMS'),
    'jupiter_v2.ARTEFACT_ANNULI_DEG': ('v2:masters/jupv2_masters.py', ('ANN_IN', 'ANN_OUT')),
    'jupiter_v2.INTEGRATION_S': ('v2:masters/jupv2_masters.py', 'INT_S'),
    'jupiter_v2.ANCHOR_MEMBERS': ('v2:anchor/jblue_build.py', 'BLUE'),
    'jupiter_v2.GATE': ('v2:integrate/jint_metrics.py', ('GATE_MIN_REFS', 'GATE_RADIUS_DEG', 'GATE_PCT')),
    'jupiter_v2.WANDER_MIN_PAIRS': ('v2:integrate/jint_metrics.py', 'WANDER_MIN_PAIRS'),
    'jupiter_v2.PROVISIONAL_K_MAD': ('v2:integrate/jint_metrics.py', 'K_MAD'),
    'jupiter_v2.RULED_ROOT': ('v2:site/jsite_render.py', 'DEFAULT_ROOT'),
    'jupiter_v2.ROOT_ORDER': ('v2:site/jsite_render.py', 'ROOT_ORDER'),
    'jupiter_v2.STRETCH_SIGMA': ('v2:site/jsite_render.py', 'STRETCH'),
    'jupiter_v2.ZOOM_FRAME': ('v2:site/jsite_render.py', ('ZOOM_CAP', 'ZOOM_MARGIN', 'ZOOM_MIN_SIDE')),
    'jupiter_v2.REGION_LEVELS': ('v2:site/jsite_render.py', 'REGION_LEVELS'),
    'jupiter_v2.R95_OVER_SIGMA': ('v2:site/jsite_lib.py', 'R95_OVER_SIGMA'),
    'jupiter_v2.HZ_API': ('v2:nightgeom/hzfetch.py', 'API'),
    'jupiter_v2.HZ_SITE': ('v2:nightgeom/hzfetch.py', 'SITE'),
    'jupiter_occurrence.NANCAY_DIR': ('v2:occurrence/nda_catalog.py', 'JOV'),
    'jupiter_occurrence.MERGE_GAP_MIN': ('v2:occurrence/nda_catalog.py', 'MERGE_GAP_MIN'),
    'jupiter_occurrence.DROP_SESSIONS': ('v2:occurrence/nda_catalog.py', 'DROP_SESSIONS'),
    'jupiter_occurrence.LITERATURE': ('v2:occurrence/digitize_literature.py', ('MARQUES', 'ZARKA', 'PROPOSAL')),
    'jupiter_occurrence.BINS': ('v2:occurrence/build_maps.py', ('NB', 'BIN')),
    'jupiter_occurrence.BOOTSTRAP': ('v2:occurrence/build_maps.py', ('B_BOOT', 'SEED')),
    'jupiter_occurrence.DE_VARIANTS': ('v2:occurrence/build_maps.py', 'DE_VARIANTS'),
    'jupiter_occurrence.HZ_STEP': ('v2:occurrence/fetch_horizons.py', ('STEP_MIN', 'CHUNK_DAYS')),
    'jupiter_occurrence.TDB_MINUS_UTC_S': ('v2:occurrence/validate_geometry.py', 'TDB_MINUS_UTC'),
    'jupiter_occurrence.FIG_DPI': ('v2:occurrence/make_figures.py', 'DPI'),
    # Phase-1 flag backfill (exopipe/phase1, production package)
    'jupiter_flag.PHASE1_ROOT': ('flag_tool', 'PHASE1_ROOT'),
    'jupiter_flag.CLEAN_EMPTY_FRACTION': ('flag_tool', 'CLEAN_EMPTY_FRACTION'),
    'jupiter_flag.RMS_TOLERANCE': ('flag_tool', 'RMS_TOLERANCE'),
    'jupiter_flag.QUANT_TOL': ('flag_tool', 'QUANT_TOL'),
    'jupiter_flag.TEST_VERSION': ('flag_qa', 'TEST_VERSION'),
    'jupiter_flag.QA_BOX_HALF_PX': ('flag_qa', 'QA_BOX_HALF'),
    'jupiter_flag.SEARCH_BEAMS': ('flag_qa', 'SEARCH_BEAMS'),
    'jupiter_flag.MASK_BEAMS': ('flag_qa', 'MASK_BEAMS'),
    'jupiter_flag.DETECT_SNR': ('flag_qa', 'DETECT_SNR'),
    'jupiter_flag.HALF_INT_S': ('flag_qa', 'HALF_INT_S'),
    'jupiter_flag.COPY_REASONS': ('flag_qa', 'COPY_REASONS'),
    # servers
    'jupiter_serve.TOL_S': ('server', 'TOL'),
}

# MIRROR keys the in-place driver can redirect (module global set only when config differs from the literal)
REDIRECTABLE = {'jupiter.SUBLST_ROOT', 'jupiter.CUBES_ROOT', 'jupiter.EPHEM_DIR', 'jupiter_measure.CROP_ROOT',
                'jupiter.DEEP_WEIGHTS'}

# which MIRROR module keys each action reads
ACTION_MODULES = {
    'measure': ('measure',),
    'v2': ('v2:masters/jupv2_masters.py', 'v2:anchor/jblue_build.py', 'v2:integrate/jint_metrics.py', 'v2:site/jsite_render.py',
           'v2:site/jsite_lib.py', 'v2:nightgeom/hzfetch.py'),
    'occurrence': ('v2:occurrence/nda_catalog.py', 'v2:occurrence/digitize_literature.py', 'v2:occurrence/build_maps.py',
                   'v2:occurrence/fetch_horizons.py', 'v2:occurrence/validate_geometry.py', 'v2:occurrence/make_figures.py'),
    'flag-backfill': ('flag_tool', 'flag_qa'),
    'serve': ('server',),
    'ephem': (),
    'quarantine': (),
}


def module_path(key):
    """Absolute path of an in-place module key ('measure', 'server', 'flag_tool', 'flag_qa', 'v2:<rel>', 'phase0:<rel>')."""
    from exopipe.phase3.config import CFG
    from exopipe.phase3.core import env
    J = CFG.jupiter
    if key == 'measure':
        return os.path.join(env.legacy_root('clv15'), J.MEASURE_MODULE)
    if key == 'server':
        return os.path.join(env.legacy_root('exopipe_ops'), J.SERVER_MODULE)
    if key == 'flag_tool':
        return CFG.jupiter_flag.TOOL
    if key == 'flag_qa':
        return CFG.jupiter_flag.QA_MODULE
    if key.startswith('v2:'):
        return os.path.join(J.V2_CODE_ROOT, key[3:])
    if key.startswith('phase0:'):
        return os.path.join(J.PHASE0_CODE_ROOT, key[7:])
    raise KeyError(f'unknown in-place module key {key!r}')


_LIT_CACHE = {}


def module_literals(path):
    """{NAME: (value, line)} of the top-level literal assignments of a module (ast, nothing imported)."""
    st = os.stat(path)
    ck = (path, st.st_mtime_ns, st.st_size)
    if ck in _LIT_CACHE:
        return _LIT_CACHE[ck]
    tree = ast.parse(open(path).read(), filename=path)
    out = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for tgt in node.targets:
            names = [tgt] if isinstance(tgt, ast.Name) else list(getattr(tgt, 'elts', []))
            vals = [node.value] if isinstance(tgt, ast.Name) else list(getattr(node.value, 'elts', []))
            if len(names) != len(vals):
                continue
            for n, v in zip(names, vals):
                if not isinstance(n, ast.Name):
                    continue
                try:
                    out[n.id] = (ast.literal_eval(v), node.lineno)
                except (ValueError, SyntaxError, TypeError):
                    pass
    _LIT_CACHE[ck] = out
    return out


def _norm(v):
    if isinstance(v, (list, tuple)):
        return tuple(_norm(x) for x in v)
    if isinstance(v, dict):
        return {k: _norm(x) for k, x in v.items()}
    return v


def literal_of(key):
    """(module path, value, line) of the module literal a MIRROR key mirrors (value a tuple for multi-name mirrors)."""
    mkey, names = MIRRORS[key]
    path = module_path(mkey)
    lits = module_literals(path)
    if isinstance(names, tuple):
        missing = [n for n in names if n not in lits]
        if missing:
            raise KeyError(f'{path}: no top-level literal {missing}')
        return path, tuple(lits[n][0] for n in names), lits[names[0]][1]
    if names not in lits:
        raise KeyError(f'{path}: no top-level literal {names}')
    return path, lits[names][0], lits[names][1]


def production_value(key):
    """The config value of a key before any override of this run."""
    from exopipe.phase3.config import CFG
    for o in CFG.overrides:
        if o[0].split('[')[0] == key:
            return o[1]
    return CFG.get(key)


def check_literals(module_keys):
    """Refuse when a MIRROR key of these modules differs from the module literal (before overrides). Returns the number
    of keys compared."""
    bad, n = [], 0
    for key, (mkey, _) in MIRRORS.items():
        if mkey not in module_keys:
            continue
        try:
            path, lit, line = literal_of(key)
        except (OSError, KeyError, SyntaxError) as e:
            bad.append(f'{key}: cannot read the in-place literal ({e})')
            continue
        n += 1
        if _norm(production_value(key)) != _norm(lit):
            bad.append(f'{key} = {production_value(key)!r} but {os.path.basename(path)}:{line} has {lit!r}')
    if bad:
        raise Refusal('config and the in-place module disagree (MIRROR keys; fix config.py or the module by a commit): '
                      + '; '.join(bad[:6]) + (' ...' if len(bad) > 6 else ''))
    return n


def check_overrides(cfg):
    """Refuse an override of a MIRROR key the driver cannot redirect (it would change nothing in the engine)."""
    bad = sorted({o[0].split('[')[0] for o in cfg.overrides if o[0].split('[')[0] in MIRRORS
                  and o[0].split('[')[0] not in REDIRECTABLE})
    if bad:
        raise Refusal(f'override of {", ".join(bad)} refused: each mirrors a literal of an in-place module that the step does '
                      f'not redirect, so the engine would ignore it (lib/jupiter/mirrors.py); change the value after absorption')


def redirect_globals(module_key, extra=None):
    """{module global: config value} for the REDIRECTABLE keys of this module whose config value differs from the literal."""
    from exopipe.phase3.config import CFG
    out = {}
    for key in REDIRECTABLE:
        mkey, name = MIRRORS[key]
        if mkey != module_key:
            continue
        _, lit, _ = literal_of(key)
        val = CFG.get(key)
        if _norm(val) != _norm(lit):
            out[name] = val
    out.update(extra or {})
    return out
