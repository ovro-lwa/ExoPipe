"""Binding check of the catalog step's ``[lit]`` config keys (spec plan/specs/catalog.md section 1.3 item 5).

During the refactor window the deep catalogue chain runs the exopipe_ops modules IN PLACE (DESIGN 11.1). Many values
config.py names are literals inside those modules, which do not read config. So config can claim a value the module
does not use. At the start of every deep run the step reads each bound literal from the module's SOURCE (``ast``; nothing
is imported or executed) and refuses when it differs from the config key. An override (--set / --config) of a key in a
section's lower-case ``lit`` or ``protected`` tuple is refused too: it would change nothing while the module carries its
own literal, and the run record would claim a value that was never used. The bound keys of the catalog sections are in
their PROTECTED_KEYS (config.py refuses the override); a bound key of a shared section (astrometry, fitting, paths) is
refused by the step at its first unit (``override_refusals``).

``BINDINGS`` lists (config key, module file relative to the exopipe_ops root, extractor, conversion).
"""
import ast
import os

_TREES = {}


def _tree(path):
    st = os.stat(path)
    k = (path, st.st_mtime_ns, st.st_size)
    if k not in _TREES:
        with open(path, 'rb') as f:
            _TREES[k] = ast.parse(f.read(), filename=path)
    return _TREES[k]


def _is_env_get(node):
    """os.environ.get(VAR, default) -> (VAR, default node); else None."""
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'get'
            and isinstance(node.func.value, ast.Attribute) and node.func.value.attr == 'environ' and node.args):
        try:
            var = ast.literal_eval(node.args[0])
        except ValueError:
            return None
        return var, (node.args[1] if len(node.args) > 1 else None)
    return None


def _value(node):
    """The literal a module assigns: a plain literal; float(...)/int(...) of an env default; an env default; the else
    branch of a conditional; the dict a `{...}.get(x)` reads."""
    try:
        return ast.literal_eval(node)
    except ValueError:
        pass
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ('float', 'int') and node.args:
        inner = _value(node.args[0])
        return float(inner) if node.func.id == 'float' else int(inner)
    eg = _is_env_get(node)
    if eg and eg[1] is not None:
        return ast.literal_eval(eg[1])
    if isinstance(node, ast.IfExp):
        return _value(node.orelse)
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
        return _value(node.values[-1])
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'get'
            and isinstance(node.func.value, ast.Dict)):
        return ast.literal_eval(node.func.value)
    raise ValueError(f'not a literal: {ast.dump(node)[:120]}')


def assign(name, entry=None):
    """The first assignment NAME = <literal> (or NAME in a tuple unpack) anywhere in the module; ``entry`` picks a dict
    entry of the value."""
    def ex(tree):
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == name:
                    v = _value(node.value)
                    return v[entry] if entry is not None else v
                if isinstance(tgt, ast.Tuple) and isinstance(node.value, ast.Tuple):
                    for t, val in zip(tgt.elts, node.value.elts):
                        if isinstance(t, ast.Name) and t.id == name:
                            return _value(val)
        raise LookupError(f'no assignment of {name}')
    ex.what = f'{name}' + (f'[{entry!r}]' if entry is not None else '')
    return ex


def envdefault(var):
    """The default of the first os.environ.get(VAR, default)."""
    def ex(tree):
        for node in ast.walk(tree):
            eg = _is_env_get(node)
            if eg and eg[0] == var and eg[1] is not None:
                return ast.literal_eval(eg[1])
        raise LookupError(f'no os.environ.get({var!r}, default)')
    ex.what = f'os.environ.get({var!r}) default'
    return ex


def assign_call_kw(name, kw):
    """NAME = f(..., kw=<number> [* unit]) -> the number (EarthLocation(lat=37.2398 * u.deg, ...))."""
    def ex(tree):
        for node in ast.walk(tree):
            if (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
                    and isinstance(node.value, ast.Call)):
                for k in node.value.keywords:
                    if k.arg == kw:
                        v = k.value.left if isinstance(k.value, ast.BinOp) else k.value
                        return ast.literal_eval(v)
        raise LookupError(f'no {name} = ...({kw}=...)')
    ex.what = f'{name}({kw}=)'
    return ex


def call_arg(func, idx=0):
    """The literal argument ``idx`` of the first call of ``func`` with a literal there (a name or attribute)."""
    def ex(tree):
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                f = node.func
                fname = f.attr if isinstance(f, ast.Attribute) else getattr(f, 'id', None)
                if fname == func and len(node.args) > idx:
                    try:
                        return ast.literal_eval(node.args[idx])
                    except ValueError:
                        continue                  # a computed argument: the next call
        raise LookupError(f'no call {func}(<literal>)')
    ex.what = f'{func}(arg {idx})'
    return ex


def strlit(substr):
    """The first string constant of the module containing ``substr`` (a path inside an expression)."""
    def ex(tree):
        for node in ast.walk(tree):
            v = getattr(node, 'value', None) if type(node).__name__ == 'Constant' else getattr(node, 's', None)
            if isinstance(v, str) and substr in v:
                return v
        raise LookupError(f'no string constant containing {substr!r}')
    ex.what = f'string containing {substr!r}'
    return ex


def _ints(d):
    return {k: int(v) for k, v in d.items()}


def in_func(fname, inner):
    """Apply ``inner`` to the body of the module-level function ``fname`` only (a literal inside a function)."""
    def ex(tree):
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == fname:
                return inner(node)
        raise LookupError(f'no function {fname}')
    ex.what = f'{fname}(): {getattr(inner, "what", "")}'
    return ex


def compare_const(name):
    """The constant a name is compared with: ``NAME > 0.01`` -> 0.01 (first such comparison)."""
    def ex(tree):
        for node in ast.walk(tree):
            if (isinstance(node, ast.Compare) and isinstance(node.left, ast.Name) and node.left.id == name
                    and len(node.comparators) == 1):
                try:
                    return ast.literal_eval(node.comparators[0])
                except ValueError:
                    continue
        raise LookupError(f'no comparison of {name} with a literal')
    ex.what = f'{name} <op> literal'
    return ex


# (config key, module file under the exopipe_ops root, extractor, conversion of the module value or None)
BINDINGS = [
    ('catalog.DEEP_ROOTS_CSV', 'p3s2_conform.py', strlit('s2search_state/roots.csv'), None),
    ('catalog.DEEP_ROOTS_CSV', 'p3s2_frames.py', envdefault('P3S2_ROOTS_CSV'), None),
    ('paths.COADD_SB_ROOT', 'p3s5_parent_union.py', assign('ROOT'), None),
    ('paths.COADD_SB_ROOT', 'gate_inputs.py', envdefault('P3_COADD_ROOT'), None),
    ('catalog.DEEP_PARENT_CANDIDATES', 'p3s5_parent_union.py', assign('CANDS'), None),
    ('catalog.DEEP_PARENT_DIR_DEFAULT', 'gate_inputs.py', envdefault('P3_PARENT_DIR'), None),
    ('catalog.SOLAR_AUDIT', 'p3s2_conform.py', assign('_SOLAR_AUDIT'), None),
    ('catalog.PULSARS_CSV', 'p3s2_conform.py', strlit('pulsars_atnf.csv'), None),
    ('paths.CLV15_ROOT', 'source_localization.py', assign('XM_DIR'), None),
    ('catalog.XMATCH_FILES', 'source_localization.py', assign('XM_FILES'), None),
    ('catalog.TABLES_DEEP', 'multiples_fork.py', assign('TABLES'), None),
    ('catalog.SITE_LAT_DEG[conform]', 'p3s2_conform.py', assign_call_kw('_OVRO', 'lat'), None),
    ('catalog.SITE_LON_DEG', 'p3s2_conform.py', assign_call_kw('_OVRO', 'lon'), None),
    ('catalog.SITE_HEIGHT_M[conform]', 'p3s2_conform.py', assign_call_kw('_OVRO', 'height'), float),
    ('catalog.SITE_LAT_DEG[gate_inputs]', 'gate_inputs.py', assign('OVRO_LAT_DEG'), None),
    ('catalog.SITE_LAT_DEG[builder]', 'build_source_catalogs_v3.py', assign('OVRO_LAT'), None),
    ('catalog.SITE_LAT_DEG[shuffle]', 'p3s3_repeat_shuffle.py', assign('LAT'), None),
    ('catalog.SITE_LAT_DEG[localize]', 'source_localization.py', assign('OVRO_LAT'), None),
    ('catalog.SITE_LAT_DEG[forced_radio]', 'deep_forced_radio.py', assign('LATD'), None),
    ('catalog.SITE_LAT_DEG[ephem]', 'p3s2_ephem_filter.py', assign_call_kw('OVRO', 'lat'), None),
    ('catalog.SITE_LON_DEG', 'p3s2_ephem_filter.py', assign_call_kw('OVRO', 'lon'), None),
    ('catalog.SITE_HEIGHT_M[ephem]', 'p3s2_ephem_filter.py', assign_call_kw('OVRO', 'height'), float),
    ('catalog.SITE_LON_DEG', 'deep_fit_positions.py', assign('OVRO_LON'), None),
    ('catalog.BAND_CLASS', 'select_v3.py', assign('GRP'), None),
    ('paths.PHASE3_ARCHIVE_ROOT', 'gate_inputs.py', envdefault('P3_ARCHIVE_ROOT'), None),
    ('paths.ARCHIVE_RUN', 'gate_inputs.py', envdefault('P3_ARCHIVE_RUN'), None),
    ('catalog.SITE_LAT_DEG[conform_mid_el]', 'p3s2_conform.py', in_func('_deep_mid_el', call_arg('radians', 0)), None),
    ('catalog_fitpos.ON_CENTRE_MAX_FRAC', 'deep_fit_positions.py', compare_const('frac_on'), None),
    ('catalog.COLLECT_ROOTS', 'p3s3_sources.py', assign('COLLECT_ROOTS'), None),
    ('catalog.DEEP_STAGING_ATTEMPTS', 'p3s3_sources.py', assign('STAGING'), None),
    ('catalog_merge.BACKING_CAP_DEG', 'p3s3_merge.py', assign('CAP_DEG'), None),
    ('catalog_merge.CADENCE_LABEL_REPAIR', 'p3s3_merge.py', assign('RUN_CADENCE'), None),
    ('catalog_conform.SIZE_REMOVE_BEAMS', 'p3s2_conform.py', assign('SIZE_REMOVE'), None),
    ('catalog_conform.SIZE_FLAG_BEAMS', 'p3s2_conform.py', assign('SIZE_FLAG'), None),
    ('catalog_conform.SIZE_SNR', 'p3s2_conform.py', assign('SNR_SIZE'), None),
    ('catalog_conform.NOMINAL_BEAM_ARCSEC', 'p3s2_conform.py', assign('BEAM'), None),
    ('catalog_conform.AXIS_REMOVE', 'p3s2_conform.py', assign('AXIS_REMOVE'), None),
    ('catalog_conform.AXIS_FLAG', 'p3s2_conform.py', assign('AXIS_FLAG'), None),
    ('catalog_conform.SANITY_X', 'p3s2_conform.py', assign('SANITY_X'), None),
    ('catalog_conform.ROWLINE_MIN', 'p3s2_conform.py', assign('ROWLINE_MIN'), None),
    ('catalog_conform.DEEP_ELEV_FLOOR_DEG', 'p3s2_conform.py', assign('DEEP_ELEV_FLOOR_DEG'), None),
    ('catalog_conform.DEEP_ELEV_FLOOR_DEG', 'gate_inputs.py', assign('DEEP_ELEV_FLOOR_DEG'), None),
    ('catalog_conform.JUPITER_TRACK_RADIUS_DEG', 'p3s2_conform.py', assign('_JUP_R'), None),
    ('catalog_conform.TABLE_MODE[deep]', 'p3s2_conform.py', envdefault('P3_CONFORM_TABLE'), None),
    ('fitting.K_TEMPLATE[deep,I]', 'deep_forced_radio.py', assign('K_TEMPLATE', 'I'), None),
    ('fitting.K_TEMPLATE[deep,V]', 'deep_forced_radio.py', assign('K_TEMPLATE', 'V'), None),
    ('catalog_select.POLICY', 'select_v3.py', assign('POLICY'), None),
    ('catalog_select.RADOP_ARCMIN', 'select_v3.py', assign('RADOP_AM'), None),
    ('catalog_select.BEAM_DEFAULT_ARCSEC', 'select_v3.py', assign('BEAM_DEFAULT'), None),
    ('catalog_select.EPHEM_JUPITER_DEG', 'p3s2_ephem_filter.py', assign('JUP'), None),
    ('catalog_associate.SYS_FLOOR_ARCSEC', 'build_source_catalogs_v3.py', assign('SYS_FLOOR_AS'), None),
    ('catalog_associate.RPT_PCT', 'build_source_catalogs_v3.py', assign('RPT_PCT'), None),
    ('catalog_associate.SURVIVOR_SNR', 'build_source_catalogs_v3.py', assign('THR'), None),
    ('catalog_associate.E_TARGET', 'build_source_catalogs_v3.py', assign('E_TARGET'), None),
    ('catalog_associate.PRIORITY_SAMPLES', 'build_source_catalogs_v3.py', assign('PRIORITY_SAMPLES'), None),
    ('catalog_associate.SIZE_GATE_7_8', 'build_source_catalogs_v3.py', assign('SIZE_GATE_7_8'), None),
    ('astrometry.K_ASSOC', 'build_source_catalogs_v3.py', assign('K_ASSOC'), None),
    ('astrometry.SYS_SCALE', 'build_source_catalogs_v3.py', assign('SYS_SCALE'), None),
    ('astrometry.SYS_SCALE', 'source_localization.py', assign('SYS_SCALE'), None),
    ('catalog_fork.SNR_MIN', 'multiples_fork.py', assign('SNR_MIN'), None),
    ('catalog_fork.MEMBER_SNR', 'multiples_fork.py', assign('MEMBER_SNR'), None),
    ('catalog_fork.MATCH_BEAMS', 'multiples_fork.py', assign('MATCH_BEAMS'), None),
    ('catalog_fork.N_RANDOM', 'multiples_fork.py', assign('N_RANDOM'), None),
    ('catalog_fork.NPROC', 'multiples_fork.py', assign('NPROC'), None),
    ('catalog_fork.TABLES_SOURCE_DIR', 'multiples_fork.py', assign('TAB_DIR'), None),
    ('catalog_fork.SHUFFLE_SEED', 'p3s3_repeat_shuffle.py', call_arg('default_rng', 0), None),
    ('catalog_fork.SHUFFLE_K', 'p3s3_repeat_shuffle.py', assign('K'), None),
    ('catalog_identity.IDENTITY_K', 'p3s3_identity.py', assign('K_ASSOC'), None),
    ('catalog_localize.RHO_EPOCH', 'source_localization.py', assign('RHO_EPOCH'), None),
    ('catalog_localize.RHO_FAR', 'source_localization.py', assign('RHO_FAR'), None),
    ('catalog_localize.CATS[deep]', 'build_localizations.py', envdefault('P3_LOC_CATS'), None),
    ('catalog_localize.LIST_TIERS', 'p3s2_transient_list.py', assign('TIERS'), None),
    ('catalog_localize.LIST_PORTS', 'p3s2_transient_list.py', assign('PORT'), _ints),
]


def _norm(v):
    if isinstance(v, (list, tuple)):
        return tuple(_norm(x) for x in v)
    if isinstance(v, dict):
        return {(_norm(k) if isinstance(k, (list, tuple)) else k): _norm(x) for k, x in v.items()}
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return float(v)
    return v


def same(a, b):
    return _norm(a) == _norm(b)


def check(cfg, ops_root):
    """[{key, file, what, ok, module, config, detail}] for every binding (nothing raised)."""
    out = []
    for key, rel, ex, conv in BINDINGS:
        path = os.path.join(ops_root, rel)
        r = {'key': key, 'file': path, 'what': getattr(ex, 'what', '')}
        try:
            want = cfg.get(key)
            got = ex(_tree(path))
            if conv is not None:
                got = conv(got)
            r.update(ok=same(got, want), module=repr(got), config=repr(want),
                     detail='' if same(got, want) else f'{rel} {r["what"]} = {got!r} but config {key} = {want!r}')
        except Exception as e:                        # an unreadable module or a moved literal refuses too
            r.update(ok=False, module=None, config=None, detail=f'{rel}: {type(e).__name__}: {e}')
        out.append(r)
    return out


def bound_keys():
    """'section.KEY' (entry brackets removed) of every binding."""
    return sorted({k.split('[', 1)[0] for k, _, _, _ in BINDINGS})


def override_refusals(cfg):
    """The overrides of this run (--set / --config) of a bound key: refused (spec 1.3 item 5)."""
    b = set(bound_keys())
    return [o[0] for o in cfg.overrides if o[0].split('[', 1)[0] in b]


def unprotected_bound(cfg):
    """Bound keys of a section that does not protect them (a shared section: astrometry, fitting, paths)."""
    out = []
    for k in bound_keys():
        sec, key = k.split('.', 1)
        if not cfg.protected(k):
            out.append(k)
    return out
