"""Gate G13 (window lock) of `exopipe3 curves`: the in-place extractor's own constants against config.

During the refactor window the product is made by clv15/phase3_snapshot_lightcurves.py run in place (DESIGN 11.1), so the
values that define the product live in that module and the modules it imports. config.curves mirrors them as
PROTECTED_KEYS; this check reads the modules' MODULE-LEVEL LITERALS from their source (``ast``, nothing imported, nothing
executed) and refuses before any extraction when they differ from config, or when a constant is no longer a literal the
check can read (fail closed: the production chat changed the code, the mirror must be re-checked).

Checked (module: constant -> config key):
  phase3_snapshot_lightcurves  STAMP -> curves.STAMP_PX; NBINS -> curves.NBINS; REF_SNR -> curves.REF_SNR;
                               os.environ.get('P3_LC_WAIT_I_S', ...) is read (curves.WAIT_I_TABLE_S is exported under it)
  phase3_snapshot_search       PB_LIBRARY -> curves.PB_LIBRARY; INTEGRATION_S -> curves.INTEGRATION_S (content test
                               light_curve.integrations); the frame roots of frame_root (discover_frames' default, clv15
                               5c83ddc): PHASE1_ROOT = os.environ.get('P3_SNAP_PHASE1_ROOT', paths.PHASE1_ROOT),
                               GPU_FRAME_ROOT -> paths.SNAPSHOT_GPU_FRAME_ROOT, GPU_CELLS_FILE =
                               os.environ.get('P3_SNAP_GPU_CELLS', paths.SNAPSHOT_GPU_FRAME_CELLS), with neither variable
                               naming another path in this environment; RUN_PINS likewise (P3_SNAP_RUN_PINS)
  phase3_snapshot_refastro     LOWRES -> curves.FLOW_LOWRES_PX; FLOWTREE = os.environ.get('P3_SNAP_FLOWTREE', ...) (the step
                               exports paths.SNAPSHOT_FLOW_ROOT under that name)
  phase3_snapshot_chain        SAMPLES -> curves.SAMPLES
With curves.SEARCH_FRAMES (clv15 55469ba) the extractor's SOURCE TEXT must also hold the option '--search-frames', the
search's frame-list name curves.SEARCH_FRAME_LIST as its '%s' literal ('noise_%s_%s_median.csv') and the refusal the step
maps to gate G18 (workers._GATES) - a module without them would fail every pair at its argument parser.
Not checkable as constants: frame_root's rule itself (a run-wide override wins, else a listed cell reads the GPU root;
plan.frame_root mirrors it for the input fingerprint) and the inline literals in extract(), LC:115-171: the stamp core
(|d| <= 2), 1.4826, the in-field margin (STAMP // 2 + 2), the 1e-3 deg WCS tolerance, the flow cell size 8.0, the
3-reference minimum, 'master:S'. Their mirrors are protected keys; the content tests (lib/curves/tests.py) check the written
tables against them.
"""
import ast
import os

MODULES = {'lc': 'phase3_snapshot_lightcurves.py', 'ss': 'phase3_snapshot_search.py',
           'ra': 'phase3_snapshot_refastro.py', 'ch': 'phase3_snapshot_chain.py'}


def _module_literals(path):
    """{name: value or a marker} of the module-level single-name assignments; os.environ.get('VAR', default) calls are
    returned as ('environ', 'VAR', default)."""
    with open(path) as f:
        tree = ast.parse(f.read(), filename=path)
    out = {}
    env_names = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name, val = node.targets[0].id, node.value
            try:
                out[name] = ast.literal_eval(val)
                continue
            except (ValueError, SyntaxError):
                pass
            if (isinstance(val, ast.Call) and isinstance(val.func, ast.Attribute) and val.func.attr == 'get'
                    and isinstance(val.func.value, ast.Attribute) and val.func.value.attr == 'environ' and val.args):
                try:
                    out[name] = ('environ',) + tuple(ast.literal_eval(a) for a in val.args)
                    continue
                except (ValueError, SyntaxError):
                    pass
            out[name] = ('unreadable', ast.dump(val)[:80])
    for node in ast.walk(tree):                   # every os.environ.get('NAME', ...) anywhere in the module
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'get'
                and isinstance(node.func.value, ast.Attribute) and node.func.value.attr == 'environ' and node.args):
            try:
                env_names.add(ast.literal_eval(node.args[0]))
            except (ValueError, SyntaxError):
                pass
    out['__environ_names__'] = env_names
    return out


def check(clv15_root, C, paths_cfg):
    """(ok, problems list, module files) for the extractor under ``clv15_root`` against config ``C`` (curves) and
    ``paths_cfg`` (paths)."""
    files = {k: os.path.join(clv15_root, v) for k, v in MODULES.items()}
    problems = []
    lit = {}
    for k, p in files.items():
        try:
            lit[k] = _module_literals(p)
        except (OSError, SyntaxError) as e:
            problems.append(f'{os.path.basename(p)} unreadable ({type(e).__name__}: {e})')
            lit[k] = {'__environ_names__': set()}

    def want(mod, name, expected, key):
        got = lit[mod].get(name, ('absent',))
        if isinstance(expected, tuple) and isinstance(got, (list, tuple)):
            got = tuple(tuple(x) if isinstance(x, (list, tuple)) else x for x in got)
            expected = tuple(tuple(x) if isinstance(x, (list, tuple)) else x for x in expected)
        if got != expected:
            problems.append(f'{MODULES[mod]} {name} = {got!r} != {key} {expected!r}')

    want('lc', 'STAMP', C.STAMP_PX, 'curves.STAMP_PX')
    want('lc', 'NBINS', C.NBINS, 'curves.NBINS')
    want('lc', 'REF_SNR', C.REF_SNR, 'curves.REF_SNR')
    want('ss', 'PB_LIBRARY', C.PB_LIBRARY, 'curves.PB_LIBRARY')
    want('ss', 'GPU_FRAME_ROOT', paths_cfg.SNAPSHOT_GPU_FRAME_ROOT, 'paths.SNAPSHOT_GPU_FRAME_ROOT')
    want('ss', 'INTEGRATION_S', C.INTEGRATION_S, 'curves.INTEGRATION_S')
    want('ra', 'LOWRES', C.FLOW_LOWRES_PX, 'curves.FLOW_LOWRES_PX')
    want('ch', 'SAMPLES', tuple(C.SAMPLES), 'curves.SAMPLES')
    ft = lit['ra'].get('FLOWTREE')
    if not (isinstance(ft, tuple) and len(ft) >= 2 and ft[0] == 'environ' and ft[1] == 'P3_SNAP_FLOWTREE'):
        problems.append(f"{MODULES['ra']} FLOWTREE = {ft!r} is no longer read from P3_SNAP_FLOWTREE (the step exports "
                        f"paths.SNAPSHOT_FLOW_ROOT under that name)")
    # the paths discover_frames reads that production takes from the environment at import, each = its config key: the
    # Phase-1 run-pin file (clv15 facb3e8), the frame root (a run-wide override, 2026-11-09) and the GPU-frame cell list
    # (clv15 5c83ddc) - their defaults, and no other path named in this environment (archive_frames runs it in process)
    for name, var, key in (('RUN_PINS', 'P3_SNAP_RUN_PINS', 'PHASE1_RUN_PINS'),
                           ('PHASE1_ROOT', 'P3_SNAP_PHASE1_ROOT', 'PHASE1_ROOT'),
                           ('GPU_CELLS_FILE', 'P3_SNAP_GPU_CELLS', 'SNAPSHOT_GPU_FRAME_CELLS')):
        want_v = getattr(paths_cfg, key)
        got = lit['ss'].get(name)
        if got != ('environ', var, want_v):
            problems.append(f"{MODULES['ss']} {name} = {got!r} != os.environ.get('{var}', paths.{key} {want_v!r})")
        env_v = os.environ.get(var)
        if env_v and os.path.realpath(env_v) != os.path.realpath(want_v):
            problems.append(f'{var}={env_v} in this environment: discover_frames would read another path than paths.{key} '
                            f'({want_v}) - unset it')
    if 'P3_LC_WAIT_I_S' not in lit['lc']['__environ_names__']:
        problems.append(f"{MODULES['lc']} no longer reads P3_LC_WAIT_I_S (curves.WAIT_I_TABLE_S is exported under it)")
    if getattr(C, 'SEARCH_FRAMES', False):
        problems += search_frames_problems(files['lc'], C.SEARCH_FRAME_LIST)
    return not problems, problems, sorted(files.values())


def search_frames_problems(lc_path, list_name):
    """The extractor's --search-frames contract (clv15 55469ba) in its source text: the option, the frame-list file name
    (``list_name`` with {sb} / {st} as the module's '%s'), and the refusal message gate G18 is keyed on."""
    from exopipe.phase3.lib.curves.workers import _GATES
    try:
        with open(lc_path) as f:
            src = f.read()
    except OSError as e:
        return [f'{os.path.basename(lc_path)} unreadable ({type(e).__name__}: {e})']
    need = {"'--search-frames' / \"--search-frames\" option": ('"--search-frames"' in src or "'--search-frames'" in src),
            f'frame-list literal {list_name.format(sb="%s", st="%s")!r}': list_name.format(sb='%s', st='%s') in src,
            'the G18 refusal text': any(gate == 'G18' and key in src for key, gate in _GATES)}
    return [f'{os.path.basename(lc_path)} lacks the {what} of --search-frames (clv15 55469ba) while curves.SEARCH_FRAMES '
            f'is True' for what, ok in need.items() if not ok]
