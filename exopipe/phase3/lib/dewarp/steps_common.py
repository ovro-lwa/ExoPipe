"""Shared pieces of `exopipe3 dewarp` (steps/dewarp.py): the roots of one run, the ONE acceptance function, the measurement
table reader, routing (gate G-D2), the window gates (G-W, G-D0, G-D1, G-D3) and the in-place consistency checks.

Light at import (stdlib only): the in-place modules (phase3_master_dewarp -> cascade_dewarp -> PyBDSF, pb_apply -> reproject)
are imported inside the functions that need them.
"""
import ast
import csv
import hashlib
import os
import re

STAGES = ('stage', 'measure', 'apply', 'verify', 'record')
# Section keys read IN PLACE during the window (their values live in the production modules' own constants); every one is
# in dewarp.PROTECTED_KEYS (no override can reach the in-place code) and gate_in_place compares the values.
IN_PLACE = ('ANCHOR_SUBBAND', 'ANCHOR_FALLBACK', 'ANCHOR_FREQ_MHZ', 'ANCHOR_MATCH_RADIUS_ARCMIN', 'ANCHOR_FLUX_RATIO',
            'CASCADE_MATCH_RADIUS_BASE_ARCMIN', 'CASCADE_MATCH_RADIUS_EXPONENT', 'CASCADE_FLUX_RATIO', 'CASCADE_FLUX_CENTER',
            'MAX_CONSECUTIVE_FAILURES', 'MIN_WARP_ANCHORS', 'OUTLIER_CLIP_SIGMA', 'OUTLIER_CLIP_ITERS', 'WARP_RBF_KERNEL',
            'WARP_RBF_SMOOTHING', 'WARP_RBF_EPSILON', 'WARP_RBF_SNR_WEIGHTING', 'ANCHOR_MIN_ELEVATION_DEG',
            'ANCHOR_EAST_MIN_ELEVATION_DEG', 'ANCHOR_MAX_ISLAND_BEAM_AREAS', 'MIN_ANCHORS', 'HELDOUT_MIN_GAIN_AS', 'CV_FOLDS',
            'CV_SEED', 'VLSSR_BORDER_FRACTION', 'VLSSR_PEAK_FILTER_PX', 'VLSSR_N_PEAKS', 'VLSSR_CENTROID_BOX_PX',
            'VLSSR_MAX_SEP_ARCMIN', 'VLSSR_TAIL_PERCENTILE', 'SCREEN_STATS_BORDER_FRACTION', 'RESAMPLE_METHOD',
            'RESAMPLE_ORDER', 'PB_FLOOR', 'D5_DECIMALS', 'D5_SRC_TOL_ARCMIN', 'D5_MIN_COMMON_PEAKS', 'D6_POOL_RANGE',
            'D6_PROD_TOL', 'D6_PROD_SE_MULT', 'D6_MIN_SOURCES', 'D6_PEAK_RMS_MULT', 'D6_PEAK_FILTER_PX', 'D6_INNER_FRACTION',
            'D6_MIN_INNER_PIXELS', 'D6_BOX_HALF_PX', 'D6_BOOTSTRAP_N', 'D6_BOOTSTRAP_SEED', 'D6_EXEMPT_NAME_PARTS',
            'D4_MIN_SIBLINGS')
# The modules check_mdewarp_gen.py imports in its child (from the production repository it inserts at sys.path[0]: core.legacy
# runs it exactly as production does) and their branch counterparts the step measures and applies with in process. D3 of the
# content test compares the step's apply with the CHILD's resample, so they must be the same code (gate_child_modules).
CHILD_MODULES = (('exopipe/phase3/phase3_master_dewarp.py', 'exopipe/phase3/lib/dewarp/master_dewarp.py'),
                 ('exopipe/phase3/pb_apply.py', 'exopipe/phase3/lib/masters/pb_apply.py'),
                 ('exopipe/phase2/cascade_dewarp.py', 'exopipe/phase2/cascade_dewarp.py'),
                 ('exopipe/phase2/extractor_pb_75.py', 'exopipe/phase2/extractor_pb_75.py'),
                 ('exopipe/common/phase2_config.py', 'exopipe/common/phase2_config.py'),
                 ('exopipe/common/pipeline_config.py', 'exopipe/common/pipeline_config.py'))
# the one permitted difference: the import line rewritten by the move commit (6bc74ca; master_dewarp.py:294)
CHILD_PERMITTED_LINE_DIFFS = {'exopipe/phase3/lib/dewarp/master_dewarp.py':
                              (('    from exopipe.phase3.lib.masters import pb_apply as PB',
                                '    from exopipe.phase3 import pb_apply as PB'),)}
CELL_STAGES = ('apply', 'verify', 'record')
THREAD_VARS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')
CONTENT_OK = 'INSTALL_CONTENT_OK'        # sidecars written by check_mdewarp_gen beside the generation (CMG:166, :177)
APPLIED_SCREEN = 'APPLIED_SCREEN.json'
REBUILD_OK = 'REBUILD_CONTENT_OK'        # <REBUILD_CHECK_ROOT>/<cell>/ (chain_cell.sh:23; phase3_install.py:224)


def cfg():
    from exopipe.phase3.config import CFG
    return CFG


def C():
    return cfg().dewarp


def branch_root():
    """The repository root this package runs from (…/exopipe_refactor): lib/dewarp/ -> lib -> phase3 -> exopipe -> root."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))


def md_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'master_dewarp.py')


def refusal(msg):
    from exopipe.phase3.core.record import Refusal
    return Refusal(msg)


# ---------------------------------------------------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------------------------------------------------

def parse_stages(spec):
    """'--stages a,b' -> the stages in pipeline order. 'stage' is part of 'measure' (it never runs alone)."""
    if not spec:
        return list(STAGES)
    want = [s.strip() for s in spec.split(',') if s.strip()]
    bad = [s for s in want if s not in STAGES]
    if bad:
        raise ValueError(f'unknown stage(s) {bad}; stages: {",".join(STAGES)}')
    if 'stage' in want and 'measure' not in want:
        raise ValueError("stage 'stage' only runs with 'measure' (it stages the measured masters to node-local scratch)")
    if 'measure' in want and 'stage' not in want:
        want.append('stage')              # spec 1.3: staging is part of the measurement (skipped when STAGE_INPUTS False)
    return [s for s in STAGES if s in want]


# ---------------------------------------------------------------------------------------------------------------------
# acceptance (ONE function: dewarp_lst.sh:44-49 = phase3_master_dewarp.py:416 = check_mdewarp_gen.py:82, :88)
# ---------------------------------------------------------------------------------------------------------------------

def accepted(row, min_gain_as):
    """(a)+(b') on the cell's FINAL table row (R-MASTER-DEWARP-ACCEPTANCE-A-AND-B-PRIME): status 'OK' and
    float(anchor_off_med_as) - float(cv_resid_med_as) >= min_gain_as arcsec on the WRITTEN values; a missing row or an
    unparsable value is not accepted (dewarp_lst.sh:44-49 = phase3_master_dewarp.py:416)."""
    try:
        return bool(row) and row['status'] == 'OK' and \
            float(row['anchor_off_med_as']) - float(row['cv_resid_med_as']) >= min_gain_as
    except (KeyError, TypeError, ValueError):
        return False


def read_table(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def final_rows(path, lst=None):
    """{(lst, scope): the LAST row} of a measurement table (rows are appended per attempt, phase3_master_dewarp.py:342-348;
    the final row decides: dewarp_lst.sh:41, check_mdewarp_gen.py:80-81)."""
    out = {}
    for r in read_table(path):
        if lst is None or r.get('lst') == lst:
            out[(r.get('lst'), r.get('scope'))] = r
    return out


class Measurement:
    """One LST's measurement in MD's layout: <dir>/tables/table_<LST>.csv and <dir>/<LST>/maps/<scope>_warp_screen.fits."""

    def __init__(self, mdir, lst):
        self.dir = os.path.abspath(mdir)
        self.lst = lst
        self.table = os.path.join(self.dir, 'tables', f'table_{lst}.csv')

    def screen(self, scope):
        return os.path.join(self.dir, self.lst, 'maps', f'{scope}_warp_screen.fits')

    def screens(self):
        import glob
        return sorted(glob.glob(os.path.join(self.dir, self.lst, 'maps', '*_warp_screen.fits')))

    def exists(self):
        return os.path.isfile(self.table)

    def final_row(self, scope):
        return final_rows(self.table, self.lst).get((self.lst, scope))


def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def sha1(path):
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------------------------------
# roots of one run (spec 4.2: --out re-roots measure/, gen/ and archive/)
# ---------------------------------------------------------------------------------------------------------------------

class Roots:
    def __init__(self, args, run):
        c, CFG = C(), cfg()
        self.rerooted = getattr(args, 'out', None) is not None
        self.out = os.path.abspath(run.out_root)
        self.working = CFG.paths.COADD_SB_ROOT          # the deep working tree (shared root, decision 3)
        self.rebuild = c.REBUILD_ROOT
        self.check = c.REBUILD_CHECK_ROOT
        if self.rerooted:
            self.gen = os.path.join(self.out, 'gen')
            self.measure_base = os.path.join(self.out, 'measure')
            self.archive = os.path.join(self.out, 'archive')
        else:
            self.gen = c.GEN_ROOT
            self.measure_base = c.MEASURE_ROOT
            self.archive = CFG.paths.PHASE3_ARCHIVE_ROOT
        self.measure_dir = os.path.join(self.measure_base, run.run_id)
        self.archive_run = CFG.paths.ARCHIVE_RUN
        self.record_dir = os.path.join(self.archive, c.SCREEN_RECORD_DIR, self.archive_run)
        self.record_csv = os.path.join(self.record_dir, c.SCREEN_RECORD_FILE)
        self.production_record = os.path.join(CFG.paths.PHASE3_ARCHIVE_ROOT, c.SCREEN_RECORD_DIR, self.archive_run,
                                              c.SCREEN_RECORD_FILE)
        self.work = getattr(args, 'work', None)
        self.records = os.path.join(self.out, '_records', 'dewarp')

    def gen_of(self, cell):
        return os.path.join(self.gen, cell, C().GEN_DIR)

    def promoted_master(self, cell):
        lst, scope = cell.split('_', 1)
        return os.path.join(self.rebuild, cell, 'coadd_final', C().MEASURE_FILES[0].format(scope=scope))

    def rebuild_ok(self, cell):
        return os.path.join(self.check, cell, REBUILD_OK)

    def source_dir(self, lst, scope):
        """Staging source of one scope: the first of (REBUILD_ROOT, paths.COADD_SB_ROOT) whose <cell>/coadd_final holds the plain
        measured file - decided per FILE, as dewarp_lst.sh:24-26."""
        plain = C().MEASURE_FILES[0].format(scope=scope)
        for r in (self.rebuild, self.working):
            d = os.path.join(r, f'{lst}_{scope}', 'coadd_final')
            if os.path.isfile(os.path.join(d, plain)):
                return d
        return None

    def write_roots(self, stages):
        """The roots this run writes under, by stage (gate G-W)."""
        out = [('records', self.records)]
        if 'measure' in stages:
            out.append(('measure dir', self.measure_dir))
        if 'apply' in stages or 'verify' in stages:
            out.append(('generation root', self.gen))
        if 'record' in stages:
            out.append(('archive record dir', self.record_dir))
        return out


def scratch_root(work):
    """The node-local scratch root (core/fitsio rules): --work, else the site file's SCRATCH_ROOT for this host."""
    from exopipe.phase3.core import fitsio
    return fitsio.scratch_root(work)


# ---------------------------------------------------------------------------------------------------------------------
# routing (G-D2)
# ---------------------------------------------------------------------------------------------------------------------

def routed_scopes(roots):
    """{(lst, scope)} routed to the LST dewarp. ROUTE_SOURCE 'accepted_list' = the round's hand list (chain_cell.sh:38,
    dewarp_driver.sh:5; today); 'screen_record' = the applied=1 rows of the archived screen record this run uses."""
    c = C()
    if c.ROUTE_SOURCE == 'accepted_list':
        path = route_list_path()
        if not os.path.isfile(path):
            raise refusal(f'G-D2: routing list {path} not found')
        rows = read_table(path)
        if rows and not {'lst', 'scope'} <= set(rows[0]):
            raise refusal(f'G-D2: routing list {path} lacks the columns lst,scope')
        return {(r['lst'].strip(), r['scope'].strip()) for r in rows}, path
    if c.ROUTE_SOURCE == 'screen_record':
        path = roots.record_csv
        if not os.path.isfile(path):
            raise refusal(f'G-D2: ROUTE_SOURCE screen_record but {path} does not exist')
        return {(r['lst'], r['scope']) for r in read_table(path) if r.get('applied') == '1'}, path
    raise refusal(f"G-D2: dewarp.ROUTE_SOURCE {c.ROUTE_SOURCE!r} is neither 'accepted_list' nor 'screen_record'")


def route_list_path():
    return os.path.join(cfg().paths.OPS_ROOT, C().ROUTE_LIST)


def carries_mdwarp(path):
    from exopipe.phase3.core.record import fits_primary_cards
    try:
        return bool(fits_primary_cards(path).get('MDWARP'))
    except OSError:
        return False


def cell_gate(roots, cell, routed):
    """None when a cell may go to the LST dewarp, else the refusal reason (G-D2, G-D7)."""
    lst, scope = cell.split('_', 1)
    if (lst, scope) not in routed:
        return (f'G-D2: {cell} is not routed to the LST dewarp (dewarp.ROUTE_SOURCE {C().ROUTE_SOURCE}: '
                f'not listed)')
    pm = roots.promoted_master(cell)
    if not os.path.isfile(pm):
        return f'G-D2: {cell} has no promoted rebuild ({pm} missing)'
    if not os.path.isfile(roots.rebuild_ok(cell)):
        return f'G-D2: {cell} has no rebuild content-test record {roots.rebuild_ok(cell)} (T1-T8)'
    pbc = os.path.join(os.path.dirname(pm), C().MEASURE_FILES[1].format(scope=scope))
    if carries_mdwarp(pm) or (os.path.isfile(pbc) and carries_mdwarp(pbc)):
        return (f'G-D7: {cell} input master already carries MDWARP - never stack a dewarp '
                f'(R-GREGG-20261009-DEWARP-CASCADE-23HGREEN-3CELLS-FAST item 1)')
    return None


def pending_cells(roots, lst, routed):
    """The routed promoted cells of an LST (what `--lsts L` expands to): routed scopes whose promoted rebuild exists."""
    from exopipe.phase3.core import cells as _cells
    out = []
    for scope in _cells.scopes():
        c = f'{lst}_{scope}'
        if (lst, scope) in routed and os.path.isfile(roots.promoted_master(c)):
            out.append(c)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# window gates
# ---------------------------------------------------------------------------------------------------------------------

def gate_write_roots(roots, stages, staging=None):
    """G-W: no derived write root under paths.PRODUCTION_ROOTS while paths.PRODUCTION_WRITES is False (README rule 10).
    Returns the list of violations (empty = pass)."""
    CFG = cfg()
    if CFG.paths.PRODUCTION_WRITES:
        return []
    items = roots.write_roots(stages) + ([('staging root', staging)] if staging else [])
    bad = []
    for what, path in items:
        pr = CFG.production_root_of(path)
        if pr is not None:
            bad.append(f'{what} {path} is under the production root {pr}')
    if 'measure' in stages and not C().STAGE_INPUTS:
        for what, path in (('working tree', roots.working), ('rebuild root', roots.rebuild)):
            pr = CFG.production_root_of(path)
            if pr is not None:
                bad.append(f'STAGE_INPUTS False measures {what} {path} in place and PyBDSF writes <file>.pybdsf.log '
                           f'beside each input, under the production root {pr}')
    return bad


def gate_input_roots(roots, stages):
    """G-D1: the input roots are directories on this host (no host names: the roots decide)."""
    need = [('dewarp.REBUILD_ROOT', roots.rebuild), ('paths.COADD_SB_ROOT', roots.working)]
    if any(s in stages for s in CELL_STAGES):
        need.append(('dewarp.REBUILD_CHECK_ROOT', roots.check))
    return [f'G-D1: {k} {p} is not a directory on this host' for k, p in need if not os.path.isdir(p)]


def gate_pinned_inputs():
    """G-D3: the pinned VLSSr files carry the sha1 recorded in config (DESIGN 11.6). Returns violations."""
    c = C()
    bad = []
    for key in ('VLSSR_PEAKS', 'VLSSR_CATALOG'):
        p = getattr(c, key)
        if not os.path.isfile(p):
            bad.append(f'G-D3: {key} {p} not found')
            continue
        h = sha1(p)
        if h != getattr(c, key + '_SHA1'):
            bad.append(f'G-D3: {key} {p} sha1 {h} != pinned {getattr(c, key + "_SHA1")}')
    return bad


def module_literals(path, names):
    """Module-level literal assignments of a source file, read with ast (no import): {name: value} for the names found.
    Tuple targets (`A, B = 1, 2`) are unpacked."""
    tree = ast.parse(open(path).read(), filename=path)
    out = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for t in node.targets:
            if isinstance(t, ast.Name) and t.id in names:
                try:
                    out[t.id] = ast.literal_eval(node.value)
                except ValueError:
                    pass
            elif isinstance(t, ast.Tuple) and isinstance(node.value, ast.Tuple) and len(t.elts) == len(node.value.elts):
                for e, v in zip(t.elts, node.value.elts):
                    if isinstance(e, ast.Name) and e.id in names:
                        try:
                            out[e.id] = ast.literal_eval(v)
                        except ValueError:
                            pass
    return out


def content_check_script():
    """check_mdewarp_gen.py (CMG) under core.env.legacy_root('exopipe_ops') (called in place, DESIGN 11.1)."""
    from exopipe.phase3.core import env
    rel = C().CONTENT_CHECK_SCRIPT
    try:
        root = env.legacy_root('exopipe_ops')
    except KeyError as e:
        raise refusal(f'dewarp.CONTENT_CHECK_SCRIPT: {e} (EXOPIPE3_SITE unset?)')
    p = os.path.join(root, rel)
    if not os.path.isfile(p):
        raise refusal(f'dewarp.CONTENT_CHECK_SCRIPT {p} not found')
    return p


def child_repo(script=None):
    """The repository check_mdewarp_gen.py puts first on sys.path (CMG:31 `sys.path.insert(0, '/lustre/gh/main/exopipe')`):
    its child imports phase3_master_dewarp, pb_apply and cascade_dewarp from there."""
    script = script or content_check_script()
    m = re.search(r"sys\.path\.insert\(\s*0\s*,\s*['\"]([^'\"]+)['\"]\s*\)", open(script).read())
    return m.group(1) if m else None


def gate_child_modules(script=None):
    """G-D0b: the production modules the content-test child imports are the code the step measures and applies with in
    process (byte-equal; master_dewarp.py may differ only by its rewritten pb_apply import). Returns (violations,
    {production path: sha1[:12]}) - the sha1s are compared with the child's own fingerprint after it ran."""
    repo = child_repo(script)
    if not repo:
        return ['G-D0b: check_mdewarp_gen.py inserts no repository at sys.path[0] - cannot tell which code its child runs'], {}
    br = branch_root()
    bad, sums = [], {}
    for prod_rel, br_rel in CHILD_MODULES:
        pp, bp = os.path.join(repo, prod_rel), os.path.join(br, br_rel)
        if not (os.path.isfile(pp) and os.path.isfile(bp)):
            bad.append(f'G-D0b: {pp if not os.path.isfile(pp) else bp} missing')
            continue
        pt, bt = open(pp).read(), open(bp).read()
        for have, want in CHILD_PERMITTED_LINE_DIFFS.get(br_rel, ()):
            bt = bt.replace(have + '\n', want + '\n', 1)
        if pt != bt:
            pl, bl = pt.splitlines(), bt.splitlines()
            first = next((i for i, (a, b) in enumerate(zip(pl, bl)) if a != b), min(len(pl), len(bl)))
            bad.append(f'G-D0b: the content-test child would run {pp}, which differs from the branch {bp} at line '
                       f'{first + 1} - the step applies with one and D3 re-samples with the other')
        sums[os.path.abspath(pp)] = hashlib.sha1(open(pp, 'rb').read()).hexdigest()[:12]
    return bad, sums


def _p2_expected(p2):
    """Section keys taken from phase2_config, as the in-place code reads them."""
    return {'ANCHOR_SUBBAND': p2.ANCHOR_SUBBAND, 'ANCHOR_FREQ_MHZ': p2.ANCHOR_FREQ_MHZ,
            'ANCHOR_MATCH_RADIUS_ARCMIN': float(p2.ANCHOR_MATCH_RADIUS.to_value('arcmin')),
            'ANCHOR_FLUX_RATIO': (p2.ANCHOR_FLUX_RATIO_MIN, p2.ANCHOR_FLUX_RATIO_MAX),
            'CASCADE_MATCH_RADIUS_BASE_ARCMIN': p2.CASCADE_MATCH_RADIUS_BASE,
            'CASCADE_MATCH_RADIUS_EXPONENT': p2.CASCADE_MATCH_RADIUS_EXPONENT,
            'CASCADE_FLUX_RATIO': (p2.CASCADE_FLUX_RATIO_MIN, p2.CASCADE_FLUX_RATIO_MAX),
            'CASCADE_FLUX_CENTER': p2.CASCADE_FLUX_CENTER, 'MAX_CONSECUTIVE_FAILURES': p2.MAX_CONSECUTIVE_FAILURES,
            'MIN_WARP_ANCHORS': p2.MIN_WARP_ANCHORS, 'OUTLIER_CLIP_SIGMA': p2.OUTLIER_CLIP_SIGMA,
            'OUTLIER_CLIP_ITERS': p2.OUTLIER_CLIP_ITERS, 'WARP_RBF_KERNEL': p2.WARP_RBF_KERNEL,
            'WARP_RBF_SMOOTHING': p2.WARP_RBF_SMOOTHING, 'WARP_RBF_EPSILON': p2.WARP_RBF_EPSILON,
            'WARP_RBF_SNR_WEIGHTING': p2.WARP_RBF_SNR_WEIGHTING,
            'ANCHOR_MIN_ELEVATION_DEG': dict(getattr(p2, 'ANCHOR_MIN_ELEVATION_DEG', {})),
            'ANCHOR_EAST_MIN_ELEVATION_DEG': dict(getattr(p2, 'ANCHOR_EAST_MIN_ELEVATION_DEG', {})),
            'ANCHOR_MAX_ISLAND_BEAM_AREAS': dict(getattr(p2, 'ANCHOR_MAX_ISLAND_BEAM_AREAS', {})),
            'VLSSR_CATALOG': p2.VLSSR_CATALOG}


def _same(a, b):
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        try:
            return float(a) == float(b)
        except (TypeError, ValueError):
            return False
    return a == b


def gate_in_place(import_heavy=True):
    """G-D0 (DESIGN 11.1/11.4): (1) no override of a protected or in-place dewarp key; (2) every section value that the
    in-place code reads from its own constants equals that constant (MD module literals and CMG's line-52 literals by ast,
    phase2_config attributes, and - with import_heavy - cascade_dewarp.ANCHOR_FALLBACK, pb_apply.PB_FLOOR,
    membership.DEWARP_INTERP / DEWARP_ACCEPTED); (3) every loaded exopipe module comes from this branch.
    Returns (violations, checked) - checked = number of values compared."""
    import sys
    CFG, c = cfg(), C()
    bad, n = [], 0
    prot = set(getattr(c, 'PROTECTED_KEYS', ()) or ())
    for k in IN_PLACE:                  # config.py refuses overrides of PROTECTED_KEYS; every in-place key must be one
        if k not in prot:
            bad.append(f'G-D0: dewarp.{k} is read IN PLACE during the window but is not in dewarp.PROTECTED_KEYS')
    for o in CFG.overrides:             # defensive: an override that reached an in-place or protected key anyway
        sec, _, rest = str(o[0]).partition('.')
        if sec == 'dewarp' and re.split(r'\[', rest, 1)[0] in prot:
            bad.append(f'G-D0: dewarp.{rest} is protected / read in place and was overridden')
    # MD module literals (ast: no PyBDSF import)
    from exopipe.phase3.core import cells as _cells
    md = module_literals(md_path(), ('MIN_ANCHORS', 'HELDOUT_MIN_GAIN_AS', 'RESAMPLE_METHOD', 'OUT_GEN', 'WIDEBANDS',
                                     'VLSSR_PEAKS'))
    for mk, ck, want in (('MIN_ANCHORS', 'MIN_ANCHORS', c.MIN_ANCHORS),
                         ('HELDOUT_MIN_GAIN_AS', 'HELDOUT_MIN_GAIN_AS', c.HELDOUT_MIN_GAIN_AS),
                         ('RESAMPLE_METHOD', 'RESAMPLE_METHOD', c.RESAMPLE_METHOD), ('OUT_GEN', 'GEN_DIR', c.GEN_DIR),
                         ('WIDEBANDS', '(core/cells.BANDS)', _cells.BANDS), ('VLSSR_PEAKS', 'VLSSR_PEAKS', c.VLSSR_PEAKS)):
        n += 1
        if mk not in md:
            bad.append(f'G-D0: master_dewarp.{mk} not found as a module-level literal in {md_path()}')
        elif not _same(md[mk], want):
            bad.append(f'G-D0: master_dewarp.{mk} = {md[mk]!r} but the section says {ck} = {want!r}')
    # CMG line-52 literals
    try:
        cmg = content_check_script()
        lit = module_literals(cmg, ('D5_SRC_TOL_ARCMIN', 'D6_POOL_LO', 'D6_POOL_HI', 'D6_PROD_TOL', 'D6_MIN_SOURCES'))
        for mk, want in (('D5_SRC_TOL_ARCMIN', c.D5_SRC_TOL_ARCMIN), ('D6_POOL_LO', c.D6_POOL_RANGE[0]),
                         ('D6_POOL_HI', c.D6_POOL_RANGE[1]), ('D6_PROD_TOL', c.D6_PROD_TOL),
                         ('D6_MIN_SOURCES', c.D6_MIN_SOURCES)):
            n += 1
            if mk not in lit:
                bad.append(f'G-D0: check_mdewarp_gen.{mk} not found as a module-level literal in {cmg}')
            elif not _same(lit[mk], want):
                bad.append(f'G-D0: check_mdewarp_gen.{mk} = {lit[mk]!r} but the section says {want!r}')
    except Exception as e:          # an unresolvable CMG is itself a G-D0 failure
        bad.append(f'G-D0: {e}')
    # phase2_config attributes
    from exopipe.common import phase2_config as p2
    for k, want in _p2_expected(p2).items():
        n += 1
        if not _same(getattr(c, k), want):
            bad.append(f'G-D0: dewarp.{k} = {getattr(c, k)!r} but phase2_config gives {want!r}')
    if import_heavy:
        from exopipe.phase2 import cascade_dewarp as CD
        from exopipe.phase3.lib.masters import pb_apply as PB, membership as MB
        for what, have, want in (('cascade_dewarp.ANCHOR_FALLBACK', tuple(CD.ANCHOR_FALLBACK), c.ANCHOR_FALLBACK),
                                 ('pb_apply.PB_FLOOR', PB.PB_FLOOR, c.PB_FLOOR),
                                 ('membership.DEWARP_INTERP', MB.DEWARP_INTERP, c.RESAMPLE_METHOD)):
            n += 1
            if not _same(have, want):
                bad.append(f'G-D0: {what} = {have!r} but the section says {want!r}')
        n += 1
        prod = os.path.join(CFG.paths.PHASE3_ARCHIVE_ROOT, c.SCREEN_RECORD_DIR, CFG.paths.ARCHIVE_RUN, c.SCREEN_RECORD_FILE)
        if MB.DEWARP_ACCEPTED != prod:
            bad.append(f'G-D0: membership.DEWARP_ACCEPTED = {MB.DEWARP_ACCEPTED} (read by verify_products) but the section '
                       f'names the screen record {prod}')
    root = branch_root().rstrip('/') + '/'
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, '__file__', None)
        if name.startswith('exopipe') and f and not os.path.abspath(f).startswith(root):
            bad.append(f'G-D0: module {name} is loaded from {f}, outside this branch ({root}) - the in-place '
                       f'sys.path.insert of /lustre/gh/main/exopipe won (spec B5)')
    return bad, n


def gate_threads():
    """G-D8 (measure_on_09.sh:13): the measurement runs with single-thread BLAS. The thread variables must equal
    MEASURE_THREADS (cli --threads N); threadpoolctl must be importable to cap the pools numpy already loaded (config import
    loads numpy before cli applies --threads: spec blocker B3). Returns violations."""
    want = str(C().MEASURE_THREADS)
    bad = [f'G-D8: {v}={os.environ.get(v)!r}, the measurement needs {want} (launch with --threads {want})'
           for v in THREAD_VARS if os.environ.get(v) != want]
    try:
        import threadpoolctl  # noqa: F401
    except ImportError:
        bad.append('G-D8: threadpoolctl is not importable - the BLAS pools numpy loaded before --threads applied cannot be '
                   'capped')
    return bad


def measurement_currency(roots, cell, row):
    """G-A11 (the step's own gate; only with --measurement): None when the measurement can be of the routed cell's current
    input, else the reason. Refuses a final row measured BEFORE the cell's rebuild passed T1-T8 (the UTC stamp on line 2 of
    <REBUILD_CHECK_ROOT>/<cell>/REBUILD_CONTENT_OK), and an ALREADY DEWARPED row while the rebuilt input carries no MDWARP
    (the measurement saw the installed master, not the rebuild). dewarp_lst.sh always measures fresh, so production never
    meets either case; a missing row is left to the acceptance rule (not accepted, 'not measured')."""
    if not row:
        return None
    try:
        lines = open(roots.rebuild_ok(cell)).read().splitlines()
    except OSError:
        lines = []
    m = re.search(r'(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ)', lines[1] if len(lines) > 1 else '')
    meas_at = (row.get('measured_at') or '').strip()
    if m and meas_at and meas_at < m.group(1):
        return (f"G-A11: the final row of {cell} was measured {meas_at}, before its rebuild passed T1-T8 ({m.group(1)}): "
                f"the measurement is not of this input - re-measure (drop --measurement)")
    if str(row.get('status', '')).startswith('ALREADY DEWARPED') and not carries_mdwarp(roots.promoted_master(cell)):
        return (f'G-A11: the final row of {cell} says ALREADY DEWARPED but its rebuilt input carries no MDWARP: the '
                f'measurement saw the installed master, not the rebuild - re-measure (drop --measurement)')
    return None


def install_meta_rebuild(roots, cell):
    return {'kind': 'rebuild', 'gen': os.path.join(roots.rebuild, cell)}


def render_not_accepted(row):
    """The applied=0 status text (dewarp_lst.sh:82-83): r.get(key, default) on the final row (an empty dict when the cell has
    no row), cut to SCREEN_STATUS_MAX_CHARS."""
    c = C()
    r = row or {}
    st = c.SCREEN_NOT_ACCEPTED_STATUS.format(status=r.get('status', 'not measured'),
                                             anchor_off_med_as=r.get('anchor_off_med_as', '?'),
                                             cv_resid_med_as=r.get('cv_resid_med_as', '?'))
    return st[:c.SCREEN_STATUS_MAX_CHARS]


def not_accepted_row(lst, scope, row, written_utc):
    """The applied=0 row of mdewarp_screens.csv (dewarp_lst.sh:84-88; columns phase3_install.SCREEN_COLS)."""
    r = row or {}
    return dict(lst=lst, scope=scope, applied=0, status=render_not_accepted(row), n_anchors=r.get('n_anchors', ''),
                vlssr_med_before_arcmin=r.get('med_before', ''), vlssr_med_after_arcmin=r.get('med_after', ''),
                heldout_resid_med_arcsec=r.get('cv_resid_med_as', ''), anchor_off_med_arcsec=r.get('anchor_off_med_as', ''),
                screen_p50_arcsec='', screen_p95_arcsec='', screen_file='', screen_md5='',
                measured_at=r.get('measured_at', ''), written_utc=written_utc)


def jsonable(v):
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [jsonable(x) for x in v]
    return str(v)
