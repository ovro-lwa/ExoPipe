"""Incremental deep catalogue: `exopipe3 catalog --tier deep --incremental --epochs <LST>_<date>` (DESIGN 12.1, 12.3).

Adds ONE epoch (one LST hour of one night: every band and subband, both Stokes) to the cumulative deep catalogue, with
results IDENTICAL to a full rebuild over the same epochs. steps/catalog.py calls ``add_arguments(p)`` for the --inc-*
flags and ``run_incremental(args, run, epochs)`` when --incremental is given.

    exopipe3 catalog --tier deep --incremental --epochs 04h_2025-10-14 --inc-parents /fast/.../p3s5_parents_union_X --out R
    exopipe3 catalog --tier deep --incremental --epochs @hours.txt --inc-phase epoch     # the per-epoch stage only
    exopipe3 forced ...                                                                   # forced photometry (its own step)
    exopipe3 catalog --tier deep --incremental --inc-phase identity --out R               # union, identity, localizations, list
    exopipe3 catalog --tier deep --incremental --inc-rebuild --out R                      # canonical FULL rebuild of the same epochs

PHASES (one record unit each; `--inc-phase`, default catalog_incremental.PHASES_DEFAULT = epoch,associate):
  epoch      unit incremental.epoch.<LST>_<date>: the epoch's rows of the merge output (per-key merge gate already applied),
             its ledger / images / refused rows and every frame of that night under the LST's cells are stored as an EPOCH
             BLOCK <out>/_incremental/epochs/<E>/ (class epoch_block), and its detections are FITTED on their searched frames
             (deep_fit_positions.py in place, R-GREGG-20261012-FITTED-DETECTION-POSITIONS-REQUIRED) - the expensive per-row
             work, done once per epoch.
  associate  unit incremental.associate: every block is assembled into the run's merge output in CANONICAL order
             (catalog_incremental.CANONICAL_ROW_ORDER), the population-coupled stages are REPLAYED in place exactly as
             p3s3_pass.sh runs them (conform, fitpos - every fit a cache hit -, select, ephemeris, split, the four catalogue
             builds, the multiples fork), and the ASSOCIATION ENGINE below updates the cumulative source catalogue from the
             replayed selection: only pairs involving added or changed detections are tested, components touched by a removal
             are re-associated locally. It writes the per-key state, the per-source DELTA (new / extended / merged / split /
             changed / removed / unchanged, renamed flag - what `pages --incremental` re-renders) and cross-checks the builder's
             member partition and names against the engine (classes incremental_state, incremental_partition,
             incremental_delta).
  identity   unit incremental.identity (after `exopipe3 forced`, gate D-G22): the union of the four primary tiers (ported
             p3s3_pass.sh PYU0), one identity per source (p3s3_identity.py), localizations, the transient list.

WHY THIS IS IDENTICAL TO A FULL REBUILD (the order-dependence analysis is in the impl note, catalog-incremental.md):
  1. Order: today's chain is order-dependent (union-find group order = first member in input order -> output row order,
     ranks / review_dir among equal |best_snr| (2,900 tied survivors in Run_P3S5 I_wb Red alone); `best` = first of equal
     |snr| (3 groups in V_wb); np.average summation order; the a < b orientation of every pair separation; identity
     tie-breaks). The assembled input is a pure function of the SET of epochs (all rows sorted by a total key), so every
     order-dependent step sees the same order however the epochs arrived.
  2. Coupling: conformance is NOT epoch-local (the bright-parent table S uses the cell's median frame rms over every night,
     p3s2_conform.py:220-223; the V 3x noise gate uses the cell's median THERMNOI over every V frame present, :144-159; the
     pulsar consistency groups span cells, :361-460), and every builder statistic is population-wide (BEAM_AS by the LST set,
     footprint AREA/RHO, nsearched, repeat-significance density, E floors, n_ge, other_bands, the sigma-stack singleton
     gate that a new epoch can LIFT, gate78, ranks). Those stages are therefore replayed over the cumulative set; only the
     epoch-local stages (merge gate per key, fitted positions per row, selection |snr| >= 5 and the verdict join, ephemeris
     per row, split per row) are computed for the epoch alone and cached. Same code + same input bytes = same products.
  3. Association: the partition is the connected components of the pair graph `sep < K_ASSOC x hypot(tot_a, tot_b)`; a
     component set is a function of the node set only (proof in the impl note), so the engine's incremental update equals the
     builder's full union-find. The engine calls the LIVE builder's own functions (_stat_as, _sys_det, jname, unique_names,
     assoc_groups, _sstack_fails - loaded from build_source_catalogs_v3.py by AST, never copied) and orients each pair as the
     builder does, so the cross-check is exact, not approximate.

COST (be clear about what is incremental today): the epoch phase does the per-row work once per epoch (the merge-gate rows
and the fitted positions: 1,329 s for the 141,964 rows of the 2026-10-13 Run_P3S5 pass). The associate phase still replays
conformance, selection, the four builds and the fork over the WHOLE cumulative set, because their statistics are global (see
point 2). Only the fits come from cache. The association itself is incremental (engine). The products of the window are
written by the in-place builder over the full set, so they are identical by construction, and the engine's state, delta and
cross-check prove the incremental association equals them. Making the builder's per-source derivation run only on the
components the engine marks as touched is an absorption task (impl note, NOT DONE).

In-place calls go through core.legacy.run (site interpreter 'default', inherited P3_* cleared, parity guard, child code
fingerprint in the unit record). The split and union ports are catalog-deep's (lib/catalog/deep_ported.py). Every value the
full chain also uses is read from the catalog-deep sections (``_deep``). Writes only under --out.
"""
import ast
import csv
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import sys
import time
from collections import Counter, OrderedDict, defaultdict

from exopipe.phase3.core import cells as _cells
from exopipe.phase3.core.record import NoInput, Refusal

STATE_VERSION = 1
ENGINE_VERSION = 'incremental-uf-1'       # bump when the engine's association code changes: the state is then rebuilt
PHASES = ('epoch', 'associate', 'identity')
CHANGES = ('new', 'extended', 'unchanged', 'changed', 'merged', 'split', 'removed')
DET_FIELDS = ('cand_id', 'band', 'lst', 'date', 'stokes', 'ra', 'dec', 'snr', 'stat', 'sys', 'sstack', 'comp')
SOURCE_FIELDS = ('band', 'comp', 'source_id', 'ra', 'dec', 'n_det', 'best_cand_id', 'members', 'sstack_removed')
DELTA_FIELDS = ('band', 'change', 'renamed', 'source_id', 'prev_source_ids', 'n_det', 'n_added', 'n_removed', 'comp')
BLOCK_FILES = ('master_catalog.csv', 'refused.csv', 'ledger.csv', 'images.csv', 'frames.csv', 'fitcache.csv', 'epoch.json')
FIT_KINDS = ('gaussian', 'template', 'peak')      # deep_fit_positions.KINDS (:41); used by the epoch_block content test
# p3s2_conform.py's per-run frame caches, keyed by (stokes, lst, band, date) and never re-read for a key already cached
# (:106-124 frame_beam_ratio, :125-147 frame_beam_bmaj (D8, ops 283a513), :349-360 frame_times). An incremental workspace keeps
# them across associate runs; a key whose frame changed since (a re-extracted epoch: new attempt, new frame) must not be served
# from the cache, or the workspace would differ from a full rebuild into an empty --out (review 2026-10-13).
FRAME_CACHES = ('frame_times.csv', 'frame_beam_ratio.csv', 'frame_beam_bmaj.csv')
FRAME_CACHE_RECORD = 'frame_cache_frames.json'
_FRAME_RE = re.compile(r'(\d{4}-\d\d-\d\d)_([IV])_redo_resid_(\w+)\.fits$')
# environment variables of the caller that an in-place stage would read and that core.legacy.run does not clear (it clears
# P3_*): refused at start unless the stage sets them itself (each changes a result silently)
LEAK_PREFIXES = ('P3S', 'FORK_', 'BDSF_')
STAGE_SET_VARS = ('P3S_LABEL', 'P3S_RUN_DIR', 'FORK_RUN_DIR', 'FORK_OUT_DIR', 'FORK_NPROC')

csv.field_size_limit(1 << 30)


# =====================================================================================================================
# small helpers
# =====================================================================================================================

def _cfg():
    from exopipe.phase3.config import CFG
    return CFG


def C():
    """The config section of this unit."""
    return _cfg().catalog_incremental


def _deep(section, key, entry=None):
    """A value the full deep chain also uses, read from the step owner's section (one place per value): refuses when the
    section or key does not exist (the fold-in renamed it - update incremental.py, never guess)."""
    sec = getattr(_cfg(), section, None)
    if sec is None or not hasattr(sec, key):
        raise Refusal(f'config {section}.{key} is not defined (catalog-deep section); the incremental path reads it')
    v = getattr(sec, key)
    if entry is not None:
        if not isinstance(v, dict) or entry not in v:
            raise Refusal(f'config {section}.{key}[{entry}] is not defined')
        v = v[entry]
    return v


def _ops(name):
    """An exopipe_ops script or module named in config, as a path under the site legacy root (core.env.legacy_root)."""
    if os.path.isabs(name):
        return name
    from exopipe.phase3.core import env
    try:
        return os.path.join(env.legacy_root('exopipe_ops'), name)
    except KeyError as e:
        raise Refusal(f'{e} - the incremental path calls exopipe_ops modules in place (site LEGACY_CODE_ROOTS)')


def _utc(fmt='%Y-%m-%dT%H:%M:%SZ'):
    return time.strftime(fmt, time.gmtime())


def _sha1_file(path):
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


def _read_csv(path):
    """(header, rows as lists of strings) - values are kept as read, so a rewrite is byte-faithful to the values."""
    with open(path, newline='') as f:
        rd = csv.reader(f)
        try:
            hdr = next(rd)
        except StopIteration:
            return [], []
        return hdr, [r for r in rd]


def _write_csv(path, header, rows):
    """Atomic CSV write (csv.writer default dialect, as the production modules write)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f'{path}.tmp.{os.getpid()}'
    with open(tmp, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    os.replace(tmp, path)


def _write_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f'{path}.tmp.{os.getpid()}'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=1, sort_keys=True, default=str)
    os.replace(tmp, path)


def _load_module_from(path, name):
    """Import a legacy module from its file IN PLACE (DESIGN 11.1). Re-uses sys.modules[name] when it is that file."""
    from exopipe.phase3.core.legacy import refuse_patch_path
    refuse_patch_path(path, 'catalog.incremental._load_module_from')
    m = sys.modules.get(name)
    if m is not None and os.path.realpath(getattr(m, '__file__', '') or '') == os.path.realpath(path):
        return m
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if name not in sys.modules:
        sys.modules[name] = mod            # so core/record.code_fingerprint identifies it (legacy root)
    return mod


# =====================================================================================================================
# epochs and the canonical order
# =====================================================================================================================

def parse_epoch(item):
    """'04h_2025-10-14' -> ('04h', '2025-10-14') (core.cells.parse_epoch, the per-hour item kind); else ValueError."""
    return _cells.parse_epoch(str(item).strip())


def _is_epoch(name):
    try:
        parse_epoch(name)
        return True
    except ValueError:
        return False


def epoch_order_key(item):
    lst, date = parse_epoch(item)
    return (date, lst)


def expand_epochs(spec):
    """Epoch items from a list or an item spec (04h_2025-10-14,05h_2025-10-14 or @file), validated, canonical order."""
    if spec is None or spec == '':
        return []
    if isinstance(spec, (list, tuple)):
        items = []
        for s in spec:
            items += expand_epochs(s) if (isinstance(s, str) and (',' in s or s.startswith('@'))) else [str(s).strip()]
    else:
        items = _cells.expand_epochs(spec)
    for e in items:
        parse_epoch(e)
    return sorted(set(items), key=epoch_order_key)


def row_order_key(row, ix):
    """Canonical order of a merge-output row (catalog_incremental.CANONICAL_ROW_ORDER); ``ix`` maps column -> index."""
    return tuple(row[ix[c]] for c in C().CANONICAL_ROW_ORDER)


# =====================================================================================================================
# the builder's own functions, in place (AST; never copied)
# =====================================================================================================================

_BUILDER_CACHE = {}


def builder_namespace(path=None):
    """The association / naming functions and constants of the LIVE build_source_catalogs_v3.py, loaded by AST (the module
    is a script: importing it would run a build). Each function body is the file's own; its globals are the builder's (the
    in-place p3s3_astrom as _astrom, the deep tier: _PERTIE None). The file's sha1 and each function's source sha1 are kept
    in the namespace (__sha1__, __segments__) for the record. Refuses when a named function or constant is missing."""
    c = C()
    path = path or _ops(c.BUILDER)
    with open(path) as f:
        src = f.read()
    sha = hashlib.sha1(src.encode()).hexdigest()
    hit = _BUILDER_CACHE.get(path)
    if hit is not None and hit['__sha1__'] == sha:
        return hit
    tree = ast.parse(src, path)
    want_f, want_c = set(c.BUILDER_FUNCS), set(c.BUILDER_CONSTS)
    body, seen, seg = [], set(), {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in want_f:
            body.append(node)
        elif (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
              and node.targets[0].id in want_c):
            body.append(node)
        else:
            continue
        name = node.name if isinstance(node, ast.FunctionDef) else node.targets[0].id
        seen.add(name)
        seg[name] = hashlib.sha1((ast.get_source_segment(src, node) or '').encode()).hexdigest()[:12]
    missing = sorted((want_f | want_c) - seen)
    if missing:
        raise Refusal(f'the builder {path} no longer defines {missing} at module level - the incremental association '
                      f'calls them in place (catalog_incremental.BUILDER_FUNCS / BUILDER_CONSTS); update the config')
    import numpy as np
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    astrom = _load_module_from(_ops(c.ASTROM_MODULE), 'p3s3_astrom')
    ns = {'__name__': 'build_source_catalogs_v3__inplace', 'np': np, 'math': math, 'os': os, 'SkyCoord': SkyCoord, 'u': u,
          'defaultdict': defaultdict, '_astrom': astrom, '_PERTIE': None, '_SP': None, 'SSTACK_THR': None}
    exec(compile(ast.Module(body=body, type_ignores=[]), path, 'exec'), ns)
    ns['__sha1__'] = sha
    ns['__segments__'] = seg
    ns['__path__'] = path
    _BUILDER_CACHE[path] = ns
    return ns


def binding_check(ns):
    """Refuse when the builder's constants disagree with config, or the process environment carries a switch the
    builder's functions read (they would associate differently from the deep build)."""
    CFG = _cfg()
    if float(ns['K_ASSOC']) != float(CFG.astrometry.K_ASSOC):
        raise Refusal(f"binding: {ns['__path__']} K_ASSOC = {ns['K_ASSOC']} but config astrometry.K_ASSOC = "
                      f"{CFG.astrometry.K_ASSOC} (R-ASSOCIATION-RADIUS-QUADRATURE) - refusing")
    for v in ('P3_SYS_AS', 'P3_TENMIN_PERTIE'):
        if os.environ.get(v):
            raise Refusal(f'binding: {v} is set in this process; the deep association reads it (_sys_det) - unset it')


def sstack_threshold():
    """(threshold, source) of the sigma-stack singleton gate exactly as the builder resolves it (CONFIG['SIGMA_THRESHOLD']
    of the production search_i_transients, :613-617), read by AST; the builder's own fallback when unreadable (B-30)."""
    c = C()
    try:
        tree = ast.parse(open(c.SSTACK_SOURCE).read())
        for node in tree.body:
            if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == 'CONFIG' and isinstance(node.value, ast.Dict)):
                for k, v in zip(node.value.keys, node.value.values):
                    if isinstance(k, ast.Constant) and k.value == 'SIGMA_THRESHOLD':
                        return float(ast.literal_eval(v)), f'{c.SSTACK_SOURCE} CONFIG[SIGMA_THRESHOLD]'
    except (OSError, SyntaxError, ValueError):
        pass
    from exopipe.phase3.core import fallback      # WAVE2 B-30
    fallback.site('fallbacks.CATALOG_SSTACK_THRESHOLD_FALLBACK', f'{c.SSTACK_SOURCE} unreadable or without CONFIG[SIGMA_THRESHOLD] - '
                  f'the sigma-stack singleton gate would use {c.SSTACK_FALLBACK} (the builder\'s :617 fallback)')
    return float(c.SSTACK_FALLBACK), f'fallback {c.SSTACK_FALLBACK} ({c.SSTACK_SOURCE} unreadable; builder :617)'


# =====================================================================================================================
# the association engine (order-independent; incremental)
# =====================================================================================================================

def make_dets(ns, header, rows, beam_as):
    """Association inputs of selected rows, in the given (builder input) order: the builder's own _stat_as / _sys_det."""
    ix = {k: i for i, k in enumerate(header)}
    bdef = float(C().BEAM_DEFAULT_ARCSEC)
    out = []
    for r in rows:
        x = {k: r[i] for k, i in ix.items()}
        x['_snr'], x['_ra'], x['_dec'] = float(x['snr']), float(x['ra']), float(x['dec'])
        x['_beam_as'] = float(beam_as.get(x['band'], bdef))
        out.append({'cand_id': x['cand_id'], 'band': x['band'], 'lst': x['lst'], 'date': x['date'],
                    'stokes': x.get('stokes', ''), 'ra': x['_ra'], 'dec': x['_dec'], 'snr': x['_snr'],
                    'stat': float(ns['_stat_as'](x)), 'sys': float(ns['_sys_det'](x)), 'sstack': x.get('sstack', '') or ''})
    return out


class _UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, i):
        p = self.p
        while p[i] != i:
            p[i] = p[p[i]]
            i = p[i]
        return i

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


class BandArrays:
    """Positions and errors of one band's detections in builder input order, with the two SkyCoords the builder's
    search_around_sky uses: the pair (a, b), a < b, is separated as c_t[a].separation(c[b]) (astropy matching.py:
    coords1 = coords1.transform_to(coords2); d2ds = coords1[idx1].separation(coords2[idx2])), so the exact test here is
    bit-for-bit the builder's."""

    def __init__(self, dets):
        import numpy as np
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        self.n = len(dets)
        self.ra = np.array([d['ra'] for d in dets], dtype=float)
        self.dec = np.array([d['dec'] for d in dets], dtype=float)
        st = [d['stat'] for d in dets]
        sy = [d['sys'] for d in dets]
        self.tot = np.sqrt(np.asarray(st) ** 2 + np.asarray(sy) ** 2)     # build_source_catalogs_v3.py:589
        r, de = np.radians(self.ra), np.radians(self.dec)
        self.v = np.stack([np.cos(de) * np.cos(r), np.cos(de) * np.sin(r), np.sin(de)], -1) if self.n else np.zeros((0, 3))
        if self.n:
            self.c = SkyCoord(list(self.ra) * u.deg, list(self.dec) * u.deg)
            self.c_t = self.c.transform_to(self.c)

    def passing_pairs(self, query, among, K):
        """Pairs (a, b), a < b, a or b in ``query``, both in ``among`` (index lists), with sep < K x hypot(tot_a, tot_b)."""
        import numpy as np
        from scipy.spatial import cKDTree
        query = np.array(sorted(set(int(i) for i in query)), dtype=int)
        among = np.array(sorted(set(int(i) for i in among)), dtype=int)
        if not len(query) or not len(among):
            return []
        tmax = float(self.tot[among].max())
        pad = float(C().PAIR_SEARCH_PAD)
        rdeg = K * np.sqrt(self.tot[query] ** 2 + tmax ** 2) / 3600.0 * pad
        chord = 2.0 * np.sin(np.radians(np.minimum(rdeg, 180.0)) / 2.0)
        tree = cKDTree(self.v[among])
        cand = set()
        for qi, hits in zip(query, tree.query_ball_point(self.v[query], chord)):
            for k in hits:
                j = int(among[k])
                if j != qi:
                    cand.add((min(qi, j), max(qi, j)))
        if not cand:
            return []
        pr = np.array(sorted(cand), dtype=int)
        lo, hi = pr[:, 0], pr[:, 1]
        sep = self.c_t[lo].separation(self.c[hi]).arcsec
        ok = sep < K * np.hypot(self.tot[lo], self.tot[hi])
        return [(int(a), int(b)) for a, b in zip(lo[ok], hi[ok])]


def _labels(uf, dets):
    """Component label of each detection: the cand_id of its component's first member in builder input order."""
    first = {}
    out = []
    for i, d in enumerate(dets):
        r = uf.find(i)
        first.setdefault(r, d['cand_id'])
        out.append(first[r])
    return out


def associate_band_full(dets, K):
    """Full association of one band (reference): every pair tested."""
    arr = BandArrays(dets)
    uf = _UF(arr.n)
    allidx = list(range(arr.n))
    for a, b in arr.passing_pairs(allidx, allidx, K):
        uf.union(a, b)
    return _labels(uf, dets), {'mode': 'full', 'pairs_tested_for': arr.n}


def associate_band_incremental(prev, dets, K):
    """Update one band's association from the previous state ``prev`` (list of det dicts carrying 'comp') to ``dets``
    (the new selection of the band, builder input order). Exactly the connected components of the new pair graph:
      - a component of the previous state none of whose members was removed or changed keeps its internal edges (the
        criterion depends only on the two detections);
      - a component that lost (or had changed) a member is dissolved and its remaining members are re-associated among
        themselves (no edge can join them to an untouched component: it would have been one component before);
      - every added or changed detection is tested against every detection of the new set."""
    def crit(d):
        return (d['ra'], d['dec'], d['stat'], d['sys'])
    pmap = {d['cand_id']: d for d in prev}
    nidx = {d['cand_id']: i for i, d in enumerate(dets)}
    changed = {c for c, i in nidx.items() if c in pmap and crit(pmap[c]) != crit(dets[i])}
    removed = (set(pmap) - set(nidx)) | changed
    added = (set(nidx) - set(pmap)) | changed
    arr = BandArrays(dets)
    uf = _UF(arr.n)
    touched = {pmap[c]['comp'] for c in removed if c in pmap}
    by_comp = defaultdict(list)
    for d in prev:
        if d['cand_id'] not in removed and d['cand_id'] in nidx:
            by_comp[d['comp']].append(nidx[d['cand_id']])
    dissolved = []
    for comp, mem in by_comp.items():
        if comp in touched:
            dissolved.extend(mem)
            continue
        for j in mem[1:]:
            uf.union(mem[0], j)
    if dissolved:
        for a, b in arr.passing_pairs(dissolved, dissolved, K):
            uf.union(a, b)
    if added:
        for a, b in arr.passing_pairs([nidx[c] for c in added], range(arr.n), K):
            uf.union(a, b)
    return _labels(uf, dets), {'mode': 'incremental', 'added': len(added - changed), 'removed': len(removed - changed),
                               'changed': len(changed), 'dissolved_components': len(touched),
                               'dissolved_members': len(dissolved)}


def associate_band_builder(ns, dets, K):
    """The builder's OWN assoc_groups on the same input (the content tests' reference): labels per detection."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    if not dets:
        return []
    c_all = SkyCoord([d['ra'] for d in dets] * u.deg, [d['dec'] for d in dets] * u.deg)
    groups = ns['assoc_groups'](c_all, [d['stat'] for d in dets], [d['sys'] for d in dets], K=K)
    lab = [None] * len(dets)
    for g in groups:
        first = dets[min(g)]['cand_id']
        for i in g:
            lab[i] = first
    return lab


def derive_sources(ns, dets, labels, sstack_thr):
    """Sources of one band, as the builder forms them (:640-660, 743-810): groups in order of first member, members in
    input order, the sigma-stack singleton gate (Stokes I primary tiers: ``sstack_thr`` not None), inverse-variance
    position, the best (first max |snr|), the J-name and the unique-names rule. Refuses on a name clash (builder :434)."""
    import numpy as np
    groups = OrderedDict()
    for d, lab in zip(dets, labels):
        groups.setdefault(lab, []).append(d)
    out = []
    if sstack_thr is not None:
        ns['SSTACK_THR'] = sstack_thr
    for lab, mem in groups.items():
        removed = []
        if sstack_thr is not None and len({(m['lst'], m['date']) for m in mem}) == 1:
            keep = [m for m in mem if not ns['_sstack_fails']({'_snr': m['snr'], 'sstack': m['sstack']})]
            removed = [m['cand_id'] for m in mem if m not in keep]
            if not keep:
                continue
            mem = keep
        st = np.array([m['stat'] for m in mem])
        sy = np.array([m['sys'] for m in mem])
        w = 1.0 / np.maximum(np.hypot(st, sy), 1e-3) ** 2
        ra = float(np.average([m['ra'] for m in mem], weights=w))
        dec = float(np.average([m['dec'] for m in mem], weights=w))
        best = max(mem, key=lambda m: abs(m['snr']))
        out.append({'band': mem[0]['band'], 'comp': lab, 'source_id': ns['jname'](ra, dec), 'ra': round(ra, 5),
                    'dec': round(dec, 5), 'n_det': len(mem), 'best_cand_id': best['cand_id'],
                    'members': [m['cand_id'] for m in mem], 'sstack_removed': removed})
    if out:
        try:
            names = ns['unique_names']([s['source_id'] for s in out], [(s['ra'], s['dec']) for s in out])
        except SystemExit as e:
            raise Refusal(f'{out[0]["band"]}: {e}')
        for s, nm in zip(out, names):
            s['source_id'] = nm
    return out


def associate_key(ns, header, rows, beam_as, sstack_thr, prev_dets=None, K=None):
    """Association of one catalogue key (all its bands). ``prev_dets`` (band -> det list with 'comp') = the previous state:
    incremental update; None = full. Returns {'dets': band -> dets with 'comp', 'sources': band -> sources, 'stats'}."""
    K = float(ns['K_ASSOC'] if K is None else K)
    dets = make_dets(ns, header, rows, beam_as)
    byband = OrderedDict()
    for d in dets:
        byband.setdefault(d['band'], []).append(d)
    res = {'dets': OrderedDict(), 'sources': OrderedDict(), 'stats': {}}
    for band, bd in byband.items():
        if prev_dets is not None:
            lab, st = associate_band_incremental(prev_dets.get(band, []), bd, K)
        else:
            lab, st = associate_band_full(bd, K)
        for d, l in zip(bd, lab):
            d['comp'] = l
        res['dets'][band] = bd
        res['sources'][band] = derive_sources(ns, bd, lab, sstack_thr)
        res['stats'][band] = st
    return res


def partition_digest(dets_by_band):
    h = hashlib.sha1()
    for band in sorted(dets_by_band):
        for d in sorted(dets_by_band[band], key=lambda d: d['cand_id']):
            h.update(f"{band}|{d['cand_id']}|{d['comp']}\n".encode())
    return h.hexdigest()


def partition_sets(dets_by_band):
    """{band: set of frozensets of cand_ids} - the partition, independent of labels."""
    out = {}
    for band, dl in dets_by_band.items():
        g = defaultdict(set)
        for d in dl:
            g[d['comp']].add(d['cand_id'])
        out[band] = {frozenset(v) for v in g.values()}
    return out


def classify_delta(prev_sources, new_sources):
    """Per-source change between two states of one band (sources matched by MEMBER sets, never by name):
    new (no previous member), unchanged (same members), extended (previous members all kept, more added), changed
    (members added and removed), merged (members of 2+ previous sources), split (a previous source's members now in 2+
    sources), removed (a previous source with no member left); 'renamed' = the source_id differs from the previous one."""
    owner = {}
    for i, s in enumerate(prev_sources):
        for c in s['members']:
            owner[c] = i
    hits_of_prev = defaultdict(set)
    rows = []
    for j, s in enumerate(new_sources):
        prev_ids = sorted({owner[c] for c in s['members'] if c in owner})
        for i in prev_ids:
            hits_of_prev[i].add(j)
    for j, s in enumerate(new_sources):
        mem = set(s['members'])
        prev_ids = sorted({owner[c] for c in mem if c in owner})
        prev_mem = set().union(*[set(prev_sources[i]['members']) for i in prev_ids]) if prev_ids else set()
        n_add = len(mem - prev_mem)
        n_rem = len(prev_mem - mem)
        if not prev_ids:
            ch = 'new'
        elif len(prev_ids) > 1:
            ch = 'merged'
        elif len(hits_of_prev[prev_ids[0]]) > 1:
            ch = 'split'
        elif mem == prev_mem:
            ch = 'unchanged'
        elif prev_mem < mem:
            ch = 'extended'
        else:
            ch = 'changed'
        pids = [prev_sources[i]['source_id'] for i in prev_ids]
        rows.append({'band': s['band'], 'change': ch, 'renamed': int(bool(pids) and pids != [s['source_id']]),
                     'source_id': s['source_id'], 'prev_source_ids': ';'.join(pids), 'n_det': s['n_det'],
                     'n_added': n_add, 'n_removed': n_rem, 'comp': s['comp']})
    for i, s in enumerate(prev_sources):
        if not hits_of_prev.get(i):
            rows.append({'band': s['band'], 'change': 'removed', 'renamed': 0, 'source_id': '',
                         'prev_source_ids': s['source_id'], 'n_det': 0, 'n_added': 0, 'n_removed': len(s['members']),
                         'comp': ''})
    return rows


# ---- engine state files ---------------------------------------------------------------------------------------------

def _fmt(v):
    return repr(v) if isinstance(v, float) else str(v)


def write_engine_state(state_dir, K, res):
    dets_rows = [[_fmt(d[f]) for f in DET_FIELDS] for band in res['dets'] for d in res['dets'][band]]
    _write_csv(os.path.join(state_dir, f'engine_{K}.csv'), list(DET_FIELDS), dets_rows)
    src_rows = []
    for band in res['sources']:
        for s in res['sources'][band]:
            src_rows.append([s['band'], s['comp'], s['source_id'], repr(s['ra']), repr(s['dec']), str(s['n_det']),
                             s['best_cand_id'], ';'.join(s['members']), ';'.join(s['sstack_removed'])])
    _write_csv(os.path.join(state_dir, f'sources_{K}.csv'), list(SOURCE_FIELDS), src_rows)


def read_engine_dets(state_dir, K):
    """band -> det dicts (with 'comp') of a key's saved state, or None when there is none."""
    p = os.path.join(state_dir, f'engine_{K}.csv')
    if not os.path.exists(p):
        return None
    out = OrderedDict()
    with open(p, newline='') as f:
        for r in csv.DictReader(f):
            d = dict(r)
            for k in ('ra', 'dec', 'snr', 'stat', 'sys'):
                d[k] = float(d[k])
            out.setdefault(d['band'], []).append(d)
    return out


def read_engine_sources(state_dir, K):
    p = os.path.join(state_dir, f'sources_{K}.csv')
    out = OrderedDict()
    if not os.path.exists(p):
        return out
    with open(p, newline='') as f:
        for r in csv.DictReader(f):
            s = dict(r)
            s['members'] = [m for m in s['members'].split(';') if m]
            s['sstack_removed'] = [m for m in (s.get('sstack_removed') or '').split(';') if m]
            s['n_det'] = int(s['n_det'])
            out.setdefault(s['band'], []).append(s)
    return out


def builder_partition(cat_dir):
    """{band: ({frozenset(cand_ids)}, {frozenset: source_id})} from a catalogue directory's <band>_source_members.csv."""
    out = {}
    for p in sorted(os.listdir(cat_dir)) if os.path.isdir(cat_dir) else []:
        if not p.endswith('_source_members.csv'):
            continue
        band = p[:-len('_source_members.csv')]
        g = OrderedDict()
        with open(os.path.join(cat_dir, p), newline='') as f:
            for r in csv.DictReader(f):
                g.setdefault(r['source_id'], set()).add(r['cand_id'])
        out[band] = ({frozenset(v) for v in g.values()}, {frozenset(v): k for k, v in g.items()})
    return out


def crosscheck(res, cat_dir):
    """(ok, detail): the builder's member partition and names in ``cat_dir`` equal the engine's sources."""
    bp = builder_partition(cat_dir)
    bad = []
    n = 0
    for band in sorted(set(bp) | set(res['sources'])):
        eng = {frozenset(s['members']): s['source_id'] for s in res['sources'].get(band, [])}
        sets, names = bp.get(band, (set(), {}))
        n += len(eng)
        if set(eng) != sets:
            bad.append(f'{band}: partition differs ({len(set(eng) - sets)} engine-only, {len(sets - set(eng))} builder-only)')
            continue
        nb = sum(1 for k, v in eng.items() if names.get(k) != v)
        if nb:
            bad.append(f'{band}: {nb} source names differ')
    return (not bad), (f'{n} sources in {len(bp)} bands equal' if not bad else '; '.join(bad[:6]))


# =====================================================================================================================
# epoch blocks (the per-epoch stage) and the canonical assembly
# =====================================================================================================================

class Source:
    """Where epochs come from (catalog_incremental.EPOCH_SOURCE): 'run:<dir>' = a deep run's merge output."""

    def __init__(self, spec, label):
        c = C()
        spec = spec or c.EPOCH_SOURCE
        if spec == 'search':
            raise Refusal('--inc-source search: the per-hour search output layout (`exopipe3 search --epochs`) is not defined '
                          'yet; use run:<deep run dir> (the merge output of a deep run)')
        if spec == 'run':
            spec = 'run:' + (c.SOURCE_RUN or os.path.join(_cfg().paths.TRANSIENTS_ROOT, f'Run_{label}'))
        if not spec.startswith('run:'):
            raise Refusal(f'--inc-source {spec!r}: expected run:<deep run dir> or search')
        self.kind = 'run'
        self.dir = spec[4:].rstrip('/')
        self.spec = f'run:{self.dir}'
        self.files = {k: os.path.join(self.dir, v) for k, v in c.SOURCE_FILES.items()}
        self._cache = {}
        self.want = []             # every epoch requested in this invocation: each table is read ONCE for all of them

    def check(self):
        if not os.path.isdir(self.dir):
            raise Refusal(f'source run {self.dir} does not exist')
        if not os.path.exists(self.files['merged']):
            raise Refusal(f'source run {self.dir} has no merge output {os.path.basename(self.files["merged"])} '
                          f'(written by the conformance stage as a copy of the merge gate output)')

    def table(self, kind, epochs):
        """(header, {epoch: rows}) of one source table filtered to the (lst, date) of every epoch requested in this
        invocation (``self.want`` + ``epochs``); each table is read once per process (the deep ledger is 785 MB)."""
        need = set(self.want) | set(epochs)
        hit = self._cache.get(kind)
        if hit is not None and need <= hit[0]:
            return hit[1], hit[2]
        want = {parse_epoch(e): e for e in need}
        p = self.files[kind]
        hdr, by = [], {e: [] for e in need}
        if os.path.exists(p):
            with open(p, newline='') as f:
                rd = csv.reader(f)
                hdr = next(rd, [])
                ix = {k: i for i, k in enumerate(hdr)}
                if 'lst' not in ix or 'date' not in ix:
                    raise Refusal(f'{p}: no lst/date columns - not a deep run table')
                il, idt = ix['lst'], ix['date']
                for r in rd:
                    e = want.get((r[il], r[idt][:10]))
                    if e is not None:
                        by[e].append(r)
        self._cache[kind] = (need, hdr, by)
        return hdr, by

    def fit_cache_rows(self, cand_ids):
        """(header, rows) of the source run's fit cache whose key starts with one of ``cand_ids`` (read once)."""
        if 'fitcache' not in self._cache:
            p = self.files['fitcache']
            self._cache['fitcache'] = _read_csv(p) if os.path.exists(p) else ([], [])
        hdr, rows = self._cache['fitcache']
        want = set(cand_ids)
        return hdr, [r for r in rows if r and cache_cand_id(r[0]) in want]

    def frames(self, lst, date):
        """[(stokes, cell, file name, realpath)] of every file of that night in the run's vresid/iresid cell dirs of the LST
        (the attempt layout: frame + pbcorr sibling; conform's V-noise median globs all of them, p3s2_conform.py:146)."""
        out = []
        for st, sub in (('V', 'vresid'), ('I', 'iresid')):
            d = os.path.join(self.dir, sub)
            if not os.path.isdir(d):
                continue
            for cell in sorted(os.listdir(d)):
                if not cell.startswith(lst + '_'):
                    continue
                cd = os.path.join(d, cell)
                try:
                    names = sorted(os.listdir(cd))
                except OSError:
                    continue
                for fn in names:
                    if fn.startswith(date + '_') and fn.endswith('.fits'):
                        out.append((st, cell, fn, os.path.realpath(os.path.join(cd, fn))))
        return out


def cache_cand_id(key):
    """The cand_id of a deep_fit_positions.py cache key: '<cand>|<frame>|<mtime>|<ra>|<dec>|<snr>' (5ad0563) or, since
    exopipe_ops 554b721 (2026-10-13, fit cache keyed by the fitter's md5), '<fitver>|<cand>|<frame>|...': the 6th field from
    the end either way (a path holds no '|')."""
    parts = str(key).split('|')
    return parts[-6] if len(parts) >= 6 else ''


def frame_path(root, stokes, lst, band, date):
    """The searched frame of a detection (deep_fit_positions.frame_path, :44-46)."""
    sub = 'iresid' if stokes == 'I' else 'vresid'
    return os.path.join(root, sub, f'{lst}_{band}', f'{date}_{stokes}_redo_resid_{band}.fits')


def epochs_in_state(state_dir):
    d = os.path.join(state_dir, 'epochs')
    if not os.path.isdir(d):
        return []
    out = []
    for e in os.listdir(d):
        if _is_epoch(e) and os.path.exists(os.path.join(d, e, 'epoch.json')):
            out.append(e)
    return sorted(out, key=epoch_order_key)


def _link(target, dst):
    if os.path.islink(dst):
        if os.readlink(dst) == target:
            return
        os.unlink(dst)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    os.symlink(target, dst)


def _excluded_scopes(excl, lst, date, scopes):
    return sorted(s for s in scopes if excl.is_excluded(lst, s, date))


# =====================================================================================================================
# the replay of the population-coupled stages (in place, as p3s3_pass.sh)
# =====================================================================================================================

def stage_env(ctx, stage, extra=None):
    """The variables one in-place stage gets ON TOP of the environment core.legacy.run builds (the caller's environment with
    every P3_* cleared; leaking P3S*/FORK_*/BDSF_* variables are refused at start, ``env_leaks``): exactly what the
    production launchers set (fork_serve_p3s5_20261008.sh:31-32 for every stage; p3s3_pass.sh:42 conform; :57 select),
    with each value read from the step owner's config section."""
    env = {'P3S_LABEL': ctx.label, 'P3_CADENCE': 'deep',
           'P3_TABLE_ELEV': '1' if _deep('catalog_conform', 'TABLE_ELEV', 'deep') else '0'}
    if ctx.parents:
        env['P3_PARENT_DIR'] = ctx.parents
    if stage == 'conform':
        env['P3_CONFORM_SIZE'] = '1' if _deep('catalog_conform', 'SIZE_STEP', 'deep') else '0'
    if stage == 'select':
        if not _deep('catalog_select', 'PARENT_EXCLUSION'):
            env['P3_NO_PARENT_EXCL'] = '1'
    env.update(extra or {})
    return env


def _fit_nproc():
    return int(_deep('catalog_fitpos', 'NPROC', 'deep'))


def env_leaks():
    """Caller environment variables an in-place stage would read and core.legacy.run does not clear."""
    return sorted(k for k in os.environ if k.startswith(LEAK_PREFIXES) and k not in STAGE_SET_VARS)


def _ported():
    """The shell logic of the deep pass ported ONCE, by the step owner (lib/catalog/deep_ported.py: split heredoc PYS
    p3s3_pass.sh:66-76, union heredoc PYU0 :103-129); the incremental path uses the same functions."""
    try:
        from exopipe.phase3.lib.catalog import deep_ported
    except ImportError as e:
        raise Refusal(f'lib/catalog/deep_ported.py (catalog-deep) is not importable ({e}): the split and union ports live there')
    return deep_ported


def split_selection(U):
    """selected_all.csv -> selected_<S>_<tier>.csv (deep_ported.split_selection). Returns {'V_wb': n, ...}."""
    n = _ported().split_selection(U)
    return {f'{k[0]}_{k[1]}': v for k, v in n.items()}


def union_tiers(U):
    """The union of the primary tiers into source_catalogs/ (deep_ported.union_tiers). The files of each tier are the ones
    its catalogue directory holds now: every directory was built fresh by this workspace's last build (no previous-pass
    file can be present, ``_build_stage``)."""
    tiers = tuple(_deep('catalog_identity', 'UNION_TIERS'))
    kinds = tuple(_deep('catalog_identity', 'UNION_KINDS'))
    # review 2026-10-13: the previous union files are moved aside first (as the deep chain does, D9): a band that has left
    # every primary tier would otherwise keep its old union file, which p3s3_identity.py globs and rewrites
    outd = f'{U}/source_catalogs'
    olds = [f for f in sorted(os.listdir(outd)) if any(f.endswith(f'_{k}.csv') for k in kinds)] if os.path.isdir(outd) else []
    if olds:
        sup = os.path.join(U, C().STATE_DIR, 'superseded', f'union_{_utc("%Y%m%dT%H%M%SZ")}')
        os.makedirs(sup, exist_ok=True)
        for f in olds:
            os.replace(os.path.join(outd, f), os.path.join(sup, f))
    files = {}
    for K in tiers:
        d = f'{U}/source_catalogs_{K}'
        files[K] = [os.path.join(d, f) for f in sorted(os.listdir(d))] if os.path.isdir(d) else []
    return _ported().union_tiers(U, files, tiers, kinds)


class _Ctx:
    """Resolved settings of one incremental invocation."""

    def __init__(self, args, run):
        c = C()
        self.args, self.run = args, run
        self.out = os.path.abspath(run.out_root)
        self.dry = bool(run.dry_run)
        self.label = getattr(args, 'run_label', None) or _deep('catalog', 'DEEP_RUN_LABEL')
        self.state_dir = os.path.join(self.out, c.STATE_DIR)
        ph = getattr(args, 'inc_phase', None)
        self.phases = tuple(p.strip() for p in ph.split(',') if p.strip()) if ph else tuple(c.PHASES_DEFAULT)
        bad = [p for p in self.phases if p not in PHASES]
        if bad:
            raise Refusal(f'--inc-phase {ph!r}: unknown phase(s) {bad}; phases: {", ".join(PHASES)}')
        par = getattr(args, 'inc_parents', None) or getattr(args, 'parents', None) or c.PARENT_DIR
        self.parents = os.path.abspath(par) if (par and par != 'snapshot') else None
        self.rebuild = bool(getattr(args, 'inc_rebuild', False) or run.fresh)
        self.source_spec = getattr(args, 'inc_source', None)
        self._source = None

    @property
    def source(self):
        if self._source is None:
            self._source = Source(self.source_spec, self.label)
        return self._source

    def say(self, text):
        print(text, flush=True)

    def log(self, name):
        return os.path.join(self.out, name)

    def run_stage(self, u, stage, argv, env, log_name, cwd=None, append=False):
        """Run one in-place stage through core.legacy.run (site interpreter 'default', P3_* cleared, parity guard, child
        code fingerprint into the unit record). ``argv[0]`` = the script (config name or path), the rest its arguments.
        Any exit other than 0 is a Refusal naming the stage, the module's reason and the log tail (D10: the pass aborts).
        ``append`` False truncates the log first, as production's `> log` does."""
        from exopipe.phase3.core import legacy
        c = C()
        target, args = _ops(str(argv[0])), [str(a) for a in argv[1:]]
        lp = self.log(log_name)
        if self.dry:
            try:
                py = legacy.interpreter('default')
            except Refusal as e:
                py = f'<{e}>'
            self.say(f'  [{stage}] {py} {target} {" ".join(args)}   (cwd {cwd or self.out}; log {lp}; env '
                     + ' '.join(f'{k}={v}' for k, v in sorted(env.items())) + ')')
            return 0
        os.makedirs(os.path.dirname(lp), exist_ok=True)
        if not append:
            open(lp, 'w').close()
        res = legacy.run(target, args, python='default', env=env, cwd=cwd or self.out, log=lp, unit=u,
                         timeout=c.STAGE_TIMEOUT_S)
        u.note(f'stage {stage}: exit {res.rc} in {res.wall_s:.0f} s ({lp})')
        if not res.ok:
            tail = ' | '.join(l.strip() for l in (res.tail or '').splitlines()[-4:])
            raise Refusal(f'stage {stage} {"refused" if res.refused else "failed"} ({res.reason}; {lp}): {tail[-600:]}')
        return res.rc


def _resumed(ctx, u, admitted=None):
    """--resume: when this unit's record is done and its inputs, admitted nights, code and config are unchanged and every
    output passes its content test now (core/record.resume_check), re-record the outputs and skip the work."""
    if not ctx.run.resume or ctx.dry:
        return False
    from exopipe.phase3.core import record
    ok, why = ctx.run.resume_check(u.name, inputs=None, admitted=admitted)
    if not ok:
        u.note(f'resume: not skipped ({why})')
        return False
    rec = record.load_record(ctx.out, ctx.run.step, u.name) or {}
    for o in rec.get('outputs', []):
        u.add_output(o['path'], o['class'], test=False)
    u.note(f'resume: skipped - {why}')
    ctx.say(f'{u.name}: skipped (resume: {why})')
    return True


def _add_output(u, path, cls, **tctx):
    """Record an output with its class's content test; a class whose test is not registered yet (owned by the step
    owner's lib/catalog/tests.py) is recorded untested WITH a note (never silently)."""
    from exopipe.phase3.core import validate
    try:
        validate.tests_for(cls)
        ok = True
    except KeyError:
        ok = False
    if ok:
        return u.add_output(path, cls, **tctx)
    u.note(f'output {path}: class {cls} has no registered content test yet (catalog-deep owns it) - recorded untested')
    return u.add_output(path, cls, test=False)


# =====================================================================================================================
# phase: epoch
# =====================================================================================================================

def _epoch_unit(ctx, epoch):
    from exopipe.phase3.core import exclusions
    c = C()
    lst, date = parse_epoch(epoch)
    block = os.path.join(ctx.state_dir, 'epochs', epoch)
    with ctx.run.unit(f'incremental.epoch.{epoch}', epoch=epoch, label=ctx.label, source=ctx.source.spec) as u:
        src = ctx.source
        src.check()
        for k in ('merged', 'refused', 'images'):
            if os.path.exists(src.files[k]):
                u.add_input(src.files[k])
        excl = exclusions.load()
        u.meta['exclusions'] = excl.fingerprint()
        hdr, by = src.table('merged', [epoch])
        rows = by.get(epoch, [])
        ix = {k: i for i, k in enumerate(hdr)}
        frames = src.frames(lst, date)
        scopes = sorted({r[ix['band']] for r in rows} | {f[1].split('_', 1)[1] for f in frames})
        xs = _excluded_scopes(excl, lst, date, scopes)
        u.set_admitted([date] if set(scopes) - set(xs) else [])
        if _resumed(ctx, u, u.admitted):
            return
        if not rows and not frames:
            raise NoInput(f'{epoch}: no detections and no frames in {src.dir} (nothing searched for this hour)')
        if scopes and set(xs) >= set(scopes):
            raise Refusal(f'{epoch}: every scope searched that night is excluded now ({", ".join(xs)}; data_decisions.csv) - '
                          f'an excluded epoch is never added (CLAUDE.md 2)')
        cids = Counter(r[ix['cand_id']] for r in rows)
        dup = [k for k, v in cids.items() if v > 1]
        if dup:
            raise Refusal(f'{epoch}: {len(dup)} duplicate cand_id(s) in the merge output, e.g. {dup[:3]}')
        have = {(f[0], f[1], f[2]) for f in frames}
        miss = sorted({(r[ix['stokes']], f"{lst}_{r[ix['band']]}") for r in rows
                       if (r[ix['stokes']], f"{lst}_{r[ix['band']]}",
                           f"{date}_{r[ix['stokes']]}_redo_resid_{r[ix['band']]}.fits") not in have})
        nrow_x = sum(1 for r in rows if r[ix['band']] in xs)
        ctx.say(f'epoch {epoch}: {len(rows)} detections of the merge output ({src.spec}); '
                f'{sum(1 for r in rows if r[ix["stokes"]] == "V")} V / {sum(1 for r in rows if r[ix["stokes"]] == "I")} I '
                f'on {len({r[ix["band"]] for r in rows})} scopes; {len(frames)} frame files; excluded now: '
                f'{", ".join(xs) or "none"} ({nrow_x} rows, removed at assembly)')
        if miss:
            raise Refusal(f'{epoch}: {len(miss)} (Stokes, cell) with detections but no searched frame in {src.dir}, '
                          f'e.g. {miss[:3]} (the fitted-position stage would refuse: deep_fit_positions.py:96-102)')
        if ctx.dry:
            ctx.say(f'  would write the epoch block {block}/ ({", ".join(BLOCK_FILES)}) [class epoch_block] and fit '
                    f'{len(rows)} detections on their frames ({_ops(c.STAGE_SCRIPTS["fitpos"])} --nproc {_fit_nproc()}; '
                    f'cache seeded from {src.files["fitcache"] if c.SEED_FIT_CACHE_FROM_SOURCE else "nothing"})')
            u.reason = 'dry run: plan printed, nothing written'
            return
        tmp = f'{block}.tmp.{os.getpid()}'
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp)
        _write_csv(os.path.join(tmp, 'master_catalog.csv'), hdr, rows)
        for kind, name in (('refused', 'refused.csv'), ('ledger', 'ledger.csv'), ('images', 'images.csv')):
            if kind == 'ledger' and os.path.exists(src.files['ledger']):
                u.add_input(src.files['ledger'])
            h2, by2 = src.table(kind, [epoch])
            _write_csv(os.path.join(tmp, name), h2 or ['lst', 'date'], by2.get(epoch, []))
        _write_csv(os.path.join(tmp, 'frames.csv'), ['stokes', 'cell', 'file', 'target'], [list(f) for f in frames])
        for st, cell, fn, tgt in frames:
            _link(tgt, os.path.join(tmp, 'vresid' if st == 'V' else 'iresid', cell, fn))
        fit_out = os.path.join(tmp, 'fitted.csv')
        if rows:
            seed = []
            if c.SEED_FIT_CACHE_FROM_SOURCE:
                old = os.path.join(block, 'fitcache.csv')
                pool = []
                if os.path.exists(old):
                    ch, crow = _read_csv(old)
                    pool.append((ch, crow))
                if os.path.exists(src.files['fitcache']):
                    pool.append(src.fit_cache_rows(cids))
                want = set(cids)
                chdr = None
                for h_, rr in pool:
                    if not h_:
                        continue
                    chdr = chdr or h_
                    if h_ != chdr:
                        continue
                    seed += [r for r in rr if r and cache_cand_id(r[0]) in want]
                if seed:
                    _write_csv(fit_out + '.cache.csv', chdr, seed)
            ctx.run_stage(u, 'fitpos(epoch)', [c.STAGE_SCRIPTS['fitpos'], tmp, '--out', fit_out, '--nproc', _fit_nproc()],
                          stage_env(ctx, 'fitpos'), os.path.join(c.STATE_DIR, 'epochs', f'{epoch}.fitpos.log'))
            shutil.copy(fit_out + '.cache.csv', os.path.join(tmp, 'fitcache.csv'))
            if os.path.exists(fit_out + '.summary.json'):
                shutil.copy(fit_out + '.summary.json', os.path.join(tmp, 'fit_summary.json'))
            for p in (fit_out, fit_out + '.cache.csv', fit_out + '.summary.json'):
                if os.path.exists(p):
                    os.unlink(p)
            u.note(f'fit cache seeded with {len(seed)} rows')
        else:
            _write_csv(os.path.join(tmp, 'fitcache.csv'), ['key', '_fracpx'], [])
        meta = {'epoch': epoch, 'lst': lst, 'date': date, 'source': src.spec, 'utc': _utc(), 'n_rows': len(rows),
                'n_frames': len(frames), 'scopes': scopes, 'excluded_scopes_at_extraction': xs,
                'files': {n: _sha1_file(os.path.join(tmp, n)) for n in BLOCK_FILES if n != 'epoch.json'
                          and os.path.exists(os.path.join(tmp, n))}}
        _write_json(os.path.join(tmp, 'epoch.json'), meta)
        old = f'{block}.old.{os.getpid()}'
        if os.path.exists(block):
            os.rename(block, old)
        os.rename(tmp, block)
        shutil.rmtree(old, ignore_errors=True)
        _add_output(u, block, 'epoch_block')


# =====================================================================================================================
# phase: associate
# =====================================================================================================================

def assemble(ctx, epochs, excl, u=None):
    """Write the run's merge output from the epoch blocks, in canonical order (the replay's only input besides the static
    reference data): master_catalog.csv (= selected_all.csv, as p3s3_merge.py:150-153), master_catalog_refused.csv,
    v_deep_ledger.csv, v_deep_images.csv (blocks in canonical epoch order), fit_positions_cache.csv, per-frame links under
    vresid/ iresid/ and per-item links of parent_catalogs/ v110_products/. Rows, ledger, images and frames of a (lst, scope,
    date) excluded NOW are left out - as the merge staleness gate (p3s3_stale_exclusions.py) leaves out an attempt that
    searched an excluded epoch, at epoch granularity. Returns a summary dict (sha1 of the assembled merge output)."""
    U = ctx.out
    hdr0, rows, refused, ledger, images, fitc = None, [], [], [], [], []
    hdrs = {}
    frames = []
    n_x = 0
    for e in epochs:
        b = os.path.join(ctx.state_dir, 'epochs', e)
        lst, date = parse_epoch(e)
        h, rr = _read_csv(os.path.join(b, 'master_catalog.csv'))
        if hdr0 is None and h:
            hdr0 = h
        if h and h != hdr0:
            raise Refusal(f'epoch block {e}: merge-output columns differ from the other blocks ({b}); re-extract it')
        ix = {k: i for i, k in enumerate(h)}
        bands = {r[ix['band']] for r in rr} if rr else set()
        fr = _read_csv(os.path.join(b, 'frames.csv'))[1]
        bands |= {f[1].split('_', 1)[1] for f in fr}
        xs = set(_excluded_scopes(excl, lst, date, bands))
        for r in rr:
            if r[ix['band']] in xs:
                n_x += 1
            else:
                rows.append(r)
        frames += [f for f in fr if f[1].split('_', 1)[1] not in xs]
        for name, acc, scope_col in (('refused.csv', refused, 'band'), ('ledger.csv', ledger, 'scope'),
                                     ('images.csv', images, 'scope')):
            h2, r2 = _read_csv(os.path.join(b, name))
            if not h2 or h2 == ['lst', 'date']:
                continue
            if name in hdrs and hdrs[name] != h2:
                raise Refusal(f'epoch block {e}: {name} columns differ from the other blocks')
            hdrs[name] = h2
            j = h2.index(scope_col) if scope_col in h2 else None
            acc += [r for r in r2 if j is None or r[j] not in xs]
        h3, r3 = _read_csv(os.path.join(b, 'fitcache.csv'))
        if h3 and len(h3) > 2:
            if 'fitcache' in hdrs and hdrs['fitcache'] != h3:
                raise Refusal(f'epoch block {e}: fitcache.csv columns differ from the other blocks')
            hdrs['fitcache'] = h3
            fitc += r3
    if hdr0 is None:
        raise NoInput('no epoch block has a merge output')
    ix = {k: i for i, k in enumerate(hdr0)}
    cid = Counter(r[ix['cand_id']] for r in rows)
    dup = [k for k, v in cid.items() if v > 1]
    if dup:
        raise Refusal(f'{len(dup)} cand_id(s) appear in two epoch blocks, e.g. {dup[:3]} - the canonical order needs unique ids')
    rows.sort(key=lambda r: row_order_key(r, ix))
    # ledger, images, refused rows and fit-cache rows carry no unique id: sorted as whole rows (a total order on distinct
    # rows; identical rows are interchangeable), so the workspace is a function of the SET of epochs, never of the order
    # in which blocks were extracted or of the row order inside a block
    for acc in (refused, ledger, images, fitc):
        acc.sort()
    summary = {'epochs': len(epochs), 'rows': len(rows), 'rows_excluded_now': n_x, 'frames': len(frames),
               'ledger_rows': len(ledger), 'image_rows': len(images), 'fit_cache_rows': len(fitc)}
    if ctx.dry:
        return summary
    _write_csv(f'{U}/master_catalog.csv', hdr0, rows)
    _write_csv(f'{U}/selected_all.csv', hdr0, rows)
    if 'refused.csv' in hdrs:
        _write_csv(f'{U}/master_catalog_refused.csv', hdrs['refused.csv'], refused)
    if 'ledger.csv' in hdrs:
        _write_csv(f'{U}/v_deep_ledger.csv', hdrs['ledger.csv'], ledger)
    if 'images.csv' in hdrs:
        _write_csv(f'{U}/v_deep_images.csv', hdrs['images.csv'], images)
    if 'fitcache' in hdrs:
        _write_csv(f'{U}/fit_positions_cache.csv', hdrs['fitcache'], fitc)
    # frames: exactly the admitted epochs' files (per-file links; never a link to a whole attempt dir)
    want = {}
    for st, cell, fn, tgt in frames:
        want[os.path.join(U, 'vresid' if st == 'V' else 'iresid', cell, fn)] = tgt
    for sub in ('vresid', 'iresid'):
        d = os.path.join(U, sub)
        if os.path.islink(d):
            raise Refusal(f'{d} is a link (a production attempt layout): the incremental workspace needs its own frame links')
        if os.path.isdir(d):
            for cell in os.listdir(d):
                cd = os.path.join(d, cell)
                if os.path.islink(cd):
                    raise Refusal(f'{cd} is a link to a whole attempt dir (production layout): not an incremental workspace')
                for fn in os.listdir(cd):
                    p = os.path.join(cd, fn)
                    if os.path.islink(p) and p not in want:
                        os.unlink(p)
    for p, tgt in want.items():
        _link(tgt, p)
    # parent catalogues and v110 products of the cells of the state's LSTs: per-item links to the source run's targets
    lsts = {parse_epoch(e)[0] for e in epochs}
    for sub in ('parent_catalogs', 'v110_products'):
        sd = os.path.join(ctx.source.dir, sub)
        if not os.path.isdir(sd):
            continue
        for item in sorted(os.listdir(sd)):
            m = re.match(r'(?:master_R075_)?(\d\dh)_', item)
            if m and m.group(1) in lsts:
                _link(os.path.realpath(os.path.join(sd, item)), os.path.join(U, sub, item))
    dropped = prune_frame_caches(U, ctx.state_dir, sorted(want))
    if dropped:
        summary['frame_cache_rows_dropped'] = dropped
    summary['merge_output_sha1'] = _sha1_file(f'{U}/master_catalog.csv')
    if u is not None:
        u.note(f'assembled: {json.dumps(summary, sort_keys=True)}')
    return summary


def _frame_key(path):
    """'S|lst|band|date' of a searched-frame link <U>/{vresid,iresid}/<lst>_<band>/<date>_<S>_redo_resid_<band>.fits, the
    key p3s2_conform.py gives it (the cell dir's LST, the band from the file name); None for another name."""
    m = _FRAME_RE.match(os.path.basename(path))
    if not m:
        return None
    return '|'.join((m.group(2), os.path.basename(os.path.dirname(path)).split('_')[0], m.group(3), m.group(1)))


def _frame_identity(path):
    try:
        st = os.stat(path)                      # follows the link: the frame file itself
        return [os.path.realpath(path), st.st_size, st.st_mtime_ns]
    except OSError:
        return None


def prune_frame_caches(U, state_dir, frame_paths):
    """Keep in each conformance frame cache of the workspace only the rows whose frame is the one the row was read from
    (same target, size and mtime as recorded at the previous assembly); every other row is dropped so conformance reads
    that header again, as a full rebuild into an empty --out does. Writes the record for the next assembly. Returns
    {cache name: rows dropped}."""
    cur = {}
    for p in frame_paths:
        k = _frame_key(p)
        if k:
            cur[k] = _frame_identity(p)
    rec_p = os.path.join(state_dir, FRAME_CACHE_RECORD)
    try:
        prev = json.load(open(rec_p)) if os.path.exists(rec_p) else {}
    except (OSError, ValueError):
        prev = {}
    dropped = {}
    for name in FRAME_CACHES:
        p = os.path.join(U, name)
        if not os.path.exists(p):
            continue
        h, rows = _read_csv(p)
        try:
            ix = [h.index(c) for c in ('stokes', 'lst', 'band', 'date')]
        except ValueError:
            os.replace(p, p + '.unreadable')     # not a cache conformance wrote: set aside, conformance rebuilds it
            dropped[name] = 'unreadable'
            continue
        keep = []
        for r in rows:
            k = '|'.join(r[i] for i in ix)
            if cur.get(k) is not None and prev.get(k) == cur[k]:
                keep.append(r)
        if len(keep) != len(rows):
            dropped[name] = len(rows) - len(keep)
            if keep:
                _write_csv(p, h, keep)
            else:
                os.remove(p)                    # empty: conformance writes it afresh
    os.makedirs(state_dir, exist_ok=True)
    _write_json(rec_p, cur)
    return dropped


def _load_state(ctx):
    p = os.path.join(ctx.state_dir, 'state.json')
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        return json.load(f)


def _build_stage(ctx, u, keys):
    """The per-(Stokes, tier) builds (p3s3_pass.sh:77-83): a key with no selected rows has its catalogue dir removed; each
    build writes into a FRESH directory that then replaces the old one (production's mkdir -p keeps stale files of an
    earlier pass - e.g. Run_P3S5/source_catalogs_I_wb/Blue_sources_removed_gate78.csv of 2026-10-09 beside the 10-12
    build - which would make an incremental workspace differ from a full rebuild)."""
    c = C()
    U = ctx.out
    built = []
    jobs = []
    for K in keys:
        st, tier = K.split('_')
        sel = f'{U}/selected_{st}_{tier}.csv'
        if ctx.dry:
            ctx.run_stage(u, f'build {K}', [c.STAGE_SCRIPTS['build'], f'selected_{st}_{tier}.csv', f'source_catalogs_{K}'
                                            f' (fresh dir, swapped in)', 'v_deep_images.csv'], stage_env(ctx, 'build'),
                          f'catalogs_inc_{K}.log')
            continue
        n = (sum(1 for _ in open(sel)) - 1) if os.path.exists(sel) else 0
        if n < 1:
            ctx.say(f'  [build {K}] no selected rows yet - source_catalogs_{K} removed')
            if not ctx.dry:
                shutil.rmtree(f'{U}/source_catalogs_{K}', ignore_errors=True)
            continue
        tmpname = f'source_catalogs_{K}.inc_new'
        if not ctx.dry:
            shutil.rmtree(f'{U}/{tmpname}', ignore_errors=True)
            os.makedirs(f'{U}/{tmpname}')
        jobs.append((K, [c.STAGE_SCRIPTS['build'], f'selected_{st}_{tier}.csv', tmpname, 'v_deep_images.csv'], tmpname))

    def one(job):
        K, argv, tmpname = job
        ctx.run_stage(u, f'build {K}', argv, stage_env(ctx, 'build'), f'catalogs_inc_{K}.log', cwd=U)
        return job
    if c.BUILD_WORKERS and int(c.BUILD_WORKERS) > 1 and len(jobs) > 1 and not ctx.dry:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(int(c.BUILD_WORKERS)) as ex:
            done = list(ex.map(one, jobs))
    else:
        done = [one(j) for j in jobs]
    for K, argv, tmpname in done:
        if ctx.dry:
            continue
        final = f'{U}/source_catalogs_{K}'
        old = f'{final}.inc_old'
        shutil.rmtree(old, ignore_errors=True)
        if os.path.exists(final):
            os.rename(final, old)
        os.rename(f'{U}/{tmpname}', final)
        shutil.rmtree(old, ignore_errors=True)
        built.append(K)
    return built


def _replay_associate(ctx, u):
    c = C()
    U = ctx.out
    stages = tuple(c.REPLAY_STAGES_ASSOCIATE)
    built = []
    for stage in stages:
        if stage == 'conform':
            ctx.run_stage(u, 'conform', [c.STAGE_SCRIPTS['conform'], U], stage_env(ctx, 'conform'), 'incremental.log',
                          append=True)
            if not ctx.dry:      # the conformant record before fitpos rewrites master_catalog.csv in place (its content test)
                sd = os.path.join(ctx.state_dir, 'stage_conform')
                os.makedirs(sd, exist_ok=True)
                shutil.copy(f'{U}/master_catalog.csv', os.path.join(sd, 'master_catalog.csv'))
        elif stage == 'fitpos':
            ctx.run_stage(u, 'fitpos', [c.STAGE_SCRIPTS['fitpos'], U, '--nproc', _fit_nproc()],
                          stage_env(ctx, 'fitpos'), 'incremental.log', append=True)
            if not ctx.dry:
                try:
                    summ = json.load(open(f'{U}/fit_positions_summary.json'))
                    u.note(f"fitpos replay: {summ.get('cached')} from the epoch fits, {summ.get('fitted_now')} fitted now")
                    if summ.get('fitted_now'):
                        u.note('fitpos replay fitted rows the epoch stage had not fitted (a frame changed since): products '
                               'still identical to a full rebuild (same function), but the epoch cache was incomplete')
                except (OSError, ValueError):
                    pass
        elif stage == 'select':
            ctx.run_stage(u, 'select', [c.STAGE_SCRIPTS['select'], 'master_catalog.csv', 'selected_all.csv',
                                        'master_catalog.csv'], stage_env(ctx, 'select'), 'select_inc.log', cwd=U)
        elif stage == 'ephem':
            ctx.run_stage(u, 'ephem', [c.STAGE_SCRIPTS['ephem']], stage_env(ctx, 'ephem', {'P3S_RUN_DIR': U}),
                          'select_inc.log', cwd=U, append=True)
        elif stage == 'split':
            if ctx.dry:
                ctx.say('  [split] ported p3s3_pass.sh:66-76 -> selected_{V,I}_{wb,sb}.csv')
            else:
                n = split_selection(U)
                u.note(f'split: {n}')
        elif stage == 'build':
            built = _build_stage(ctx, u, c.CATALOG_KEYS)
        elif stage == 'fork':
            for st in _deep('catalog_fork', 'STOKES'):
                env = stage_env(ctx, 'fork', {'FORK_RUN_DIR': U, 'FORK_OUT_DIR': U,
                                              'FORK_NPROC': str(_deep('catalog_fork', 'NPROC'))})
                ctx.run_stage(u, f'fork {st}', [c.STAGE_SCRIPTS['fork'], 'run', st], env, f'fork_{st}.log', cwd=U)
                ctx.run_stage(u, f'fork check {st}', [c.STAGE_SCRIPTS['fork'], 'check', st], env, f'fork_{st}.log', cwd=U,
                              append=True)
        else:
            raise Refusal(f'catalog_incremental.REPLAY_STAGES_ASSOCIATE names an unknown stage {stage!r}')
    return built


def _engine(ctx, u, ns, sstack, keys_sel):
    """Update the association state of every key from the replayed selections; cross-check against the builds."""
    c = C()
    state_dir = ctx.state_dir
    prev_state = _load_state(ctx)
    valid_prev = (not ctx.rebuild and prev_state.get('engine', {}).get('version') == ENGINE_VERSION
                  and prev_state.get('engine', {}).get('builder_sha1') == ns['__sha1__']
                  and float(prev_state.get('engine', {}).get('K_ASSOC', -1)) == float(ns['K_ASSOC']))
    if prev_state and not valid_prev:
        u.note('association state rebuilt in full: ' + ('--inc-rebuild/--fresh' if ctx.rebuild else
                                                        'engine version, builder or K_ASSOC changed since the last state'))
    out = {}
    for K, sel_path, cat_dir, is_fork in keys_sel:
        prev = read_engine_dets(state_dir, K) if valid_prev else None
        prev_src = read_engine_sources(state_dir, K) if valid_prev else OrderedDict()
        if not os.path.exists(sel_path) or not os.path.isdir(cat_dir):
            # no rows for this key now: every previous source is removed
            res = {'dets': OrderedDict(), 'sources': OrderedDict(), 'stats': {}}
        else:
            h, rows = _read_csv(sel_path)
            beam = {}
            ep = os.path.join(cat_dir, 'E_params.json')
            if os.path.exists(ep):
                beam = json.load(open(ep)).get('beam_as', {})
            st = {r[h.index('stokes')] for r in rows} if rows else set()
            thr = sstack[0] if (st == {'I'} and not is_fork) else None
            res = associate_key(ns, h, rows, beam, thr, prev_dets=prev, K=float(ns['K_ASSOC']))
            if c.CROSSCHECK:
                ok, det = crosscheck(res, cat_dir)
                u.check(f'incremental.crosscheck.{K}', ok, det, det, cls='incremental_partition', path=cat_dir)
                if not ok and not ctx.run.waived(f'incremental.crosscheck.{K}'):
                    raise Refusal(f'{K}: the builder and the incremental association disagree - {det} (state not advanced)')
        delta = []
        for band in sorted(set(res['sources']) | set(prev_src)):
            delta += classify_delta(prev_src.get(band, []), res['sources'].get(band, []))
        out[K] = (res, delta)
    # write the state only when every key passed
    for K, (res, delta) in out.items():
        write_engine_state(state_dir, K, res)
        _write_csv(os.path.join(state_dir, f'delta_{K}.csv'), list(DELTA_FIELDS),
                   [[str(r[f]) for f in DELTA_FIELDS] for r in delta])
    return out


def _associate_unit(ctx):
    from exopipe.phase3.core import exclusions
    c = C()
    with ctx.run.unit('incremental.associate', label=ctx.label) as u:
        epochs = epochs_in_state(ctx.state_dir)
        if not epochs:
            if ctx.dry and getattr(ctx, 'planned_epochs', None):
                epochs = list(ctx.planned_epochs)
                ctx.say(f'associate: (plan) the state would hold {len(epochs)} epoch(s) after the epoch phase')
            else:
                raise NoInput(f'no epoch blocks under {ctx.state_dir}/epochs - run the epoch phase first')
        if not ctx.parents or not os.path.isdir(ctx.parents):
            msg = (f'P3_PARENT_DIR is required and is {ctx.parents or "unset"}: pass --inc-parents DIR (the parent-catalogue '
                   f'snapshot of the current masters; B-59 - gate_inputs.py:19 would otherwise read Run_P3S4/parent_catalogs)')
            if ctx.dry:
                ctx.say(f'  GATE would refuse: {msg}')
            else:
                raise Refusal(msg)
        ns = builder_namespace()
        binding_check(ns)
        sstack = sstack_threshold()
        u.meta.update({'builder_sha1': ns['__sha1__'], 'builder_functions': ns['__segments__'], 'sstack': sstack,
                       'K_ASSOC': ns['K_ASSOC'], 'epochs': len(epochs)})
        u.add_input(_ops(c.BUILDER))
        u.add_input(_ops(c.ASTROM_MODULE))
        for s in c.STAGE_SCRIPTS.values():
            if os.path.exists(_ops(s)):
                u.add_input(_ops(s))
        excl = exclusions.load()
        u.meta['exclusions'] = excl.fingerprint()
        xl = []
        for e in epochs:
            b = os.path.join(ctx.state_dir, 'epochs', e)
            if os.path.exists(os.path.join(b, 'epoch.json')):
                for n in ('epoch.json', 'master_catalog.csv', 'frames.csv', 'fitcache.csv'):
                    if os.path.exists(os.path.join(b, n)):
                        u.add_input(os.path.join(b, n))
                meta = json.load(open(os.path.join(b, 'epoch.json')))
                lst, date = parse_epoch(e)
                xl += [f'excluded:{e}:{s}' for s in _excluded_scopes(excl, lst, date, meta.get('scopes', []))]
        u.set_admitted(list(epochs) + xl)
        if _resumed(ctx, u, u.admitted):
            return
        ctx.say(f'associate: {len(epochs)} epoch block(s) -> canonical merge output in {ctx.out} '
                f'(order {",".join(c.CANONICAL_ROW_ORDER)}); builder {c.BUILDER} sha1 {ns["__sha1__"][:12]}; '
                f'sigma-stack {sstack[0]} ({sstack[1]})')
        summ = assemble(ctx, [e for e in epochs if os.path.exists(os.path.join(ctx.state_dir, 'epochs', e, 'epoch.json'))],
                        excl, u) if epochs_in_state(ctx.state_dir) else {'rows': '(after the epoch phase)'}
        ctx.say(f'  assembled merge output: {json.dumps(summ, sort_keys=True)}')
        built = _replay_associate(ctx, u)
        if ctx.dry:
            ctx.say(f'  engine: incremental association of {", ".join(c.CATALOG_KEYS)} (+ forks {", ".join(c.FORK_KEYS)}) '
                    f'from the replayed selections; cross-check against the builds; writes {ctx.state_dir}/'
                    f'{{engine,sources,delta}}_<K>.csv and state.json [classes incremental_state, incremental_partition, '
                    f'incremental_delta]')
            u.reason = 'dry run: plan printed, nothing written'
            return
        keys_sel = [(K, f'{ctx.out}/selected_{K}.csv', f'{ctx.out}/source_catalogs_{K}', False) for K in c.CATALOG_KEYS]
        if 'fork' in c.REPLAY_STAGES_ASSOCIATE:
            keys_sel += [(K, os.path.join(ctx.out, rel), f'{ctx.out}/source_catalogs_{K}', True)
                         for K, rel in c.FORK_KEYS.items()]
        res = _engine(ctx, u, ns, sstack, keys_sel)
        prev = _load_state(ctx)
        state = {'version': STATE_VERSION, 'label': ctx.label, 'utc': _utc(), 'source': ctx.source.spec,
                 'canonical_row_order': list(c.CANONICAL_ROW_ORDER),
                 'engine': {'version': ENGINE_VERSION, 'builder': _ops(c.BUILDER), 'builder_sha1': ns['__sha1__'],
                            'builder_functions': ns['__segments__'], 'K_ASSOC': ns['K_ASSOC'], 'sstack': list(sstack)},
                 'exclusions': excl.fingerprint(), 'assembled': summ,
                 'epochs': {e: json.load(open(os.path.join(ctx.state_dir, 'epochs', e, 'epoch.json')))['files']
                            for e in epochs},
                 'previous_epochs': sorted(prev.get('epochs', {}), key=epoch_order_key) if prev else [],
                 'keys': {K: {'n_dets': sum(len(v) for v in r[0]['dets'].values()),
                              'n_sources': sum(len(v) for v in r[0]['sources'].values()),
                              'partition_sha1': partition_digest(r[0]['dets']),
                              'stats': r[0]['stats'],
                              'delta': dict(Counter(x['change'] for x in r[1])),
                              'renamed': sum(int(x['renamed']) for x in r[1])} for K, r in res.items()}}
        _write_json(os.path.join(ctx.state_dir, 'state.json'), state)
        for K, (r, delta) in res.items():
            ctx.say(f'  {K}: {state["keys"][K]["n_dets"]} detections -> {state["keys"][K]["n_sources"]} sources; '
                    f'delta {state["keys"][K]["delta"]}; renamed {state["keys"][K]["renamed"]}')
        _add_output(u, ctx.state_dir, 'incremental_state')
        for K, (r, delta) in res.items():
            _add_output(u, os.path.join(ctx.state_dir, f'delta_{K}.csv'), 'incremental_delta')
            cd = f'{ctx.out}/source_catalogs_{K}'
            if os.path.isdir(cd):
                _add_output(u, cd, 'incremental_partition')
                _add_output(u, cd, 'fork_catalog' if K in c.FORK_KEYS else 'source_catalog')
        U, conf = ctx.out, os.path.join(ctx.state_dir, 'stage_conform', 'master_catalog.csv')
        if os.path.exists(conf):
            _add_output(u, conf, 'conformant_catalog', **{'master_catalog_asrun.csv': f'{U}/master_catalog_asrun.csv',
                                                          'master_catalog_removed.csv': f'{U}/master_catalog_removed.csv'})
            _add_output(u, f'{U}/master_catalog.csv', 'fitted_catalog', input=conf)
        for p in [f'{U}/selected_all.csv'] + [f'{U}/selected_{K}.csv' for K in c.CATALOG_KEYS]:
            if os.path.exists(p):
                _add_output(u, p, 'selection')


# =====================================================================================================================
# phase: identity (after `exopipe3 forced`)
# =====================================================================================================================

def _identity_unit(ctx):
    from exopipe.phase3.core import record
    c = C()
    with ctx.run.unit('incremental.identity', label=ctx.label) as u:
        arec = record.load_record(ctx.out, ctx.run.step, 'incremental.associate')
        if not arec or arec.get('status') != 'done':
            msg = (f'the associate phase of {ctx.out} is not done (record incremental.associate: '
                   f'{(arec or {}).get("status", "missing")}) - run it first')
            if not ctx.dry:
                raise Refusal(msg)
            ctx.say(f'identity: GATE would refuse now: {msg}')
            arec = arec or {}
        if c.REQUIRE_FORCED_BEFORE_IDENTITY:
            fdir = c.FORCED_RECORDS.format(out=ctx.out)
            newest = ''
            if os.path.isdir(fdir):
                for fn in os.listdir(fdir):
                    if fn.endswith('.json'):
                        try:
                            r = json.load(open(os.path.join(fdir, fn)))
                        except (OSError, ValueError):
                            continue
                        if r.get('status') == 'done' and str(r.get('finished', '')) > newest:
                            newest = str(r.get('finished', ''))
            ok = bool(newest) and newest > str(arec.get('finished', ''))
            name = 'incremental.forced_before_identity'
            if not ok:
                msg = (f'D-G22: no done `forced` record in {fdir} newer than the associate phase ({arec.get("finished")}); '
                       f'the union carries the forced_* columns forced writes into the tier catalogues (p3s3_forced_phot.py)')
                if ctx.dry:
                    ctx.say(f'  GATE would refuse: {msg}')
                elif not ctx.run.waived(name):
                    raise Refusal(msg)
                else:
                    u.check(name, False, newest, msg)
        stages = tuple(c.REPLAY_STAGES_IDENTITY)
        U = ctx.out
        listp = None
        for stage in stages:
            if stage == 'union':
                if ctx.dry:
                    ctx.say('  [union] ported p3s3_pass.sh:103-129 (four primary tiers, one row per (Stokes, source_id))')
                else:
                    u.note(f'union: {len(union_tiers(U))} files')
            elif stage == 'identity':
                ctx.run_stage(u, 'identity', [c.STAGE_SCRIPTS['identity'], U], stage_env(ctx, 'identity'),
                              'incremental.log', append=True)
            elif stage == 'localize':
                ctx.run_stage(u, 'localize', [c.STAGE_SCRIPTS['localize'], U], stage_env(ctx, 'localize'),
                              'incremental.log', append=True)
            elif stage == 'list':
                listp = os.path.join(U, c.TRANSIENT_LIST_NAME.format(label=ctx.label, utc=_utc('%Y%m%dT%H%M')))
                ctx.run_stage(u, 'list', [c.STAGE_SCRIPTS['list'], U, listp], stage_env(ctx, 'list'), 'incremental.log',
                              append=True)
                if not ctx.dry:
                    latest = os.path.join(U, c.TRANSIENT_LIST_NAME.format(label=ctx.label, utc='latest'))
                    if os.path.islink(latest) or os.path.exists(latest):
                        os.unlink(latest)
                    os.symlink(listp, latest)
            else:
                raise Refusal(f'catalog_incremental.REPLAY_STAGES_IDENTITY names an unknown stage {stage!r}')
        if ctx.dry:
            u.reason = 'dry run: plan printed, nothing written'
            return
        for p in sorted(os.listdir(f'{U}/source_catalogs')) if os.path.isdir(f'{U}/source_catalogs') else []:
            if p.endswith('_sources_survivors.csv'):
                _add_output(u, f'{U}/source_catalogs/{p}', 'union_catalog')
        for p, cls in ((f'{U}/source_catalogs/localizations.csv', 'localization_table'), (listp, 'transient_list')):
            if p and os.path.exists(p):
                _add_output(u, p, cls)


# =====================================================================================================================
# entry points
# =====================================================================================================================

def add_arguments(p):
    """The --inc-* flags (steps/catalog.py adds them beside its own --incremental)."""
    import argparse
    g = p if isinstance(p, argparse._ArgumentGroup) else p.add_argument_group(
        'incremental (--incremental: lib/catalog/incremental.py; config catalog_incremental)')
    g.add_argument('--inc-phase', metavar='PHASES',
                   help='comma list of epoch, associate, identity (default catalog_incremental.PHASES_DEFAULT = '
                        'epoch,associate; identity runs after `exopipe3 forced`, gate D-G22)')
    g.add_argument('--inc-source', metavar='SPEC',
                   help="where an epoch's detections come from: run:<deep run dir> (its merge output; default "
                        "run:<paths.TRANSIENTS_ROOT>/Run_<label>) or search (per-hour search output; refuses "
                        "until that layout exists)")
    g.add_argument('--inc-parents', metavar='DIR',
                   help='parent-catalogue snapshot passed as P3_PARENT_DIR to conformance and the fork (required for '
                        'the associate phase; never defaulted, B-59)')
    g.add_argument('--inc-rebuild', action='store_true',
                   help='rebuild the association state from every epoch block (the canonical FULL rebuild of the same '
                        'epochs) instead of updating it')


def run_incremental(args, run, epochs=None):
    """Add the epochs (items <LST>_<date>, a list or a spec) to the cumulative deep catalogue under run.out_root.
    Returns None (the CLI derives the exit code from the units) or, in --dry-run, 0 after printing the plan."""
    c = C()
    tier = getattr(args, 'tier', None) or 'deep'
    if tier not in c.TIERS:
        raise Refusal(f'catalog --incremental: tier {tier} has no incremental path (catalog_incremental.TIERS = '
                      f'{c.TIERS}); the sub-LST and snapshot catalogues are rebuilt by their own chains')
    ctx = _Ctx(args, run)
    if epochs is None:
        epochs = getattr(args, 'epochs', None)
    items = expand_epochs(epochs)
    if 'epoch' in ctx.phases and not items:
        raise NoInput('--incremental with the epoch phase needs --epochs <LST>_<date>[,...] (or --inc-phase associate|identity)')
    from exopipe.phase3.core import env as _env
    _env.legacy()
    leaks = env_leaks()
    if leaks:
        msg = (f'the environment carries {", ".join(leaks)}: in-place stages would read them (core.legacy.run clears only '
               f'P3_*) and associate differently from the production pass - unset them')
        if not ctx.dry:
            raise Refusal(msg)
        ctx.say(f'GATE would refuse: {msg}')
    ctx.planned_epochs = sorted(set(epochs_in_state(ctx.state_dir)) | set(items), key=epoch_order_key)
    if 'epoch' in ctx.phases:
        ctx.source.want = list(items)          # each source table is read once for every requested epoch
    ctx.say(f'catalog --incremental (deep, label {ctx.label}): out {ctx.out}; phases {",".join(ctx.phases)}; epochs '
            f'{",".join(items) or "(none new)"}; state {len(epochs_in_state(ctx.state_dir))} epoch(s) in {ctx.state_dir}'
            + ('; association state rebuilt in full' if ctx.rebuild else ''))
    if 'epoch' in ctx.phases:
        for e in items:
            _epoch_unit(ctx, e)
    if 'associate' in ctx.phases:
        failed = [x for x in ctx.run.units if x.name.startswith('incremental.epoch.') and x.status in ('refused', 'error')]
        if failed and not ctx.dry:
            with ctx.run.unit('incremental.associate', label=ctx.label):
                raise Refusal(f'not associating: {len(failed)} epoch unit(s) refused or failed '
                              f'({", ".join(x.name for x in failed[:5])})')
        else:
            _associate_unit(ctx)
    if 'identity' in ctx.phases:
        _identity_unit(ctx)
    if ctx.dry:
        return 0
    return None
