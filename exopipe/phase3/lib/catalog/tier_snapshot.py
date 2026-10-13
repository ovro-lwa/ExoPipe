"""catalog --tier snapshot - the snapshot batch chain of ONE cumulative run (Run_SNAP...) over its cell list.

Ports the catalogue part of ops/snapshot_rechain_batch.sh (steps 0-5 and the run content test): preconditions, the fit-store
gate and repair, the prefit loop, the per-cell tie pass, the multi-cell chain (association, routing, 3-D bright-source gate,
reference astrometry, anchor association, check sources, run writer: clv15/phase3_snapshot_chain.py IN PLACE) and
snapshot_rechain_content_test.py. In-place scripts run through core/legacy.run under the site interpreter 'py36' (the
development Python 3.6 production uses); the shell logic and heredocs are Python here. Pages, the filter catalogue, carry,
HOLD and serve (driver steps 6-8) are `pages` / `serve`; the per-tier localization (snapshot_localize_tiers.sh) is `serve`'s in
the window; the wideband fit-store quarantine is `archive quarantine-rows` (refactor chat decision 6).

Unit of work: the run (`--run Run_SNAP...`, item kind = the run; its cells are night cells 18h_2025-04-11 from
`--cells @CELLS_FILE`, one per line, LST_DATE or 'LST DATE', '#' comments - the production cells file). Records:
'<NAME>.<stage>' per stage, '<NAME>.tie.<cell>' per tie item, '<NAME>.fit.<tag>.<part>' per fit-round item. The per-cell tie
pass and the fit rounds are core/runner jobs of this same step (internal stage 'fit-shard', and 'tie' with --snap-item) on
the nodes of --snap-chain-nodes (default the site pool 'snapshot'); the stage that submits them waits for the job record
(never log text or pids).

Stages (--stages / --from / --until; default all, in this order):
  preflight  P-G1 run name; P-G2 approval (--snap-approval "<Gregg's words, UTC, ledger ID>", or the production approval marker
             with CELLS_FILE_SHA256 = the list now) + the list's sha256 in the record; P-G3 the tested-code sha256 file verifies;
             P-G4 batch-1 store on this host; P-G5 single writer (runner leases, K7: production refuses until they exist;
             non-production writes only copies); P-G6 eligibility (snapshot_batch_audit.py); P-G7 every served cell kept
             (waiver snapshot.served-cells-kept); P-G8 prefit writer released; P-G9 band classes (Q14); P-G10 frame-time
             pre-fix inputs (waiver snapshot.frame-time-prefix); P-G11 tie model (waiver snapshot.tie-model); P-G12 /fast free
             space; binding check of every [lit] key
  store      stores quiet (P-G13); unmarked double-registered wideband rows REFUSE (archive quarantine-rows moves them);
             repair lists + one repair fit round; combined store + header equality + identity-screen count + store content
             test snapshot_store_ready.py (P-G14)
  prefit     up to PREFIT_MAX_ITER rounds: per-cell --prefit-list-only (12 at a time), union, fit round, append (P-G15)
  tie        per-cell --stop-after-tie items over the nodes, 16 imfit shards for >= 30,000 detections at the fit step else 8
             (P-G16)
  chain      memory model + node_mem_gate.sh (P-G17); the multi-cell chain over every cell (P-G18)
  test       snapshot_rechain_content_test.py R1-R5, R7 (P-G19; R6 after the pages is `pages`')
P-G20 (runaway guard, LIMIT_H hours, from the marker's LIMIT_HOURS or config) is checked before every stage.

Inputs:  the cells file; paths.SNAPSHOT_FLEET_OUT/<cell>/ (search products, runinfo); catalog_snapshot.MASTER_CATALOGS,
         CURVE_CONSTANTS, LIGHTCURVES_ROOT, CHECKSOURCES, DIRTY_CELLS; the per-cell fit stores BATCH_DIR/fits/fits_<cell>.csv
         (+ _template.csv) and the batch-1 store; the tie cache BATCH_DIR/tiecache; the served run on SERVED_PORT_CHECK
         (paths.SNAPSHOT_REVIEW_ROOT, or --snap-served-run); data_decisions.csv is applied inside the chain (in place).
Outputs (product classes): RUN = --out (production catalog_snapshot.WORK_ROOT/snapshot_rechain_<NAME>/<NAME>):
         chain_summary.json, candidates/, source_catalogs*/, queue*.txt, rejected.csv, vetoed/, ... (snapshot_run); the
         combined store L/fits_store.csv (csv_table + the in-place store content test; fit_store is search's class, not
         registered yet); the tie cache rows of the run's cells (tie_cache); prefit and repair lists (csv_table); the
         preflight report (catalog_preflight). Non-production runs keep every side file under RUN/_side/ (fit-store and
         tie-cache COPIES, L, W); production writes the production paths of the batch driver.
Failure modes (exit 2, gate named in the record): P-G1..P-G20 of spec section 5.3; any in-place script exiting non-zero;
         a non-production store stage that needed repair rows (snapshot_store_ready.py reads only the production store
         directory: refused until the test is absorbed); a [lit] key differing from its module literal. Exit 1 = error.
Per-hour vs intermittent: the chain associates over EVERY cell of the cumulative run at once (2 d 18 h on batch 2), so it is
         an intermittent (batch) step today; the per-hour chain 'snapshot-cell' stops at `curves` for a new cell and the
         cell joins the next cumulative run. A per-cell incremental chain is not implemented (spec speed-up 9).
"""
import csv
import glob
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time

TIER = 'snapshot'
STAGES = ('preflight', 'store', 'prefit', 'tie', 'chain', 'test')
INTERNAL = ('fit-shard',)
WAIVERS = {'served': 'snapshot.served-cells-kept', 'frametime': 'snapshot.frame-time-prefix', 'tiemodel': 'snapshot.tie-model'}
MARKER_SWITCH = {'served': 'ALLOW-DROP-SERVED-CELLS', 'frametime': 'ALLOW-PREFIX-WIDEBAND-1MIN',
                 'tiemodel': 'ALLOW-PRE-BUNDLE-TIES'}
WIDE = ('Red', 'Green', 'Blue', 'Full')
MARK = 'identity screen'
ANALYSIS = ('no-neg-lobe', 'no-tie', 'allow-fit-pending', 'allow-missing-models', 'allow-missing-lightcurves',
            'no-identity-merge', 'no-anchor-assoc', 'rotate-parents')
# module switches the snapshot scripts read that core/legacy.run does not clear (it clears P3_* only)
LEAKY_ENV = ('RECHAIN_HOST', 'RECHAIN_NODES', 'RECHAIN_CHAIN_NEED_GB', 'RECHAIN_LC_CACHE_MAX', 'RECHAIN_FITS_DIR')


def _T():
    """The shared helpers (lib/catalog/tier_sublst.py); imported lazily so this file also runs as a Python 3.6 child script."""
    from exopipe.phase3.lib.catalog import tier_sublst
    return tier_sublst


def _cfg():
    from exopipe.phase3.config import CFG
    return CFG


def refuse(msg):
    from exopipe.phase3.core.record import Refusal
    raise Refusal(msg)


# =====================================================================================================================
# child entry points (run through core/legacy.run under the production interpreter; stdlib + clv15 only)
# =====================================================================================================================

def child_errmodel(clv):
    """phase3_snapshot_tiecache._errmodel() (snapshot_rechain_batch.sh:164)."""
    sys.path.insert(0, clv)
    import phase3_snapshot_tiecache as t
    print(t._errmodel())
    return 0


def child_fitsplit(clv, src, out, nparts):
    """snapshot_rechain_batch.sh:197-224 (the shard split heredoc), with the parts weighted equally: whole (cell, minute,
    band group) tasks (phase3_snapshot_fit._group_order) are given to the least-loaded part, largest first. Production
    weighted parts by each node's free cores read over ssh; a part is now a runner item that any node claims and fits with
    its own free cores - load balance only, the rows fitted are the same."""
    sys.path.insert(0, clv)
    from phase3_snapshot_fit import _group_order
    os.makedirs(out, exist_ok=True)
    rows = list(csv.DictReader(open(src)))
    g = {}
    for r in rows:
        g.setdefault((r['cell'], r['ts'][:13], _group_order(r)[0]), []).append(r)
    parts = ['part%02d' % (i + 1) for i in range(max(1, int(nparts)))]
    load = dict((p, 0) for p in parts)
    part = dict((p, []) for p in parts)
    for k in sorted(g, key=lambda k: -len(g[k])):
        p = min(parts, key=lambda p: load[p])
        load[p] += len(g[k])
        part[p] += g[k]
    for p in parts:
        if not part[p]:
            continue
        with open(os.path.join(out, p + '.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(part[p])
    print('FITSPLIT ' + ' '.join('%s:%d' % (p, len(part[p])) for p in parts if part[p]))
    return 0


# =====================================================================================================================
# ported shell / heredoc logic (pure functions; unit-tested on synthetic data)
# =====================================================================================================================

def parse_cells_file(path):
    """snapshot_rechain_batch.sh:78: awk '!/^#/ && NF {print (NF>=2 && $1 ~ /h$/) ? $1"_"$2 : $1}' | awk '!s[$0]++'."""
    out, seen = [], set()
    for line in open(path):
        if line.startswith('#'):
            continue
        f = line.split()
        if not f:
            continue
        c = f[0] + '_' + f[1] if (len(f) >= 2 and f[0].endswith('h')) else f[0]
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


def marker_lines(path):
    try:
        return [ln.rstrip('\n') for ln in open(path)]
    except OSError:
        return None


def marker_has(lines, prefix):
    return any(ln.startswith(prefix) for ln in (lines or ()))


def marker_value(lines, key):
    for ln in lines or ():
        m = re.match(r'^%s=(.*)$' % re.escape(key), ln)
        if m:
            return m.group(1)
    return None


def verify_code_sha256(path):
    """`(cd / && sha256sum -c --quiet FILE)` (snapshot_rechain_batch.sh:76): every 'sha256  path' line verifies. Returns the
    list of failures ('path: reason')."""
    bad = []
    for ln in open(path):
        ln = ln.strip()
        if not ln:
            continue
        m = re.match(r'^([0-9a-f]{64})\s+\*?(.+)$', ln)
        if not m:
            bad.append('%s: malformed line' % ln[:80])
            continue
        p = m.group(2) if m.group(2).startswith('/') else os.path.join('/', m.group(2))
        try:
            if sha256_file(p) != m.group(1):
                bad.append('%s: FAILED' % p)
        except OSError as e:
            bad.append('%s: %s' % (p, e))
    return bad


def eligibility_bad(eligibility_csv, statuses):
    """snapshot_rechain_batch.sh:97-100: the cells whose status is not eligible ('cell STATUS')."""
    return ['%s %s' % (r['cell'], r['status']) for r in csv.DictReader(open(eligibility_csv)) if r['status'] not in statuses]


def frame_time_marks(fleet_out, cells, basis):
    """snapshot_rechain_batch.sh:135-155 (PYF): (n searches, n pre-fix 1-minute tiers, cells, n pre-fix wideband pairings,
    cells) from the runinfo_*_median.json of every cell."""
    n = n1 = nw = 0
    c1, cw = set(), set()
    for c in cells:
        for rp in sorted(glob.glob(os.path.join(fleet_out, c, 'runinfo_*_median.json'))):
            try:
                ri = json.load(open(rp))
            except (ValueError, OSError):
                continue
            n += 1
            if os.path.basename(rp).split('_')[1] in WIDE:
                if 'wideband_pairing' not in ri:
                    nw += 1
                    cw.add(c)
            elif ri.get('frame_time_1min') != basis:
                n1 += 1
                c1.add(c)
    return n, n1, len(c1), nw, len(cw)


def unmarked_wideband_rows(store):
    """Wideband rows of a fit store without the identity-screen mark (what archive quarantine-rows moves out)."""
    csv.field_size_limit(10 ** 9)
    n = 0
    with open(store, newline='') as f:
        for r in csv.DictReader(f):
            if r.get('subband') in WIDE and MARK not in (r.get('image') or ''):
                n += 1
    return n


def template_of(store):
    return store[:-4] + '_template.csv' if store.endswith('.csv') else store + '_template.csv'


def combine_stores(b1store, stores, out):
    """snapshot_rechain_batch.sh:264-268: S = the batch-1 store, then every cell store's rows (its header must equal the
    batch-1 header, byte for byte), and the template side stores likewise (a missing T is created by the first copy).
    Returns (rows, template rows). Raises ValueError naming a store whose header differs."""
    T = template_of(out)
    for p in (out, T):
        if os.path.exists(p):
            os.remove(p)
    shutil.copy2(b1store, out)
    if os.path.exists(template_of(b1store)):
        shutil.copy2(template_of(b1store), T)
    with open(b1store, 'rb') as f:
        H = f.readline()
    for st in stores:
        with open(st, 'rb') as f:
            if f.readline().rstrip(b'\n') != H.rstrip(b'\n'):
                raise ValueError('%s header differs' % st)
            rest = f.read()
        with open(out, 'ab') as o:
            o.write(rest)
        t = template_of(st)
        if os.path.exists(t) and os.path.getsize(t) > 0:
            if os.path.exists(T):
                with open(t, 'rb') as f:
                    f.readline()
                    rest = f.read()
                with open(T, 'ab') as o:
                    o.write(rest)
            else:
                shutil.copy2(t, T)
    nrows = _T().count_lines(out) - 1
    nt = _T().count_lines(T) - 1 if os.path.exists(T) else 0
    return nrows, nt


def union_csv(paths, out, fieldnames=None):
    """The union heredocs (PYU, :251-258 and :288-295): every row of every list, one header (the fixed repair columns, or the
    last list's own). Returns the row count."""
    rows, hdr = [], fieldnames
    for p in paths:
        rd = csv.DictReader(open(p))
        if fieldnames is None:
            hdr = rd.fieldnames
        rows += list(rd)
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=hdr or ['cell'])
        w.writeheader()
        w.writerows(rows)
    return len(rows)


REPAIR_COLS = ['cell', 'subband', 'stokes', 'tier', 'ts', 'ra_deg', 'dec_deg', 'snr']


def prefit_counts(prefit_dir):
    """(n at the per-frame fit step, n loaded) summed over the cells' chain_summary.json (snapshot_rechain_batch.sh:297-300),
    and {cell: n at the fit step} for the tie split."""
    ndet = nload = 0
    per = {}
    for p in sorted(glob.glob(os.path.join(prefit_dir, '*', 'chain_summary.json'))):
        d = json.load(open(p))
        n = d['prefit']['n_at_per_frame_fit_step']
        per[os.path.basename(os.path.dirname(p))] = n
        ndet += n
        nload += sum(v for k, v in d.items() if k.startswith('n_loaded_'))
    return ndet, nload, per


def tie_shards(n, rule):
    """snapshot_rechain_batch.sh:322: >= rule[0] detections at the fit step -> rule[1] imfit shards, else rule[2]."""
    return rule[1] if n >= rule[0] else rule[2]


def memory_need(model, nload, ndet, ncells, lc_cache_max):
    """snapshot_rechain_batch.sh:349: max(min, ceil(base + 2.2 GB per million loaded + 10 GB per million at the fit step +
    min(LC cache, 30 x cells) x 0.031 GB))."""
    return max(model['min_gb'], int(math.ceil(model['base_gb'] + model['gb_per_m_loaded'] * nload / 1e6
                                              + model['gb_per_m_fitstep'] * ndet / 1e6
                                              + min(lc_cache_max, model['tables_per_cell'] * ncells) * model['gb_per_table'])))


def argparse_defaults(path):
    """{'--flag': default} of the add_argument calls of a script (AST; defaults given as literals or module-level names)."""
    import ast
    T = _T()
    lits = T.module_literals(path)
    out = {}
    for node in ast.walk(ast.parse(open(path).read(), path)):
        if isinstance(node, ast.Call) and getattr(node.func, 'attr', '') == 'add_argument' and node.args \
                and isinstance(node.args[0], ast.Constant) and str(node.args[0].value).startswith('--'):
            for kw in node.keywords:
                if kw.arg == 'default':
                    try:
                        out[node.args[0].value] = T._ev(kw.value, lits)
                    except T._Unresolved:
                        out[node.args[0].value] = None
    return out


# =====================================================================================================================
# the snapshot context
# =====================================================================================================================

def _cells_file(args):
    spec = (getattr(args, 'cells', None) or '').strip()
    if not (spec.startswith('@') and ',' not in spec):
        raise SystemExit('error: catalog --tier snapshot needs --cells @CELLS_FILE (the cumulative cell list of the run, '
                         'one cell per line, LST_DATE or "LST DATE"; its sha256 binds the approval, B-52)')
    return os.path.abspath(spec[1:])


def default_out(args):
    c = _cfg().catalog_snapshot
    name = getattr(args, 'run_label', None) or 'Run_SNAP'
    return os.path.join(c.WORK_ROOT, 'snapshot_rechain_%s' % name, name)


class Ctx:
    def __init__(self, args, run):
        T = _T()
        cfg = _cfg()
        c = cfg.catalog_snapshot
        self.T, self.args, self.run, self.cfg, self.c = T, args, run, cfg, c
        self.dry = bool(run.dry_run)
        self.name = getattr(args, 'run_label', None)
        if not self.name:
            raise SystemExit('error: catalog --tier snapshot needs --run Run_SNAP<...> (the batch run name)')
        self.cells_file = _cells_file(args)
        if not os.path.exists(self.cells_file):
            raise SystemExit('error: cells file %s not found' % self.cells_file)
        self.cells = parse_cells_file(self.cells_file)
        from exopipe.phase3.core import cells as C
        for cell in self.cells:
            try:
                lst, date, sb = C.parse_night_cell(cell)
            except ValueError as e:
                raise SystemExit('error: %s: %s' % (self.cells_file, e))
            if sb:
                raise SystemExit('error: %s: %s names a subband - the chain takes whole night cells' % (self.cells_file, cell))
        self.sha = sha256_file(self.cells_file)
        self.RUN = os.path.abspath(run.out_root)
        self.prod = cfg.production_root_of(self.RUN) is not None
        self.B = c.BATCH_DIR
        self.fits_prod = os.path.join(self.B, 'fits')
        self.tiecache_prod = os.path.join(self.B, 'tiecache')
        if self.prod:
            self.W = os.path.dirname(self.RUN)
            self.L = os.path.join(self.B, 'rechain_%s' % self.name)
            self.fits = self.fits_prod
            self.tiecache = self.tiecache_prod
            self.log = T.Log(os.path.join(cfg.paths.OPS_ROOT, 'snapshot_rechain_%s.log' % self.name),
                             stamp='%a %b %d %H:%M:%S UTC %Y')
            self.node_fast = os.path.join(c.NODE_FAST_ROOT, 'rechain_%s' % self.name)
            self.active = os.path.join(self.B, 'rechain_active', '%s.cells' % self.name)
            self.refastro_scratch = None
        else:
            side = os.path.join(self.RUN, '_side')
            self.W = os.path.join(side, 'work')
            self.L = os.path.join(side, 'rechain_%s' % self.name)
            self.fits = os.path.join(side, 'fits')
            self.tiecache = os.path.join(side, 'tiecache')
            self.log = T.Log(os.path.join(side, 'snapshot_rechain_%s.log' % self.name), stamp='%a %b %d %H:%M:%S UTC %Y')
            self.node_fast = None             # node-local: <scratch root of the item's node>/rechain_<NAME> (core/fitsio)
            self.active = None
            self.refastro_scratch = True
        self.S = os.path.join(self.L, 'fits_store.csv')
        self.recdir = os.path.join(self.RUN, '_records', 'catalog', self.name)
        self.ops = T.legacy_root('exopipe_ops')
        self.clv = T.legacy_root('clv15')
        self.py36 = T.interpreter('py36', self.dry)
        self.chain_py = os.path.join(self.clv, 'phase3_snapshot_chain.py')
        self.marker = c.APPROVAL_MARKER.format(name=self.name)
        self.marker_lines = marker_lines(self.marker)
        self.t0 = time.time()
        self.analysis = self._analysis(arg_list(args, 'snap_analysis'))

    def _analysis(self, items):
        out = []
        for it in items:
            k, _, v = it.partition('=')
            if k not in ANALYSIS:
                raise SystemExit('error: --snap-analysis %s: switches are %s' % (k, ', '.join(ANALYSIS)))
            out.append((k, v))
        if out and self.prod:
            refuse('--snap-analysis switches are analysis only: refused for a production --out (B-83)')
        return out

    def opsc(self, n):
        return os.path.join(self.ops, n)

    def node_local(self, work=None):
        if self.node_fast:
            return self.node_fast
        from exopipe.phase3.core import fitsio
        from exopipe.phase3.core.record import Refusal
        try:
            return os.path.join(fitsio.scratch_root(work), 'rechain_%s' % self.name)
        except Refusal:
            if not self.dry:
                raise
            return '<node-local scratch>/rechain_%s' % self.name

    def chainargs(self):
        """CHAINARGS of snapshot_rechain_batch.sh:280 plus every [arg] key whose config value differs from the script's own
        default (production passes none of those) and the analysis switches."""
        c = self.c
        a = ['--proto-root', self.cfg.paths.SNAPSHOT_FLEET_OUT, '--master-catalogs', c.MASTER_CATALOGS,
             '--curve-constants', c.CURVE_CONSTANTS, '--fit-store', self.S, '--lightcurves-root', c.LIGHTCURVES_ROOT]
        d = argparse_defaults(self.chain_py) if os.path.exists(self.chain_py) else {}
        for flag, val in (('--samples-dir', self.cfg.paths.CLV15_ROOT), ('--store-floor', c.STORE_FLOOR),
                          ('--seed-floor', c.SEED_FLOOR), ('--assoc-floor', c.ASSOC_FLOOR),
                          ('--repeat-floor', c.REPEAT_FLOOR), ('--singleton-floor', c.SINGLETON_FLOOR['I']),
                          ('--singleton-floor-v', c.SINGLETON_FLOOR['V']), ('--struct-k', c.STRUCT['k']),
                          ('--dirty-cells', c.DIRTY_CELLS), ('--dirty-i-floor', c.DIRTY_I_FLOOR)):
            if flag not in d or self.T._norm(d[flag]) != self.T._norm(val):
                a += [flag, str(val)]
        for k, v in self.analysis:
            a += ['--' + k] + ([v] if v else [])
        return a

    def chain_env(self, **extra):
        e = {}
        if not self.c.FLOW_MODE:
            e['P3_SNAP_FIT_RAW'] = '1'
        if self.refastro_scratch:            # refastro's default scratch is under a production root (/fast/gh/main/phase3)
            from exopipe.phase3.core import fitsio
            from exopipe.phase3.core.record import Refusal
            try:
                root = fitsio.scratch_root(arg_val(self.args, 'work'))
            except Refusal:
                if not self.dry:
                    raise
                root = '<node-local scratch: --work or site SCRATCH_ROOT>'
            e['P3_SNAP_REFASTRO_SCRATCH'] = os.path.join(root, 'catalog_snapshot_%s_refastro' % self.name)
        e.update(dict((k, str(v)) for k, v in extra.items()))
        return e

    def waived(self, key):
        """A gate bypass: --waive <name> --ruling (recorded), or the production marker's switch line (window parity)."""
        r = self.run.waived(WAIVERS[key])
        if r:
            return 'waiver %s: %s' % (WAIVERS[key], r)
        if marker_has(self.marker_lines, MARKER_SWITCH[key]):
            ln = [x for x in self.marker_lines if x.startswith(MARKER_SWITCH[key])][0]
            return 'marker %s: %s' % (self.marker, ln[:200])
        return None

    def limit_h(self):
        v = marker_value(self.marker_lines, 'LIMIT_HOURS')
        return int(v) if v and v.isdigit() else int(self.c.LIMIT_H)


def arg_val(args, name):
    return getattr(args, name, None)


def arg_list(args, name):
    v = getattr(args, name, None)
    return [x.strip() for x in str(v).split(',') if x.strip()] if v else []


# =====================================================================================================================
# [lit] bindings of the snapshot chain (spec section 3; module-level names)
# =====================================================================================================================

CH, RA = 'phase3_snapshot_chain.py', 'phase3_snapshot_refastro.py'
LIT = {
    'catalog_snapshot.MIN_ELEVATION_DEG': ('clv15', CH, 'MIN_ELEVATION_DEG'),
    'catalog_snapshot.ATEAM_ANCHOR_DEG': ('clv15', CH, 'ATEAM_ANCHOR_DEG'),
    'catalog_snapshot.ATEAM_MIN_DEG': ('clv15', CH, 'ATEAM_MIN_DEG'),
    'catalog_snapshot.ATEAM_SLOPE': ('clv15', CH, 'ATEAM_SLOPE'),
    'catalog_snapshot.ATEAM_FIXED_DEG': ('clv15', CH, 'ATEAM_FIXED_DEG'),
    'catalog_snapshot.ATEAM_FLUX_NPZ': ('clv15', CH, 'ATEAM_FLUX_NPZ'),
    'catalog_snapshot.EPHEM_DEG': ('clv15', CH, 'EPHEM_DEG'),
    'catalog_snapshot.JUPITER_BEAMS': ('clv15', CH, 'JUPITER_BEAMS'),
    'catalog_snapshot.GHOST_DATES': ('clv15', CH, 'GHOST_DATES'),
    'catalog_snapshot.GHOST_OFFSET_DEG': ('clv15', CH, 'GHOST_OFFSET_DEG'),
    'catalog_snapshot.GHOST_SIGMA_DEG': ('clv15', CH, 'GHOST_SIGMA_DEG'),
    'catalog_snapshot.GHOST_MATCH_PERCENTILE': ('clv15', CH, 'GHOST_MATCH_PERCENTILE'),
    'catalog_snapshot.NEG_LOBE': ('clv15', CH, ('NEG_LOBE_RMS', 'NEG_LOBE_RATIO', 'NEG_LOBE_BEAMS')),
    'catalog_snapshot.VHF_FLOOR': ('clv15', CH, 'VHF_FLOOR'),
    'catalog_snapshot.VHF_EVENT_VETO': ('clv15', CH, 'VHF_EVENT_VETO'),
    'catalog_snapshot.ASYM': ('clv15', CH, {'angle_deg': 'ASYM_ANGLE_DEG', 'half_deg': 'ASYM_HALF_DEG',
                                            'r_beams': 'ASYM_R_BEAMS', 'snap_px': 'ASYM_SNAP_PX', 'd': 'ASYM_D',
                                            'z': 'ASYM_Z'}),
    'catalog_snapshot.E_MAX': ('clv15', CH, 'E_MAX'),
    'catalog_snapshot.VETO_RECORD_SNR': ('clv15', CH, 'VETO_RECORD_SNR'),
    'catalog_snapshot.E_PROMOTE': ('clv15', CH, 'E_PROMOTE'),
    'catalog_snapshot.XM_PERCENTILE': ('clv15', CH, 'XM_PERCENTILE'),
    'catalog_snapshot.INSTANT_BIN_S': ('clv15', CH, 'INSTANT_BIN_S'),
    'catalog_snapshot.REPEAT_CONSTRAIN_K': ('clv15', CH, 'REPEAT_CONSTRAIN_K'),
    'catalog_snapshot.REPEAT_SKY_RATE_ARCSEC_S': ('clv15', CH, 'REPEAT_SKY_RATE_ARCSEC_S'),
    'catalog_snapshot.CHECKSOURCES': ('clv15', CH, 'CHECKSOURCES'),
    'catalog_snapshot.GATE3D': ('clv15', CH, {'ratio': 'GATE3D_RATIO', 'snr': 'GATE3D_SNR',
                                              'parent_scale_refs': 'PARENT_SCALE_REFS',
                                              'match_percentile': 'GATE3D_MATCH_PERCENTILE',
                                              'self_parent_all_bands': 'GATE3D_SELF_PARENT_ALL_BANDS',
                                              'ratio_high': 'GATE3D_RATIO_HIGH', 'alpha': 'GATE3D_ALPHA',
                                              'alpha_min_sb': 'GATE3D_ALPHA_MINSB'}),
    'catalog_snapshot.GATE3D_NTHR': ('clv15', CH, 'GATE3D_NTHR'),
    'catalog_snapshot.REFASTRO_SHARDS_DEFAULT': ('clv15', RA, 'REFASTRO_SHARDS'),
    'catalog_snapshot.REF_MAX_SEP_DEG': ('clv15', RA, 'REF_MAX_SEP_DEG'),
}


# =====================================================================================================================
# runner jobs of this step (fit rounds, tie pass)
# =====================================================================================================================

def _nodes(ctx):
    from exopipe.phase3.core import runner
    spec = arg_val(ctx.args, 'snap_chain_nodes') or 'snapshot'
    return runner.parse_nodes(spec)


def _item_argv(ctx, stage):
    from exopipe.phase3.steps.run import forward_argv
    a = ['catalog', '--tier', 'snapshot', '--run', ctx.name, '--cells', '@' + ctx.cells_file, '--stages', stage,
         '--snap-item', '{item}', '--out', ctx.RUN]
    if arg_val(ctx.args, 'work'):
        a += ['--work', ctx.args.work]
    if arg_val(ctx.args, 'cores') not in (None, 'auto'):
        a += ['--cores', ctx.args.cores]        # an explicit core list / 'none' travels to the items; 'auto' is per node
    return a + forward_argv(ctx.args)


def submit_and_wait(ctx, s, name, stage, items, lanes, plan):
    """Submit a core/runner job of this step over ``items`` on the --snap-chain-nodes (``lanes`` per node) and wait for its
    record (DESIGN 6: never log text or pids). Refuses unless every item ends done."""
    from exopipe.phase3.core import runner
    nodes = [(h, lanes) for h, _ in _nodes(ctx)]
    argv = _item_argv(ctx, stage)
    plan.append('runner job %s: %d items on %s: exopipe3 %s' % (name, len(items), ','.join('%s:%d' % n for n in nodes),
                                                               ' '.join(argv)))
    if ctx.dry:
        return None
    job = runner.submit('%s_%s' % (ctx.name, name), argv, items, nodes)
    s.u.meta.setdefault('runner_jobs', []).append(job)
    poll = ctx.cfg.runner.POLL_S
    while runner.job_state(job)[0] != 'finished':
        time.sleep(poll)
    bad = [(it, runner.item_state(job, it).get('status')) for it in runner.items(job)
           if runner.item_state(job, it).get('status') != 'done']
    if bad:
        refuse('runner job %s (%s): %d of %d items not done: %s' % (job, name, len(bad), len(items),
                                                                   '; '.join('%s %s' % b for b in bad[:5])))
    return job


def fit_round(ctx, s, tag, pending, plan):
    """snapshot_rechain_batch.sh:195-242 (fit_round): split the pending rows into whole (cell, minute, band group) tasks,
    fit them as runner items (internal stage fit-shard), append the new rows to every cell's store (and to S when it
    exists). The prefit rounds skip keys already in S (--skip-store S); the repair round refits every row."""
    T = ctx.T
    sh = os.path.join(ctx.L, 'shards_%s' % tag)
    nparts = len(_nodes(ctx))        # one part per node (production: one shard per node, weighted by its free cores)
    plan.append('fit round %s: %s -> %d parts in %s (child %s fitsplit), fit-grouped per part, then snapshot_store_append.py '
                '--fits-dir %s%s' % (tag, pending, nparts, sh, ctx.py36, ctx.fits,
                                     ' --also %s' % ctx.S if os.path.exists(ctx.S) else ''))
    if ctx.dry:
        submit_and_wait(ctx, s, 'fit_%s' % tag, 'fit-shard', ['%s:part01' % tag], 1, plan)
        return
    if os.path.lexists(sh):
        shutil.rmtree(sh)
    res = T.call(ctx, s, os.path.abspath(__file__), ['fitsplit', ctx.clv, pending, sh, str(nparts)], 'py36',
                 log=ctx.log.path, cwd=ctx.L)
    T.need_ok(res, 'shard split %s' % tag, r'FITSPLIT', say=lambda: ctx.log.say('REFUSING: shard split %s' % tag))
    parts = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(sh, 'part*.csv')))
    ctx.log.say('fit round %s: %s' % (tag, ' '.join('%s=%d' % (p, T.count_lines(os.path.join(sh, p + '.csv')) - 1)
                                                    for p in parts)))
    submit_and_wait(ctx, s, 'fit_%s' % tag, 'fit-shard', ['%s:%s' % (tag, p) for p in parts], 1, plan)
    new = sorted(p for p in glob.glob(os.path.join(sh, 'fits_%s_*.csv' % tag)) if not p.endswith('_template.csv'))
    also = ['--also', ctx.S] if os.path.exists(ctx.S) else []
    res = T.call(ctx, s, ctx.opsc('snapshot_store_append.py'), new + also + ['--fits-dir', ctx.fits], 'py36',
                 log=ctx.log.path, cwd=ctx.L)
    T.need_ok(res, 'append %s' % tag, r'APPENDED|REFUS', say=lambda: ctx.log.say('REFUSING: append %s' % tag))
    s.u.note(T.last_line(res.tail, 'APPENDED'))


def item_fit_shard(ctx, s, item, plan):
    """One part of a fit round on this node (snapshot_rechain_batch.sh:228-236): fit-grouped into a node-local store with
    (free cores - FIT_PROCS_RESERVE) processes, one BLAS thread each, then copied to the shard dir."""
    T = ctx.T
    tag, _, part = item.partition(':')
    sh = os.path.join(ctx.L, 'shards_%s' % tag)
    shard = os.path.join(sh, part + '.csv')
    nl_dir = ctx.node_local(arg_val(ctx.args, 'work'))
    nl = os.path.join(nl_dir, 'fits_%s_%s.csv' % (tag, part))
    procs = max(1, len(os.sched_getaffinity(0)) - ctx.c.FIT_PROCS_RESERVE)
    skip = ['--skip-store', ctx.S] if tag.startswith('prefit') else []
    a = ['fit-grouped', '--pending', shard, '--store', nl] + skip + ['--proto-root', ctx.cfg.paths.SNAPSHOT_FLEET_OUT,
                                                                     '--procs', str(procs)]
    plan.append('fit-shard %s: %s %s %s' % (item, ctx.py36, os.path.join(ctx.clv, 'phase3_snapshot_fit.py'), ' '.join(a)))
    if ctx.dry:
        return
    if not os.path.exists(shard):
        refuse('fit-shard %s: no shard %s' % (item, shard))
    os.makedirs(nl_dir, exist_ok=True)
    for p in (nl, template_of(nl)):
        if os.path.exists(p):
            os.remove(p)
    s.inp(shard)
    s.code(os.path.join(ctx.clv, 'phase3_snapshot_fit.py'))
    res = T.call(ctx, s, os.path.join(ctx.clv, 'phase3_snapshot_fit.py'), a, 'py36',
                 env={'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'},
                 log=os.path.join(ctx.L, 'fit_%s_%s.log' % (tag, part)), cwd=ctx.L)
    T.need_ok(res, 'fit-shard %s (phase3_snapshot_fit.py fit-grouped)' % item, r'DONE|REFUS|Error')
    dst = os.path.join(sh, os.path.basename(nl))
    shutil.copy2(nl, dst)
    if os.path.exists(template_of(nl)):
        shutil.copy2(template_of(nl), os.path.join(sh, os.path.basename(template_of(nl))))
    s.out(dst, 'csv_table')


def item_tie(ctx, s, cell, plan):
    """One cell of the tie pass (snapshot_rechain_batch.sh:326-332): the chain with --stop-after-tie writes the cell's tie
    cache rows; 16 imfit shards when the cell has >= 30,000 detections at the per-frame fit step, else 8."""
    T = ctx.T
    pr = T.stage_record(ctx.RUN, '%s.prefit' % ctx.name) or {}
    per = (pr.get('meta') or {}).get('per_cell_fitstep') or {}
    n = per.get(cell)
    if n is None and not ctx.dry:
        refuse('tie %s: the prefit record of %s holds no count for the cell (run the prefit stage)' % (cell, ctx.name))
    shards = tie_shards(n or 0, ctx.c.TIE_SHARDS)
    out = os.path.join(ctx.node_local(arg_val(ctx.args, 'work')), 'tie', cell)
    a = ctx.chainargs() + ['--cells', cell, '--out', out, '--veto-record-root', '', '--stop-after-tie']
    plan.append('tie %s: P3_TIE_CACHE_DIR=%s P3_REFASTRO_SHARDS=%d %s %s %s' % (cell, ctx.tiecache, shards, ctx.py36,
                                                                             ctx.chain_py, ' '.join(a)))
    if ctx.dry:
        return
    os.makedirs(out, exist_ok=True)
    s.code(ctx.chain_py)
    s.inp(ctx.S)
    res = T.call(ctx, s, ctx.chain_py, a, 'py36',
                 env=ctx.chain_env(P3_TIE_CACHE_DIR=ctx.tiecache, P3_REFASTRO_SHARDS=shards),
                 log=os.path.join(ctx.L, 'tie', '%s.log' % cell), cwd=ctx.L)
    T.need_ok(res, 'tie %s (--stop-after-tie)' % cell, r'STOP-AFTER-TIE|REFUS|Error')
    s.u.note(T.last_line(res.tail, 'tie cache:'))
    s.out(os.path.join(out, 'chain_summary.json'), 'snapshot_tie_summary')
    tc = os.path.join(ctx.tiecache, 'tie_%s.csv' % cell)
    if os.path.exists(tc):
        s.out(tc, 'tie_cache')


# =====================================================================================================================
# stages
# =====================================================================================================================

def served_run(ctx):
    """The run served on SERVED_PORT_CHECK (snapshot_rechain_batch.sh:103): --snap-served-run, else the --run of the process
    listening on the port, read on the host that holds paths.SNAPSHOT_REVIEW_ROOT (site NODE_LOCAL_ROOTS; ssh with
    SSH_AUTH_SOCK cleared, CLAUDE.md 8, when that is another host). Returns (run dir or None, how)."""
    over = arg_val(ctx.args, 'snap_served_run')
    if over:
        return over, '--snap-served-run'
    import socket
    review = ctx.cfg.paths.SNAPSHOT_REVIEW_ROOT
    host = next((h for h, roots in (ctx.cfg.site.NODE_LOCAL_ROOTS or {}).items()
                 if any(review.startswith(r.rstrip('/') + '/') for r in roots)), None)
    port = ctx.c.SERVED_PORT_CHECK
    sh = ('p=$(ss -tlnp 2>/dev/null | grep ":%d " | grep -oE "pid=[0-9]+" | cut -d= -f2 | head -1); [ -n "$p" ] && '
          'tr "\\0" " " < /proc/$p/cmdline | grep -oE -- "--run [^ ]+" | cut -d" " -f2' % port)
    if host in (None, socket.gethostname().split('.')[0], socket.gethostname()):
        p = subprocess.run(['bash', '-c', sh], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, universal_newlines=True)
        how = 'local ss on %s' % socket.gethostname()
    else:
        env = dict(os.environ)
        env.update(ctx.cfg.runner.SSH_PREFIX_ENV)
        p = subprocess.run(['ssh', '-n', '-o', 'ConnectTimeout=20', host, sh], stdout=subprocess.PIPE,
                           stderr=subprocess.DEVNULL, universal_newlines=True, env=env, timeout=120)
        how = 'ssh %s ss' % host
    return (p.stdout.strip().splitlines() or [None])[0], how


def st_preflight(ctx, s, plan):
    T, c, u = ctx.T, ctx.c, s.u
    ok = ctx.name.startswith(c.RUN_PREFIX)
    plan.append('P-G1 run name %s starts with %s: %s' % (ctx.name, c.RUN_PREFIX, ok))
    if not ok:
        refuse('P-G1: RUN_NAME must start with %s (%s)' % (c.RUN_PREFIX, ctx.name))
    plan.append('cells file %s: %d cells, sha256 %s' % (ctx.cells_file, len(ctx.cells), ctx.sha))
    if not ctx.cells:
        refuse('empty cell list %s' % ctx.cells_file)
    u.meta.update(cells=ctx.cells, cells_file=ctx.cells_file, cells_sha256=ctx.sha)
    # P-G2 approval and list binding
    appr = arg_val(ctx.args, 'snap_approval')
    if appr:
        how = '--snap-approval: %s' % appr
        good = True
    else:
        good = ctx.marker_lines is not None and ('CELLS_FILE_SHA256=%s' % ctx.sha) in ctx.marker_lines
        how = 'marker %s %s CELLS_FILE_SHA256=%s' % (ctx.marker, 'carries' if good else
                                                     ('lacks' if ctx.marker_lines is not None else 'MISSING:'), ctx.sha)
    plan.append('P-G2 approval: %s' % how)
    u.check('P-G2.approval', good, detail=how)
    u.meta['approval'] = how
    if not good and not ctx.dry:
        refuse('P-G2: no approval for %s: give --snap-approval "<Gregg\'s words, UTC, ledger ID>" or the marker %s with '
               'CELLS_FILE_SHA256=%s (the list as it is now)' % (ctx.name, ctx.marker, ctx.sha))
    # P-G3 tested code (window form: the session's sha256 file of the in-place code)
    cs = c.CODE_SHA256.format(name=ctx.name)
    if os.path.exists(cs):
        bad = verify_code_sha256(cs)
        plan.append('P-G3 tested code %s: %s' % (cs, 'verifies' if not bad else '%d FAIL, e.g. %s' % (len(bad), bad[:2])))
        u.check('P-G3.tested-code', not bad, value=len(bad), detail='; '.join(bad[:3]))
        if bad and not ctx.dry:
            refuse('P-G3: installed code differs from the tested code (%s): %s' % (cs, bad[:3]))
    else:
        plan.append('P-G3 tested code: no %s - would refuse (run --code COMMIT replaces it at cutover, B-51)' % cs)
        if not ctx.dry:
            refuse('P-G3: no code fingerprint %s (sha256sum of the installed code the session tested)' % cs)
    # P-G4 batch-1 store
    b1 = os.path.exists(c.BATCH1_STORE) and os.path.getsize(c.BATCH1_STORE) > 0
    plan.append('P-G4 batch-1 store %s on this host: %s' % (c.BATCH1_STORE, b1))
    if not b1 and not ctx.dry:
        refuse('P-G4: no batch-1 store %s on this host (the chain host of the site file)' % c.BATCH1_STORE)
    # P-G5 single writer
    if ctx.prod:
        plan.append('P-G5 single writer: needs the runner leases snapshot-store + per cell (K7, not in core/runner yet)')
        if not ctx.dry:
            refuse('P-G5: a production snapshot chain needs the runner leases snapshot-store and per-cell (spec K7; the '
                   'pgrep-over-ssh scan is not ported) - not in core/runner yet')
    else:
        plan.append('P-G5 single writer: non-production run - every store and tie-cache write goes to copies under %s '
                    '(production writers are not blocked; the store-quiet wait below still applies)' % os.path.dirname(ctx.L))
    # P-G6 eligibility
    elig = os.path.join(ctx.L, 'eligibility.csv')
    plan.append('P-G6 eligibility: %s %s --cells-file %s --no-store --out %s (statuses %s)' % (
        ctx.py36, ctx.opsc('snapshot_batch_audit.py'), ctx.cells_file, elig, ', '.join(c.ELIGIBLE_STATUSES)))
    if not ctx.dry:
        os.makedirs(ctx.L, exist_ok=True)
        res = T.call(ctx, s, ctx.opsc('snapshot_batch_audit.py'), ['--cells-file', ctx.cells_file, '--no-store', '--out', elig],
                     'py36', log=os.path.join(ctx.L, 'eligibility.log'), cwd=ctx.L)
        T.need_ok(res, 'P-G6 eligibility audit failed (%s/eligibility.log)' % ctx.L)
        badc = eligibility_bad(elig, c.ELIGIBLE_STATUSES)
        u.check('P-G6.eligibility', not badc, value=len(badc), detail='; '.join(badc[:5]))
        if badc:
            refuse('P-G6: cells not eligible: %s' % '; '.join(badc))
        s.inp(elig)
    # P-G7 served cells kept
    try:
        srun, how = served_run(ctx)
    except Exception as e:
        srun, how = None, 'served-run lookup failed: %s' % e
    if srun and os.path.exists(os.path.join(srun, 'chain_summary.json')):
        scells = json.load(open(os.path.join(srun, 'chain_summary.json')))['cells']
        drop = [x for x in scells if x not in ctx.cells]
        nnew = sum(1 for x in ctx.cells if x not in scells)
        w = ctx.waived('served') if drop else None
        plan.append('P-G7 served run %s (%s): %d cells; the list drops %d%s; %d new' % (
            srun, how, len(scells), len(drop), ' (%s)' % w if w else '', nnew))
        u.meta.update(served_run=srun, n_new_cells=nnew, dropped_served=drop)
        u.check('P-G7.served-cells-kept', not drop or bool(w), value=len(drop), detail=w or ' '.join(drop[:10]))
        if drop and not w and not ctx.dry:
            refuse('P-G7: the list drops %d cell(s) of the served run %s: %s (waive snapshot.served-cells-kept with a '
                   'ruling)' % (len(drop), os.path.basename(srun), ' '.join(drop)))
    else:
        plan.append('P-G7 served run: %s (%s) - would refuse (no run served on %d to check against)' % (srun, how,
                                                                                                    c.SERVED_PORT_CHECK))
        if not ctx.dry:
            refuse('P-G7: no run served on %d (its cells are kept and its judgements carry): %s; give --snap-served-run'
                   % (c.SERVED_PORT_CHECK, how))
    # P-G8 prefit writer released (production publishes the active list first: snapshot_cell_prefit.sh skips its cells)
    claims = os.path.join(ctx.B, 'prefit_claims')
    held = [x for x in ctx.cells if os.path.isdir(os.path.join(claims, x))]
    plan.append('P-G8 prefit claims held on listed cells now: %d%s' % (len(held), ' (%s)' % ' '.join(held[:6]) if held else ''))
    if not ctx.dry:
        if ctx.active:
            os.makedirs(os.path.dirname(ctx.active), exist_ok=True)
            shutil.copy(ctx.cells_file, ctx.active + '.tmp')
            os.replace(ctx.active + '.tmp', ctx.active)
            u.note('active list published: %s' % ctx.active)
        tries, wait = c.PREFIT_RELEASE_WAIT
        for i in range(tries):
            held = [x for x in ctx.cells if os.path.isdir(os.path.join(claims, x))]
            if not held:
                break
            if i == tries - 1:
                refuse('P-G8: snapshot_cell_prefit.sh still holds %s after %d min' % (' '.join(held), tries * wait // 60))
            if i == 0:
                ctx.log.say('waiting for snapshot_cell_prefit.sh to finish %s' % ' '.join(held))
            time.sleep(wait)
    # P-G9 band classes (Q14) - p3s3_astrom.band_class, in process (it imports math only)
    try:
        A = T._imp(ctx.ops, 'p3s3_astrom')
        q14 = (A.band_class('41MHz'), A.band_class('64MHz')) == ('Green', 'Blue')
    except Exception as e:
        q14 = False
        plan.append('P-G9 band_class import failed: %s' % e)
    plan.append('P-G9 Q14 band classes (41 MHz Green, 64 MHz Blue): %s' % q14)
    u.check('P-G9.band-classes', q14)
    if not q14 and not ctx.dry:
        refuse('P-G9: Q14 is not implemented (p3s3_astrom.band_class: 41 MHz -> Green, 64 MHz -> Blue; Gregg 2026-10-07)')
    # P-G10 frame-time pre-fix inputs
    basis = T.module_literals(os.path.join(ctx.clv, 'phase3_snapshot_search.py')).get('TIER1MIN_TIME_BASIS')
    ft = frame_time_marks(ctx.cfg.paths.SNAPSHOT_FLEET_OUT, ctx.cells, basis)
    w = ctx.waived('frametime') if (ft[1] or ft[3]) else None
    plan.append('P-G10 frame-time marks (basis %r): %d searches, %d pre-fix 1-minute tiers in %d cells, %d pre-fix wideband in '
                '%d cells%s' % ((basis,) + ft + ((' - %s' % w) if w else '',)))
    u.meta['frame_time_marks'] = list(ft)
    u.check('P-G10.frame-time-prefix', not (ft[1] or ft[3]) or bool(w), value=ft[1] + ft[3], detail=w or '')
    if (ft[1] or ft[3]) and not w and not ctx.dry:
        refuse('P-G10: %d subband searches in %d cells have a pre-fix 1-minute tier and %d wideband searches in %d cells '
               'pre-fix pairing (of %d): waive snapshot.frame-time-prefix with Gregg\'s ruling '
               '(RULING-GREGG-20261008-FRAMETIME-RERUN-REJECTED-ACCEPT-ERROR-UNTIL-MAJOR-REDO)' % (ft[1], ft[2], ft[3], ft[4], ft[0]))
    # P-G11 tie model
    if ctx.dry:
        plan.append('P-G11 tie model: %s %s errmodel %s (refused if = %s unless waived snapshot.tie-model)' % (
            ctx.py36, __file__, ctx.clv, c.PRE_BUNDLE_TIE_MODEL))
    else:
        res = T.call(ctx, s, os.path.abspath(__file__), ['errmodel', ctx.clv], 'py36',
                     log=os.path.join(ctx.L, 'tie_model.log'), cwd=ctx.L)
        tm = ([ln for ln in (res.tail or '').splitlines() if ln.strip() and not ln.startswith('=====')] or [''])[-1].strip()
        if not res.ok or not tm:
            refuse('P-G11: could not read the tie-cache model (%s)' % res.reason)
        u.meta['tie_model'] = tm
        w = ctx.waived('tiemodel') if tm == c.PRE_BUNDLE_TIE_MODEL else None
        plan.append('P-G11 tie model %s%s' % (tm, ' = the pre-bundle model (%s)' % w if w else ''))
        if tm == c.PRE_BUNDLE_TIE_MODEL and not w:
            refuse('P-G11: the installed tie-cache model is still the pre-bundle one (%s): the astrometry bundle is not '
                   'installed - waive snapshot.tie-model with a ruling to chain on it' % tm)
    # P-G12 /fast free space (production: the chain host's /fast)
    where = ctx.W if ctx.prod else (arg_val(ctx.args, 'work') or ctx.RUN)
    probe = where
    while probe and not os.path.exists(probe):
        probe = os.path.dirname(probe)
    free = shutil.disk_usage(probe or '/').free // 10 ** 9
    plan.append('P-G12 free space at %s: %d GB (production needs >= %d GB on the chain host /fast)' % (
        probe, free, c.FAST_FREE_MIN_GB))
    u.meta['free_gb'] = free
    if ctx.prod and free < c.FAST_FREE_MIN_GB and not ctx.dry:
        refuse('P-G12: /fast has %d GB free (< %d)' % (free, c.FAST_FREE_MIN_GB))
    # binding check
    T.binding_gate(u, T.binding_check(LIT), plan, ctx.dry)
    plan.append('chain arguments: %s' % ' '.join(ctx.chainargs()))
    if not ctx.dry:
        ctx.log.say('RECHAIN-START %s (exopipe3 catalog --tier snapshot) host=%s cells=%d list=%s sha256=%s approval=%s' % (
            ctx.name, os.uname()[1], len(ctx.cells), ctx.cells_file, ctx.sha, u.meta.get('approval', '')[:200]))
        rep = os.path.join(ctx.recdir, 'preflight.json')
        os.makedirs(ctx.recdir, exist_ok=True)
        with open(rep, 'w') as f:
            json.dump({'unit': ctx.name, 'plan': plan, 'binding': u.meta.get('binding'), 'cells_sha256': ctx.sha,
                       'utc': T.utc()}, f, indent=1)
        s.out(rep, 'catalog_preflight')


def _store_list(ctx, root):
    return [os.path.join(root, 'fits_%s.csv' % x) for x in ctx.cells if os.path.exists(os.path.join(root, 'fits_%s.csv' % x))]


def st_store(ctx, s, plan):
    T, c = ctx.T, ctx.c
    prod_stores = _store_list(ctx, ctx.fits_prod)
    plan.append('P-G13 stores quiet: %d per-cell stores in %s + %s unwritten for %d min (%d x %d s)' % (
        len(prod_stores), ctx.fits_prod, c.BATCH1_STORE, c.STORE_QUIET_MIN, c.STORE_QUIET_WAIT[0], c.STORE_QUIET_WAIT[1]))
    plan.append('unmarked double-registered wideband rows in those stores REFUSE (archive quarantine-rows moves them; '
                'refactor chat decision 6)')
    if not ctx.prod:
        plan.append('non-production: the stores are COPIED to %s and every append goes there' % ctx.fits)
    plan.append('store repair: per cell %s %s <cell> %s/repair/<cell>.csv%s (%d at a time)' % (
        ctx.py36, ctx.opsc('snapshot_store_repair.py'), ctx.L, '' if ctx.prod else ' --store <copy>', c.LIST_PARALLEL))
    plan.append('combined store %s = batch-1 store + every cell store (header equality; 0 unmarked wideband rows); content '
                'test %s %s <cells>' % (ctx.S, ctx.py36, ctx.opsc('snapshot_store_ready.py')))
    if ctx.dry:
        fit_round(ctx, s, 'repair', os.path.join(ctx.L, 'repair.csv'), plan)
        return
    tries, wait = c.STORE_QUIET_WAIT
    for i in range(tries):
        recent = [p for p in prod_stores + [c.BATCH1_STORE] if os.path.exists(p)
                  and time.time() - os.path.getmtime(p) < c.STORE_QUIET_MIN * 60]
        if not recent:
            break
        if i == tries - 1:
            refuse('P-G13: %d fit stores were still written within %d min after %d min of waiting' % (
                len(recent), c.STORE_QUIET_MIN, tries * wait // 60))
        time.sleep(wait)
    unm = {}
    for p in prod_stores + [c.BATCH1_STORE]:
        n = unmarked_wideband_rows(p)
        if n:
            unm[p] = n
    s.u.meta['unmarked_wideband_rows'] = unm
    s.u.check('P-G13.unmarked-wideband-rows', not unm, value=sum(unm.values()), detail='; '.join(
        '%s %d' % (os.path.basename(k), v) for k, v in list(unm.items())[:5]))
    if unm:
        refuse('%d double-registered wideband rows without the identity-screen mark in %d stores (e.g. %s): run `exopipe3 '
               'archive quarantine-rows` first (catalog never moves store rows)' % (
                   sum(unm.values()), len(unm), ', '.join(list(unm)[:3])))
    for p in prod_stores + [c.BATCH1_STORE]:
        s.inp(p)
    if not ctx.prod:
        os.makedirs(ctx.fits, exist_ok=True)
        copied = {}
        for p in prod_stores:
            for q in (p, template_of(p)):
                if os.path.exists(q):
                    shutil.copy2(q, os.path.join(ctx.fits, os.path.basename(q)))
                    copied[os.path.basename(q)] = T.sha1_file(q)
        s.u.meta['store_copies_sha1'] = copied
    stores = _store_list(ctx, ctx.fits)
    rep = os.path.join(ctx.L, 'repair')
    if os.path.lexists(rep):
        shutil.rmtree(rep)
    os.makedirs(rep)
    from concurrent.futures import ThreadPoolExecutor

    def one(cell):
        a = [cell, os.path.join(rep, '%s.csv' % cell)]
        if not ctx.prod:
            a += ['--store', os.path.join(ctx.fits, 'fits_%s.csv' % cell)]
        r = T.call(ctx, s, ctx.opsc('snapshot_store_repair.py'), a, 'py36', log=os.path.join(rep, '%s.log' % cell),
                   cwd=ctx.L)
        return cell, r
    with ThreadPoolExecutor(c.LIST_PARALLEL) as ex:
        results = list(ex.map(one, [os.path.basename(p)[5:-4] for p in stores]))
    failed = [cl for cl, r in results if not r.ok]
    if failed:
        refuse('P-G14: repair list failed for %s (%s/<cell>.log)' % (' '.join(failed[:3]), rep))
    nrep = union_csv(sorted(glob.glob(os.path.join(rep, '*.csv'))), os.path.join(ctx.L, 'repair.csv'), REPAIR_COLS)
    ctx.log.say('store repair: %d rows the store content test refuses (%d stores listed)' % (nrep, len(results)))
    s.u.meta['n_repair_rows'] = nrep
    if nrep > 0:
        if not ctx.prod:
            refuse('P-G14: %d repair rows: the store content test (snapshot_store_ready.py) reads only the production store '
                   'directory %s, so a non-production run cannot test its repaired copies - refused until the test is '
                   'absorbed (or run in production)' % (nrep, ctx.fits_prod))
        fit_round(ctx, s, 'repair', os.path.join(ctx.L, 'repair.csv'), plan)
        stores = _store_list(ctx, ctx.fits)
    try:
        nrows, nt = combine_stores(c.BATCH1_STORE, stores, ctx.S)
    except ValueError as e:
        refuse('P-G14: %s' % e)
    nb = unmarked_wideband_rows(ctx.S)
    if nb:
        refuse('P-G14: %d wideband rows without the identity-screen mark remain in %s' % (nb, ctx.S))
    if not ctx.prod:
        changed = [p for p in prod_stores if T.sha1_file(p) != s.u.meta['store_copies_sha1'].get(os.path.basename(p))]
        if changed:
            refuse('P-G14: production stores changed while copied (%s): the store test would read other rows' % changed[:3])
    res = T.call(ctx, s, ctx.opsc('snapshot_store_ready.py'), ctx.cells, 'py36', log=os.path.join(ctx.L, 'store_ready.log'),
                 cwd=ctx.L)
    lines = [ln for ln in open(os.path.join(ctx.L, 'store_ready.log')).read().splitlines()
             if re.match(r'^(READY|NOT-READY) ', ln)][-len(ctx.cells):]
    nready = sum(1 for ln in lines if ln.startswith('READY '))
    notready = [ln for ln in lines if ln.startswith('NOT-READY ')]
    if len(lines) != len(ctx.cells):
        refuse('P-G14: store content test reported %d of %d cells' % (len(lines), len(ctx.cells)))
    if notready:
        for ln in notready:
            ctx.log.say('STORE %s' % ln)
        refuse('P-G14: store content test: %d cells NOT-READY after the repair (every line in the log)' % len(notready))
    s.u.check('P-G14.store-ready', True, value=nready, detail='snapshot_store_ready READY for %d cells' % nready)
    ctx.log.say('store: %d rows, template %d; 0 double-registered wideband rows; snapshot_store_ready READY for %d cells' % (
        nrows, nt, nready))
    s.out(ctx.S, 'csv_table')
    s.out(os.path.join(ctx.L, 'repair.csv'), 'csv_table')


def st_prefit(ctx, s, plan):
    T, c = ctx.T, ctx.c
    maxit = int(arg_val(ctx.args, 'snap_prefit_iterations') or c.PREFIT_MAX_ITER)
    if maxit > c.PREFIT_MAX_ITER and not ctx.run.waived('snapshot.prefit-iterations'):
        refuse('--snap-prefit-iterations %d > catalog_snapshot.PREFIT_MAX_ITER %d needs --waive snapshot.prefit-iterations '
               '--ruling (Gregg 2026-10-07 Q6)' % (maxit, c.PREFIT_MAX_ITER))
    plan.append('prefit loop (<= %d rounds): per cell %s %s <CHAINARGS> --cells <cell> --out %s/prefit<i>/<cell> '
                '--veto-record-root "" --prefit-list-only (%d at a time); union -> %s/prefit<i>.csv; fit round; append' % (
                    maxit, ctx.py36, ctx.chain_py, ctx.W, c.LIST_PARALLEL, ctx.L))
    if ctx.dry:
        fit_round(ctx, s, 'prefit1', os.path.join(ctx.L, 'prefit1.csv'), plan)
        return
    if not os.path.exists(ctx.S):
        refuse('prefit: no combined store %s (run the store stage)' % ctx.S)
    from concurrent.futures import ThreadPoolExecutor
    ok, it = False, 0
    ndet = nload = 0
    per = {}
    for it in range(1, maxit + 1):
        lim_check(ctx, 'prefit%d' % it)
        d = os.path.join(ctx.W, 'prefit%d' % it)
        if os.path.lexists(d):
            shutil.move(d, os.path.join(ctx.recdir, 'superseded', 'prefit%d_%s' % (it, T.utc('%Y%m%dT%H%M%SZ'))))
        os.makedirs(d)

        def one(cell):
            a = ctx.chainargs() + ['--cells', cell, '--out', os.path.join(d, cell), '--veto-record-root', '',
                                   '--prefit-list-only']
            return cell, T.call(ctx, s, ctx.chain_py, a, 'py36', env=ctx.chain_env(), log=os.path.join(d, '%s.log' % cell),
                                cwd=ctx.W)
        with ThreadPoolExecutor(c.LIST_PARALLEL) as ex:
            res = list(ex.map(one, ctx.cells))
        failed = [cl for cl, r in res if not r.ok]
        if failed:
            refuse('P-G15: prefit list %d failed for %s' % (it, ' '.join(failed[:3])))
        n = union_csv(sorted(glob.glob(os.path.join(d, '*', 'prefit_requests.csv'))), os.path.join(ctx.L, 'prefit%d.csv' % it))
        ndet, nload, per = prefit_counts(d)
        ctx.log.say('prefit %d: %d of %d detections at the per-frame fit step without their own fit' % (it, n, ndet))
        if n == 0:
            ok = True
            break
        fit_round(ctx, s, 'prefit%d' % it, os.path.join(ctx.L, 'prefit%d.csv' % it), plan)
        ctx.log.say('prefit %d: %d store rows after the append' % (it, T.count_lines(ctx.S) - 1))
    s.u.meta.update(prefit_iterations=it, prefit_dir=os.path.join(ctx.W, 'prefit%d' % it), n_at_fit_step=ndet,
                    n_loaded=nload, per_cell_fitstep=per)
    if not ok:
        refuse('P-G15: detections still without their own fit after %d prefit iterations (%s/prefit%d.csv)' % (it, ctx.L, it))
    s.out(os.path.join(ctx.L, 'prefit%d.csv' % it), 'csv_table')


def ensure_tiecache_copy(ctx, s):
    """Non-production: the tie cache is a COPY of the run's cells' rows (tie_<cell>.csv), taken once, so the tie pass and the
    chain write only under --out."""
    if ctx.prod or os.path.isdir(ctx.tiecache):
        return
    os.makedirs(ctx.tiecache)
    got = {}
    for cell in ctx.cells:
        p = os.path.join(ctx.tiecache_prod, 'tie_%s.csv' % cell)
        if os.path.exists(p):
            shutil.copy2(p, ctx.tiecache)
            got[os.path.basename(p)] = ctx.T.sha1_file(p)
    s.u.meta['tiecache_copy'] = got


def st_tie(ctx, s, plan):
    item = arg_val(ctx.args, 'snap_item')
    if item:
        return item_tie(ctx, s, item, plan)
    T = ctx.T
    pr = T.stage_record(ctx.RUN, '%s.prefit' % ctx.name) or {}
    per = (pr.get('meta') or {}).get('per_cell_fitstep') or {}
    order = sorted(ctx.cells, key=lambda x: -per.get(x, 0))
    plan.append('tie pass (model %s): %d cells, largest first; shards %s' % (
        (T.stage_record(ctx.RUN, '%s.preflight' % ctx.name) or {}).get('meta', {}).get('tie_model', '?'), len(order),
        ' '.join('%s:%d' % (x, tie_shards(per.get(x, 0), ctx.c.TIE_SHARDS)) for x in order[:6])
        + (' ...' if len(order) > 6 else '')))
    if not ctx.dry:
        if not per:
            refuse('tie: no per-cell counts in the prefit record of %s (run the prefit stage)' % ctx.name)
        ensure_tiecache_copy(ctx, s)
        os.makedirs(os.path.join(ctx.L, 'tie'), exist_ok=True)
    submit_and_wait(ctx, s, 'tie', 'tie', order, ctx.c.TIE_CELLS_PER_NODE, plan)
    if ctx.dry:
        return
    a = b = 0
    for cell in order:
        ln = T.last_line(open(os.path.join(ctx.L, 'tie', '%s.log' % cell), errors='replace').read(), 'tie cache:')
        m = re.match(r'.*tie cache: (\d+) detections from .*, (\d+) tied now', ln)
        if m:
            a += int(m.group(1))
            b += int(m.group(2))
    ctx.log.say('tie pass: %d cached, %d tied now' % (a, b))
    s.u.meta.update(tie_cached=a, tie_tied_now=b)
    for cell in order:
        tc = os.path.join(ctx.tiecache, 'tie_%s.csv' % cell)
        if os.path.exists(tc):
            s.out(tc, 'tie_cache', errmodel=(T.stage_record(ctx.RUN, '%s.preflight' % ctx.name) or {}).get('meta', {})
                  .get('tie_model'))


def st_chain(ctx, s, plan):
    T, c = ctx.T, ctx.c
    pr = T.stage_record(ctx.RUN, '%s.prefit' % ctx.name) or {}
    meta = pr.get('meta') or {}
    nload, ndet = meta.get('n_loaded', 0), meta.get('n_at_fit_step', 0)
    need = arg_val(ctx.args, 'snap_chain_need_gb') or memory_need(c.MEM_MODEL, nload, ndet, len(ctx.cells), c.LC_CACHE_MAX)
    env = ctx.chain_env(P3_TIE_CACHE_DIR=ctx.tiecache, P3_REFASTRO_SHARDS=c.CHAIN_REFASTRO_SHARDS,
                        P3_VETO_PROCS=c.VETO_PROCS, P3_LC_CACHE_MAX=c.LC_CACHE_MAX)
    a = ctx.chainargs() + ['--cells'] + ctx.cells + ['--out', ctx.RUN, '--veto-record-root',
                                                     c.VETO_RECORD_ROOT.format(run=ctx.RUN)]
    plan.append('memory model: %d loaded, %d at the fit step, %d cells -> %s GB (max %d)' % (
        nload, ndet, len(ctx.cells), need, c.MEM_MODEL['max_gb']))
    plan.append('chain: %s %s %s   env %s > %s/chain.log' % (ctx.py36, ctx.chain_py, ' '.join(a[:12]) + ' ...',
                                                            ' '.join('%s=%s' % kv for kv in sorted(env.items())), ctx.W))
    if ctx.dry:
        return
    if not pr or pr.get('status') != 'done':
        refuse('chain: the prefit record of %s is not done (memory model and fitted detections come from it)' % ctx.name)
    if int(need) > c.MEM_MODEL['max_gb']:
        refuse('P-G17: the chain needs about %s GB (%d detections loaded, %d at the fit step, %d cells) - more than a node\'s '
               '400 GiB /user.slice can give' % (need, nload, ndet, len(ctx.cells)))
    ctx.log.say('memory model: %d detections loaded, %d at the per-frame fit step, %d cells -> the chain host must have %s GB '
                'free' % (nload, ndet, len(ctx.cells), need))
    ok, line = T.mem_gate(ctx.ops, need, c.MEM_WAIT_S, 'rechain-chain-%s' % ctx.name, ctx.log.path)
    if not ok:
        refuse('P-G17 memory gate: %s' % (line or 'node_mem_gate.sh refused'))
    ensure_tiecache_copy(ctx, s)
    # `rm -rf $RUN` (snapshot_rechain_batch.sh:172): the previous pass's files are moved aside, never deleted (D10)
    keep = ('_records', '_side')
    old = [e for e in os.listdir(ctx.RUN) if e not in keep] if os.path.isdir(ctx.RUN) else []
    if old:
        dst = os.path.join(ctx.recdir, 'superseded', 'run_%s' % T.utc('%Y%m%dT%H%M%SZ'))
        os.makedirs(dst)
        for e in old:
            shutil.move(os.path.join(ctx.RUN, e), os.path.join(dst, e))
        s.u.note('previous pass files (%d entries) moved to %s' % (len(old), dst))
    os.makedirs(ctx.W, exist_ok=True)
    s.code(ctx.chain_py)
    s.inp(ctx.S, ctx.cells_file)
    import resource
    res = T.call(ctx, s, ctx.chain_py, a, 'py36', env=env, log=os.path.join(ctx.W, 'chain.log'), cwd=ctx.W)
    peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e6
    if not res.ok:
        tail = [ln for ln in (res.tail or '').splitlines() if 'warn' not in ln.lower() and ln.strip() and not ln[0].isspace()]
        refuse('P-G18: chain rc=%s (%s)' % (res.rc, (tail[-1] if tail else '')[:160]))
    summ = json.load(open(os.path.join(ctx.RUN, 'chain_summary.json')))
    ctx.log.say('chain: served %s prefit %s ghost %s checksources_missing %s; tied in the multi-cell chain: %s; peak memory '
                '%.1f GB (model %s GB)' % (summ.get('n_served'), summ.get('prefit'), (summ.get('ghost_gate_removed') or {}).get('n'),
                                           summ.get('checksources_missing'), (summ.get('tie_cache') or {}).get('tied_now'),
                                           peak, need))
    s.u.meta.update(peak_rss_gb=round(peak, 1), mem_need_gb=need)
    s.out(ctx.RUN, 'snapshot_run')


def st_test(ctx, s, plan):
    T = ctx.T
    t = ctx.opsc('snapshot_rechain_content_test.py')
    plan.append('run content test: %s %s %s >> %s (R1-R5, R7)' % (ctx.py36, t, ctx.RUN, ctx.log.path))
    if ctx.dry:
        return
    s.code(t)
    res = T.call(ctx, s, t, [ctx.RUN], 'py36', log=ctx.log.path, cwd=ctx.RUN)
    line = T.last_line(res.tail, r'^B2R-RUN-CONTENT')
    s.u.check('P-G19.run-content-test', res.ok, value=res.rc, detail=line)
    if not res.ok:
        refuse('P-G19: run content test failed (%s)' % (line or res.reason))
    ctx.log.say('run content test: %s' % line)
    s.out(ctx.RUN, 'snapshot_run')


def st_fit_shard(ctx, s, plan):
    item = arg_val(ctx.args, 'snap_item')
    if not item or ':' not in item:
        raise SystemExit('error: --stages fit-shard is a runner item: give --snap-item <tag>:<part>')
    item_fit_shard(ctx, s, item, plan)


FUNCS = {'preflight': st_preflight, 'store': st_store, 'prefit': st_prefit, 'tie': st_tie, 'chain': st_chain,
         'test': st_test, 'fit-shard': st_fit_shard}


def lim_check(ctx, what):
    """P-G20: the runaway guard before a stage starts (LIMIT_HOURS of the marker, else catalog_snapshot.LIMIT_H)."""
    e = time.time() - ctx.t0
    lim = ctx.limit_h()
    if e >= lim * 3600:
        refuse('P-G20: %d-hour limit reached before %s (%d h elapsed) - waits for Gregg' % (lim, what, e // 3600))


# =====================================================================================================================
# interface for steps/catalog.py
# =====================================================================================================================

def add_arguments(p):
    """Tier-unique flags (prefixed --snap-)."""
    p.add_argument('--snap-approval', metavar='TEXT',
                   help="Gregg's words, UTC and ledger ID approving this run (P-G2); default: the production approval marker "
                        "catalog_snapshot.APPROVAL_MARKER with CELLS_FILE_SHA256 = the cells file now")
    p.add_argument('--snap-chain-nodes', metavar='NODES',
                   help='nodes for the tie items and fit rounds (core/runner spec host:lanes or a site pool; default the '
                        'site pool snapshot)')
    p.add_argument('--snap-prefit-iterations', type=int, default=None,
                   help='prefit loop bound (default catalog_snapshot.PREFIT_MAX_ITER = 3; more needs --waive '
                        'snapshot.prefit-iterations --ruling)')
    p.add_argument('--snap-served-run', metavar='DIR',
                   help='the served run whose cells must be kept (P-G7); default: the --run of the server on '
                        'catalog_snapshot.SERVED_PORT_CHECK')
    p.add_argument('--snap-chain-need-gb', type=int, default=None,
                   help='memory the chain host must have free (default: the memory model, catalog_snapshot.MEM_MODEL)')
    p.add_argument('--snap-analysis', metavar='LIST',
                   help='ANALYSIS ONLY chain switches (%s; rotate-parents=DEG): refused for a production --out, record stamped '
                        'analysis, never served (B-83)' % ', '.join(ANALYSIS))
    p.add_argument('--snap-item', metavar='ITEM', help='runner item of the internal stages (tie: a cell; fit-shard: '
                                                       '<tag>:<part>) - set by the stage that submits the job')


def run(args, run):
    T = _T()
    sel = T.select_stages(args, STAGES, internal=INTERNAL, label='snapshot')
    item = arg_val(args, 'snap_item')
    if item and sel not in (['tie'], ['fit-shard']):
        raise SystemExit('error: --snap-item goes with --stages tie or --stages fit-shard only')
    ctx = Ctx(args, run)
    first = STAGES.index(sel[0]) if sel[0] in STAGES else None
    where = 'production' if ctx.prod else 'non-production; side files under ' + os.path.join(ctx.RUN, '_side')
    print('catalog --tier snapshot %s: RUN=%s (%s); %d cells; code %s, %s; stages %s%s' % (
        ctx.name, ctx.RUN, where, len(ctx.cells), ctx.ops, ctx.clv, ','.join(sel), (' item %s' % item) if item else ''),
        file=sys.stderr)
    if ctx.dry:
        for name in sel:
            plan = []
            FUNCS[name](ctx, T.Stage(ctx, name, T.DryUnit(), tier=TIER, in_place=()), plan)
            print('[%s.%s%s]' % (ctx.name, name, ('.' + item) if item else ''))
            for ln in plan:
                print('  ' + ln)
        return 0
    T.env_leaks(LEAKY_ENV)
    restored = []
    if first and not item:
        restored = T.check_from(ctx.RUN, ctx.name, STAGES, first, ctx.recdir)
    finished = False
    try:
        for name in sel:
            unit = '%s.%s' % (ctx.name, name) if not item else '%s.%s.%s' % (
                ctx.name, 'tie' if name == 'tie' else 'fit', item.replace(':', '.'))
            with run.unit(unit, tier=TIER, run=ctx.name, stage=name, item=item,
                          analysis=bool(ctx.analysis)) as u:
                st = T.Stage(ctx, name, u, tier=TIER, in_place=())
                for n_ in restored:
                    u.note(n_)
                restored = []
                plan = []
                try:
                    if not item:
                        lim_check(ctx, name)
                    FUNCS[name](ctx, st, plan)
                finally:
                    u.meta['plan'] = plan
                    if ctx.analysis:
                        u.meta['analysis'] = ctx.analysis
                    st.close()
            if u.status != 'done':
                if not item:
                    ctx.log.say('REFUSING: %s %s: %s' % (name, u.status, u.reason))
                break
        else:
            finished = 'test' in sel
    finally:
        if ctx.active and not item and (finished or any(x.status != 'done' for x in run.units)):
            try:
                os.remove(ctx.active)
            except OSError:
                pass
    return None


if __name__ == '__main__':          # child entry points through core/legacy.run (Python 3.6 + clv15 only)
    if len(sys.argv) >= 3 and sys.argv[1] == 'errmodel':
        sys.exit(child_errmodel(sys.argv[2]))
    if len(sys.argv) >= 6 and sys.argv[1] == 'fitsplit':
        sys.exit(child_fitsplit(*sys.argv[2:6]))
    sys.exit('usage: tier_snapshot.py errmodel CLV | fitsplit CLV PENDING OUTDIR NPARTS')
