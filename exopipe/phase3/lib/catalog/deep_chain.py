"""The deep catalogue chain of `exopipe3 catalog --tier deep` (full mode; spec plan/specs/catalog.md).

Stages, in order (CFG.catalog.DEEP_STAGES): preflight (per-pass parent-catalogue snapshot), harvest, merge (merge gate +
staleness gate), conform, fitpos (fitted detection positions), select (selection, ephemeris filter, split), associate
(one unit per catalogue key: association, identity merge, E statistic, promotion, unique J-names), fork (multiples, one
unit per Stokes), union, identity, localize, list (the deep transient list). `exopipe3 forced` runs between fork and
union (it writes forced_* columns into the tier survivors in place), so a plain run stops after the fork
(CFG.catalog.DEEP_DEFAULT_UNTIL) and `--from union` finishes the pass.

How it differs from ops/p3s3_pass.sh + ops/fork_serve_p3s5_20261008.sh (which it replaces; their shell logic is ported
here, the modules are called IN PLACE unchanged, DESIGN 11.1):
- every stage is a unit with a run record; a stage reads only the files its predecessors' records list and refuses
  unless those records are current (lib/catalog/deep_records.py) - no glob decides an input (D9); a stage whose
  in-place module globs is preceded by a check that the glob would read exactly the recorded files;
- every stage writes a fresh directory: its stage directory <out>/_records/catalog/<label>/stage_<name>/ (stage.json,
  copies of the files a later stage rewrites in place) and, where the stage builds a product directory wholesale
  (source_catalogs_<K>, fork_<S>, the union files, the localization tables), the previous pass's copy is moved aside to
  <out>/_records/catalog/<label>/_superseded/<run id>/ first, so no previous-pass file survives a stage (D10);
- the chain stops at the first stage that does not end done (exit 2 refused, 1 error); exit codes, never log lines;
- outside the production roots the side effects are redirected: the staleness gate edits a COPY of the search state,
  the parent snapshot goes under --out, the harvest (which writes the production collect root) refuses;
- parent catalogues are built ONLY by `exopipe3 masters --tier deep --stages sources` (refactor decision 2026-10-13):
  preflight assembles the per-pass snapshot of the current ones (ops/p3s5_parent_union.py, read-only on its sources)
  and refuses, naming that command, when an installed cell has none (fork_serve_p3s5_20261008.sh ran ops/p3s5_parents.py);
- every in-place module runs through core.legacy.run (site interpreter catalog.DEEP_PYTHON, inherited P3_* cleared, parity
  guard in the child, child code fingerprint in the unit record) with the stage's environment set explicitly.
"""
import csv
import fnmatch
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections import deque

from exopipe.phase3.core import legacy, record
from exopipe.phase3.core.record import NoInput, Refusal
from exopipe.phase3.lib.catalog import deep_binding, deep_ported, deep_records

ENV_PREFIXES = ('P3S', 'FORK_', 'BDSF_')   # module switches blanked in the child unless the stage sets them (core.legacy.run
                                           # also clears runner.CHILD_ENV_CLEAR_PREFIXES; BDSF_ is not among those)
CACHES = ('frame_times.csv', 'frame_beam_ratio.csv', 'frame_beam_bmaj.csv', 'fit_positions_cache.csv', 'stale_attempts.csv')
DEEP_SECTIONS = ('catalog', 'catalog_merge', 'catalog_conform', 'catalog_fitpos', 'catalog_select', 'catalog_associate',
                 'catalog_fork', 'catalog_identity', 'catalog_localize', 'astrometry', 'fitting')
# The keys of the catalog sections the deep chain USES: its own orchestration and the values it passes to the in-place
# modules (environment / arguments). Every other key of these sections documents a literal inside an in-place module (or a
# value of another tier): an override of it (or of its 'deep'/'fork' entry) would change nothing in the deep chain while the
# record claimed it did, so the first unit refuses it (spec 1.3 item 5). Overrides of 'sublst'/'snapshot' entries are left
# to the tier modules.
LIVE_KEYS = {
    'catalog': ('DEEP_RUN_LABEL', 'DEEP_LABEL_PREFIX', 'DEEP_FORK_SKIP_PREFIX', 'DEEP_STATE_DIR', 'DEEP_PARENT_SNAPSHOT',
                'DEEP_PARENT_TEST_SAMPLE', 'INSTALL_LOCK', 'DEEP_STAGES', 'DEEP_DEFAULT_UNTIL', 'DEEP_ASSOC_KEYS',
                'DEEP_ASSOC_WORKERS', 'DEEP_LOG', 'DEEP_PYTHON', 'PARENTS_COMMAND', 'BINDING_CHECK', 'FORCED_STEP'),
    'catalog_merge': ('HARVEST_STOKES', 'HARVEST_SSH_SLEEP_S', 'HARVEST_DEST_ROOT', 'UNHARVESTED', 'TEST_BACKING_KEYS'),
    'catalog_conform': ('SIZE_STEP', 'TABLE_ELEV'),
    'catalog_fitpos': ('NPROC', 'ZERO_STAT_ERROR'),       # ZERO_STAT_ERROR: read by the fitted_catalog content test of the fitpos stage
    'catalog_select': ('PARENT_EXCLUSION', 'VERDICT_INPUT'),
    'catalog_associate': (),
    'catalog_fork': ('STOKES', 'NPROC', 'N_RANDOM', 'TABLES_SOURCE_DIR'),
    'catalog_identity': ('UNION_TIERS', 'UNION_KINDS'),
    'catalog_localize': ('LIST_NAME',),
}
OTHER_TIER_ENTRIES = ('sublst', 'snapshot')


def dead_overrides(cfg):
    """Overrides of keys of the catalog sections the deep chain does not use (see LIVE_KEYS)."""
    from exopipe.phase3.config import _parse_key
    out = []
    for o in cfg.overrides:
        sec, key, path = _parse_key(o[0])
        if sec not in LIVE_KEYS or key in LIVE_KEYS[sec]:
            continue
        if path and path[0] and path[0][0] in OTHER_TIER_ENTRIES:
            continue
        out.append(o[0])
    return out


# in-place scripts of each stage (their sha1 goes into the record: code identity for --resume, DESIGN 11.5)
SCRIPTS = {
    'preflight': ('p3s5_parent_union.py',),
    'harvest': ('p3s3_sources.py',),
    'merge': ('p3s3_merge.py', 'p3s3_sources.py', 'p3s3_stale_exclusions.py'),
    'conform': ('p3s2_conform.py', 'gate_inputs.py', 'p3s3_astrom.py', 'source_localization.py'),
    'fitpos': ('deep_fit_positions.py', 'deep_forced_radio.py'),
    'select': ('select_v3.py', 'p3s2_ephem_filter.py'),
    'associate': ('build_source_catalogs_v3.py', 'p3s3_astrom.py', 'select_v3.py'),
    'fork': ('multiples_fork.py', 'p3s2_conform.py', 'gate_inputs.py', 'select_v3.py', 'p3s2_ephem_filter.py',
             'build_source_catalogs_v3.py', 'p3s3_repeat_shuffle.py'),
    'union': (),
    'identity': ('p3s3_identity.py',),
    'localize': ('build_localizations.py', 'source_localization.py'),
    'list': ('p3s2_transient_list.py', 'gate_inputs.py'),
}


def utcnow():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def _sha1(path):
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


def ops_root(cfg):
    """The exopipe_ops root of the in-place modules: the site's legacy code root (core.env.legacy_root)."""
    from exopipe.phase3.core import env as core_env
    try:
        return core_env.legacy_root('exopipe_ops').rstrip('/')
    except KeyError as e:
        raise Refusal(f'{e} (EXOPIPE3_SITE): the deep chain calls the exopipe_ops modules in place')


def python_for(cfg):
    """The interpreter of the in-place calls: the site PYTHON key catalog.DEEP_PYTHON (refuses when absent)."""
    return legacy.interpreter(cfg.catalog.DEEP_PYTHON)


def _rows_n(path):
    with open(path, newline='') as f:
        return max(0, sum(1 for _ in csv.reader(f)) - 1)


class DeepRun:
    """One invocation of the deep chain over a selected stage range."""

    def __init__(self, args, run, stages):
        from exopipe.phase3.config import CFG
        self.cfg = CFG
        c = CFG.catalog
        self.args, self.run = args, run
        self.stages = list(stages)
        self.first_stage = self.stages[0]
        self.label = args.run_label
        self.low = self.label.lower()
        self.U = os.path.abspath(run.out_root)
        self.prod = CFG.production_root_of(self.U) is not None
        self.dry = run.dry_run
        from exopipe.phase3.core import env as core_env
        self.host = core_env.host()          # short host name (the harvest compares it with the published node names)
        self.label_dir = os.path.join(self.U, '_records', 'catalog', self.label)
        self.sup_root = os.path.join(self.label_dir, '_superseded', run.run_id)
        self.log = os.path.join(self.U, c.DEEP_LOG)
        st = [s.strip() for s in (args.stokes or 'V,I').split(',') if s.strip()]
        bad = [s for s in st if s not in ('V', 'I')]
        if bad:
            raise ValueError(f'--stokes {args.stokes}: V and/or I')
        self.stokes = tuple(s for s in ('V', 'I') if s in st)
        self.partial_stokes = self.stokes != ('V', 'I')
        self.deep_label = self.label.startswith(c.DEEP_LABEL_PREFIX)
        self.fork_on = not self.label.startswith(c.DEEP_FORK_SKIP_PREFIX)
        self.keys = [K for K in c.DEEP_ASSOC_KEYS if K[0] in self.stokes]
        self.fork_stokes = [S for S in CFG.catalog_fork.STOKES if S in self.stokes]
        self.ts = time.strftime('%Y%m%dT%H%M', time.gmtime())
        self.lock_fd = None
        self.checked_first = False
        self.binding = None
        self._cell_filter()
        self._build_graph()

    # ---- graph ------------------------------------------------------------------------------------------------------
    def u(self, stage, key=None):
        return f'{self.label}.{stage}' + (f'.{key}' if key else '')

    def _build_graph(self):
        c = self.cfg
        u = self.u
        keys_all = list(c.catalog.DEEP_ASSOC_KEYS)
        self.sel_pred = u('fitpos') if self.deep_label else u('conform')
        forks = [u('fork', S) for S in c.catalog_fork.STOKES] if self.fork_on else []
        self.preds = {u('preflight'): [], u('harvest'): [], u('merge'): [u('harvest')],
                      u('conform'): [u('merge'), u('preflight')], u('fitpos'): [u('conform')],
                      u('select'): [self.sel_pred],
                      u('union'): [u('associate', K) for K in c.catalog_identity.UNION_TIERS],
                      u('identity'): [u('union')] + [u('associate', K) for K in keys_all] + forks,
                      u('localize'): [u('identity')],
                      u('list'): [u('localize'), u('identity'), self.sel_pred]}
        for K in keys_all:
            self.preds[u('associate', K)] = [u('select'), u('merge')]
        for S in c.catalog_fork.STOKES:
            self.preds[u('fork', S)] = [u('conform'), u('merge'), u('preflight')]
        self.units = list(self.preds)
        self.chain = deep_records.Chain(self.U, self.label, self.units, self.preds,
                                        may_be_noinput=[u('associate', K) for K in keys_all]
                                        + [u('fork', S) for S in c.catalog_fork.STOKES] + [u('fitpos')],
                                        forced_step=c.catalog.FORCED_STEP)

    def _cell_filter(self):
        """--cells / --exclude-cells / --lsts on the deep tier: a TEST filter of the published lists the merge reads
        (spec 1.1; the rule of ops/multiples_fork.py:61-67 FORK_TEST_CELLS), only with --out outside every production root."""
        from exopipe.phase3.core import cells as C
        a = self.args
        self.cells = set(C.expand_cells(a.cells)) if getattr(a, 'cells', None) else None
        self.xcells = set(C.expand_cells(a.exclude_cells)) if getattr(a, 'exclude_cells', None) else set()
        self.lsts = set(C.expand(a.lsts, universe=C.LSTS, kind='LST')) if getattr(a, 'lsts', None) else None
        self.filtered = bool(self.cells or self.xcells or self.lsts)

    def keep_cell(self, cell):
        if self.cells is not None and cell not in self.cells:
            return False
        if cell in self.xcells:
            return False
        if self.lsts is not None and cell.split('_')[0] not in self.lsts:
            return False
        return True

    # ---- paths and environment ----------------------------------------------------------------------------------------
    @property
    def ops(self):
        return ops_root(self.cfg)

    @property
    def py(self):
        return python_for(self.cfg)

    def script(self, name):
        return os.path.join(self.ops, name)

    def prod_state_dir(self):
        return self.cfg.catalog.DEEP_STATE_DIR.format(ops_root=self.cfg.paths.OPS_ROOT, label_lower=self.low)

    def state_dir(self):
        if self.prod:
            return self.prod_state_dir()
        return os.path.join(self.U, '_state', f'{self.low}_state')

    def parent_dir(self):
        return self.chain.meta(self.u('preflight'), 'parent_dir')

    def env(self, extra=None, state_dir=None, parent_dir=None):
        """(full environment for a non-legacy child (rsync), the stage's own values for core.legacy.run). The production
        pass exports, for every stage (ops/fork_serve_p3s5_20261008.sh:31-32): P3S_LABEL P3_CADENCE=deep P3_TABLE_ELEV=1
        P3_PARENT_DIR FORK_NPROC (NUMPY_MADVISE_HUGEPAGE=0 comes from core.legacy.run). The values the modules would take
        from their own defaults are set explicitly (each equals the default the binding check verified): the collect
        roots and state dir (ops/p3s3_sources.py), the roots map (ops/p3s2_frames.py), the run dir (ops/p3s2_ephem_filter.py,
        ops/multiples_fork.py), the fork knobs. core.legacy.run clears inherited P3_*; any other inherited P3S*/FORK_*/
        BDSF_* variable is blanked here (every module reads those through `.get(X) or default`, or not at all)."""
        c = self.cfg
        own = {k: '' for k in os.environ if k.startswith(ENV_PREFIXES)}
        own.update({'P3S_LABEL': self.label, 'P3_CADENCE': 'deep',
                    'P3_TABLE_ELEV': '1' if c.catalog_conform.TABLE_ELEV['deep'] else '0',
                    'P3S3_COLLECT_ROOTS': ':'.join(c.catalog.COLLECT_ROOTS),
                    'P3S3_STATE_DIR': state_dir or self.state_dir(),
                    'P3S2_ROOTS_CSV': c.catalog.DEEP_ROOTS_CSV, 'P3S_RUN_DIR': self.U,
                    'FORK_RUN_DIR': self.U, 'FORK_OUT_DIR': self.U, 'FORK_TEST_CELLS': '',
                    'FORK_NPROC': str(c.catalog_fork.NPROC), 'FORK_N_RANDOM': str(c.catalog_fork.N_RANDOM),
                    'FORK_TABLES_DIR': c.catalog_fork.TABLES_SOURCE_DIR})
        par = parent_dir or (self.parent_dir() if hasattr(self, 'chain') else None)
        if par:
            own['P3_PARENT_DIR'] = par
        own.update(extra or {})
        full = {k: v for k, v in os.environ.items() if not k.startswith(ENV_PREFIXES + ('P3_',))}
        full.update({k: v for k, v in own.items() if v != ''})
        return full, own

    # ---- small helpers ----------------------------------------------------------------------------------------------
    def say(self, text):
        """A line in <run>/incremental.log, as the pass writes them (people read it; the record is the state)."""
        if self.dry:
            return
        os.makedirs(self.U, exist_ok=True)
        with open(self.log, 'a') as f:
            f.write(f'{time.strftime("%H:%M", time.gmtime())} {text}\n')

    def call(self, u, script, args, env, log, mode='a', cwd=None, keep=(), python=None):
        """Run one in-place exopipe_ops script through core.legacy.run (output appended to ``log``, or ``log`` started
        afresh with mode 'w' as the pass's '>' redirects); the command, the stage environment, exit status, last line and
        the output lines matching ``keep`` go into the unit record. Returns (rc, tail, hits)."""
        own = env[1]
        os.makedirs(os.path.dirname(log), exist_ok=True)
        if mode == 'w':
            open(log, 'w').close()
        off = os.path.getsize(log) if os.path.exists(log) else 0
        entry = {'script': script, 'args': [str(a) for a in args], 'env': own, 'cwd': cwd or self.U, 'log': log,
                 'started': utcnow()}
        res = legacy.run(self.script(script), list(args), python=python or self.cfg.catalog.DEEP_PYTHON, env=own,
                         cwd=cwd or self.U, log=log, unit=u)
        with open(log, 'rb') as f:
            f.seek(off)
            lines = [x for x in f.read().decode('utf-8', 'replace').splitlines() if not x.startswith('===== ')]
        hits = [x for x in lines if any(k in x for k in keep)]
        entry.update(rc=res.rc, finished=utcnow(), argv=res.argv, last=lines[-1] if lines else '', hits=hits[-40:])
        u.meta.setdefault('commands', []).append(entry)
        return res.rc, lines[-60:], hits

    def shell(self, u, argv, env, log):
        """A non-Python child (rsync of the harvest): plain subprocess, output appended to ``log``."""
        entry = {'argv': list(argv), 'env': env[1], 'log': log, 'started': utcnow()}
        os.makedirs(os.path.dirname(log), exist_ok=True)
        with open(log, 'a') as lf:
            rc = subprocess.run(list(argv), env=env[0], stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL).returncode
        entry.update(rc=rc, finished=utcnow())
        u.meta.setdefault('commands', []).append(entry)
        return rc

    def attempts(self, state, u=None, log=None):
        """ops/p3s3_sources.py <stokes...> run in place (core.legacy.run) with the stage environment: the published
        attempts, parsed from its own output (deep_ported.parse_sources_output). Without a unit (--dry-run) the output
        goes to a temporary file that is removed."""
        import tempfile
        c = self.cfg
        tmpd = None
        if log is None:
            tmpd = tempfile.mkdtemp(prefix='exopipe3_catalog_sources_')
            log = os.path.join(tmpd, 'sources.log')
        try:
            open(log, 'w').close()
            res = legacy.run(self.script('p3s3_sources.py'), list(c.catalog_merge.HARVEST_STOKES),
                             python=c.catalog.DEEP_PYTHON, env=self.env(state_dir=state)[1], cwd=self.U if
                             os.path.isdir(self.U) else None, log=log, unit=u)
            text = open(log, errors='replace').read()
        finally:
            if tmpd:
                shutil.rmtree(tmpd, ignore_errors=True)
        if res.rc != 0:
            raise Refusal(f'ops/p3s3_sources.py exit {res.rc}: {res.tail.splitlines()[-1:] if res.tail else ""}')
        return deep_ported.parse_sources_output(text, self.label, c.catalog.COLLECT_ROOTS)

    def move_aside(self, path, u=None):
        """Move a previous-pass product or stage dir out of the way (never deleted): <label>/_superseded/<run id>/<rel>."""
        if not os.path.lexists(path):
            return None
        rel = os.path.relpath(os.path.abspath(path), self.U)
        if rel.startswith('..'):
            raise Refusal(f'{path} is outside the run directory {self.U}: not moved')
        dst = os.path.join(self.sup_root, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.rename(path, dst)
        if u is not None:
            u.meta.setdefault('superseded', []).append([os.path.abspath(path), dst])
        return dst

    def stage_dir(self, u, stage, key=None):
        d = os.path.join(self.label_dir, f'stage_{stage}' + (f'_{key}' if key else ''))
        self.move_aside(d, u)
        os.makedirs(d)
        u.meta['stage_dir'] = d
        return d

    def write_stage_json(self, d, **kw):
        record.atomic_write_json(os.path.join(d, 'stage.json'),
                                 dict(kw, run_dir=self.U, label=self.label, run_id=self.run.run_id, written=utcnow()))

    def out(self, u, path, cls=None, cache=False, test=True, **ctx):
        """Record an output: its fingerprint (files only - a directory's mtime moves whenever a later stage renames a
        file inside it, so a directory output is identified by the files recorded under it) and, with ``cls``, the
        output with its content tests."""
        p = os.path.abspath(path)
        if not os.path.isdir(p):
            u.meta.setdefault('out_fp', {})[p] = deep_records.out_fingerprint(p)
        if cache:
            u.meta.setdefault('caches', []).append(p)
        if cls:
            u.add_output(p, cls, test=test, **ctx)
        return p

    def copy(self, u, src, d, name=None):
        dst = os.path.join(d, name or os.path.basename(src))
        tmp = f'{dst}.tmp.{os.getpid()}'
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
        u.meta.setdefault('copies', {})[os.path.basename(src)] = dst
        self.out(u, dst)
        return dst

    def scripts_sha1(self, stage):
        out = {}
        for n in SCRIPTS.get(stage, ()):
            p = self.script(n)
            out[n] = _sha1(p)[:12] if os.path.exists(p) else None
        return out

    # ---- first-unit gates (recorded in the first unit of the invocation) -----------------------------------------------
    def compute_binding(self):
        if self.binding is None:
            if self.cfg.catalog.BINDING_CHECK:
                res = deep_binding.check(self.cfg, self.ops)
            else:
                res = []
            self.binding = res
        return self.binding

    def first_unit_gates(self, u):
        res = self.compute_binding()
        u.meta['binding'] = {'checked': len(res), 'mismatch': [r['detail'] for r in res if not r['ok']],
                             'enabled': bool(self.cfg.catalog.BINDING_CHECK),
                             'bound_keys': len(deep_binding.bound_keys())}
        if self.checked_first:
            return
        self.checked_first = True
        bad = [r for r in res if not r['ok']]
        if bad:
            u.refuse('binding check: config does not match the in-place module(s): ' + '; '.join(r['detail'] for r in bad[:4])
                     + (f' (+{len(bad) - 4} more)' if len(bad) > 4 else ''))
        dead = dead_overrides(self.cfg)
        if dead:
            u.refuse(f'override of key(s) the deep chain does not use (they document literals inside the in-place modules '
                     f'or another tier): {dead} - the run would claim a value that was never used (spec 1.3 item 5)')
        ov = deep_binding.override_refusals(self.cfg)
        if ov:
            u.refuse(f'override of key(s) the in-place modules carry as literals: {ov} - an '
                     f'override would change nothing while the module carries its own value (spec 1.3 item 5); change '
                     f'the module (owning chat) or wait for absorption into lib/catalog')
        if self.filtered and self.prod:
            u.refuse('--cells/--exclude-cells/--lsts on the deep tier are a TEST filter: they need --out outside every '
                     'production root (a production pass never runs on a subset; ops/multiples_fork.py:61-67)')

    # ---- predecessor records ----------------------------------------------------------------------------------------
    def preds_current(self, u, unit):
        self.chain.reload()
        for pu in self.preds.get(unit, ()):
            ok, why = self.chain.check(pu)
            if not ok:
                u.refuse(f'predecessor record not current: {why} - re-run from stage {pu.split(".")[1]}')
            u.note(f'predecessor current: {why}')

    def ensure_master(self, u, want, stage):
        """<run>/master_catalog.csv must be the predecessor's recorded output ``want`` (its stage copy). When this stage
        starts the invocation, the copy is restored (the re-pass of ops/p3s3_pass.sh:32-36, P3S3_SKIP_MERGE=1, generalised
        to every stage after a rewrite); otherwise a difference means something outside the chain changed it: refused."""
        cur = os.path.join(self.U, 'master_catalog.csv')
        if os.path.exists(cur) and _sha1(cur) == _sha1(want):
            return
        if stage == self.first_stage:
            tmp = f'{cur}.tmp.{os.getpid()}'
            shutil.copy2(want, tmp)
            os.replace(tmp, cur)
            u.note(f'restored master_catalog.csv from the recorded predecessor output {want}')
            self.say(f'{stage}: restored master_catalog.csv from {want} (re-pass)')
            return
        u.refuse(f'{cur} is not the predecessor output recorded in {want} (changed outside this chain)')

    def skip_if_valid(self, unit, stage):
        """--resume: skip a unit whose record is current, written from a clean tree with this config hash and these
        in-place scripts, and whose outputs pass their content tests now."""
        if not self.run.resume or self.dry:
            return False
        self.chain.reload()
        ok, why = self.chain.check(unit)
        rec = self.chain.get(unit)
        if not ok or rec is None or rec.get('status') != 'done':
            return False
        if str(rec.get('commit') or '').endswith('-dirty') or rec.get('config_hash') != self.run.config_hash:
            return False
        if (rec.get('meta') or {}).get('scripts') != self.scripts_sha1(stage):
            return False
        from exopipe.phase3.core import validate
        for o in rec.get('outputs', []):
            if not os.path.exists(o['path']) or any(not r['passed'] for r in validate.run(o['class'], o['path'],
                                                                                          **(o.get('ctx') or {}))):
                return False
        x = self.run.unit(unit, stage=stage)
        x.status, x.reason = 'skipped-valid', why
        print(f'{unit}: skipped (record current; --resume)', file=sys.stderr)
        return True

    def new_unit(self, stage, key=None):
        name = self.u(stage, key)
        u = self.run.unit(name, stage=stage, key=key, tier='deep', label=self.label, run_dir=self.U)
        u.meta['scripts'] = self.scripts_sha1(stage)
        u.meta['ops_root'] = self.ops
        u.meta['production'] = self.prod
        return u

    # ---- install lease (ops/fork_serve_p3s5_20261008.sh:18-19, 38-39) ---------------------------------------------------
    def take_lock(self):
        if self.lock_fd is not None or self.dry:
            return
        if not self.prod:
            print(f'install lease not taken: run outside the production roots (a master installed meanwhile makes '
                  f'gate_inputs.parents refuse in conform/fork)', file=sys.stderr)
            return
        import fcntl
        p = self.cfg.catalog.INSTALL_LOCK.format(ops_root=self.cfg.paths.OPS_ROOT)
        self.lock_fd = os.open(p, os.O_RDWR | os.O_CREAT, 0o664)
        print(f'waiting for the install lock {p}', file=sys.stderr)
        fcntl.flock(self.lock_fd, fcntl.LOCK_EX)
        self.say('install lock held')

    def release_lock(self):
        if self.lock_fd is None:
            return
        import fcntl
        fcntl.flock(self.lock_fd, fcntl.LOCK_UN)
        os.close(self.lock_fd)
        self.lock_fd = None
        self.say('install lock released')

    # =================================================================================================================
    # stages
    # =================================================================================================================
    def st_preflight(self):
        c = self.cfg.catalog
        root = self.cfg.paths.COADD_SB_ROOT
        u = self.new_unit('preflight')
        with u:
            self.first_unit_gates(u)
            if not os.path.isdir(root):
                u.refuse(f'the deep masters {root} are not on this host ({self.host}); site NODE_LOCAL_ROOTS: '
                         f'{self.cfg.site.NODE_LOCAL_ROOTS} (blocker K5: run where the masters are)')
            d = self.stage_dir(u, 'preflight')
            given = self.args.deep_parents not in (None, '', 'snapshot')
            if given:
                par = os.path.abspath(self.args.deep_parents)
                if not os.path.isdir(par):
                    u.refuse(f'--parents {par}: no such directory')
            else:
                par = (c.DEEP_PARENT_SNAPSHOT.format(utc=self.ts) if self.prod
                       else os.path.join(self.U, '_parents', f'p3s5_parents_union_{self.ts}'))
                rc, tail, _ = self.call(u, 'p3s5_parent_union.py', [par], self.env(), f'{par}.union_1.log', mode='w')
                if rc != 0:
                    u.refuse(f'p3s5_parent_union.py exit {rc}: {tail[-1] if tail else ""}')
                self.say(tail[-1][:400] if tail else 'parent union: no output')
            why = {}
            cells, miss = deep_ported.parents_missing(par, root, why)
            u.meta['parents_missing'] = {x: why.get(x, '') for x in miss}
            nomaster = [x for x in miss if why.get(x) == 'no master']
            if nomaster:
                # production counts a cell directory without a master as missing and refuses the pass on it
                # (ops/p3s5_parent_union.py:19-20; fork_serve_p3s5_20261008.sh:22-27)
                u.refuse(f'D-G1: {len(nomaster)} cell director(ies) under {root} without a master '
                         f'(I_deep_Taper_R075_<band>.fits): {nomaster[:12]}{" ..." if len(nomaster) > 12 else ""} - '
                         f'install or remove them before the pass (the production pass refuses on them too)')
            if miss:
                # D-G1: parents are built ONLY by masters --stages sources (refactor decision 2026-10-13, group 8)
                u.refuse(f'D-G1: {len(miss)} installed cell(s) without a parent catalogue on the current master in {par}: '
                         f'{miss[:12]}{" ..." if len(miss) > 12 else ""} - build them with '
                         f'`{c.PARENTS_COMMAND.format(cells=",".join(miss))}` and re-run')
            man = os.path.join(d, 'parents_manifest.csv')
            n = deep_ported.parents_manifest(par, man)
            u.meta['parent_dir'] = par
            self.write_stage_json(d, parent_dir=par, cells=len(cells), catalogues=n, given=given)
            self.out(u, man)
            self.out(u, par, 'parent_snapshot', coadd_root=root)
            u.note(f'parent snapshot {par}: {n} catalogues for {len(cells)} installed cells')
        return u

    def st_harvest(self):
        c = self.cfg
        u = self.new_unit('harvest')
        with u:
            self.first_unit_gates(u)
            d = self.stage_dir(u, 'harvest')
            src_state = self.prod_state_dir()
            state = self.state_dir()
            if state != src_state:
                n = deep_ported.copy_state(src_state, state, self.keep_cell if self.filtered else None)
                u.note(f'search state copied {src_state} -> {state} ({n}); the staleness gate edits the copy only')
            elif self.filtered:
                u.refuse('a cell filter needs a state copy (run outside the production roots)')
            for st in ('V', 'I'):
                p = os.path.join(state, f'published_{st}.csv')
                if os.path.exists(p):
                    u.add_input(p)
            env = self.env(state_dir=state)
            slog = os.path.join(d, 'published_attempts.log')
            atts = self.attempts(state, u, slog)
            leftover = [a for a in atts if os.path.isdir(os.path.join(a['dir'], 'parent_catalogs'))]
            if leftover:
                u.refuse(f'{len(leftover)} published attempt(s) hold a parent_catalogs dir inside the attempt (the v3.4 layout '
                         f'the shell migrated with rm -rf, not ported: it contradicts "never deletes"), e.g. '
                         f'{leftover[0]["dir"]}')
            todo = [a for a in atts if not a['harvested']]
            if todo:
                if not self.prod:
                    u.refuse(f'D-G3: {len(todo)} published attempt(s) not harvested (e.g. {todo[0]["cell"]}/{todo[0]["stokes"]}); '
                             f'the harvest writes into the production collect root {c.catalog_merge.HARVEST_DEST_ROOT}')
                fails = []
                for a in todo:
                    dest, argv, extra, remote = deep_ported.harvest_commands(a, self.label, self.host,
                                                                            c.catalog_merge.HARVEST_DEST_ROOT,
                                                                            c.catalog.DEEP_STAGING_ATTEMPTS)
                    os.makedirs(dest, exist_ok=True)
                    rc = self.shell(u, argv, self.env(extra, state_dir=state), self.log)
                    if rc != 0:
                        fails.append(f"{a['cell']} {a['node']} {a['att']}")
                    if remote:
                        time.sleep(c.catalog_merge.HARVEST_SSH_SLEEP_S)
                self.say(f'harvest: {len(todo)} to pull, {len(todo) - len(fails)} pulled, {len(fails)} failed')
                if fails:
                    u.refuse(f'D-G3: harvest failed for {len(fails)} attempt(s): {fails[:6]}')
                atts = self.attempts(state, u, slog)
            if not atts:
                raise NoInput('no published attempt in the search state')
            man = os.path.join(d, 'harvest_manifest.csv')
            with open(man + '.tmp', 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=['cell', 'stokes', 'node', 'att', 'dir', 'harvested', 'label'])
                w.writeheader()
                for a in sorted(atts, key=lambda a: (a['stokes'], a['cell'])):
                    w.writerow(dict(cell=a['cell'], stokes=a['stokes'], node=a['node'], att=a['att'], dir=a['dir'],
                                    harvested='1' if a['harvested'] else '0', label=self.label))
            os.replace(man + '.tmp', man)
            u.meta['state_dir'] = state
            u.meta['harvest_manifest'] = man
            self.write_stage_json(d, state_dir=state, harvest_manifest=man, filtered=self.filtered,
                                  cells=sorted(self.cells) if self.cells else None)
            self.out(u, man, 'harvest_manifest')
        return u

    def st_merge(self):
        c = self.cfg
        u = self.new_unit('merge')
        with u:
            self.first_unit_gates(u)
            self.preds_current(u, u.name)
            hv = self.u('harvest')
            state, man = self.chain.meta(hv, 'state_dir'), self.chain.meta(hv, 'harvest_manifest')
            d = self.stage_dir(u, 'merge')
            u.add_input(man)
            for st in ('V', 'I'):
                p = os.path.join(state, f'published_{st}.csv')
                if os.path.exists(p):
                    u.add_input(p)
            u.add_input(c.paths.DATA_DECISIONS)
            env = self.env(state_dir=state)
            now = {(a['cell'], a['stokes'], a['node'], a['att'])
                   for a in self.attempts(state, u, os.path.join(d, 'published_attempts.log'))}
            with open(man, newline='') as f:
                rec = {(r['cell'], r['stokes'], r['node'], r['att']) for r in csv.DictReader(f)}
            if now != rec:
                u.refuse(f'the published attempts changed since the harvest ({len(now ^ rec)} differ, e.g. '
                         f'{sorted(now ^ rec)[:3]}): re-run from harvest')
            rc, tail, hits = self.call(u, 'p3s3_merge.py', [self.U], env, self.log,
                                       keep=('merge gate:', 'merge: WARNING'))
            if rc != 0:
                self.say('merge gate REFUSED - pass aborted')
                u.refuse(f'merge gate exit {rc}: {tail[-1] if tail else ""}')
            stale = []
            for h in hits:
                if h.startswith('merge gate: REFUSED') and '):' in h:
                    for tok in h.split('):', 1)[1].split():
                        if '/' in tok and ':' in tok:
                            cs = tok.split(':', 1)[0]
                            stale.append(list(cs.rsplit('/', 1)))
                if h.startswith('merge: WARNING'):
                    u.meta['unharvested_warning'] = h       # B-30: today a warning, counted
                    if c.catalog_merge.UNHARVESTED == 'refuse':
                        u.refuse(f'B-30 (catalog_merge.UNHARVESTED=refuse): {h[:300]}')
            for name in ('master_catalog.csv', 'master_catalog_refused.csv', 'v_deep_images.csv'):
                self.copy(u, os.path.join(self.U, name), d)
            for name in ('master_catalog.csv', 'master_catalog_refused.csv', 'selected_all.csv', 'v_deep_ledger.csv',
                         'v_deep_images.csv'):
                self.out(u, os.path.join(self.U, name))
            if os.path.exists(os.path.join(self.U, 'stale_attempts.csv')):
                self.out(u, os.path.join(self.U, 'stale_attempts.csv'), cache=True)
            u.meta['stale_attempts'] = stale
            self.write_stage_json(d, harvest_manifest=man, stale_attempts=stale, state_dir=state)
            from exopipe.phase3.core import exclusions
            u.meta['exclusions'] = exclusions.load().fingerprint()
            with open(os.path.join(self.U, 'v_deep_images.csv'), newline='') as f:
                u.set_admitted({f"{r.get('lst')}_{r.get('scope')}:{r.get('date')}" for r in csv.DictReader(f)})
            self.out(u, os.path.join(d, 'master_catalog.csv'), 'merged_record')
            u.note(next((h for h in reversed(hits) if 'published attempts read' in h), tail[-1] if tail else ''))
        return u

    def st_conform(self):
        c = self.cfg
        u = self.new_unit('conform')
        with u:
            self.first_unit_gates(u)
            self.preds_current(u, u.name)
            merge_copy = self.chain.meta(self.u('merge'), 'copies')['master_catalog.csv']
            par = self.chain.meta(self.u('preflight'), 'parent_dir')
            for p, gate in ((c.catalog.SOLAR_AUDIT, 'D-G14 solar audit'), (c.catalog.PULSARS_CSV, 'B-58 pulsar list'),
                            (c.catalog.TABLES_DEEP, 'bright-parent tables'), (c.catalog.DEEP_ROOTS_CSV, 'roots.csv')):
                if not os.path.exists(p):
                    u.refuse(f'{gate} missing: {p}')
                u.add_input(p)
            u.add_input(merge_copy)
            u.add_input(os.path.join(self.U, 'v_deep_ledger.csv'))
            u.add_input(os.path.join(self.chain.meta(self.u('preflight'), 'stage_dir'), 'parents_manifest.csv'))
            u.add_input(c.paths.DATA_DECISIONS)
            self.ensure_master(u, merge_copy, 'conform')
            d = self.stage_dir(u, 'conform')
            env = self.env({'P3_CONFORM_SIZE': '1' if c.catalog_conform.SIZE_STEP['deep'] else '0'}, parent_dir=par)
            rc, tail, hits = self.call(u, 'p3s2_conform.py', [self.U], env, self.log,
                                       keep=('conform:',))
            if rc != 0:
                self.say('conform REFUSED - pass aborted')
                u.refuse(f'conform exit {rc}: {tail[-1] if tail else ""}')
            for name in ('master_catalog.csv', 'master_catalog_removed.csv'):
                self.copy(u, os.path.join(self.U, name), d)
            for name in ('master_catalog.csv', 'master_catalog_asrun.csv', 'master_catalog_removed.csv'):
                self.out(u, os.path.join(self.U, name))
            self.write_stage_json(d, merge_copy=merge_copy, parent_dir=par)
            self.out(u, os.path.join(d, 'master_catalog.csv'), 'conformant_catalog')
            for name in ('frame_times.csv', 'frame_beam_ratio.csv', 'frame_beam_bmaj.csv'):   # bmaj: D8, ops 283a513
                p = os.path.join(self.U, name)
                if os.path.exists(p):
                    self.out(u, p, 'frame_table', cache=True, run_dir=self.U)
            from exopipe.phase3.core import exclusions
            u.meta['exclusions'] = exclusions.load().fingerprint()
            u.note(next((h for h in reversed(hits) if ' raw -> ' in h), tail[-1] if tail else '')[:600])
        return u

    def st_fitpos(self):
        c = self.cfg
        u = self.new_unit('fitpos')
        with u:
            self.first_unit_gates(u)
            if not self.deep_label:
                self.say(f'fitted positions: label {self.label} is not a deep run - its tier fits in its own catalogue stage')
                raise NoInput(f'label {self.label} is not a deep ({c.catalog.DEEP_LABEL_PREFIX}*) run (ops/p3s3_pass.sh:48-52)')
            self.preds_current(u, u.name)
            src = self.chain.meta(self.u('conform'), 'copies')['master_catalog.csv']
            u.add_input(src)
            self.ensure_master(u, src, 'fitpos')
            d = self.stage_dir(u, 'fitpos')
            env = self.env()
            env = self.env({'PYTHONUNBUFFERED': '1'})          # the pass runs it with PY8 -u (p3s3_pass.sh:49)
            rc, tail, hits = self.call(u, 'deep_fit_positions.py', [self.U, '--nproc', str(c.catalog_fitpos.NPROC['deep'])],
                                       env, self.log,
                                       keep=('FIT-POSITIONS', 'REFUSED'))
            if rc != 0:
                self.say('fitted positions REFUSED - pass aborted')
                u.refuse(f'D-G17 fitted positions exit {rc}: {(hits or tail or [""])[-1]}')
            for name in ('master_catalog.csv', 'fit_positions_summary.json'):
                self.copy(u, os.path.join(self.U, name), d)
            self.out(u, os.path.join(self.U, 'master_catalog.csv'))
            self.out(u, os.path.join(self.U, 'fit_positions_summary.json'))
            self.out(u, os.path.join(self.U, 'fit_positions_cache.csv'), cache=True)
            self.write_stage_json(d, input=src)
            self.out(u, os.path.join(d, 'master_catalog.csv'), 'fitted_catalog')
            u.note((hits or tail or [''])[-1])
        return u

    def st_select(self):
        c = self.cfg
        u = self.new_unit('select')
        with u:
            self.first_unit_gates(u)
            self.preds_current(u, u.name)
            src = self.chain.meta(self.sel_pred, 'copies')['master_catalog.csv']
            u.add_input(src)
            self.ensure_master(u, src, 'select')
            d = self.stage_dir(u, 'select')
            extra = {} if c.catalog_select.PARENT_EXCLUSION else {'P3_NO_PARENT_EXCL': '1'}
            slog = os.path.join(self.U, 'select_inc.log')
            rc, tail, _ = self.call(u, 'select_v3.py', ['master_catalog.csv', 'selected_all.csv', c.catalog_select.VERDICT_INPUT],
                                    self.env(extra), slog, mode='w')
            if rc != 0:
                self.say('selection (select_v3) FAILED - pass aborted (D10)')
                u.refuse(f'D-G18 select_v3 exit {rc}: {tail[-1] if tail else ""}')
            n_retro = deep_ported.ephem_retro_rows(self.U)       # WAVE2 B-30: pre-check (M4)
            if n_retro:
                from exopipe.phase3.core import fallback      # WAVE2 B-30
                fallback.site('fallbacks.CATALOG_EPHEM_RETRO_FALLBACK', f'{n_retro} I row(s) of selected_all.csv without in-chain '
                              f'Jupiter / Moon separations - p3s2_ephem_filter.py would compute them at the epoch DATE-OBS, not '
                              f'the frame mid-time', n=n_retro)
            rc, tail, _ = self.call(u, 'p3s2_ephem_filter.py', [], self.env(), slog)
            if rc != 0:
                self.say('ephemeris filter FAILED - pass aborted (D10)')
                u.refuse(f'D-G18 ephemeris filter exit {rc}: {tail[-1] if tail else ""}')
            n_lst = deep_ported.ephem_time_fallbacks('\n'.join(tail or []))
            if n_lst:
                from exopipe.phase3.core import fallback      # WAVE2 B-30
                fallback.site('fallbacks.CATALOG_EPHEM_TIME_FALLBACK', f'p3s2_ephem_filter.py timed {n_lst} (cell, date) '
                              f'epoch(s) by date + LST hour as UT (no Phase-2 or frame DATE-OBS)', n=n_lst)
            u.note(tail[-1] if tail else '')
            counts = deep_ported.split_selection(self.U)
            u.meta['split'] = {f'{s}_{t}': n for (s, t), n in counts.items()}
            self.write_stage_json(d, input=src, split=u.meta['split'])
            self.out(u, os.path.join(self.U, 'selected_all.csv'), 'selection')
            for s in ('V', 'I'):
                for t in ('wb', 'sb'):
                    self.out(u, os.path.join(self.U, f'selected_{s}_{t}.csv'), 'selection')
        return u

    def _associate_one(self, K):
        c = self.cfg
        u = self.new_unit('associate', K)
        with u:
            self.first_unit_gates(u)
            self.preds_current(u, u.name)
            st, tier = K.split('_')
            sel = os.path.join(self.U, f'selected_{K}.csv')
            u.add_input(sel)
            u.add_input(os.path.join(self.U, 'v_deep_images.csv'))
            u.add_input(c.paths.DATA_DECISIONS)
            self.stage_dir(u, 'associate', K)
            out_dir = os.path.join(self.U, f'source_catalogs_{K}')
            self.move_aside(out_dir, u)               # fresh: no previous-pass band file survives (D10)
            if _rows_n(sel) < 1:
                self.say(f'{K}: no selected rows yet')
                raise NoInput(f'{K}: no selected rows (ops/p3s3_pass.sh:79-80: its catalogue directory removed)')
            os.makedirs(out_dir)
            if st == 'I':          # WAVE2 B-30 pre-check (M4): the builder's sigma-stack threshold (:613-617) falls back to 7.0
                from exopipe.phase3.lib.catalog import incremental as _inc      # silently when its source cannot be read
                u.meta['sstack_threshold'] = list(_inc.sstack_threshold())
            rc, tail, _ = self.call(u, 'build_source_catalogs_v3.py', [f'selected_{st}_{tier}.csv', f'source_catalogs_{K}',
                                                                      'v_deep_images.csv'], self.env(),
                                    os.path.join(self.U, f'catalogs_inc_{K}.log'), mode='w')
            if rc != 0:
                self.say(f'catalogue build {K} FAILED - pass aborted (D10)')
                u.refuse(f'D-G19 catalogue build {K} exit {rc}: {tail[-1] if tail else ""}')
            for p in deep_records.files_under(out_dir):
                self.out(u, p)
            self.out(u, out_dir, 'source_catalog')
        return u

    def st_associate(self):
        workers = int(self.args.workers or self.cfg.catalog.DEEP_ASSOC_WORKERS or 1)
        if workers <= 1 or len(self.keys) <= 1:
            return [self._associate_one(K) for K in self.keys]
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(self._associate_one, self.keys))

    def _fork_one(self, S):
        c = self.cfg
        u = self.new_unit('fork', S)
        with u:
            self.first_unit_gates(u)
            if not self.fork_on:
                raise NoInput(f'label {self.label}: the fork is skipped for {c.catalog.DEEP_FORK_SKIP_PREFIX}* runs '
                              f'(ops/p3s3_pass.sh:89)')
            self.preds_current(u, u.name)
            if self.first_stage == 'fork':            # D-G23: the resume of ops/p3s3_pass.sh:23-30, on records
                for K in c.catalog.DEEP_ASSOC_KEYS:
                    ok, why = self.chain.check(self.u('associate', K))
                    if not ok:
                        u.refuse(f'D-G23 --from fork: {why}')
            par = self.chain.meta(self.u('preflight'), 'parent_dir')
            u.add_input(os.path.join(self.U, 'master_catalog_asrun.csv'))
            u.add_input(os.path.join(self.U, 'v_deep_ledger.csv'))
            u.add_input(os.path.join(self.chain.meta(self.u('preflight'), 'stage_dir'), 'parents_manifest.csv'))
            u.add_input(c.catalog.TABLES_DEEP)
            u.add_input(c.paths.DATA_DECISIONS)
            self.stage_dir(u, 'fork', S)
            for name in (f'fork_{S}', f'source_catalogs_{S}_fwb', f'source_catalogs_{S}_fsb'):
                self.move_aside(os.path.join(self.U, name), u)
            env = self.env(parent_dir=par)
            flog = os.path.join(self.U, f'fork_{S}.log')
            rc, tail, _ = self.call(u, 'multiples_fork.py', ['run', S], env, flog, mode='w')
            if rc != 0:
                self.say(f'multiples fork {S} REFUSED ({flog}) - pass aborted')
                u.refuse(f'D-G20 multiples fork {S} exit {rc}: {tail[-1] if tail else ""}')
            with open(flog, errors='replace') as _fl:              # WAVE2 B-30: the fork's own ephemeris filter run
                n_lst = deep_ported.ephem_time_fallbacks(_fl.read())
            if n_lst:
                from exopipe.phase3.core import fallback      # WAVE2 B-30
                fallback.site('fallbacks.CATALOG_EPHEM_TIME_FALLBACK', f'multiples fork {S}: p3s2_ephem_filter.py timed {n_lst} '
                              f'(cell, date) epoch(s) by date + LST hour as UT (no Phase-2 or frame DATE-OBS)', n=n_lst)
            rc, tail, hits = self.call(u, 'multiples_fork.py', ['check', S], env, flog,
                                       keep=('FORK CONTENT TEST',))
            line = hits[-1] if hits else ''
            if rc != 0:
                self.say(f'multiples fork {S} content test FAILED ({flog}) - pass aborted')
                u.refuse(f'D-G20 FORK CONTENT TEST {S} exit {rc}: {line or (tail[-1] if tail else "")}')
            self.say(f'multiples fork {S}: {line}')
            u.check(f'fork.content_test_{S}', True, value=line, detail='ops/multiples_fork.py check (in place)')
            for p in deep_records.files_under(os.path.join(self.U, f'fork_{S}')):
                self.out(u, p)
            dirs = [os.path.join(self.U, f'source_catalogs_{S}_f{t}') for t in ('wb', 'sb')]
            dirs = [x for x in dirs if os.path.isdir(x)]
            for x in dirs:
                for p in deep_records.files_under(x):
                    self.out(u, p)
                self.out(u, x, 'fork_catalog')
            if not dirs:
                raise NoInput(f'the fork kept no {S} source ({line})')
        return u

    def st_fork(self):
        out = []
        for S in self.fork_stokes:
            u = self._fork_one(S)
            out.append(u)
            if S == 'I':
                self.release_lock()     # fork_serve releases after the Stokes I fork content test
            if u.status not in ('done', 'no-input', 'skipped-valid'):
                break
        return out

    def _require_both_stokes(self, u, stage):
        if self.partial_stokes:
            u.refuse(f'{stage} is defined over both Stokes: --stokes {",".join(self.stokes)} refused (spec 1.1)')

    def _forced_gate(self, u):
        """D-G22: `exopipe3 forced` has written its columns since the last associate / fork record."""
        last = max([(self.chain.get(x) or {}).get('finished') or '' for x in self.preds[self.u('identity')]
                    if x != self.u('union')] or [''])
        newer = [t for t, _ in self.chain.forced_records() if t >= last]
        if not newer:
            u.refuse(f'D-G22: no done record of `exopipe3 {self.cfg.catalog.FORCED_STEP}` under {self.U}/_records/ newer '
                     f'than the last associate/fork record ({last}); run forced on this run first '
                     f'(ops/p3s3_pass.sh:101 runs it between the fork and the union)')
        u.note(f'forced record {newer[-1]} is newer than the last associate/fork record {last}')

    def st_union(self):
        c = self.cfg
        u = self.new_unit('union')
        with u:
            self.first_unit_gates(u)
            self._require_both_stokes(u, 'union')
            self.preds_current(u, u.name)
            self._forced_gate(u)
            self.stage_dir(u, 'union')
            files = {}
            for K in c.catalog_identity.UNION_TIERS:
                d = os.path.join(self.U, f'source_catalogs_{K}')
                files[K] = [p for p in self.chain.outputs(self.u('associate', K), '.csv', under=d)]
                for p in files[K]:
                    u.add_input(p)
            outd = os.path.join(self.U, 'source_catalogs')
            kinds = c.catalog_identity.UNION_KINDS
            for p in sorted(os.listdir(outd)) if os.path.isdir(outd) else []:
                if any(p.endswith(f'_{k}.csv') for k in kinds):
                    self.move_aside(os.path.join(outd, p), u)      # fresh union files (D9)
            written = deep_ported.union_tiers(self.U, files, c.catalog_identity.UNION_TIERS, kinds, outd)
            if not written:
                raise NoInput('no primary-tier catalogue to union')
            self.say(f'per-(Stokes,tier) catalogs unioned into source_catalogs/ ({len(written)} files)')
            for p in written:
                self.out(u, p, 'union_catalog')
        return u

    def _expected_survivors(self, tiers=None):
        """The survivors files the associate and fork records list (optionally only those of ``tiers``)."""
        c = self.cfg
        exp = set()
        keys = [('associate', K) for K in c.catalog.DEEP_ASSOC_KEYS]
        if self.fork_on:
            keys += [('fork', S) for S in c.catalog_fork.STOKES]
        for stage, k in keys:
            for p in self.chain.outputs(self.u(stage, k), '_sources_survivors.csv'):
                K = os.path.basename(os.path.dirname(p)).replace('source_catalogs_', '')
                if 'ALL_' in os.path.basename(p) or (tiers is not None and K not in tiers):
                    continue
                exp.add(p)
        return exp

    def _same_reads(self, u, module, actual, expected):
        extra, missing = sorted(actual - expected), sorted(expected - actual)
        if extra or missing:
            u.refuse(f'{module} would read files that no record of this pass lists ({len(extra)}: {extra[:4]}) or miss '
                     f'recorded ones ({len(missing)}: {missing[:4]}) - a stale directory (D9/D10); move it out of the run '
                     f'directory or re-run the stage that owns it')
        u.note(f'{module} reads exactly the {len(expected)} recorded survivors files')

    def st_identity(self):
        c = self.cfg
        u = self.new_unit('identity')
        with u:
            self.first_unit_gates(u)
            self._require_both_stokes(u, 'identity')
            self.preds_current(u, u.name)
            self.stage_dir(u, 'identity')
            tier_files, union_files = deep_ported.identity_reads(self.U, c.catalog_identity.IDENTITY_SKIP_DIRS)
            exp_union = set(self.chain.outputs(self.u('union'), '_sources_survivors.csv'))
            self._same_reads(u, 'ops/p3s3_identity.py', tier_files | union_files, self._expected_survivors() | exp_union)
            for p in sorted(tier_files | union_files):
                u.add_input(p)
            noerr, unparsed = deep_ported.identity_row_fallbacks(tier_files)      # WAVE2 B-30: pre-check (M4)
            if noerr:
                from exopipe.phase3.core import fallback      # WAVE2 B-30
                fallback.site('fallbacks.CATALOG_IDENTITY_POS_ERR_FALLBACK', f'{len(noerr)} survivors row(s) without a positive '
                              f'finite pos_err_arcsec, e.g. {noerr[:3]} - p3s3_identity.py would take 60 arcsec (empty / 0), '
                              f'associate no row of the run (one NaN: the search radius is K x sqrt 2 x the largest error), '
                              f'associate everything (inf) or let it name the identity (negative)', n=len(noerr))
            if unparsed:
                from exopipe.phase3.core import fallback      # WAVE2 B-30 (a per-item drop: counted)
                fallback.site('fallbacks.CATALOG_IDENTITY_ROW_FALLBACK', f'{len(unparsed)} survivors row(s) p3s3_identity.py '
                              f'cannot parse, e.g. {unparsed[:3]} - left out of every identity', n=len(unparsed))
            rc, tail, _ = self.call(u, 'p3s3_identity.py', [self.U], self.env(), self.log)
            if rc != 0:
                self.say('identity FAILED - pass aborted (D10)')
                u.refuse(f'D-G21 identity exit {rc}: {tail[-1] if tail else ""}')
            u.note(tail[-1] if tail else '')
            for p in sorted(tier_files | union_files):
                self.out(u, p, 'identity_table')
        return u

    def st_localize(self):
        c = self.cfg
        u = self.new_unit('localize')
        with u:
            self.first_unit_gates(u)
            self._require_both_stokes(u, 'localize')
            self.preds_current(u, u.name)
            self.stage_dir(u, 'localize')
            actual = deep_ported.localize_reads(self.U, c.catalog_localize.CATS['deep'])
            exp = {p for p in self._expected_survivors()
                   if fnmatch.fnmatch(os.path.basename(os.path.dirname(p)), c.catalog_localize.CATS['deep'])}
            self._same_reads(u, 'ops/build_localizations.py', actual, exp)
            for p in sorted(actual):
                u.add_input(p)
            outd = os.path.join(self.U, 'source_catalogs')
            for name in ('localizations.csv', 'localization_offsets.csv'):
                self.move_aside(os.path.join(outd, name), u)
            rc, tail, _ = self.call(u, 'build_localizations.py', [self.U], self.env(), self.log)
            if rc != 0:
                self.say('localization tables FAILED - pass aborted')
                u.refuse(f'D-G21 localizations exit {rc}: {tail[-1] if tail else ""}')
            u.note(tail[-1] if tail else '')
            for name in ('localizations.csv', 'localization_offsets.csv'):
                p = os.path.join(outd, name)
                if os.path.exists(p):
                    self.out(u, p, 'localization_table')
        return u

    def st_list(self):
        c = self.cfg
        u = self.new_unit('list')
        with u:
            self.first_unit_gates(u)
            self._require_both_stokes(u, 'list')
            self.preds_current(u, u.name)
            self.stage_dir(u, 'list')
            tiers = c.catalog_localize.LIST_TIERS
            self._same_reads(u, 'ops/p3s2_transient_list.py', deep_ported.list_reads(self.U, tiers),
                             self._expected_survivors(tiers))
            u.add_input(os.path.join(self.U, 'master_catalog.csv'))
            out = os.path.join(self.U, c.catalog_localize.LIST_NAME.format(label=self.label, utc=self.ts))
            rc, tail, _ = self.call(u, 'p3s2_transient_list.py', [self.U, out], self.env(), self.log)
            if rc != 0:
                self.say('transient list FAILED - pass aborted (D10)')
                u.refuse(f'D-G21 transient list exit {rc}: {tail[-1] if tail else ""}')
            link = os.path.join(self.U, f'TRANSIENT_LIST_{self.label}_latest.csv')
            tmp = f'{link}.tmp.{os.getpid()}'
            if os.path.lexists(tmp):
                os.unlink(tmp)
            os.symlink(out, tmp)
            os.replace(tmp, link)
            u.note(tail[-1] if tail else '')
            self.out(u, out, 'transient_list')
        return u

    # =================================================================================================================
    # the chain
    # =================================================================================================================
    def execute(self):
        lock_stages = {'preflight', 'conform', 'fork'}
        if lock_stages & set(self.stages):
            self.take_lock()
        try:
            for stage in self.stages:
                fn = getattr(self, f'st_{stage}')
                if stage in ('associate', 'fork'):
                    keys = self.keys if stage == 'associate' else self.fork_stokes
                    pending = [k for k in keys if not self.skip_if_valid(self.u(stage, k), stage)]
                    if stage == 'associate':
                        self.keys = pending
                    else:
                        self.fork_stokes = pending
                    units = fn() if pending else []
                elif self.skip_if_valid(self.u(stage), stage):
                    continue
                else:
                    units = [fn()]
                bad = [x for x in units if x.status not in ('done', 'no-input', 'skipped-valid')]
                stop_noinput = [x for x in units if x.status == 'no-input' and stage not in ('associate', 'fork', 'fitpos')]
                if bad or stop_noinput:
                    x = (bad or stop_noinput)[0]
                    print(f'catalog deep: chain stopped at stage {stage} ({x.name} {x.status}: {x.reason})', file=sys.stderr)
                    return
            last = self.stages[-1]
            if last == 'fork' and 'list' not in self.stages:
                print(f'catalog deep: stages {self.stages[0]}..{last} done. Next: exopipe3 {self.cfg.catalog.FORCED_STEP} '
                      f'--tier deep --run {self.label} ..., then exopipe3 catalog --tier deep --run {self.label} '
                      f'--from union' + (f' --out {self.U}' if not self.prod else ''), file=sys.stderr)
        finally:
            self.release_lock()

    # =================================================================================================================
    # --dry-run: the plan (reads, commands with their environment, writes, gates); nothing is written or run
    # =================================================================================================================
    def plan(self):
        c = self.cfg
        p = print
        p(f'exopipe3 catalog --tier deep --run {self.label}: DRY RUN (nothing written, nothing run)')
        p(f'  run dir (--out) {self.U}  [{"PRODUCTION root: writes refused while paths.PRODUCTION_WRITES is False" if self.prod and not c.paths.PRODUCTION_WRITES else ("production" if self.prod else "non-production: side effects redirected under --out")}]')
        p(f'  stages {",".join(self.stages)}  (default range ends at {c.catalog.DEEP_DEFAULT_UNTIL}; forced runs between fork and union)')
        p(f'  in-place modules {self.ops}  python {self.py}  host {self.host}')
        res = self.compute_binding()
        bad = [r for r in res if not r['ok']]
        p(f'  binding check: {len(res)} literals read, {len(bad)} mismatched' + ('' if c.catalog.BINDING_CHECK else ' (disabled)'))
        for r in bad[:10]:
            p(f'    MISMATCH {r["detail"]}')
        ov = deep_binding.override_refusals(c) + dead_overrides(c)
        if ov:
            p(f'  REFUSED at the first unit: overrides of keys the in-place modules carry as literals or the deep chain '
              f'does not use {ov}')
        if self.filtered:
            p(f'  test filter cells={sorted(self.cells) if self.cells else "-"} exclude={sorted(self.xcells)} '
              f'lsts={sorted(self.lsts) if self.lsts else "-"}' + ('  REFUSED: production out' if self.prod else ''))
        if {'preflight', 'conform', 'fork'} & set(self.stages):
            lock = c.catalog.INSTALL_LOCK.format(ops_root=c.paths.OPS_ROOT)
            p(f'  install lease: {"flock " + lock + " from the first of preflight/conform/fork to the fork I content test" if self.prod else "not taken (non-production run)"}')
        self.chain.reload()
        env_state = self.state_dir()

        def ex(path):
            return 'ok' if os.path.exists(path) else 'MISSING'

        base = self.env()[1]
        p('  stage environment of every in-place call (core.legacy.run; inherited P3_* cleared): '
          + ' '.join(f'{k}={v}' for k, v in sorted(base.items())))

        def cmd(argv, own, log=None, cwd=None):
            e = ' '.join(f'{k}={v}' for k, v in sorted(own.items()) if base.get(k) != v)
            p(f'    run: (cd {cwd or self.U}; {e} {" ".join(argv)}' + (f' >> {log})' if log else ')'))

        def preds(unit):
            for pu in self.preds.get(unit, ()):
                if pu.split('.')[1] in self.stages:
                    p(f'    after: {pu} (this invocation)')
                    continue
                ok, why = self.chain.check(pu)
                p(f'    needs: {pu} record - {"current" if ok else "NOT CURRENT"}: {why}')

        for stage in self.stages:
            p(f'\n  stage {stage}')
            if stage == 'preflight':
                root = c.paths.COADD_SB_ROOT
                p(f'    unit {self.u("preflight")}; masters {root} [{ex(root)}]')
                if self.args.deep_parents not in (None, '', 'snapshot'):
                    par = os.path.abspath(self.args.deep_parents)
                    p(f'    parents given: {par} [{ex(par)}]')
                else:
                    par = (c.catalog.DEEP_PARENT_SNAPSHOT.format(utc=self.ts) if self.prod
                           else os.path.join(self.U, '_parents', f'p3s5_parents_union_{self.ts}'))
                    cmd([self.py, self.script('p3s5_parent_union.py'), par], self.env()[1], f'{par}.union_1.log')
                    p(f'    candidates: ' + ', '.join(f'{x} [{ex(x)}]' for x in c.catalog.DEEP_PARENT_CANDIDATES))
                if os.path.isdir(root):
                    why = {}
                    cells_now, miss = deep_ported.parents_missing(par, root, why)
                    nm = [x for x in miss if why.get(x) == 'no master']
                    if nm:
                        p(f'    cell dirs without a master: {len(nm)} {nm[:8]} - REFUSED (D-G1, as production)')
                    if os.path.isdir(par):
                        p(f'    installed cells {len(cells_now)}; without a catalogue in {par}: {len(miss)}'
                          + (f' - REFUSED: `{c.catalog.PARENTS_COMMAND.format(cells=",".join(miss[:8]))}`' if miss else ''))
                    else:
                        p(f'    installed cells {len(cells_now)} (the snapshot is assembled at run time)')
                p('    missing cells refuse, naming `' + c.catalog.PARENTS_COMMAND.format(cells='<cells>')
                  + '` (parents are built only by masters --stages sources)')
                p('    writes: the snapshot dir, stage_preflight/parents_manifest.csv; class parent_snapshot')
                p('    gates: D-G1 every installed cell has a catalogue on its current master; K5 masters on this host')
            elif stage == 'harvest':
                src_state = self.prod_state_dir()
                p(f'    unit {self.u("harvest")}; search state {src_state} [{ex(src_state)}]'
                  + (f' -> copy {env_state}' if env_state != src_state else ''))
                try:
                    atts = self.attempts(src_state)
                    if self.filtered:
                        atts = [a for a in atts if self.keep_cell(a['cell'])]
                    n_un = sum(1 for a in atts if not a['harvested'])
                    p(f'    published attempts: {len(atts)} ({sum(1 for a in atts if a["stokes"] == "V")} V, '
                      f'{sum(1 for a in atts if a["stokes"] == "I")} I); not harvested: {n_un}'
                      + (' - would be pulled by rsync' if n_un and self.prod else (' - REFUSED (D-G3) outside production' if n_un else '')))
                except Exception as e:
                    p(f'    published attempts: not readable here ({e})')
                p('    writes: stage_harvest/harvest_manifest.csv; class harvest_manifest')
            elif stage == 'merge':
                preds(self.u('merge'))
                cmd([self.py, self.script('p3s3_merge.py'), self.U], self.env(state_dir=env_state)[1], self.log)
                p('    writes: master_catalog.csv, master_catalog_refused.csv, selected_all.csv, v_deep_ledger.csv, '
                  'v_deep_images.csv, stale_attempts.csv (append), candidates/ refusals/ sigstack/ vresid/ iresid/ '
                  'v110_products/ parent_catalogs/; stage copies; class merged_record')
                p('    gates: published attempts = the harvest manifest; D-G4 staleness (markers dropped in '
                  + ('the production state' if self.prod else 'the state COPY') + '); D-G6 backing; D-G7 duplicates; D-G8 zero accepted')
            elif stage == 'conform':
                preds(self.u('conform'))
                par = self.chain.meta(self.u('preflight'), 'parent_dir') or '<parent dir from the preflight record>'
                cmd([self.py, self.script('p3s2_conform.py'), self.U],
                    self.env({'P3_CONFORM_SIZE': '1' if c.catalog_conform.SIZE_STEP['deep'] else '0'}, parent_dir=par)[1], self.log)
                for x in (c.catalog.SOLAR_AUDIT, c.catalog.PULSARS_CSV, c.catalog.TABLES_DEEP, c.catalog.DEEP_ROOTS_CSV):
                    p(f'    reads {x} [{ex(x)}]')
                p('    writes: master_catalog.csv (conformant), master_catalog_asrun.csv, master_catalog_removed.csv, '
                  'frame_beam_ratio.csv + frame_beam_bmaj.csv + frame_times.csv (caches); classes conformant_catalog, frame_table')
                p('    gates: master_catalog.csv = the merge record copy (restored when conform starts the run); D-G9..D-G16')
            elif stage == 'fitpos':
                if not self.deep_label:
                    p(f'    label {self.label} is not a deep run: no fitted-position stage (no-input)')
                    continue
                preds(self.u('fitpos'))
                cmd([self.py, self.script('deep_fit_positions.py'), self.U, '--nproc',
                     str(c.catalog_fitpos.NPROC['deep'])], self.env({'PYTHONUNBUFFERED': '1'})[1], self.log)
                p('    writes: master_catalog.csv (fitted), fit_positions_summary.json, fit_positions_cache.csv; class fitted_catalog')
                p('    gates: D-G17 (input fitted already, frame missing, cand_id set, kind, > 1 per cent on pixel centres)')
            elif stage == 'select':
                preds(self.u('select'))
                extra = {} if c.catalog_select.PARENT_EXCLUSION else {'P3_NO_PARENT_EXCL': '1'}
                cmd([self.py, self.script('select_v3.py'), 'master_catalog.csv', 'selected_all.csv',
                     c.catalog_select.VERDICT_INPUT], self.env(extra)[1], os.path.join(self.U, 'select_inc.log'))
                cmd([self.py, self.script('p3s2_ephem_filter.py')], self.env()[1],
                    os.path.join(self.U, 'select_inc.log'))
                p('    then the split (ported PYS): selected_{V,I}_{wb,sb}.csv; class selection')
                p('    not here: TESS coverage (pages; spec Q10)')
            elif stage == 'associate':
                for K in self.keys:
                    p(f'    unit {self.u("associate", K)}')
                    st, tier = K.split('_')
                    cmd([self.py, self.script('build_source_catalogs_v3.py'), f'selected_{st}_{tier}.csv',
                         f'source_catalogs_{K}', 'v_deep_images.csv'], self.env()[1],
                        os.path.join(self.U, f'catalogs_inc_{K}.log'))
                preds(self.u('associate', self.keys[0])) if self.keys else None
                p(f'    previous source_catalogs_<K>/ moved aside first (fresh); a key with no row: no-input; '
                  f'{int(self.args.workers or c.catalog.DEEP_ASSOC_WORKERS or 1)} at a time; class source_catalog')
            elif stage == 'fork':
                if not self.fork_on:
                    p(f'    label {self.label}: fork skipped ({c.catalog.DEEP_FORK_SKIP_PREFIX}*)')
                    continue
                par = self.chain.meta(self.u('preflight'), 'parent_dir') or '<parent dir from the preflight record>'
                for S in self.fork_stokes:
                    p(f'    unit {self.u("fork", S)}')
                    env = self.env(parent_dir=par)[1]
                    cmd([self.py, self.script('multiples_fork.py'), 'run', S], env, os.path.join(self.U, f'fork_{S}.log'))
                    cmd([self.py, self.script('multiples_fork.py'), 'check', S], env, os.path.join(self.U, f'fork_{S}.log'))
                if self.fork_stokes:
                    preds(self.u('fork', self.fork_stokes[0]))
                p('    writes: fork_<S>/, source_catalogs_<S>_f{wb,sb}/ (previous moved aside); class fork_catalog; '
                  'gates D-G20' + (', D-G23 (associate records current)' if self.first_stage == 'fork' else ''))
            elif stage in ('union', 'identity', 'localize', 'list'):
                if self.partial_stokes:
                    p(f'    REFUSED: {stage} needs both Stokes (--stokes {",".join(self.stokes)})')
                preds(self.u(stage))
                if stage == 'union':
                    last = max([(self.chain.get(x) or {}).get('finished') or '' for x in self.preds[self.u('identity')]
                                if x != self.u('union')] or [''])
                    newer = [t for t, _ in self.chain.forced_records() if t >= last]
                    p(f'    D-G22 forced record newer than {last or "(no associate/fork record)"}: '
                      + (f'yes ({newer[-1]})' if newer else 'NO - would refuse'))
                    p('    ported PYU0 over the files the associate records list; writes source_catalogs/<band>_{sources_survivors,'
                      'sources_master,source_members}.csv (previous moved aside); class union_catalog')
                elif stage == 'identity':
                    cmd([self.py, self.script('p3s3_identity.py'), self.U], self.env()[1], self.log)
                    p('    gate: the files the module globs = the recorded survivors (stale dirs refuse); class identity_table')
                elif stage == 'localize':
                    cmd([self.py, self.script('build_localizations.py'), self.U], self.env()[1], self.log)
                    p('    gate: the glob = the recorded survivors; writes source_catalogs/localizations.csv, '
                      'localization_offsets.csv; class localization_table')
                else:
                    out = os.path.join(self.U, c.catalog_localize.LIST_NAME.format(label=self.label, utc=self.ts))
                    cmd([self.py, self.script('p3s2_transient_list.py'), self.U, out], self.env()[1], self.log)
                    p(f'    then TRANSIENT_LIST_{self.label}_latest.csv -> it; class transient_list')
        p(f'\n  records: {self.U}/_records/catalog/{self.label}.<stage>.json; stage dirs {self.label_dir}/stage_<stage>/')
        return 0


def run(args, run, stages):
    """Entry point of the deep full path (steps/catalog.py dispatches here)."""
    d = DeepRun(args, run, stages)
    if run.dry_run:
        return d.plan()
    d.execute()
    return None
