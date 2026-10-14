"""catalog --tier sublst - the 10-minute (sub-LST) catalogue chain of ONE LST of one search arm.

Ports ops/sublst_catalog_chain.sh, ops/sublst_catalogs.sh (stages prep, veto, build, buildv), the search-complete and
parent CHECKS of ops/sublst_catalog_publish.sh and the reference tie of ops/sublst_resume_from_tie.sh. The in-place modules
(exopipe_ops, clv15; roots = the site file's LEGACY_CODE_ROOTS, which name the frozen tree of the run being shadowed) run
unchanged in children through core/legacy.run with the environment, arguments and working directory production gives them;
the shell logic (gates, heredocs, sequencing, refusals) is Python here.

Unit of work: ONE LST of one arm per invocation (item kind 'lst': `--lsts 05h`; the runner chain 'sublst-lst' fans LSTs
out). Every stage is its own record unit '<LST>_<arm>.<stage>' (e.g. 05h_B.merge) under <out>/_records/catalog/, holding the
files it read (fingerprints), the files it wrote (product class + content tests), the in-place calls (argv, rc, log, child
code sha1) and the module's machine line. The catalogue directory U is --out (production: catalog_sublst.CAT_ROOT/catalog_<SFX>,
SFX = <arm> for 19h, else <LST>_<arm>; a production root, so refused while paths.PRODUCTION_WRITES is False).

Stages (--stages a,b / --from S / --until S; default all, in this order):
  preflight   search of the LST complete on its tree (S-G1: done >= 36 - 2 x NO_WINDOWS, no claim, no failure, search ended
              after the last tree-B build); parent catalogues current (S-G2: gate_inputs.parents on every subband of
              coadd_<LST> - CHECK ONLY: a stale set refuses naming `exopipe3 masters --tier sublst --stages sources`); flux
              bake (S-G5); sub-LST tables installed (S-G12); binding check of every [lit] key; pinned window-tie tables (md5)
  merge       collect layout by symlinks from search_<arm> (sublst_catalogs.sh:94-105) + p3s3_merge.py (S-G7)
  tier-rules  tier rule 3 (deep-search refusals), tier rule 4 (failed flux scale), 10-minute elevation floor (S-G8..10)
  tie-pulsars sublst_pertie.py measure + frames on the rows near ATNF pulsars (S-G11)
  conform     p3s2_conform.py with the sub-LST tables, always (S-G6, S-G13, exopipe_ops 7042670) + TABLE-NOTES content test
  fitpos      sublst_fit_positions.py, template K per Stokes = fitting.K_TEMPLATE[('sublst', S)] (S-G14, exopipe_ops 70003ad),
              then the 10-minute elevation floor again at the FITTED positions (S-G10; exopipe_ops 24d2b39)
  select      select_v3.py (no verdict file, as production: question Q7) + p3s2_ephem_filter.py + split (S-G15)
  tie         sublst_pertie.py measure + frames on every selected row (S-G11: the window tie T1-T13 at the fitted positions)
  veto        leak_veto_sublst.py per (band, Stokes) of master_catalog.csv (S-G16; output unserved, B-24)
  build       source catalogues without the leak veto + localizations + rejected-source record (S-G18..S-G20)
  buildv      source catalogues with the leak veto + localizations (S-G17)
  test        PERTIE-TEST (flow interpreter) + ASTROM-FRAME-TEST (S-G21); copy of the small products to
              compare_<LST>/new_<arm>/catalog and the byte check of pertie_detections.csv (S-G22)
  compare     sublst_compare_19h.py at the levels selected, noveto, veto (non-gating, B-24)
  reftie      sublst_tie.py (reference-tie annotation, output unread; ruling S1 open)
`--from X` needs the record of the stage before X done and its outputs unchanged; master_catalog.csv (rewritten in place by
merge, tier-rules, conform, fitpos) is restored from the stage copy <out>/_records/catalog/<LST>_<arm>/stage_<name>/ when it
changed since. The tests (a) windows and (b) residuals of the sub-LST hand-over are masters / frames content-test classes run
by the chain before `pages` (refactor chat decision 6), not stages of this step.

Inputs:  search outputs paths.SUBLST_ROOT/search_<arm>/<LST>_*_[IV]/, the arm's window tree SUBLST_ROOT/tenmin_root_x2/<LST>_*/
         (frames.csv, NO_WINDOWS.txt), SUBLST_ROOT/coadd_<LST>/ masters, SUBLST_ROOT/parent_catalogs_<LST>/ (built by
         masters --stages sources), SUBLST_ROOT/catalog_inputs/ (roots, footprint catalogues, v1 flux table), the window
         stacks SUBLST_ROOT/<LST>/, the cubes, the search state ops/sublst_search_state_<arm>/{done,claims,failed},
         data_decisions.csv (merge staleness gate, in place), the sub-LST bright-parent tables of the code tree.
Outputs (product classes): master_catalog.csv (merged_record -> tier_rule_removals -> conformant_catalog -> fitted_catalog),
         master_catalog_{deepsearch_refused,fluxscale_refused,elevation_floor,elevation_floor_fitted}.csv
         (tier_rule_removals), frame_times.csv
         (frame_table), selected_*.csv (selection), pertie/pertie_detections.csv (window_tie), leakveto/*.csv
         (leak_veto_table), source_catalogs_{noveto,veto}_<K>/ (source_catalog), localizations_{noveto,veto}/localizations.csv
         (localization_table), rejected.csv + rejected_index.npz + detection_ts.csv (rejected_record), the compare copy
         (compare_copy), compare summaries and tie_members.csv (csv_table), the preflight report (catalog_preflight).
         Same names and layout as production (DESIGN 11.1: no renames in the window).
Failure modes (exit 2, gate named in the record): S-G1..S-G22 of spec section 5.2; any in-place module exiting non-zero
         (its last machine line recorded); a [lit] key whose module literal differs from config; a module switch set in this
         process environment that would leak into the children; a missing shared config key (named). Exit 1 = error.
Per-hour vs intermittent: per LST once its search is complete (DESIGN 12.1: the sub-LST per-hour chain runs `catalog` after
         `search`). `--incremental` (one epoch into the cumulative catalogue) is NOT implemented for this tier: the chain is
         LST-wide today (one merge spans the LST's 36 search jobs).
"""
import ast
import contextlib
import csv
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time

TIER = 'sublst'
STAGES = ('preflight', 'merge', 'tier-rules', 'tie-pulsars', 'conform', 'fitpos', 'select', 'tie', 'veto', 'build',
          'buildv', 'test', 'compare', 'reftie')
# stages whose in-place modules are the heavy part of the chain: they run under the node's heavy-stage lock and memory gate
HEAVY = ('merge', 'tier-rules', 'tie-pulsars', 'conform', 'fitpos', 'select', 'tie', 'veto', 'build', 'buildv', 'test',
         'compare')
OPTIONAL = ('build', 'buildv', 'compare', 'reftie')   # may end no-input by configuration (not selected / switched off)
IN_PLACE = ('master_catalog.csv',)                    # rewritten in place by merge, tier-rules, conform, fitpos
SPLIT_KEYS = ('V_sb', 'I_sb', 'V_wb', 'I_wb')         # sublst_catalogs.sh:235, :255 (order of the --rows and of the builds)
# environment variables the in-place sub-LST modules read that core/legacy.run does not clear (it clears P3_* only): a value
# inherited from the caller's shell would change a child's result silently, so the step refuses while one is set
LEAKY_ENV = ('ELEV_FLOOR', 'ENVELOPE_PCT', 'REF_MIN', 'MASTER_MIN_JY', 'K_ASTROM', 'R_ASTROM_DEG', 'SUBLST_TIE_NEED_GB',
             'SUBLST_CODE_ROOT', 'SUBLST_STACK_ROOT', 'SUBLST_CAT_DIR')
# read IN THIS PROCESS by the tier-rule functions (phase3_tier_rules imported here, as the heredocs import it)
INPROC_ENV = ('P3_COADD_RUN', 'P3_DEEP_SEARCH_STATE')


# =====================================================================================================================
# helpers shared by the tier modules (tier_snapshot imports them)
# =====================================================================================================================

def _cfg():
    from exopipe.phase3.config import CFG
    return CFG


def _record():
    from exopipe.phase3.core import record
    return record


def refuse(msg):
    raise _record().Refusal(msg)


def utc(fmt='%Y-%m-%dT%H:%M:%SZ'):
    return time.strftime(fmt, time.gmtime())


def arg(args, *names, default=None):
    """The first of ``names`` present on args with a value (the dispatcher owns some dest names)."""
    for n in names:
        v = getattr(args, n, None)
        if v not in (None, '', []):
            return v
    return default


def shared(section, key, entry=None):
    """A value of a SHARED config section (owned by the catalog-deep draft); refuses, naming the key, when missing."""
    sec = _cfg().sections().get(section)
    if sec is None or not hasattr(sec, key):
        refuse(f'config key {section}.{key} is missing (spec catalog.md section 3; the shared catalog sections are the '
               f'catalog-deep draft\'s)')
    v = getattr(sec, key)
    if entry is not None:
        if not isinstance(v, dict) or entry not in v:
            refuse(f'config key {section}.{key}[{entry!r}] is missing (spec catalog.md section 3)')
        v = v[entry]
    return v


def legacy_root(name):
    """core.env.legacy_root, refusing (not KeyError) when the site file lacks the root."""
    from exopipe.phase3.core import env
    try:
        return env.legacy_root(name).rstrip('/')
    except KeyError as e:
        refuse(f'{e} - the in-place modules cannot be found (EXOPIPE3_SITE)')


def interpreter(key, dry=False):
    """The site interpreter PYTHON[key] (key in default, flow, py36); in --dry-run an unset key is shown, not refused."""
    from exopipe.phase3.core import legacy
    try:
        return legacy.interpreter(key)
    except _record().Refusal:
        if dry:
            return f'<site PYTHON[{key!r}] UNSET: would refuse>'
        raise


def sha1_file(path):
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


def md5_file(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


def count_lines(path):
    """`wc -l` of a file (newline characters), 0 when it cannot be read."""
    try:
        with open(path, 'rb') as f:
            return sum(blk.count(b'\n') for blk in iter(lambda: f.read(1 << 20), b''))
    except OSError:
        return 0


def select_stages(args, stages, internal=(), label='sub-LST'):
    """Stages to run from --stages / --from / --until (dest names as the dispatcher may spell them). Unknown names exit 1.
    ``internal`` stages (runner items) run only when named in --stages."""
    raw = arg(args, 'stages')
    frm = arg(args, 'from_stage', 'stage_from', 'from_')
    until = arg(args, 'until_stage', 'stage_until', 'until')
    known = tuple(stages) + tuple(internal)
    if raw:
        want = [s.strip() for s in str(raw).split(',') if s.strip()]
        bad = [s for s in want if s not in known]
        if bad:
            raise SystemExit(f'error: unknown {label} stage(s) {bad}; stages: {", ".join(stages)}')
        sel = [s for s in known if s in want]
    else:
        sel = list(stages)
    for name, v in (('--from', frm), ('--until', until)):
        if v and v not in stages:
            raise SystemExit(f'error: {name} {v!r} is not a {label} stage; stages: {", ".join(stages)}')
    if frm:
        sel = [s for s in sel if s in internal or stages.index(s) >= stages.index(frm)]
    if until:
        sel = [s for s in sel if s in internal or stages.index(s) <= stages.index(until)]
    if not sel:
        raise SystemExit('error: no stage selected')
    return sel


class Log:
    """A driver log the way the shell drivers write it: `say` lines '<UTC> msg' appended (and echoed to stderr)."""

    def __init__(self, path, stamp='%Y-%m-%dT%H:%M:%SZ', tag=''):
        self.path, self.stamp, self.tag = path, stamp, tag

    def say(self, msg):
        line = f'{utc(self.stamp)} {self.tag}{msg}'
        print(line, file=sys.stderr, flush=True)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, 'a') as f:
            f.write(line + '\n')

    def raw(self, text):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, 'a') as f:
            f.write(text if text.endswith('\n') else text + '\n')


def last_line(text, pattern):
    """The last line of ``text`` matching the regular expression ``pattern`` ('' when none)."""
    hits = [ln for ln in (text or '').splitlines() if re.search(pattern, ln)]
    return hits[-1] if hits else ''


def call(ctx, s, target, args, python='default', env=None, log=None, cwd=None):
    """core/legacy.run of one in-place script (the ONE child runner, refactor chat decision 1); the call is recorded on the
    stage's unit. Returns the LegacyResult."""
    from exopipe.phase3.core import legacy
    os.makedirs(os.path.dirname(log), exist_ok=True)
    return legacy.run(target, args, python=python, env=env or {}, cwd=cwd, log=log, unit=s.u)


def need_ok(res, what, pattern=None, say=None):
    """Refuse the stage unless the in-place call ended rc 0 (production: any non-zero exit stops the chain)."""
    if res.ok:
        return
    line = last_line(res.tail, pattern) if pattern else ''
    tail = [ln for ln in (res.tail or '').splitlines() if ln.strip()][-1:] or ['(no output)']
    if say:
        say()
    refuse(f'{what}: exit {res.rc}: {(line or tail[0])[:300]}')


def env_leaks(names, prefixes=()):
    """Refuse while the process environment holds a module switch a child would inherit (core/legacy.run clears P3_* only)."""
    hit = sorted(k for k in os.environ if k in names or any(k.startswith(p) for p in prefixes))
    if hit:
        refuse(f'module switch(es) {hit} set in this environment would reach the in-place modules (legacy.run clears only '
               f'P3_*): unset them; every value comes from config')


@contextlib.contextmanager
def stdout_to(path):
    """Send this process's stdout (prints of an in-process legacy function) to a log, as the heredoc's `>> log`."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a') as f, contextlib.redirect_stdout(f):
        yield


# ---- binding check: config value == the in-place module's literal (AST, no import side effects) ----------------------

class _Unresolved(Exception):
    pass


def _ev(node, env):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.Tuple, ast.List)):
        return tuple(_ev(e, env) for e in node.elts)
    if isinstance(node, ast.Dict):
        return {_ev(k, env): _ev(v, env) for k, v in zip(node.keys, node.values)}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _ev(node.operand, env)
        return -v if isinstance(node.op, ast.USub) else +v
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _ev(node.left, env) + _ev(node.right, env)
    if isinstance(node, ast.Name):
        if node.id in env and env[node.id] is not _Unresolved:
            return env[node.id]
        raise _Unresolved(node.id)
    if isinstance(node, ast.JoinedStr):
        out = []
        for v in node.values:
            if isinstance(v, ast.Constant):
                out.append(str(v.value))
            elif isinstance(v, ast.FormattedValue) and v.format_spec is None and v.conversion in (-1, None):
                out.append(str(_ev(v.value, env)))
            else:
                raise _Unresolved('f-string')
        return ''.join(out)
    if isinstance(node, ast.Call):
        f, name = node.func, ''
        while isinstance(f, ast.Attribute):
            name = '.' + f.attr + name
            f = f.value
        if isinstance(f, ast.Name):
            name = f.id + name
        a = [_ev(x, env) for x in node.args]
        if name == 'os.path.join':
            return os.path.join(*a)
        if name == 'os.environ.get' and len(a) == 2:
            return a[1]                                     # the module's DEFAULT (environment not read)
        if name in ('float', 'int', 'str', 'tuple') and len(a) == 1:
            return {'float': float, 'int': int, 'str': str, 'tuple': tuple}[name](a[0])
        raise _Unresolved(name)
    raise _Unresolved(type(node).__name__)


def module_literals(path):
    """{name: value} of the module-level assignments of a Python file whose value is a literal (or built from earlier
    literals by os.path.join, f-strings, os.environ.get defaults, float/int); other names map to _Unresolved."""
    tree = ast.parse(open(path).read(), path)
    env = {}

    def bind(tgt, val):
        if isinstance(tgt, ast.Name):
            try:
                env[tgt.id] = _ev(val, env)
            except _Unresolved:
                env[tgt.id] = _Unresolved
        elif isinstance(tgt, (ast.Tuple, ast.List)) and isinstance(val, (ast.Tuple, ast.List)) \
                and len(tgt.elts) == len(val.elts):
            for t, v in zip(tgt.elts, val.elts):
                bind(t, v)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                bind(tgt, node.value)
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            bind(node.target, node.value)
    return env


def _norm(v):
    if isinstance(v, (list, tuple)):
        return tuple(_norm(x) for x in v)
    if isinstance(v, dict):
        return {k: _norm(x) for k, x in v.items()}
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return float(v)
    return v


def _resolve(target, lits):
    if isinstance(target, str):
        if target not in lits:
            raise _Unresolved(f'{target} not defined at module level')
        if lits[target] is _Unresolved:
            raise _Unresolved(f'{target} is not a literal')
        return lits[target]
    if isinstance(target, dict):
        return {k: _resolve(t, lits) for k, t in target.items()}
    if isinstance(target, (list, tuple)):
        return tuple(_resolve(t, lits) for t in target)
    raise _Unresolved(repr(target))


def binding_check(table):
    """Check [lit] bindings: table = {'section.KEY': (root name, file, target)}; target = a module-level name, or a tuple /
    dict of names (composite keys). Returns [(key, status, config value, module value, where)], status 'ok', 'MISMATCH' or
    'unbound' (file missing, or the name is not a literal)."""
    cfg = _cfg()
    cache, out = {}, []
    for key, (root, fname, target) in table.items():
        sec, _, k = key.partition('.')
        want = getattr(cfg.sections()[sec], k)
        path = os.path.join(legacy_root(root), fname)
        tgt = target if isinstance(target, str) else '+'.join(map(str, target.values() if isinstance(target, dict) else target))
        where = f'{root}/{fname}:{tgt}'
        try:
            if path not in cache:
                cache[path] = module_literals(path)
            got = _resolve(target, cache[path])
        except (OSError, SyntaxError, _Unresolved) as e:
            out.append((key, 'unbound', want, None, f'{where} ({e})'))
            continue
        out.append((key, 'ok' if _norm(got) == _norm(want) else 'MISMATCH', want, got, where))
    return out


def binding_gate(u, rows, plan, dry):
    mism = [r for r in rows if r[1] == 'MISMATCH']
    unb = [r for r in rows if r[1] == 'unbound']
    plan.append(f'binding check: {sum(1 for r in rows if r[1] == "ok")} ok, {len(mism)} mismatch, {len(unb)} unbound')
    for r in mism + unb:
        plan.append(f'   {r[1]:8s} {r[0]} config={r[2]!r} module={r[3]!r} ({r[4]})')
    u.meta['binding'] = [{'key': r[0], 'status': r[1], 'where': r[4]} for r in rows]
    u.check('binding.lit-keys', not mism, value=len(mism),
            detail='; '.join(f'{r[0]}: config {r[2]!r} != {r[3]!r} ({r[4]})' for r in mism[:5]))
    if mism and not dry:
        refuse('binding check: config differs from the in-place literal for ' + '; '.join(
            f'{r[0]} (config {r[2]!r}, {r[4]} {r[3]!r})' for r in mism[:5]))


def env_defaults_differ(pairs, lits):
    """{VAR: str(value)} for each (config value, VAR, module name) whose config value differs from the module's own default:
    production passes none of these, so the environment equals production's while they agree."""
    out = {}
    for value, var, name in pairs:
        d = lits.get(name, _Unresolved)
        if d is _Unresolved or _norm(d) != _norm(value):
            out[var] = str(value)
    return out


# ---- stage records ---------------------------------------------------------------------------------------------------

def stage_record(out_root, unit):
    return _record().load_record(out_root, 'catalog', unit)


def output_fingerprints(paths):
    rec = _record()
    return {os.path.abspath(p): rec.fingerprint(p) for p in paths if os.path.isfile(p)}


def check_from(out_root, unit_prefix, stages, first, recdir, in_place=(), ok_status=('done', 'no-input')):
    """`--from stages[first]`: the record of the stage before must be done (or a recorded no-input of an optional stage) and
    its file outputs unchanged; each file rewritten IN PLACE is checked against the latest earlier stage that wrote it and,
    when it changed since, restored from that stage's copy (<recdir>/stage_<name>/) if the copy matches the record.
    Returns notes; refuses otherwise (DESIGN 11.5: a stage reads only what the previous stage's record lists)."""
    rec = _record()
    prev = f'{unit_prefix}.{stages[first - 1]}'
    r = stage_record(out_root, prev)
    if r is None:
        refuse(f'--from {stages[first]}: no record of the previous stage {prev} under {out_root}/_records/catalog/')
    if r.get('status') not in ok_status:
        refuse(f'--from {stages[first]}: the previous stage {prev} is {r.get("status")} ({r.get("reason")})')
    notes = []
    for p, fp in ((r.get('meta') or {}).get('output_fingerprints') or {}).items():
        if os.path.basename(p) in in_place:
            continue
        if not rec.same_fingerprint(fp, rec.fingerprint(p)):
            refuse(f'--from {stages[first]}: {p} changed since the stage {prev} recorded it')
    for name in in_place:
        for st in reversed(stages[:first]):
            rr = stage_record(out_root, f'{unit_prefix}.{st}')
            fps = ((rr or {}).get('meta') or {}).get('output_fingerprints') or {}
            hit = [p for p in fps if os.path.basename(p) == name]
            if not hit:
                continue
            p, fp = hit[0], fps[hit[0]]
            if rr.get('status') != 'done':
                refuse(f'--from {stages[first]}: {name} was last written by {unit_prefix}.{st}, whose record is '
                       f'{rr.get("status")}')
            if rec.same_fingerprint(fp, rec.fingerprint(p)):
                break
            copy = os.path.join(recdir, f'stage_{st}', name)
            if os.path.exists(copy) and rec.same_fingerprint(fp, rec.fingerprint(copy)):
                shutil.copy2(copy, p)
                notes.append(f'restored {p} from the stage copy {copy} ({unit_prefix}.{st})')
                break
            refuse(f'--from {stages[first]}: {p} changed since {unit_prefix}.{st} wrote it and no matching stage copy')
    return notes


def _exclusions_fp():
    try:
        from exopipe.phase3.core import exclusions
        return exclusions.load().fingerprint()
    except Exception as e:          # recorded; the merge's staleness gate (in place) is what enforces exclusions
        return {'error': str(e)}


class Stage:
    """One stage run: its unit record, the files it read and wrote, the stage copy of in-place files."""

    def __init__(self, ctx, name, u, tier=TIER, in_place=IN_PLACE):
        self.ctx, self.name, self.u, self.tier, self.in_place = ctx, name, u, tier, in_place
        self.written = []

    def inp(self, *paths):
        for p in paths:
            if p and os.path.exists(p):
                self.u.add_input(p)

    def code(self, *scripts):
        """The in-place scripts this stage runs: their bytes identify the code of the unit (DESIGN 11.5)."""
        self.inp(*scripts)

    def out(self, path, cls, **kw):
        if not os.path.exists(path):
            refuse(f'{self.name}: expected output {path} was not written')
        self.u.add_output(path, cls, tier=self.tier, **kw)
        self.written.append(path)

    def keep_stage_copy(self):
        d = os.path.join(self.ctx.recdir, f'stage_{self.name}')
        for name in self.in_place:
            p = os.path.join(self.ctx.U, name)
            if os.path.exists(p):
                os.makedirs(d, exist_ok=True)
                shutil.copy2(p, os.path.join(d, name))

    def close(self):
        self.u.meta['output_fingerprints'] = output_fingerprints(self.written)
        self.u.meta['exclusions'] = _exclusions_fp()


class DryUnit:
    """Stand-in unit for --dry-run planning (records nothing)."""

    def __init__(self):
        self.meta, self.notes = {}, []

    def check(self, *a, **k):
        pass

    def note(self, t):
        self.notes.append(t)

    def add_input(self, *a, **k):
        pass

    def add_output(self, *a, **k):
        pass

    def set_admitted(self, *a):
        pass


def heavy_lock(path):
    """flock of the node's heavy-stage lock (sublst_catalog_chain.sh:26); held until the returned file is closed."""
    import fcntl
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fh = open(path, 'a')
    fcntl.flock(fh, fcntl.LOCK_EX)
    return fh


def mem_gate(ops, need_gb, wait_s, label, log):
    """ops/node_mem_gate.sh NEED WAIT LABEL (bash, in place; runner admission --mem-gb pending, K7). Returns (ok, line)."""
    os.makedirs(os.path.dirname(log), exist_ok=True)
    with open(log, 'a') as lf:
        p = subprocess.run(['bash', os.path.join(ops, 'node_mem_gate.sh'), str(need_gb), str(wait_s), label],
                           stdout=subprocess.PIPE, stderr=lf, universal_newlines=True, stdin=subprocess.DEVNULL)
        lf.write(p.stdout)
    return p.returncode == 0, last_line(p.stdout, 'MEM-GATE')


# =====================================================================================================================
# the sub-LST context
# =====================================================================================================================

def parse_lst(args):
    raw = arg(args, 'lsts')
    if not raw:
        raise SystemExit('error: catalog --tier sublst needs --lsts <LST> (one LST per invocation, e.g. --lsts 05h)')
    from exopipe.phase3.core import cells
    lsts = cells.expand(raw, universe=cells.LSTS, kind='LST')
    try:
        lsts = [cells.parse_lst(x) for x in lsts]
    except ValueError as e:
        raise SystemExit(f'error: {e}')
    if len(lsts) != 1:
        raise SystemExit(f'error: catalog --tier sublst runs ONE LST per invocation (the catalogue directory --out is per '
                         f'LST); got {lsts} - submit one item per LST (exopipe3 run, chain sublst-lst)')
    return lsts[0]


def sfx_of(lst, arm):
    return arm if (lst == '19h' and _cfg().catalog_sublst.SFX_19H_LEGACY) else f'{lst}_{arm}'


def default_out(args):
    """Production catalogue directory of the LST: catalog_sublst.CAT_ROOT/catalog_<SFX> (sublst_catalogs.sh:27)."""
    c = _cfg().catalog_sublst
    arm = arg(args, 'sublst_arm', default=c.ARM)
    try:
        lst = parse_lst(args)
    except SystemExit:
        return c.CAT_ROOT
    return os.path.join(c.CAT_ROOT, f'catalog_{sfx_of(lst, arm)}')


class Ctx:
    """Everything one LST's chain needs (paths as sublst_catalogs.sh:20-30 builds them)."""

    def __init__(self, args, run):
        cfg = _cfg()
        c = cfg.catalog_sublst
        self.args, self.run, self.cfg, self.c = args, run, cfg, c
        self.dry = bool(run.dry_run)
        self.lst = parse_lst(args)
        self.arm = arg(args, 'sublst_arm', default=c.ARM)
        if self.arm not in c.ARM_TREES:
            raise SystemExit(f'error: --sublst-arm {self.arm!r}: arms {", ".join(c.ARM_TREES)}')
        self.hh = self.lst.rstrip('h')
        self.label = c.LABEL.format(hh=self.hh, arm=self.arm)
        self.low = self.label.lower()
        self.sfx = sfx_of(self.lst, self.arm)
        self.unit = f'{self.lst}_{self.arm}'
        builds = arg(args, 'sublst_builds')
        self.builds = tuple(b.strip() for b in builds.split(',') if b.strip()) if builds else tuple(c.BUILDS)
        bad = [b for b in self.builds if b not in ('noveto', 'veto')]
        if bad:
            raise SystemExit(f'error: --sublst-builds {bad}: builds are noveto, veto')
        sl = self.sl = cfg.paths.SUBLST_ROOT
        f = dict(root=sl, lst=self.lst, arm=self.arm, tree=c.ARM_TREES[self.arm])
        self.tree = c.TENMIN_TREE.format(**f)
        self.src = c.SEARCH_DIR.format(**f)
        self.coadd = c.COADD_DIR.format(**f)
        self.parents = c.PARENT_DIR.format(**f)
        self.roots_csv = c.ROOTS_CSV.format(**f)
        self.stack_root = c.STACK_ROOT.format(**f)
        self.compare_root = c.COMPARE_ROOT.format(**f)
        self.search_state = c.SEARCH_STATE.format(ops=cfg.paths.OPS_ROOT, arm=self.arm)
        foot = shared('catalog', 'FOOTPRINT_CAT_DIR', 'sublst')
        if self.lst == '19h' and c.SFX_19H_LEGACY:
            foot = foot.replace('_{lst}', '')
        self.foot = foot.format(lst=self.lst)
        self.U = os.path.abspath(run.out_root)
        self.prod = cfg.production_root_of(self.U) is not None
        ops_state = cfg.paths.OPS_ROOT
        # node-local and ops-side paths: production's in production; under <out>/_side otherwise (never outside --out: the
        # merge's collect layout and state copy, the compare copy, the lock, the logs production keeps in ops)
        if self.prod:
            self.coll = os.path.join(c.CAT_ROOT, f'collect_{self.sfx}')
            self.state = os.path.join(c.CAT_ROOT, f'state_{self.sfx}')
            self.cmp_new = os.path.join(self.compare_root, f'new_{self.arm}')
            self.lock = os.path.join(os.path.dirname(c.CAT_ROOT), 'sublst_heavy.lock')
            self.scratch = None
            self.chain_log = os.path.join(ops_state, f'sublst_catalog_chain_{self.sfx}.log')
            self.pcheck_txt = os.path.join(ops_state, f'sublst_parentcheck_{self.lst}.txt')
        else:
            side = os.path.join(self.U, '_side')
            self.coll = os.path.join(side, f'collect_{self.sfx}')
            self.state = os.path.join(side, f'state_{self.sfx}')
            self.cmp_new = os.path.join(side, f'compare_{self.lst}', f'new_{self.arm}')
            self.lock = os.path.join(side, 'sublst_heavy.lock')
            self.scratch = os.path.join(arg(args, 'work', default=side), f'catalog_sublst_{self.unit}_scratch')
            self.chain_log = os.path.join(side, f'sublst_catalog_chain_{self.sfx}.log')
            self.pcheck_txt = os.path.join(side, f'sublst_parentcheck_{self.lst}.txt')
        self.cmp_catalog = os.path.join(self.cmp_new, 'catalog')
        self.pt = os.path.join(self.U, 'pertie')
        self.recdir = os.path.join(self.U, '_records', 'catalog', self.unit)
        self.log = Log(os.path.join(self.U, 'catalogs.log'))
        self.chainlog = Log(self.chain_log, stamp='%a %b %d %H:%M:%S UTC %Y')
        self.ops = legacy_root('exopipe_ops')
        self.clv = legacy_root('clv15')
        self.xc = os.path.dirname(self.ops)
        self.py = {k: interpreter(k, self.dry) for k in ('default', 'flow', 'py36')}
        # the sub-LST tables OF THE CODE TREE (sublst_catalogs.sh:75: $XC/exopipe/exopipe/common/...)
        tables = shared('catalog', 'TABLES_SUBLST')
        live = '/lustre/gh/main'
        self.tables = (os.path.join(self.xc, tables[len(live) + 1:]) if tables.startswith(live + '/') and self.xc != live
                       else tables)
        self.flux = self.flux_bake()

    # ---- flux bake (sublst_catalogs.sh:58-70, the PYB heredoc) -----------------------------------------------------
    def flux_bake(self):
        s = set()
        for fp in sorted(glob.glob(f'{self.tree}/{self.lst}_*/frames.csv')):
            with open(fp, newline='') as fh:
                for r in csv.DictReader(fh):
                    s.add(str(r.get('fluxbake', '0')).strip() in ('1', 'True', 'true'))
        return 'mixed' if len(s) > 1 else 'baked' if s == {True} else 'unbaked'

    def flux_table(self):
        """P3_FLUX_CORR_TABLE: None (baked), the v1 table (unbaked); refuses a mixed tree (S-G5) and, under the
        'baked-only' policy (B-72 a), an unbaked one."""
        if self.flux == 'baked':
            return None
        if self.flux == 'unbaked' and self.c.FLUX_BAKE_POLICY == 'branch':
            return self.c.FLUX_V1_TABLE
        refuse(f'S-G5: {self.tree} frames for {self.lst} '
               f'{"mix baked and un-baked flux scales" if self.flux == "mixed" else "are un-baked"} ({self.flux}; '
               f'catalog_sublst.FLUX_BAKE_POLICY={self.c.FLUX_BAKE_POLICY})')

    # ---- environment: sublst_catalogs.sh's exports (lines 28-30, 44, 67-68, 81); legacy.run adds it to the parent's
    # environment with P3_* cleared ----------------------------------------------------------------------------------
    def env(self, **extra):
        e = dict(SUBLST_LST=self.lst,
                 P3S_LABEL=self.label, P3S_RUN_DIR=self.U, P3S3_STATE_DIR=self.state, P3S3_COLLECT_ROOTS=self.coll,
                 P3_CADENCE='10min', P3_TIER=shared('catalog_associate', 'TIER_TAG', 'sublst'),
                 P3_TENMIN_ROOT=self.tree, P3_COADD_ROOT=self.coadd, P3_PARENT_DIR=self.parents,
                 P3S2_ROOTS_CSV=self.roots_csv, P3_FOOTPRINT_CAT_DIR=self.foot,
                 P3_TENMIN_PERTIE=os.path.join(self.pt, 'pertie_detections.csv'),
                 P3_LOC_COMMON_FRAME='1' if shared('catalog_localize', 'COMMON_FRAME', 'sublst') else '0',
                 P3_TABLE_PREFIX=shared('catalog_conform', 'TABLE_PREFIX', 'sublst'), P3_TABLE_EXTRA=self.tables,
                 P3_TABLE_ELEV='1' if shared('catalog_conform', 'TABLE_ELEV', 'sublst') else '0')
        ft = self.flux_table() if not self.dry else (None if self.flux == 'baked' else self.c.FLUX_V1_TABLE)
        if ft:
            e['P3_FLUX_CORR_TABLE'] = ft
        e.update({k: str(v) for k, v in extra.items()})
        return e

    def chain_env(self, **extra):
        """sublst_catalog_chain.sh's own environment for its test / compare / tie commands (not sublst_catalogs.sh's)."""
        e = dict(SUBLST_LST=self.lst)
        e.update({k: str(v) for k, v in extra.items()})
        return e

    def opsc(self, name):
        return os.path.join(self.ops, name)


# =====================================================================================================================
# binding table of this tier's [lit] keys (spec section 3) - module-level names of the in-place files
# =====================================================================================================================

LIT = {
    'catalog_sublst.TIER_RULE3_PIN': ('clv15', 'phase3_tier_rules.py', 'DEEP_SEARCH_PIN'),
    'catalog_sublst.TIER_RULE3_PIN_MD5': ('clv15', 'phase3_tier_rules.py', 'DEEP_SEARCH_PIN_MD5'),
    'catalog_sublst.COADD_RECORDS_FROZEN': ('clv15', 'phase3_tier_rules.py', 'COADD_FROZEN'),
    'catalog_sublst.MEMBERSHIP_VETO_PIN': ('clv15', 'phase3_tier_rules.py', 'MEMBERSHIP_VETO_PIN'),
    'catalog_sublst.MEMBERSHIP_VETO_PIN_MD5': ('clv15', 'phase3_tier_rules.py', 'MEMBERSHIP_VETO_PIN_MD5'),
    'catalog_sublst.TIER_RULE4_RANGE': ('clv15', 'phase3_tier_rules.py', 'FLUX_SCALE_RANGE'),
    'catalog_sublst.ELEV_FLOOR_DEG': ('exopipe_ops', 'gate_inputs.py', 'TENMIN_ELEV_FLOOR_DEG'),
    'catalog_sublst.REJECTED_SYS_ARCSEC': ('exopipe_ops', 'emit_rejected_record.py', 'SYS_ARCSEC'),
    'catalog_sublst.REJECTED_BEAM_ARCSEC': ('exopipe_ops', 'emit_rejected_record.py', 'BEAM'),
    'catalog_sublst_tie.WIDEN_DEG': ('exopipe_ops', 'sublst_pertie.py', 'WIDEN_DEG'),
    'catalog_sublst_tie.REF_TRY': ('exopipe_ops', 'sublst_pertie.py', 'REF_TRY'),
    'catalog_sublst_tie.REF_N': ('exopipe_ops', 'sublst_pertie.py', 'REF_N'),
    'catalog_sublst_tie.TIE_MIN_REFS': ('exopipe_ops', 'sublst_pertie.py', 'TIE_MIN_REFS'),
    'catalog_sublst_tie.TIE_CONSISTENCY_K': ('exopipe_ops', 'sublst_pertie.py', 'TIE_CONSISTENCY_K'),
    'catalog_sublst_tie.TIE_SYS_FLOOR_ARCMIN': ('exopipe_ops', 'sublst_pertie.py', 'TIE_SYS_FLOOR_ARCMIN'),
    'catalog_sublst_tie.ACCEPT_SNR': ('exopipe_ops', 'sublst_pertie.py', 'ACCEPT_SNR'),
    'catalog_sublst_tie.ACCEPT_OFFSET_BEAMS': ('exopipe_ops', 'sublst_pertie.py', 'ACCEPT_OFFSET_BEAMS'),
    'catalog_sublst_tie.STAT_COEFF_SNR_FLOOR': ('exopipe_ops', 'sublst_pertie.py', ('STAT_COEFF', 'SNR_FLOOR')),
    'catalog_sublst_tie.ELEV_BANDS_DEG': ('exopipe_ops', 'sublst_pertie.py', 'ELEV_BANDS'),
    'catalog_sublst_tie.FALLBACK_PCT': ('exopipe_ops', 'sublst_pertie.py', 'FALLBACK_PCT'),
    'catalog_sublst_tie.FALLBACK_MIN_N': ('exopipe_ops', 'sublst_pertie.py', 'FALLBACK_MIN_N'),
    'catalog_sublst_tie.FEW_Z': ('exopipe_ops', 'sublst_pertie.py', 'FEW_Z'),
    'catalog_sublst_tie.FEW_FRAC': ('exopipe_ops', 'sublst_pertie.py', 'FEW_FRAC'),
    'catalog_sublst_tie.MAX_SIGNAL_FRAMES': ('exopipe_ops', 'sublst_pertie.py', 'MAX_SIGNAL_FRAMES'),
    'catalog_sublst_tie.CUBE_ROOTS': ('exopipe_ops', 'sublst_pertie.py', 'CUBE_ROOTS'),
    'catalog_sublst_tie.CALIB_MIN_N': ('exopipe_ops', 'sublst_pertie.py', 'CALIB_MIN_N'),
    'catalog_sublst_tie.PAIR_CAP_PER_REF': ('exopipe_ops', 'sublst_pertie.py', 'PAIR_CAP_PER_REF'),
    'catalog_sublst_tie.RHO_PAIR_CAP': ('exopipe_ops', 'sublst_pertie.py', 'RHO_PAIR_CAP'),
    'catalog_sublst_tie.RADIUS99_EDGES': ('exopipe_ops', 'sublst_pertie.py', 'RADIUS99_EDGES'),
    'catalog_sublst_tie.WINERR_SNR_EDGES': ('exopipe_ops', 'sublst_pertie.py', 'WINERR_SNR_EDGES'),
    'catalog_sublst_tie.ERR_MIN_N': ('exopipe_ops', 'sublst_pertie.py', 'ERR_MIN_N'),
    'catalog_sublst_tie.PINNED_TABLES': ('exopipe_ops', 'sublst_pertie.py', {
        'sublst_pertie_fallback.json': 'FALLBACK_MD5', 'sublst_pertie_frame_fallback.json': 'FRAME_FALLBACK_MD5',
        'sublst_pertie_calib.json': 'CALIB_MD5', 'sublst_pertie_winerr.json': 'WINERR_MD5',
        'sublst_pertie_frame_scale.json': 'FRAME_SCALE_MD5', 'sublst_pertie_rho.json': 'TENMIN_RHO_MD5',
        'sublst_pertie_radius99.json': 'RADIUS99_MD5'}),
    'catalog_sublst_tie.ERRMODEL_FITS': ('exopipe_ops', 'sublst_pertie.py', 'ERRMODEL_FITS'),
    'catalog_sublst_tie.ERRMODEL_FITS_SHA256': ('exopipe_ops', 'sublst_pertie.py', 'ERRMODEL_FITS_SHA256'),
    'catalog_sublst_tie.VLSSR_CAT': ('exopipe_ops', 'sublst_pertie.py', 'VLSSR_CAT'),
    'catalog_sublst_tie.VLSSR_SHA256': ('exopipe_ops', 'sublst_pertie.py', 'VLSSR_SHA256'),
    'catalog_sublst_tie.ASTROM_FRAME_INPUT_DIR': ('exopipe_ops', 'astrom_frame.py', 'INPUT_DIR'),
    'catalog_sublst_tie.ASTROM_FRAME_SCREEN_MANIFEST_MD5': ('exopipe_ops', 'astrom_frame.py', 'SCREEN_MANIFEST_MD5'),
    'catalog_sublst_tie.ASTROM_FRAME_BAND_FALLBACK_PCT': ('exopipe_ops', 'astrom_frame.py', 'BAND_FALLBACK_PCT'),
    'catalog_sublst_veto.ELEV_FLOOR_DEG': ('exopipe_ops', 'leak_veto_sublst.py', 'ELEV_FLOOR'),
    'catalog_sublst_veto.ENVELOPE_PCT': ('exopipe_ops', 'leak_veto_sublst.py', 'ENVELOPE_PCT'),
    'catalog_sublst_veto.REF_MIN': ('exopipe_ops', 'leak_veto_sublst.py', 'REF_MIN'),
    'catalog_sublst_veto.MASTER_MIN_JY': ('exopipe_ops', 'leak_veto_sublst.py', 'MASTER_MIN_JY'),
    'catalog_sublst_veto.EL_BINS_DEG': ('exopipe_ops', 'leak_veto_sublst.py', 'EL_BINS'),
    'catalog_sublst_veto.K_ASTROM': ('exopipe_ops', 'leak_veto_sublst.py', 'K_ASTROM'),
    'catalog_sublst_veto.R_ASTROM_DEG': ('exopipe_ops', 'leak_veto_sublst.py', 'R_ASTROM'),
}
# the veto's [env] keys (overridable): passed only when config differs from the module default (production passes none)
VETO_ENV = (('ELEV_FLOOR_DEG', 'ELEV_FLOOR', 'ELEV_FLOOR'), ('ENVELOPE_PCT', 'ENVELOPE_PCT', 'ENVELOPE_PCT'),
            ('REF_MIN', 'REF_MIN', 'REF_MIN'), ('MASTER_MIN_JY', 'MASTER_MIN_JY', 'MASTER_MIN_JY'),
            ('K_ASTROM', 'K_ASTROM', 'K_ASTROM'), ('R_ASTROM_DEG', 'R_ASTROM_DEG', 'R_ASTROM'))
_FITPOS_K_RE = re.compile(r'^\s*FITPOS_K="([^"]*)"', re.M)


def k_arg():
    """The --k arguments of sublst_fit_positions.py from fitting.K_TEMPLATE[('sublst', S)] (sublst_catalogs.sh:213)."""
    kt = _cfg().fitting.K_TEMPLATE
    out = []
    for s in ('I', 'V'):
        if ('sublst', s) not in kt:
            refuse(f"fitting.K_TEMPLATE[('sublst', '{s}')] missing (R-GREGG-20261013-TENMIN-TEMPLATE-K-MEASURED)")
        a, b = kt[('sublst', s)]
        out += ['--k', f'{s}={a!r},{b!r}']
    return out


def k_binding(ops):
    """FITPOS_K of sublst_catalogs.sh (a shell literal) against fitting.K_TEMPLATE and catalog_fitpos.SUBLST_K_ARG."""
    want = ' '.join(k_arg())
    sh = os.path.join(ops, 'sublst_catalogs.sh')
    try:
        m = _FITPOS_K_RE.search(open(sh).read())
    except OSError as e:
        return [('fitting.K_TEMPLATE[sublst]', 'unbound', want, None, f'{sh} ({e})')]
    got = m.group(1).strip() if m else None
    # compare the NUMBERS per Stokes ('0.450' in the shell literal == 0.45 in config), never the formatted strings
    rows = [('fitting.K_TEMPLATE[sublst]', ('unbound' if got is None else
                                            ('ok' if _k_values(got) == _k_values(want) else 'MISMATCH')), want, got,
             'exopipe_ops/sublst_catalogs.sh:FITPOS_K')]
    ka = shared('catalog_fitpos', 'SUBLST_K_ARG')
    norm = ' '.join(f'--k {t}' for t in str(ka).split())
    rows.append(('catalog_fitpos.SUBLST_K_ARG', 'ok' if _k_values(norm) == _k_values(want) else 'MISMATCH', ka, want,
                 'fitting.K_TEMPLATE[sublst] (two keys must agree)'))
    return rows


def _k_values(arg):
    """{'I': (kmaj, kmin), 'V': (...)} from '--k I=0.464,0.450 --k V=0.599,0.602' (or without the --k)."""
    out = {}
    for tok in str(arg).replace('--k', ' ').split():
        if '=' in tok:
            st, vals = tok.split('=', 1)
            try:
                out[st.strip()] = tuple(round(float(v), 6) for v in vals.split(','))
            except ValueError:
                out[st.strip()] = vals
    return out


# =====================================================================================================================
# ported heredocs (sublst_catalogs.sh, sublst_catalog_publish.sh) - Python, same reads, same writes, same printed lines
# =====================================================================================================================

def pcheck_main(lst, ops):
    """sublst_catalog_publish.sh:52-61 (pcheck), run in a child (`python tier_sublst.py pcheck LST OPS`) with P3_PARENT_DIR /
    P3_COADD_ROOT of the LST: gate_inputs.parents on every subband of the coadd tree. Exit 1 if any refuses."""
    sys.path.insert(0, ops)
    import gate_inputs as G
    bad = []
    sbs = sorted(os.path.basename(d)[len(lst) + 1:] for d in glob.glob(f'{G.COADD}/{lst}_*'))
    for b in sbs:
        try:
            G.parents(lst, b)
        except Exception as e:
            bad.append(b)
            print('PARENT-STALE', e)
    print('PARENT-CHECK %s: %d subbands, %d refused%s' % (lst, len(sbs), len(bad), (' (' + ' '.join(bad) + ')') if bad else ''))
    return 1 if bad or not sbs else 0


def _read_catalog(U):
    with open(f'{U}/master_catalog.csv', newline='') as f:
        rows = list(csv.DictReader(f))
    return rows, (list(rows[0].keys()) if rows else ['cand_id'])


def _write_rule(U, hdr, keep, gone, extra_col, removed_name):
    """The heredocs' writes: csv.DictWriter, master_catalog.csv.new + os.replace."""
    with open(f'{U}/master_catalog.csv.new', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=hdr)
        w.writeheader()
        w.writerows(keep)
    with open(f'{U}/{removed_name}', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=hdr + [extra_col])
        w.writeheader()
        w.writerows(gone)
    os.replace(f'{U}/master_catalog.csv.new', f'{U}/master_catalog.csv')


def _imp(root, name):
    if root not in sys.path:
        sys.path.insert(0, root)
    import importlib
    return importlib.import_module(name)


def tier_rule3(U, clv):
    """sublst_catalogs.sh:112-128 (PYR): rows of an (LST, night, band, Stokes) the deep search refused are removed."""
    TR = _imp(clv, 'phase3_tier_rules')
    rows, hdr = _read_catalog(U)
    keep, gone = [], []
    for r in rows:
        g = TR.deep_search_refused(r["lst"], r["date"][:10], r["band"], r["stokes"])
        (gone if g else keep).append(dict(r, deep_search_gate=g) if g else r)
    _write_rule(U, hdr, keep, gone, 'deep_search_gate', 'master_catalog_deepsearch_refused.csv')
    return ("tier rule 3: %d of %d detections removed (deep search refused their night/band/Stokes; records %s)"
            % (len(gone), len(rows), TR.DEEP_REFUSALS))


def tier_rule4(U, clv):
    """sublst_catalogs.sh:134-150 (PYF): rows of an (LST, night, band) with a failed flux scale are removed."""
    TR = _imp(clv, 'phase3_tier_rules')
    rows, hdr = _read_catalog(U)
    keep, gone = [], []
    for r in rows:
        g = TR.flux_scale_failed(r["lst"], r["date"][:10], r["band"])
        (gone if g else keep).append(dict(r, flux_scale_rule=g) if g else r)
    _write_rule(U, hdr, keep, gone, 'flux_scale_rule', 'master_catalog_fluxscale_refused.csv')
    return ("tier rule 4: %d of %d detections removed (failed flux scale: degenerate-flatDC and outside 0.33-3)"
            % (len(gone), len(rows)))


def elevation_floor(U, ops):
    """sublst_catalogs.sh:155-174 (PYE): rows below gate_inputs.TENMIN_ELEV_FLOOR_DEG at their window mid-time are removed;
    a row whose elevation is not finite REFUSES."""
    import numpy as np
    G = _imp(ops, 'gate_inputs')
    rows, hdr = _read_catalog(U)
    keep, gone = [], []
    for r in rows:
        e = float(G.window_elevation(r["lst"], r["date"], float(r["ra"]), float(r["dec"])))
        if not np.isfinite(e):
            refuse(f"S-G10: REFUSING: no elevation for {r['cand_id']}")
        (keep if e >= G.TENMIN_ELEV_FLOOR_DEG else gone).append(
            r if e >= G.TENMIN_ELEV_FLOOR_DEG else dict(r, window_elevation_deg=round(e, 2)))
    _write_rule(U, hdr, keep, gone, 'window_elevation_deg', 'master_catalog_elevation_floor.csv')
    return ("elevation floor: %d of %d detections removed (below %g deg at their window mid-time; Gregg 2026-10-01)"
            % (len(gone), len(rows), G.TENMIN_ELEV_FLOOR_DEG))


def elevation_floor_fitted(U, ops):
    """sublst_catalogs.sh:221-247 (PYE2; exopipe_ops 24d2b39, 2026-10-14): the 10-minute elevation floor again, at the
    FITTED position - the fit moves a detection by up to ~100 arcsec, so one just above the floor at its peak can fall below
    it (06h 2025-01-06 59 MHz I_0156: 24.902 deg) and the window tie has no bin there; the ruled floor (Gregg 2026-10-01)
    applies at the detection's best position. List rows, not dicts: after the fit master_catalog.csv repeats column names
    (axis_flag, axis_norm, fp_prediction, size_beams, size_flag) - a column is read at its LAST occurrence, as the heredoc's
    index map. A row whose elevation is not finite REFUSES."""
    import numpy as np
    G = _imp(ops, 'gate_inputs')
    with open(f"{U}/master_catalog.csv", newline="") as f:
        rd = csv.reader(f)
        hdr = next(rd)
        rows = list(rd)
    ix = {c: i for i, c in enumerate(hdr)}
    keep, gone = [], []
    for r in rows:
        e = float(G.window_elevation(r[ix["lst"]], r[ix["date"]], float(r[ix["ra"]]), float(r[ix["dec"]])))
        if not np.isfinite(e):
            refuse(f"S-G10: REFUSING: no elevation for {r[ix['cand_id']]} at its fitted position")
        (keep if e >= G.TENMIN_ELEV_FLOOR_DEG else gone).append(r if e >= G.TENMIN_ELEV_FLOOR_DEG else r + ["%.2f" % e])
    with open(f"{U}/master_catalog.csv.new", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        w.writerows(keep)
    with open(f"{U}/master_catalog_elevation_floor_fitted.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(hdr + ["window_elevation_deg"])
        w.writerows(gone)
    os.replace(f"{U}/master_catalog.csv.new", f"{U}/master_catalog.csv")
    return ("elevation floor (fitted positions): %d of %d detections removed (below %g deg at their window mid-time)"
            % (len(gone), len(rows), G.TENMIN_ELEV_FLOOR_DEG))


def table_notes(removed_csv):
    """sublst_catalogs.sh:189-201 (PYT): every bright-parent table removal names its elevation band and the image-pixel
    metric. Returns (line, n, bad)."""
    n = bad = 0
    ex = ''
    with open(removed_csv, newline='') as f:
        for r in csv.DictReader(f):
            t = ' '.join(str(r.get(k) or '') for k in ('removal_reason', 'conform_note'))
            if 'table:' not in t or 'parent S=' not in t:
                continue
            n += 1
            if '[el' not in t or '(image-pixel)' not in t:
                bad += 1
                ex = ex or t[:200]
    return (f'TABLE-NOTES {n} bright-parent table removals, {bad} without an elevation band and the image-pixel metric'
            + (f' (e.g. {ex})' if bad else '')), n, bad


def tables_present(tables_file, metric):
    """sublst_catalogs.sh:182: sublst_ keys exist, some with an elevation band, every one with the image-pixel metric."""
    t = json.load(open(tables_file))['tables']
    s = [k for k in t if k.startswith('sublst_')]
    return bool(s and any('_el' in k for k in s) and all(t[k].get('metric') == metric for k in s)), len(s)


def split(U, wb_bands, sb_token):
    """sublst_catalogs.sh:222-233 (PYS): selected_all.csv split by Stokes into subbands ('MHz' in band) and wideband groups.
    Returns the printed lines."""
    with open(f'{U}/selected_all.csv', newline='') as f:
        rows = list(csv.DictReader(f))
    lines = []
    for st in ('V', 'I'):
        for tag, keep in (('sb', lambda b: sb_token in b), ('wb', lambda b: b in wb_bands)):
            sub = [r for r in rows if r['stokes'] == st and keep(r['band'])]
            with open(f'{U}/selected_{st}_{tag}.csv', 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ['cand_id'])
                w.writeheader()
                w.writerows(sub)
            lines.append(f'selected {st} {tag} {len(sub)}')
    return lines


def veto_pairs(U):
    """(band, Stokes) pairs of master_catalog.csv (sublst_catalogs.sh:246 reads columns 4 and 5 by position; here by name,
    refusing when those columns are not band, stokes), in the order of `sort -u` of 'band_stokes'."""
    with open(f'{U}/master_catalog.csv', newline='') as f:
        rd = csv.DictReader(f)
        cols = rd.fieldnames or []
        if len(cols) < 5 or cols[3] != 'band' or cols[4] != 'stokes':
            refuse(f'master_catalog.csv columns 4-5 are {cols[3:5]}, not band, stokes: the production veto loop reads them by '
                   f'position (sublst_catalogs.sh:246) - refusing rather than vetoing other pairs')
        pairs = sorted({f"{r['band']}_{r['stokes']}" for r in rd})
    return [tuple(p.rsplit('_', 1)) for p in pairs]


def search_complete(ctx):
    """S-G1 (sublst_catalog_chain.sh:17-22; sublst_catalog_publish.sh:38-47): done >= 36 - 2 x NO_WINDOWS, no claim, no
    failure in the arm's search state; and the LST's run log <OPS_ROOT>/sublst_run_<LST>.log with a search end after its last
    '4 tree B built' line (or every done marker newer than it) - no run log or no tree-B line refuses, as the production
    launcher does. Returns (ok, detail). The search run records replace this at cutover."""
    c = ctx.c
    st = ctx.search_state
    nowin = sorted(glob.glob(f'{ctx.tree}/{ctx.lst}_*/NO_WINDOWS.txt'))
    nj = c.EXPECTED_JOBS[0] - c.EXPECTED_JOBS[1] * len(nowin)

    def ls(sub):
        try:
            return sorted(x for x in os.listdir(os.path.join(st, sub)) if x.startswith(f'{ctx.lst}_'))
        except OSError:
            return []
    done, claims, failed = ls('done'), ls('claims'), ls('failed')
    detail = (f'search state {st}: {len(done)} done of {nj} expected (36 - 2 x {len(nowin)} NO_WINDOWS'
              f'{": " + " ".join(os.path.basename(os.path.dirname(p)) for p in nowin) if nowin else ""}), '
              f'{len(claims)} claims, {len(failed)} failed')
    ok = len(done) >= nj and not claims and not failed
    runlog = os.path.join(ctx.cfg.paths.OPS_ROOT, f'sublst_run_{ctx.lst}.log')
    # review 2026-10-13: the production launcher (sublst_catalog_publish.sh, run by sublst_followon / sublst_frozen_run)
    # REFUSES when the run log has no '4 tree B built' line or does not exist: `NTB=$(grep -n ...)` (:39) and
    # `[ -n "$NTB" ] && {...} && ... || REFUSING catpub` (:44-47). The port skipped the condition in that case.
    if ok and not os.path.exists(runlog):
        detail += f'; no run log {runlog}: REFUSED (sublst_catalog_publish.sh:39, :44-47)'
        ok = False
    if ok:
        lines = open(runlog, errors='replace').read().splitlines()
        tb = [i for i, ln in enumerate(lines) if '4 tree B built' in ln]
        if not tb:
            detail += f"; run log {runlog} has no '4 tree B built' line: REFUSED (sublst_catalog_publish.sh:39, :44-47)"
            ok = False
        else:
            ended = any(re.search(r'5 search complete|6 catalogues \+ comparison done', ln) for ln in lines[tb[-1]:])
            if not ended:
                try:
                    import calendar
                    t0 = calendar.timegm(time.strptime(lines[tb[-1]][:28].strip(), '%a %b %d %H:%M:%S UTC %Y'))
                except ValueError:
                    t0 = None
                newer = [d for d in done if t0 is not None and os.path.getmtime(os.path.join(st, 'done', d)) > t0]
                ended = len(newer) >= nj
            detail += f'; run log {runlog}: search ended on the current tree: {ended}'
            ok = ok and ended
    return ok, detail


def _admitted(U):
    p = os.path.join(U, 'master_catalog.csv')
    if not os.path.exists(p):
        return []
    with open(p, newline='') as f:
        return sorted({(r.get('date') or '')[:10] for r in csv.DictReader(f) if r.get('date')})


# =====================================================================================================================
# stages
# =====================================================================================================================

def st_preflight(ctx, s, plan):
    c, u = ctx.c, s.u
    ok, detail = search_complete(ctx)
    plan.append(f'S-G1 search complete: {ok} - {detail}')
    u.check('S-G1.search-complete', ok, detail=detail)
    if not ok and not ctx.dry:
        refuse(f'S-G1: no completed search of {ctx.lst} arm {ctx.arm} on its current tree ({detail})')
    plan.append(f'S-G5 flux bake of {ctx.tree}/{ctx.lst}_*/frames.csv: {ctx.flux}'
                f' -> P3_FLUX_CORR_TABLE {"unset" if ctx.flux == "baked" else c.FLUX_V1_TABLE}')
    if not ctx.dry:
        ctx.flux_table()
    try:
        tp, nk = tables_present(ctx.tables, c.TABLES_REQUIRED_METRIC)
    except (OSError, ValueError, KeyError) as e:
        tp, nk = False, f'unreadable ({e})'
    plan.append(f'S-G12 sub-LST tables {ctx.tables}: {nk} sublst_ keys, elevation bands + {c.TABLES_REQUIRED_METRIC}: {tp}')
    u.check('S-G12.sublst-tables', tp, value=nk, detail=ctx.tables)
    s.inp(ctx.tables)
    if not tp and not ctx.dry:
        refuse(f'S-G12: no installed sub-LST bright-parent tables with elevation bands and the {c.TABLES_REQUIRED_METRIC} '
               f'metric in {ctx.tables} (catalog never derives or installs them: exopipe3 calibrate)')
    binding_gate(u, binding_check(LIT) + k_binding(ctx.ops), plan, ctx.dry)
    bad_md5 = []
    for name, md5 in ctx.cfg.catalog_sublst_tie.PINNED_TABLES.items():
        p = os.path.join(ctx.ops, name)
        got = md5_file(p) if os.path.exists(p) else 'missing'
        if got != md5:
            bad_md5.append(f'{name} {got}')
    plan.append(f'pinned window-tie tables beside {ctx.opsc("sublst_pertie.py")}: '
                f'{len(ctx.cfg.catalog_sublst_tie.PINNED_TABLES) - len(bad_md5)} ok' + (f', BAD {bad_md5}' if bad_md5 else ''))
    u.check('pinned-tables.md5', not bad_md5, value=len(bad_md5), detail='; '.join(bad_md5))
    if bad_md5 and not ctx.dry:
        refuse(f'pinned window-tie tables differ from config md5: {bad_md5}')
    # S-G2: CHECK ONLY (refactor chat decision 6: masters --tier sublst --stages sources builds the parents)
    sbs = sorted(os.path.basename(d)[len(ctx.lst) + 1:] for d in glob.glob(f'{ctx.coadd}/{ctx.lst}_*'))
    cats = sorted(glob.glob(f'{ctx.parents}/master_R075_{ctx.lst}_*.gaul.csv'))
    if ctx.dry:
        nside = sum(1 for p in cats if os.path.exists(p + '.gen.json'))
        plan.append(f'S-G2 parents (check only): {len(sbs)} subbands in {ctx.coadd}, {len(cats)} catalogues ({nside} with '
                    f'.gen.json) in {ctx.parents}; md5 against the masters at run: {ctx.py["default"]} {__file__} pcheck '
                    f'{ctx.lst} {ctx.ops} > {ctx.pcheck_txt}')
    else:
        res = call(ctx, s, os.path.abspath(__file__), ['pcheck', ctx.lst, ctx.ops], 'default',
                   env={'P3_PARENT_DIR': ctx.parents, 'P3_COADD_ROOT': ctx.coadd}, log=ctx.pcheck_txt, cwd=ctx.U)
        line = last_line(res.tail, r'^PARENT-CHECK')
        u.check('S-G2.parents-current', res.ok, detail=line, path=ctx.pcheck_txt)
        if not res.ok:
            refuse(f'S-G2: {line or "parent check failed"} - the parent catalogues of {ctx.lst} are stale or missing: build '
                   f'them with `exopipe3 masters --tier sublst --lsts {ctx.lst} --stages sources` (catalog never builds them)')
        plan.append(f'S-G2 {line}')
        s.code(ctx.opsc('gate_inputs.py'))
    plan.append(f'S-G3 memory: {c.CHAIN_MEM_GB} GB free in /user.slice within {c.CHAIN_MEM_WAIT_S} s (node_mem_gate.sh, under '
                f'the heavy lock {ctx.lock}) before the first heavy stage')
    plan.append(f'S-G4 code: in-place modules of {ctx.ops} and {ctx.clv} (site LEGACY_CODE_ROOTS); child code sha1 recorded')
    plan.append('S-G6 sub-LST tables only: no switch exists (SUBLST_TABLES gone since exopipe_ops 7042670)')
    if not ctx.dry:
        os.makedirs(ctx.recdir, exist_ok=True)
        rep = os.path.join(ctx.recdir, 'preflight.json')
        with open(rep, 'w') as f:
            json.dump({'unit': ctx.unit, 'plan': plan, 'binding': u.meta.get('binding'), 'flux': ctx.flux,
                       'tables': ctx.tables, 'utc': utc()}, f, indent=1)
        s.out(rep, 'catalog_preflight')


def st_merge(ctx, s, plan):
    U, A = ctx.U, os.path.join(ctx.coll, f'{ctx.low}_collect_sublst')
    script = ctx.opsc('p3s3_merge.py')
    jobs = sorted(glob.glob(f'{ctx.src}/{ctx.lst}_*_[IV]'))
    usable = [d for d in jobs if os.path.isfile(f'{d}/transients/Run_{ctx.label}/v_deep_ledger.csv')]
    plan.append(f'collect layout {A} by symlinks: {len(usable)} of {len(jobs)} search jobs in {ctx.src} carry '
                f'transients/Run_{ctx.label}/v_deep_ledger.csv; state {ctx.state}')
    plan.append(f'merge: {ctx.py["default"]} {script} {U}   (cwd {U}; P3S_LABEL={ctx.label} P3S3_STATE_DIR={ctx.state} '
                f'P3S3_COLLECT_ROOTS={ctx.coll}) >> catalogs.log')
    if ctx.dry:
        return
    os.makedirs(U, exist_ok=True)
    s.code(script, ctx.opsc('p3s3_sources.py'), ctx.opsc('p3s3_stale_exclusions.py'))
    # `rm -rf "$COLL" "$STATE" "$PT"` (sublst_catalogs.sh:96): the previous copies are moved aside, never deleted
    stamp = utc('%Y%m%dT%H%M%SZ')
    for d in (ctx.coll, ctx.state, ctx.pt):
        if os.path.lexists(d):
            dst = os.path.join(ctx.recdir, 'superseded', f'{os.path.basename(d)}_{stamp}')
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(d, dst)
            s.u.note(f'previous {d} kept as {dst}')
    for d in (f'{A}/attempts', f'{A}/parent_catalogs', f'{A}/v110_products', ctx.state):
        os.makedirs(d, exist_ok=True)
    for st in ('V', 'I'):
        open(f'{ctx.state}/published_{st}.csv', 'w').close()
    for d in jobs:
        j = os.path.basename(d)
        if not os.path.isfile(f'{d}/transients/Run_{ctx.label}/v_deep_ledger.csv'):
            ctx.log.say(f'SKIP {j}: no ledger (not a completed search)')
            continue
        cell, st = j.rsplit('_', 1)
        os.symlink(d, f'{A}/attempts/{cell}_{st}_x')
        with open(f'{ctx.state}/published_{st}.csv', 'a') as f:
            f.write(f'{cell},sublst,x\n')
        if os.path.isdir(f'{d}/catdir'):
            for fn in sorted(os.listdir(f'{d}/catdir')):        # cp -n: never clobber
                dst = f'{A}/parent_catalogs/{fn}'
                if not os.path.lexists(dst) and os.path.isfile(f'{d}/catdir/{fn}'):
                    shutil.copy(f'{d}/catdir/{fn}', dst)
        if os.path.isdir(f'{d}/v110_out'):
            dst = f'{A}/v110_products/{cell}'
            if os.path.islink(dst):
                os.unlink(dst)
            os.symlink(f'{d}/v110_out', dst)
        s.inp(f'{d}/transients/Run_{ctx.label}/v_deep_ledger.csv', f'{d}/transients/Run_{ctx.label}/master_catalog.csv')
    nv, ni = count_lines(f'{ctx.state}/published_V.csv'), count_lines(f'{ctx.state}/published_I.csv')
    ctx.log.say(f'published V={nv} I={ni} cell-Stokes from {ctx.src}')
    ctx.log.say('merge gate')
    res = call(ctx, s, script, [U], env=ctx.env(), log=ctx.log.path, cwd=U)
    need_ok(res, 'S-G7 merge gate (p3s3_merge.py)', r'REFUS|merge', say=lambda: ctx.log.say('MERGE GATE REFUSED - stopping'))
    s.u.note(last_line(res.tail, r'merge') or 'merge done')
    s.out(f'{U}/master_catalog.csv', 'merged_record')
    for name in ('master_catalog_refused.csv', 'v_deep_images.csv'):
        if os.path.exists(f'{U}/{name}'):
            s.out(f'{U}/{name}', 'csv_table')
    s.u.set_admitted(_admitted(U))
    s.keep_stage_copy()


def st_tier_rules(ctx, s, plan):
    U, c = ctx.U, ctx.c
    plan.append(f'tier rule 3 (in process, as the heredoc): phase3_tier_rules.deep_search_refused of {ctx.clv}; pin '
                f'{c.TIER_RULE3_PIN} md5 {c.TIER_RULE3_PIN_MD5} -> master_catalog_deepsearch_refused.csv')
    plan.append(f'tier rule 4 (in process): phase3_tier_rules.flux_scale_failed (frozen records {c.COADD_RECORDS_FROZEN}) '
                f'-> master_catalog_fluxscale_refused.csv')
    plan.append(f'elevation floor (in process): gate_inputs.window_elevation >= {c.ELEV_FLOOR_DEG} deg -> '
                f'master_catalog_elevation_floor.csv')
    if ctx.dry:
        return
    env_leaks(INPROC_ENV)
    s.inp(f'{U}/master_catalog.csv', c.TIER_RULE3_PIN, c.MEMBERSHIP_VETO_PIN)
    s.code(os.path.join(ctx.clv, 'phase3_tier_rules.py'), ctx.opsc('gate_inputs.py'))
    for fn, root, name in ((tier_rule3, ctx.clv, 'tier rule 3 (deep-search refusals)'),
                           (tier_rule4, ctx.clv, 'tier rule 4 (failed flux scales)'),
                           (elevation_floor, ctx.ops, 'elevation floor')):
        try:
            with stdout_to(ctx.log.path):
                line = fn(U, root)
                print(line, flush=True)
        except _record().Refusal:
            raise
        except Exception as e:
            ctx.log.say(f'REFUSING: {name} failed')
            refuse(f'{name} failed: {type(e).__name__}: {e}')
        ctx.log.say(line)
        s.u.note(line)
    for name in ('master_catalog_deepsearch_refused.csv', 'master_catalog_fluxscale_refused.csv',
                 'master_catalog_elevation_floor.csv', 'master_catalog.csv'):
        s.out(f'{U}/{name}', 'tier_rule_removals')
    s.u.set_admitted(_admitted(U))
    s.keep_stage_copy()


def _pertie(ctx, s, plan, rows, near_pulsars, what):
    U, c = ctx.U, ctx.c
    script = ctx.opsc('sublst_pertie.py')
    common = ['--lst', ctx.lst, '--out', ctx.pt, '--stack-root', ctx.stack_root, '--parents', ctx.parents]
    rows_a = [x for r in rows for x in ('--rows', r)] + (['--near-pulsars'] if near_pulsars else [])
    scr = ['--scratch', os.path.join(ctx.scratch, 'sublst_pertie_scratch')] if ctx.scratch else []
    m_args = ['measure'] + common + ['--shards', str(c.PERTIE_SHARDS)] + rows_a + scr
    f_args = ['frames'] + common + ['--workers', str(c.PERTIE_WORKERS), '--shards', str(c.PERTIE_SHARDS)] + rows_a + scr
    plan.append(f'window tie measure: {ctx.py["default"]} {script} ' + ' '.join(m_args) + f'   (cwd {U}) >> catalogs.log')
    plan.append(f'window tie frames:  {ctx.py["flow"]} {script} ' + ' '.join(f_args))
    if ctx.dry:
        return
    s.code(script)
    s.inp(*[os.path.join(U, r) for r in rows])
    ctx.log.say(f'window tie ({what})')
    # production runs `python -u` (unbuffered); legacy.run takes no interpreter options - output order in the log is the same
    res = call(ctx, s, script, m_args, 'default', env=ctx.env(), log=ctx.log.path, cwd=U)
    need_ok(res, f'S-G11 window tie measure ({what})', r'REFUS|MEASURED|Error',
            say=lambda: ctx.log.say(f'REFUSING: window tie ({what}) failed - see catalogs.log'))
    ml = last_line(res.tail, 'MEASURED')
    ctx.log.say(ml[21:] if len(ml) > 21 else ml)
    res = call(ctx, s, script, f_args, 'flow', env=ctx.env(), log=ctx.log.path, cwd=U)
    need_ok(res, f'S-G11 few-frame stage ({what})', r'REFUS|FRAMES-DONE|Error',
            say=lambda: ctx.log.say(f'REFUSING: few-frame stage ({what}) failed - see catalogs.log'))
    fl = last_line(res.tail, 'FRAMES-DONE')
    ctx.log.say(fl[21:] if len(fl) > 21 else fl)
    s.u.note(ml)
    s.u.note(fl)
    s.out(os.path.join(ctx.pt, 'pertie_detections.csv'), 'window_tie', selected_dir=U, near_pulsars=near_pulsars)


def st_tie_pulsars(ctx, s, plan):
    _pertie(ctx, s, plan, ['master_catalog.csv'], True, '1 of 2: detections within one frame beam of an ATNF pulsar')


def st_tie(ctx, s, plan):
    _pertie(ctx, s, plan, [f'selected_{k}.csv' for k in SPLIT_KEYS], False, '2 of 2: every selected detection')


def st_conform(ctx, s, plan):
    U, c = ctx.U, ctx.c
    script = ctx.opsc('p3s2_conform.py')
    env_x = dict(P3_TABLE_RECORD_ONLY='1' if shared('catalog_conform', 'TABLE_RECORD_ONLY') else '0',
                 P3_CONFORM_SIZE='1' if shared('catalog_conform', 'SIZE_STEP', 'sublst') else '0')
    plan.append(f'conform: {" ".join(f"{k}={v}" for k, v in env_x.items())} {ctx.py["default"]} {script} {U}   >> '
                f'catalogs.log; then TABLE-NOTES on master_catalog_removed.csv (S-G13)')
    if ctx.dry:
        return
    tp, nk = tables_present(ctx.tables, c.TABLES_REQUIRED_METRIC)
    if not tp:
        refuse(f'S-G12: no installed sub-LST bright-parent tables with elevation bands and the {c.TABLES_REQUIRED_METRIC} '
               f'metric in {ctx.tables}')
    ctx.log.say(f'sub-LST tables present in {ctx.tables} (P3_TABLE_PREFIX={shared("catalog_conform", "TABLE_PREFIX", "sublst")}'
                f', elevation bands on) - applied by the conformance stage (never derived here)')
    ctx.log.say('conformance check with the sub-LST tables')
    s.code(script, ctx.opsc('gate_inputs.py'))
    s.inp(f'{U}/master_catalog.csv', ctx.tables, ctx.roots_csv)
    res = call(ctx, s, script, [U], env=ctx.env(**env_x), log=ctx.log.path, cwd=U)
    need_ok(res, 'S-G13 conformance (p3s2_conform.py)', r'REFUS', say=lambda: ctx.log.say('CONFORM REFUSED - stopping'))
    line, n, bad = table_notes(f'{U}/master_catalog_removed.csv')
    ctx.log.raw(line)
    s.u.check('S-G13.table-notes', bad == 0, value=bad, detail=line, cls='conformant_catalog',
              path=f'{U}/master_catalog_removed.csv')
    if bad:
        ctx.log.say(f'REFUSING: {line[:300]}')
        refuse(f'S-G13: {line}')
    ctx.log.say(line)
    s.out(f'{U}/master_catalog.csv', 'conformant_catalog')
    s.out(f'{U}/master_catalog_removed.csv', 'csv_table')
    s.out(f'{U}/master_catalog_asrun.csv', 'csv_table')
    if os.path.exists(f'{U}/frame_times.csv'):
        s.out(f'{U}/frame_times.csv', 'frame_table')
    s.u.set_admitted(_admitted(U))
    s.keep_stage_copy()


def st_fitpos(ctx, s, plan):
    U = ctx.U
    script = ctx.opsc('sublst_fit_positions.py')
    a = ([U, '--search-root', ctx.src, '--label', ctx.label] + k_arg()
         + ['--nproc', str(shared('catalog_fitpos', 'NPROC', 'sublst'))])
    plan.append(f'fitted positions: {ctx.py["default"]} {script} ' + ' '.join(a) + '   >> catalogs.log (S-G14: F1-F4 in the '
                f'module)')
    plan.append(f'elevation floor at the fitted positions (in process): gate_inputs.window_elevation >= '
                f'{ctx.c.ELEV_FLOOR_DEG} deg -> master_catalog_elevation_floor_fitted.csv (S-G10; exopipe_ops 24d2b39)')
    if ctx.dry:
        return
    env_leaks(INPROC_ENV)
    s.code(script, ctx.opsc('deep_forced_radio.py'), ctx.opsc('gate_inputs.py'))
    s.inp(f'{U}/master_catalog.csv')
    ctx.log.say(f'fitted detection positions on the searched frames (template K: {" ".join(k_arg())})')
    res = call(ctx, s, script, a, env=ctx.env(), log=ctx.log.path, cwd=U)
    if not res.ok:
        ln = last_line(res.tail, r'^(REFUSED|FIT-POSITIONS)')
        ctx.log.say(f'REFUSING: fitted positions failed - {ln[:300]}')
        refuse(f'S-G14 fitted positions: sublst_fit_positions.py exit {res.rc}: {ln[:300]}')
    ok = last_line(res.tail, r'^FIT-POSITIONS-OK')
    ctx.log.say(ok[:400])
    s.u.note(ok)
    try:
        with stdout_to(ctx.log.path):
            line = elevation_floor_fitted(U, ctx.ops)
            print(line, flush=True)
    except _record().Refusal:
        raise
    except Exception as e:
        ctx.log.say('REFUSING: elevation floor (fitted positions) failed')
        refuse(f'S-G10 elevation floor (fitted positions) failed: {type(e).__name__}: {e}')
    ctx.log.say(line)
    s.u.note(line)
    s.out(f'{U}/master_catalog_elevation_floor_fitted.csv', 'tier_rule_removals')
    s.out(f'{U}/master_catalog.csv', 'fitted_catalog')
    if os.path.exists(f'{U}/fit_positions_record.csv'):
        s.out(f'{U}/fit_positions_record.csv', 'csv_table')
    if os.path.exists(f'{U}/fit_positions_summary.json'):
        s.inp(f'{U}/fit_positions_summary.json')
    s.keep_stage_copy()


def st_select(ctx, s, plan):
    U = ctx.U
    sel, eph = ctx.opsc('select_v3.py'), ctx.opsc('p3s2_ephem_filter.py')
    excl = '0' if shared('catalog_select', 'PARENT_EXCLUSION') else '1'
    split_cfg = shared('catalog_select', 'SPLIT', 'sublst')
    plan.append(f'selection: P3_NO_PARENT_EXCL={excl} {ctx.py["default"]} {sel} master_catalog.csv selected_all.csv   (cwd '
                f'{U}; no verdict file, as production: question Q7)')
    plan.append(f'ephemeris filter: {ctx.py["default"]} {eph}   (P3S_RUN_DIR={U}); split by Stokes: sb = {split_cfg["sb"]!r} '
                f'in band, wb = {split_cfg["wb"]}')
    if ctx.dry:
        return
    s.code(sel, eph)
    s.inp(f'{U}/master_catalog.csv')
    ctx.log.say('selection')
    res = call(ctx, s, sel, ['master_catalog.csv', 'selected_all.csv'], env=ctx.env(P3_NO_PARENT_EXCL=excl),
               log=ctx.log.path, cwd=U)
    need_ok(res, 'S-G15 select_v3.py', say=lambda: ctx.log.say('SELECT FAILED'))
    from exopipe.phase3.lib.catalog import deep_ported as _DP
    n_retro = _DP.ephem_retro_rows(U)                     # WAVE2 B-30: pre-check (M4)
    if n_retro:
        from exopipe.phase3.core import fallback      # WAVE2 B-30
        fallback.site('fallbacks.CATALOG_EPHEM_RETRO_FALLBACK', f'{n_retro} I row(s) of {U}/selected_all.csv without in-chain '
                      f'Jupiter / Moon separations - p3s2_ephem_filter.py would compute them at the epoch DATE-OBS, not the '
                      f'frame mid-time', n=n_retro)
    res = call(ctx, s, eph, [], env=ctx.env(), log=ctx.log.path, cwd=U)
    need_ok(res, 'S-G15 p3s2_ephem_filter.py', say=lambda: ctx.log.say('EPHEMERIS FILTER FAILED'))
    n_lst = _DP.ephem_time_fallbacks(res.tail)
    if n_lst:
        from exopipe.phase3.core import fallback      # WAVE2 B-30
        fallback.site('fallbacks.CATALOG_EPHEM_TIME_FALLBACK', f'p3s2_ephem_filter.py timed {n_lst} (cell, date) epoch(s) by '
                      f'date + LST hour as UT (no Phase-2 or frame DATE-OBS)', n=n_lst)
    ctx.log.say('split by Stokes: subbands and wideband groups')
    try:
        lines = split(U, tuple(split_cfg['wb']), split_cfg['sb'])
    except Exception as e:           # the production heredoc's failure was unchecked (:222-233); the step refuses
        refuse(f'S-G15 split failed: {type(e).__name__}: {e}')
    ctx.log.raw('\n'.join(lines))
    s.out(f'{U}/selected_all.csv', 'selection')
    for k in SPLIT_KEYS:
        s.out(f'{U}/selected_{k}.csv', 'selection')


def st_veto(ctx, s, plan):
    U = ctx.U
    script = ctx.opsc('leak_veto_sublst.py')
    lits = module_literals(script) if os.path.exists(script) else {}
    v = ctx.cfg.catalog_sublst_veto
    env_x = env_defaults_differ([(getattr(v, k), var, name) for k, var, name in VETO_ENV], lits)
    pairs = veto_pairs(U) if os.path.exists(f'{U}/master_catalog.csv') else []
    plan.append(f'leak veto: for (band, Stokes) of master_catalog.csv ({len(pairs) or "?"} pairs): {ctx.py["default"]} '
                f'{script} {U} {U}/leakveto {ctx.lst} <band> <S> {ctx.arm} >> leakveto.err' + (f'  env {env_x}' if env_x else ''))
    if ctx.dry:
        return
    s.code(script)
    s.inp(f'{U}/master_catalog.csv', f'{U}/master_catalog_asrun.csv')
    os.makedirs(f'{U}/leakveto', exist_ok=True)
    nfail = 0
    for band, st in pairs:
        # production: stderr -> leakveto.err, stdout tee'd into catalogs.log and grepped for 'kept candidates'; legacy.run
        # sends both to one log (leakveto.err) and the machine line is copied into catalogs.log
        res = call(ctx, s, script, [U, f'{U}/leakveto', ctx.lst, band, st, ctx.arm], env=ctx.env(**env_x),
                   log=f'{U}/leakveto.err', cwd=U)
        line = last_line(res.tail, 'kept candidates')
        if line:
            ctx.log.raw(line)
        if not res.ok or not line:
            ctx.log.say(f'VETO FAILED {ctx.lst} {band} {st}')
            nfail += 1
    n = len(os.listdir(f'{U}/leakveto'))
    ctx.log.say(f'veto tables: {n} written, {nfail} failed')
    if nfail:
        refuse(f'S-G16 leak veto: {nfail} of {len(pairs)} (band, Stokes) failed (see {U}/leakveto.err)')
    for band, st in pairs:
        s.out(f'{U}/leakveto/leakveto_{ctx.lst}_{band}_{st}.csv', 'leak_veto_table', run_dir=U)


def _build(ctx, s, plan, tag):
    U = ctx.U
    script, loc, rej = (ctx.opsc('build_source_catalogs_v3.py'), ctx.opsc('build_localizations.py'),
                        ctx.opsc('emit_rejected_record.py'))
    lv = '' if tag == 'noveto' else f'{U}/leakveto'
    cats = shared('catalog_localize', 'CATS', 'sublst').format(build=tag)
    py = ctx.py['default']
    plan.append(f'build ({tag}): for K in {", ".join(SPLIT_KEYS)} with >= 1 selected row: P3_LEAKVETO_DIR={lv!r} {py} {script} '
                f'selected_K.csv source_catalogs_{tag}_K v_deep_images.csv >> catalogs_{tag}_K.log')
    plan.append(f'localizations ({tag}): P3_LOC_CATS={cats!r} P3_LOC_OUT={U}/localizations_{tag} {py} {loc} {U}')
    if tag == 'noveto':
        plan.append(f'rejected-source record: P3_SERVED_CATALOGS={U}/master_catalog.csv {py} {rej} {U} 10min')
    if ctx.dry:
        return
    if tag == 'veto' and not os.path.isdir(lv):
        ctx.log.say('REFUSING buildv: P3_LEAKVETO_DIR unset')
        refuse(f'S-G17: build (veto) needs the leak-veto tables {lv} (run the veto stage)')
    s.code(script, loc, rej, ctx.opsc('source_localization.py'))
    s.inp(f'{U}/v_deep_images.csv', os.path.join(ctx.pt, 'pertie_detections.csv'))
    built = []
    for k in SPLIT_KEYS:
        n = count_lines(f'{U}/selected_{k}.csv') - 1 if os.path.exists(f'{U}/selected_{k}.csv') else 0
        if n < 1:
            ctx.log.say(f'{k}: no selected rows')
            continue
        ctx.log.say(f'{k} ({tag}): {n} selected rows -> source catalogs')
        d = f'source_catalogs_{tag}_{k}'
        os.makedirs(f'{U}/{d}', exist_ok=True)
        s.inp(f'{U}/selected_{k}.csv')
        res = call(ctx, s, script, [f'selected_{k}.csv', d, 'v_deep_images.csv'], env=ctx.env(P3_LEAKVETO_DIR=lv),
                   log=f'{U}/catalogs_{tag}_{k}.log', cwd=U)
        need_ok(res, f'S-G18 build_source_catalogs_v3.py {k} ({tag})', r'REFUS|Error',
                say=lambda: ctx.log.say(f'{k} ({tag}): build_source_catalogs_v3 FAILED (see catalogs_{tag}_{k}.log)'))
        built.append(d)
    ctx.log.say(f'source catalogues ({tag}) complete')
    res = call(ctx, s, loc, [U], env=ctx.env(P3_LEAKVETO_DIR=lv, P3_LOC_CATS=cats, P3_LOC_OUT=f'{U}/localizations_{tag}'),
               log=f'{U}/catalogs_{tag}_loc.log', cwd=U)
    need_ok(res, f'S-G19 build_localizations.py ({tag})', r'REFUS|Error',
            say=lambda: ctx.log.say(f'localizations ({tag}) FAILED (see catalogs_{tag}_loc.log)'))
    lp = f'{U}/localizations_{tag}/localizations.csv'
    if glob.glob(f'{U}/{cats}/*_sources_survivors.csv') and not (os.path.exists(lp) and os.path.getsize(lp) > 0):
        ctx.log.say(f'REFUSING: survivors exist but no localizations ({tag})')
        refuse(f'S-G19: survivors exist but no localizations ({tag})')
    ctx.log.say(f'localizations ({tag}): {count_lines(lp) - 1} identities -> {U}/localizations_{tag}')
    for d in built:
        s.out(f'{U}/{d}', 'source_catalog', run_dir=U)
    if os.path.exists(lp):
        s.out(lp, 'localization_table', run_dir=U, cats=cats)
    if tag == 'noveto':
        if not (os.path.exists(f'{U}/frame_times.csv') and os.path.getsize(f'{U}/frame_times.csv') > 0):
            ctx.log.say(f'REFUSING: no {U}/frame_times.csv for the rejected-source record')
            refuse(f'S-G20: no {U}/frame_times.csv for the rejected-source record')
        res = call(ctx, s, rej, [U, '10min'], env=ctx.env(P3_LEAKVETO_DIR=lv, P3_SERVED_CATALOGS=f'{U}/master_catalog.csv'),
                   log=ctx.log.path, cwd=U)
        need_ok(res, 'S-G20 emit_rejected_record.py', r'REFUS',
                say=lambda: ctx.log.say('REFUSING: rejected-source record failed (see catalogs.log)'))
        ctx.log.say(f'rejected-source record: {last_line(res.tail, "rejected rows")[:220]}')
        s.out(f'{U}/rejected.csv', 'rejected_record')


def st_build(ctx, s, plan):
    if 'noveto' in ctx.builds:
        return _build(ctx, s, plan, 'noveto')
    plan.append('build (noveto): not selected (--sublst-builds)')
    if not ctx.dry:
        raise _record().NoInput('build noveto not selected (--sublst-builds)')


def st_buildv(ctx, s, plan):
    if 'veto' in ctx.builds:
        return _build(ctx, s, plan, 'veto')
    plan.append('build (veto): not selected (--sublst-builds)')
    if not ctx.dry:
        raise _record().NoInput('build veto not selected (--sublst-builds)')


def compare_copy(U, cmp_catalog):
    """sublst_catalog_chain.sh:44-52: the small products to the comparison dir (cp -p; errors ignored as production), then
    the window-tie tables whole (rm -rf + cp -rp; the caller compares pertie_detections.csv byte for byte)."""
    os.makedirs(cmp_catalog, exist_ok=True)
    done = []
    for p in sorted(glob.glob(f'{U}/*.csv')) + sorted(glob.glob(f'{U}/catalogs*.log')) + [f'{U}/rejected_index.npz']:
        try:
            shutil.copy2(p, cmp_catalog)
            done.append(p)
        except OSError:
            pass
    for p in sorted(glob.glob(f'{U}/source_catalogs_*')) + [f'{U}/leakveto'] + sorted(glob.glob(f'{U}/localizations_*')):
        if os.path.isdir(p):
            try:
                shutil.copytree(p, os.path.join(cmp_catalog, os.path.basename(p)), copy_function=shutil.copy2,
                                dirs_exist_ok=True, symlinks=True)
                done.append(p)
            except (OSError, shutil.Error):
                pass
    dst = os.path.join(cmp_catalog, 'pertie')
    if os.path.lexists(dst):
        shutil.rmtree(dst)
    shutil.copytree(f'{U}/pertie', dst, copy_function=shutil.copy2, symlinks=True)
    done.append(f'{U}/pertie')
    return done


def files_equal(a, b):
    try:
        return os.path.getsize(a) == os.path.getsize(b) and sha1_file(a) == sha1_file(b)
    except OSError:
        return False


def st_test(ctx, s, plan):
    U, c = ctx.U, ctx.c
    pert, af = ctx.opsc('sublst_pertie.py'), ctx.opsc('astrom_frame.py')
    t_args = ['test', U, '--pixels', str(c.PERTIE_TEST_PIXELS), '--lightcurves', str(c.PERTIE_TEST_LIGHTCURVES)]
    a_args = ['test', str(c.ASTROM_FRAME_TEST_SCREENS)]
    pd = os.path.join(ctx.pt, 'pertie_detections.csv')
    plan.append(f'PERTIE-TEST: P3_TENMIN_PERTIE={pd} P3S2_ROOTS_CSV={ctx.roots_csv} {ctx.py["flow"]} {pert} ' + ' '.join(t_args))
    plan.append(f'ASTROM-FRAME-TEST: {ctx.py["default"]} {af} ' + ' '.join(a_args))
    plan.append(f'copy the small products to {ctx.cmp_catalog}; pertie_detections.csv byte-identical there (S-G22)')
    if ctx.dry:
        return
    s.code(pert, af)
    s.inp(pd)
    res = call(ctx, s, pert, t_args, 'flow', env=ctx.chain_env(P3_TENMIN_PERTIE=pd, P3S2_ROOTS_CSV=ctx.roots_csv),
               log=ctx.chain_log, cwd=U)
    line = last_line(res.tail, r'^PERTIE-TEST')
    s.u.check('S-G21.PERTIE-TEST', res.ok, value=res.rc, detail=line, cls='window_tie', path=pd)
    if not res.ok:
        ctx.chainlog.say(f'REFUSING: {line}')
        refuse(f'S-G21: {line or "PERTIE-TEST exit %d" % res.rc}')
    ctx.chainlog.say(line)
    res = call(ctx, s, af, a_args, 'default', env=ctx.chain_env(), log=ctx.chain_log, cwd=U)
    line = last_line(res.tail, r'^ASTROM-FRAME-TEST')
    s.u.check('S-G21.ASTROM-FRAME-TEST', res.ok, value=res.rc, detail=line)
    if not res.ok:
        ctx.chainlog.say(f'REFUSING: {line}')
        refuse(f'S-G21: {line or "ASTROM-FRAME-TEST exit %d" % res.rc}')
    ctx.chainlog.say(line)
    try:
        compare_copy(U, ctx.cmp_catalog)
    except (OSError, shutil.Error) as e:
        ctx.chainlog.say(f'REFUSING: the window-tie tables ({ctx.pt}) were not copied to the comparison dir')
        refuse(f'S-G22: the window-tie tables were not copied to {ctx.cmp_catalog} ({e})')
    b = os.path.join(ctx.cmp_catalog, 'pertie', 'pertie_detections.csv')
    if not files_equal(pd, b):
        ctx.chainlog.say(f'REFUSING: the window-tie tables ({ctx.pt}) were not copied to the comparison dir')
        refuse(f'S-G22: {b} differs from {pd}')
    s.out(ctx.cmp_catalog, 'compare_copy', source=U)


def st_compare(ctx, s, plan):
    script = ctx.opsc('sublst_compare_19h.py')
    old = os.path.join(ctx.compare_root, 'old')
    if not ctx.c.RUN_COMPARE:
        plan.append('compare: off (catalog_sublst.RUN_COMPARE)')
        if not ctx.dry:
            raise _record().NoInput('compare off (catalog_sublst.RUN_COMPARE = False)')
        return
    for lv in ('selected', 'noveto', 'veto'):
        plan.append(f'compare {lv} (non-gating): {ctx.py["default"]} {script} {old} {ctx.U} {ctx.tree} {ctx.cmp_new} {lv}')
    if ctx.dry:
        return
    s.code(script)
    wrote = []
    for lv in ('selected', 'noveto', 'veto'):
        res = call(ctx, s, script, [old, ctx.U, ctx.tree, ctx.cmp_new, lv], env=ctx.chain_env(), log=ctx.chain_log,
                   cwd=ctx.U)
        p = os.path.join(ctx.cmp_new, f'summary_{lv}.csv')
        if not res.ok:
            ctx.chainlog.say(f'compare {lv} failed')
            s.u.meta.setdefault('fallbacks', []).append(f'compare {lv} failed rc {res.rc} (logged only, B-24/B-30)')
        elif os.path.exists(p):
            wrote.append(p)
    for p in wrote:
        s.out(p, 'csv_table')
    if not wrote:
        raise _record().NoInput('no comparison summary written (every level failed; non-gating)')


def _slice_used_gb():
    for f in ('/sys/fs/cgroup/memory/user.slice/memory.usage_in_bytes', '/sys/fs/cgroup/user.slice/memory.current'):
        try:
            return int(int(open(f).read()) / 1e9)
        except (OSError, ValueError):
            continue
    return None


def st_reftie(ctx, s, plan):
    c = ctx.c
    script = ctx.opsc('sublst_tie.py')
    tdir = os.path.join(ctx.cmp_new, 'tie')
    work = ['--work', os.path.join(ctx.scratch, 'sublst_tie_work')] if ctx.scratch else []
    a = [ctx.U, ctx.lst, tdir] + work
    if not c.RUN_REFTIE:
        plan.append('reftie: off (catalog_sublst.RUN_REFTIE)')
        if not ctx.dry:
            raise _record().NoInput('reftie off (catalog_sublst.RUN_REFTIE = False)')
        return
    plan.append(f'reference tie: P3_REFASTRO_SHARDS={c.REFTIE_SHARDS} {ctx.py["py36"]} {script} ' + ' '.join(a)
                + f' > {tdir}/tie.log (starts while /user.slice < {c.REFTIE_MEMCAP_GB} GB)')
    if ctx.dry:
        return
    used = _slice_used_gb()
    if used is not None and used >= c.REFTIE_MEMCAP_GB:
        refuse(f'reftie: /user.slice holds {used} GB >= catalog_sublst.REFTIE_MEMCAP_GB {c.REFTIE_MEMCAP_GB} '
               f'(sublst_resume_from_tie.sh waits; the runner admission that replaces the wait is pending, K7)')
    s.code(script)
    os.makedirs(tdir, exist_ok=True)
    tl = os.path.join(tdir, 'tie.log')
    if os.path.exists(tl):
        os.replace(tl, f'{tl}.failed_{utc("%Y%m%dT%H%M%S")}')
    res = call(ctx, s, script, a, 'py36', env=ctx.chain_env(P3_REFASTRO_SHARDS=c.REFTIE_SHARDS), log=tl, cwd=ctx.U)
    need_ok(res, 'reftie: sublst_tie.py')
    s.u.note('7 tie: ' + last_line(res.tail, r'^registration'))
    s.out(os.path.join(tdir, 'tie_members.csv'), 'csv_table')


FUNCS = {'preflight': st_preflight, 'merge': st_merge, 'tier-rules': st_tier_rules, 'tie-pulsars': st_tie_pulsars,
         'conform': st_conform, 'fitpos': st_fitpos, 'select': st_select, 'tie': st_tie, 'veto': st_veto,
         'build': st_build, 'buildv': st_buildv, 'test': st_test, 'compare': st_compare, 'reftie': st_reftie}


# =====================================================================================================================
# interface for steps/catalog.py
# =====================================================================================================================

def add_arguments(p):
    """Tier-unique flags (prefixed --sublst-)."""
    g = p        # the dispatcher passes its 'tier sublst' argument group (item kind lst: ONE LST per invocation, --lsts 05h)
    g.add_argument('--sublst-arm', choices=('A', 'B', 'BI'), default=None,
                   help='search arm (default catalog_sublst.ARM = B, the served arm; A/BI are comparison arms)')
    g.add_argument('--sublst-builds', default=None, metavar='LIST',
                   help='catalogue builds, comma list of noveto,veto (default catalog_sublst.BUILDS; only noveto is served, '
                        'B-24 open)')


def run(args, run):
    """Run the selected stages of one LST, in order; the first stage that does not end done stops the chain (D10)."""
    ctx = Ctx(args, run)
    sel = select_stages(args, STAGES)
    first = STAGES.index(sel[0])
    where = 'production' if ctx.prod else 'non-production; side files under ' + os.path.join(ctx.U, '_side')
    print(f'catalog --tier sublst {ctx.unit}: U={ctx.U} ({where}); label {ctx.label}; tree {ctx.tree}; code {ctx.ops}, '
          f'{ctx.clv}; stages {",".join(sel)}', file=sys.stderr)
    if ctx.dry:
        for name in sel:
            plan = []
            FUNCS[name](ctx, Stage(ctx, name, DryUnit()), plan)
            print(f'[{ctx.unit}.{name}]')
            for ln in plan:
                print(f'  {ln}')
        if first > 0:
            prev = f'{ctx.unit}.{STAGES[first - 1]}'
            r = stage_record(ctx.U, prev)
            print(f'  --from {sel[0]}: needs the record {prev} done (now: {"missing" if r is None else r.get("status")}) '
                  f'and its outputs unchanged')
        return 0
    env_leaks(LEAKY_ENV)
    restored = check_from(ctx.U, ctx.unit, STAGES, first, ctx.recdir, IN_PLACE) if first > 0 else []
    lock = None
    if any(st in HEAVY for st in sel):
        lock = heavy_lock(ctx.lock)
        ok, line = mem_gate(ctx.ops, ctx.c.CHAIN_MEM_GB, ctx.c.CHAIN_MEM_WAIT_S, f'catalogue chain {ctx.sfx}', ctx.chain_log)
        if not ok:
            ctx.chainlog.say('REFUSING: no memory headroom (see MEM-GATE)')
            lock.close()
            with run.unit(f'{ctx.unit}.{sel[0]}', tier=TIER, lst=ctx.lst, arm=ctx.arm, stage=sel[0]) as u:
                u.refuse(f'S-G3 memory: {line or "node_mem_gate.sh refused"}')
            return None
    ctx.chainlog.say(f'CHAIN-START arm {ctx.arm} lst {ctx.lst} (exopipe3 catalog --tier sublst; stages {",".join(sel)})')
    try:
        for name in sel:
            with run.unit(f'{ctx.unit}.{name}', tier=TIER, lst=ctx.lst, arm=ctx.arm, stage=name) as u:
                st = Stage(ctx, name, u)
                i_ = STAGES.index(name)
                if name != sel[0] and STAGES[i_ - 1] not in sel:
                    # review 2026-10-13: a gap in --stages (e.g. conform,select without fitpos) is checked like a --from:
                    # the skipped stage's record must be done and current, else this stage would read outputs no record
                    # vouches for (DESIGN 11.5)
                    restored = check_from(ctx.U, ctx.unit, STAGES, i_, ctx.recdir, IN_PLACE)
                for n_ in restored:
                    u.note(n_)
                restored = []
                plan = []
                try:
                    FUNCS[name](ctx, st, plan)
                finally:
                    u.meta['plan'] = plan
                    st.close()
            if u.status != 'done' and not (u.status == 'no-input' and name in OPTIONAL):
                ctx.chainlog.say(f'REFUSING: {name} {u.status}: {u.reason}')
                break
        else:
            ctx.chainlog.say(f'CHAIN-END arm {ctx.arm}')
    finally:
        if lock is not None:
            lock.close()
    return None


if __name__ == '__main__':          # child entry points run through core/legacy.run (stdlib + in-place modules only)
    if len(sys.argv) >= 4 and sys.argv[1] == 'pcheck':
        sys.exit(pcheck_main(sys.argv[2], sys.argv[3]))
    sys.exit('usage: tier_sublst.py pcheck LST OPS_ROOT')
