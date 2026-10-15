"""Shared machinery of `exopipe3 calibrate`: table classes, versions, stage directories, record lookup, in-place helpers,
digests, and the install bundle (versioned file, config patch, draft ledger row).

Layout of one calibration (DESIGN 11.5: each sub-stage writes a FRESH directory and reads only the files the previous
sub-stage's record lists):

    <out>/calibrate/<class>/<tier>/<version>/<stage>/...      stage = a derive sub-stage, 'diff', 'install' or 'verify-<label>'
    <out>/_records/calibrate/<class>.<tier>.<action>.<version>.json      action = derive-<stage> | diff | install | verify

A stage directory that exists is never overwritten: --resume skips a stage whose record is current; --fresh moves the old
directory to <version>/_superseded/<stage>.<UTC> (nothing is deleted).
"""
import contextlib
import csv
import difflib
import glob
import hashlib
import importlib
import json
import os
import re
import shutil
import sys
import time

from exopipe.phase3.core.record import Refusal

# table class -> (implementing module, tiers, derive sub-stages per tier)
CLASSES = {
    'bright-parent': ('bright_parent', ('deep', 'sublst'), {'deep': ('inputs', 'curves', 'build'),
                                                            'sublst': ('inputs', 'curves', 'build')}),
    'multiples': ('bright_parent', ('deep',), {'deep': ('inputs', 'curves', 'build')}),
    'scale': ('fluxscale', ('deep',), {'deep': ('measure', 'table')}),
    'fluxcorr': ('fluxscale', ('deep',), {'deep': ('stageA', 'bandvec', 'table')}),
    'template-k': ('template_k', ('deep', 'sublst', 'snapshot'), {'deep': ('inject', 'summary'), 'sublst': ('inject', 'summary'),
                                                                  'snapshot': ()}),
}
# classes of the spec (plan/specs/calibrate.md 1.2) NOT implemented by this unit yet (plan/impl/calibrate.md section 6)
NOT_IMPLEMENTED = ('snapshot-curve', 'assoc-cp', 'untied-sys', 'forced-ceiling', 'completeness', 'pertie')
ACTIONS = ('derive', 'diff', 'install', 'verify')


def CFG():
    from exopipe.phase3.config import CFG as _C
    return _C


def utcnow():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def utcstamp():
    return time.strftime('%Y%m%dT%H%MZ', time.gmtime())


# ---------------------------------------------------------------------------------------------------------------------
# digests and atomic writes
# ---------------------------------------------------------------------------------------------------------------------

def digests(path):
    """{'md5', 'sha1', 'sha256'} of a file's bytes."""
    hs = {'md5': hashlib.md5(), 'sha1': hashlib.sha1(), 'sha256': hashlib.sha256()}
    with open(path, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            for h in hs.values():
                h.update(blk)
    return {k: h.hexdigest() for k, h in hs.items()}


def sha256(path):
    return digests(path)['sha256']


def _atomic(path, write):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, f'.{os.path.basename(path)}.tmp.{os.getpid()}')
    with open(tmp, 'w', newline='') as f:
        write(f)
    os.replace(tmp, path)
    return path


def write_json(path, obj, indent=1):
    """Atomic JSON write (indent=1 like every production table file)."""
    return _atomic(path, lambda f: json.dump(obj, f, indent=indent))


def write_text(path, text):
    return _atomic(path, lambda f: f.write(text))


def write_csv(path, fieldnames, rows):
    def w(f):
        wr = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction='ignore', lineterminator='\n')
        wr.writeheader()
        wr.writerows(rows)
    return _atomic(path, w)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def read_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def canon(obj):
    """The object as a JSON round trip gives it (tuples -> lists): tables are compared as parsed objects (A2)."""
    return json.loads(json.dumps(obj))


def num(x):
    """25.0 -> 25 (production writes elevation bands as ints)."""
    return int(x) if isinstance(x, float) and x.is_integer() else x


# ---------------------------------------------------------------------------------------------------------------------
# versions, stage directories, records
# ---------------------------------------------------------------------------------------------------------------------

_SAFE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._+-]*$')


def check_version(v):
    if not v or not _SAFE.match(v):
        raise ValueError(f'--version {v!r}: letters, digits and . _ + - only')
    return v


def class_root(out, cls, tier):
    return os.path.join(out, 'calibrate', cls, tier)


def version_dir(out, cls, tier, version):
    return os.path.join(class_root(out, cls, tier), version)


def unit_name(cls, tier, action, version):
    return f'{cls}.{tier}.{action}.{version}'


def record_path(out, unit):
    return os.path.join(out, '_records', 'calibrate', f'{unit}.json')


def load_record(out, unit):
    p = record_path(out, unit)
    if not os.path.exists(p):
        return None
    return read_json(p)


def versions(out, cls, tier):
    root = class_root(out, cls, tier)
    if not os.path.isdir(root):
        return []
    return sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)) and not d.startswith('_'))


def newest_done(out, cls, tier, action):
    """(version, record) of the newest done record of `action` for this class and tier, or (None, None)."""
    best = (None, None, '')
    for p in glob.glob(os.path.join(out, '_records', 'calibrate', f'{cls}.{tier}.{action}.*.json')):
        try:
            rec = read_json(p)
        except (OSError, ValueError):
            continue
        if rec.get('status') != 'done':
            continue
        fin = rec.get('finished') or ''
        if fin >= best[2]:
            v = os.path.basename(p)[len(f'{cls}.{tier}.{action}.'):-len('.json')]
            best = (v, rec, fin)
    return best[0], best[1]


def outputs_of(rec, cls=None):
    return [o['path'] for o in (rec or {}).get('outputs', []) if cls is None or o.get('class') == cls]


def require_done(out, unit, what):
    rec = load_record(out, unit)
    if rec is None or rec.get('status') != 'done':
        raise Refusal(f'{what}: no done record {record_path(out, unit)} '
                      f'(status {None if rec is None else rec.get("status")!r}) - run that stage first')
    return rec


def fresh_stage_dir(path, fresh):
    """Create the stage directory; an existing one refuses unless --fresh (then it is moved aside, never deleted)."""
    if os.path.exists(path):
        if not fresh:
            raise Refusal(f'{path} exists: a stage directory is never overwritten - pass --fresh (it is moved to '
                          f'_superseded/) or --resume (a current record is skipped)')
        sup = os.path.join(os.path.dirname(path), '_superseded')
        os.makedirs(sup, exist_ok=True)
        shutil.move(path, os.path.join(sup, f'{os.path.basename(path)}.{time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())}'))
    os.makedirs(path)
    return path


def legacy_code_current(rec):
    """(True, '') when every in-place file the record's children imported still has the recorded sha1."""
    for f, h in ((rec or {}).get('meta', {}).get('legacy_code') or {}).items():
        try:
            now = hashlib.sha1(open(f, 'rb').read()).hexdigest()[:12]
        except OSError:
            return False, f'in-place code missing: {f}'
        if now != h:
            return False, f'in-place code changed: {f}'
    return True, ''


# ---------------------------------------------------------------------------------------------------------------------
# in-place calls (refactor window, DESIGN 11.1)
# ---------------------------------------------------------------------------------------------------------------------

def ops_root():
    from exopipe.phase3.core import env
    return env.legacy_root('exopipe_ops')


def ops_script(key):
    return os.path.join(ops_root(), CFG().calibrate.LEGACY_SCRIPTS[key])


@contextlib.contextmanager
def legacy_env(env):
    """Set environment variables for an IN-PROCESS call of a legacy module that reads them; restore afterwards."""
    old = {k: os.environ.get(k) for k in env}
    try:
        for k, v in env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = str(v)
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def import_inplace(name, env=None, reload=False):
    """Import a clv15 / exopipe_ops module by name for an in-process call. sys.path is restored afterwards (legacy modules
    insert their own roots at import); ``env`` is set while the module executes (modules that read the environment at
    import); ``reload`` re-executes a module already imported (e.g. gate_inputs with another P3_PARENT_DIR)."""
    from exopipe.phase3.core import env as _env
    from exopipe.phase3.core.legacy import refuse_patch_path
    refuse_patch_path(name, 'calibrate.common.import_inplace')
    _env.legacy()
    saved = list(sys.path)
    try:
        with legacy_env(env or {}):
            if name in sys.modules and reload:
                return importlib.reload(sys.modules[name])
            return importlib.import_module(name)
    finally:
        sys.path[:] = saved


def run_legacy(u, target, args, env=None, log=None, cwd=None, module=False, timeout=None):
    """core.legacy.run with the step's interpreter; a refusal of the child refuses the unit, an error raises."""
    from exopipe.phase3.core import legacy
    C = CFG()
    full_env = {'OMP_NUM_THREADS': str(C.calibrate.THREADS)}
    full_env.update({k: v for k, v in (env or {}).items() if v is not None})
    res = legacy.run(target, args, python=C.calibrate.LEGACY_PYTHON, module=module, env=full_env,
                     cwd=cwd or ops_root(), log=log, unit=u, timeout=timeout)
    if res.refused:
        raise Refusal(f'{os.path.basename(str(target))}: {res.reason}')
    if not res.ok:
        raise RuntimeError(f'{os.path.basename(str(target))} failed: {res.reason}; log {log}; tail: {res.tail[-800:]}')
    return res


def parallel(calls, workers):
    """Run zero-argument callables concurrently (threads: each one waits on a child process); re-raise the first failure
    (a Refusal before any other error)."""
    from concurrent.futures import ThreadPoolExecutor
    if not calls:
        return []
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
        futs = [ex.submit(c) for c in calls]
        res, errs = [], []
        for f in futs:
            try:
                res.append(f.result())
            except Exception as e:      # noqa: BLE001 - collected and re-raised below
                errs.append(e)
    for e in errs:
        if isinstance(e, Refusal):
            raise e
    if errs:
        raise errs[0]
    return res


def need_file(path, what):
    if not path or not os.path.exists(path):
        raise Refusal(f'{what}: {path} not found')
    return path


# ---------------------------------------------------------------------------------------------------------------------
# install bundle: config patch, pointer consistency, draft ledger row
# ---------------------------------------------------------------------------------------------------------------------

def repo_root():
    import exopipe
    return os.path.dirname(os.path.dirname(os.path.abspath(exopipe.__file__)))


def config_sources():
    """The config source files: the package exopipe/phase3/config/ (__init__.py, shared.py, steps/<step>.py)."""
    here = os.path.join(repo_root(), 'exopipe', 'phase3', 'config')
    return [os.path.join(here, '__init__.py'), os.path.join(here, 'shared.py')] + sorted(
        p for p in glob.glob(os.path.join(here, 'steps', '*.py')) if os.path.basename(p) != '__init__.py')


def _section_spans():
    """{section: (source file, {KEY: (first line, last line)})} for every config section defined in the config package
    (exopipe/phase3/config/), from each file's syntax tree (1-based inclusive line numbers)."""
    import ast
    want = set(CFG().sections()) - {'site'}
    out = {}
    for f in config_sources():
        try:
            tree = ast.parse(open(f).read())
        except (OSError, SyntaxError):
            continue
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name in want and node.name not in out:
                spans = {}
                for st in node.body:
                    if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
                        spans[st.targets[0].id] = (st.lineno, getattr(st, 'end_lineno', st.lineno))
                out[node.name] = (f, spans)
    return out


def _strings_in(v):
    if isinstance(v, str):
        yield v
    elif isinstance(v, dict):
        for k, x in v.items():
            yield from _strings_in(k)
            yield from _strings_in(x)
    elif isinstance(v, (list, tuple, set, frozenset)):
        for x in v:
            yield from _strings_in(x)


PROVENANCE_SUFFIXES = ('_INPUTS', '_DERIVATION', '_DERIVATIONS', '_PREVIOUS', '_CONVERGENCE', '_RECORDS')


def config_patch(replace_strings=None, entry_edits=(), sections=None, scoped=()):
    """A unified diff (text) of the config source files that makes the config point at a new table.

    replace_strings: {old string: new string} - every quoted occurrence inside the assignment of any key whose value holds
                     the old string (paths, digests, version strings), in EVERY section (pointer consistency: the same
                     table is mirrored in tables.*, search_tables.*, masters_fluxscale.*, catalog.* during the window).
    entry_edits:     [(section, KEY, regex, replacement)] - an edit inside one key's assignment (e.g. one entry of
                     fitting.K_TEMPLATE); the regex must match exactly once.
    scoped:          [(section, KEY, {old: new})] - string replacements inside ONE key only (provenance keys: *_INPUTS,
                     *_PREVIOUS, ... are never touched by replace_strings, they name what each installed table was made from).
    Returns (diff text, [(section, KEY)] changed). The patch is never applied by the step (DESIGN 11.4: by a commit)."""
    replace_strings = {k: v for k, v in (replace_strings or {}).items() if k and k != v}
    spans = _section_spans()
    files, changed = {}, []

    def _edit(f, a, b, fn):
        lines = files.setdefault(f, open(f).read().split('\n'))
        text = '\n'.join(lines[a - 1:b])
        new = fn(text)
        if new.count('\n') != text.count('\n'):
            raise Refusal(f'config patch: an edit of {f}:{a}-{b} would change its line count')
        lines[a - 1:b] = new.split('\n')
        return new != text

    for sec, (f, keys) in spans.items():
        if sections is not None and sec not in sections:
            continue
        cls = CFG()._section(sec)
        for key, (a, b) in keys.items():
            if not key.isupper() or key in ('RULED', 'PROTECTED_KEYS') or key.endswith(PROVENANCE_SUFFIXES):
                continue
            hits = [s for s in set(_strings_in(getattr(cls, key, None))) if s in replace_strings]
            if not hits:
                continue

            def _rep(text, hits=hits):
                for old in hits:
                    for q in ("'", '"'):
                        text = text.replace(f'{q}{old}{q}', repr(replace_strings[old]))
                return text
            if _edit(f, a, b, _rep):
                changed.append((sec, key))
    for sec, key, mapping in scoped:
        if sec not in spans or key not in spans[sec][1]:
            raise Refusal(f'config patch: no assignment of {sec}.{key} in the package config sources')
        f, (a, b) = spans[sec][0], spans[sec][1][key]

        def _rep2(text, mapping=mapping):
            for old, newv in mapping.items():
                if old == newv:
                    continue
                for q in ("'", '"'):
                    text = text.replace(f'{q}{old}{q}', repr(newv))
            return text
        if _edit(f, a, b, _rep2):
            changed.append((sec, key))
    for sec, key, rx, rep in entry_edits:
        if sec not in spans or key not in spans[sec][1]:
            raise Refusal(f'config patch: no assignment of {sec}.{key} in the package config sources')
        f, (a, b) = spans[sec][0], spans[sec][1][key]
        count = []

        def _sub(text, rx=rx, rep=rep, count=count):
            new, n = re.subn(rx, rep, text)
            count.append(n)
            return new
        _edit(f, a, b, _sub)
        if count != [1]:
            raise Refusal(f'config patch: {sec}.{key}: the edit {rx!r} matched {count[0]} times (expected once)')
        changed.append((sec, key))
    out = []
    root = repo_root()
    for f, lines in sorted(files.items()):
        old = open(f).read().split('\n')
        new = lines
        rel = os.path.relpath(f, root)
        out.extend(difflib.unified_diff([x + '\n' for x in old], [x + '\n' for x in new], f'a/{rel}', f'b/{rel}'))
    return ''.join(out), changed


def pointer_replacements(old_file, new_file, old_path, new_path, extra=None):
    """{old: new} for a table moving from old_file (bytes; at old_path in config) to new_file (at new_path): the path and
    each digest kind (md5, sha1, sha256) that config may record."""
    rep = {old_path: new_path}
    if old_file and os.path.exists(old_file):
        do, dn = digests(old_file), digests(new_file)
        for k in do:
            rep[do[k]] = dn[k]
    rep.update(extra or {})
    return rep


def ledger_draft(path, cls, tier, version, files, description, ruling_id, words):
    """One draft row in the shape of clv15/fix_ledger.csv (status 'draft'): written for the session and Gregg, never
    appended to the ledger by the step."""
    C = CFG()
    fid = f'CALIBRATE-{cls}-{tier}-{version}-INSTALL'.upper().replace('_', '-')
    row = {'date': time.strftime('%Y-%m-%dT%H:%MZ', time.gmtime()), 'fix_id': fid, 'files': '; '.join(files),
           'description': description,
           'authorized_by': f'{ruling_id}' + (f' ("{words}")' if words else ''), 'status': 'draft'}
    write_csv(path, C.calibrate.LEDGER_DRAFT_COLUMNS, [row])
    return row


def check_ruling(args):
    """Gates G18/G22: install needs --install-ruling <ledger ID> (a row of paths.FIX_LEDGER)."""
    C = CFG()
    rid = getattr(args, 'install_ruling', None)
    if not rid:
        if C.calibrate.INSTALL_RULING_REQUIRED:
            raise Refusal('install needs --install-ruling <ledger ID> naming Gregg\'s ruling for this table (gates G18/G22; '
                          'calibrate.INSTALL_RULING_REQUIRED)')
        return None
    from exopipe.phase3.core import ledger
    if not ledger.has_id(rid, C.paths.FIX_LEDGER):
        raise Refusal(f'--install-ruling {rid}: no row with that fix_id in {C.paths.FIX_LEDGER} (gate G22)')
    return rid


def diff_gate(out, cls, tier, version, cand_sha, installed_sha=None):
    """Gate G22: a done diff record of the same candidate sha256 (calibrate.DIFF_REQUIRED_FOR_INSTALL) - and, for a table
    file, made against the installed file as it is NOW (a diff against another file, or before the installed table changed,
    is stale); and no derive / diff record of this version overrides a RULED key without a ruling (DESIGN 11.4: install
    refuses a product whose record shows such an override)."""
    from exopipe.phase3.core.record import unruled_overrides
    bad = []
    for p in sorted(glob.glob(os.path.join(out, '_records', 'calibrate', f'{cls}.{tier}.*.{version}.json'))):
        act = os.path.basename(p)[len(f'{cls}.{tier}.'):-len(f'.{version}.json')]
        if act.startswith('derive-') or act == 'diff':
            try:
                rec_ = read_json(p)
            except (OSError, ValueError):
                continue
            bad += [f"{act}: {o.get('key')} ({o.get('authority')})" for o in unruled_overrides(rec_)]
    if bad:
        raise Refusal(f'install: the records of {cls} {tier} {version} override ruled keys without a ruling: {bad[:5]} '
                      f'(re-derive with --ruling-for SECTION.KEY=<ledger ID>, or without the override; gate G22, DESIGN 11.4)')
    if not CFG().calibrate.DIFF_REQUIRED_FOR_INSTALL:
        return None
    rec = load_record(out, unit_name(cls, tier, 'diff', version))
    if rec is None or rec.get('status') != 'done':
        raise Refusal(f'install: no done diff record for {cls} {tier} {version} (run `calibrate {cls} diff` first; gate G22)')
    if (rec.get('meta') or {}).get('candidate_sha256') != cand_sha:
        raise Refusal(f'install: the diff record compared candidate sha256 {(rec.get("meta") or {}).get("candidate_sha256")} '
                      f'but the candidate is now {cand_sha} (gate G22)')
    if installed_sha is not None:
        ds = outputs_of(rec, 'calibration_diff')
        got = read_json(ds[0]).get('installed_sha256') if ds and os.path.exists(ds[0]) else None
        if got != installed_sha:
            raise Refusal(f'install: the diff compared against installed sha256 {got}, the installed table is now '
                          f'{installed_sha}: the diff is stale - run diff again (gate G22)')
    return record_path(out, unit_name(cls, tier, 'diff', version))


# ---------------------------------------------------------------------------------------------------------------------
# the per-invocation context handed to the class modules
# ---------------------------------------------------------------------------------------------------------------------

class Ctx:
    """One `calibrate <table> <action>` invocation for one tier: args, run, config, version, paths, unit helpers."""

    def __init__(self, args, run, cls, tier, action, version):
        self.args, self.run, self.cls, self.tier, self.action = args, run, cls, tier, action
        self.C = CFG()
        self.out = run.out_root
        self.version = version
        self.vdir = version_dir(self.out, cls, tier, version)
        self.dry = run.dry_run
        self.stages = []            # derive sub-stages selected in this invocation (set by steps/calibrate._one)

    def unit_name(self, action=None):
        return unit_name(self.cls, self.tier, action or self.action, self.version)

    def record_of(self, action):
        return load_record(self.out, self.unit_name(action))

    def stage_dir(self, stage):
        return os.path.join(self.vdir, stage)

    def skip_if_current(self, action):
        """--resume: (True, why) and a 'skipped-valid' unit when the stage's record is current (inputs, code incl. the
        in-place children's files, config hash, outputs passing their content tests now)."""
        if not self.run.resume or self.dry:
            return False
        name = self.unit_name(action)
        ok, why = self.run.resume_check(name, inputs=None, admitted=None)
        if ok:
            ok, why2 = legacy_code_current(load_record(self.out, name))
            why = why2 or why
        if not ok:
            print(f'{name}: not resumed ({why})', file=sys.stderr)
            return False
        u = self.run.unit(name, cls=self.cls, tier=self.tier, version=self.version)
        u.status, u.reason = 'skipped-valid', why
        print(f'{name}: skipped ({why})', file=sys.stderr)
        return True

    def say(self, msg):
        print(f'calibrate {self.cls} {self.tier} {self.action} {self.version}: {msg}', file=sys.stderr, flush=True)

    def workers(self, default):
        return int(self.args.workers or default)


def verify(ctx, targets):
    """`calibrate <table> verify`: run every content test of each target's class, record each result as a check (a failure
    refuses the unit), write verify_report.json (class calibration_verify). targets = [(path, label, product class)]."""
    from exopipe.phase3.core import validate
    plan = [{'path': p, 'label': lab, 'class': c, 'exists': bool(p) and os.path.exists(p)} for p, lab, c in targets]
    if ctx.dry:
        return {'targets': plan}
    label = targets[0][1] if len(targets) == 1 else 'set'
    action = f'verify-{label}'
    d = os.path.join(ctx.vdir, action)
    if os.path.exists(d):
        sup = os.path.join(ctx.vdir, '_superseded')
        os.makedirs(sup, exist_ok=True)
        shutil.move(d, os.path.join(sup, f'{action}.{time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())}'))
    os.makedirs(d)
    with ctx.run.unit(ctx.unit_name(action), cls=ctx.cls, tier=ctx.tier, version=ctx.version, stage=action) as u:
        rep = {'class': ctx.cls, 'tier': ctx.tier, 'utc': utcnow(), 'targets': []}
        for p, lab, c in targets:
            if not p or not os.path.exists(p):
                u.check(f'{c}.exists', False, value=p, detail=f'{lab}: file not found', cls=c, path=p or None)
                rep['targets'].append({'path': p, 'label': lab, 'class': c, 'sha256': None, 'results': []})
                continue
            u.add_input(p)
            res = validate.run(c, p)
            pin = pinned_sha(p)
            if pin:                      # gate G19 for every pinned table (scale, fluxcorr, band vector, ...): sha256 = its pin
                key, want = pin
                got = sha256(p)
                res.append({'name': 'calibrate.sha_pinned', 'passed': got == want, 'value': got[:16],
                            'detail': f'tables.{key}_SHA256 {"=" if got == want else "!="} {want[:16]} (gate G19)'})
            for r in res:
                u.check(r['name'], r['passed'], value=r['value'], detail=r['detail'], cls=c, path=p)
            rep['targets'].append({'path': p, 'label': lab, 'class': c, 'sha256': sha256(p), 'results': res})
        rp = write_json(os.path.join(d, 'verify_report.json'), rep)
        u.add_output(rp, 'calibration_verify')
    return None


def pinned_sha(path):
    """(tables key, pinned sha256) when ``path`` is the file a `tables.<KEY>` names and `tables.<KEY>_SHA256` exists; else None."""
    T = CFG().tables
    rp = os.path.realpath(path)
    for k, v in T.items():
        if isinstance(v, str) and v.startswith('/') and os.path.realpath(v) == rp and hasattr(T, k + '_SHA256'):
            return k, getattr(T, k + '_SHA256')
    return None


def code_literals(strings):
    """[(file:line, string)] of package source lines (exopipe/phase3, config sources excluded: they are patched) that hold
    one of the strings - consumers that read a table or version by a LITERAL and need a commit with the install."""
    hits = []
    root = os.path.join(repo_root(), 'exopipe', 'phase3')
    strings = [x for x in strings if x and len(x) >= 8]
    for dp, dn, fn in os.walk(root):
        dn[:] = [x for x in dn if x != '__pycache__' and not (os.path.basename(dp) == 'lib' and x == 'calibrate')
                 and not (dp == root and x == 'config')]          # the config sources are patched, not scanned
        for n in fn:
            if not n.endswith('.py'):
                continue
            p = os.path.join(dp, n)
            try:
                for i, ln in enumerate(open(p, errors='replace'), 1):
                    for x in strings:
                        if x in ln:
                            hits.append((f'{os.path.relpath(p, repo_root())}:{i}', x))
            except OSError:
                continue
    return hits
