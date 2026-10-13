"""Shared helpers of `exopipe3 forced`: stage selection, the record chain between stages, resume, the one legacy call
(core.legacy.run, decision 1), host placement of node-local stages, rsync between hosts, and the refusal of config overrides
the in-place code would ignore. Used by lib/forced/tier_deep.py and tier_snapshot.py."""
import csv
import hashlib
import os
import re
import socket
import subprocess

from exopipe.phase3.core import record
from exopipe.phase3.core.record import Refusal

# Config keys the WRAPPER acts on (paths it passes, stages it runs, processes it starts). Every other key of the forced
# sections describes a literal of the in-place scripts (or a pinned input): an override would change nothing in the
# products, so check_overrides refuses it. fitting.* and astrometry.* are used in place by the clv15/exopipe_ops modules
# with their own values: refused as well while the forced chain runs in place.
STEERING = {
    'forced.STAGES', 'forced.DEFAULT_STAGES', 'forced.EPOCH_STAGES', 'forced.INPLACE_SCRIPTS', 'forced.INPLACE_GRANDCHILDREN',
    'forced.PYTHON_KEY', 'forced.NODE_LOCAL_RE', 'forced.LOG_TAIL_CHARS',
    'forced_panels.EVAL_RECORD', 'forced_panels.EVAL_TARGETS', 'forced_panels.QUEUE', 'forced_panels.VERIFY_OK',
    'forced_panels.TESS_OK',
    'forced_catflux.CATALOG_GLOB', 'forced_catflux.SURVIVORS_GLOB', 'forced_catflux.DONE_RE',
    'forced_catflux.RECHECK_ROWS', 'forced_catflux.RECHECK_SEED', 'forced_catflux.RECHECK_TOL_JY',
    'forced_snapshot.WORK_ROOT', 'forced_snapshot.REVIEW_SUBDIR_UNDER_OUT', 'forced_snapshot.EVAL_SB',
    'forced_snapshot.EVAL_WB', 'forced_snapshot.ZONE_FLAGS', 'forced_snapshot.EPOCH_NAME', 'forced_snapshot.NPROC',
    'forced_snapshot.OK_LINES', 'forced_snapshot.REFASTRO_SCRATCH_PRODUCTION', 'forced_snapshot.LOC_COPY_BACK',
    'forced_snapshot.FLOW_ENV',
}
OWN_SECTIONS = ('forced', 'forced_panels', 'forced_catflux', 'forced_snapshot', 'forced_assoc')
IN_PLACE_SHARED = ('fitting', 'astrometry')


def cfg():
    from exopipe.phase3.config import CFG
    return CFG


def check_overrides(c=None):
    """G37 in-place config truth: refuse a --set/--config of a value the in-place scripts carry as a literal (it would not
    change the products) or of fitting.* / astrometry.* (the in-place modules use their own values)."""
    c = c or cfg()
    bad = []
    for o in c.overrides:
        key = o[0].split('[', 1)[0]
        sec = key.split('.', 1)[0]
        if (sec in OWN_SECTIONS and key not in STEERING) or sec in IN_PLACE_SHARED:
            bad.append(o[0])
    if bad:
        raise Refusal('G37 in-place config truth: %s describe(s) literals of the production scripts the forced chain runs in '
                      'place (DESIGN 11.1); an override would not change the products. Change them at absorption, not '
                      'by --set/--config' % ', '.join(bad))


def plan(text):
    """One plan line (--dry-run prints the plan and writes nothing)."""
    print('PLAN forced: %s' % text, flush=True)


def selected_stages(args, tier):
    """Stages to run, in pipeline order: --stages (any order given), else the tier's default list (the per-hour list with
    --epochs), cut by --from / --until."""
    f = cfg().forced
    allst = list(f.STAGES[tier])
    if args.stages:
        names = [s.strip() for s in args.stages.split(',') if s.strip()]
        unknown = [s for s in names if s not in allst]
        if unknown:
            raise ValueError('unknown %s stage(s) %s; stages: %s' % (tier, unknown, ', '.join(allst)))
    else:
        names = list((f.EPOCH_STAGES if getattr(args, 'epochs', None) else f.DEFAULT_STAGES)[tier])
    seq = [s for s in allst if s in names]
    for flag, val in (('--from', args.stage_from), ('--until', args.stage_until)):
        if val and val not in seq:
            raise ValueError('%s %s is not among the selected stages %s' % (flag, val, seq))
    if args.stage_from:
        seq = seq[seq.index(args.stage_from):]
    if args.stage_until:
        seq = seq[:seq.index(args.stage_until) + 1]
    if not seq:
        raise ValueError('no stage selected')
    return seq


def epochs(args):
    """[(lst, date)] of the common --epochs selector (core.cells item kind 'epoch'), or []."""
    from exopipe.phase3.core import cells as C
    if not getattr(args, 'epochs', None):
        return []
    return [C.parse_epoch(e) for e in C.expand_epochs(args.epochs)]


# ---------------------------------------------------------------------------------------------------------------------
# record chain and resume
# ---------------------------------------------------------------------------------------------------------------------

def load_done(out_root, step, unit, what):
    """The done record ``unit`` of ``step`` under ``out_root``; refuses when absent or not done (DESIGN 11.5: a stage reads
    only the files the previous stage's record lists)."""
    rec = record.load_record(out_root, step, unit)
    if rec is None:
        raise Refusal('no record of %s (%s) under %s' % (what, unit, os.path.join(out_root, '_records', step)))
    if rec.get('status') != 'done':
        raise Refusal('%s record %s is %s (%s), not done' % (what, unit, rec.get('status'), (rec.get('reason') or '')[:200]))
    return rec


def outputs_of(rec, cls=None, basename=None):
    return [o['path'] for o in rec.get('outputs', [])
            if (not cls or o.get('class') == cls) and (not basename or os.path.basename(o['path']) == basename)]


def _sha1(path):
    try:
        with open(path, 'rb') as f:
            return hashlib.sha1(f.read()).hexdigest()[:12]
    except OSError:
        return None


def legacy_code_changed(rec):
    """The first legacy file whose sha1 differs from the record's meta.legacy_code (core.legacy.run's child fingerprint plus
    the grandchildren this step adds), or None. core.record.resume_check compares only the step process's own modules."""
    for p, h in ((rec.get('meta') or {}).get('legacy_code') or {}).items():
        if _sha1(p) != h:
            return p
    return None


def resumed(run, name, inputs, admitted=None, **meta):
    """--resume: True (and a 'skipped-valid' unit in the run summary; the done record stays as it is) when the record of
    ``name`` is done, its inputs, admitted nights, code (the step's modules AND the legacy code its children ran) and config
    are unchanged, and its outputs pass their content tests now."""
    if not run.resume:
        return False
    ok, why = run.resume_check(name, inputs=inputs, admitted=admitted)
    if ok:
        changed = legacy_code_changed(record.load_record(run.out_root, run.step, name) or {})
        if changed:
            ok, why = False, 'legacy code changed: %s' % changed
    if not ok:
        print('resume %s: not skipped (%s)' % (name, why), flush=True)
        return False
    u = run.unit(name, **meta)
    u.status, u.reason = 'skipped-valid', why
    print('resume %s: skipped (%s)' % (name, why), flush=True)
    return True


def records_log(run, unit):
    """Log of a unit's legacy calls: <out>/_records/forced/logs/<unit>.log (production wrote these to ad-hoc log files)."""
    return os.path.join(run.out_root, '_records', run.step, 'logs', '%s.log' % unit)


# ---------------------------------------------------------------------------------------------------------------------
# the legacy call (core.legacy.run, decision 1)
# ---------------------------------------------------------------------------------------------------------------------

def legacy_ops(rel=''):
    from exopipe.phase3.core import env
    try:
        root = env.legacy_root('exopipe_ops')
    except KeyError as e:
        raise Refusal(str(e))
    return os.path.join(root, rel) if rel else root


def script(key):
    """Path of the in-place exopipe_ops script forced.INPLACE_SCRIPTS[key] (refuses when absent)."""
    p = legacy_ops(cfg().forced.INPLACE_SCRIPTS[key])
    if not os.path.exists(p):
        raise Refusal('in-place script %s not found' % p)
    return p


class Call:
    """Result of one legacy call: ``res`` (core.legacy.LegacyResult) and ``text`` (this call's whole log section)."""

    def __init__(self, res, text):
        self.res, self.text = res, text
        self.rc, self.ok, self.refused = res.rc, res.ok, res.refused

    def lines(self):
        return self.text.splitlines()

    def has(self, s):
        return self.rc == 0 and s in self.text

    def line(self, s):
        return next((l for l in self.lines() if s in l), '')

    def why(self):
        m = re.search(r'(REFUS(?:ED|ING)[^\n]*)', self.text)
        if m:
            return m.group(1)[:600]
        return (self.res.reason or 'exit %d' % self.rc) + ': ' + self.text[-cfg().forced.LOG_TAIL_CHARS // 4:]


def call(u, target, args, python, log, env=None, cwd=None, stage=None):
    """core.legacy.run(target, args) with the site interpreter ``python``, stdout+stderr appended to ``log``; adds the sha1
    of the grandchild scripts of ``stage`` (forced.INPLACE_GRANDCHILDREN) to u.meta['legacy_code']. Returns a Call."""
    from exopipe.phase3.core import legacy
    os.makedirs(os.path.dirname(os.path.abspath(log)), exist_ok=True)
    off = os.path.getsize(log) if os.path.exists(log) else 0
    res = legacy.run(target, [str(a) for a in args], python=python, env=env or {}, cwd=cwd, log=log, unit=u)
    with open(log, 'rb') as f:
        f.seek(off)
        text = f.read().decode('utf-8', 'replace')
    if u is not None:
        code = u.meta.setdefault('legacy_code', {})
        code[os.path.abspath(target)] = _sha1(target)
        for rel in (cfg().forced.INPLACE_GRANDCHILDREN.get(stage, ()) if stage else ()):
            p = legacy_ops(rel)
            if os.path.exists(p):
                code[p] = _sha1(p)
        u.add_input(target)
    return Call(res, text)


# ---------------------------------------------------------------------------------------------------------------------
# hosts (node-local data) and rsync / ssh (runner.SSH_PREFIX_ENV, CLAUDE.md 8)
# ---------------------------------------------------------------------------------------------------------------------

def hostname():
    return socket.gethostname().split('.')[0]


def is_node_local(path):
    return bool(re.match(cfg().forced.NODE_LOCAL_RE, os.path.abspath(path)))


def host_of(path):
    """The host whose site NODE_LOCAL_ROOTS hold ``path`` (also after resolving symlinks), or None (shared / unknown)."""
    roots = getattr(cfg().site, 'NODE_LOCAL_ROOTS', None) or {}
    cands = {os.path.abspath(path)}
    try:
        cands.add(os.path.realpath(path))
    except OSError:
        pass
    for host, rr in roots.items():
        for r in rr:
            r = r.rstrip('/')
            if any(x == r or x.startswith(r + '/') for x in cands):
                return host
    return None


def require_host(path, what):
    """Refuse unless ``path`` is readable on this host (node-local data of another host is not)."""
    h = host_of(path)
    if h and h != hostname():
        raise Refusal('%s (%s) lives on %s, this is %s: run this stage there (exopipe3 run submit --nodes %s:1 -- forced '
                      '...)' % (what, path, h, hostname(), h))
    if not os.path.exists(path):
        raise Refusal('%s not found: %s (on %s)' % (what, path, hostname()))


def refuse_production_target(path, what):
    """rsync and the legacy children write outside this process: while paths.PRODUCTION_WRITES is False a target under a
    production root is refused here (the cli guard covers --out only)."""
    c = cfg()
    if not c.paths.PRODUCTION_WRITES and c.production_root_of(path) is not None:
        raise Refusal('%s %s is under the production root %s and paths.PRODUCTION_WRITES is False (refactor window)'
                      % (what, path, c.production_root_of(path)))


def _ssh_env():
    env = dict(os.environ)
    env.update(cfg().runner.SSH_PREFIX_ENV)
    return env


def remote(host):
    return '' if host in (None, '', hostname()) else '%s:' % host


def rsync(src, dst, src_host=None, dst_host=None, opts=('-a',), log=None):
    """rsync src -> dst (directories end with '/'), either side optionally on another host. Returns (rc, output)."""
    argv = ['rsync'] + list(opts) + [remote(src_host) + src, remote(dst_host) + dst]
    p = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, env=_ssh_env())
    if log:
        os.makedirs(os.path.dirname(os.path.abspath(log)), exist_ok=True)
        with open(log, 'a') as fh:
            fh.write('$ %s\n%s\n' % (' '.join(argv), p.stdout))
    return p.returncode, p.stdout


def path_exists_on(path, host):
    if host in (None, '', hostname()):
        return os.path.lexists(path)
    p = subprocess.run(['ssh', '-n', host, 'test', '-e', path], env=_ssh_env(), stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
    return p.returncode == 0


def nonrecord_entries(d):
    """Entries of directory ``d`` other than the run records (_records/): a product directory 'exists' only with these."""
    try:
        return [x for x in os.listdir(d) if x != '_records']
    except OSError:
        return []


def roots_cells(path=None):
    """{cell: coadd root} of forced_catflux.ROOTS_CSV (the root map p3s2_frames resolves Stokes I fallback frames with)."""
    out = {}
    with open(path or cfg().forced_catflux.ROOTS_CSV) as f:
        for line in f:
            p = line.strip().split(',')
            if len(p) >= 3 and p[0]:
                out[p[0]] = p[2]
    return out


def write_rows(path, rows, fieldnames):
    """CSV written as the production scripts write theirs (csv.DictWriter, newline=''), tmp + rename."""
    tmp = '%s.tmp.%d' % (path, os.getpid())
    with open(tmp, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)
