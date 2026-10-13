"""Shared orchestration of `exopipe3 photometry` (steps/photometry.py): run-level gates, engine resolution, stage selection,
the in-place call wrapper, record helpers and the dry-run plan printer. No result-changing constant lives here (config
sections photometry*); the engines run IN PLACE through core/legacy.run (DESIGN 11.1)."""
import hashlib
import os
import socket
import sys

from exopipe.phase3.core import record
from exopipe.phase3.core.record import NoInput, Refusal


def cfg():
    from exopipe.phase3.config import CFG
    return CFG


def hostname():
    return socket.gethostname().split('.')[0]


def plan(run, text):
    """One line of the plan (--dry-run prints the plan; a real run prints it too, as its log)."""
    print('PLAN %s' % text, flush=True)


def sha1_file(path):
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------------------------------
# run-level gates
# ---------------------------------------------------------------------------------------------------------------------

def gate_env():
    """PHOT-ENV: the in-place modules and their launchers read switches from the environment (spec 7.2); core/legacy.run
    clears inherited P3_* only, so any other forbidden prefix in the caller's environment refuses."""
    pre = tuple(cfg().photometry.ENV_FORBIDDEN_PREFIXES)
    bad = sorted(k for k in os.environ if k.startswith(pre))
    if bad:
        raise Refusal('PHOT-ENV: the caller environment sets %s (prefixes %s are read by the in-place engines or their '
                      'launchers; unset them - every engine input is a flag or a config key)' % (', '.join(bad), pre))


def sample_paths():
    c = cfg()
    return {t: os.path.join(c.paths.CLV15_ROOT, c.photometry.SAMPLE_FILES[t]) for t in c.photometry.SAMPLES_MEASURED}


def gate_samples():
    """PHOT-SAMPLE-PIN: each measured sample list must be the file config pins (sha1). The engines read these files by
    literal path; a rebuilt list needs a config update (B-58 lean shape, pure: same files, same values)."""
    c = cfg()
    out = {}
    for t, p in sample_paths().items():
        if not os.path.exists(p):
            raise Refusal('PHOT-SAMPLE-PIN: sample list %s missing (%s)' % (t, p))
        h = sha1_file(p)
        if h != c.photometry.SAMPLE_SHA1[t]:
            raise Refusal('PHOT-SAMPLE-PIN: %s sha1 %s != photometry.SAMPLE_SHA1[%s] %s - the list changed; update the '
                          'pin in config (by a commit) once the new list is accepted' % (p, h[:12], t,
                                                                                         c.photometry.SAMPLE_SHA1[t][:12]))
        out[t] = p
    return out


def gate_tier(tier):
    if tier not in cfg().photometry.TIERS_WITH_ENGINE:
        raise Refusal('PHOT-SUBLST-NO-ENGINE: no production sub-LST target photometry exists (the 19h pilot '
                      'sublst_forced_phot.py / sublst_forced_eval.py is one-off and no launcher runs it; ledger '
                      'R-FORCED-PHOTOMETRY-ELEMENTS-APPROVED-V-FAINT-PARENT-ENTRIES and R-FORCED-THRESHOLDS-PROVISIONAL-AND-'
                      'VETO-SCOPE notes) - tier %s is refused' % tier)


def gate_pinned_inputs(paths):
    """Pinned inputs the engines read by literal path must exist (a missing one would fail inside the engine)."""
    miss = [p for p in paths if p and not os.path.exists(p)]
    if miss:
        raise Refusal('pinned input missing: %s' % ', '.join(miss))


# ---------------------------------------------------------------------------------------------------------------------
# engines (in place, core/legacy.run)
# ---------------------------------------------------------------------------------------------------------------------

def engine(key):
    """Absolute path of the in-place engine photometry.ENGINE[key]: exactly one hit among the site LEGACY_CODE_ROOTS."""
    c = cfg()
    name = c.photometry.ENGINE[key]
    hits = [os.path.join(r, name) for r in c.site.LEGACY_CODE_ROOTS if os.path.isfile(os.path.join(r, name))]
    if len(hits) != 1:
        raise Refusal('engine %s: %d files named %s in the site LEGACY_CODE_ROOTS %s (exactly one expected)'
                      % (key, len(hits), name, c.site.LEGACY_CODE_ROOTS))
    return hits[0]


def python_key(tier):
    return cfg().photometry.ENGINE_PYTHON[tier]


def engine_env(name, threads=None):
    env = dict(cfg().photometry.ENV.get(name, {}))
    if threads:
        for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            env[k] = str(threads)
    return env


def logs_dir(run):
    return os.path.join(run.out_root, '_records', run.step, 'logs')


def call(u, key, args, tier, env_name, run, log_name, env_extra=None, threads=None):
    """Run engine ``key`` with ``args`` in a child process (core/legacy.run): site interpreter of the tier, cwd = the
    engine's own root (production `cd $OPS`), inherited P3_* cleared, config environment, parity guard inherited, the
    child's code fingerprint added to the unit record AND every fingerprinted file added as an input (so --resume notices an
    edit of an in-place module, spec blocker 21). Returns (result, the log text of this call)."""
    from exopipe.phase3.core import legacy
    script = engine(key)
    env = engine_env(env_name, threads)
    env.update(env_extra or {})
    os.makedirs(logs_dir(run), exist_ok=True)
    log = os.path.join(logs_dir(run), log_name)
    off = os.path.getsize(log) if os.path.exists(log) else 0
    res = legacy.run(script, args, python=python_key(tier), env=env, cwd=os.path.dirname(script), log=log, unit=u,
                     tail_lines=cfg().photometry.LOG_TAIL_LINES)
    for f in sorted(res.fingerprint or {}):
        if os.path.isfile(f):
            u.add_input(f)
    u.add_input(script)
    with open(log, 'rb') as fh:
        fh.seek(off)
        text = fh.read().decode('utf-8', 'replace')
    return res, text


def legacy_code_changed(rec):
    """Files of a record's child fingerprints whose sha1 differs now ('' when none)."""
    changed = []
    for f, h in ((rec.get('meta') or {}).get('legacy_code') or {}).items():
        try:
            now = hashlib.sha1(open(f, 'rb').read()).hexdigest()[:12]
        except OSError:
            now = None
        if now != h:
            changed.append(os.path.basename(f))
    return ', '.join(changed[:5])


def resumed(run, name, admitted=None, **meta):
    """--resume: True (a 'skipped-valid' unit; the done record stays as written) when the unit's record is done, its
    recorded inputs (incl. the in-place engine files), admitted set, code and config are unchanged and its outputs pass
    their content tests now (core/record.resume_check)."""
    if not run.resume or run.fresh:
        return False
    ok, why = run.resume_check(name, inputs=None, admitted=admitted)
    if ok:
        ch = legacy_code_changed(record.load_record(run.out_root, run.step, name) or {})
        if ch:
            ok, why = False, 'in-place code changed: %s' % ch
    if not ok:
        print('resume %s: not skipped (%s)' % (name, why), flush=True)
        return False
    u = run.unit(name, **meta)
    u.status, u.reason = 'skipped-valid', why
    print('resume %s: skipped (%s)' % (name, why), flush=True)
    return True


def run_resume_ok(run, name, admitted, meta_equal):
    """resumed() plus equality of recorded meta values the core identity cannot see (e.g. the digest of a cell's cutout
    listing: each cutout is ~15 MB, which record.fingerprint would hash in full - the step compares path, size and mtime
    of every cutout through one digest instead)."""
    if not run.resume or run.fresh:
        return False
    rec = record.load_record(run.out_root, run.step, name) or {}
    for k, v in (meta_equal or {}).items():
        if (rec.get('meta') or {}).get(k) != v:
            print('resume %s: not skipped (%s changed)' % (name, k), flush=True)
            return False
    return resumed(run, name, admitted=admitted)


def prev_record(run, name, what, root=None):
    """The done record of an earlier stage's unit (the only source of that stage's outputs, DESIGN 11.5), else Refusal."""
    rec = record.load_record(root or run.out_root, run.step, name)
    if rec is None:
        raise Refusal('M-CHAIN: no record of unit %s under %s (run --stages %s first)' % (name, root or run.out_root, what))
    if rec.get('status') not in ('done',):
        raise Refusal('M-CHAIN: unit %s is %s (%s) - run --stages %s again' % (name, rec.get('status'), rec.get('reason'),
                                                                             what))
    return rec


def outputs_of(rec, cls):
    return [o['path'] for o in rec.get('outputs', []) if o.get('class') == cls]


def atomic_copy(src, dst):
    import shutil
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = os.path.join(os.path.dirname(dst), '.%s.tmp.%d' % (os.path.basename(dst), os.getpid()))
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)
    return dst


def write_text_atomic(path, text):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = '%s.tmp.%d' % (path, os.getpid())
    with open(tmp, 'w', newline='') as f:
        f.write(text)
    os.replace(tmp, path)
    return path


def resolve_stages(args, tier):
    """The stages to run, in tier order: --stages (any order given), else the per-hour set with --epochs, else all; then
    --from / --until cut the list. Unknown stage names are an error."""
    c = cfg()
    allst = tuple(c.photometry.STAGES[tier])
    if args.stages:
        want = [s.strip() for s in args.stages.split(',') if s.strip()]
        bad = [s for s in want if s not in allst]
        if bad:
            raise ValueError('unknown stage(s) %s for tier %s (stages: %s)' % (bad, tier, ','.join(allst)))
        sel = [s for s in allst if s in want]
    elif getattr(args, 'epochs', None):
        sel = list(c.photometry.EPOCH_STAGES[tier])
    else:
        sel = list(allst)
    for flag, val in (('--from', args.stage_from), ('--until', args.stage_until)):
        if val and val not in allst:
            raise ValueError('%s %s: not a stage of tier %s (%s)' % (flag, val, tier, ','.join(allst)))
    if args.stage_from:
        sel = [s for s in sel if allst.index(s) >= allst.index(args.stage_from)]
    if args.stage_until:
        sel = [s for s in sel if allst.index(s) <= allst.index(args.stage_until)]
    if not sel:
        raise ValueError('no stage selected (tier %s stages: %s)' % (tier, ','.join(allst)))
    return sel


def stage_ok(units):
    """True when every unit of a stage ended done or skipped-valid (no-input units do not block)."""
    return all(u.status in ('done', 'skipped-valid', 'no-input') for u in units) and \
        any(u.status in ('done', 'skipped-valid') for u in units)


def admitted_epochs(rows):
    """'<lst>_<band>:<date>' of the frame rows whose (lst, band, date) core.exclusions admits (resume identity)."""
    from exopipe.phase3.core import exclusions
    ex = exclusions.load()
    return sorted({'%s_%s:%s' % (r['lst'], r['band'], r['date']) for r in rows
                   if not ex.is_excluded(r['lst'], r['band'], r['date'])})


def say(text):
    print(text, file=sys.stderr, flush=True)


__all__ = ['NoInput', 'Refusal']
