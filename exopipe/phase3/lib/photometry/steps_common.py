"""Shared orchestration of `exopipe3 photometry` (steps/photometry.py): run-level gates, engine resolution, stage selection,
the in-place call wrapper, record helpers and the dry-run plan printer. No result-changing constant lives here (config
sections photometry*); the engines run IN PLACE through core/legacy.run (DESIGN 11.1)."""
import hashlib
import os
import sys

from exopipe.phase3.core import record
from exopipe.phase3.core.record import NoInput, Refusal


def cfg():
    from exopipe.phase3.config import CFG
    return CFG


def hostname():
    """This host's short name (core/env.host: the key of every per-host site table)."""
    from exopipe.phase3.core import env
    return env.host()


def plan(run, text):
    """One line of the plan (--dry-run prints the plan; a real run prints it too, as its log)."""
    print('PLAN %s' % text, flush=True)


def would_refuse(run, stage, msg):
    """--dry-run: print the refusal the real run would raise and remember it (the dry run then exits 2, as a refused run
    would; steps/photometry.py)."""
    plan(run, '[%s]   WOULD REFUSE %s' % (stage, msg))
    run.photometry_would_refuse = getattr(run, 'photometry_would_refuse', []) + ['%s: %s' % (stage, msg)]


def earlier_in_this_run(args, stage, before):
    """True when ``stage`` is selected in THIS invocation ahead of ``before`` (run_tier sets args.photometry_stages):
    in a --dry-run nothing is written, so a later stage's inputs are the outputs that stage would write, not records on
    disk (DESIGN 11.5: each stage reads the files the previous stage's record lists)."""
    st = list(getattr(args, 'photometry_stages', ()) or ())
    return stage in st and before in st and st.index(stage) < st.index(before)


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


def gate_tier(tier, stages_arg=None):
    """PHOT-SUBLST-NO-ENGINE: the sub-LST tier has the 10-minute tier cutouts and their forced measurement (photometry.STAGES
    ['sublst']), no target-photometry measure / evaluate engine: naming one of those refuses (as does a tier with no engine)."""
    c = cfg()
    want = [s.strip() for s in (stages_arg or '').split(',') if s.strip()]
    other = sorted({s for t, st in c.photometry.STAGES.items() if t != tier for s in st} - set(c.photometry.STAGES.get(tier, ())))
    if tier not in c.photometry.TIERS_WITH_ENGINE or (tier == 'sublst' and any(s in other for s in want)):
        raise Refusal('PHOT-SUBLST-NO-ENGINE: no production sub-LST target photometry exists beyond the 10-minute tier cutouts '
                      'and their forced measurement (stages %s; the 19h pilot sublst_forced_phot.py / sublst_forced_eval.py is '
                      'one-off and no launcher runs it; ledger R-FORCED-PHOTOMETRY-ELEMENTS-APPROVED-V-FAINT-PARENT-ENTRIES and '
                      'R-FORCED-THRESHOLDS-PROVISIONAL-AND-VETO-SCOPE notes) - tier %s stage(s) %s refused'
                      % (','.join(c.photometry.STAGES.get('sublst', ())), tier, ','.join(s for s in want if s in other) or 'all'))


# ---------------------------------------------------------------------------------------------------------------------
# generation roots and ad-hoc target lists (CUTOVER_READINESS 'PHOTOMETRY CUTOUTS ...', 'DATED DRIVERS TO COVER')
# ---------------------------------------------------------------------------------------------------------------------

def cutout_root(args):
    """The cutout generation a stage reads or writes: --cutout-root, else the served generation (photometry.CUTOUT_ROOT)."""
    return os.path.abspath(args.cutout_root) if getattr(args, 'cutout_root', None) else cfg().photometry.CUTOUT_ROOT


def gate_write_root(path, what):
    """C-ROOT: a stage writes products outside --out (the cutout generation, the tier forced roots): never into a retired
    generation (photometry.RETIRED_CUTOUT_ROOTS: the gen-4 root held cutouts an injected run wrote), and - like --out
    (cli._guard_out) - never into a production root while paths.PRODUCTION_WRITES is False."""
    c = cfg()
    real = os.path.realpath(path)
    for r in c.photometry.RETIRED_CUTOUT_ROOTS:
        rr = os.path.realpath(r)
        if real == rr or real.startswith(rr.rstrip('/') + '/'):
            raise Refusal('C-ROOT: %s %s is under the retired cutout generation %s (photometry.RETIRED_CUTOUT_ROOTS: it held '
                          'cutouts an injected run wrote) - write a generation root' % (what, path, r))
    root = c.production_root_of(path)
    if root is not None and not c.paths.PRODUCTION_WRITES:
        raise Refusal('C-ROOT: %s %s is under the production root %s and paths.PRODUCTION_WRITES is False (refactor window) '
                      '- write parity / scratch products to your own root' % (what, path, root))
    return path


def adhoc_targets(args):
    """[(tag, absolute csv path)] of --targets TAG=CSV[,TAG=CSV] (the ad-hoc target sets of phase3_snapshot_cutouts._samples),
    validated: tags unique, a sample directory name, never the planet sample; every file readable with at least one target row
    by the cutout writer's column rule (common_name|name|Name, ra_current|ra|RA, dec_current|dec|DEC). [] without --targets."""
    import csv
    import re
    spec = getattr(args, 'targets', None)
    if not spec:
        return []
    ph = cfg().photometry
    out = []
    for item in [x.strip() for x in spec.split(',') if x.strip()]:
        tag, sep, path = item.partition('=')
        tag, path = tag.strip(), path.strip()
        if not sep or not tag or not path:
            raise Refusal('PHOT-ADHOC: --targets %r: expected TAG=CSV[,TAG=CSV]' % item)
        if not re.match(ph.ADHOC_TAG_RE, tag) or tag == ph.PLANET_SAMPLE:
            raise Refusal('PHOT-ADHOC: --targets tag %r: a sample directory name (%s), not %s' % (tag, ph.ADHOC_TAG_RE,
                                                                                               ph.PLANET_SAMPLE))
        if tag in [t for t, _ in out]:
            raise Refusal('PHOT-ADHOC: --targets names the tag %s twice' % tag)
        path = os.path.abspath(path)
        if not os.path.isfile(path):
            raise Refusal('PHOT-ADHOC: --targets %s=%s: no such file' % (tag, path))
        n = 0
        with open(path, encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                ra = r.get('ra_current') or r.get('ra') or r.get('RA')
                dec = r.get('dec_current') or r.get('dec') or r.get('DEC')
                try:
                    float(ra), float(dec)
                except (TypeError, ValueError):
                    continue
                n += 1
        if not n:
            raise Refusal('PHOT-ADHOC: --targets %s=%s: no row with a name and a parsable RA / Dec (columns common_name|name|'
                          'Name, ra_current|ra|RA, dec_current|dec|DEC)' % (tag, path))
        out.append((tag, path))
    return out


def adhoc_env(targets, planets=False):
    """The process environment the in-place engines read an ad-hoc target set from (photometry.ADHOC_ENV /
    ADHOC_PLANETS_ENV: phase3_snapshot_cutouts._samples / _planets, snapshot_forced_phot SAMPLES); {} without one."""
    ph = cfg().photometry
    if not targets:
        return {}
    env = {ph.ADHOC_ENV: ','.join('%s=%s' % tp for tp in targets)}
    if planets:
        env[ph.ADHOC_PLANETS_ENV] = '1'
    return env


def gate_adhoc_roots(targets, roots):
    """PHOT-ADHOC: an ad-hoc target set never writes into the served generation (production wrote its own trees,
    forced_adhoc_20261028/photometry and /stage1): each of ``roots`` ({what: path}) must differ from the served one."""
    if not targets:
        return
    c = cfg()
    served = {'cutout root': c.photometry.CUTOUT_ROOT,
              'stage-1 root': os.path.join(c.photometry.OUT_ROOT['snapshot'], c.photometry.STAGE1_DIR)}
    served.update({'tier forced root (%s)' % s: r for s, r in c.photometry.TIER_FORCED_ROOTS.items()})
    for what, p in roots.items():
        for sw, sp in served.items():
            if p and os.path.realpath(p) == os.path.realpath(sp):
                raise Refusal('PHOT-ADHOC: --targets with the %s %s = the served generation\'s %s - an ad-hoc target set '
                              'writes its own root (give --cutout-root / --out / --tier-out)' % (what, p, sw))


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


def python_key(tier, key=None):
    """The site interpreter key of engine ``key`` (photometry.ENGINE_PYTHON_BY_KEY), else its tier's (ENGINE_PYTHON)."""
    ph = cfg().photometry
    return ph.ENGINE_PYTHON_BY_KEY.get(key) or ph.ENGINE_PYTHON[tier]


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
    res = legacy.run(script, args, python=python_key(tier, key), env=env, cwd=os.path.dirname(script), log=log, unit=u,
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
    """The stages to run, in tier order: --stages (any order given), else the per-hour set with --epochs, else the tier's
    default set (photometry.DEFAULT_STAGES: a cutout regeneration runs only when named); then --from / --until cut the list.
    Unknown stage names are an error."""
    c = cfg()
    allst = tuple(c.photometry.STAGES[tier])
    if args.stages:
        want = [s.strip() for s in args.stages.split(',') if s.strip()]
        bad = [s for s in want if s not in allst]
        if bad:
            raise ValueError('unknown stage(s) %s for tier %s (stages: %s)' % (bad, tier, ','.join(allst)))
        sel = [s for s in allst if s in want]
    elif getattr(args, 'epochs', None):
        sel = list(c.photometry.EPOCH_STAGES.get(tier, ()))
        if not sel:
            raise ValueError('tier %s has no per-hour stage (--epochs): its tier cutouts hold every night of an LST hour - '
                             'select <LST>_<band> cells with --cells' % tier)
    else:
        default = tuple(c.photometry.DEFAULT_STAGES.get(tier, allst))
        sel = [s for s in allst if s in default]
    for flag, val in (('--from', args.stage_from), ('--until', args.stage_until)):
        if val and val not in allst:
            raise ValueError('%s %s: not a stage of tier %s (%s)' % (flag, val, tier, ','.join(allst)))
        if val and val not in sel and not args.stages and not getattr(args, 'epochs', None):
            raise ValueError('%s %s: stage %s runs only when named (--stages; tier %s default: %s)'
                             % (flag, val, val, tier, ','.join(sel)))
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
