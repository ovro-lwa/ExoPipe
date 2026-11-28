"""Shared orchestration of `exopipe3 photometry` (steps/photometry.py): run-level gates, engine resolution, stage selection,
the in-place call wrapper, record helpers and the dry-run plan printer. No result-changing constant lives here (config
sections photometry*); the engines run IN PLACE through core/legacy.run (DESIGN 11.1)."""
import hashlib
import os
import sys

from exopipe.phase3.core import placement, record
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


def samples_measured(tier):
    """The samples ``tier`` measures (photometry.SAMPLES_MEASURED[tier]: snapshot HWJ, 10pc, LPT; deep and sub-LST HWJ, 10pc)."""
    return tuple(cfg().photometry.SAMPLES_MEASURED[tier])


def samples_cut(tier):
    """The samples ``tier``'s cutouts hold: photometry.SAMPLES_CUT, the one writer list every cutter takes (HWJ, 10pc, LPT),
    the same for every tier."""
    return tuple(cfg().photometry.SAMPLES_CUT)


def sample_paths(tier, cut=False):
    """{sample: list path} of the samples ``tier`` measures, or (cut=True) of the samples its cutters read."""
    c = cfg()
    return {t: os.path.join(c.paths.CLV15_ROOT, c.photometry.SAMPLE_FILES[t])
            for t in (samples_cut(tier) if cut else samples_measured(tier))}


def gate_samples(tier, cut=False):
    """PHOT-SAMPLE-PIN: each sample list ``tier`` measures - or with cut=True (a cutouts stage) each list its cutters read -
    must be the file config pins (sha1). The engines read these files by literal path; a rebuilt list needs a config update
    (B-58 lean shape, pure: same files, same values)."""
    c = cfg()
    out = {}
    for t, p in sample_paths(tier, cut=cut).items():
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

def served_cutout_root(tier):
    """The served cutout generation of ``tier`` (photometry.CUTOUT_ROOTS: snapshot = the per-night cutouts, deep / sublst =
    the tier cutouts); a tier without an entry refuses (C-TIER-ROOT), never another tier's root."""
    roots = cfg().photometry.CUTOUT_ROOTS
    if tier not in roots:
        raise Refusal('C-TIER-ROOT: no served cutout root for tier %r (photometry.CUTOUT_ROOTS names %s)'
                      % (tier, ', '.join(sorted(roots))))
    return roots[tier]


def cutout_root(args, tier=None):
    """The cutout generation a stage reads or writes: a campaign's generation (--campaign: photometry_campaign.GENERATION,
    lib/photometry/campaign.py), else --cutout-root, else the tier's served generation (photometry.CUTOUT_ROOTS[tier]; tier =
    ``tier``, else args.tier)."""
    if getattr(args, 'campaign', False):
        from exopipe.phase3.lib.photometry import campaign
        return campaign.cutout_root()
    if getattr(args, 'cutout_root', None):
        return os.path.abspath(args.cutout_root)
    return served_cutout_root(tier or getattr(args, 'tier', None))


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
    by the cutout writer's column rule (common_name|name|Name, ra_current|ra|RA, dec_current|dec|DEC), and every row with a
    position named (review F11). [] without --targets."""
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
        from exopipe.phase3.lib.photometry.tests import safe_name
        n, unnamed = 0, []
        with open(path, encoding='utf-8-sig') as f:
            for i, r in enumerate(csv.DictReader(f), 2):
                ra = r.get('ra_current') or r.get('ra') or r.get('RA')
                dec = r.get('dec_current') or r.get('dec') or r.get('DEC')
                try:
                    float(ra), float(dec)
                except (TypeError, ValueError):
                    continue
                # the writer cuts every row with a position, under safe_name(name) (phase3_snapshot_cutouts._targets): a row
                # without a usable name would be cut as <TAG>/<band>/_<LST>_..., an object with no directory (review F11)
                if not safe_name(r.get('common_name') or r.get('name') or r.get('Name') or ''):
                    unnamed.append(i)
                    continue
                n += 1
        if unnamed:
            raise Refusal('PHOT-ADHOC: --targets %s=%s: %d row(s) with a position but no usable name (line(s) %s; columns '
                          'common_name|name|Name) - the writer would cut them as an object without a directory; name every '
                          'target' % (tag, path, len(unnamed), ', '.join(map(str, unnamed[:5]))))
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


def served_roots():
    """[(what, root)] of the SERVED generations' product roots: photometry.CUTOUT_ROOTS (one entry per distinct root, naming
    its tiers: 'cutout root (deep, sublst)'), the stage-1 root <OUT_ROOT[snapshot]>/<STAGE1_DIR>, and photometry.
    TIER_FORCED_ROOTS per sample."""
    ph = cfg().photometry
    tiers = {}
    for t, r in sorted(ph.CUTOUT_ROOTS.items()):
        tiers.setdefault(r, []).append(t)
    return ([('cutout root (%s)' % ', '.join(ts), r) for r, ts in tiers.items()]
            + [('stage-1 root', os.path.join(ph.OUT_ROOT['snapshot'], ph.STAGE1_DIR))]
            + [('tier forced root (%s)' % s, r) for s, r in sorted(ph.TIER_FORCED_ROOTS.items())])


def served_root_of(path):
    """(what, root) of the served root that equals or holds ``path`` (real paths; the most specific), else None."""
    if not path:
        return None
    real = os.path.realpath(path)
    hits = []
    for what, r in served_roots():
        rr = os.path.realpath(r).rstrip('/') or '/'
        if real == rr or real.startswith(rr + '/'):
            hits.append((len(rr), what, r))
    return max(hits)[1:] if hits else None


def gate_adhoc_roots(targets, roots):
    """PHOT-ADHOC: an ad-hoc target set never writes into the served generation (production wrote its own trees,
    forced_adhoc_20261028/photometry and /stage1): each of ``roots`` ({what: path}) must lie outside every served root - equal
    to one or anywhere under one refuses (review F10)."""
    if not targets:
        return
    for what, p in roots.items():
        hit = served_root_of(p)
        if hit:
            same = os.path.realpath(p) == os.path.realpath(hit[1])
            raise Refusal('PHOT-ADHOC: --targets with the %s %s %s the served generation\'s %s %s - an ad-hoc target set '
                          'writes its own root (give --cutout-root / --out / --tier-out outside it)'
                          % (what, p, '=' if same else 'inside', hit[0], hit[1]))


def gate_pinned_inputs(paths):
    """Pinned inputs the engines read by literal path must exist (a missing one would fail inside the engine)."""
    miss = [p for p in paths if p and not os.path.exists(p)]
    if miss:
        raise Refusal('pinned input missing: %s' % ', '.join(miss))


# ---------------------------------------------------------------------------------------------------------------------
# engines (in place, core/legacy.run)
# ---------------------------------------------------------------------------------------------------------------------

def engine(key):
    """Absolute path of engine ``key``: a v2 engine (photometry.ENGINE_V2, a path under exopipe/phase3/), else the in-place
    engine photometry.ENGINE[key] - exactly one hit among the site LEGACY_CODE_ROOTS."""
    c = cfg()
    if key in c.photometry.ENGINE_V2:
        import exopipe.phase3 as p3
        p = os.path.join(os.path.dirname(os.path.abspath(p3.__file__)), c.photometry.ENGINE_V2[key])
        if not os.path.isfile(p):
            raise Refusal('engine %s: no file %s in the v2 package' % (key, p))
        return p
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
    edit of an in-place module, spec blocker 21). ``u`` None: a call outside any unit (once per stage); the caller adds the
    fingerprint to the units it serves. Returns (result, the log text of this call)."""
    from exopipe.phase3.core import legacy
    script = engine(key)
    env = engine_env(env_name, threads)
    env.update(env_extra or {})
    os.makedirs(logs_dir(run), exist_ok=True)
    log = os.path.join(logs_dir(run), log_name)
    off = os.path.getsize(log) if os.path.exists(log) else 0
    res = legacy.run(script, args, python=python_key(tier, key), env=env, cwd=os.path.dirname(script), log=log, unit=u,
                     tail_lines=cfg().photometry.LOG_TAIL_LINES)
    if u is not None:
        for f in sorted(res.fingerprint or {}):
            if os.path.isfile(f):
                u.add_input(f)
        u.add_input(script)
    with open(log, 'rb') as fh:
        fh.seek(off)
        text = fh.read().decode('utf-8', 'replace')
    return res, text


def code_sha12(path):
    """sha1[:12] of a code file's bytes - the fingerprint core/legacy.run records per child module and the campaign pins
    (lib/photometry/campaign.CodePin) - or None when it cannot be read."""
    try:
        with open(path, 'rb') as f:
            return hashlib.sha1(f.read()).hexdigest()[:12]
    except OSError:
        return None


def legacy_code_changed(rec):
    """Files of a record's child fingerprints whose sha1 differs now ('' when none)."""
    changed = []
    for f, h in ((rec.get('meta') or {}).get('legacy_code') or {}).items():
        now = code_sha12(f)
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


def place_copies(pairs):
    """Place copies of the work files [(src, dst)] (node-local scratch) together: core/placement.atomic_many - each copied to a
    temporary file beside its destination and checked against its source (size, md5; placement.copy_verified, with the
    source's times as copy2 gave them), then all given the replaced file's group, mode and ACL or the directory's rule for a
    new one (never the work file's), then renamed in order. A copy that differs (CopyNotVerified) or an attribute refusal
    places none of them; a failed rename after others leaves those placed and its PlacementRefusal names them. Either way
    no temporary file is left. A killed earlier write's temporary files are removed first (placement.sweep). Returns the
    destinations. Two pairs with one destination refuse (PlacementRefusal) before anything is written."""
    import collections
    dsts = [d for _s, d in pairs]
    twice = sorted(a for a, n in collections.Counter(os.path.abspath(d) for d in dsts).items() if n > 1)
    if twice:
        raise placement.PlacementRefusal(f'placement: destinations named twice {twice} (two sources for one product) - '
                                         f'nothing written')
    for d in sorted({os.path.dirname(d) for d in dsts}):
        os.makedirs(d, exist_ok=True)
    placement.sweep(dsts)
    with placement.atomic_many(dsts) as tmps:
        for s, d in pairs:
            placement.copy_verified(s, tmps[d], times=True)
    return dsts


def write_text_atomic(path, text):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    # the existing file's group, mode and ACL, or the directory's rule for a new file (core/placement.atomic)
    with placement.atomic(path) as tmp, open(tmp, 'w', newline='') as f:
        f.write(text)
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
        if not default:
            raise ValueError('tier %s has no default stages: its stages %s regenerate products, so a run names them '
                             '(--stages; photometry.DEFAULT_STAGES)' % (tier, ','.join(allst)))
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


def stop_after(args, stage, units):
    """True when the stages after ``stage`` must not run: one of its units did not complete. Except the cutouts stage of an
    --in-place regeneration of the served generation (review F4): its measurement runs once any unit placed its files,
    whatever the others' outcomes - a refused or failed unit leaves its files as they were (steps_cutouts.placement), so the
    measurement is that of the served root as it now stands."""
    if not units or stage_ok(units):
        return False
    if stage == 'cutouts' and getattr(args, 'in_place', False):
        return not any(u.status in ('done', 'skipped-valid') for u in units)
    return True


def admitted_epochs(rows):
    """'<lst>_<band>:<date>' of the frame rows whose (lst, band, date) core.exclusions admits (resume identity)."""
    from exopipe.phase3.core import exclusions
    ex = exclusions.load()
    return sorted({'%s_%s:%s' % (r['lst'], r['band'], r['date']) for r in rows
                   if not ex.is_excluded(r['lst'], r['band'], r['date'])})


def say(text):
    print(text, file=sys.stderr, flush=True)


__all__ = ['NoInput', 'Refusal']
