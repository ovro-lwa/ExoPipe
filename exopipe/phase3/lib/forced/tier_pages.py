"""exopipe3 forced --tier snapshot --stages tier_eval,tier_site,tier_runs: family T, the HWJ / 10 pc / LPT forced-photometry
pages of the 10-minute and deep tiers (Gregg 2026-10-27, at the terminal: "HWJ and 10 pc forced photometry at 10-minute and deep
- that is very valuable"). Port of the dated driver exopipe_ops tier_forced_page_20261027.py (ledger 3079 site, 3081 chain,
3394-3395 Epoch panels), run IN PLACE through core.legacy.run - every subcommand unchanged - with the site interpreter
forced.PYTHON_KEY['tier_pages'] (py36: the fit imports bdsf).

THE SAMPLES (forced_tier_pages.SAMPLES): HWJ, 10 pc and, since 2026-11-24, the known long-period transients (LPT) - a sample
inside the page (Gregg's hub ruling of 2026-10-27: a new page variant is a sample inside a page, not a hub button), measured
and shown by the same chain, panels and statistics (replacing the one-off /s/lpt/ page of the dated drivers phase3/
forced_lpt_20261118, forced_lpt_20261123). The script's own literal is HWJ / 10 pc: the stages pass SAMPLES and SAMPLE_LABEL
through the in-place driver's --globals when they differ from it, and the root of the evaluations as its SF.

Campaign stages, named only (never a default), one family per invocation (as families W and N):
  tier_eval  unit T_eval: the four stage-2 evaluations of the pages, snapshot_forced_eval.py in place (production's command,
             run_evals.sh): `<FT>/<tier> <out>/<EVAL_NAME> --kind sb|wb --nproc N --tier 10min|deep --survey-cells
             EVAL_SURVEY_CELLS --samples <SAMPLES>` with P3_PHOT_ROOT = the cutout root (the cutouts' TIMES), per tier and
             kind. The thresholds, counts and exceedance fractions are per sample, so a sample added leaves the other samples'
             rows as they were. Outputs <out>/eval_<tier>_[wb_]gen5_hwj10pc/thresholds.csv (class forced_tier_eval). FT =
             --tier-forced-root, else photometry.TIER_FORCED_ROOTS (one root for every sample); the cutout root =
             --tier-cutout-root, else photometry.CUTOUT_ROOT.
  tier_site  unit T_site: `build SITE --nproc N`, then its content test `verify SITE` (HWJ10PC-TIERS-PAGE-OK: the events are
             exactly the clean measurements at or above their Gaussian thresholds in the four evaluations, every card's forced
             numbers recomputed from the tier files, every yellow reading from the saved crops). SITE = --tier-site, else
             <out>/forced_tier_pages.SITE_NAME; its Masters link tree is SITE_masters. Output SITE/page_record.json
             (class forced_tier_site). The evaluations: this --out's T_eval (tier_eval in this run, or its done record),
             else production's forced_tier_pages.EVAL_DIRS (HWJ / 10 pc only: T-EVAL-SAMPLES refuses them for an LPT page).
  tier_runs  unit T_runs: `chain SITE --root ROOT --nproc N` (the four review runs ROOT/Run_SNAPFORCED_{V,I}{10min,deep}_gen5.new
             from the site's cards: fit in each event's own image, localization, association, render, the served tree), then
             its content test `verify-chain SITE --root ROOT` (TIER-CHAIN-OK). Outputs ROOT/<run>.new/queue_<S>.txt (class
             forced_tier_run, one per run). ROOT = --snap-review-root, else <out>/forced_snapshot.REVIEW_SUBDIR_UNDER_OUT under
             --out, else paths.SNAPSHOT_REVIEW_ROOT. The site is read through the T_site record of this --out when there is
             one, else content-tested as an input (the served site was built by production). The chain reads the tier files
             and cutouts the site was built on (its page_record inputs tiers / phot, passed as P3_FORCED_TIERS / P3_PHOT_ROOT).
Directories: --out = the site root W (default forced_tier_pages.SITE_ROOT); records W/_records/forced/T_<stage>.json, logs
W/_records/forced/logs/. The fit's work directory is the script's literal TIER_FIT_WORK for a production run; under --out it is
<--work, else the site scratch root>/tier_forced_fit_work, set through lib/jupiter/_inplace.py --globals (the in-place driver
that sets a module literal ONLY when the config value differs from it; a production run passes none).
The chain's localization is the in-place source_localization, wrapped under the ruled config (core/wrap.py, B-32 (a): the
inverse-variance estimator with fitted errors), so its localizations differ from production's. A parity build therefore passes
--parity --config exopipe/phase3/parity/production_20261013.cfg (--parity alone only installs the write guard).
Not ported: installing the runs (production's `swap`: <run> -> <run>.superseded_*, <run>.new -> <run>), the one-off `runs` /
`verify-runs` (the no-fit generation of 2026-10-27, superseded the same morning) and `rerender` / `swap-stage` (2026-11-10, the
Epoch panels: the chain renders them now). Serving them (p9120-p9123) is not part of `serve` either.
Gates (exit 2, named in the record): T-G35 the site directory, its Masters tree, a <run>.new or its Masters tree, an
evaluation directory exists (a name is used once); T-EVAL an evaluation file missing, or this --out's tier_eval record not done;
T-EVAL-SAMPLES an evaluation without the thresholds of every SAMPLES (sample, Stokes) of its tier; T-SAMPLES a sample without
a pinned list (photometry.SAMPLE_FILES / SAMPLE_SHA1) or a label; T-ROOTS the samples' tier forced roots differ, or the
evaluations were made on other roots than this run names; T-INPUT a sample without a measurement of a tier and kind in the tier
forced root, or without cutouts; T-EVAL-RUN an evaluation exiting non-zero or without its success line; T-EXCL an event on an
excluded cell-band (forced_tier_pages.EXCLUDED_EVENTS 'refuse'), read now: tier_site checks the evaluations' selection,
tier_runs the site's cards (a site production built had no such gate); T-HOST tier_runs off the review host (the localization reads paths.COADD_SB_ROOT) or the review
root node-local to another host; T-SITE the site's record or content test failing; T-WORKERS --workers above
forced_tier_pages.MAX_WORKERS (the in-place pool reads Lustre); T-WORK the fit's work directory not node-local
(core.fitsio.scratch_root: none configured, or on Lustre; never a fallback to --out); T-BUILD / T-CHAIN the in-place call
exiting non-zero or its success line missing; T-VERIFY / T-VERIFY-CHAIN the in-place content test failing; a production
target while paths.PRODUCTION_WRITES is False. A dry run exits 2 when its plan names a gate the real run would hit.
--resume adopts a T_* record only for the same site (tier_runs: and review root and fit work) and the same admitted
cell-bands (the events not excluded now), then as every unit (inputs, code, config, outputs re-tested)."""
import json
import os

from exopipe.phase3.core.record import Refusal
from exopipe.phase3.lib.forced import common as K

UNITS = {'tier_eval': 'T_eval', 'tier_site': 'T_site', 'tier_runs': 'T_runs'}
# the in-place driver: runs one function of a production module unchanged, module globals set only when passed
DRIVER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'jupiter', '_inplace.py')


def add_arguments(p):
    g = p.add_argument_group('snapshot tier, family T (the HWJ / 10 pc / LPT forced-photometry pages of the 10-minute and deep '
                             'tiers: --stages tier_eval,tier_site,tier_runs)')
    g.add_argument('--tier-site', metavar='DIR', help='the site directory tier_site builds and tier_runs reads (default '
                                                      '<out>/forced_tier_pages.SITE_NAME); a build never reuses one')
    g.add_argument('--tier-forced-root', metavar='DIR', help='tier_eval / tier_site: the tier forced root (<DIR>/<10min|deep>/'
                                                             '<LST>/forced_<sample>_<band>_<S>.npz) the evaluations and the '
                                                             'site read (default photometry.TIER_FORCED_ROOTS of the samples)')
    g.add_argument('--tier-cutout-root', metavar='DIR', help='tier_eval / tier_site: the tier cutouts those were measured on '
                                                             '(default photometry.CUTOUT_ROOT)')


def default_out(args):
    return K.cfg().forced_tier_pages.SITE_ROOT


def run_names(t=None):
    """[(tier, Stokes, run name)] in the in-place chain's order (tier_forced_page_20261027.py:944-945)."""
    t = t or K.cfg().forced_tier_pages
    return [(tier, S, t.RUN_NAME.format(S=S, tier=tier)) for tier in t.TIERS for S in ('V', 'I')]


def selected_events(t=None, ev_dirs=None, samples=None):
    """The rows the in-place build lists (tier_forced_page_20261027.py:41-65): per tier and kind, the clean rows of the
    evaluation's targets_ge4.csv of the samples (default forced_tier_pages.SAMPLES) at or above thr_survey_gauss of their
    (sample, tier, kind, Stokes) (Stokes V: |S/N|). Own code, for the dry-run count and the exclusion gate."""
    import csv
    t = t or K.cfg().forced_tier_pages
    samples = tuple(samples or t.SAMPLES)
    out = []
    for (tier, kind), ev in sorted((ev_dirs or t.EVAL_DIRS).items()):
        tp, rp = os.path.join(ev, 'thresholds.csv'), os.path.join(ev, 'targets_ge4.csv')
        if not (os.path.exists(tp) and os.path.exists(rp)):
            continue
        with open(tp) as f:
            th = {(r['sample'], r['stokes']): r for r in csv.DictReader(f) if r['tier'] == tier and r['sample'] in samples}
        with open(rp) as f:
            for r in csv.DictReader(f):
                if r['tier'] != tier or r['screen'] != 'clean' or r['sample'] not in samples:
                    continue
                th_ = th.get((r['sample'], r['stokes']))
                if th_ is None:
                    continue
                v = float(r['snr'])
                if (abs(v) if r['stokes'] == 'V' else v) >= float(th_['thr_survey_gauss']):
                    out.append(dict(r, kind=kind))
    return out


class TCtx:
    def __init__(self, args, run, stages=()):
        self.args, self.run, self.cfg = args, run, K.cfg()
        self.stages = tuple(stages)
        t, paths = self.cfg.forced_tier_pages, self.cfg.paths
        self.t, self.dry = t, run.dry_run
        self.W = os.path.abspath(run.out_root)
        self.site = os.path.abspath(args.tier_site) if getattr(args, 'tier_site', None) else os.path.join(self.W, t.SITE_NAME)
        if args.out:
            root = args.snap_review_root or os.path.join(self.W, self.cfg.forced_snapshot.REVIEW_SUBDIR_UNDER_OUT)
        else:
            root = args.snap_review_root or paths.SNAPSHOT_REVIEW_ROOT
        self.root = os.path.abspath(root)
        self.review_host = K.host_of(paths.COADD_SB_ROOT)   # the localization reads the deep co-add record there
        self.root_host = K.host_of(self.root)
        self.fit_work, self.fit_work_refusal = None, None
        if 'tier_runs' in self.stages:
            try:
                self.fit_work = self._fit_work()
            except Refusal as e:           # T-WORK: refused in the unit (or named by the dry run's plan)
                self.fit_work_refusal = str(e)
        self.py = self.cfg.forced.PYTHON_KEY['tier_pages']
        self.runs = run_names(t)
        self.would_refuse = []             # gates a dry run's plan names (the dry run then exits 2, as family W)
        self.ft, self.phot, self.roots_refusal = self._roots()

    def _roots(self):
        """(tier forced root, cutout root, T-ROOTS refusal or None): --tier-forced-root, else the one root
        photometry.TIER_FORCED_ROOTS names for every forced_tier_pages.SAMPLES (the script reads one, PP.FT);
        --tier-cutout-root, else photometry.CUTOUT_ROOT."""
        ph = self.cfg.photometry
        phot = os.path.abspath(self.args.tier_cutout_root) if getattr(self.args, 'tier_cutout_root', None) else ph.CUTOUT_ROOT
        if getattr(self.args, 'tier_forced_root', None):
            return os.path.abspath(self.args.tier_forced_root), phot, None
        roots = {s: ph.TIER_FORCED_ROOTS.get(s) for s in self.t.SAMPLES}
        if None in roots.values() or len(set(roots.values())) != 1:
            return None, phot, ('T-ROOTS the samples\' tier forced roots (photometry.TIER_FORCED_ROOTS) are %s: the pages read '
                                'one root - give --tier-forced-root' % roots)
        return roots[self.t.SAMPLES[0]], phot, None

    def eval_dirs(self):
        """{(tier, kind): <out>/<EVAL_NAME>}: the evaluations tier_eval writes (the script's ev_dir with SF = <out>)."""
        return {(tier, kind): os.path.join(self.W, self.t.EVAL_NAME.format(tier=tier, wb='wb_' if kind == 'wb' else ''))
                for tier, kind in sorted(self.t.EVAL_DIRS)}

    def eval_source(self):
        """(evaluations {(tier, kind): dir}, their root SF, how, T-EVAL / T-ROOTS refusal or None) that tier_site reads:
        this --out's tier_eval evaluations (tier_eval in this run, or its done T_eval record, made on the same roots), else
        production's forced_tier_pages.EVAL_DIRS (made on FT_PRODUCTION / PHOT_PRODUCTION)."""
        from exopipe.phase3.core import record
        t = self.t
        if 'tier_eval' in self.stages:
            return self.eval_dirs(), self.W, 'tier_eval in this run', None
        rec = record.load_record(self.W, self.run.step, UNITS['tier_eval'])
        if rec is not None:
            meta = rec.get('meta') or {}
            if rec.get('status') != 'done':
                return self.eval_dirs(), self.W, None, ('T-EVAL the tier_eval record of this --out is %s (%s), not done'
                                                        % (rec.get('status'), (rec.get('reason') or '')[:200]))
            if (meta.get('tier_forced_root'), meta.get('tier_cutout_root')) != (self.ft, self.phot):
                return self.eval_dirs(), self.W, None, ('T-ROOTS the evaluations of this --out read %s / %s, this run names '
                                                        '%s / %s' % (meta.get('tier_forced_root'), meta.get('tier_cutout_root'),
                                                                     self.ft, self.phot))
            return self.eval_dirs(), self.W, 'record %s' % UNITS['tier_eval'], None
        if (self.ft, self.phot) != (t.FT_PRODUCTION, t.PHOT_PRODUCTION):
            return dict(t.EVAL_DIRS), t.SITE_ROOT, None, (
                'T-ROOTS production\'s evaluations (forced_tier_pages.EVAL_DIRS) read %s / %s; this run names %s / %s - '
                'evaluate them first (--stages tier_eval)' % (t.FT_PRODUCTION, t.PHOT_PRODUCTION, self.ft, self.phot))
        return dict(t.EVAL_DIRS), t.SITE_ROOT, 'forced_tier_pages.EVAL_DIRS (production)', None

    def page_globals(self, sf):
        """The in-place script's module globals a site build sets, only where config differs from its literal: SF (the root of
        the evaluations, ev_dir :37-38), SAMPLES and SAMPLE_LABEL (:31-32)."""
        t = self.t
        gl = {}
        if os.path.abspath(sf) != os.path.abspath(t.SITE_ROOT):
            gl[t.EVAL_ROOT_GLOBAL] = os.path.abspath(sf)
        if tuple(t.SAMPLES) != tuple(t.SCRIPT_SAMPLES):
            gl['SAMPLES'] = list(t.SAMPLES)
        labels = {s: t.SAMPLE_LABEL.get(s) for s in t.SAMPLES}
        if labels != {s: t.SCRIPT_SAMPLE_LABEL.get(s) for s in t.SAMPLES}:
            gl['SAMPLE_LABEL'] = labels
        return gl

    def page_env(self, ft=None, phot=None):
        """P3_FORCED_TIERS / P3_PHOT_ROOT of the in-place page script (planets_forced_page_20261026.py:54, :61 read them at
        import), passed explicitly: core.legacy.run clears the caller's P3_*."""
        return {'P3_FORCED_TIERS': ft or self.ft, 'P3_PHOT_ROOT': phot or self.phot}

    def _fit_work(self):
        """--work, else the production literal for a production run, else the site scratch root of this host. The fit
        writes one plane FITS per event, so node-local only: core.fitsio.scratch_root refuses a missing root and a root on
        Lustre (no fallback to --out)."""
        from exopipe.phase3.core import fitsio
        sub = 'tier_forced_fit_work'
        if self.args.work:
            return os.path.join(fitsio.scratch_root(os.path.abspath(self.args.work)), sub)
        if not self.args.out:
            return self.t.FIT_WORK_PRODUCTION
        return os.path.join(fitsio.scratch_root(None), sub)

    def log(self, stage):
        return K.records_log(self.run, UNITS[stage])

    def new_dirs(self):
        """[(tier, Stokes, name, <root>/<name>.new)] of the chain's four runs."""
        return [(tier, S, n, os.path.join(self.root, n + self.t.NEW_SUFFIX)) for tier, S, n in self.runs]

    def inplace(self, u, argv, log, globals_=None, test_stage=False, stage=None, env=None):
        """One subcommand of the in-place script through the driver (core.legacy.run), with the environment ``env``. The
        script and the driver are the unit's inputs, their sha1 its legacy code."""
        script = K.script('tier_pages')
        res_path = os.path.join(os.path.dirname(log), '%s.inplace_%d.json' % (os.path.basename(log)[:-4], len(
            u.meta.get('inplace', []))))
        dargs = ['--module', script, '--call', 'main', '--argv', json.dumps([str(a) for a in argv]),
                 '--globals', json.dumps(globals_ or {}), '--result', res_path] + (['--test-stage'] if test_stage else [])
        res = K.call(u, DRIVER, dargs, self.py, log, env=env, cwd=K.legacy_ops(), stage=stage)
        u.meta.setdefault('legacy_code', {})[os.path.abspath(script)] = K._sha1(script)
        u.add_input(script)
        try:
            with open(res_path) as f:
                result = json.load(f)
        except (OSError, ValueError):
            result = {'kind': 'error', 'reason': 'the in-place driver wrote no result (exit %d)' % res.rc}
        # every exopipe_ops module the child loaded (planets_forced_page_20261026, snapshot_forced_panels, std7_panel, ...):
        # its code identity, re-checked by --resume (common.legacy_code_changed)
        for p in sorted(result.get('code_sha256') or {}):
            u.meta['legacy_code'][p] = K._sha1(p)
        u.meta.setdefault('inplace', []).append({'argv': [str(a) for a in argv], 'globals': globals_ or {}, 'env': env or {},
                                                 'kind': result.get('kind'), 'reason': (result.get('reason') or '')[:300],
                                                 'wall_s': result.get('wall_s'), 'rc': res.rc})
        return res


def run(args, run, stages):
    from exopipe.phase3.lib.forced.wide_stages import gate_args
    gate_args(args, 'T')
    c = TCtx(args, run, stages)
    K.plan('family T (%s forced-photometry pages, 10-minute and deep tiers): stages %s; site %s; review root %s '
           '(review host %s); runs %s; fit work %s; tier forced root %s; cutout root %s; tier_forced_page_20261027.py in place '
           '(site PYTHON %r)'
           % (' / '.join(c.t.SAMPLES), ','.join(stages), c.site, c.root, c.review_host or 'shared',
              ', '.join(n + c.t.NEW_SUFFIX for _, _, n in c.runs), c.fit_work or c.fit_work_refusal or '- (no tier_runs)',
              c.ft or c.roots_refusal, c.phot, c.py))
    for st in stages:
        STAGE_FN[st](c)
    if c.dry and c.would_refuse:           # the plan named a gate the real run would hit: the dry run exits 2 as well
        print('dry run: %d gate(s) would refuse: %s' % (len(c.would_refuse), '; '.join(c.would_refuse)[:800]), flush=True)
        from exopipe.phase3.core.record import EXIT_REFUSED
        return EXIT_REFUSED
    return None


# ---------------------------------------------------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------------------------------------------------

def _evals(evs):
    """Evaluations {(tier, kind): dir} keyed as the site's record keys them ('<tier>|<kind>')."""
    return {'%s|%s' % k: v for k, v in sorted(evs.items())}


def _samples_refusal(c):
    """T-SAMPLES: each of forced_tier_pages.SAMPLES has a label and a list photometry pins (SAMPLE_FILES, SAMPLE_SHA1: the
    evaluation's current targets are those lists, snapshot_forced_phot.current_targets); None when all do."""
    import hashlib
    t, ph = c.t, c.cfg.photometry
    for s in t.SAMPLES:
        if s not in t.SAMPLE_LABEL or s not in ph.SAMPLE_FILES or s not in ph.SAMPLE_SHA1:
            return ('T-SAMPLES sample %s has no label (forced_tier_pages.SAMPLE_LABEL) or no pinned list '
                    '(photometry.SAMPLE_FILES / SAMPLE_SHA1)' % s)
        p = os.path.join(c.cfg.paths.CLV15_ROOT, ph.SAMPLE_FILES[s])
        try:
            with open(p, 'rb') as f:
                h = hashlib.sha1(f.read()).hexdigest()
        except OSError as e:
            return 'T-SAMPLES the %s list %s cannot be read (%s)' % (s, p, e)
        if h != ph.SAMPLE_SHA1[s]:
            return ('T-SAMPLES %s sha1 %s != photometry.SAMPLE_SHA1[%s] %s - the list changed; update the pin first'
                    % (p, h[:12], s, ph.SAMPLE_SHA1[s][:12]))
    return None


def _eval_sample_gaps(t, evs):
    """T-EVAL-SAMPLES: [(tier, kind, sample, Stokes)] that an evaluation's thresholds.csv has no row for (the build refuses
    a missing threshold, tier_forced_page_20261027.py:218-220; checked before any in-place call)."""
    import csv
    have = set()
    for (tier, kind), ev in sorted(evs.items()):
        p = os.path.join(ev, 'thresholds.csv')
        if os.path.exists(p):
            with open(p) as f:
                have |= {(tier, kind, r['sample'], r['stokes']) for r in csv.DictReader(f) if r.get('tier') == tier}
    return [(tier, kind, s, S) for (tier, kind) in sorted(evs) for s in t.SAMPLES for S in ('I', 'V')
            if (tier, kind, s, S) not in have]


def _eval_inputs(c):
    """T-INPUT: [what is missing] of the inputs tier_eval reads: per tier and kind (sb = the subbands, wb = the widebands),
    every sample's forced_<sample>_<band>_<S>.npz in some <FT>/<tier>/<LST>/ (globs one level per LST, no recursion), and
    every sample's cutout directory <cutout root>/<sample> (the evaluation reads the cutouts' TIMES)."""
    import glob
    import re
    wb = re.compile(r'_(Red|Green|Blue|Full)_[IV]\.npz$')
    miss = []
    for (tier, kind) in sorted(c.t.EVAL_DIRS):
        lsts = sorted(glob.glob(os.path.join(c.ft, tier, '[0-2][0-9]h')))
        if not lsts:
            miss.append('no LST directory under %s' % os.path.join(c.ft, tier))
            continue
        for s in c.t.SAMPLES:
            n = sum(1 for d in lsts for p in glob.glob(os.path.join(d, 'forced_%s_*_[IV].npz' % s))
                    if bool(wb.search(p)) == (kind == 'wb'))
            if not n:
                miss.append('%s %s %s: no forced_%s_*.npz under %s/<LST>/' % (s, tier, kind, s, os.path.join(c.ft, tier)))
    for s in c.t.SAMPLES:
        if not os.path.isdir(os.path.join(c.phot, s)):
            miss.append('%s: no cutouts %s' % (s, os.path.join(c.phot, s)))
    return miss


def stage_eval(c):
    """tier_eval: the four stage-2 evaluations of SAMPLES (module docstring), one unit T_eval."""
    import glob
    import re
    t = c.t
    name = UNITS['tier_eval']
    nproc, too_many = _nproc(c, 'eval')
    evs = c.eval_dirs()
    script = K.script('tier_eval')
    argv_of = lambda tier, kind, out: [os.path.join(c.ft or '<FT>', tier), out, '--kind', kind, '--nproc', nproc, '--tier',
                                       tier, '--survey-cells', t.EVAL_SURVEY_CELLS, '--samples', ','.join(t.SAMPLES)]
    env = dict(t.EVAL_ENV, P3_PHOT_ROOT=c.phot)
    for (tier, kind), out in sorted(evs.items()):
        K.plan('[tier_eval] %s %s (%s)' % (os.path.basename(script), ' '.join(str(a) for a in argv_of(tier, kind, out)),
                                           ' '.join('%s=%s' % kv for kv in sorted(env.items()))))
    if c.dry:
        gates = [too_many, c.roots_refusal, _samples_refusal(c)]
        ex = [p for p in evs.values() if os.path.lexists(p)]
        gates.append(('T-G35 %s exist(s)' % ', '.join(ex)) if ex else None)
        for p in evs.values():
            try:
                K.refuse_production_target(p, 'tier evaluation')
            except Refusal as e:
                gates.append(str(e)[:200])
                break
        if not c.roots_refusal:
            miss = _eval_inputs(c)
            gates.append(('T-INPUT %s' % '; '.join(miss)[:400]) if miss else None)
        for g in gates:
            if g:
                K.plan('[tier_eval]   gate %s - refused' % g)
                c.would_refuse.append(g.split(' (')[0][:200])
        if not any(gates):
            K.plan('[tier_eval]   gates T-ROOTS, T-SAMPLES, T-G35, T-INPUT, T-WORKERS: pass')
        return
    want = dict(tier_forced_root=c.ft, tier_cutout_root=c.phot, samples=list(t.SAMPLES))
    if c.run.resume:
        from exopipe.phase3.core import record
        meta = (record.load_record(c.W, c.run.step, name) or {}).get('meta') or {}
        diff = ['%s %s -> %s' % (k, meta.get(k), v) for k, v in sorted(want.items()) if meta.get(k) != v]
        if diff:
            print('resume %s: not skipped (%s)' % (name, '; '.join(diff)), flush=True)
        elif K.resumed(c.run, name, None, tier='snapshot', stage='tier_eval', family='T', **want):
            return
    with c.run.unit(name, tier='snapshot', stage='tier_eval', family='T', tier_forced_root=c.ft, tier_cutout_root=c.phot,
                    samples=list(t.SAMPLES)) as u:
        if c.roots_refusal:
            u.refuse(c.roots_refusal)
        why = _samples_refusal(c)
        if why:
            u.refuse(why)
        for s in t.SAMPLES:
            u.add_input(os.path.join(c.cfg.paths.CLV15_ROOT, c.cfg.photometry.SAMPLE_FILES[s]))
        for p in evs.values():
            K.refuse_production_target(p, 'tier evaluation')
        if too_many:
            u.refuse(too_many)
        ex = [p for p in evs.values() if os.path.lexists(p)]
        if ex:
            u.refuse('T-G35 %s exist(s): an evaluation writes a new directory (production\'s run_evals.sh refuses one); name '
                     'a new --out' % ', '.join(ex))
        miss = _eval_inputs(c)
        if miss:
            u.refuse('T-INPUT %s - measure the samples first (photometry --tier deep|sublst --stages cutouts,cutmeasure)'
                     % '; '.join(miss)[:900])
        for tier in t.TIERS:
            for mp in sorted(glob.glob(os.path.join(c.ft, tier, '[0-2][0-9]h', 'forced_manifest.csv'))):
                u.add_input(mp)
        u.meta['host'] = K.hostname()
        for (tier, kind), out in sorted(evs.items()):
            u.plan(os.path.join(out, 'thresholds.csv'), 'forced_tier_eval')
        log = c.log('tier_eval')
        ok = re.compile(t.EVAL_OK_RE)
        lines = {}
        for (tier, kind), out in sorted(evs.items()):
            res = K.call(u, script, argv_of(tier, kind, out), c.cfg.forced.PYTHON_KEY['tier_eval'], log, env=env,
                         cwd=K.legacy_ops())
            done = [m for m in (ok.match(l) for l in res.lines()) if m and os.path.abspath(m.group(1)) == out]
            if res.rc != 0 or not done:
                u.refuse('T-EVAL-RUN the %s %s evaluation (exit %s) ended without its success line: %s'
                         % (tier, kind, res.rc, res.why()))
            lines['%s|%s' % (tier, kind)] = done[-1].group(0)
        u.meta['eval_lines'] = lines
        for (tier, kind), out in sorted(evs.items()):
            u.add_output(os.path.join(out, 'thresholds.csv'), 'forced_tier_eval', tier=tier, kind=kind,
                         samples=list(t.SAMPLES), inp=os.path.join(c.ft, tier))


def _exists(c):
    """The directories a new build may not find (T-G35)."""
    out = [p for p in (c.site, c.site.rstrip('/') + '_masters') if os.path.lexists(p)]
    return out


def _nproc(c, stage):
    """(processes, T-WORKERS refusal or None): --workers, else forced_tier_pages.NPROC; at most MAX_WORKERS (the in-place
    pool reads tier cutouts, Masters and epoch images from Lustre: at most 6 Lustre-heavy workers)."""
    n = c.args.workers or c.t.NPROC[stage]
    if n > c.t.MAX_WORKERS:
        return n, ('T-WORKERS %d processes > forced_tier_pages.MAX_WORKERS %d (the in-place pool reads tier cutouts, Masters '
                   'and epoch images from Lustre)' % (n, c.t.MAX_WORKERS))
    return n, None


def _site_cards(rp):
    """The event rows (cell, band, ts, ...) of a site's page_record.json, one per card."""
    with open(rp) as f:
        return [card.get('row') or {} for card in json.load(f).get('cards', [])]


def _excluded(c, rows):
    """T-EXCL: [(cell, band, ts)] of the rows on an excluded cell-band, core.exclusions read now; [] under
    forced_tier_pages.EXCLUDED_EVENTS 'build' (production: no such gate)."""
    if c.t.EXCLUDED_EVENTS != 'refuse':
        return []
    from exopipe.phase3.lib.forced.tier_snapshot import excluded_events
    return excluded_events(rows)


def _admitted(rows, bad):
    """The unit's admitted set, part of its resume identity: the sorted 'cell|band' of its events not excluded now. An
    exclusion added later changes it, so --resume re-runs the unit (whose gates then refuse) instead of skipping it."""
    drop = {'%s|%s' % (cell, band) for cell, band, _ts in bad}
    return sorted({'%s|%s' % (r.get('cell', ''), r.get('band', '')) for r in rows} - drop)


def _resumed(c, stage, rows, **want):
    """--resume for a family T unit. Its name is fixed (T_site, T_runs), so a record is adopted only when its meta names
    the same site (tier_runs: and review root and fit work) and, through K.resumed, the same admitted cell-bands, inputs,
    code and config, with outputs that pass their content tests now."""
    if not c.run.resume:
        return False
    from exopipe.phase3.core import record
    name = UNITS[stage]
    meta = (record.load_record(c.W, c.run.step, name) or {}).get('meta') or {}
    diff = ['%s %s -> %s' % (k, meta.get(k), v) for k, v in sorted(want.items()) if meta.get(k) != v]
    if diff:
        print('resume %s: not skipped (%s)' % (name, '; '.join(diff)), flush=True)
        return False
    if rows is None:
        print('resume %s: not skipped (no site record to read its events from)' % name, flush=True)
        return False
    return K.resumed(c.run, name, None, admitted=_admitted(rows, _excluded(c, rows)), tier='snapshot', stage=stage,
                     family='T', **want)


def stage_site(c):
    t = c.t
    name = UNITS['tier_site']
    nproc, too_many = _nproc(c, 'site')
    evs, sf, how, ev_refusal = c.eval_source()
    gl, env = c.page_globals(sf), c.page_env()
    K.plan('[tier_site] tier_forced_page_20261027.py build %s --nproc %d; verify %s (samples %s; evaluations: %s; %s%s)'
           % (c.site, nproc, c.site, ','.join(t.SAMPLES), how or ev_refusal, ' '.join('%s=%s' % kv for kv in sorted(env.items())),
              (' with globals %s' % json.dumps(gl, sort_keys=True)) if gl else ''))
    if c.dry:
        for g in (too_many, c.roots_refusal, ev_refusal, _samples_refusal(c)):
            if g:
                K.plan('[tier_site]   gate %s - refused' % g)
                c.would_refuse.append(g.split(' (')[0][:200])
        if 'tier_eval' in c.stages:
            K.plan('[tier_site]   gates T-EVAL, T-EVAL-SAMPLES, T-EXCL: the evaluations are made by tier_eval in this run '
                   '(read once it is done)')
        else:
            for (tier, kind), ev in sorted(evs.items()):
                miss = [f for f in t.EVAL_FILES if not os.path.exists(os.path.join(ev, f))]
                K.plan('[tier_site]   evaluation %s %s: %s' % (tier, kind, ev if not miss else 'MISSING %s (T-EVAL refuses)'
                                                                                     % miss))
                if miss:
                    c.would_refuse.append('T-EVAL %s %s' % (tier, kind))
            gaps = _eval_sample_gaps(t, evs)
            K.plan('[tier_site]   gate T-EVAL-SAMPLES: %s' % (('%d (tier, kind, sample, Stokes) without a threshold, e.g. %s - '
                                                              'refused' % (len(gaps), gaps[:4])) if gaps else
                                                             'every sample has its thresholds'))
            if gaps:
                c.would_refuse.append('T-EVAL-SAMPLES %d, e.g. %s' % (len(gaps), gaps[:2]))
        ex = _exists(c)
        K.plan('[tier_site]   gate T-G35: %s' % ('EXISTS %s - refused' % ex if ex else 'site and Masters tree new'))
        if ex:
            c.would_refuse.append('T-G35 %s' % ', '.join(ex))
        try:
            K.refuse_production_target(c.site, 'tier page site')
        except Refusal as e:
            K.plan('[tier_site]   %s - refused' % e)
            c.would_refuse.append(str(e)[:200])
        if 'tier_eval' not in c.stages:
            sel = selected_events(t, evs)
            K.plan('[tier_site]   %d events selected by the evaluations' % len(sel))
            if t.EXCLUDED_EVENTS == 'refuse':
                bad = _excluded(c, sel)
                K.plan('[tier_site]   gate T-EXCL: %d on an excluded cell-band%s' % (len(bad), (' - refused, e.g. %s' % bad[:3])
                                                                                      if bad else ''))
                if bad:
                    c.would_refuse.append('T-EXCL %d selected event(s)' % len(bad))
        return
    if _resumed(c, 'tier_site', selected_events(t, evs) if c.run.resume else None, site=c.site):
        return
    with c.run.unit(name, tier='snapshot', stage='tier_site', family='T', site=c.site, evals=_evals(evs),
                    samples=list(t.SAMPLES), tier_forced_root=c.ft, tier_cutout_root=c.phot) as u:
        K.refuse_production_target(c.site, 'tier page site')
        if too_many:
            u.refuse(too_many)
        for g in (c.roots_refusal, ev_refusal, _samples_refusal(c)):
            if g:
                u.refuse(g)
        ex = _exists(c)
        if ex:
            u.refuse('T-G35 %s exist(s): a build writes a new directory (tier_forced_page_20261027.py:215-216); name a new '
                     'one with --tier-site' % ', '.join(ex))
        for (tier, kind), ev in sorted(evs.items()):
            for fn in t.EVAL_FILES:
                p = os.path.join(ev, fn)
                if not os.path.exists(p):
                    u.refuse('T-EVAL the %s %s evaluation has no %s (%s)' % (tier, kind, fn, p))
                K.require_host(p, 'the %s %s evaluation' % (tier, kind))
                u.add_input(p)
        gaps = _eval_sample_gaps(t, evs)
        if gaps:
            u.refuse('T-EVAL-SAMPLES %d (tier, kind, sample, Stokes) of forced_tier_pages.SAMPLES without a threshold in the '
                     'evaluations (%s), e.g. %s - evaluate every sample first (--stages tier_eval)' % (len(gaps), how, gaps[:4]))
        if t.EXCLUDED_EVENTS not in ('refuse', 'build'):
            u.refuse('forced_tier_pages.EXCLUDED_EVENTS %r: refuse or build' % (t.EXCLUDED_EVENTS,))
        sel = selected_events(t, evs)      # after T-EVAL: every evaluation file is there
        bad = _excluded(c, sel)
        u.meta['excluded_events'] = len(bad)
        if bad:
            u.refuse('T-EXCL %d selected event(s) on an excluded cell-band, e.g. %s - the evaluation predates the '
                     'exclusion: redo it first (forced_tier_pages.EXCLUDED_EVENTS refuse)' % (len(bad), bad[:3]))
        u.set_admitted(_admitted(sel, bad))
        u.meta['host'] = K.hostname()
        u.meta['evals_from'] = how
        rp = os.path.join(c.site, 'page_record.json')
        u.plan(rp, 'forced_tier_site')
        log = c.log('tier_site')
        res = c.inplace(u, ['build', c.site, '--nproc', nproc], log, globals_=gl, env=env)
        if res.rc != 0 or not res.has(t.OK_LINES['build']):
            u.refuse('T-BUILD build (exit %s): %s' % (res.rc, res.why()))
        u.meta['build_line'] = res.line(t.OK_LINES['build'])
        res = c.inplace(u, ['verify', c.site], log, globals_=gl, test_stage=True, env=env)
        if res.rc != 0 or not res.has(t.OK_LINES['verify']):
            u.refuse('T-VERIFY the site\'s content test (verify, exit %s): %s' % (res.rc, res.why()))
        u.meta['verify_line'] = res.line(t.OK_LINES['verify'])
        u.add_output(rp, 'forced_tier_site', evals=_evals(evs), samples=list(t.SAMPLES))


def _site_check(c):
    """(page_record path, None, how) for the site tier_runs reads, else (None, T-SITE refusal, None): this --out's T_site
    record when it names the site (DESIGN 11.5), else (a site production built) the site's content test run now as an
    input check."""
    from exopipe.phase3.core import record, validate
    rp = os.path.join(c.site, 'page_record.json')
    rec = record.load_record(c.W, c.run.step, UNITS['tier_site'])
    if rec is not None:
        if rec.get('status') != 'done':
            return None, 'T-SITE the tier_site record is %s (%s), not done' % (rec.get('status'),
                                                                               (rec.get('reason') or '')[:200]), None
        if rp in K.outputs_of(rec, 'forced_tier_site'):
            return rp, None, 'record %s' % UNITS['tier_site']
    if not os.path.exists(rp):
        return None, 'T-SITE no site %s (no tier_site record of this --out names it)' % rp, None
    tests = validate.run('forced_tier_site', rp, evals=_evals(c.eval_source()[0]), samples=list(c.t.SAMPLES))
    bad = [x for x in tests if not x['passed']]
    if bad:
        return None, 'T-SITE the site %s fails its content test: %s' % (c.site, '; '.join(x['detail'][:300] for x in bad)), None
    return rp, None, 'content-tested input (%s)' % '; '.join(str(x['detail'])[:200] for x in tests)


def _site_input(c, u):
    """The site tier_runs reads (_site_check), recorded as the unit's input; refused (T-SITE) otherwise."""
    rp, why, how = _site_check(c)
    if why:
        u.refuse(why)
    u.meta['site_from'] = how
    u.add_input(rp)
    return rp


def _site_roots(c, rp):
    """(the chain's environment {P3_FORCED_TIERS, P3_PHOT_ROOT}, T-SITE / T-ROOTS refusal or None): the tier files and cutouts
    the site was built on (its page_record inputs tiers / phot: the chain re-reads the cards' targets and planes there); a
    --tier-forced-root / --tier-cutout-root naming others refuses."""
    try:
        with open(rp) as f:
            inp = json.load(f).get('inputs') or {}
    except (OSError, ValueError) as e:
        return None, 'T-SITE the site record %s cannot be read (%s)' % (rp, e)
    ft, phot = inp.get('tiers'), inp.get('phot')
    if not (ft and phot and os.path.isdir(ft) and os.path.isdir(phot)):
        return None, 'T-SITE the site record names tier files %r and cutouts %r: not both directories here' % (ft, phot)
    for flag, attr, have in (('--tier-forced-root', 'tier_forced_root', ft), ('--tier-cutout-root', 'tier_cutout_root', phot)):
        given = getattr(c.args, attr, None)
        if given and os.path.abspath(given) != os.path.abspath(have):
            return None, 'T-ROOTS %s %s, but the site was built on %s' % (flag, given, have)
    return c.page_env(ft, phot), None


def stage_runs(c):
    t = c.t
    name = UNITS['tier_runs']
    nproc, too_many = _nproc(c, 'runs')
    gl = {t.FIT_WORK_GLOBAL: c.fit_work} if c.fit_work and c.fit_work != t.FIT_WORK_PRODUCTION else {}
    K.plan('[tier_runs] tier_forced_page_20261027.py chain %s --root %s --nproc %d%s; verify-chain %s --root %s (review host)'
           % (c.site, c.root, nproc, (' with %s=%s' % (t.FIT_WORK_GLOBAL, c.fit_work)) if gl else '', c.site, c.root))
    rp = os.path.join(c.site, 'page_record.json')
    if c.dry:
        here = K.hostname()
        off = [x for x in (('the review host is %s' % c.review_host) if c.review_host and c.review_host != here else '',
                           ('the review root is node-local on %s' % c.root_host) if c.root_host and c.root_host != here else '')
               if x]
        K.plan('[tier_runs]   gate T-HOST: review host %s, this host %s%s' % (c.review_host or 'shared file system', here,
                                                                           (' - refused (%s)' % '; '.join(off)) if off else ''))
        if off:
            c.would_refuse.append('T-HOST %s' % '; '.join(off))
        for why in (too_many, c.fit_work_refusal and 'T-WORK %s' % c.fit_work_refusal):
            if why:
                K.plan('[tier_runs]   gate %s - refused' % why)
                c.would_refuse.append(why.split(' (')[0][:200])
        for target, what in ((c.root, 'review root'), (c.fit_work, 'fit work directory')):
            if target:
                try:
                    K.refuse_production_target(target, what)
                except Refusal as e:
                    K.plan('[tier_runs]   %s - refused' % e)
                    c.would_refuse.append(str(e)[:200])
        for _t, _s, n, D in c.new_dirs():
            ex = [p for p in (D, D + '_masters') if os.path.lexists(p)]
            K.plan('[tier_runs]   gate T-G35: %s %s' % (D, ('EXISTS %s - refused' % ex) if ex else 'new'))
            if ex:
                c.would_refuse.append('T-G35 %s' % ', '.join(ex))
        if 'tier_site' in c.stages:
            K.plan('[tier_runs]   gates T-SITE, T-EXCL: the site is built by tier_site in this run (its gates apply)')
        else:
            site_rp, why, how = _site_check(c)
            if not why:
                env, why = _site_roots(c, site_rp)
                how = '%s; the chain reads %s' % (how, env)
            K.plan('[tier_runs]   gate T-SITE: %s' % (('%s - refused' % why) if why else how[:300]))
            if why:
                c.would_refuse.append(why[:200])
            else:
                bad = _excluded(c, _site_cards(site_rp))
                K.plan('[tier_runs]   gate T-EXCL: %d site card(s) on an excluded cell-band%s'
                       % (len(bad), (' - refused, e.g. %s' % bad[:3]) if bad else ''))
                if bad:
                    c.would_refuse.append('T-EXCL %d site card(s)' % len(bad))
        return
    rows = _site_cards(rp) if (c.run.resume and os.path.exists(rp)) else None
    if _resumed(c, 'tier_runs', rows, site=c.site, review_root=c.root, fit_work=c.fit_work):
        return
    with c.run.unit(name, tier='snapshot', stage='tier_runs', family='T', site=c.site, review_root=c.root,
                    fit_work=c.fit_work) as u:
        if c.review_host and c.review_host != K.hostname():
            u.refuse('T-HOST tier_runs runs on the review host %s (the localization reads %s there), this is %s: exopipe3 run '
                     'submit --nodes %s:1 -- forced --tier snapshot --stages tier_runs ...'
                     % (c.review_host, c.cfg.paths.COADD_SB_ROOT, K.hostname(), c.review_host))
        if c.root_host and c.root_host != K.hostname():
            u.refuse('T-HOST the review root %s is node-local on %s, this is %s' % (c.root, c.root_host, K.hostname()))
        K.refuse_production_target(c.root, 'review root')
        if c.fit_work_refusal:
            u.refuse('T-WORK the fit\'s work directory (one plane FITS per event): %s' % c.fit_work_refusal)
        K.refuse_production_target(c.fit_work, 'fit work directory')
        if too_many:
            u.refuse(too_many)
        news = c.new_dirs()
        ex = [p for _t, _s, _n, D in news for p in (D, D + '_masters') if os.path.lexists(p)]
        if ex:
            u.refuse('T-G35 %s exist(s): the chain writes new runs (tier_forced_page_20261027.py:946-947); install or remove '
                     'the earlier ones first' % ', '.join(ex))
        _site_input(c, u)
        env, why = _site_roots(c, rp)
        if why:
            u.refuse(why)
        u.meta['env'] = env
        rows = _site_cards(rp)
        bad = _excluded(c, rows)           # a site production built had no exclusion gate; one built before an exclusion
        u.meta['excluded_events'] = len(bad)
        if bad:
            u.refuse('T-EXCL %d site card(s) on an excluded cell-band, e.g. %s - rebuild the site (tier_site) from evaluations '
                     'redone after the exclusion (forced_tier_pages.EXCLUDED_EVENTS refuse)' % (len(bad), bad[:3]))
        u.set_admitted(_admitted(rows, bad))
        u.meta['host'] = K.hostname()
        # the chain copies each installed run's judgements/ into its <run>.new (tier_forced_page_20261027.py:1096-1098);
        # labels made after the chain are re-synced by production's swap (:1250-1251), which is not ported
        u.meta['carried_judgements'] = [n for _t, _s, n, _D in news if os.path.isdir(os.path.join(c.root, n, 'judgements'))]
        qs = [(tier, S, os.path.join(D, 'queue_%s.txt' % S)) for tier, S, _n, D in news]
        for _tier, _S, q in qs:
            u.plan(q, 'forced_tier_run')
        os.makedirs(c.root, exist_ok=True)
        log = c.log('tier_runs')
        res = c.inplace(u, ['chain', c.site, '--root', c.root, '--nproc', nproc], log, globals_=gl, stage='tier_runs', env=env)
        done = [l for l in res.lines() if l.startswith(t.OK_LINES['chain'])]
        if res.rc != 0 or len(done) != len(news):
            u.refuse('T-CHAIN chain: %d of %d runs ended with %r: %s' % (len(done), len(news), t.OK_LINES['chain'], res.why()))
        u.meta['chain_lines'] = done
        res = c.inplace(u, ['verify-chain', c.site, '--root', c.root], log, test_stage=True, stage='tier_runs', env=env)
        if res.rc != 0 or not res.has(t.OK_LINES['verify_chain']):
            u.refuse('T-VERIFY-CHAIN the runs\' content test (verify-chain, exit %s): %s' % (res.rc, res.why()))
        u.meta['verify_line'] = res.line(t.OK_LINES['verify_chain'])
        for tier, S, q in qs:
            u.add_output(q, 'forced_tier_run', site=c.site, tier=tier, stokes=S)


STAGE_FN = {'tier_eval': stage_eval, 'tier_site': stage_site, 'tier_runs': stage_runs}
