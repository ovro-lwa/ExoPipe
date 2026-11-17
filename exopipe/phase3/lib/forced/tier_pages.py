"""exopipe3 forced --tier snapshot --stages tier_site,tier_runs: family T, the HWJ / 10 pc forced-photometry pages of the
10-minute and deep tiers (Gregg 2026-10-27, at the terminal: "HWJ and 10 pc forced photometry at 10-minute and deep - that is
very valuable"). Port of the dated driver exopipe_ops tier_forced_page_20261027.py (ledger 3079 site, 3081 chain, 3394-3395
Epoch panels), run IN PLACE through core.legacy.run - every subcommand unchanged - with the site interpreter
forced.PYTHON_KEY['tier_pages'] (py36: the fit imports bdsf).

Campaign stages, named only (never a default), one family per invocation (as families W and N):
  tier_site  unit T_site: `build SITE --nproc N`, then its content test `verify SITE` (HWJ10PC-TIERS-PAGE-OK: the events are
             exactly the clean measurements at or above their Gaussian thresholds in the four evaluations, every card's forced
             numbers recomputed from the tier files, every yellow reading from the saved crops). SITE = --tier-site, else
             <out>/forced_tier_pages.SITE_NAME; its Masters link tree is SITE_masters. Output SITE/page_record.json
             (class forced_tier_site).
  tier_runs  unit T_runs: `chain SITE --root ROOT --nproc N` (the four review runs ROOT/Run_SNAPFORCED_{V,I}{10min,deep}_gen5.new
             from the site's cards: fit in each event's own image, localization, association, render, the served tree), then
             its content test `verify-chain SITE --root ROOT` (TIER-CHAIN-OK). Outputs ROOT/<run>.new/queue_<S>.txt (class
             forced_tier_run, one per run). ROOT = --snap-review-root, else <out>/forced_snapshot.REVIEW_SUBDIR_UNDER_OUT under
             --out, else paths.SNAPSHOT_REVIEW_ROOT. The site is read through the T_site record of this --out when there is
             one, else content-tested as an input (the served site was built by production).
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
Gates (exit 2, named in the record): T-G35 the site directory, its Masters tree, a <run>.new or its Masters tree exists (a
name is used once); T-EVAL an evaluation file missing; T-EXCL an event on an excluded cell-band (forced_tier_pages.
EXCLUDED_EVENTS 'refuse'); T-HOST tier_runs off the review host (the localization reads paths.COADD_SB_ROOT) or the review
root node-local to another host; T-SITE the site's record or content test failing; T-BUILD / T-CHAIN the in-place success
line missing; T-VERIFY / T-VERIFY-CHAIN the in-place content test failing; a production target while paths.PRODUCTION_WRITES
is False."""
import json
import os

from exopipe.phase3.core.record import Refusal
from exopipe.phase3.lib.forced import common as K

UNITS = {'tier_site': 'T_site', 'tier_runs': 'T_runs'}
# the in-place driver: runs one function of a production module unchanged, module globals set only when passed
DRIVER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'jupiter', '_inplace.py')


def add_arguments(p):
    g = p.add_argument_group('snapshot tier, family T (the HWJ / 10 pc forced-photometry pages of the 10-minute and deep '
                             'tiers: --stages tier_site,tier_runs)')
    g.add_argument('--tier-site', metavar='DIR', help='the site directory tier_site builds and tier_runs reads (default '
                                                      '<out>/forced_tier_pages.SITE_NAME); a build never reuses one')


def default_out(args):
    return K.cfg().forced_tier_pages.SITE_ROOT


def run_names(t=None):
    """[(tier, Stokes, run name)] in the in-place chain's order (tier_forced_page_20261027.py:944-945)."""
    t = t or K.cfg().forced_tier_pages
    return [(tier, S, t.RUN_NAME.format(S=S, tier=tier)) for tier in t.TIERS for S in ('V', 'I')]


def selected_events(t=None, ev_dirs=None):
    """The rows the in-place build lists (tier_forced_page_20261027.py:41-65): per tier and kind, the clean rows of the
    evaluation's targets_ge4.csv of the two samples at or above thr_survey_gauss of their (sample, tier, kind, Stokes)
    (Stokes V: |S/N|). Own code, for the dry-run count and the exclusion gate."""
    import csv
    t = t or K.cfg().forced_tier_pages
    out = []
    for (tier, kind), ev in sorted((ev_dirs or t.EVAL_DIRS).items()):
        tp, rp = os.path.join(ev, 'thresholds.csv'), os.path.join(ev, 'targets_ge4.csv')
        if not (os.path.exists(tp) and os.path.exists(rp)):
            continue
        with open(tp) as f:
            th = {(r['sample'], r['stokes']): r for r in csv.DictReader(f) if r['tier'] == tier and r['sample'] in t.SAMPLES}
        with open(rp) as f:
            for r in csv.DictReader(f):
                if r['tier'] != tier or r['screen'] != 'clean' or r['sample'] not in t.SAMPLES:
                    continue
                th_ = th.get((r['sample'], r['stokes']))
                if th_ is None:
                    continue
                v = float(r['snr'])
                if (abs(v) if r['stokes'] == 'V' else v) >= float(th_['thr_survey_gauss']):
                    out.append(dict(r, kind=kind))
    return out


class TCtx:
    def __init__(self, args, run):
        self.args, self.run, self.cfg = args, run, K.cfg()
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
        self.fit_work = self._fit_work()
        self.py = self.cfg.forced.PYTHON_KEY['tier_pages']
        self.runs = run_names(t)

    def _fit_work(self):
        """--work, else the production literal for a production run, else the site scratch root of this host, else the
        site root (as tier_snapshot._scratch)."""
        sub = 'tier_forced_fit_work'
        if self.args.work:
            return os.path.join(os.path.abspath(self.args.work), sub)
        if not self.args.out:
            return self.t.FIT_WORK_PRODUCTION
        from exopipe.phase3.core import fitsio
        try:
            return os.path.join(fitsio.scratch_root(None), sub)
        except Refusal:
            return os.path.join(self.W, sub)

    def log(self, stage):
        return K.records_log(self.run, UNITS[stage])

    def new_dirs(self):
        """[(tier, Stokes, name, <root>/<name>.new)] of the chain's four runs."""
        return [(tier, S, n, os.path.join(self.root, n + self.t.NEW_SUFFIX)) for tier, S, n in self.runs]

    def inplace(self, u, argv, log, globals_=None, test_stage=False, stage=None):
        """One subcommand of the in-place script through the driver (core.legacy.run). The script and the driver are the
        unit's inputs, their sha1 its legacy code."""
        script = K.script('tier_pages')
        res_path = os.path.join(os.path.dirname(log), '%s.inplace_%d.json' % (os.path.basename(log)[:-4], len(
            u.meta.get('inplace', []))))
        dargs = ['--module', script, '--call', 'main', '--argv', json.dumps([str(a) for a in argv]),
                 '--globals', json.dumps(globals_ or {}), '--result', res_path] + (['--test-stage'] if test_stage else [])
        res = K.call(u, DRIVER, dargs, self.py, log, cwd=K.legacy_ops(), stage=stage)
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
        u.meta.setdefault('inplace', []).append({'argv': [str(a) for a in argv], 'globals': globals_ or {},
                                                 'kind': result.get('kind'), 'reason': (result.get('reason') or '')[:300],
                                                 'wall_s': result.get('wall_s'), 'rc': res.rc})
        return res


def run(args, run, stages):
    from exopipe.phase3.lib.forced.wide_stages import gate_args
    gate_args(args, 'T')
    c = TCtx(args, run)
    K.plan('family T (HWJ / 10 pc forced-photometry pages, 10-minute and deep tiers): stages %s; site %s; review root %s '
           '(review host %s); runs %s; fit work %s; tier_forced_page_20261027.py in place (site PYTHON %r)'
           % (','.join(stages), c.site, c.root, c.review_host or 'shared', ', '.join(n + c.t.NEW_SUFFIX for _, _, n in c.runs),
              c.fit_work, c.py))
    for st in stages:
        STAGE_FN[st](c)
    return None


# ---------------------------------------------------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------------------------------------------------

def _evals(t):
    """forced_tier_pages.EVAL_DIRS keyed as the site's record keys them ('<tier>|<kind>')."""
    return {'%s|%s' % k: v for k, v in sorted(t.EVAL_DIRS.items())}


def _exists(c):
    """The directories a new build may not find (T-G35)."""
    out = [p for p in (c.site, c.site.rstrip('/') + '_masters') if os.path.lexists(p)]
    return out


def stage_site(c):
    t = c.t
    name = UNITS['tier_site']
    nproc = c.args.workers or t.NPROC['site']
    K.plan('[tier_site] tier_forced_page_20261027.py build %s --nproc %d; verify %s' % (c.site, nproc, c.site))
    if c.dry:
        for (tier, kind), ev in sorted(t.EVAL_DIRS.items()):
            miss = [f for f in t.EVAL_FILES if not os.path.exists(os.path.join(ev, f))]
            K.plan('[tier_site]   evaluation %s %s: %s' % (tier, kind, ev if not miss else 'MISSING %s (T-EVAL refuses)' % miss))
        ex = _exists(c)
        K.plan('[tier_site]   gate T-G35: %s' % ('EXISTS %s - refused' % ex if ex else 'site and Masters tree new'))
        sel = selected_events(t)
        K.plan('[tier_site]   %d events selected by the evaluations' % len(sel))
        if t.EXCLUDED_EVENTS == 'refuse':
            from exopipe.phase3.lib.forced.tier_snapshot import excluded_events
            bad = excluded_events(sel)
            K.plan('[tier_site]   gate T-EXCL: %d on an excluded cell-band%s' % (len(bad), (' - refused, e.g. %s' % bad[:3])
                                                                                  if bad else ''))
        return
    if K.resumed(c.run, name, None, tier='snapshot', stage='tier_site'):
        return
    with c.run.unit(name, tier='snapshot', stage='tier_site', family='T', site=c.site,
                    evals=_evals(t)) as u:
        K.refuse_production_target(c.site, 'tier page site')
        ex = _exists(c)
        if ex:
            u.refuse('T-G35 %s exist(s): a build writes a new directory (tier_forced_page_20261027.py:215-216); name a new '
                     'one with --tier-site' % ', '.join(ex))
        for (tier, kind), ev in sorted(t.EVAL_DIRS.items()):
            for fn in t.EVAL_FILES:
                p = os.path.join(ev, fn)
                if not os.path.exists(p):
                    u.refuse('T-EVAL the %s %s evaluation has no %s (%s)' % (tier, kind, fn, p))
                K.require_host(p, 'the %s %s evaluation' % (tier, kind))
                u.add_input(p)
        if t.EXCLUDED_EVENTS not in ('refuse', 'build'):
            u.refuse('forced_tier_pages.EXCLUDED_EVENTS %r: refuse or build' % (t.EXCLUDED_EVENTS,))
        if t.EXCLUDED_EVENTS == 'refuse':
            from exopipe.phase3.lib.forced.tier_snapshot import excluded_events
            bad = excluded_events(selected_events(t))
            u.meta['excluded_events'] = len(bad)
            if bad:
                u.refuse('T-EXCL %d selected event(s) on an excluded cell-band, e.g. %s - the evaluation predates the '
                         'exclusion: redo it first (forced_tier_pages.EXCLUDED_EVENTS refuse)' % (len(bad), bad[:3]))
        u.meta['host'] = K.hostname()
        rp = os.path.join(c.site, 'page_record.json')
        u.plan(rp, 'forced_tier_site')
        log = c.log('tier_site')
        res = c.inplace(u, ['build', c.site, '--nproc', nproc], log)
        if not res.has(t.OK_LINES['build']):
            u.refuse('T-BUILD build: %s' % res.why())
        u.meta['build_line'] = res.line(t.OK_LINES['build'])
        res = c.inplace(u, ['verify', c.site], log, test_stage=True)
        if not res.has(t.OK_LINES['verify']):
            u.refuse('T-VERIFY the site\'s content test (verify): %s' % res.why())
        u.meta['verify_line'] = res.line(t.OK_LINES['verify'])
        u.add_output(rp, 'forced_tier_site', evals=_evals(t))


def _site_input(c, u):
    """The site tier_runs reads: this --out's T_site record when it names the site (DESIGN 11.5), else (a site production
    built) the site's content test run now as an input check."""
    from exopipe.phase3.core import record, validate
    rp = os.path.join(c.site, 'page_record.json')
    rec = record.load_record(c.W, c.run.step, UNITS['tier_site'])
    if rec is not None:
        if rec.get('status') != 'done':
            u.refuse('T-SITE the tier_site record is %s (%s), not done' % (rec.get('status'), (rec.get('reason') or '')[:200]))
        if rp in K.outputs_of(rec, 'forced_tier_site'):
            u.meta['site_from'] = 'record %s' % UNITS['tier_site']
            u.add_input(rp)
            return rp
    if not os.path.exists(rp):
        u.refuse('T-SITE no site %s (no tier_site record of this --out names it)' % rp)
    tests = validate.run('forced_tier_site', rp, evals=_evals(c.t))
    bad = [x for x in tests if not x['passed']]
    if bad:
        u.refuse('T-SITE the site %s fails its content test: %s' % (c.site, '; '.join(x['detail'][:300] for x in bad)))
    u.meta['site_from'] = 'content-tested input (%s)' % '; '.join(str(x['detail'])[:200] for x in tests)
    u.add_input(rp)
    return rp


def stage_runs(c):
    t = c.t
    name = UNITS['tier_runs']
    nproc = c.args.workers or t.NPROC['runs']
    gl = {t.FIT_WORK_GLOBAL: c.fit_work} if c.fit_work != t.FIT_WORK_PRODUCTION else {}
    K.plan('[tier_runs] tier_forced_page_20261027.py chain %s --root %s --nproc %d%s; verify-chain %s --root %s (review host)'
           % (c.site, c.root, nproc, (' with %s=%s' % (t.FIT_WORK_GLOBAL, c.fit_work)) if gl else '', c.site, c.root))
    if c.dry:
        K.plan('[tier_runs]   gate T-HOST: review host %s, this host %s' % (c.review_host or 'shared file system', K.hostname()))
        for _t, _s, n, D in c.new_dirs():
            K.plan('[tier_runs]   gate T-G35: %s %s' % (D, 'EXISTS - refused' if os.path.lexists(D) else 'new'))
        return
    if K.resumed(c.run, name, None, tier='snapshot', stage='tier_runs'):
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
        K.refuse_production_target(c.fit_work, 'fit work directory')
        news = c.new_dirs()
        ex = [p for _t, _s, _n, D in news for p in (D, D + '_masters') if os.path.lexists(p)]
        if ex:
            u.refuse('T-G35 %s exist(s): the chain writes new runs (tier_forced_page_20261027.py:946-947); install or remove '
                     'the earlier ones first' % ', '.join(ex))
        _site_input(c, u)
        u.meta['host'] = K.hostname()
        u.meta['carried_judgements'] = [n for _t, _s, n, _D in news if os.path.isdir(os.path.join(c.root, n, 'judgements'))]
        qs = [(tier, S, os.path.join(D, 'queue_%s.txt' % S)) for tier, S, _n, D in news]
        for _tier, _S, q in qs:
            u.plan(q, 'forced_tier_run')
        os.makedirs(c.root, exist_ok=True)
        log = c.log('tier_runs')
        res = c.inplace(u, ['chain', c.site, '--root', c.root, '--nproc', nproc], log, globals_=gl, stage='tier_runs')
        done = [l for l in res.lines() if l.startswith(t.OK_LINES['chain'])]
        if res.rc != 0 or len(done) != len(news):
            u.refuse('T-CHAIN chain: %d of %d runs ended with %r: %s' % (len(done), len(news), t.OK_LINES['chain'], res.why()))
        u.meta['chain_lines'] = done
        res = c.inplace(u, ['verify-chain', c.site, '--root', c.root], log, test_stage=True, stage='tier_runs')
        if not res.has(t.OK_LINES['verify_chain']):
            u.refuse('T-VERIFY-CHAIN the runs\' content test (verify-chain): %s' % res.why())
        u.meta['verify_line'] = res.line(t.OK_LINES['verify_chain'])
        for tier, S, q in qs:
            u.add_output(q, 'forced_tier_run', site=c.site, tier=tier, stokes=S)


STAGE_FN = {'tier_site': stage_site, 'tier_runs': stage_runs}
