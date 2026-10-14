"""exopipe3 forced --tier snapshot: family C, the snapshot forced-photometry EVENT PANELS of one (Stokes, cadence) run -
build -> fit -> tie -> loc (review host) -> assoc -> render -> verify (review host). Port of
snapshot_forced_pages_run_20261007.sh (the launcher of the 2026-10-07/08 re-serve); every subcommand runs in place through
core.legacy.run (snapshot_forced_panels.py, snapshot_forced_assoc.py, test_snapshot_forced_reserve.py). Spec forced.md 1.3,
4.3, 5, 7.

Directories: --out is the work root W (default forced_snapshot.WORK_ROOT); the work run is RUN = W/<NAME>; the records are
W/_records/forced/C_<stage>_<NAME>.json. The review copy is D = <review root>/<NAME>: paths.SNAPSHOT_REVIEW_ROOT for a
production run, <out>/<forced_snapshot.REVIEW_SUBDIR_UNDER_OUT> (or --snap-review-root) under --out. loc and verify run on the
REVIEW HOST = the host holding paths.COADD_SB_ROOT (the deep masters and co-add record the localization and the filter panel
read; lwacalim07 today); the other stages run on any node that can read the inputs, the work run staying on the node that
built it when it is node-local.

The run name is used once (G35): a stage that fails is re-run on the same RUN (--stages <stage> ...); a new build needs a new
--snap-name. Per hour (--epochs E): one run per (Stokes, cadence) of the selection (default I and V, 10s and 1min), named
forced_snapshot.EPOCH_NAME, built from restricted copies of the two evaluations (that hour's rows); a combination without an
event in the hour ends no-input; such runs are recorded partial and not servable. --cells <LST>_<date>[_<band>] restricts a
single run the same way (an investigation).
"""
import csv
import glob
import os
import shutil

from exopipe.phase3.core.record import Refusal, NoInput
from exopipe.phase3.lib.forced import common as K


def add_arguments(p):
    g = p.add_argument_group('snapshot tier (family C: forced event panels of one Stokes and one cadence per run)')
    g.add_argument('--snap-cadence', choices=('10s', '1min'),
                   help='the run\'s one cadence (SNAP-FORCED-ONE-TIER-PER-PAGE); required without --epochs')
    g.add_argument('--snap-name', metavar='NAME',
                   help='run name Run_SNAPFORCED_<S><cadence>_<TAG>; required without --epochs; the build refuses a name '
                        'whose work run or review copy exists')
    g.add_argument('--snap-eval-sb', metavar='DIR', help='subband evaluation of the photometry step (default '
                                                          'forced_snapshot.EVAL_SB, the served generation eval_20261007)')
    g.add_argument('--snap-eval-wb', metavar='DIR', help='wideband evaluation (default forced_snapshot.EVAL_WB)')
    g.add_argument('--snap-zone-flags', metavar='FILE',
                   help='independent zone recomputation (test_snapshot_forced_reserve.py zones --write) read by the '
                        'served-zones test Z2 at render (default forced_snapshot.ZONE_FLAGS)')
    g.add_argument('--snap-review-root', metavar='DIR',
                   help='review root under a non-production --out (default <out>/review); production: '
                        'paths.SNAPSHOT_REVIEW_ROOT')
    g.add_argument('--snap-masters-root', metavar='DIR',
                   help='master link tree of the panels (default <work root>/masters_<NAME>)')
    g.add_argument('--snap-no-assoc', action='store_true',
                   help='no association (assoc skipped, render --no-assoc, verify without it): recorded not servable')


def default_out(args):
    return K.cfg().forced_snapshot.WORK_ROOT


def combos(args):
    """[(Stokes, cadence, name)] of this invocation (module docstring)."""
    s = K.cfg().forced_snapshot
    ep = K.epochs(args)
    st = [x.strip() for x in (args.stokes or ('I,V' if ep else '')).split(',') if x.strip()]
    if not st or any(x not in ('I', 'V') for x in st):
        raise Refusal('the snapshot forced chain runs ONE Stokes per run: --stokes I or --stokes V (with --epochs: I, V '
                      'or I,V)')
    cads = [args.snap_cadence] if args.snap_cadence else (list(s.CADENCES) if ep else [])
    if not cads:
        raise Refusal('G24 --snap-cadence 10s or 1min is required (one cadence per run, SNAP-FORCED-ONE-TIER-PER-PAGE)')
    if not ep:
        if len(st) != 1:
            raise Refusal('one Stokes per run: --stokes I or --stokes V')
        if not args.snap_name:
            raise Refusal('--snap-name NAME is required (Run_SNAPFORCED_<S><cadence>_<TAG>)')
        return [(st[0], cads[0], args.snap_name)]
    if len(ep) != 1:
        raise Refusal('--epochs takes one epoch per invocation for the snapshot tier (one run per Stokes and cadence each)')
    n = len(st) * len(cads)
    if args.snap_name and n > 1:
        raise Refusal('--snap-name names one run; --epochs with %d Stokes x cadence combinations names them %s'
                      % (n, s.EPOCH_NAME))
    e = '%s_%s' % ep[0]
    return [(S, cad, args.snap_name or s.EPOCH_NAME.format(S=S, cadence=cad, epoch=e)) for S in st for cad in cads]


class Ctx:
    def __init__(self, args, run, S, cad, name):
        self.args, self.run, self.cfg = args, run, K.cfg()
        s, paths = self.cfg.forced_snapshot, self.cfg.paths
        self.dry = run.dry_run
        self.S, self.cad, self.name = S, cad, name
        if cad not in s.CADENCES:
            raise Refusal('cadence %s not in forced_snapshot.CADENCES %s' % (cad, s.CADENCES))
        self.W = os.path.abspath(run.out_root)
        self.RUN = os.path.join(self.W, name)
        if args.out:
            rr = args.snap_review_root or os.path.join(self.W, s.REVIEW_SUBDIR_UNDER_OUT)
        else:
            rr = args.snap_review_root or paths.SNAPSHOT_REVIEW_ROOT
        self.D = os.path.join(rr, name)
        self.review_host = K.host_of(paths.COADD_SB_ROOT)      # loc / verify run here (the deep masters, coadd record)
        self.d_host = K.host_of(self.D)
        self.ev_sb = args.snap_eval_sb or s.EVAL_SB
        self.ev_wb = args.snap_eval_wb or s.EVAL_WB
        self.items = self._items()
        self.ev_used = (self.ev_sb, self.ev_wb)
        if self.items:
            base = os.path.join(self.W, 'restricted_eval_%s' % name)
            self.ev_used = (os.path.join(base, 'sb'), os.path.join(base, 'wb'))
        self.flags = args.snap_zone_flags or s.ZONE_FLAGS
        self.masters = args.snap_masters_root or os.path.join(self.W, 'masters_%s' % name)
        self.servable = not self.items and not args.snap_no_assoc
        self.py = self.cfg.forced.PYTHON_KEY['snapshot']

    def _items(self):
        from exopipe.phase3.core import cells as C
        out = [(l, d, None) for l, d in K.epochs(self.args)]
        if self.args.cells:
            out += [C.parse_night_cell(x) for x in C.expand(self.args.cells, kind='night cell')]
        return out

    def unit_name(self, stage):
        return 'C_%s_%s' % (stage, self.name)

    def env(self, **extra):
        e = dict(self.cfg.forced_snapshot.FLOW_ENV)
        e.update(extra)
        return e

    def log(self, stage):
        return K.records_log(self.run, self.unit_name(stage))

    def work_host(self):
        """The host holding RUN when it is node-local (the build unit's host), else None."""
        if not K.is_node_local(self.RUN):
            return None
        rec = K.load_done(self.W, self.run.step, self.unit_name('build'), 'the build')
        return (rec.get('meta') or {}).get('host') or (rec.get('host') or '').split('.')[0] or None

    def need_work_here(self, what):
        h = self.work_host()
        if h and h != K.hostname():
            raise Refusal('%s: the work run %s is node-local on %s, this is %s - run this stage there'
                          % (what, self.RUN, h, K.hostname()))

    def need_review_host(self, what):
        if self.review_host and self.review_host != K.hostname():
            raise Refusal('%s runs on the review host %s (the deep masters and %s are node-local there), this is %s: '
                          'exopipe3 run submit --nodes %s:1 -- forced --tier snapshot --stages %s ...'
                          % (what, self.review_host, self.cfg.paths.COADD_SB_ROOT, K.hostname(), self.review_host, what))
        if self.d_host and self.d_host != (self.review_host or K.hostname()):
            raise Refusal('the review copy %s is node-local on %s, not on the review host %s'
                          % (self.D, self.d_host, self.review_host))


def run(args, run):
    stages = K.selected_stages(args, 'snapshot')
    for S, cad, name in combos(args):
        c = Ctx(args, run, S, cad, name)
        K.plan('tier snapshot, Stokes %s, cadence %s, run %s: stages %s; work %s, review %s (review host %s); evaluations '
               '%s + %s%s' % (S, cad, name, ','.join(stages), c.RUN, c.D, c.review_host or 'shared', c.ev_sb, c.ev_wb,
                              ('; RESTRICTED to %s (partial, not servable)' % c.items) if c.items else ''))
        if c.items and 'build' in stages and not _has_events(c):
            K.plan('  no Stokes %s %s event in the restricted evaluations: no-input' % (S, cad))
            if not c.dry:
                with run.unit(c.unit_name('build'), tier='snapshot', stage='build', family='C', stokes=S, cadence=cad,
                              partial=True, servable=False):
                    raise NoInput('no Stokes %s %s event of %s in %s + %s' % (S, cad, c.items, c.ev_sb, c.ev_wb))
            continue
        for st in stages:
            STAGE_FN[st](c)
    return None


# ---------------------------------------------------------------------------------------------------------------------
# restricted evaluations (per hour / --cells)
# ---------------------------------------------------------------------------------------------------------------------

def _keep(c, r):
    return any(r.get('cell') == '%s_%s' % (l, d) and (b is None or r.get('band') == b) for l, d, b in c.items)


def selected_events(ev_dirs, S, cad, keep=None):
    """Rows of the two evaluations the in-place build would list for (S, cad) - snapshot_forced_panels.event_rows :51-75:
    Stokes S, screen 'clean', S/N (V: |S/N|) >= thr_survey_gauss of (sample, Stokes, tier), tier = cad. Full rows are
    counted (the in-place build refuses them, G23). Used only to decide no-input for a restricted run."""
    out = []
    for ev in ev_dirs:
        tp, rp = os.path.join(ev, 'thresholds.csv'), os.path.join(ev, 'targets_ge4.csv')
        if not (os.path.exists(tp) and os.path.exists(rp)):
            continue
        with open(tp) as f:
            th = {(r['sample'], r['stokes'], r['tier']): r for r in csv.DictReader(f)}
        with open(rp) as f:
            for r in csv.DictReader(f):
                if r.get('stokes') != S or r.get('screen') != 'clean' or r.get('tier') != cad:
                    continue
                if keep is not None and not keep(r):
                    continue
                t = th.get((r['sample'], r['stokes'], r['tier']))
                if t is None:
                    continue
                v = float(r['snr'])
                v = abs(v) if S == 'V' else v
                if v >= float(t['thr_survey_gauss']):
                    out.append(r)
    return out


def excluded_events(rows, ex=None):
    """[(cell, band, ts)] of the selected evaluation rows on an excluded (LST, band, night) - core.exclusions read now (a
    wideband is excluded with any excluded member subband, the 18 MHz carve-out kept)."""
    from exopipe.phase3.core.exclusions import Exclusions
    ex = ex or Exclusions()
    out = []
    for r in rows:
        cell = r.get('cell') or ''
        if '_' not in cell:
            continue
        lst, date = cell.split('_', 1)
        if ex.is_excluded(lst, r.get('band', ''), date[:10]):
            out.append((cell, r.get('band', ''), r.get('ts', '')))
    return out


def _has_events(c):
    return bool(selected_events((c.ev_sb, c.ev_wb), c.S, c.cad, keep=lambda r: _keep(c, r)))


def _restrict_evals(c, u):
    """Restricted copies of the two evaluations: thresholds.csv as it is, the targets_ge4.csv rows of the selected items."""
    n = 0
    for src, dst in zip((c.ev_sb, c.ev_wb), c.ev_used):
        os.makedirs(dst, exist_ok=True)
        if os.path.exists(os.path.join(src, 'thresholds.csv')):
            shutil.copyfile(os.path.join(src, 'thresholds.csv'), os.path.join(dst, 'thresholds.csv'))
        tp = os.path.join(src, 'targets_ge4.csv')
        if os.path.exists(tp):
            with open(tp, newline='') as f:
                rd = csv.DictReader(f)
                rows = [r for r in rd if _keep(c, r)]
                cols = rd.fieldnames
            K.write_rows(os.path.join(dst, 'targets_ge4.csv'), rows, cols)
            n += len(rows)
    u.meta['restricted_rows'] = n


# ---------------------------------------------------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------------------------------------------------

def stage_build(c):
    s = c.cfg.forced_snapshot
    name = c.unit_name('build')
    args = ['build', '--stokes', c.S, '--tier', c.cad, '--eval-sb', c.ev_used[0], '--eval-wb', c.ev_used[1], '--out', c.RUN]
    K.plan('[build] snapshot_forced_panels.py %s' % ' '.join(args))
    if c.dry:
        for ev in (c.ev_sb, c.ev_wb):
            K.plan('[build]   input evaluation %s: %s' % (ev, 'thresholds.csv found' if os.path.exists(
                os.path.join(ev, 'thresholds.csv')) else 'NO thresholds.csv (G22 refuses)'))
        K.plan('[build]   gate G35: work run %s %s; review copy %s%s %s' % (
            c.RUN, 'EXISTS - refused' if os.path.lexists(c.RUN) else 'new', K.remote(c.d_host), c.D,
            'EXISTS - refused' if K.path_exists_on(c.D, c.d_host) else 'new'))
        K.plan('[build]   %d Stokes %s %s events selected by the evaluations%s' % (
            len(selected_events((c.ev_sb, c.ev_wb), c.S, c.cad, keep=(lambda r: _keep(c, r)) if c.items else None)),
            c.S, c.cad, ' (restricted)' if c.items else ''))
        return
    if K.resumed(c.run, name, None, tier='snapshot', stage='build'):
        return
    with c.run.unit(name, tier='snapshot', stage='build', family='C', stokes=c.S, cadence=c.cad, run=c.RUN, review=c.D,
                    partial=bool(c.items), servable=c.servable, eval_sb=c.ev_used[0], eval_wb=c.ev_used[1]) as u:
        if os.path.lexists(c.RUN):
            u.refuse('G35 %s exists: a run name is used once (pages_run:25)' % c.RUN)
        if K.path_exists_on(c.D, c.d_host):
            u.refuse('G35 %s%s exists: a run name is used once (pages_run:26)' % (K.remote(c.d_host), c.D))
        for ev in (c.ev_sb, c.ev_wb):
            for fn in ('thresholds.csv', 'targets_ge4.csv'):
                if os.path.exists(os.path.join(ev, fn)):
                    u.add_input(os.path.join(ev, fn))
        if s.EXCLUDED_EVENTS not in ('refuse', 'build'):
            u.refuse(f'forced_snapshot.EXCLUDED_EVENTS {s.EXCLUDED_EVENTS!r}: refuse or build')
        if s.EXCLUDED_EVENTS == 'refuse':        # Gregg 2026-10-14: exclusions enforced downstream
            bad = excluded_events(selected_events((c.ev_sb, c.ev_wb), c.S, c.cad,
                                                  keep=(lambda r: _keep(c, r)) if c.items else None))
            u.meta['excluded_events'] = len(bad)
            if bad:
                u.refuse(f'{len(bad)} selected event(s) on an excluded night, e.g. {bad[:3]} - the evaluation predates the '
                         f'exclusion: redo it first (forced_snapshot.EXCLUDED_EVENTS refuse)')
        if c.items:
            _restrict_evals(c, u)
        u.meta['host'] = K.hostname()
        os.makedirs(c.RUN)
        q = os.path.join(c.RUN, 'queue.txt')
        u.plan(q, 'forced_event_build')
        res = K.call(u, K.script('snapshot'), args, c.py, c.log('build'), env=c.env())
        if not res.has(s.OK_LINES['build']):
            u.refuse('G22-G26 build: %s' % res.why())
        u.meta['build_line'] = res.line(s.OK_LINES['build'])
        u.add_output(q, 'forced_event_build', eval_sb=c.ev_used[0], eval_wb=c.ev_used[1], stokes=c.S, cadence=c.cad)


def _scratch(c, sub, production):
    """Node-local work directory: --work, else the production path for a production run, else the site scratch root of
    this host, else the work root (the production layout)."""
    if c.args.work:
        return os.path.join(c.args.work, sub)
    if not c.args.out:
        return production
    from exopipe.phase3.core import fitsio
    try:
        return os.path.join(fitsio.scratch_root(None), sub)
    except Refusal:
        return os.path.join(c.W, sub)


def stage_fit(c):
    s = c.cfg.forced_snapshot
    name = c.unit_name('fit')
    nproc = c.args.workers or s.NPROC['fit']
    wd = _scratch(c, 'work_%s' % c.name, os.path.join(c.W, 'work_%s' % c.name))     # pages_run:30 ($W/work_$NAME)
    K.plan('[fit] snapshot_forced_panels.py fit --run %s --nproc %d --workdir %s (in-place template fit: %s beam; '
           'core/fitting switch fitting.TEMPLATE_BEAM_ORIENTATION = %s)' % (c.RUN, nproc, wd,
                                                                         c.cfg.forced.INPLACE_TEMPLATE_ORIENTATION,
                                                                         c.cfg.fitting.TEMPLATE_BEAM_ORIENTATION))
    if c.dry:
        return
    if K.resumed(c.run, name, None, tier='snapshot', stage='fit'):
        return
    with c.run.unit(name, tier='snapshot', stage='fit', family='C', run=c.RUN,
                    template_orientation={'in_place': c.cfg.forced.INPLACE_TEMPLATE_ORIENTATION,
                                          'core_fitting_switch': c.cfg.fitting.TEMPLATE_BEAM_ORIENTATION}) as u:
        K.load_done(c.W, c.run.step, c.unit_name('build'), 'the build')
        c.need_work_here('fit')
        if c.cfg.forced.INPLACE_TEMPLATE_ORIENTATION != c.cfg.fitting.TEMPLATE_BEAM_ORIENTATION:
            u.note('D5: the in-place template fit uses the %s beam (clv15 44f0155) while core/fitting\'s switch says %s; '
                   'the products follow the in-place code (its K: phase3_template_fit.K_MAJOR / K_MINOR / '
                   'K_MAJOR_HIGH, re-measured un-mirrored in clv15 a715a1b)'
                   % (c.cfg.forced.INPLACE_TEMPLATE_ORIENTATION, c.cfg.fitting.TEMPLATE_BEAM_ORIENTATION))
        fp = os.path.join(c.RUN, 'forced_fits.csv')
        u.plan(fp, 'forced_fits')
        res = K.call(u, K.script('snapshot'), ['fit', '--run', c.RUN, '--nproc', nproc, '--workdir', wd], c.py,
                     c.log('fit'), env=c.env())
        if not res.has(s.OK_LINES['fit']):
            u.refuse('G27 fit: %s' % res.why())
        u.meta['fit_line'] = res.line(s.OK_LINES['fit'])
        u.add_output(fp, 'forced_fits')


def stage_tie(c):
    s = c.cfg.forced_snapshot
    name = c.unit_name('tie')
    nproc = c.args.workers or s.NPROC['tie']
    scr = _scratch(c, 'refastro_%s' % c.name, s.REFASTRO_SCRATCH_PRODUCTION)
    K.plan('[tie] snapshot_forced_panels.py tie --run %s --nproc %d (P3_SNAP_REFASTRO_SCRATCH=%s)' % (c.RUN, nproc, scr))
    if c.dry:
        return
    if K.resumed(c.run, name, None, tier='snapshot', stage='tie'):
        return
    with c.run.unit(name, tier='snapshot', stage='tie', family='C', run=c.RUN) as u:
        K.load_done(c.W, c.run.step, c.unit_name('fit'), 'the fit')
        c.need_work_here('tie')
        tp = os.path.join(c.RUN, 'forced_ties.csv')
        u.plan(tp, 'forced_ties')
        casa = c.cfg.site.PYTHON.get(c.cfg.forced.PYTHON_KEY['casa'])
        if not casa:
            u.refuse('the site file has no PYTHON[%r] (the CASA imfit worker of the tie)' % c.cfg.forced.PYTHON_KEY['casa'])
        res = K.call(u, K.script('snapshot'), ['tie', '--run', c.RUN, '--nproc', nproc], c.py, c.log('tie'),
                     env=c.env(P3_SNAP_REFASTRO_SCRATCH=scr, P3_CASA_PYTHON=casa), stage='tie')
        if not res.has(s.OK_LINES['tie']):
            u.refuse('G28 tie: %s' % res.why())
        u.meta['tie_line'] = res.line(s.OK_LINES['tie'])
        u.add_output(tp, 'forced_ties')


def stage_loc(c):
    s = c.cfg.forced_snapshot
    name = c.unit_name('loc')
    back = [x.format(S=c.S) for x in s.LOC_COPY_BACK]
    K.plan('[loc] copy %s -> %s; snapshot_forced_panels.py loc --run %s (review host %s); copy back %s'
           % (c.RUN, c.D, c.D, c.review_host or K.hostname(), ', '.join(back)))
    if c.dry:
        K.plan('[loc]   gate review host: %s, this host %s' % (c.review_host or 'shared file system', K.hostname()))
        return
    if K.resumed(c.run, name, None, tier='snapshot', stage='loc'):
        return
    with c.run.unit(name, tier='snapshot', stage='loc', family='C', run=c.RUN, review=c.D) as u:
        K.load_done(c.W, c.run.step, c.unit_name('tie'), 'the tie')
        c.need_review_host('loc')
        K.refuse_production_target(c.D, 'review copy')
        wh = c.work_host()
        log = c.log('loc')
        os.makedirs(c.D, exist_ok=True)
        rc, out = K.rsync(c.RUN.rstrip('/') + '/', c.D.rstrip('/') + '/', src_host=wh, log=log)
        if rc != 0:
            u.refuse('copy %s -> %s failed (rsync exit %d): %s' % (c.RUN, c.D, rc, out[-300:]))
        lp = os.path.join(c.D, 'source_catalogs', 'localizations.csv')
        u.plan(lp, 'forced_localization')
        res = K.call(u, K.script('snapshot'), ['loc', '--run', c.D], c.py, log, env=c.env(), cwd=K.legacy_ops(),
                     stage='loc')
        if not res.has(s.OK_LINES['loc']):
            u.refuse('G29 localization: %s' % res.why())
        for x in back:
            srcp = os.path.join(c.D, x)
            if not os.path.exists(srcp):
                u.refuse('the localization wrote no %s' % srcp)
            if os.path.isdir(srcp):
                rc, out = K.rsync(srcp.rstrip('/') + '/', os.path.join(c.RUN, x).rstrip('/') + '/', dst_host=wh, log=log)
            else:
                rc, out = K.rsync(srcp, c.RUN.rstrip('/') + '/', dst_host=wh, log=log)
            if rc != 0:
                u.refuse('copy back %s failed (rsync exit %d): %s' % (x, rc, out[-300:]))
        u.meta['loc_line'] = res.line(s.OK_LINES['loc'])
        u.add_output(lp, 'forced_localization')


def stage_assoc(c):
    s = c.cfg.forced_snapshot
    name = c.unit_name('assoc')
    if c.args.snap_no_assoc:
        K.plan('[assoc] skipped (--snap-no-assoc: the run is not servable)')
        return
    K.plan('[assoc] snapshot_forced_assoc.py run %s; verify %s (P3_RADIO_CATDIR=%s)'
           % (c.RUN, c.RUN, c.cfg.forced_assoc.RADIO_CATDIR))
    if c.dry:
        return
    if K.resumed(c.run, name, None, tier='snapshot', stage='assoc'):
        return
    with c.run.unit(name, tier='snapshot', stage='assoc', family='C', run=c.RUN) as u:
        K.load_done(c.W, c.run.step, c.unit_name('loc'), 'the localization')
        c.need_work_here('assoc')
        ip = os.path.join(c.RUN, 'forced_assoc_identity.csv')
        u.plan(ip, 'forced_assoc')
        env = c.env(P3_RADIO_CATDIR=c.cfg.forced_assoc.RADIO_CATDIR)
        res = K.call(u, K.script('assoc'), ['run', c.RUN], c.py, c.log('assoc'), env=env)
        if not res.has(s.OK_LINES['assoc']):
            u.refuse('G30 association: %s' % res.why())
        res = K.call(u, K.script('assoc'), ['verify', c.RUN], c.py, c.log('assoc'), env=env)
        if not res.has(s.OK_LINES['assoc_verify']):
            u.refuse('G31 association content test: %s' % res.why())
        u.add_output(ip, 'forced_assoc', run_dir=c.RUN)


def render_cleanup(run_dir):
    """pages_run:50-51: the served tree is rebuilt whole; the build tree candidates/<band>/ and the localization's own
    catalogue source_catalogs_<S>_loc stay."""
    for d in glob.glob(os.path.join(run_dir, 'source_catalogs_[IV]*')):
        if os.path.isdir(d) and not d.endswith('_loc'):
            shutil.rmtree(d)
    for d in (os.path.join(run_dir, 'candidates', 'V'), os.path.join(run_dir, 'candidates', 'I')):
        if os.path.isdir(d):
            shutil.rmtree(d)
    for p in glob.glob(os.path.join(run_dir, 'queue_V*.txt')) + glob.glob(os.path.join(run_dir, 'queue_I*.txt')) + \
            [os.path.join(run_dir, 'missing_panels.csv')]:
        if os.path.exists(p):
            os.remove(p)


def build_lsts(run_dir):
    """pages_run:52 (awk FNR>1 {print $1} candidates/*/*/detections.csv | sort -u): the LSTs of the build tree."""
    out = set()
    for p in glob.glob(os.path.join(run_dir, 'candidates', '*', '*', 'detections.csv')):
        with open(p) as f:
            next(f, None)
            for line in f:
                if line.strip():
                    out.add(line.split(',', 1)[0])
    return sorted(out)


def stage_render(c):
    s = c.cfg.forced_snapshot
    name = c.unit_name('render')
    nproc = c.args.workers or s.NPROC['render']
    K.plan('[render] masters %s; render --run %s --nproc %d%s; tests yellow and served (--flags %s); copy to %s '
           '(--delete --exclude yellow/)' % (c.masters, c.RUN, nproc, ' --no-assoc' if c.args.snap_no_assoc else '',
                                            c.flags, c.D))
    if c.dry:
        return
    if K.resumed(c.run, name, None, tier='snapshot', stage='render'):
        return
    with c.run.unit(name, tier='snapshot', stage='render', family='C', run=c.RUN, review=c.D) as u:
        K.load_done(c.W, c.run.step, c.unit_name('loc' if c.args.snap_no_assoc else 'assoc'), 'the association')
        c.need_work_here('render')
        K.refuse_production_target(c.D, 'review copy')
        if not (c.flags and os.path.exists(c.flags)):
            u.refuse('G36 render needs the zone recomputation (--snap-zone-flags / forced_snapshot.ZONE_FLAGS) for the '
                     'served-zones test Z2: %s' % c.flags)
        u.add_input(c.flags)
        log = c.log('render')
        render_cleanup(c.RUN)
        lsts = build_lsts(c.RUN)
        if not lsts:
            u.refuse('no LST in the build tree of %s' % c.RUN)
        mp = os.path.join(c.RUN, 'missing_panels.csv')
        u.plan(mp, 'forced_event_panels')
        res = K.call(u, K.script('snapshot'), ['masters', c.masters] + lsts, c.py, log, env=c.env())
        u.meta['masters_line'] = res.line(s.OK_LINES['masters'])
        args = ['render', '--run', c.RUN, '--masters-root', c.masters, '--nproc', nproc]
        if c.args.snap_no_assoc:
            args.append('--no-assoc')
        res = K.call(u, K.script('snapshot'), args, c.py, log, env=c.env())
        if not res.has(s.OK_LINES['render']):
            u.refuse('G32/G33 render: %s' % res.why())
        res = K.call(u, K.script('reserve_test'), ['yellow', c.RUN], c.py, log, env=c.env())
        u.check('G36_yellow', res.has(s.OK_LINES['yellow']), None, res.line(s.OK_LINES['yellow']) or res.why())
        res = K.call(u, K.script('reserve_test'), ['served', '--flags', c.flags, '--run', c.RUN], c.py, log, env=c.env())
        u.check('G36_served_zones', res.has(s.OK_LINES['served']), None, res.line(s.OK_LINES['served']) or res.why())
        bad = [t for t in u.tests if not t['passed']]
        if bad:
            u.refuse('G36 render-host tests failed: %s' % '; '.join(t['detail'][:200] for t in bad))
        rc, out = K.rsync(c.RUN.rstrip('/') + '/', c.D.rstrip('/') + '/', dst_host=c.d_host,
                          opts=('-a', '--delete', '--exclude', 'yellow/'), log=log)
        if rc != 0:
            u.refuse('copy %s -> %s failed (rsync exit %d): %s' % (c.RUN, c.D, rc, out[-300:]))
        u.add_output(mp, 'forced_event_panels', stokes=c.S)


def stage_verify(c):
    s = c.cfg.forced_snapshot
    name = c.unit_name('verify')
    args = ['verify', '--stokes', c.S, '--loc'] + ([] if c.args.snap_no_assoc else ['--assoc']) + \
        ['--eval-sb', c.ev_used[0], '--eval-wb', c.ev_used[1], '--run', c.D]
    K.plan('[verify] snapshot_forced_panels.py %s (on the review host)' % ' '.join(args))
    if c.dry:
        return
    if K.resumed(c.run, name, None, tier='snapshot', stage='verify'):
        return
    with c.run.unit(name, tier='snapshot', stage='verify', family='C', review=c.D, servable=c.servable,
                    partial=bool(c.items)) as u:
        K.load_done(c.W, c.run.step, c.unit_name('render'), 'the render')
        c.need_review_host('verify')
        for ev in c.ev_used:
            if not os.path.exists(os.path.join(ev, 'thresholds.csv')):
                u.refuse('evaluation %s not readable on %s' % (ev, K.hostname()))
        mp = os.path.join(c.D, 'missing_panels.csv')
        u.plan(mp, 'forced_event_panels')
        res = K.call(u, K.script('snapshot'), args, c.py, c.log('verify'), env=c.env(), cwd=K.legacy_ops())
        if not res.has(s.OK_LINES['verify']):
            u.refuse('content test (verify --loc%s): %s' % ('' if c.args.snap_no_assoc else ' --assoc', res.why()))
        u.meta['verify_line'] = res.line(s.OK_LINES['verify'])
        u.add_output(mp, 'forced_event_panels', stokes=c.S)


STAGE_FN = {'build': stage_build, 'fit': stage_fit, 'tie': stage_tie, 'loc': stage_loc, 'assoc': stage_assoc,
            'render': stage_render, 'verify': stage_verify}
