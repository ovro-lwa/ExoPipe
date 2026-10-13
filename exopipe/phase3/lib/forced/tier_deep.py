"""exopipe3 forced --tier deep: family B (stage catflux: forced flux at each served source's catalogue position on the run's
own search frame, p3s3_forced_phot.py in place, p3s3_pass.sh:100-101) and the family-A page stage (stage panels: review pages
of the deep HWJ / 10 pc forced measurements, deep_forced_panels.py in place on the rows of photometry's evaluation).

The deep family-A MEASUREMENT (frame list, staging, 32 shards, evaluation) is `exopipe3 photometry --tier deep` (refactor
chat decision 6, 2026-10-13); this module reads its evaluate record.

--out is the directory the stage writes into (its records go to <out>/_records/forced/):
  catflux  the deep run directory whose catalogues it completes - the catalog step's --out (default
           <paths.TRANSIENTS_ROOT>/Run_<LABEL>), so the catalog's D-G22 gate finds this record beside its own;
  panels   the isolated page run directory (default <paths.TRANSIENTS_ROOT>/<--deep-panels-run>).
"""
import glob
import os
import re

from exopipe.phase3.core.record import Refusal, NoInput
from exopipe.phase3.lib.forced import common as K


def add_arguments(p):
    g = p.add_argument_group('deep tier (catflux: forced flux at the catalogue position, family B; panels: pages of the '
                             'photometry step\'s deep HWJ / 10 pc evaluation)')
    g.add_argument('--deep-panels-run', metavar='NAME',
                   help='panels: the isolated page run directory under paths.TRANSIENTS_ROOT (run 3: Run_P3S4_forced3); '
                        'not needed with --out (= the page run directory); refused when it holds products unless --fresh')
    g.add_argument('--deep-eval-run', metavar='DIR',
                   help='panels: the photometry step\'s deep run directory (its --out); the rows are the '
                        'targets_above_threshold.csv its evaluate record lists (forced_panels.EVAL_RECORD)')
    g.add_argument('--deep-targets-csv', metavar='FILE',
                   help='panels: render these rows instead (production passes the eval file explicitly; runs 1-3 have no '
                        'photometry record)')


def default_out(args, stages):
    """The production directory of the selected deep stage (see the module docstring)."""
    c = K.cfg()
    if 'panels' in stages:
        return os.path.join(c.paths.TRANSIENTS_ROOT, args.deep_panels_run or '_forced_panels_run_not_given')
    return os.path.join(c.paths.TRANSIENTS_ROOT, 'Run_%s' % _label(args)) if args.run_label else \
        '/tmp/exopipe3_forced_no_run_label'


def _label(args):
    lab = args.run_label or ''
    return lab[4:] if lab.startswith('Run_') else lab


def run(args, run):
    stages = K.selected_stages(args, 'deep')
    if len(stages) > 1:
        raise Refusal('catflux and panels write into different directories (--out is the catalog run dir for catflux, the '
                      'page run dir for panels): run them as two invocations')
    if args.cells:
        raise Refusal('the deep forced stages take no --cells: catflux completes every catalogue of the run directory, panels '
                      'renders every row of the evaluation (the photometry step selects cells)')
    if not args.run_label:
        raise Refusal('G1 searched run required: --run LABEL (e.g. P3S5) - no default (D7, '
                      'FIX-REFACTOR-DEFECTS-D2-D6-D7-D9-D10-AND-RULINGS-D1-D5-D8-20261013)')
    src = os.path.join(K.cfg().paths.TRANSIENTS_ROOT, 'Run_%s' % _label(args))
    {'catflux': stage_catflux, 'panels': stage_panels}[stages[0]](args, run, src)
    return None


# ---------------------------------------------------------------------------------------------------------------------
# stage catflux (family B)
# ---------------------------------------------------------------------------------------------------------------------

def stage_catflux(args, run, src):
    c = K.cfg()
    cf = c.forced_catflux
    U = os.path.abspath(run.out_root)
    ep = K.epochs(args)
    files = sorted(p for d in sorted(glob.glob(os.path.join(U, cf.CATALOG_GLOB)))
                   for p in glob.glob(os.path.join(d, cf.SURVIVORS_GLOB)))
    K.plan('[catflux] p3s3_forced_phot.py %s in place: %d survivors catalogues in %s (P3S2_ROOTS_CSV=%s)%s'
           % (U, len(files), cf.CATALOG_GLOB, cf.ROOTS_CSV,
              ('; per hour %s: the catalogues as catalog --incremental left them (every row is re-measured on the frame '
               'of its best detection - a pure function of frame and position)' % ', '.join('%s_%s' % e for e in ep))
              if ep else ''))
    if run.dry_run:
        for what, p in (('run directory', U), ('its vresid/', os.path.join(U, 'vresid')),
                        ('its iresid/', os.path.join(U, 'iresid')), ('Stokes I fallback root', c.paths.COADD_SB_ROOT)):
            h = K.host_of(p)
            K.plan('[catflux]   input %s %s: %s%s' % (what, p, 'found' if os.path.exists(p) else 'MISSING',
                                                     '' if h in (None, K.hostname()) else ' (on %s: run there)' % h))
        return
    name = 'catflux'
    if K.resumed(run, name, None, tier='deep', stage='catflux'):
        return
    with run.unit(name, tier='deep', stage='catflux', family='B', run_dir=U, src_run=src,
                  epochs=['%s_%s' % e for e in ep]) as u:
        K.require_host(U, 'the run directory')
        for sub in ('vresid', 'iresid'):
            if not os.path.isdir(os.path.join(U, sub)):
                u.refuse('G21 %s/%s missing: the forced flux is measured on the run\'s OWN search frames only '
                         '(p3s3_forced_phot.py:26-31)' % (U, sub))
        K.require_host(c.paths.COADD_SB_ROOT, 'the Stokes I fallback frames (p3s2_frames.isub_path)')
        if not files:
            raise NoInput('no %s/%s in %s' % (cf.CATALOG_GLOB, cf.SURVIVORS_GLOB, U))
        for p in files:
            u.plan(p, 'catalog_forced_columns')
        res = K.call(u, K.script('catflux'), [U], c.forced.PYTHON_KEY['catflux'], K.records_log(run, name),
                     env={'P3S2_ROOTS_CSV': cf.ROOTS_CSV}, cwd=U, stage='catflux')
        m = re.search(cf.DONE_RE, res.text)
        if res.rc != 0 or not m:
            u.refuse('G21 p3s3_forced_phot failed (exit %d; the publish chain stops, D10): %s' % (res.rc, res.why()))
        u.meta['measured'], u.meta['without_frame'] = int(m.group(1)), int(m.group(2))
        # inputs = the catalogues as catflux left them (it rewrites its own inputs): --resume skips only while nobody
        # changed them since
        for p in files:
            u.add_input(p)
            u.add_output(p, 'catalog_forced_columns', run_dir=U)


# ---------------------------------------------------------------------------------------------------------------------
# stage panels (family A pages)
# ---------------------------------------------------------------------------------------------------------------------

def _targets(args, run):
    """(targets csv, where it came from)."""
    pc = K.cfg().forced_panels
    if args.deep_targets_csv:
        return os.path.abspath(args.deep_targets_csv), 'given (--deep-targets-csv)'
    if not args.deep_eval_run:
        raise Refusal('panels needs the rows: --deep-eval-run DIR (the photometry step\'s deep run, its evaluate record) or '
                      '--deep-targets-csv FILE')
    step, unit = pc.EVAL_RECORD
    if run.dry_run and not os.path.exists(os.path.join(args.deep_eval_run, '_records', step, unit + '.json')):
        return os.path.join(args.deep_eval_run, 'out', 'eval', pc.EVAL_TARGETS), 'photometry record not yet written'
    rec = K.load_done(args.deep_eval_run, step, unit, 'the photometry evaluation')
    hits = K.outputs_of(rec, basename=pc.EVAL_TARGETS)
    if len(hits) != 1:
        raise Refusal('photometry evaluate record lists %d %s' % (len(hits), pc.EVAL_TARGETS))
    return hits[0], 'photometry record %s' % os.path.join(args.deep_eval_run, '_records', step, unit + '.json')


def stage_panels(args, run, src):
    c = K.cfg()
    pc = c.forced_panels
    P = os.path.abspath(run.out_root)
    tcsv, origin = _targets(args, run)
    K.plan('[panels] deep_forced_panels.py %s %s, then --verify (DEEP_FORCED_SRC_RUN=%s; rows: %s; %s only when the '
           'content test passes)' % (tcsv, P, src, origin, pc.QUEUE))
    if run.dry_run:
        K.plan('[panels]   gate page run directory: %s' % ('HOLDS PRODUCTS (refused without --fresh)'
                                                           if K.nonrecord_entries(P) else 'new'))
        h = K.host_of(src)
        K.plan('[panels]   gate searched run %s: %s%s' % (src, 'found' if os.path.exists(src) else 'MISSING (G1)',
                                                         '' if h in (None, K.hostname()) else ' (on %s: run there)' % h))
        return
    name = 'panels'
    if K.resumed(run, name, None, tier='deep', stage='panels'):
        return
    with run.unit(name, tier='deep', stage='panels', family='A', panels_run=P, targets_csv=tcsv, src_run=src,
                  rows_from=origin) as u:
        K.require_host(src, 'G1 the searched run (the page run links its frames)')
        if K.nonrecord_entries(P) and not run.fresh:
            u.refuse('%s holds products: a page run name is used once (a new --deep-panels-run, or --fresh)' % P)
        if not os.path.exists(tcsv):
            u.refuse('rows %s not found' % tcsv)
        u.add_input(tcsv)
        for fn in c.forced.SAMPLE_FILES.values():
            u.add_input(os.path.join(c.paths.CLV15_ROOT, fn))
        q = os.path.join(P, pc.QUEUE)
        u.plan(q, 'forced_pages')
        os.makedirs(P, exist_ok=True)
        tess = c.site.PYTHON.get(c.forced.PYTHON_KEY['tess'])
        if not tess:
            u.refuse('the site file has no PYTHON[%r] (deep_forced_panels.py runs the TESS coverage with it)'
                     % c.forced.PYTHON_KEY['tess'])
        res = K.call(u, K.script('panels'), [tcsv, P], c.forced.PYTHON_KEY['panels'], K.records_log(run, name),
                     env={'DEEP_FORCED_SRC_RUN': src, 'TESS_PY': tess}, cwd=P, stage='panels')
        if res.rc != 0:
            if res.refused:
                u.refuse('G17-G20 deep_forced_panels: %s' % res.why())
            raise RuntimeError('deep_forced_panels.py exit %d: %s' % (res.rc, res.why()))
        u.meta['tess'] = 'written' if pc.TESS_OK in res.text else 'FAILED (annotation, never a refusal; ledger 1756)'
        u.add_output(q, 'forced_pages', targets_csv=tcsv, src_run=src)
