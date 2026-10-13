"""exopipe3 followup - follow-up of high-interest candidates and the external-coverage records the review needs.

    exopipe3 followup ep-plan --action fetch                     # one Einstein Probe WXT plan download (cron, every 3 h)
    exopipe3 followup ep-plan --action audit                     # coverage gaps of the stored plans (cron, daily)
    exopipe3 followup ep-plan --action replay --raw @raws.txt --out DIR     # stored downloads through today's content test
    exopipe3 followup tess --name "HD 154088" --radio-start 2025-04-12T11:29:23.8 --radio-end 2025-04-12T12:27:34.5 \\
        --radio-label "J1704-2833 deep Stokes V, 17h 2025-04-12, 78 MHz" --out DIR
    exopipe3 followup investigate --cand 2,5 --out DIR          # candidate dossiers from the top-50 list (--cand all)
    exopipe3 followup investigate --ra 94.895 --dec 41.092 --lst 07h --date 2024-12-31 --band 23MHz --out DIR
    exopipe3 followup summary --out DIR                         # ranked summary of the dossiers (re-scored)
    exopipe3 followup hone --cand-csv cands.csv --out DIR        # honing in the 10-minute x subband plane
    exopipe3 followup periodicity --stages lightcurves --cells '*_Red' --stokes I --out DIR
    exopipe3 followup periodicity --stages search,page --lc-root /lustre/gh/main/phase3/periodicity5 --out DIR
    exopipe3 followup fold --lightcurve lc.csv --prefix J1810 --ra 272.6 --dec 17.7 --period-d 0.14817 \\
        --fold-bands Red,Green --out DIR
    exopipe3 followup source-page --source-id S0064780 --ra 279.6 --dec -6.9 --out DIR
    exopipe3 followup dynspec | lpt-period | dedisperse | reprocess        # design pending: refused (exit 2)

Sub-operations (positional OP; each independent; one run record per unit, <out>/_records/followup/<unit>.json):
  ep-plan      Einstein Probe WXT 24-hour PLANNED footprint archive (ledger IMPL-EP-WXT-PLAN-ARCHIVE). Units
               fetch_<UTC>, audit_<UTC date>, replay_<raw|refused>_<tag>.
  tess         TESS light curve of a star around a radio epoch (the tess-light-curve skill). Unit <key>.
  investigate  candidate dossier (exopipe/phase3/followup/, the candinv harness): morphology, astrometry, spectrum, I/V
               coincidence, 10-minute curve, context, 10-s snapshot escalation, P(astrophysical). Unit = candidate id
               rank<NN>_<Jname>_LST<lst>; one candidate's failure never stops the others.
  summary      ranked summary CSV of existing dossiers; re-scores each evidence file and writes the score back (today's
               attic/run_all_top50.py --assemble). Unit summary_<name>.
  hone         architecture 9.2 honing (phase3_honing.py): reproduction gate on S/N, enumerated box bank, control null.
               Unit = candidate id of the CSV.
  periodicity  the 10-minute wideband periodicity search on catalogued positions, pass 5 (the scope ruled 'for now',
               R-PERIODICITY-SCOPE-WIDEBAND-10MIN-CATALOGUED-POSITIONS-FOR-NOW). Stages lightcurves (unit
               lc3_<LST>_<band>_<S>; item = cell <LST>_<band>, band Red/Green/Blue) -> search (unit <band>_<S>) -> page
               (unit page_<S>); each stage reads only files a done unit of the previous stage recorded.
               NOT the exoplanet-class periodicity of architecture 9.4 (known period first, every tier): design pending.
  fold         barycentre + PDM + MAD-cleaned fold of one 10-minute light curve (steep_fold.py). Unit <prefix>.
               NOT the folding of architecture 9.5 (association evidence, window function): design pending.
  source-page  one periodicity diagnostic page for one source (source_diag_page.py). Unit <source_id>.
  dynspec (9.3), lpt-period (9.6), dedisperse (9.7), reprocess (9.1): DESIGN PENDING - refused before any read (exit 2).

Inputs (every path is a config key: exopipe3 config --step followup):
  ep-plan      HTTPS GET of followup_ep_plan.URL; replay: stored raw_<tag>.json.gz / refused_<tag>.json.gz (--raw)
  tess         SIMBAD, MAST (TIC, light-curve products, TESScut), NASA Exoplanet Archive; followup_tess.ORBIT_TABLE
  investigate  followup_investigate.TOP50_CSV, BUNDLE_DIR (June bundle), PROD_ROOT trees, PHASE2_ARCHIVE / PHASE2_FAST
               images and warp screens, Phase-1 10-s snapshots, VLSS_UNRESOLVED
  summary      <out>/transients/Run_<OUT_LABEL>/<OUT_SUBDIR>/<cand_id>/<cand_id>_evidence.json
  hone         --cand-csv (cand_id,lst,date,ra,dec,snr[,band][,disc_sb]); 10-minute V msub frames under
               followup_hone.INPUT_ROOT (deleted 2026-09-01: every candidate ends no-products today)
  periodicity  LC_POSITIONS; archive frames <LC_ARCHIVE_ROOT>/<LST>/*/<LC_GENERATION>/<band>/<S>/10min/...pbcorr...msub...;
               search: the 24 lc3_<LST>_<band>_<S>.npz (this run's lightcurves or --lc-root) and SOURCES_CSV
  fold         --lightcurve CSV (band, mjd_mid, flux_jy, err_jy, night[, lst, cadence])
  source-page  archive frames and pass-4 tables (literals of source_diag_page.py)
  exclusions   paths.DATA_DECISIONS, report only (followup.EXCLUSION_MODE 'report' = today's behaviour): every unit with
               nights records its admitted night set and notes the excluded nights it used
Outputs (production layout, no renames; with --out DIR the same relative layout under DIR):
  ep-plan      <root>/YYYY-MM/wxtplan_<tag>.npz [ep_wxt_plan], raw_<tag>.json.gz [ep_wxt_raw], refused_<tag>.json.gz
               [ep_wxt_refused]; <root>/fetch_log.csv [ep_wxt_fetch_log]; <root>/coverage_gaps.csv [ep_wxt_coverage_gaps];
               replay: <out>/replay/wxtplan_<tag>.npz [ep_wxt_plan] + replay_log.csv [csv_table] (never the archive root)
  tess         <out>/<key>/<key>_summary.txt [tess_summary], <key>_s<NN>_<SPOC120|SPOC20|QLP200|own>.csv [tess_lightcurve],
               <key>_s<NN>.png [followup_figure]
  investigate  <out>/transients/Run_PROD/candinv/<cand_id>/: _evidence.json [followup_dossier]; _spectrum.csv,
               _snapshot_lc.csv, _snapshot_spectrum.csv [followup_table]; .reg [followup_region]; conformant FITS
               (all-sky, cutouts, 10-s burst frames) [followup_fits]; _snapshot_lc.png, _index.png [followup_figure]
  summary      <out>/transients/Run_PROD/candinv/<summary-name> [followup_summary]; re-scored evidence [followup_dossier]
  hone         <out>/honing/honed.csv [honing_table]; <out>/honing/stamps/<cand_id>_honedbox.fits [honing_stamp]
  periodicity  lightcurves <LC_ROOT | out/lightcurves>/lc3_<LST>_<band>_<S>.npz [periodicity_lightcurves];
               search <SEARCH_OUT | out/search>/periodicity_<band>_<S>.csv [periodicity_table];
               page <MERGED_DIR | out/page>/allsky_periodicity_<S>.csv [periodicity_merged] and <page dir>/index.html
               [followup_page] (--page-dir, default <merged dir>/periodicity_page_<S>)
  fold         <out>/<prefix>_folded.csv [fold_table]; _folded.png, _smoothed.png, _period_search.png [followup_figure]
  source-page  <out>/index.html [followup_page] + <out>/img/
  records      <out>/_records/followup/<unit>.json; child logs <out>/_records/followup/logs/<unit>.log
Failure modes (exit 2 = refused, the gate named; 3 = nothing to do; 1 = error):
  before any unit: G5 design-pending op; a flag the op does not use (exit 1); G6 override of a PROTECTED key (an in-place
  literal or a production root: config refuses it) or of a pinned input (TOP50_CSV, BUNDLE_DIR, LC_POSITIONS) while --out
  is a production root; G6b an in-place source is missing or no longer carries a literal config mirrors
  (followup.MIRRORS); G7 --out (or --page-dir) under a production root (paths.PRODUCTION_ROOTS or
  followup.PRODUCTION_ROOTS) while paths.PRODUCTION_WRITES is False; G8 the site interpreter missing or not executable on
  this host (tess: lwacalim07 only).
  ep-plan: G1 plan content test / G2 lossless round trip failed (raw kept as refused_<tag>.json.gz; unit refused); G3
  download or parse failure (error); G4 fetch_log.csv did not grow by exactly one row (error); audit with no stored plans
  (exit 3); replay into the archive root (refused).
  tess: G9 star not resolved by SIMBAD (refused); G10 no TESS sector observed the star in the epoch (exit 3, summary kept).
  investigate: G11 discovery V residual not in the bundle, G12 discovery fit failed (refused); G13 the package config was
  imported before its paths were set (refused); no node-local scratch (pass --work); --no-write ends no-input (exit 3).
  hone: G16 reproduction gate failed (refused, per candidate); G17 no frames / no box (exit 3 per candidate).
  periodicity: G26 no frames for (LST, band, S), G27 no light curves for (band, S) (exit 3); G28 a light curve or table not
  recorded by a done unit of the previous stage (nor in the pinned pass-5 roots), fewer than 24 cells' light curves, or
  fewer than the three band tables (refused).
  hone, fold, source-page: no production root - --out is required.
Fallbacks kept as today (batch rows B-30, B-41, C-33, C-37 lie with Gregg; recorded as unit notes where visible): dossier
FITS stamped unvalidated on header failure; no Phase-2 epoch time -> no conformant FITS; any-variant gap-fill; V spectrum
from the non-subtracted image; undewarped snapshot frames when the screen grid differs; CASA import failure -> matched
filter; empty cross-match (catalogue directory: a note on every dossier unit); TESS service failures printed and continued.

Modes (DESIGN 12): intermittent (survey) only - nothing here runs in the per-hour chain. ep-plan fetch/audit run from the
lwacalim07 crontab (during the window the crontab still calls ep_wxt_plan_fetch.py itself).
Refactor window (DESIGN 11.1): every sub-operation calls today's code IN PLACE - the exopipe_ops scripts through
core.legacy.run with production's argv and environment (values from config), the follow-up package in-process with its
CANDINV_* paths set from config (followup/_cfg_adapter.py). The content tests of the 20 follow-up classes live here.
"""
import csv
import gzip
import json
import os
import re
import sys
import time

from exopipe.phase3.core.validate import content_test

SUMMARY = ('follow-up: EP WXT plan archive, TESS curve, candidate dossiers + summary, honing, 10-min periodicity, fold, '
           'source page; design-pending sub-operations refuse')
TIERS = None
SELECTORS = ('cells', 'stokes')          # periodicity only; every other op refuses them
CONFIG = ('followup', 'followup_ep_plan', 'followup_tess', 'followup_investigate', 'followup_hone',
          'followup_periodicity', 'followup_fold', 'fitting', 'paths', 'exclusions', 'runner', 'site')
WRITES = True

# design-pending sub-operations: (architecture section, what, why) - gate G5
PENDING = {
    'dynspec': ('9.3', 'dynamic spectra (contract 6.3)', 'no code exists in any repository'),
    'lpt-period': ('9.6', 'LPT periodicity (contract 6.6)',
                   'no code; the blind test showed the periodogram unsuited to the LPT (ledger 1266-1269)'),
    'dedisperse': ('9.7', 'dedispersion (contract 6.7)', 'no code exists in any repository'),
    'reprocess': ('9.1', 'reprocessing (contract 6.1)',
                  'no Phase 3 code - reprocess with the Phase 1/2 controllers'),
}
OPS = ('ep-plan', 'tess', 'investigate', 'summary', 'hone', 'periodicity', 'fold', 'source-page') + tuple(PENDING)
STAGES = ('lightcurves', 'search', 'page')
# step flags (argparse dest) each sub-operation uses; any other one given is an error (exit 1)
OP_FLAGS = {
    'ep-plan': {'action', 'raw'},
    'tess': {'name', 'radio_start', 'radio_end', 'radio_label', 'sector', 'key'},
    'investigate': {'cand', 'ra', 'dec', 'lst', 'date', 'band', 'cadence', 'iv', 'no_escalate', 'no_snapshot', 'no_write'},
    'summary': {'cand', 'summary_name'},
    'hone': {'cand_csv', 'band', 'controls', 'stamps', 'snapshots'},
    'periodicity': {'stages', 'bands', 'lc_root', 'ncpu', 'page_dir', 'cells', 'stokes'},
    'fold': {'lightcurve', 'prefix', 'period_d', 'fold_bands', 'ylim', 'ra', 'dec'},
    'source-page': {'source_id', 'ra', 'dec', 'page_lsts'},
}
_ALL_FLAGS = sorted(set().union(*OP_FLAGS.values()))
_TAG_RE = re.compile(r'^(raw|refused)_(\d{8}T\d{6}Z)\.json\.gz$')


# =====================================================================================================================
# command line
# =====================================================================================================================

def add_arguments(p):
    p.add_argument('op', choices=OPS, metavar='OP', help='sub-operation: ' + ', '.join(OPS) + ' (the last four refuse: '
                   'design pending)')
    g = p.add_argument_group('ep-plan')
    g.add_argument('--action', choices=('fetch', 'audit', 'replay'),
                   help='fetch (default): one download, content test before any write; audit: coverage gaps of the '
                        'stored plans; replay: stored downloads through today\'s content test into --out/replay/')
    g.add_argument('--raw', action='append', metavar='PATH|@FILE', help='replay: raw_<tag>.json.gz / refused_<tag>.json.gz')
    g = p.add_argument_group('tess')
    g.add_argument('--name', help='star name SIMBAD resolves (or a Gaia DR3 id)')
    g.add_argument('--radio-start', help='radio epoch start, ISO UTC')
    g.add_argument('--radio-end', help='radio epoch end, ISO UTC')
    g.add_argument('--radio-label', help='label printed and plotted')
    g.add_argument('--sector', type=int, help='one TESS sector (default: every sector overlapping the epoch that covers the star)')
    g.add_argument('--key', help='file prefix and product directory (default tess_ + the alphanumerics of --name)')
    g = p.add_argument_group('investigate / summary / hone')
    g.add_argument('--cand', help='top-50 ranks: 2,5 | @FILE | all (summary default: all)')
    g.add_argument('--ra', type=float, help='deg (investigate rank-0 candidate; fold; source-page)')
    g.add_argument('--dec', type=float, help='deg')
    g.add_argument('--lst', help='investigate: discovery LST, e.g. 07h')
    g.add_argument('--date', help='investigate: discovery night YYYY-MM-DD')
    g.add_argument('--band', help='investigate: discovery scope (subband or band); hone: Red/Green/Blue frequency axis '
                                  '(default followup_hone.DEFAULT_BAND)')
    g.add_argument('--cadence', choices=('deep', '10min'), help='investigate: discovery cadence (default deep)')
    g.add_argument('--iv', type=int, help='investigate: 10-minute bin index of a 10-minute discovery')
    g.add_argument('--no-escalate', action='store_true', help='investigate: skip T4 multi-epoch and T5 snapshot')
    g.add_argument('--no-snapshot', action='store_true', help='investigate: skip T5 only')
    g.add_argument('--no-write', action='store_true', help='investigate: evidence in the record only (unit ends no-input)')
    g.add_argument('--summary-name', help='summary: file name (default investigation_summary.csv)')
    g.add_argument('--cand-csv', help='hone: candidates cand_id,lst,date,ra,dec,snr[,band][,disc_sb]')
    g.add_argument('--controls', type=int, help='hone: control positions (default followup_hone.N_CONTROLS)')
    g.add_argument('--stamps', action='store_true', help='hone: write each maximizing box stack (161 x 161 px)')
    g.add_argument('--snapshots', action='store_true', help='hone: flag best boxes for snapshot refinement (note only)')
    g = p.add_argument_group('periodicity (uses the common --cells <LST>_<band> and --stokes)')
    g.add_argument('--stages', help='stages in order: lightcurves,search,page (default all three; declared here until '
                                    'cli.py adds --stages)')
    g.add_argument('--bands', help='band axis of the search jobs (default followup_periodicity.BANDS)')
    g.add_argument('--lc-root', help='light curves for search/page when lightcurves does not run in this invocation '
                                     '(default followup_periodicity.LC_ROOT, the pinned pass-5 root)')
    g.add_argument('--ncpu', type=int, help='search processes (default min(followup_periodicity.NCPU, free cores))')
    g.add_argument('--page-dir', help='page output directory (default <merged dir>/periodicity_page_<S>)')
    g = p.add_argument_group('fold / source-page')
    g.add_argument('--lightcurve', help='fold: light-curve CSV')
    g.add_argument('--prefix', help='fold: output prefix')
    g.add_argument('--period-d', type=float, help='fold: catalogue period, days')
    g.add_argument('--fold-bands', help='fold: bands to fold, comma-separated')
    g.add_argument('--ylim', help='fold: plot limits lo:hi per band, comma-separated')
    g.add_argument('--source-id', help='source-page: source id of the all-sky catalogue')
    g.add_argument('--page-lsts', help='source-page: cells, comma-separated (default the transit hour and its neighbours)')


def default_out(args):
    from exopipe.phase3.config import CFG
    op = args.op
    if op in PENDING:
        return '/tmp'                          # nothing is written (run.no_records): the refusal comes first
    if op == 'ep-plan':
        return CFG.followup_ep_plan.ARCHIVE_ROOT
    if op == 'tess':
        return CFG.followup_tess.OUT_PARENT
    if op in ('investigate', 'summary'):
        return CFG.followup_investigate.OUT_ROOT
    if op == 'periodicity':
        P = CFG.followup_periodicity
        return {'lightcurves': P.LC_ROOT, 'search': P.SEARCH_OUT, 'page': P.MERGED_DIR}[_stages(args)[0]]
    raise SystemExit(f'followup {op}: no production root - give --out DIR')


def _stages(args):
    st = tuple(s.strip() for s in (args.stages or ','.join(STAGES)).split(',') if s.strip())
    bad = [s for s in st if s not in STAGES]
    if bad or not st:
        raise ValueError(f'--stages: unknown {bad}; stages are {",".join(STAGES)}')
    if list(st) != sorted(st, key=STAGES.index) or len(set(st)) != len(st):
        raise ValueError(f'--stages {",".join(st)}: give each stage once, in pipeline order ({",".join(STAGES)})')
    return st


def _given(args, dest):
    v = getattr(args, dest, None)
    return v not in (None, False, [], '')


def _check_flags(args):
    """A step flag the chosen op does not use is an error (spec section 2)."""
    allowed = OP_FLAGS.get(args.op, set())
    extra = [d for d in _ALL_FLAGS if d not in allowed and _given(args, d)]
    if extra:
        raise ValueError(f'followup {args.op} does not use ' + ', '.join('--' + d.replace('_', '-') for d in extra)
                         + f'; its flags: ' + (', '.join('--' + d.replace('_', '-') for d in sorted(allowed)) or 'none'))


# =====================================================================================================================
# the step
# =====================================================================================================================

def run(args, run):
    from exopipe.phase3.core import record
    op = args.op
    if op in PENDING:                                                    # G5: refuse before any read
        run.no_records = True
        sec, what, why = PENDING[op]
        raise record.Refusal(f"G5: followup {op} - {what} - is DESIGN PENDING (architecture {sec}): {why}. Architecture "
                             f"9.0.2: no novel statistical test enters an evidence file without Gregg's prior sign-off; "
                             f"its design card goes to Gregg first. Nothing was read or written.")
    try:
        _check_flags(args)
        if op == 'periodicity':
            _stages(args)
    except ValueError as e:
        run.no_records = True
        print(f'error: {e}', file=sys.stderr)
        return record.EXIT_ERROR
    problems = _gates(args, run)
    if problems and not run.dry_run:
        run.no_records = True                                            # nothing may land under a production root
        raise record.Refusal(' | '.join(problems))
    for p in problems:
        print(f'WOULD REFUSE  {p}')
    fn = {'ep-plan': _op_ep_plan, 'tess': _op_tess, 'investigate': _op_investigate, 'summary': _op_summary,
          'hone': _op_hone, 'periodicity': _op_periodicity, 'fold': _op_fold, 'source-page': _op_source_page}[op]
    try:
        rc = fn(args, run)
    except (ValueError, OSError) as e:                                   # argument / input-list errors found in an op
        if not run.units:
            run.no_records = True
        print(f'error: {e}', file=sys.stderr)
        return record.EXIT_ERROR
    if run.dry_run:
        return record.EXIT_REFUSED if problems else (record.EXIT_DONE if rc is None else rc)
    return rc


# ---------------------------------------------------------------------------------------------------------------------
# gates before any unit: G6 (pinned inputs), G6b (mirrored literals), G7 (production roots), G8 (interpreter)
# ---------------------------------------------------------------------------------------------------------------------

def _under(path, roots):
    real = os.path.realpath(path)
    for r in roots:
        rr = os.path.realpath(r)
        if real == rr or real.startswith(rr.rstrip('/') + '/'):
            return r
    return None


def _production_root(CFG, path):
    """The production root holding ``path``: paths.PRODUCTION_ROOTS or followup.PRODUCTION_ROOTS; None otherwise."""
    return CFG.production_root_of(path) or _under(path, CFG.followup.PRODUCTION_ROOTS)


def _script(key):
    """Absolute path of an in-place script (followup.SCRIPTS, relative to the site's legacy root 'exopipe_ops')."""
    from exopipe.phase3.config import CFG
    from exopipe.phase3.core import env
    rel = CFG.followup.SCRIPTS[key]
    return rel if os.path.isabs(rel) else os.path.join(env.legacy_root('exopipe_ops'), rel)


def _scripts_for(args):
    op = args.op
    if op == 'periodicity':
        return list(_stages(args))
    return {'ep-plan': ['ep_plan'], 'tess': ['tess'], 'hone': ['hone'], 'fold': ['fold'],
            'source-page': ['source_page']}.get(op, [])


def _mirror_problems(CFG, key):
    try:
        path = _script(key)
    except KeyError as e:
        return [f'G6b: {e}']
    if not os.path.isfile(path):
        return [f'G6b: in-place script {path} (followup.SCRIPTS[{key!r}]) not found']
    text = open(path, encoding='utf-8', errors='replace').read()
    return [f'G6b: {ck} mirrors the literal "{snip}" of {path}, which the source no longer contains - the in-place code '
            f'changed: update config (and its content tests) with it before running'
            for ck, snip in CFG.followup.MIRRORS.get(key, ()) if snip not in text]


def _write_roots(args, run):
    CFG = run.cfg
    roots = [run.out_root]
    if args.op == 'periodicity' and not args.out:
        P = CFG.followup_periodicity
        roots += [{'lightcurves': P.LC_ROOT, 'search': P.SEARCH_OUT, 'page': P.MERGED_DIR}[s] for s in _stages(args)]
    if args.op == 'periodicity' and args.page_dir:
        roots.append(args.page_dir)
    return list(dict.fromkeys(roots))


def _gates(args, run):
    CFG = run.cfg
    problems = []
    prod = _production_root(CFG, run.out_root)
    for o in CFG.overrides:                                              # G6: pinned inputs only into a scratch --out
        base = o[0].split('[')[0]
        sec, _, key = base.partition('.')
        if key in getattr(CFG.sections().get(sec), 'pinned_inputs', ()) and prod:
            problems.append(f'G6: {base} is a pinned input; override it only into a --out outside every production root '
                            f'(this run writes {run.out_root}, under {prod})')
    for key in _scripts_for(args):                                       # G6b
        problems += _mirror_problems(CFG, key)
    if not CFG.paths.PRODUCTION_WRITES:                                  # G7
        for d in _write_roots(args, run):
            r = _production_root(CFG, d)
            if r:
                problems.append(f'G7: {d} is under the production root {r} and paths.PRODUCTION_WRITES is False '
                                f'(refactor window): write to your own root with --out')
    if args.op == 'tess':                                                # G8
        py = CFG.site.PYTHON.get(CFG.followup.TESS_PYTHON_KEY)
        if not py or not os.access(py, os.X_OK):
            problems.append(f'G8: the TESS interpreter site.PYTHON[{CFG.followup.TESS_PYTHON_KEY!r}] = {py!r} is not '
                            f'executable on {run.host} (the TESS environment exists on lwacalim07 only)')
    return problems


# ---------------------------------------------------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------------------------------------------------

def _log_path(run, unit):
    d = os.path.join(run.out_root, '_records', 'followup', 'logs')
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f'{unit}.log')


def _call(u, run, key, argv, python=None, env=None, cwd=None, log_name=None):
    """Run the in-place script ``key`` as production runs it (core.legacy.run): the script and every legacy module its
    child imported become inputs of the unit (code identity for --resume)."""
    from exopipe.phase3.core import legacy
    script = _script(key)
    log = _log_path(run, log_name or (u.name if u is not None else key))
    if u is not None:
        u.add_input(script)
    res = legacy.run(script, argv, python=python or run.cfg.followup.PYTHON_KEY, env=env,
                     cwd=cwd or os.path.dirname(log), log=log, unit=u, refusal_codes=(), refusal_text=('REFUSED:',))
    if u is not None:
        for f in sorted(res.fingerprint or {}):
            if os.path.abspath(f) not in u.inputs and os.path.isfile(f):
                u.add_input(f)
    return res


def _guarded(run, name, fn, **meta):
    """One unit; an error is recorded in its record and the next unit still runs."""
    try:
        with run.unit(name, **meta) as u:
            fn(u)
    except Exception as e:
        print(f'followup {name}: ERROR {type(e).__name__}: {e}', file=sys.stderr)
        return None
    print(f'followup {name}: {u.status}' + (f' - {u.reason}' if u.reason else ''), file=sys.stderr)
    return u


def _resumed(run, name, admitted=None, **meta):
    """--resume: True (and a skipped-valid unit) when the unit's record is valid now (core.record.resume_check)."""
    if not run.resume or run.fresh:
        return False
    ok, why = run.resume_check(name, inputs=None, admitted=admitted)
    if ok:
        u = run.unit(name, **meta)
        u.status, u.reason = 'skipped-valid', why
        print(f'followup {name}: skipped ({why})', file=sys.stderr)
        return True
    print(f'followup {name}: not resumed ({why})', file=sys.stderr)
    return False


def _atomic_text(path, text):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, f'.{os.path.basename(path)}.tmp.{os.getpid()}')
    with open(tmp, 'w', newline='') as f:
        f.write(text)
    os.replace(tmp, path)


def _read_csv(path):
    with open(path, newline='') as f:
        rd = csv.DictReader(f)
        return list(rd.fieldnames or []), list(rd)


def _new_files(paths, t0):
    return [p for p in paths if os.path.isfile(p) and os.path.getmtime(p) >= t0 - 1.0]


def _exclusion_report(u, lst, scope, dates):
    """EXCLUSION_MODE 'report' (today's behaviour): record the admitted nights; note excluded nights used."""
    from exopipe.phase3.core import exclusions
    ex = exclusions.load()
    dates = sorted(set(dates))
    bad = ex.excluded_constituents(lst, scope, dates)
    u.set_admitted([d for d in dates if d not in bad])
    u.meta['exclusions'] = dict(ex.fingerprint(), mode='report', excluded_used=bad)
    if bad:
        u.note(f'EXCLUSION_MODE report: {len(bad)} excluded night(s) of ({lst}, {scope}) used as today: {", ".join(bad)}')
    return bad


def _plan(lines):
    for line in lines:
        print(line)


# =====================================================================================================================
# ep-plan
# =====================================================================================================================

_EP_MOD = {}


def _ep_module():
    """The in-place EP module (content_test, to_compact, read_compact, roundtrip_ok, LOG_COLS), imported once by path."""
    if 'm' not in _EP_MOD:
        import importlib.util
        spec = importlib.util.spec_from_file_location('exopipe3_legacy_ep_wxt_plan_fetch', _script('ep_plan'))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        _EP_MOD['m'] = m
    return _EP_MOD['m']


def _ep_tag(fetch_utc):
    return fetch_utc.replace('-', '').replace(':', '')[:15] + 'Z'


def _utc_tag():
    return time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())


def _op_ep_plan(args, run):
    action = args.action or 'fetch'
    if action == 'replay':
        return _ep_replay(args, run)
    root = run.out_root
    log = os.path.join(root, 'fetch_log.csv')
    if run.dry_run:
        n = len(_read_csv(log)[1]) if os.path.exists(log) else 0
        npz = sum(1 for dp, _, fs in os.walk(root) for f in fs if f.startswith('wxtplan_') and f.endswith('.npz')
                  and '.part' not in f) if os.path.isdir(root) else 0
        _plan([f'ep-plan {action}: {_script("ep_plan")} {root}' + (' --audit' if action == 'audit' else ''),
               f'  archive root {root}: {npz} stored plans, fetch_log.csv {n} rows',
               f'  outputs: ' + ('<root>/YYYY-MM/wxtplan_<tag>.npz [+ raw_/refused_<tag>.json.gz], fetch_log.csv (+1 row)'
                                 if action == 'fetch' else '<root>/coverage_gaps.csv')])
        return 0 if (action == 'fetch' or npz) else 3

    if action == 'fetch':
        def body(u):
            os.makedirs(root, exist_ok=True)
            before = open(log, 'rb').read() if os.path.exists(log) else b''
            n0 = len(_read_csv(log)[1]) if before else 0
            res = _call(u, run, 'ep_plan', [root])
            after = open(log, 'rb').read() if os.path.exists(log) else b''
            rows = _read_csv(log)[1] if after else []
            grew = after.startswith(before) and len(rows) == n0 + 1
            u.check('ep_plan.log_appended_one_row', grew, value=f'{n0} -> {len(rows)} rows',
                    detail='fetch_log.csv grew by exactly one row; earlier rows byte-identical' if grew else
                    'G4: fetch_log.csv did not grow by exactly one row with the earlier rows unchanged')
            if not grew:
                raise RuntimeError(f'G4: fetch_log.csv {n0} -> {len(rows)} rows (rc {res.rc}); see {res.log}')
            row = rows[-1]
            tag = _ep_tag(row['fetch_utc'])
            mdir = os.path.join(root, f'{tag[:4]}-{tag[4:6]}')
            u.meta['fetch'] = dict(row, tag=tag, rc=res.rc)
            if row['status'] == 'pass':
                if res.rc != 0:
                    raise RuntimeError(f'pass row logged but the script exited {res.rc}; see {res.log}')
                u.add_output(os.path.join(root, row['file']), 'ep_wxt_plan')
                raw = os.path.join(mdir, f'raw_{tag}.json.gz')
                if os.path.exists(raw):
                    u.add_output(raw, 'ep_wxt_raw')
                u.add_output(log, 'ep_wxt_fetch_log')
                return
            u.add_output(log, 'ep_wxt_fetch_log')
            if row['detail'].startswith('download/parse'):
                raise RuntimeError(f'G3: download failed: {row["detail"]}')
            refused = os.path.join(mdir, f'refused_{tag}.json.gz')
            if os.path.exists(refused):
                u.add_output(refused, 'ep_wxt_refused')
            u.refuse(f'G1/G2: plan refused, nothing archived: {row["detail"]}')
        _guarded(run, f'fetch_{_utc_tag()}', body, op='ep-plan', action='fetch')
        return None

    def audit(u):
        from exopipe.phase3.core.record import NoInput
        res = _call(u, run, 'ep_plan', [root, '--audit'])
        if res.rc == 1 and 'no stored plans' in res.tail:
            raise NoInput(f'no stored plans under {root}')
        if res.rc != 0:
            raise RuntimeError(f'ep_wxt_plan_fetch.py --audit exit {res.rc}; see {res.log}')
        for dp, _, fs in sorted(os.walk(root)):
            for f in sorted(fs):
                if f.startswith('wxtplan_') and f.endswith('.npz') and '.part' not in f:
                    u.add_input(os.path.join(dp, f))
        u.add_output(os.path.join(root, 'coverage_gaps.csv'), 'ep_wxt_coverage_gaps')
    _guarded(run, f'audit_{time.strftime("%Y%m%d", time.gmtime())}', audit, op='ep-plan', action='audit')
    return None


def _expand_paths(items):
    out = []
    for it in items or []:
        for tok in [t.strip() for t in it.split(',') if t.strip()]:
            if tok.startswith('@'):
                with open(tok[1:]) as f:
                    out += [l.split('#', 1)[0].strip() for l in f if l.split('#', 1)[0].strip()]
            else:
                out.append(tok)
    return list(dict.fromkeys(os.path.abspath(p) for p in out))


def _ep_replay(args, run):
    from exopipe.phase3.core import record
    CFG = run.cfg
    raws = _expand_paths(args.raw)
    if not raws:
        raise ValueError('ep-plan --action replay needs --raw PATH|@FILE')
    bad = [r for r in raws if not _TAG_RE.match(os.path.basename(r))]
    if bad:
        raise ValueError(f'--raw: not raw_<tag>.json.gz / refused_<tag>.json.gz: {bad[:3]}')
    arch = CFG.followup_ep_plan.ARCHIVE_ROOT
    if _under(run.out_root, [arch]):
        run.no_records = True
        raise record.Refusal(f'replay writes {run.out_root}, inside the archive root {arch}: replay never writes the '
                             f'archive - give --out DIR elsewhere')
    rdir = os.path.join(run.out_root, 'replay')
    if run.dry_run:
        _plan([f'ep-plan replay: {len(raws)} stored download(s) through today\'s content test -> {rdir}/'] +
              [f'  {r}' + ('' if os.path.exists(r) else '  (MISSING)') for r in raws])
        return 0 if all(os.path.exists(r) for r in raws) else 2
    m = _ep_module()
    rlog = os.path.join(rdir, 'replay_log.csv')
    cols = ['replay_utc', 'raw', 'kind', 'tag', 'status', 'n_pointings', 'file', 'detail']

    def log_row(row):
        new = not os.path.exists(rlog)
        os.makedirs(rdir, exist_ok=True)
        with open(rlog, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=cols)
            if new:
                w.writeheader()
            w.writerow(row)

    for raw in raws:
        kind, tag = _TAG_RE.match(os.path.basename(raw)).groups()
        name = f'replay_{kind}_{tag}'
        if _resumed(run, name, op='ep-plan', action='replay'):
            continue

        def body(u, raw=raw, kind=kind, tag=tag):
            import numpy as np
            u.add_input(raw)
            u.add_input(_script('ep_plan'))
            plan = json.loads(gzip.open(raw).read())
            fails = m.content_test(plan)
            row = dict(replay_utc=time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime()), raw=raw, kind=kind, tag=tag,
                       status='FAIL', n_pointings=len(plan) if isinstance(plan, list) else '', file='', detail='')
            if fails:
                row['detail'] = '; '.join(fails)[:480]
                log_row(row)
                u.add_output(rlog, 'csv_table', required_columns=cols)
                u.refuse(f'G1: today\'s content test refuses the stored download: {row["detail"]}')
            out = os.path.join(rdir, f'wxtplan_{tag}.npz')
            tmp = out[:-4] + '.part.npz'
            os.makedirs(rdir, exist_ok=True)
            np.savez_compressed(tmp, **m.to_compact(plan))
            rt = m.roundtrip_ok(plan, tmp)
            if rt:
                os.remove(tmp)
                row['detail'] = f'round trip: {rt}'
                log_row(row)
                u.add_output(rlog, 'csv_table', required_columns=cols)
                u.refuse(f'G2: {rt}')
            os.replace(tmp, out)
            row.update(status='pass', file=os.path.relpath(out, rdir), detail='content test passed; lossless round trip')
            log_row(row)
            u.add_output(out, 'ep_wxt_plan', raw=raw)
            u.add_output(rlog, 'csv_table', required_columns=cols)
            arch_npz = os.path.join(arch, f'{tag[:4]}-{tag[4:6]}', f'wxtplan_{tag}.npz')
            if os.path.exists(arch_npz):                                   # parity P-F1: arrays equal to the archive's
                a, b = np.load(arch_npz), np.load(out)
                same = all(np.array_equal(a[k], b[k]) for k in ('nside', 'start_utc', 'end_utc', 'pix', 'offsets'))
                u.check('ep_plan.replay_equals_archive', same, value=os.path.basename(arch_npz),
                        detail='arrays nside, start_utc, end_utc, pix, offsets equal to the archive plan of the same tag'
                        if same else f'arrays differ from {arch_npz}')
            else:
                u.note(f'no archive plan of tag {tag} (a refused download, or not archived): nothing to compare')
        _guarded(run, name, body, op='ep-plan', action='replay')
    return None


# =====================================================================================================================
# tess
# =====================================================================================================================

def _op_tess(args, run):
    from exopipe.phase3.core.record import NoInput
    CFG = run.cfg
    T = CFG.followup_tess
    missing = [f for f in ('name', 'radio_start', 'radio_end', 'radio_label') if not _given(args, f)]
    if missing:
        raise ValueError('followup tess needs ' + ', '.join('--' + m.replace('_', '-') for m in missing))
    key = args.key or 'tess_' + ''.join(ch for ch in args.name if ch.isalnum())
    pdir = os.path.join(run.out_root, key)
    argv = ['--name', args.name, '--radio-start', args.radio_start, '--radio-end', args.radio_end,
            '--radio-label', args.radio_label, '--out', pdir, '--key', key] + (
        ['--sector', str(args.sector)] if args.sector else [])
    if run.dry_run:
        _plan([f'tess {key}: {CFG.site.PYTHON.get(CFG.followup.TESS_PYTHON_KEY)} {_script("tess")} ' + ' '.join(
            repr(a) if ' ' in a else a for a in argv),
            f'  product dir {pdir}; orbit table {T.ORBIT_TABLE}; network: SIMBAD, MAST, NASA Exoplanet Archive'])
        return 0
    if _resumed(run, key, op='tess'):
        return None

    def body(u):
        t0 = time.time()
        u.add_input(T.ORBIT_TABLE)
        res = _call(u, run, 'tess', argv, python=CFG.followup.TESS_PYTHON_KEY)
        if res.rc != 0:
            if 'not resolved by SIMBAD' in res.tail:
                u.refuse(f'G9: {args.name!r} not resolved by SIMBAD - give the SIMBAD name (or a Gaia DR3 id)')
            raise RuntimeError(f'tess_lightcurve.py exit {res.rc}; see {res.log}')
        summ = os.path.join(pdir, f'{key}_summary.txt')
        if not os.path.exists(summ):
            raise RuntimeError(f'{summ} not written')
        text = open(summ).read()
        if 'NO TESS sector observed' in text:
            u.note(f'summary kept: {summ}')
            raise NoInput(f'G10: no TESS sector observed {args.name} during {args.radio_start} - {args.radio_end}')
        listing = [os.path.join(pdir, f) for f in sorted(os.listdir(pdir))]
        for p in _new_files(listing, t0):
            b = os.path.basename(p)
            if re.match(re.escape(key) + r'_s\d+_[A-Za-z0-9]+\.csv$', b):
                u.add_output(p, 'tess_lightcurve')
            elif re.match(re.escape(key) + r'_s\d+\.png$', b):
                u.add_output(p, 'followup_figure')
            elif b.endswith('_astrocut.fits'):
                u.add_input(p)
        u.add_output(summ, 'tess_summary')
    _guarded(run, key, body, op='tess', star=args.name)
    return None


# =====================================================================================================================
# investigate / summary (the follow-up package, in-process)
# =====================================================================================================================

def _package(run, out_root, modules):
    """Set the package's CANDINV_* paths from config (G13) and import ``modules`` from exopipe.phase3.followup."""
    import importlib
    from exopipe.phase3.core.record import Refusal
    from exopipe.phase3.followup import _cfg_adapter as A
    env = A.apply(run.cfg.followup_investigate, out_root)
    mods = {m: importlib.import_module(f'exopipe.phase3.followup.{m}') for m in modules}
    bad = A.check_applied(env)
    if bad:
        raise Refusal('G13: the follow-up package did not take its configured paths: ' + '; '.join(bad))
    return mods, env


def _candidates(args, CM, top50, default_all=False):
    from exopipe.phase3.core import cells
    spec = args.cand or ('all' if default_all and args.ra is None else None)
    if spec:
        allc = CM.load_top50(top50)
        if spec.strip() == 'all':
            return allc
        by = {c.rank: c for c in allc}
        ranks = [int(t) for t in cells.expand(spec)]
        missing = [r for r in ranks if r not in by]
        if missing:
            raise ValueError(f'--cand: rank(s) {missing} not in {top50}')
        return [by[r] for r in ranks]
    vals = (args.ra, args.dec, args.lst, args.date, args.band)
    if None in vals:
        raise ValueError('give --cand RANKS|@FILE|all, or --ra --dec --lst --date --band (rank-0 candidate)')
    cells.parse_lst(args.lst)
    cells.parse_date(args.date)
    if args.band not in cells.scopes():
        raise ValueError(f'--band {args.band!r}: a subband or band ({", ".join(cells.scopes())})')
    return [CM.Candidate(rank=0, ra=args.ra, dec=args.dec, band=args.band, date=args.date,
                         cadence=args.cadence or 'deep', lst=args.lst, iv=args.iv)]


# product class of each key of dossier.investigate()['outputs'] (dossier.py:397-464)
_DOSSIER_CLASSES = {'evidence': 'followup_dossier', 'spectrum_csv': 'followup_table', 'snapshot_csv': 'followup_table',
                    'snapshot_spectrum_csv': 'followup_table', 'reg': 'followup_region', 'allsky_V': 'followup_fits',
                    'cutout_V': 'followup_fits', 'allsky_I': 'followup_fits', 'cutout_I': 'followup_fits',
                    'snapshot_burst_fits': 'followup_fits', 'snapshot_png': 'followup_figure',
                    'index_png': 'followup_figure'}


def _op_investigate(args, run):
    from exopipe.phase3.core import fitsio
    CFG = run.cfg
    S = CFG.followup_investigate
    out = run.out_root
    if run.dry_run:
        mods, env = _package(run, out, ('candidate', 'resolve', 'config'))
        CM, R, C = mods['candidate'], mods['resolve'], mods['config']
        cands = _candidates(args, CM, S.TOP50_CSV)
        lines, ok = [f'investigate: {len(cands)} candidate(s); top-50 {S.TOP50_CSV}; bundle {S.BUNDLE_DIR}; '
                     f'dossiers under {os.path.dirname(C.candidate_dir("x"))}/'], 0
        for c in cands:
            got = {k: R.bundle_path(c.rank, c.lst, c.band, c.date, c.cadence, c.iv, k)
                   for k in ('Vresidual', 'Iresidual', 'Imaster')}
            ok += got['Vresidual'] is not None
            lines.append(f'  {c.cand_id}: {c.lst} {c.band} {c.date} {c.cadence}' + (f' bin {c.iv}' if c.iv is not None else '')
                         + ''.join(f'\n      {k}: {v or "NOT FOUND"}' for k, v in got.items())
                         + ('' if got['Vresidual'] else '\n      WOULD REFUSE G11: discovery V residual not in the bundle'))
        _plan(lines + [f'  escalation: T4 {"off" if args.no_escalate else "on"}, T5 '
                       f'{"off" if (args.no_escalate or args.no_snapshot) else "on"}; '
                       f'{"no files (--no-write)" if args.no_write else "dossier files written"}'])
        return 0 if ok else 2
    scratch = fitsio.scratch_dir('followup_investigate', args.work)      # CASA logs and T5 work stay node-local
    here = os.getcwd()
    os.chdir(scratch)
    try:
        mods, env = _package(run, out, ('candidate', 'resolve', 'refs', 'config', 'dossier'))
        CM, R, REFS, C, D = (mods[k] for k in ('candidate', 'resolve', 'refs', 'config', 'dossier'))
        cands = _candidates(args, CM, S.TOP50_CSV)
        no_xmatch = all(REFS.target_catalog(k) is None for k in C.TARGET_CSVS)
        for c in cands:
            if _resumed(run, c.cand_id, op='investigate'):
                continue

            def body(u, c=c):
                _investigate_one(u, run, args, c, R, D, C, env, no_xmatch)
            _guarded(run, c.cand_id, body, op='investigate', rank=c.rank, lst=c.lst, band=c.band, date=c.date)
    finally:
        os.chdir(here)
        fitsio.cleanup(scratch)
    return None


def _investigate_one(u, run, args, c, R, D, C, env, no_xmatch):
    import contextlib
    from exopipe.phase3.core.record import NoInput
    S = run.cfg.followup_investigate
    u.meta['candinv_env'] = env
    if c.rank:
        u.add_input(S.TOP50_CSV)
    for k in ('Vresidual', 'Iresidual', 'Imaster'):
        p = R.bundle_path(c.rank, c.lst, c.band, c.date, c.cadence, c.iv, k)
        if p:
            u.add_input(p)
    _exclusion_report(u, c.lst, c.band, [c.date])
    log = _log_path(run, c.cand_id)
    with open(log, 'a') as lf, contextlib.redirect_stdout(lf):
        print(f'===== {time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())} investigate rank{c.rank} {c.cand_id} =====')
        ev = D.investigate(c, escalate=not args.no_escalate, write=not args.no_write, verbose=True,
                           snapshot=not args.no_snapshot)
    u.meta['log'] = log
    for line in open(log, errors='replace'):
        if 'WARN' in line:
            u.note('in-place fallback/warning: ' + line.strip()[:300])
    if no_xmatch:
        u.note('cross-match empty: none of the catalogues in followup/config.py TARGET_CSVS exists (the CLV path points at '
               'exopipe/phase3, not clv15) - context.xmatch empty and xmatch_stellar 0, as in production (batch C-33/B-41)')
    pa = ev.get('p_astro') or {}
    u.meta['p_astro'] = {k: pa.get(k) for k in ('p_astro', 'logit', 'ateam_veto', 'components')}
    if 'error' in ev:
        gate = 'G11' if 'not found in bundle' in ev['error'] else 'G12'
        u.refuse(f'{gate}: {ev["error"]}')
    if args.no_write:
        raise NoInput(f'--no-write: evidence kept in this record only (P(astro) {pa.get("p_astro")})')
    for key, val in sorted((ev.get('outputs') or {}).items()):
        cls = _DOSSIER_CLASSES.get(key)
        if cls is None:
            continue
        for p in (val if isinstance(val, (list, tuple)) else [val]):
            if p:
                u.add_output(p, cls)


# ---- summary: port of attic/run_all_top50.py (SUMMARY_COLS :27-38, _round :41-42, _snap_peak_vi :45-55, _row :58-108,
# ---- --assemble :131-149, write :156-170); the attic module cannot be imported (flat imports candinv, phase3_paths).
_SUMMARY_COLS = ['rank', 'p_astro', 'ateam_veto', 'logit', 'cand_id', 'ra', 'dec', 'lst',
                 'date', 'band', 'cadence', 'v_peak_jy', 'v_err_jy', 'v_snr',
                 'n_v_bands', 'v_bands', 'n_epoch_detect', 'n_epoch_searched',
                 'persistent', 'iv_coincident', 'vi_ratio',
                 'point_like', 'morph_subband', 'maj_ratio',
                 'snap_point_like', 'snap_morph_band', 'snap_maj_ratio',
                 'snap_peak_snr', 'snap_peak_time', 'snap_burst_dur_s',
                 'snap_vi_peak', 'snap_spec_peak_band', 'snap_unphysical',
                 'elevation_deg', 'azimuth_deg', 'gal_b_deg',
                 'ecl_lat_deg', 'ecl_lon_deg', 'n_peaks_gt7sigma',
                 'short_structure_10min',
                 'nearest_ateam', 'ephemeris_body', 'xmatch', 'error']


def _round(x, n):
    return round(x, n) if isinstance(x, (int, float)) and x == x else ''


def _snap_peak_vi(ev):
    ss = [r for r in ev.get('snapshot_spectrum', []) if r.get('V_snr') is not None]
    if not ss:
        return '', '', False
    best = max(ss, key=lambda r: abs(r.get('V_snr', 0)))
    unphys = any(abs(r.get('V_snr', 0) or 0) >= 7.0 and (r.get('vi_ratio') or 0) > 1.25 for r in ss)
    return _round(best.get('vi_ratio'), 3), best.get('scope', ''), unphys


def _summary_row(cand, ev):
    sp = ev.get('spectrum_summary', {})
    me = ev.get('multi_epoch', {})
    iv = ev.get('iv_coincidence', {})
    mo = ev.get('morphology', {})
    sm = ev.get('snapshot_morphology', {})
    sn = ev.get('snapshot', {})
    cx = ev.get('context', {})
    tm = ev.get('tenmin', {})
    pa = ev.get('p_astro', {})
    df = ev.get('discovery_fit', {})
    snap_vi, snap_band, snap_unphys = _snap_peak_vi(ev)
    return {
        'rank': cand.rank, 'p_astro': pa.get('p_astro', ''),
        'ateam_veto': pa.get('ateam_veto', ''),
        'logit': pa.get('logit', ''), 'cand_id': cand.cand_id,
        'ra': cand.ra, 'dec': cand.dec,
        'lst': cand.lst, 'date': cand.date, 'band': cand.band,
        'cadence': cand.cadence,
        'v_peak_jy': _round(df.get('peak'), 4),
        'v_err_jy': _round(df.get('err') if (df.get('err') == df.get('err')
                           and df.get('err') is not None) else df.get('local_rms'), 4),
        'v_snr': _round(df.get('snr'), 1),
        'n_v_bands': sp.get('n_v_bands'),
        'v_bands': '|'.join(sp.get('v_bands', []) or []),
        'n_epoch_detect': me.get('n_epoch_detect'),
        'n_epoch_searched': me.get('n_epoch_searched'),
        'persistent': me.get('persistent'),
        'iv_coincident': iv.get('iv_coincident'),
        'vi_ratio': _round(iv.get('vi_ratio'), 3),
        'point_like': mo.get('point_like'),
        'morph_subband': mo.get('morph_subband', ''),
        'maj_ratio': _round(mo.get('maj_ratio'), 2),
        'snap_point_like': sm.get('point_like'),
        'snap_morph_band': sm.get('scope', ''),
        'snap_maj_ratio': _round(sm.get('v_maj_ratio', sm.get('i_maj_ratio')), 2),
        'snap_peak_snr': _round(sn.get('peak_snr'), 1),
        'snap_peak_time': sn.get('peak_time', ''),
        'snap_burst_dur_s': sn.get('burst_dur_s', ''),
        'snap_vi_peak': snap_vi, 'snap_spec_peak_band': snap_band,
        'snap_unphysical': snap_unphys,
        'azimuth_deg': _round(cx.get('azimuth_deg'), 0),
        'elevation_deg': _round(cx.get('elevation_deg'), 1),
        'gal_b_deg': _round(cx.get('gal_b_deg'), 1),
        'ecl_lat_deg': _round(cx.get('ecl_lat_deg'), 1),
        'ecl_lon_deg': _round(cx.get('ecl_lon_deg'), 1),
        'n_peaks_gt7sigma': cx.get('n_peaks_gt7sigma'),
        'short_structure_10min': tm.get('short_structure'),
        'nearest_ateam': cx.get('nearest_ateam'),
        'ephemeris_body': cx.get('ephemeris_body', ''),
        'xmatch': '|'.join(ev.get('context', {}).get('xmatch', {}).keys()),
        'error': ev.get('error', '')}


def _op_summary(args, run):
    import io
    CFG = run.cfg
    S = CFG.followup_investigate
    mods, env = _package(run, run.out_root, ('candidate', 'score', 'config'))
    CM, SC, C = mods['candidate'], mods['score'], mods['config']
    cands = _candidates(args, CM, S.TOP50_CSV, default_all=True)
    name = args.summary_name or 'investigation_summary.csv'
    cand_root = os.path.dirname(C.candidate_dir('x'))
    out_csv = os.path.join(cand_root, name)
    evid = {c.cand_id: os.path.join(C.candidate_dir(c.cand_id), f'{c.cand_id}_evidence.json') for c in cands}
    if run.dry_run:
        have = sum(os.path.exists(p) for p in evid.values())
        _plan([f'summary: {len(cands)} candidate(s) of {S.TOP50_CSV}; {have} with an evidence file under {cand_root}/',
               f'  output {out_csv} (+ .provenance.json); each evidence file re-scored and written back'])
        return 0 if have else 3

    def body(u):
        from exopipe.common.phase2_config import PIPELINE_GITTAG
        u.meta['candinv_env'] = env
        u.add_input(S.TOP50_CSV)
        rows, rescored = [], []
        for c in cands:
            ej = evid[c.cand_id]
            try:
                with open(ej) as f:
                    ev = json.load(f)
                ev['p_astro'] = SC.p_astrophysical(ev)
                _atomic_text(ej, json.dumps(ev, indent=2, default=str))
                rescored.append(ej)
            except Exception:
                ev = {'error': 'no dossier / evidence.json'}
            rows.append(_summary_row(c, ev))
        rows.sort(key=lambda r: (r['p_astro'] if isinstance(r['p_astro'], (int, float)) else -1), reverse=True)
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=_SUMMARY_COLS, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)
        _atomic_text(out_csv, buf.getvalue())
        _atomic_text(out_csv + '.provenance.json', json.dumps(
            {'gittag': PIPELINE_GITTAG, 'generator': 'candinv/run_all_top50', 'n_candidates': len(rows),
             'source_csv': S.TOP50_CSV, 'escalate': True, 'snapshot': True, 'rank_metric': 'p_astrophysical'}, indent=2))
        u.meta['n_without_evidence'] = len(cands) - len(rescored)
        if len(rescored) < len(cands):
            u.note(f'{len(cands) - len(rescored)} of {len(cands)} candidates have no evidence file: rows with error '
                   f'"no dossier / evidence.json" (today\'s behaviour)')
        for ej in rescored:
            u.add_output(ej, 'followup_dossier')
        u.add_output(out_csv, 'followup_summary')
    stem = os.path.splitext(name)[0]
    _guarded(run, f'summary_{stem}', body, op='summary')
    return None


# =====================================================================================================================
# hone
# =====================================================================================================================

def _hone_frames(H, band_subbands, cand, band):
    """{subband: (tree, [frames])} as phase3_honing.manifest_cell finds them (HO:60-68)."""
    import glob
    out = {}
    for sb in band_subbands.get(cand.get('band') or band, ()):
        for tree in H.INPUT_TREES:
            g = sorted(glob.glob(os.path.join(H.INPUT_ROOT, cand['lst'], cand['date'], tree, sb, 'V', '10min',
                                              H.INPUT_GLOB)))
            if g:
                out[sb] = (tree, g)
                break
    return out


def _op_hone(args, run):
    import shutil
    from exopipe.phase3.core import fitsio, legacy
    from exopipe.phase3.core.record import NoInput
    CFG = run.cfg
    H = CFG.followup_hone
    if not args.cand_csv:
        raise ValueError('followup hone needs --cand-csv')
    cols, cands = _read_csv(args.cand_csv)
    need = ['cand_id', 'lst', 'date', 'ra', 'dec', 'snr']
    if [c for c in need if c not in cols]:
        raise ValueError(f'--cand-csv {args.cand_csv}: columns {need} required, has {cols}')
    ids = [c['cand_id'] for c in cands]
    if len(set(ids)) != len(ids):
        raise ValueError(f'--cand-csv: duplicate cand_id values ({len(ids) - len(set(ids))})')
    band = args.band or H.DEFAULT_BAND
    if band not in ('Red', 'Green', 'Blue'):
        raise ValueError(f'hone --band {band!r}: Red, Green or Blue')
    controls = H.N_CONTROLS if args.controls is None else args.controls
    subs = {b: tuple(v) for b, v in CFG.instrument.SUBBAND_BANDS.items()}
    hdir = os.path.join(run.out_root, 'honing')
    honed = os.path.join(hdir, 'honed.csv')
    stamps = os.path.join(hdir, 'stamps') if args.stamps else None
    frames = {c['cand_id']: _hone_frames(H, subs, c, band) for c in cands}
    if run.dry_run:
        lines = [f'hone: {len(cands)} candidate(s), band {band}, {controls} controls -> {honed}'
                 + (f', stamps {stamps}/' if stamps else '')]
        for c in cands:
            f = frames[c['cand_id']]
            lines.append(f'  {c["cand_id"]}: {c["lst"]} {c["date"]} band {c.get("band") or band}: ' + (', '.join(
                f'{sb} {t} {len(g)} frames' for sb, (t, g) in sorted(f.items())) if f else
                'no 10-minute frames under ' + H.INPUT_ROOT + ' - would end no-products (G17, exit 3)'))
        _plan(lines)
        return 0 if any(frames.values()) else 3
    scratch = fitsio.scratch_dir('followup_hone', args.work)
    try:
        csv_in = os.path.join(scratch, os.path.basename(args.cand_csv))
        shutil.copyfile(args.cand_csv, csv_in)
        os.makedirs(hdir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(args.cand_csv))[0]
        t0 = time.time()
        res = legacy.run(_script('hone'), ['--csv', csv_in, '--out', honed, '--band', band, '--controls', str(controls)]
                         + (['--stamps', stamps] if stamps else []) + (['--snapshots'] if args.snapshots else []),
                         python=CFG.followup.PYTHON_KEY, cwd=scratch, log=_log_path(run, f'hone_{stem}'),
                         refusal_codes=(), refusal_text=('REFUSED:',))
        rows = {}
        if res.rc == 0 and os.path.exists(honed) and os.path.getmtime(honed) >= t0 - 1:
            rows = {r['cand_id']: r for r in _read_csv(honed)[1]}
        for c in cands:
            cid = c['cand_id']

            def body(u, c=c, cid=cid):
                u.add_input(_script('hone'))
                u.add_input(args.cand_csv)
                u.meta['legacy_calls'] = [{'argv': res.argv, 'rc': res.rc, 'wall_s': res.wall_s, 'log': res.log}]
                u.meta['legacy_code'] = res.fingerprint
                for f in sorted(res.fingerprint or {}):
                    if os.path.isfile(f):
                        u.add_input(f)
                for sb, (tree, g) in sorted(frames[cid].items()):
                    for p in g:
                        u.add_input(p)
                _exclusion_report(u, c['lst'], c.get('disc_sb') or c.get('band') or band, [c['date']])
                if res.rc != 0:
                    raise RuntimeError(f'phase3_honing.py exit {res.rc}; see {res.log}')
                row = rows.get(cid)
                if row is None:
                    raise RuntimeError(f'{cid} has no row in {honed}')
                u.meta['honing'] = row
                st = row.get('status')
                if st == 'honed':
                    sp = os.path.join(stamps, f'{cid}_honedbox.fits') if stamps else None
                    if sp and os.path.exists(sp):
                        u.add_output(sp, 'honing_stamp')
                    u.add_output(honed, 'honing_table', cand_id=cid)
                elif st == 'repro-fail':
                    u.add_output(honed, 'honing_table', cand_id=cid)
                    u.refuse(f'G16: reproduction gate failed: own-product stack S/N {row.get("stack_all_snr")} vs '
                             f'discovery {row.get("snr")} (tolerance {H.REPRO_TOL}, same sign)')
                elif st in ('no-products', 'no-box'):
                    raise NoInput(f'G17: {st}' + (' (the 10-minute frames under ' + H.INPUT_ROOT + ' were deleted on '
                                                  '2026-09-01)' if st == 'no-products' else ''))
                else:
                    raise RuntimeError(f'unknown honing status {st!r}')
            _guarded(run, cid, body, op='hone', lst=c['lst'], date=c['date'])
    finally:
        fitsio.cleanup(scratch)
    return None


# =====================================================================================================================
# periodicity (pass 5: lightcurves -> search -> page)
# =====================================================================================================================

def _gfmt(v):
    return str(v) if isinstance(v, int) and not isinstance(v, bool) else '%g' % v


def _search_env(P):
    """The search environment exactly as run_p4_fleet.sh:12 sets it (POINT_NOISE_GATE = stage2h4's own default)."""
    return {'ELEV_FLOOR': _gfmt(P.ELEV_FLOOR_DEG), 'MIN_BLOCKS': _gfmt(P.MIN_BLOCKS),
            'MIN_NIGHTS_LONG': _gfmt(P.MIN_NIGHTS_LONG), 'FRAME_NOISE_GATE': _gfmt(P.FRAME_NOISE_GATE),
            'POINT_NOISE_GATE': _gfmt(P.POINT_NOISE_GATE), 'NULL_N': _gfmt(P.NULL_N), 'PYTHONDONTWRITEBYTECODE': '1'}


def _jobs(args, P):
    """[(band, S)] of the search/page jobs: DEFAULT_JOBS unless --bands / --stokes are given."""
    if not args.bands and not args.stokes:
        return [tuple(j.split('_')) for j in P.DEFAULT_JOBS]
    bands = [b.strip() for b in (args.bands or ','.join(P.BANDS)).split(',') if b.strip()]
    bad = [b for b in bands if b not in P.BANDS]
    if bad:
        raise ValueError(f'--bands {bad}: periodicity bands are {",".join(P.BANDS)} (ruled scope)')
    stokes = [s.strip().upper() for s in (args.stokes or 'I').split(',') if s.strip()]
    if [s for s in stokes if s not in ('I', 'V')]:
        raise ValueError(f'--stokes {args.stokes!r}: I, V or I,V')
    return [(b, s) for s in stokes for b in bands]


def _lc_frames(P, lst, band, S):
    import glob
    return sorted(glob.glob(os.path.join(P.LC_ARCHIVE_ROOT, lst, '*', P.LC_GENERATION, band, S, '10min',
                                         f'{band}_{S}_10min_Taper_Robust+0.0_pbcorr_dewarped_aligned_msub_LST{lst}_bin*'
                                         f'.fits')))


def _night_of(path, lst):
    m = re.search(r'/' + lst + r'/(\d{4}-\d{2}-\d{2})/', path)
    return m.group(1) if m else None


def _recorded(run, path, unit, record_dirs):
    """Where a file's provenance comes from: a done unit of this run, or a done record of an earlier run that lists it as
    an output and finished after the file was written. None when neither (gate G28)."""
    from exopipe.phase3.core import record as R
    ap = os.path.abspath(path)
    for u in run.units:
        if u.status == 'done' and any(o['path'] == ap for o in u.outputs):
            return f'this run, unit {u.name}'
    try:
        mt = os.path.getmtime(path)
    except OSError:
        return None
    for d in record_dirs:
        p = os.path.join(d, f'{unit}.json')
        if not os.path.exists(p):
            continue
        try:
            rec = json.load(open(p))
        except ValueError:
            continue
        fin = rec.get('finished') or ''
        try:
            fin_t = time.mktime(time.strptime(fin, '%Y-%m-%dT%H:%M:%SZ')) - time.timezone
        except ValueError:
            continue
        if rec.get('status') == 'done' and any(o.get('path') == ap for o in rec.get('outputs', [])) and fin_t + 1 >= mt:
            return f'record {p}'
    return None


def _op_periodicity(args, run):
    from exopipe.phase3.core import cells
    CFG = run.cfg
    P = CFG.followup_periodicity
    stages = _stages(args)
    jobs = _jobs(args, P)
    if args.cells and 'lightcurves' not in stages:
        raise ValueError('--cells selects light-curve cells: it needs the lightcurves stage')
    with_out = bool(args.out)
    lc_out = os.path.join(run.out_root, 'lightcurves') if with_out else P.LC_ROOT
    search_out = os.path.join(run.out_root, 'search') if with_out else P.SEARCH_OUT
    merged_out = os.path.join(run.out_root, 'page') if with_out else P.MERGED_DIR
    lc_dir = lc_out if 'lightcurves' in stages else os.path.abspath(args.lc_root or P.LC_ROOT)
    rec_dirs = list(dict.fromkeys([os.path.join(run.out_root, '_records', 'followup'),
                                   os.path.join(os.path.dirname(os.path.abspath(lc_dir)), '_records', 'followup'),
                                   os.path.join(os.path.dirname(os.path.abspath(search_out)), '_records', 'followup')]))
    rc = 0
    if 'lightcurves' in stages:
        stokes = sorted({s for _, s in jobs})
        if args.cells:
            items = cells.expand_cells(args.cells)
            bad = [c for c in items if cells.parse_cell(c)[1] not in P.BANDS]
            if bad:
                raise ValueError(f'--cells {bad[:4]}: light-curve cells are <LST>_<band>, band {",".join(P.BANDS)}')
        else:
            items = [f'{l}_{b}' for b in sorted({b for b, _ in jobs}, key=P.BANDS.index) for l in cells.LSTS]
        units = [(cells.parse_cell(c)[0], cells.parse_cell(c)[1], s) for c in items for s in stokes]
        rc = max(rc, _pd_lightcurves(args, run, P, units, lc_out))
    if 'search' in stages:
        rc = max(rc, _pd_search(args, run, P, jobs, lc_dir, search_out, rec_dirs, 'lightcurves' in stages))
    if 'page' in stages:
        rc = max(rc, _pd_page(args, run, P, sorted({s for _, s in jobs}), lc_dir, search_out, merged_out, rec_dirs,
                              'search' in stages))
    return rc if run.dry_run else None


def _pd_lightcurves(args, run, P, units, lc_out):
    from concurrent.futures import ThreadPoolExecutor
    from exopipe.phase3.core import exclusions
    from exopipe.phase3.core.record import NoInput
    if run.dry_run:
        lines, n_any = [f'periodicity lightcurves: {len(units)} unit(s) -> {lc_out}/ (positions {P.LC_POSITIONS}, '
                        f'generation {P.LC_GENERATION})'], 0
        for lst, band, S in units:
            fr = _lc_frames(P, lst, band, S)
            nights = sorted({_night_of(f, lst) for f in fr} - {None})
            n_any += bool(fr)
            lines.append(f'  lc3_{lst}_{band}_{S}: {len(fr)} frames, {len(nights)} nights'
                         + ('' if fr else ' - would end no-input (G26)'))
        _plan(lines)
        return 0 if n_any else 3

    def one(item):
        lst, band, S = item
        name = f'lc3_{lst}_{band}_{S}'
        fr = _lc_frames(P, lst, band, S)
        nights = sorted({_night_of(f, lst) for f in fr} - {None})
        adm = exclusions.load().admitted(lst, band, nights)
        if _resumed(run, name, admitted=adm, op='periodicity', stage='lightcurves'):
            return

        def body(u):
            if not fr:
                raise NoInput(f'G26: no {P.LC_GENERATION} 10-minute pbcorr msub frames for {lst} {band} {S}')
            u.add_input(P.LC_POSITIONS)
            for f in fr:
                u.add_input(f)
            _exclusion_report(u, lst, band, nights)
            t0 = time.time()
            res = _call(u, run, 'lightcurves', [P.LC_POSITIONS, lc_out, lst, band, S],
                        env={'P3_GEN': P.LC_GENERATION, 'P3_ARCHIVE_ROOT': P.LC_ARCHIVE_ROOT,
                             'PYTHONDONTWRITEBYTECODE': '1'})
            if res.rc != 0:
                raise RuntimeError(f'stage2g2_lightcurves_p3.py exit {res.rc}; see {res.log}')
            npz = os.path.join(lc_out, f'{name}.npz')
            if not _new_files([npz], t0):
                raise RuntimeError(f'{npz} not written by this run; see {res.log}')
            u.add_output(npz, 'periodicity_lightcurves')
        _guarded(run, name, body, op='periodicity', stage='lightcurves', lst=lst, band=band, stokes=S)

    lanes = max(1, args.workers or P.LC_LANES)
    with ThreadPoolExecutor(max_workers=lanes) as ex:
        list(ex.map(one, units))
    return 0


def _pd_search(args, run, P, jobs, lc_dir, search_out, rec_dirs, lc_this_run):
    from exopipe.phase3.core import cells
    from exopipe.phase3.core.record import NoInput, Refusal
    pinned = os.path.realpath(lc_dir) == os.path.realpath(P.LC_ROOT)
    ncpu = args.ncpu or min(P.NCPU, len(os.sched_getaffinity(0)))
    if run.dry_run:
        lines = [f'periodicity search: {len(jobs)} job(s), light curves from '
                 + (f'this invocation\'s lightcurves ({lc_dir})' if lc_this_run else
                    f'{lc_dir}' + (' (pinned pass-5 root)' if pinned else '')) + f' -> {search_out}/; {ncpu} processes; '
                 f'env {" ".join(f"{k}={v}" for k, v in _search_env(P).items() if k != "PYTHONDONTWRITEBYTECODE")}']
        for band, S in jobs:
            have = [l for l in cells.LSTS if os.path.exists(os.path.join(lc_dir, f'lc3_{l}_{band}_{S}.npz'))]
            unrec = [] if (pinned or lc_this_run) else [
                l for l in have if not _recorded(run, os.path.join(lc_dir, f'lc3_{l}_{band}_{S}.npz'),
                                                 f'lc3_{l}_{band}_{S}', rec_dirs)]
            lines.append(f'  {band}_{S}: {len(have)} of 24 cells present now' + (
                f'; {len(unrec)} without a done lightcurves record - would refuse (G28)' if unrec else '') + (
                '' if lc_this_run or len(have) == 24 else ' - would refuse (G28: 24 cells needed)'))
        _plan(lines)
        return 0

    for band, S in jobs:
        name = f'{band}_{S}'
        if _resumed(run, name, op='periodicity', stage='search'):
            continue

        def body(u, band=band, S=S):
            files = [os.path.join(lc_dir, f'lc3_{l}_{band}_{S}.npz') for l in cells.LSTS]
            present = [f for f in files if os.path.exists(f)]
            if not present:
                raise NoInput(f'G27: no lc3_*_{band}_{S}.npz in {lc_dir}')
            prov = {}
            for f in present:
                src = 'pinned pass-5 root ' + P.LC_ROOT if pinned else _recorded(
                    run, f, os.path.basename(f)[:-4], rec_dirs)
                if src is None:
                    raise Refusal(f'G28: {f} is not an output of a done lightcurves unit (this run or a record under '
                                  f'{", ".join(rec_dirs)}) - the search reads only recorded light curves')
                prov[f] = src
            if len(present) < len(files):
                raise Refusal(f'G28: {len(present)} of 24 cells\' light curves present for {band} {S} in {lc_dir} '
                              f'(missing {", ".join(os.path.basename(f) for f in files if f not in present)[:300]})')
            u.meta['lc_provenance'] = prov
            for f in present:
                u.add_input(f)
            u.add_input(P.SOURCES_CSV)
            t0 = time.time()
            res = _call(u, run, 'search', [lc_dir, band, S, search_out, str(ncpu)], env=_search_env(P))
            if res.rc != 0:
                if 'no lc3 files' in res.tail:
                    raise NoInput(f'G27: {res.tail.splitlines()[-1][:200]}')
                raise RuntimeError(f'stage2h4_periodicity_win.py exit {res.rc}; see {res.log}')
            out = os.path.join(search_out, f'periodicity_{band}_{S}.csv')
            if not _new_files([out], t0):
                raise RuntimeError(f'{out} not written by this run; see {res.log}')
            u.add_output(out, 'periodicity_table', lc_files=present)
        _guarded(run, name, body, op='periodicity', stage='search', band=band, stokes=S)
    return 0


def _pd_page(args, run, P, stokes_list, lc_dir, search_out, merged_out, rec_dirs, search_this_run):
    from exopipe.phase3.core.record import Refusal
    pinned = os.path.realpath(search_out) == os.path.realpath(P.SEARCH_OUT)
    env_base = {'FRAME_NOISE_GATE': _gfmt(P.FRAME_NOISE_GATE), 'POINT_NOISE_GATE': _gfmt(P.POINT_NOISE_GATE),
                'ELEV_FLOOR': _gfmt(P.ELEV_FLOOR_DEG), 'P3_ARCHIVE_ROOT': P.LC_ARCHIVE_ROOT,
                'P3_GEN': P.LC_GENERATION, 'PYTHONDONTWRITEBYTECODE': '1'}
    for S in stokes_list:
        page_dir = os.path.abspath(args.page_dir) if args.page_dir else os.path.join(merged_out, f'periodicity_page_{S}')
        tables = [os.path.join(search_out, f'periodicity_{b}_{S}.csv') for b in ('Red', 'Green', 'Blue')]
        if run.dry_run:
            _plan([f'periodicity page_{S}: tables ' + ', '.join(
                f'{os.path.basename(t)}{"" if os.path.exists(t) or search_this_run else " (MISSING: would refuse G28)"}'
                for t in tables) + f' -> {merged_out}/allsky_periodicity_{S}.csv, {page_dir}/index.html'])
            continue
        name = f'page_{S}'
        if _resumed(run, name, op='periodicity', stage='page'):
            continue

        def body(u, S=S, page_dir=page_dir, tables=tables):
            prov = {}
            for t in tables:
                if not os.path.exists(t):
                    raise Refusal(f'G28: {t} missing - the page merges the three band tables (stage2i reads Red, Green, '
                                  f'Blue)')
                band = os.path.basename(t)[len('periodicity_'):-len(f'_{S}.csv')]
                src = ('pinned pass-5 root ' + P.SEARCH_OUT) if pinned else _recorded(run, t, f'{band}_{S}', rec_dirs)
                if src is None:
                    raise Refusal(f'G28: {t} is not an output of a done search unit')
                prov[t] = src
                u.add_input(t)
            u.meta['table_provenance'] = prov
            t0 = time.time()
            res = _call(u, run, 'page', [search_out, lc_dir, page_dir, merged_out], env=dict(env_base, P3_STOKES=S))
            if res.rc != 0:
                raise RuntimeError(f'stage2i_periodicity_page.py exit {res.rc}; see {res.log}')
            merged = os.path.join(merged_out, f'allsky_periodicity_{S}.csv')
            index = os.path.join(page_dir, 'index.html')
            for p in (merged, index):
                if not _new_files([p], t0):
                    raise RuntimeError(f'{p} not written by this run; see {res.log}')
            u.add_output(merged, 'periodicity_merged', tables=tables)
            u.add_output(index, 'followup_page', merged=merged)
        _guarded(run, name, body, op='periodicity', stage='page', stokes=S)
    return 0


# =====================================================================================================================
# fold, source-page
# =====================================================================================================================

def _op_fold(args, run):
    missing = [f for f in ('lightcurve', 'prefix', 'period_d', 'fold_bands', 'ra', 'dec') if not _given(args, f)]
    if missing:
        raise ValueError('followup fold needs ' + ', '.join('--' + m.replace('_', '-') for m in missing))
    if not os.path.isfile(args.lightcurve):
        raise ValueError(f'--lightcurve {args.lightcurve}: not a file')
    out = run.out_root
    argv = [os.path.abspath(args.lightcurve), out, args.prefix, repr(args.ra), repr(args.dec), repr(args.period_d),
            args.fold_bands] + ([args.ylim] if args.ylim else [])
    outs = {'fold_table': [f'{args.prefix}_folded.csv'],
            'followup_figure': [f'{args.prefix}_folded.png', f'{args.prefix}_smoothed.png',
                                f'{args.prefix}_period_search.png']}
    if run.dry_run:
        _plan([f'fold {args.prefix}: {_script("fold")} ' + ' '.join(argv),
               '  outputs ' + ', '.join(os.path.join(out, f) for v in outs.values() for f in v)])
        return 0
    if _resumed(run, args.prefix, op='fold'):
        return None

    def body(u):
        u.add_input(args.lightcurve)
        os.makedirs(out, exist_ok=True)
        t0 = time.time()
        res = _call(u, run, 'fold', argv)
        if res.rc != 0:
            raise RuntimeError(f'steep_fold.py exit {res.rc}; see {res.log}')
        for cls, names in outs.items():
            for n in names:
                p = os.path.join(out, n)
                if not _new_files([p], t0):
                    raise RuntimeError(f'{p} not written by this run')
                u.add_output(p, cls, **({'source': os.path.abspath(args.lightcurve),
                                         'bands': args.fold_bands.split(',')} if cls == 'fold_table' else {}))
    _guarded(run, args.prefix, body, op='fold')
    return None


def _op_source_page(args, run):
    missing = [f for f in ('source_id', 'ra', 'dec') if not _given(args, f)]
    if missing:
        raise ValueError('followup source-page needs ' + ', '.join('--' + m.replace('_', '-') for m in missing))
    out = run.out_root
    argv = [args.source_id, repr(args.ra), repr(args.dec), out] + ([args.page_lsts] if args.page_lsts else [])
    if run.dry_run:
        _plan([f'source-page {args.source_id}: {_script("source_page")} ' + ' '.join(argv),
               f'  output {os.path.join(out, "index.html")} + {out}/img/ (reads the archive and the pass-4 tables named '
               f'inside the script)'])
        return 0
    if _resumed(run, args.source_id, op='source-page'):
        return None

    def body(u):
        os.makedirs(out, exist_ok=True)
        t0 = time.time()
        res = _call(u, run, 'source_page', argv)
        if res.rc != 0:
            raise RuntimeError(f'source_diag_page.py exit {res.rc}; see {res.log}')
        index = os.path.join(out, 'index.html')
        if not _new_files([index], t0):
            raise RuntimeError(f'{index} not written by this run')
        u.add_output(index, 'followup_page')
    _guarded(run, args.source_id, body, op='source-page')
    return None


# =====================================================================================================================
# content tests - one per product class (spec section 6); each reads the file's CONTENT and derives what it needs from
# the path (or the optional context the step passes)
# =====================================================================================================================

def _cfg():
    from exopipe.phase3.config import CFG
    return CFG


def _script_literal(key, name):
    """The value of the LAST top-level assignment ``name = <literal>`` in an in-place script (read, never run)."""
    import ast
    tree = ast.parse(open(_script(key), encoding='utf-8', errors='replace').read())
    val = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            val = ast.literal_eval(node.value)
    if val is None:
        raise KeyError(f'no top-level {name} = <literal> in {_script(key)}')
    return val


def _iso(s):
    from datetime import datetime
    return datetime.fromisoformat(str(s))


# ---- Einstein Probe -------------------------------------------------------------------------------------------------

@content_test('ep_wxt_plan')
def plan_valid(path, raw=None, **ctx):
    import numpy as np
    E = _cfg().followup_ep_plan
    z = np.load(path)
    nside = int(z['nside'])
    st, en, pix, off = z['start_utc'], z['end_utc'], z['pix'], z['offsets']
    bad = []
    if nside != E.NSIDE:
        bad.append(f'nside {nside} != {E.NSIDE}')
    if len(off) != len(st) + 1 or len(en) != len(st) or not len(st):
        bad.append(f'{len(st)} starts, {len(en)} ends, {len(off)} offsets')
    elif np.any(np.diff(off) <= 0) or off[0] != 0 or off[-1] != len(pix):
        bad.append('offsets not strictly increasing from 0 to len(pix)')
    if len(pix) and int(pix.max()) >= 12 * nside ** 2:
        bad.append(f'cell {int(pix.max())} >= 12 nside^2')
    cell_deg2 = 4 * np.pi * (180 / np.pi) ** 2 / (12 * nside ** 2)
    prev_end = None
    for i in range(min(len(st), len(off) - 1)):
        t0, t1 = _iso(st[i]), _iso(en[i])
        dur_h = (t1 - t0).total_seconds() / 3600
        if not E.POINTING_MIN_MIN / 60 <= dur_h <= E.POINTING_MAX_H:
            bad.append(f'pointing {i}: duration {dur_h:.2f} h outside {E.POINTING_MIN_MIN} min .. {E.POINTING_MAX_H:g} h')
        if prev_end is not None and t0 < prev_end:
            bad.append(f'pointing {i}: starts before the previous ends')
        prev_end = t1
        cells_i = pix[off[i]:off[i + 1]]
        if len(set(cells_i.tolist())) != len(cells_i):
            bad.append(f'pointing {i}: duplicate cells')
        area = len(cells_i) * cell_deg2
        if not E.FOOTPRINT_MIN_DEG2 <= area <= E.FOOTPRINT_MAX_DEG2:
            bad.append(f'pointing {i}: footprint {area:.0f} deg2')
    span = (_iso(en[-1]) - _iso(st[0])).total_seconds() / 3600 if len(st) else 0
    if span < E.MIN_SPAN_H:
        bad.append(f'plan spans {span:.1f} h < {E.MIN_SPAN_H:g}')
    m = re.search(r'wxtplan_(\d{8}T\d{6}Z)\.npz$', path)
    raw = raw or (os.path.join(os.path.dirname(path), f'raw_{m.group(1)}.json.gz') if m else None)
    rt = ''
    if raw and os.path.exists(raw):
        rt = _ep_module().roundtrip_ok(json.loads(gzip.open(raw).read()), path)
        if rt:
            bad.append(f'round trip against {os.path.basename(raw)}: {rt}')
    return (not bad, len(st), '; '.join(bad[:4]) if bad else
            f'{len(st)} pointings, span {span:.1f} h' + (f'; lossless against {os.path.basename(raw)}'
                                                          if raw and os.path.exists(raw) else ''))


@content_test('ep_wxt_raw')
def raw_passes_and_matches(path, **ctx):
    m = _ep_module()
    plan = json.loads(gzip.open(path).read())
    fails = m.content_test(plan)
    if fails:
        return False, len(fails), 'stored passing download fails today\'s content test: ' + '; '.join(fails[:3])
    mm = _TAG_RE.match(os.path.basename(path))
    npz = os.path.join(os.path.dirname(path), f'wxtplan_{mm.group(2)}.npz') if mm else None
    if not npz or not os.path.exists(npz):
        return False, len(plan), f'no stored plan of the same tag next to it ({npz})'
    rt = m.roundtrip_ok(plan, npz)
    return not rt, len(plan), rt or f'{len(plan)} pointings; passes; {os.path.basename(npz)} reproduces it losslessly'


@content_test('ep_wxt_refused')
def raw_refusal_recorded(path, **ctx):
    """A kept refused download: it parses, and fetch_log.csv holds its FAIL row naming it. Today's content test is re-run;
    when it still refuses, its first failure must be the logged reason (when a bound changed since - e.g. the pointing
    bound 3 h -> 6 h, e8338e6 - it may now pass: reported, not failed; `ep-plan --action replay` archives such a plan)."""
    m = _ep_module()
    plan = json.loads(gzip.open(path).read())
    mm = _TAG_RE.match(os.path.basename(path))
    if not mm or mm.group(1) != 'refused':
        return False, None, 'not a refused_<tag>.json.gz name'
    log = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(path))), 'fetch_log.csv')
    rows = [r for r in _read_csv(log)[1] if r['status'] == 'FAIL' and f'refused_{mm.group(2)}.json.gz' in r['detail']] \
        if os.path.exists(log) else []
    if len(rows) != 1:
        return False, len(rows), f'{len(rows)} FAIL rows of {log} name this file (expected 1)'
    fails = m.content_test(plan)
    if fails and fails[0][:60] not in rows[0]['detail']:
        return False, len(fails), f'today\'s first failure "{fails[0][:80]}" is not the logged reason'
    return True, len(fails), (f'logged refusal reproduced: {fails[0][:80]}' if fails else
                              f'logged refusal ({rows[0]["detail"][:60]}...) no longer reproduces under today\'s bounds')


@content_test('ep_wxt_fetch_log')
def log_rows_consistent(path, **ctx):
    cols, rows = _read_csv(path)
    want = list(_ep_module().LOG_COLS)
    if cols != want:
        return False, len(rows), f'header {cols} != {want}'
    bad, prev = [], None
    root = os.path.dirname(os.path.abspath(path))
    for i, r in enumerate(rows):
        if r['status'] not in ('pass', 'FAIL'):
            bad.append(f'row {i}: status {r["status"]!r}')
        if prev and r['fetch_utc'] <= prev:
            bad.append(f'row {i}: fetch_utc not increasing')
        prev = r['fetch_utc']
        if r['status'] == 'pass':
            f = os.path.join(root, r['file'])
            if not os.path.exists(f):
                bad.append(f'row {i}: {r["file"]} missing')
                continue
            import numpy as np
            z = np.load(f)
            if (len(z['start_utc']) != int(r['n_pointings']) or str(z['start_utc'][0]) != r['plan_start']
                    or str(z['end_utc'][-1]) != r['plan_end']):
                bad.append(f'row {i}: {r["file"]} does not match the row (pointings, start, end)')
        elif not r['detail'].strip():
            bad.append(f'row {i}: FAIL without detail')
    npass = sum(r['status'] == 'pass' for r in rows)
    return not bad, len(rows), '; '.join(bad[:4]) if bad else f'{len(rows)} rows ({npass} pass), every pass file matches'


@content_test('ep_wxt_coverage_gaps')
def gaps_recomputed(path, **ctx):
    """The gaps recomputed as audit() computes them, over the stored plans that existed when the CSV was written (npz
    mtime <= the CSV's); plans stored later are reported (the next audit adds them), not failed."""
    import numpy as np
    from datetime import datetime
    root = os.path.dirname(os.path.abspath(path))
    t_csv = os.path.getmtime(path)
    iv, newer = [], 0
    for dp, _, fs in os.walk(root):
        for f in fs:
            if f.startswith('wxtplan_') and f.endswith('.npz') and '.part' not in f:
                p = os.path.join(dp, f)
                if os.path.getmtime(p) > t_csv:
                    newer += 1
                    continue
                z = np.load(p)
                iv += [(datetime.fromisoformat(str(a)), datetime.fromisoformat(str(b)))
                       for a, b in zip(z['start_utc'], z['end_utc'])]
    if not iv:
        return False, 0, 'no stored plan older than the CSV'
    iv.sort()
    gaps, cur = [], iv[0][1]
    for a, b in iv[1:]:
        if a > cur:
            gaps.append([cur.isoformat(), a.isoformat(), f'{(a - cur).total_seconds() / 60:.1f}'])
        cur = max(cur, b)
    cols, rows = _read_csv(path)
    have = [[r['gap_start_utc'], r['gap_end_utc'], r['minutes']] for r in rows]
    ok = cols == ['gap_start_utc', 'gap_end_utc', 'minutes'] and have == gaps
    return ok, len(rows), ((f'{len(rows)} gaps, identical to the recomputation over {len(iv)} stored pointings'
                            if ok else f'CSV {have[:3]} != recomputed {gaps[:3]}')
                           + (f'; {newer} plan(s) stored after it (next audit)' if newer else ''))


# ---- TESS -----------------------------------------------------------------------------------------------------------

@content_test('tess_summary')
def summary_facts(path, **ctx):
    text = open(path, errors='replace').read()
    key = os.path.basename(path)[:-len('_summary.txt')]
    d = os.path.dirname(os.path.abspath(path))
    bad = []
    if not re.search(r'^STAR ', text, re.M):
        bad.append('no STAR line')
    if not re.search(r'^RADIO ', text, re.M):
        bad.append('no RADIO line')
    secs = re.findall(r'^SECTOR (\d+)', text, re.M)
    if not secs and 'NO TESS sector observed' not in text:
        bad.append('neither SECTOR lines nor "NO TESS sector observed"')
    blocks = re.split(r'^SECTOR \d+', text, flags=re.M)[1:]
    for s, blk in zip(secs, blocks):
        csvs = [f for f in os.listdir(d) if re.match(re.escape(key) + rf'_s{s}_[A-Za-z0-9]+\.csv$', f)]
        if not csvs and 'download failed' not in blk:
            bad.append(f'sector {s}: no {key}_s{s}_*.csv and no recorded download failure')
    return not bad, len(secs), '; '.join(bad) if bad else (f'sectors {", ".join(secs)}' if secs else 'no sector observed')


@content_test('tess_lightcurve')
def lc_columns(path, **ctx):
    import numpy as np
    lo, hi = _cfg().followup_tess.BTJD_RANGE_D
    cols, rows = _read_csv(path)
    if 'time' not in cols or 'flux' not in cols:
        return False, len(rows), f'columns {cols[:6]}... lack time/flux'
    t = np.array([float(r['time']) if r['time'] else np.nan for r in rows])
    f = np.array([float(r['flux']) if r['flux'] else np.nan for r in rows])
    bad = []
    if not len(rows):
        bad.append('no rows')
    elif not np.all(np.isfinite(t)) or t.min() < lo or t.max() > hi:
        bad.append(f'time outside BTJD {lo:g}-{hi:g} or not finite')
    elif not np.isfinite(np.nanmedian(f)) or np.nanmedian(f) <= 0:
        bad.append(f'median flux {np.nanmedian(f)}')
    return not bad, len(rows), '; '.join(bad) if bad else (f'{len(rows)} cadences BTJD {t.min():.2f}-{t.max():.2f}, '
                                                           f'median flux {np.nanmedian(f):.4f}')


# ---- dossier --------------------------------------------------------------------------------------------------------

_SPECTRUM_COLS = ['scope', 'freq_mhz', 'is_subband', 'v_source', 'v_peak', 'v_err', 'v_snr', 'v_sign', 'i_peak', 'i_err',
                  'i_snr', 'vi_ratio', 'v_fwhm_maj_deg', 'v_axis_ratio', 'v_ra', 'v_dec', 'v_detected', 'i_detected']
_SNAP_LC_COLS = ['time', 'mjd', 'peak', 'snr', 'sign']                              # io_products.py:143
_SNAP_SPEC_COLS = ['scope', 'freq_mhz', 'dewarped', 'V_peak', 'V_err', 'V_snr', 'V_sign', 'I_peak', 'I_err', 'I_snr',
                   'I_sign', 'vi_ratio']                                             # io_products.py:158-159
# _SPECTRUM_COLS = io_products.py:126-129. tests/test_followup.py writes a dossier through the package writers and checks
# these three lists against what they write.


def _sidecar_ok(path):
    side = path.replace('.json', '.provenance.json') if path.endswith('.json') else path + '.provenance.json'
    if not os.path.exists(side):
        return f'no sidecar {os.path.basename(side)}'
    try:
        pv = json.load(open(side))
    except ValueError as e:
        return f'sidecar unreadable ({e})'
    miss = [k for k in ('gittag', 'inputs', 'params', 'generator') if k not in pv]
    if miss or pv.get('generator') != 'candinv':
        return f'sidecar lacks {miss} or generator {pv.get("generator")!r} != candinv'
    return ''


@content_test('followup_dossier')
def evidence_consistent(path, **ctx):
    import math
    tol = _cfg().followup_investigate.P_ASTRO_TOL
    ev = json.load(open(path))
    cid = os.path.basename(path)[:-len('_evidence.json')]
    bad = []
    if (ev.get('candidate') or {}).get('cand_id') != cid:
        bad.append(f'candidate.cand_id {(ev.get("candidate") or {}).get("cand_id")!r} != {cid!r}')
    pa = ev.get('p_astro') or {}
    p, lg = pa.get('p_astro'), pa.get('logit')
    if not isinstance(p, (int, float)) or not 0 <= p <= 1:
        bad.append(f'p_astro {p!r} not in [0, 1]')
    elif pa.get('ateam_veto'):
        if p != 0:
            bad.append(f'A-team veto with p_astro {p}')
    elif not isinstance(lg, (int, float)) or abs(p - 1 / (1 + math.exp(-lg))) > tol:
        bad.append(f'p_astro {p} != logistic(logit {lg})')
    for k, v in (ev.get('outputs') or {}).items():
        for q in (v if isinstance(v, (list, tuple)) else [v]):
            if k != 'dir' and q and not os.path.exists(q):
                bad.append(f'output {k} missing: {q}')
    side = _sidecar_ok(path)
    if side:
        bad.append(side)
    return not bad, p, '; '.join(bad[:4]) if bad else f'P(astro) {p} (logit {lg}); provenance sidecar ok'


@content_test('followup_table')
def table_columns(path, **ctx):
    b = os.path.basename(path)
    cols, rows = _read_csv(path)
    bad = []
    if b.endswith('_snapshot_spectrum.csv'):
        want = _SNAP_SPEC_COLS
    elif b.endswith('_snapshot_lc.csv'):
        want = _SNAP_LC_COLS
        mjd = [float(r['mjd']) for r in rows if r.get('mjd') not in ('', None)]
        if any(b2 < a for a, b2 in zip(mjd, mjd[1:])):
            bad.append('mjd not increasing')
    elif b.endswith('_spectrum.csv'):
        want = _SPECTRUM_COLS
        ej = path[:-len('_spectrum.csv')] + '_evidence.json'
        if os.path.exists(ej):
            n = len(json.load(open(ej)).get('spectrum') or [])
            if n != len(rows):
                bad.append(f'{len(rows)} rows != {n} spectrum entries in {os.path.basename(ej)}')
    else:
        return False, None, f'unknown follow-up table {b}'
    if cols != want:
        bad.append(f'columns {cols} != {want}')
    side = _sidecar_ok(path)
    if side:
        bad.append(side)
    return not bad, len(rows), '; '.join(bad) if bad else f'{len(rows)} rows, columns and sidecar ok'


@content_test('followup_region')
def region_circles(path, **ctx):
    r_deg = _cfg().followup_investigate.REG_RADIUS_DEG
    lines = open(path).read().splitlines()
    bad = []
    if len(lines) < 2 or lines[1].strip() != 'fk5':
        bad.append('line 2 is not fk5')
    circ = [re.match(r'^circle\(([-+0-9.eE]+),([-+0-9.eE]+),([0-9.]+)d\)', l) for l in lines[2:]]
    if any(c is None for c in circ) or len(circ) < 2:
        bad.append(f'{len(circ)} lines, not all circle(ra,dec,Rd) (>= 2 needed)')
    else:
        for c in circ:
            if abs(float(c.group(3)) - r_deg) > 1e-9:
                bad.append(f'radius {c.group(3)} != {r_deg}')
            if not -90 <= float(c.group(2)) <= 90:
                bad.append(f'dec {c.group(2)}')
    side = _sidecar_ok(path)
    if side:
        bad.append(side)
    return not bad, len(circ), '; '.join(bad[:3]) if bad else f'{len(circ)} circles of {r_deg} deg'


@content_test('followup_fits')
def standard_header_and_pixels(path, **ctx):
    import numpy as np
    from astropy.io import fits
    from astropy.wcs import WCS
    from exopipe.common.validate_fits_headers import validate_header
    half = _cfg().followup_investigate.CUTOUT_HALF_PX
    rep = validate_header(path, level='standard')
    bad = [f'header: {e}' for e in rep.errors[:3]]
    with fits.open(path) as h:
        hdr, data = h[0].header, h[0].data
    d = np.squeeze(data) if data is not None else None
    if d is None or not np.isfinite(d).any():
        bad.append('no finite pixel')
    try:
        if not WCS(hdr).celestial.has_celestial:
            bad.append('no celestial WCS')
    except Exception as e:
        bad.append(f'WCS: {e}')
    if d is not None and path.endswith('_cutout.fits') and (d.ndim != 2 or max(d.shape) > 2 * half + 1):
        bad.append(f'cutout shape {d.shape} (at most {2 * half + 1} px a side)')
    cid = os.path.basename(os.path.dirname(os.path.abspath(path)))
    if 'CANDID' in hdr and str(hdr['CANDID']) != cid:
        bad.append(f'CANDID {hdr["CANDID"]} != {cid}')
    return not bad, None if d is None else list(d.shape), '; '.join(bad) if bad else (
        f'standard header ok; {d.shape}; {int(np.isfinite(d).sum())} finite pixels')


@content_test('followup_summary')
def summary_rows(path, **ctx):
    cols, rows = _read_csv(path)
    bad = []
    if cols != _SUMMARY_COLS:
        bad.append('columns differ from SUMMARY_COLS (attic/run_all_top50.py:27-38)')
    keyed = [float(r['p_astro']) if r['p_astro'] not in ('', None) else -1.0 for r in rows]
    if keyed != sorted(keyed, reverse=True):
        bad.append('rows not sorted by p_astro (descending, missing last)')
    root = os.path.dirname(os.path.abspath(path))
    for r in rows:
        ej = os.path.join(root, r['cand_id'], f'{r["cand_id"]}_evidence.json')
        if r['p_astro'] in ('', None):
            if not r['error']:
                bad.append(f'{r["cand_id"]}: no p_astro and no error')
            continue
        if not os.path.exists(ej):
            bad.append(f'{r["cand_id"]}: p_astro without evidence file')
            continue
        pe = (json.load(open(ej)).get('p_astro') or {}).get('p_astro')
        if pe is None or abs(float(r['p_astro']) - float(pe)) > 1e-9:
            bad.append(f'{r["cand_id"]}: p_astro {r["p_astro"]} != evidence {pe}')
    side = path + '.provenance.json'
    if not os.path.exists(side) or json.load(open(side)).get('n_candidates') != len(rows):
        bad.append('provenance sidecar missing or n_candidates != rows')
    return not bad, len(rows), '; '.join(bad[:4]) if bad else f'{len(rows)} candidates, ranked, scores match evidence'


# ---- honing ---------------------------------------------------------------------------------------------------------

@content_test('honing_table')
def honing_rows(path, cand_id=None, **ctx):
    tol = _cfg().followup_hone.REPRO_TOL
    cols, rows = _read_csv(path)
    bad = [f'missing column {c}' for c in ('cand_id', 'lst', 'date', 'ra', 'dec', 'snr', 'status') if c not in cols]
    for r in rows:
        st = r.get('status')
        if st not in ('honed', 'repro-fail', 'no-products', 'no-box'):
            bad.append(f'{r.get("cand_id")}: status {st!r}')
        elif st == 'honed':
            if not re.match(r'^\d+-\d+$', r.get('box_t', '')) or not re.match(r'^\d+MHz-\d+MHz$', r.get('box_f', '')):
                bad.append(f'{r["cand_id"]}: box_t {r.get("box_t")!r} / box_f {r.get("box_f")!r}')
            if r.get('emp_p') not in ('', None) and not 0 < float(r['emp_p']) <= 1:
                bad.append(f'{r["cand_id"]}: emp_p {r["emp_p"]}')
        elif st == 'repro-fail':
            own, disc = float(r['stack_all_snr']), float(r['snr'])
            if abs(abs(own) - abs(disc)) / max(abs(disc), 1e-9) <= tol and (own > 0) == (disc > 0):
                bad.append(f'{r["cand_id"]}: repro-fail but within tolerance {tol}')
    if cand_id is not None and sum(r.get('cand_id') == cand_id for r in rows) != 1:
        bad.append(f'{cand_id}: not exactly one row')
    by = {}
    for r in rows:
        by[r.get('status')] = by.get(r.get('status'), 0) + 1
    return not bad, len(rows), '; '.join(bad[:4]) if bad else f'{len(rows)} rows {by}'


@content_test('honing_stamp')
def stamp_shape(path, **ctx):
    import numpy as np
    from astropy.io import fits
    n = 2 * _cfg().followup_hone.STAMP_HALF_PX + 1
    d = fits.getdata(path)
    ok = d is not None and d.shape == (n, n) and d.dtype.kind == 'f' and d.dtype.itemsize == 4 \
        and np.isfinite(d[n // 2, n // 2])
    return ok, None if d is None else list(d.shape), f'{None if d is None else d.shape} {None if d is None else d.dtype}' \
        + ('' if ok else f' (want ({n}, {n}) float32 with a finite centre)')


# ---- periodicity ----------------------------------------------------------------------------------------------------

_LC3_KEYS = ('source_id', 'ra', 'dec', 'el_mid', 'mjd', 'night', 'bin', 'path', 'peak', 'ctr', 'max3', 'rms', 'master_jy',
             'fixedpos', 'off_ra_arcsec', 'off_dec_arcsec', 'master_path')        # stage2g2_lightcurves_p3.py:150-154


@content_test('periodicity_lightcurves')
def lc3_structure(path, **ctx):
    import numpy as np
    P = _cfg().followup_periodicity
    m = re.match(r'^lc3_(\d\dh)_(Red|Green|Blue|Full)_([IV])\.npz$', os.path.basename(path))
    if not m:
        return False, None, 'name is not lc3_<LST>_<band>_<S>.npz'
    lst, band, S = m.groups()
    z = np.load(path, allow_pickle=True)
    miss = [k for k in _LC3_KEYS if k not in z.files]
    if miss:
        return False, None, f'missing arrays {miss}'
    nf, npos = len(z['mjd']), len(z['source_id'])
    bad = []
    for k in ('peak', 'rms', 'fixedpos'):
        if z[k].shape != (nf, npos):
            bad.append(f'{k} shape {z[k].shape} != ({nf}, {npos})')
    for k in ('night', 'bin', 'path'):
        if len(z[k]) != nf:
            bad.append(f'{k} length {len(z[k])} != {nf} frames')
    if npos and float(np.nanmin(z['el_mid'])) < P.LC_ELEV_MIN_DEG:
        bad.append(f'a position below {P.LC_ELEV_MIN_DEG} deg at mid-LST')
    pat = re.compile(r'/' + lst + r'/\d{4}-\d{2}-\d{2}/' + re.escape(P.LC_GENERATION) + rf'/{band}/{S}/10min/{band}_{S}_'
                     rf'10min_Taper_Robust\+0\.0_pbcorr_dewarped_aligned_msub_LST{lst}_bin\d+_[^/]*\.fits$')
    badp = [str(p) for p in z['path'] if not pat.search(str(p))]
    if badp:
        bad.append(f'{len(badp)} frame paths off the {P.LC_GENERATION} pattern, e.g. {badp[0][-90:]}')
    return not bad, [nf, npos], '; '.join(bad[:3]) if bad else f'{nf} frames x {npos} positions'


@content_test('periodicity_table')
def periodicity_rows(path, lc_files=None, **ctx):
    import numpy as np
    P = _cfg().followup_periodicity
    m = re.match(r'^periodicity_(Red|Green|Blue|Full)_([IV])\.csv$', os.path.basename(path))
    if not m:
        return False, None, 'name is not periodicity_<band>_<S>.csv'
    S = m.group(2)
    want = list(_script_literal('search', 'cols'))
    cols, rows = _read_csv(path)
    bad = []
    if cols != want:
        bad.append(f'header ({len(cols)} columns) != stage2h4 cols ({len(want)})')
    ids = [r['source_id'] for r in rows]
    if len(set(ids)) != len(ids):
        bad.append(f'{len(ids) - len(set(ids))} duplicate source_id')
    f_ = lambda x: float(x) if x not in ('', None) else None   # noqa: E731
    lo, hi = P.SHORT_PERIOD_D
    llo, lhi = P.LONG_PERIOD_D
    nbad = {'veto': 0, 'blocks': 0, 'short': 0, 'long': 0, 'pct': 0}
    for r in rows:
        if r.get('veto'):
            d = f_(r.get('nearest_ateam_deg'))
            if r.get('dip_P') or d is None or d >= P.ATEAM_AVOID_DEG[S]:
                nbad['veto'] += 1
        nb = f_(r.get('n_blocks'))
        if r.get('dip_P') and nb is not None and nb < P.MIN_BLOCKS:
            nbad['blocks'] += 1
        for k in ('dip_P', 'burst_P'):
            v = f_(r.get(k))
            if v is not None and not lo <= v <= hi:
                nbad['short'] += 1
        for k in ('long_dip_P', 'long_burst_P'):
            v = f_(r.get(k))
            if v is not None and not llo <= v <= lhi:
                nbad['long'] += 1
        for k in cols:
            if k.endswith('_pct') and r.get(k) not in ('', None) and not 0 <= float(r[k]) <= 100:
                nbad['pct'] += 1
    bad += [f'{v} rows violate the {k} rule' for k, v in nbad.items() if v]
    if lc_files:
        want_ids = set()
        for f in lc_files:
            want_ids |= set(np.load(f, allow_pickle=True)['source_id'].tolist())
        if set(ids) != want_ids:
            bad.append(f'{len(set(ids))} sources != {len(want_ids)} distinct source_id of the input light curves')
    nsearch = sum(1 for r in rows if r.get('dip_P'))
    return not bad, len(rows), '; '.join(bad[:4]) if bad else f'{len(rows)} sources, {nsearch} searched'


@content_test('periodicity_merged')
def merged_rows(path, tables=None, **ctx):
    m = re.match(r'^allsky_periodicity_([IV])\.csv$', os.path.basename(path))
    if not m:
        return False, None, 'name is not allsky_periodicity_<S>.csv'
    keys = list(_script_literal('page', 'keys'))
    want = ['source_id', 'ra', 'dec'] + [f'{t}_{k}' for t in ('Red', 'Green', 'Blue') for k in keys]
    cols, rows = _read_csv(path)
    bad = [] if cols == want else [f'header ({len(cols)} columns) != stage2i layout ({len(want)})']
    ids = [r['source_id'] for r in rows]
    if len(set(ids)) != len(ids):
        bad.append('duplicate source_id')
    if tables:
        union = set()
        by = {}
        for t in tables:
            band = os.path.basename(t).split('_')[1]
            by[band] = {r['source_id']: r for r in _read_csv(t)[1]}
            union |= set(by[band])
        if set(ids) != union:
            bad.append(f'{len(set(ids))} sources != union {len(union)} of the band tables')
        else:
            diff = sum(1 for r in rows for band, tb in by.items() for k in keys
                       if r[f'{band}_{k}'] != tb.get(r['source_id'], {}).get(k, ''))
            if diff:
                bad.append(f'{diff} band cells differ from their band table')
    return not bad, len(rows), '; '.join(bad[:3]) if bad else f'{len(rows)} sources x {len(cols)} columns'


@content_test('followup_page')
def page_links(path, merged=None, **ctx):
    from html.parser import HTMLParser
    refs = []

    class _P(HTMLParser):
        def handle_starttag(self, tag, attrs):
            a = dict(attrs)
            if tag == 'img' and a.get('src'):
                refs.append(a['src'])
            if tag == 'a' and a.get('href'):
                refs.append(a['href'])
    text = open(path, errors='replace').read()
    _P().feed(text)
    d = os.path.dirname(os.path.abspath(path))
    local = [r.split('#')[0].split('?')[0] for r in refs if not re.match(r'^(https?:|mailto:|#|javascript:|data:)', r)]
    local = [r for r in local if r]
    missing = [r for r in local if not os.path.exists(r if os.path.isabs(r) else os.path.join(d, r))]
    bad = [f'{len(missing)} of {len(local)} local links missing, e.g. {missing[:2]}'] if missing else []
    if '</html>' not in text.lower():
        bad.append('no closing </html>')
    copy = os.path.join(d, 'allsky_periodicity.csv')
    if merged and os.path.exists(copy) and open(copy, 'rb').read() != open(merged, 'rb').read():
        bad.append('allsky_periodicity.csv differs from the merged table')
    return not bad, len(local), '; '.join(bad) if bad else f'{len(local)} local links resolve'


@content_test('fold_table')
def fold_rows(path, source=None, bands=None, **ctx):
    want = ['band', 'lst', 'night', 'mjd_mid_utc', 'mjd_bary_tdb', 'phase', 'flux_jy', 'err_jy', 'normalised_flux',
            'kept_after_mad']                                                        # steep_fold.py:58
    cols, rows = _read_csv(path)
    bad = [] if cols == want else [f'columns {cols} != {want}']
    if any(not -0.5 <= float(r['phase']) <= 0.5 for r in rows):
        bad.append('phase outside [-0.5, 0.5]')
    if any(r['kept_after_mad'] not in ('0', '1') for r in rows):
        bad.append('kept_after_mad not 0/1')
    if source and bands:
        n = sum(1 for r in _read_csv(source)[1] if r['band'] in bands and r.get('cadence', '10min') == '10min'
                and r['flux_jy'] != '')
        if n != len(rows):
            bad.append(f'{len(rows)} rows != {n} selected rows of {os.path.basename(source)}')
    kept = sum(r['kept_after_mad'] == '1' for r in rows)
    return not bad, len(rows), '; '.join(bad) if bad else f'{len(rows)} blocks, {kept} kept after MAD'


@content_test('followup_figure')
def png_decodes(path, **ctx):
    import struct
    b = open(path, 'rb').read(32)
    size = os.path.getsize(path)
    if b[:8] != b'\x89PNG\r\n\x1a\n' or b[12:16] != b'IHDR':
        return False, size, 'not a PNG (signature / IHDR)'
    w, h = struct.unpack('>II', b[16:24])
    ok = w > 0 and h > 0 and size > 1024
    return ok, size, f'{w} x {h} px, {size} bytes' + ('' if ok else ' (want > 0 px and > 1 kB)')
