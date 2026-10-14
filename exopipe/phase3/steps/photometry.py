"""exopipe3 photometry - photometry of the input samples (Hot/Warm Jupiters, 10 pc) at their catalogue positions, with the
12-control local-empirical null, the a-priori bright-source screen, control-derived thresholds and completeness.

    exopipe3 photometry --tier deep --run P3S5 --parent-dir DIR --out /fast/gh/refactor_scratch/phot/Run_P3S5   (campaign)
    exopipe3 photometry --tier deep --run P3S5 --parent-dir DIR --epochs 04h_2025-10-14 --out ...                (one hour)
    exopipe3 photometry --tier snapshot --stages measure --cells 18h_2025-04-11 --out ... --work /fast/...      (one cell)
    exopipe3 photometry --tier snapshot --stages evaluate --stage1 ROOT --eval-tag 20261013 --out ...          (campaign)

Per hour vs intermittent (DESIGN 12): per hour, --epochs <LST>_<date> measures ONE night of one LST hour - deep: the
frames, stage and measure stages over every band of that LST, restricted to that night (the Stokes V 3x noise gate still
decided over the cell's full night list); snapshot: the measure stage of that night cell (the snapshot-cell chain step after
`search --tier snapshot`). The evaluation (thresholds from the controls, completeness, the >= 4 sigma lists) is a CAMPAIGN
step over a whole measurement (intermittent, DESIGN 12.2): deep `--stages evaluate` on a run directory, snapshot
`--stages evaluate` over the measured cells. Item kinds (core/cells): deep cell 04h_Full (--cells), epoch 04h_2025-10-14
(--epochs), snapshot night cell 18h_2025-04-11[_<band>] (--cells / --epochs). Tier sublst: refused (no production engine).

Deep tier (lib/photometry/steps_deep.py). --out = the RUN directory (default <photometry.OUT_ROOT[deep]>/Run_<label>).
Inputs:  the searched run (--src-run DIR or --run LABEL -> <paths.TRANSIENTS_ROOT>/Run_<LABEL>; required, no default):
         iresid/ vresid/ <cell>/<date>_<S>_redo_resid_<band>.fits; the cell roots of photometry_frames.ROOTS_CSV
         (coadd_final/PB_deep_R075_<band>.fits for Stokes I, PB_deep_<band>.fits for V, weights.csv, the R075 master for the
         parent digest); the parent catalogues (--parent-dir, required: master_R075_<lst>_<band>.gaul.csv + .gen.json); the
         sample lists (photometry.SAMPLE_FILES, sha1-pinned); data_decisions.csv; the solar audit; parent_curve_tables.json.
Outputs: frames_07.csv (photometry_frame_list); stage: frames_09.csv (photometry_frame_list), stage_list.txt
         (photometry_stage_list), node-local copies under --stage-root; out/forced_NNN.csv.gz (photometry_table),
         out/blank_NNN.csv (photometry_blank_null), out/frames_NNN.csv (photometry_admission), NNN = shard; out/eval/
         {exceedance, thresholds, completeness, targets_ge4, targets_above_threshold}.csv + exceedance.png (photometry_eval).
Stages:  frames (deep_forced_frames.py in place, on the host that reads the cell roots), stage (copies the inputs this host
         cannot read; skipped when every input is readable), measure (deep_forced_phot.py, one child per shard owning cells,
         one BLAS thread each), evaluate (deep_forced_eval.py). Units: frames, stage, measure.shard_NN, evaluate.

Snapshot tier (lib/photometry/steps_snapshot.py). --out = the snapshot forced root (default photometry.OUT_ROOT[snapshot]).
Inputs:  the search's photometry cutouts <cutout root>/<sample>/<band>/<object>/<object>_<LST>_<UTC>_<S>.fits (default
         paths.PHASE3_ARCHIVE_ROOT/photometry; class photometry_cutout, its content test reported per band); the gen-4 Master
         catalogues; evaluate: the stage-1 files (--stage1, default <out>/stage1), the search's noise medians and runinfo
         (paths.SNAPSHOT_FLEET_OUT), the curve constants, the tier-3 pin, data_decisions.csv, the refused-cell list.
Outputs: stage1/<cell>/forced_<sample>_<band>_<S>.npz (photometry_snapshot_npz) + forced_manifest.csv (photometry_manifest);
         eval_<tag>/ and eval_wb_<tag>/ (photometry_eval: exceedance, thresholds, thresholds_grid, completeness, targets_ge4,
         refused_cells, cells, drops), refused_cells_<kind>_<tag>.csv, zones_recomputed_<tag>.csv (photometry_zone_flags).
Stages:  measure (snapshot_forced_phot.py per band, --band-procs at once, node-local work dir: --work or the site
         SCRATCH_ROOT), evaluate (snapshot_forced_eval.py --kind sb and wb, then the zone flags). Units: measure.<cell>
         (measure.<cell>.<band+band> for a band subset), evaluate.<kind>.<tag>, evaluate.zones.<tag>.

Records: <out>/_records/photometry/<unit>.json; engine logs <out>/_records/photometry/logs/. No marker files.
--dry-run: prints the plan and writes nothing; exit 2 when a gate would refuse (the plan names it), else 0. A stage whose
  inputs an EARLIER stage of the same invocation writes (stage/measure after frames, evaluate after measure) is planned from
  that stage's outputs, not from records on disk.
Failure modes (exit 2 = refused, the gate named in the record; 3 = nothing to do; 1 = error):
  run level: PHOT-ENV (caller environment sets P3S*/DEEP_FORCED_*/SF_*/FORCED_TEST_*/SUBLST_*), PHOT-SAMPLE-PIN (a sample
  list differs from its pinned sha1), PHOT-SUBLST-NO-ENGINE, PHOT-FULL (snapshot Full), a missing pinned input.
  deep: F-SRCRUN, F-PARENTDIR, a cell root not readable here; F-PARENTS / F-PB / F-NOFRAMES per cell (item refusals, the
  run goes on); F-NOFRAMES for every cell (exit 3); S-STAGE (copy failed or incomplete); M-CHAIN (no done record of the
  previous stage, a frame list changed since its record, a staged list made from another frame list, inputs not readable
  without a stage, shards missing, forced or blank files no shard record lists); F-NOFRAMES vs F-PARENTS/F-PB when the
  frame-list engine wrote no row (exit 3 only when no selected cell has a search frame, else refused); E-ORDER /
  E-POSCOLS / E-RADIO (the evaluation's own refusals); M-NOISE per hour (item refusal).
  snapshot: SM-NOCUT (no cutout: exit 3), SM-DONE (a band refused, a REFUSED manifest row, no ok row; a crash = error),
  SM-SUBSET (a band subset over a full-band manifest), SM-STALE (npz of measured bands not rewritten; --fresh moves them
  to <out>/quarantine_superseded_photometry_<run id>/<cell>/ with a sha256 manifest - never deleted), SE-LISTED, SE-NONE,
  SE-DONE (a cell without a complete stage 1), the evaluation's own refusals, Z (no targets_ge4 rows, or a recomputed
  zone screen differs: the zone-flag file is not written).
  Any failing content test refuses its unit.
"""
import importlib
import sys

SUMMARY = 'photometry of input samples (HWJ, 10 pc; flare/UCD and 25 pc listed, not measured): deep and snapshot'
TIERS = ('deep', 'snapshot', 'sublst')
SELECTORS = ('tier', 'cells', 'scopes', 'stokes', 'run_label', 'exclude_cells', 'epochs')
CONFIG = ('photometry', 'photometry_frames', 'photometry_measure', 'photometry_evaluate', 'photometry_cutouts',
          'photometry_phase2', 'catalog_snapshot', 'runner')
WRITES = True
_TIER_MODULES = {'deep': 'exopipe.phase3.lib.photometry.steps_deep', 'snapshot': 'exopipe.phase3.lib.photometry.steps_snapshot'}


def add_arguments(p):
    g = p.add_argument_group('stages (deep: frames,stage,measure,evaluate; snapshot: measure,evaluate; '
                             'with --epochs the per-hour set photometry.EPOCH_STAGES)')
    g.add_argument('--stages', metavar='LIST', help='stages to run, in tier order whatever the order given')
    g.add_argument('--from', dest='stage_from', metavar='STAGE', help='first stage of the selected list')
    g.add_argument('--until', dest='stage_until', metavar='STAGE', help='last stage of the selected list')
    d = p.add_argument_group('deep tier')
    d.add_argument('--src-run', metavar='DIR', help='the SEARCHED deep run directory (required unless --run LABEL)')
    d.add_argument('--parent-dir', metavar='DIR', help='the Robust -0.75 parent catalogue directory the deep table pass used '
                                                       '(frames stage; required)')
    d.add_argument('--frames', metavar='FILE', help='measure/evaluate from this frame list instead of the frames/stage '
                                                     'record (e.g. a list built by the deep chat; its sha1 is recorded)')
    d.add_argument('--shards', type=int, help='measure: shards (default photometry_measure.SHARDS[deep] = 32; another value '
                                              'changes the blank-sky rows and is recorded as a config override)')
    d.add_argument('--stage-root', metavar='DIR', help='stage: node-local root of the copies (default --work or the site '
                                                       'SCRATCH_ROOT + photometry_frames.STAGE_SUBDIR)')
    s = p.add_argument_group('snapshot tier')
    s.add_argument('--cutout-root', metavar='DIR', help='measure: the photometry cutouts (default '
                                                        'paths.PHASE3_ARCHIVE_ROOT/photometry_measure.CUTOUT_SUBDIR)')
    s.add_argument('--index', metavar='FILE', help='measure: listing of cutout paths relative to --cutout-root')
    s.add_argument('--band-procs', type=int, help='measure: band processes at once in a cell (default '
                                                  'photometry_measure.BAND_PROCS[snapshot] = 8)')
    s.add_argument('--manifest-name', help='measure: name of the merged manifest (default forced_manifest.csv; a band '
                                           'subset needs another name when a full-band manifest exists)')
    s.add_argument('--stage1', metavar='DIR', help='evaluate: the stage-1 root (default <out>/stage1)')
    s.add_argument('--kinds', help='evaluate: sb, wb or sb,wb (default photometry_evaluate.KINDS[snapshot])')
    s.add_argument('--survey-cells', type=int, help='evaluate: survey size for the survey thresholds (default '
                                                    'photometry_evaluate.SURVEY_CELLS[snapshot] = 402; a RULED key: another '
                                                    'value needs --ruling-for or a non-production --out, and is recorded as '
                                                    'a config override)')
    s.add_argument('--refused-cells', action='append', metavar='KIND=FILE',
                   help='evaluate: the per-kind cell,reason list of cells refused before evaluation (e.g. the '
                        'refused_cells_sb_<tag>.csv of a production evaluation, to replay it); it must list every cell '
                        'photometry_evaluate.REFUSED_CELLS refuses for that kind. Default: that pinned file plus the cells '
                        'whose measure unit was refused')
    s.add_argument('--eval-procs', type=int, help='evaluate: cells evaluated at a time (--nproc; outputs identical)')
    s.add_argument('--eval-tag', help='evaluate: output directories eval_<tag>, eval_wb_<tag> (default the UTC date)')


def default_out(args):
    from exopipe.phase3.config import CFG
    tier = getattr(args, 'tier', 'deep')
    if tier == 'deep':
        return importlib.import_module(_TIER_MODULES['deep']).default_out(args)
    if tier == 'snapshot':
        return CFG.photometry.OUT_ROOT['snapshot']
    return '/tmp/exopipe3_photometry_%s' % tier


def run(args, run):
    from exopipe.phase3.core import record
    from exopipe.phase3.lib.photometry import steps_common as K
    K.gate_tier(args.tier)
    K.gate_env()
    K.gate_samples()
    c = K.cfg()
    m = c.photometry_measure
    if args.tier == 'deep':
        K.gate_pinned_inputs([c.photometry_frames.ROOTS_CSV, m.PARENT_TABLES['deep'], m.SOLAR_AUDIT['deep'],
                              c.paths.DATA_DECISIONS])
    else:
        e = c.photometry_evaluate
        K.gate_pinned_inputs([e.REFUSED_CELLS['snapshot'], e.DEEP_SEARCH_PIN['snapshot'], c.paths.DATA_DECISIONS,
                              c.catalog_snapshot.CURVE_CONSTANTS])
    try:
        flag_overrides(args, c)
        stages = K.resolve_stages(args, args.tier)
        K.plan(run, 'photometry --tier %s: stages %s; out %s%s' % (args.tier, ','.join(stages), run.out_root,
                                                                   ' (dry run: nothing written)' if run.dry_run else ''))
        rc = importlib.import_module(_TIER_MODULES[args.tier]).run_tier(run, args, stages)
    except ValueError as e:
        print('error: %s' % e, file=sys.stderr)
        return record.EXIT_ERROR
    if run.dry_run:            # the plan printed; a gate the real run would hit makes the dry run exit 2 as well
        wr = getattr(run, 'photometry_would_refuse', [])
        if wr:
            print('dry run: %d gate(s) would refuse: %s' % (len(wr), '; '.join(wr)[:600]), file=sys.stderr)
            return record.EXIT_REFUSED
        return record.EXIT_DONE
    return rc


def flag_overrides(args, c):
    """--shards and --survey-cells change products (the blank-sky rows; the survey thresholds), so a value other than the
    config one is applied AS a config override (photometry_measure.SHARDS[deep], photometry_evaluate.SURVEY_CELLS[snapshot]):
    recorded in every unit record's config_overrides and config hash, and a RULED key refuses without --ruling-for or a
    non-production --out (config._authority) - never a silent flag."""
    from exopipe.phase3.config import ConfigRefusal
    from exopipe.phase3.core import record
    for flag, dotted, val, cur in (('--shards', 'photometry_measure.SHARDS[deep]', args.shards,
                                    c.photometry_measure.SHARDS['deep']),
                                   ('--survey-cells', 'photometry_evaluate.SURVEY_CELLS[snapshot]', args.survey_cells,
                                    c.photometry_evaluate.SURVEY_CELLS['snapshot'])):
        if val is not None and val < 1:
            raise ValueError('%s %d: must be >= 1' % (flag, val))
        if val is not None and val != cur:
            try:
                c.set(dotted, str(val), source=flag)
            except ConfigRefusal as e:
                raise record.Refusal('%s %s: %s' % (flag, val, e))
