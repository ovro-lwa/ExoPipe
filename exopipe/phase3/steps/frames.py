"""exopipe3 frames - residuals and search frames: deep per-epoch, sub-LST per-window, snapshot per-frame products.

  per hour (deep, frame-only rebuild of one night in every band):
      exopipe3 frames --tier deep --epochs 07h_2025-10-15 --deep-rebuild --out /data09/gh/frames_test --work /data09/gh/w
  bulk re-run of a cell after a master rebuild (or --nights to rebuild listed nights only):
      exopipe3 frames --tier deep --cells 07h_18MHz --stokes V --deep-rebuild --out ... --work ...
  verify production attempts (read only; records under --out):
      exopipe3 frames --tier deep --stages verify --deep-adopt @attempt_dirs.txt --out /tmp/frames_verify
  plan: add --dry-run (inputs found, pre-gates, predicted epoch outcomes, planned outputs, the engine command)
  sub-LST / snapshot: --tier sublst | snapshot (lib/frames/tier_sublst.py, tier_snapshot.py; flags --sublst-*, --snap-*)

DEEP TIER - who makes the deep frames (refactor chat 2026-10-13, crosscheck group 14): during the refactor window
`search --tier deep` (stage detect) PRODUCES the deep search frames, as production does. `frames --tier deep` is for
FRAME-ONLY REBUILDS (e.g. re-make a cell's frames after a product defect) and is never chained before search: while
frames_residuals.DEEP_PRODUCER is 'search' a residuals unit refuses unless --deep-rebuild states that the run is one. The
engine is lib/search/deep_search.run_deep_cell(..., frames_only=True); the keyword is the search owner's one reviewed
commit, byte-neutral for the default call - until it lands every residuals unit refuses (gate engine-seam;
`python -m exopipe.phase3.lib.frames.deep_seam --check` shows the proposed edit and its AST proof).

Unit of work: one (cell, Stokes) - record <cell>_<S> (bulk), <cell>_<S>_<date> (per hour, item kind 'epoch'
<LST>_<date> = that night in every band), <cell>_<S>_n<k>_<hash> (--nights). Per owed epoch the engine rebuilds the epoch
from the co-add's stage-1 product plus its master, refines it with the x2-upsampled flow against the I master, subtracts
the epoch's own leave-one-out master where it exists (a night outside the master - weights.csv in_master 0 - has none and
subtracts the central master), removes the striation model (8 x beam along RA), writes the frame conformant from birth
with its primary-beam-corrected sibling and archive copies; in bulk mode the cell's sigma-stack noise map.

Inputs (fingerprinted in the unit record; FITS by DATASUM and processing cards): the co-add cell <root>/<lst>_<scope>/
(root from search_inputs.ROOTS_CSV, or --coadd-root): coadd_final/I_deep_Taper_R075 and V_deep_Taper_R0 masters,
loo/<date>_<S>_loo_*.fits, weights.csv, night_image_gate.csv, epoch_gate.csv, stage2_verdict.csv, PB_deep[_R075] maps,
../residuals_provisional.csv, subtracted/<date>_{I,V}_sub_<scope>.fits; data_decisions.csv (core/exclusions; the admitted
night set is recorded); the membership references; the node-local x2 flow cache <work>/<frames_residuals.WORK_SUBDIR>/
<cell>/ (computed by the engine on a GPU host when missing). The engine values (gates, refine, striation, streaming,
minimum epochs) are search's config keys - the effective ones while search produces the deep frames.
Outputs (production layout, no renames; with --out D rebased to D/deep_attempts and D/archive):
  <attempt>/Run_<label>/{iresid,vresid}/<cell>/<date>_<S>_redo_resid_<scope>.fits          [search_frame]
  <attempt>/Run_<label>/{iresid,vresid}/<cell>/<date>_<S>_redo_resid_pbcorr_<scope>.fits   [search_frame (sibling)]
  <archive>/<LST>/<night>/<paths.ARCHIVE_RUN>/<scope>/<S>/deep/<PRODNAME>                    [search_frame (archive copy)]
  <attempt>/transients/Run_<label>/sigstack/<cell>_<S>_sigstack.fits (bulk only)             [noise_map]
  <attempt>/transients/Run_<label>/refusals/<cell>_<gate>_excluded.csv                       [refusal_record (search's)]
  <attempt>/pbcorr_missing_<cell>.csv (a PB map missing)                                     [frame_refusals]
  <attempt>/<cell>.log (the engine's log); records <out>/_records/frames/<unit>.json
  <attempt> = <attempts root>/<cell>_<S>_<UTC>, a fresh directory per unit run.
Content tests (lib/frames/tests.py): search_frame (full header, own time, Stokes, PRODNAME round trip, rms cards from the
pixels, STRIWIN, master used, pixels, archive copy identical, sibling x PB == plain at PB > 0.05), noise_map (floor 1,
clean-sky median 1, cards), frame_refusals (pbcorr-missing rows), epoch_image (= masters' coadd_epoch). Per unit:
frame_count, exclusion_conformance, archive_copies, removed_epochs_have_no_frame, engine_reported_set.

Failure modes (exit 2 = refused, the gate named in the record; 3 = nothing to do; 1 = error):
  deep-producer (no --deep-rebuild while search produces the deep frames); engine-seam (no frames_only keyword yet);
  D-G1 no I master; D-G2 master generation (COADDOP) is not search_inputs.MASTER_GEN; D-G3 membership rulings violated;
  D-G4 no epochs; D-G9/D-G10 night-image / epoch gate record missing or stale; D-G12 search-tier record missing or stale;
  D-G22 x2 flows missing on a host that cannot compute them (or --deep-x2-flows require); any engine refusal (LOO master
  missing or stale for an in-master epoch, header validation, archive naming, ...); an override of an engine literal
  (deep_engine.FRAME_FIXED); no non-production --out while paths.PRODUCTION_WRITES is False (records included).
  Exit 3: fewer epochs than search_detect.MIN_EPOCHS (P3-POLICY-SKIP), a per-hour night not in a band's build or excluded.
  Epochs removed by a whole-epoch gate (night-image, epoch gate, 10x sanity, provenance, beam) are recorded item refusals
  (no frame owed); a listed night with every item refused refuses the unit; epochs the engine drops without a record
  (flow rc 2, residual task; carried defects R1/R2) are recorded item refusals per search.UNACCOUNTED_EPOCHS.

Per hour vs intermittent (DESIGN 12): --epochs <LST>_<date> rebuilds one night against the INSTALLED masters (never
rebuilt here); --cells re-runs every hour of a cell after `masters` + `dewarp` + `install`. In the window neither is a
chain step before `search` (search makes the deep frames); runner.CHAINS['epoch'] must not run frames --tier deep.
"""
import importlib

from exopipe.phase3.lib.frames import tests as _tests          # noqa: F401  registers the content tests (validate.OWNERS)

SUMMARY = 'per-epoch / per-window / per-frame residuals and search frames, pbcorr siblings, noise maps'
TIERS = ('deep', 'sublst', 'snapshot')
SELECTORS = ('tier', 'cells', 'epochs', 'lsts', 'nights', 'scopes', 'stokes', 'exclude_cells', 'run_label')
WRITES = True

_TIER_MODULES = {'sublst': 'exopipe.phase3.lib.frames.tier_sublst',
                 'snapshot': 'exopipe.phase3.lib.frames.tier_snapshot'}
# search's sections the deep frame-only rebuild reads (the effective engine keys while search produces the deep frames)
_SEARCH_SECTIONS = ('search', 'search_inputs', 'search_detect')


def _config_sections():
    """Every frames section (frames, frames_residuals, the tier sections frames_*), search's engine sections, and the
    shared ones the step reads."""
    from exopipe.phase3.config import CFG
    secs = CFG.sections()
    own = tuple(s for s in secs if s == 'frames' or s.startswith('frames_'))
    return own + tuple(s for s in _SEARCH_SECTIONS if s in secs) + ('paths', 'exclusions', 'instrument', 'site', 'runner')


CONFIG = _config_sections()


def _tier_module(tier):
    return importlib.import_module(_TIER_MODULES[tier])


def add_arguments(p):
    from exopipe.phase3.config import CFG
    from exopipe.phase3.lib.frames import deep_step
    st = f'deep: {",".join(CFG.frames.STAGES["deep"])} (frames.STAGES)'
    secs = CFG.sections()
    for t in ('sublst', 'snapshot'):            # the tiers' own sections (unit frames-tiers): frames_<tier>.ORDER
        order = getattr(secs.get(f'frames_{t}'), 'ORDER', None)
        if order:
            st += f'; {t}: {",".join(order)} (frames_{t}.ORDER)'
    g = p.add_argument_group('frames: stages (every tier)')
    g.add_argument('--stages', help=f'stages to run, always in the tier\'s order: {st}')
    g.add_argument('--from', dest='from_stage', metavar='STAGE', help='first stage of the tier\'s order')
    g.add_argument('--until', dest='until_stage', metavar='STAGE', help='last stage of the tier\'s order')
    g = p.add_argument_group('frames: deep (and sub-LST) search frames')
    g.add_argument('--archive-copy', dest='archive_copy', action='store_true', default=None,
                   help='copy each frame and sibling into the archive tree (default search_inputs.ARCHIVE_WRITE[tier]: '
                        'deep on); with --out the archive root is <out>/archive')
    g.add_argument('--no-archive-copy', dest='archive_copy', action='store_false', help='no archive copies')
    g.add_argument('--archive-run', metavar='NAME', help='generation Run directory of the archive copies (default '
                                                          'paths.ARCHIVE_RUN)')
    g.add_argument('--coadd-root', metavar='DIR', help='input co-add root holding <lst>_<scope>/ (default: per cell from '
                                                        'search_inputs.ROOTS_CSV; recorded when given)')
    deep_step.add_arguments(p)
    for tier, modname in _TIER_MODULES.items():
        try:
            mod = importlib.import_module(modname)
        except ImportError as e:
            p.add_argument_group(f'tier {tier}', f'tier module {modname} is not available ({e})')
            continue
        try:
            mod.add_arguments(p)
        except Exception as e:          # a broken tier module must not take the other tiers' --help down
            p.add_argument_group(f'tier {tier}', f'{modname}.add_arguments failed: {type(e).__name__}: {e}')


def default_out(args):
    tier = getattr(args, 'tier', None) or TIERS[0]
    if tier == 'deep':
        from exopipe.phase3.lib.frames import deep_step
        return deep_step.default_out(args)
    try:
        mod = _tier_module(tier)
        if hasattr(mod, 'default_out'):
            return mod.default_out(args)
    except ImportError:
        pass
    from exopipe.phase3.config import CFG
    return CFG.paths.SUBLST_ROOT if tier == 'sublst' else CFG.paths.SNAPSHOT_CUBES_ROOT


def run(args, run):
    import sys
    from exopipe.phase3.core.record import EXIT_ERROR
    if args.tier == 'deep':
        from exopipe.phase3.lib.frames import deep_step, select_stages
        try:
            stages = select_stages(args, 'deep')
            return deep_step.run(args, run, stages)
        except ValueError as e:
            print(f'error: {e}', file=sys.stderr)
            return EXIT_ERROR
    try:
        mod = _tier_module(args.tier)
    except ImportError as e:
        print(f'error: tier {args.tier} is not available: {_TIER_MODULES[args.tier]} ({e})', file=sys.stderr)
        return EXIT_ERROR
    return mod.run(args, run)
