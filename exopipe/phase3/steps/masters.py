"""exopipe3 masters - multi-night co-add masters with every ruled membership gate, proved by content tests (all tiers).

INTERMITTENT step (DESIGN 12.2): masters are rebuilt when the images on disk have grown (status --rebuild-due), never in the
per-hour chain - the per-hour steps (frames, search, ...) read the INSTALLED masters. Tier modules: deep =
lib/masters/deep_stages.py; sub-LST = lib/masters/tier_sublst.py; snapshot = lib/masters/tier_snapshot.py. Stages run in
the tier's fixed order; --stages a,b / --from S / --until S select a contiguous part. A stage refuses without a done record
of the stage it depends on (unless that stage ran earlier in the same invocation).

  exopipe3 masters --tier deep --cells 23h_23MHz --stages stage-in --out /data09/gh/refactor_parity/masters/deep
  exopipe3 flow --tier deep --cells 23h_23MHz --out /data09/gh/refactor_parity/masters/deep        # (the flow step, GPU)
  exopipe3 masters --tier deep --cells 23h_23MHz --out /data09/gh/refactor_parity/masters/deep --work /fast/gh/x [--dry-run]
  exopipe3 masters --tier deep --cells 23h_Green --stages sources --out P --waive masters.sources.install_record --ruling "..."
  exopipe3 run submit --chain deep-build-cell --items @cells.txt --nodes lwacalim09:2   # stage-in -> flow -> build..verify

DEEP TIER (item: deep cell 04h_Full, 13h_41MHz; unit of a run record: <stage>_<cell>; default stages
build,pbmaps,vclean,fluxscale,verify - stage-in runs alone, BEFORE the flow step: chain 'deep-build-cell')
  stage-in   the cell's native (unscaled) un-warped reference written under the canonical names (port of rebuild_cell.sh
             :23, 34-43, 47-120: ref_override_I / pre-dewarp quarantine / superseded archive by CONSTDIG / current master
             divided by its pixel factor), copies in coadd_final_ref/, the carried epoch-gate record. The working cell is
             paths.COADD_SB_ROOT/<cell> (or its ref_override.csv root). --reference previous-master (default) | path:P |
             night:YYYY-MM-DD (the last two need --waive masters.reference --ruling "<Gregg's words, UTC, ledger ID>").
  (flow)     `exopipe3 flow --tier deep` (GPU) copies the stored flows into <out>/<cell>/flow/ and computes the stale ones
             against the stage-in reference; build refuses without <out>/<cell>/flow/ or with a not-done flow record.
  build      provisional + final co-add in place (lib/masters/coadd.py run_provisional_lst_freq, run_final_lst_freq) with
             P3_ARCHIVE_WRITE=0, P3_FORCE_REBUILD=1, P3_REF_MODE=master and every masters_build value applied.
  pbmaps     PB_deep_<scope> (Robust 0 beam), PB_deep_R075_<scope> (survey beam), a pbcorr sibling per variant
             (pbcorr_masters.do_cell). A master-dewarped cell refuses (its maps come from the dewarp install).
  vclean     the cleaned Stokes V master and its sibling (vclean_masters.build_vclean_cell -> search_v.py --clean_only).
  fluxscale  pixels x SHSCALE[LST] (scale table v4) x FLUXCORR[scope] (correction table v2) into <cell>/gen_shflux/
             (shflux.apply) + one row in <out>/shflux_table.csv.
  verify     content tests T1-T8 of the scaled generation alone (replaces REBUILD_CONTENT_OK; waivable one by one:
             --waive master_generation.T1 --ruling "...").
  sources    AFTER dewarp + install: the master source (parent) catalogue on the INSTALLED master (p3s5_parents.one in
             place): <out>/parent_catalogs/master_R075_<cell>.gaul.csv + .gen.json. Refuses without a done install record
             (window: --waive masters.sources.install_record --ruling "...") and when the screen record says the dewarp was
             applied but the installed master carries no MDWARP.
  variants   optional: rebuild non-survey I variants from the generation record (coadd.rebuild_variants_from_record).

Inputs (deep; read only, fingerprinted in the unit record): Phase-2 images /lustre/pipeline/exopipe/phase2/<LST>/<date>/
  Science_*/<scope>/..., their QA thermal-noise CSVs (PB weights); data_decisions.csv (core.exclusions AND the co-add's own
  reader must agree); the working cell paths.COADD_SB_ROOT/<cell> (or its ref_override.csv root) with coadd_final/ and
  epoch_gate.csv; the flows the flow step put in <out>/<cell>/flow/ (and its record); ref_override_I.csv paths; the pre-dewarp quarantine and the superseded archive; pinned tables
  (sha1 in config): solar audit, named_out_of_master.csv, variant_band_medians.json, scale table v4, correction table v2,
  ref_override*.csv; the PB library and residual table; the archived dewarp screen record (sources, T8).
Outputs (deep; production's names and layout, DESIGN 11.1), under --out (default: the site file's MASTERS_GEN_ROOT for
  this host, else masters.DEEP_GEN_ROOT - production roots refuse writes while paths.PRODUCTION_WRITES is False):
  <out>/<cell>/coadd_final_ref/{I_deep_Taper_R075,V_deep_Taper_R0}_<scope>.fits        [reference_master]
  <out>/<cell>/coadd_final/epoch_gate.csv (carried)                                     [csv_table]
  <out>/<cell>/coadd_final/{I_deep_Taper_R075,V_deep_Taper_R0,I_deep_Taper_R0,I_deep_NoTaper_R0,I_deep_NoTaper_R075}_<scope>.fits
                                                                                        [master]
  <out>/<cell>/coadd_final/loo/<date>_{I_loo_Taper_R075,V_loo_Taper_R0}_<scope>.fits     [loo_master]
  <out>/<cell>/coadd_final/weights.csv, <out>/<cell>/residuals_provisional.csv          [weights_table]
  <out>/<cell>/coadd_final/stage2_verdict.csv (+ night_image_gate.csv)                  [membership_record]
  <out>/<cell>/subtracted/<date>_{I,V}_{epoch,sub,clean}_<scope>.fits                   [coadd_epoch]
  <out>/<cell>/coadd_final/PB_deep_<scope>.fits, PB_deep_R075_<scope>.fits              [pb_map]
  <out>/<cell>/coadd_final/<variant>_pbcorr_<scope>.fits                                [pbcorr_sibling]
  <out>/<cell>/coadd_final/V_deep_Taper_R0_clean_<scope>.fits                           [vclean_master]
  <out>/<cell>/gen_shflux/ (every product x the flux scale, '_shflux' token)            [master_generation: T1-T8]
  <out>/shflux_table.csv (one row per cell; appended)
  <out>/parent_catalogs/master_R075_<cell>.gaul.csv (+ .gen.json)                       [master_catalog]
  run records <out>/_records/masters/<stage>_<cell>.json; in-place logs <out>/_records/masters/logs/<stage>_<cell>.log
  Not written: the archive (P3_ARCHIVE_WRITE=0), the working tree, markers (.stage_out, REBUILD_CONTENT_OK, promoted links).

SUB-LST TIER (lib/masters/tier_sublst.py): prep, admit, stack, loo, sources - 10-minute window masters (the wideband
  windows are the frames step's), see `exopipe3 config --section masters_sublst`. SNAPSHOT TIER (lib/masters/tier_snapshot.py): verify-frozen (the frozen
  generation-4 masters, their link tree and the model-beam PB library), freeze (plan only in the window).

Failure modes (exit 2 = refused, the gate named in the record; 3 = no input; 1 = error):
  G-M01 exclusion list missing/unreadable or the two exclusion readers disagree; every night excluded -> exit 3.
  G-M02 a master/variant/LOO holds an excluded constituent.  G-M03 off-grid nights are dropped (refusals_grid.csv).
  G-M04 reference fails the VLSSr gate / none resolvable / reference night not a member.
  G-M05 no <out>/<cell>/flow/, a not-done flow-step record, a non-reference night without a fresh flow, a flow > 50 per
        cent NaN.
  G-M09 final stage without the ruled stage-2 record.  G-M10..G-M14 per-night membership gates (recorded, not refusals).
  G-M15 fewer than 3 good nights (survey I/V).  G-M16 membership.verify violation.  G-M19 epoch header refusal (R1).
  G-M21 master-dewarped cell / corrupt variant / PB error in pbmaps.  G-M22 vclean status != OK.
  G-M23 no scale or correction row, table sha1 differs, product already scaled (D4).  G-M24 a T1-T8 test fails unwaived.
  G-M25 earlier outputs present without --fresh --supersede TAG; override master inconsistent; pixel factor outside 0.3-3;
        not exactly one superseded master; pinned table sha1 differs.  G-M26 sources: no install record / sidecar md5 /
        0 rows / dewarp screen applied but no MDWARP.  G-M27 < 40 GB free on the --out filesystem.
  Window: --set on a key in a section's PROTECTED_KEYS (pinned inputs, ruling-fixed values, and every MIRROR of a literal
  inside the in-place code) refuses (exit 2, config.py). --epochs / --nights refuse for the deep tier (exit 2).
  Inherited P3_* environment variables are removed before the in-place modules are imported (recorded in the build note).
"""
import importlib
import sys

SUMMARY = 'co-add masters + LOO + QA gates; V clean; flux scale; PB maps; source catalogue (all tiers)'
TIERS = ('deep', 'sublst', 'snapshot')
# 'epochs' is offered for the sub-LST tier's one-night stacks (tier_sublst reads args.epochs); the deep tier refuses it
SELECTORS = ('tier', 'cells', 'lsts', 'nights', 'scopes', 'stokes', 'exclude_cells', 'epochs')
WRITES = True
TIER_MODULES = {'deep': 'exopipe.phase3.lib.masters.deep_stages', 'sublst': 'exopipe.phase3.lib.masters.tier_sublst',
                'snapshot': 'exopipe.phase3.lib.masters.tier_snapshot'}
_WANT_CONFIG = ('masters', 'masters_stagein', 'masters_build', 'masters_pbmaps', 'masters_vclean', 'masters_fluxscale',
                'masters_verify', 'masters_sources', 'masters_sublst', 'masters_snapshot', 'paths', 'exclusions',
                'instrument', 'runner', 'site')


def _declared_config():
    """CONFIG = the sections the step reads (spec 3), limited to those that exist: during the refactor window the tier
    sections masters_sublst / masters_snapshot come from the tier agents' config drafts and may not be loaded yet (an
    unknown section in CONFIG would make every masters command exit 1)."""
    try:
        from exopipe.phase3.config import CFG
        have = CFG.sections()
    except Exception:
        return _WANT_CONFIG
    return tuple(s for s in _WANT_CONFIG if s in have)


CONFIG = _declared_config()
_TIER_IMPORT_ERRORS = {}


def tier_module(tier):
    """The tier module (lazily imported), or None with the import error kept in _TIER_IMPORT_ERRORS."""
    name = TIER_MODULES[tier]
    try:
        return importlib.import_module(name)
    except Exception as e:                 # a tier module not written yet (window) or broken: the others still work
        _TIER_IMPORT_ERRORS[tier] = f'{type(e).__name__}: {e}'
        return None


def add_arguments(p):
    g = p.add_argument_group('stages (every tier)')
    g.add_argument('--stages', help='comma list of stages to run, in the tier\'s fixed order (default masters.DEFAULT_STAGES'
                                    '[tier]; deep: build,pbmaps,vclean,fluxscale,verify; stage-in alone before the flow step; '
                                    'after dewarp + install: sources; optional: variants)')
    g.add_argument('--from', dest='from_stage', metavar='STAGE', help='first stage of a contiguous slice of the tier order')
    g.add_argument('--until', dest='until_stage', metavar='STAGE', help='last stage of a contiguous slice of the tier order')
    g.add_argument('--reference', default='previous-master',
                   help="deep stage-in: alignment/flux reference - previous-master (default), path:P or night:YYYY-MM-DD "
                        "(path/night need --waive masters.reference --ruling \"...\")")
    g.add_argument('--supersede', metavar='TAG',
                   help='with --fresh: move a unit\'s earlier outputs aside under <out>/_superseded/<TAG>/ with a sha1 '
                        'manifest instead of refusing')
    for tier in TIERS:
        mod = tier_module(tier)
        if mod is not None and hasattr(mod, 'add_arguments'):
            mod.add_arguments(p)
    if _TIER_IMPORT_ERRORS:
        p.epilog = ('tier modules not available: ' + '; '.join(f'{t} ({e})' for t, e in _TIER_IMPORT_ERRORS.items()))


def default_out(args):
    mod = tier_module(getattr(args, 'tier', None) or TIERS[0])
    if mod is not None and hasattr(mod, 'default_out'):
        return mod.default_out(args)
    return importlib.import_module(TIER_MODULES['deep']).default_out(args)


def resolve_stages(args):
    """--stages / --from / --until -> the tuple of stages to run, in the tier's fixed order (masters.STAGES[tier]).
    An unknown stage name is an error (exit 1)."""
    from exopipe.phase3.config import CFG
    tier = args.tier
    order = tuple(CFG.masters.STAGES[tier])
    want = [s.strip() for s in (args.stages or '').split(',') if s.strip()] or list(CFG.masters.DEFAULT_STAGES[tier])
    frm, until = args.from_stage, args.until_stage
    bad = [s for s in want + [x for x in (frm, until) if x] if s not in order]
    if bad:
        raise SystemExit(f'masters --tier {tier}: unknown stage(s) {bad}; stages in order: {", ".join(order)}')
    if frm or until:            # spec 2: a contiguous slice of the tier order; defaults = first / last of --stages
        lo = order.index(frm) if frm else min(order.index(s) for s in want)
        hi = order.index(until) if until else max(order.index(s) for s in want)
        want = list(order[lo:hi + 1])
    sel = tuple(s for s in order if s in want)
    if not sel:
        raise SystemExit(f'masters --tier {tier}: no stage selected')
    return sel


def run(args, run):
    """Resolve the stages into args.stages_resolved, then hand the run to the tier module's run(args, run)."""
    tier = args.tier
    mod = tier_module(tier)
    if mod is None:
        print(f'error: masters --tier {tier}: tier module {TIER_MODULES[tier]} not available '
              f'({_TIER_IMPORT_ERRORS.get(tier)})', file=sys.stderr)
        return 1
    args.stages_resolved = resolve_stages(args)
    if args.supersede and not args.fresh:
        raise SystemExit('--supersede TAG needs --fresh')
    return mod.run(args, run)
