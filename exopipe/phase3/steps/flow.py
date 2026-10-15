"""exopipe3 flow - optical-flow alignment on the GPU (image_plane_correction Brox flow, JAX, site PYTHON['flow']).

The ONLY GPU command (DESIGN 11.2; the one documented exception is the deep x2 epoch flows inside run_deep_cell, which
stay in `search`). Three tiers:

  per hour (DESIGN 12.1; --epochs <LST>_<date> = one LST hour of one night; chains 'epoch', 'snapshot-cell', 'sublst-lst')
    snapshot  every 10-s Stokes I frame -> its subband's multi-night Master (gen-4 I Taper Robust-0); the V frame of the
              same instant takes the I flow (clv15/phase3_snapshot_flow.py in place, one child per subband, subbands
              round-robin over the node's GPUs). Item: night cell <LST>_<date>[_<subband>] (--cells, --epochs).
              Units flow_<cell>_<sb> (stage flow), ship_<cell> (stage ship).
    sublst    second alignment: each night's 10-min window stack, x2 FFT-upsampled -> the x2 leave-one-out window Master
              that is subtracted; the I flow registers V; writes the arm-B residual (clv15/phase3_sublst_refine.py in
              place). Items: --lsts L (every subband), --cells L_<subband>, --cells / --epochs L_<date>[_<subband>] (that
              night's residuals; the leave-one-out Masters still use every night). Units refine_<LST>_<sb>[_<night>],
              check_<LST>_<sb>[_<night>].
  survey / intermittent (DESIGN 12.2; part of a masters build, chain 'deep-build-cell': masters stage-in -> flow -> build)
    deep      each night's flux-scaled I Taper R-0.75 alignment image -> the cell's reference (the previous-generation
              master, native and un-warped, written by `masters --tier deep --stages stage-in`; the fixed-table night
              only without one). Item: deep cell <LST>_<scope> (--cells, --lsts). One unit per cell, named <cell>,
              cumulative over the stages it ran (meta.stages_done).

Stages (--stages a,b / --from S / --until S; always in pipeline order; default config flow.DEFAULT_STAGES[tier]):
  deep      stage-in, prep, solve, check   (stage-in and prep are CPU only; solve needs the GPU; check is also the
                                            content verdict of flows already on disk)
            admit (on demand, alone; GPU)  the flows of the nights an installed cell admits (D-ADMIT): the pairs the done
                                            record of `masters --tier deep --stages admit-prep` lists, into <cell>/admit/flow/
  snapshot  flow, ship                     (the flow movies are `frames --tier snapshot --stages movies`)
  sublst    refine, check                  (check runs only when refine did not run in the same invocation)

Inputs:
  deep      the done record <out>/_records/masters/stage-in_<cell>.json and the reference it wrote in <out>/<cell>/coadd_final/
            (masters owns the reconstruction, _CROSSCHECK decision 6); the working cell's stored flows (masters' working-cell
            rule: paths.COADD_SB_ROOT/<cell> or its ref_override.csv root; --deep-flow-from DIR replaces it); Phase-2
            alignment images <PHASE2_ROOT>/<LST>/<date>/Science_*/<sb>MHz|<band>/I/deep/*Taper*Robust-0.75* (dewarped
            preferred, raw fallback); data_decisions.csv (core.exclusions AND the co-add module - they must agree).
  snapshot  the done record of `frames --tier snapshot --stages stage` for the cell (<cell>_stage; flow_snapshot.REQUIRE_STAGE_RECORD)
            and the staged frames <stage>/<cell>/<sb>/*-image.fits it lists; the subband Masters
            <flow_snapshot.MASTERS_ROOT>/<LST>_<sb>/coadd_final/I_deep_Taper_R0_<sb>.fits.
  sublst    window stacks <paths.SUBLST_ROOT>/<LST>/<LST>_<night>/<sb>/<S>/sublst_*_w<k>.fits (masters sub-LST); frozen night
            weights and the membership-veto pin (protected); arm-A window Masters <SUBLST_ROOT>/<LST>_masters/; the approved
            alignment refusals (protected CSV).

Outputs (no product renames in the window; --out D rebases every product root):
  deep      <out>/<cell>/flow/<date>_scaled_flow.npy [coadd_flow]; flow/<date>_scaled_dewarp.fits [flow_dewarp_image];
            prep/<date>_scaled.fits + _scale.json [flow_prep_scaled; intermediates the masters provisional stage deletes];
            prep/ref_master_prev.fits, prep/ref_master_prev_V.fits [flow_reference: the co-add's snapshot of masters'
            reference]; flow/.reference, flow/.reference_resolved (module state, as today). Default --out: the masters deep
            generation root (the same root as `masters --tier deep`, so the chain finds the stage-in record).
  snapshot  <stage>/<cell>/<sb>/<frame>_flowed.fits [snapshot_flowed_frame; node-local, beside the staged frame];
            <work>/<cell>/flow/flow_<sb>_<ts>.npy, flow_<sb>.csv [snapshot_flow_stats], flow_<sb>.log (node-local), shipped
            to <out>/snapshot_flow/<cell>/ [snapshot_flow_field] (production paths.SNAPSHOT_FLOW_ROOT) and
            <out>/snapshot_fleet_out/<cell>/flow/ (production paths.SNAPSHOT_FLEET_OUT). Default --out paths.PHASE3_ARCHIVE_ROOT.
  sublst    <out>/<LST>_x2/<LST>_<night>/<sb>/<S>/sublst_<LST>_<night>_<sb>_<S>_w<k>_resid_loo.fits [sublst_x2_residual];
            <out>/<LST>_x2/logs/<LST>_<sb>.log (the module's own lines). Default --out paths.SUBLST_ROOT.
  every unit: <out>/_records/flow/<unit>.json (inputs with fingerprints, admitted nights, outputs, content tests, the child
            processes and the sha1 of every legacy module they imported).

Failure modes (exit 2 = refused, the gate named in the record; 3 = nothing to do; 1 = error):
  deep      G-D0 no done masters stage-in record of the cell (masters owns the reference); G-D1 exclusion list missing or
            core.exclusions and the co-add module disagree; G-D2 no Phase-2 input (exit 3); G-D7 stored-flow copy failed;
            G-D8 no reference; G-D9 reference astrometry (A-02, fail closed, in place); G-D11 fewer than 2 target nights;
            G-D14 a target night with a missing or stale flow after solve (mtime, as today, and the recorded input digest);
            G-D16 a flow less than 50 per cent finite; G-D18 a solve with pairs to compute on a host with no GPU id or no
            site GPU lock. Recorded per night: G-D10 wrong grid, G-D12 shape mismatch; G-D13 stale pairs moved aside
            before solve; G-D15 batch exit code.
  snapshot  G-S1 no GPU id (--gpus or site GPU_IDS) or no site GPU lock for one; G-S2 no done frames stage record, one
            made on another node, or a staging that differs from it (.filelist, names_sha1); G-S3 a broken Master link /
            G-S4 a subband without a Master / G-S5 staged frames off their own Master's grid (the whole cell is
            refused); G-S6 per subband, BY NAME: an expected frame (staged minus WSCNVIS = 0) without its
            _flowed.fits - a set comparison, never a count (snapshot defect D12; cf72d51); G-S7 FAILED rows; G-S8 a flowed
            Stokes I frame without its field; G-S9 ship: a destination with different bytes, or a copy whose checksum differs.
  sublst    G-C1 no night-weight record; G-C2 mixed flux scales in a window (in the module); G-C3/G-C4 window and per-night
            rules (refusals BY RULING, the unit still completes); G-C5 residuals of an earlier run present (use --fresh
            --supersede TAG); G-C6 aligner exit code (module); G-C7 expected residuals missing unless exactly the approved
            (night, window) rows; G-C9 a residual without JUPMASK or whose LOOMSTR names no arm-A Master; G-C11 no GPU id
            or no site GPU lock for it (the lock is the child's P3_FLOW_LOCK); G-C12 a stack of an excluded night
            (flow_refine.EXCLUDED_STACK).
  every tier: a drift-guard difference between config and an in-place module; an override that cannot reach a child.
  GPU locks: every GPU child runs under the lock production holds on that GPU, named in the site file (GPU_LOCKS); a
            host/GPU the site names no lock for is refused (the site file's rule), never run lock-less.

Examples:
  exopipe3 flow --tier deep --cells 23h_23MHz --out /lustre/gh/main/exopipe_ops/refactor_20261012/parity/flow --dry-run
  exopipe3 flow --tier deep --cells 23h_23MHz --out D --stages stage-in,prep            # CPU host
  exopipe3 flow --tier deep --cells 23h_23MHz --out D --stages solve,check --gpus 0     # GPU host, same D
  exopipe3 flow --tier snapshot --epochs 18h_2025-04-11 --work /fast/gh/main/phase3 --out D --gpus 0,1
  exopipe3 flow --tier snapshot --cells 18h_2025-04-11_36MHz --snap-frames list.txt --snap-against-shipped --out D --work W
  exopipe3 flow --tier sublst --cells 09h_55MHz --out D --work /fast/gh/x --gpus 0
  exopipe3 flow --tier sublst --epochs 09h_2025-01-03 --out D --work /fast/gh/x --gpus 1      # per hour: one night
  exopipe3 verify sublst_x2_residual /lustre/gh/main/phase3/sublst/09h_x2
"""
import importlib

SUMMARY = 'alignment flows: snapshot frames and sub-LST windows (per hour); per-night co-add flows (masters builds)'
TIERS = ('deep', 'snapshot', 'sublst')
SELECTORS = ('tier', 'cells', 'lsts', 'nights', 'scopes', 'exclude_cells', 'epochs')
CONFIG = ('flow', 'flow_prep', 'flow_snapshot', 'flow_refine', 'masters', 'masters_build', 'masters_stagein', 'runner')
WRITES = True
STAGES = {'deep': ('stage-in', 'prep', 'solve', 'check', 'admit'), 'snapshot': ('flow', 'ship'), 'sublst': ('refine', 'check')}
# deep 'admit' (D-ADMIT slice 3) is in no default: the flows of the nights `masters --stages admit-prep` prepared, alone

_TIER_MODULES = {t: f'exopipe.phase3.lib.flow.steps_{t}' for t in TIERS}


def _tier(t):
    return importlib.import_module(_TIER_MODULES[t])


def add_arguments(p):
    g = p.add_argument_group('flow (every tier)')
    g.add_argument('--stages', help='stages to run, comma list, always in pipeline order (default flow.DEFAULT_STAGES[tier]): '
                                    'deep stage-in,prep,solve,check; snapshot flow,ship; sublst refine,check')
    g.add_argument('--from', dest='from_stage', metavar='S', help='first stage (of --stages or the default list)')
    g.add_argument('--until', dest='until_stage', metavar='S', help='last stage (of --stages or the default list)')
    g.add_argument('--gpus', help="GPU ids this process may use, comma list (default: the site file's GPU_IDS[host]); snapshot "
                                  "assigns subbands round-robin, deep and sublst use the first id")
    g.add_argument('--supersede', metavar='TAG', help='with --fresh: earlier outputs of the selected units are moved to '
                                                      '<out>/_superseded/<TAG>/ with a MANIFEST.csv (never deleted)')
    for t in TIERS:            # tier modules keep heavy imports inside their functions
        _tier(t).add_arguments(p)


def default_out(args):
    return _tier(getattr(args, 'tier', None) or TIERS[0]).default_out(args)


def resolve_stages(args):
    """--stages / --from / --until -> the stages to run, in the tier's pipeline order. ValueError on an unknown stage."""
    from exopipe.phase3.config import CFG
    allowed = STAGES[args.tier]
    want = [s.strip() for s in (args.stages or '').split(',') if s.strip()] or list(CFG.flow.DEFAULT_STAGES[args.tier])
    bad = [s for s in want + [x for x in (args.from_stage, args.until_stage) if x] if s not in allowed]
    if bad:
        raise ValueError(f'unknown stage(s) {bad} for tier {args.tier}; stages in order: {", ".join(allowed)}')
    stages = [s for s in allowed if s in want]
    if args.from_stage or args.until_stage:
        # --from/--until narrow --stages (or the default list); they never add a stage the list leaves out
        lo = allowed.index(args.from_stage) if args.from_stage else 0
        hi = allowed.index(args.until_stage) if args.until_stage else len(allowed) - 1
        stages = [s for s in stages if lo <= allowed.index(s) <= hi]
    if not stages:
        raise ValueError('no stage selected')
    return stages


def run(args, run):
    import sys
    from exopipe.phase3.core.record import Refusal
    from exopipe.phase3.lib.flow import tests  # noqa: F401  (registers the content tests of every flow class)
    if args.supersede and not args.fresh:
        raise Refusal('--supersede TAG needs --fresh (it moves the earlier outputs of the selected units aside)')
    try:
        args.flow_stages = resolve_stages(args)
    except ValueError as e:
        print(f'error: {e}', file=sys.stderr)
        return 1
    print(f'flow --tier {args.tier}: stages {",".join(args.flow_stages)}; out {run.out_root}', file=sys.stderr)
    return _tier(args.tier).run(args, run)
