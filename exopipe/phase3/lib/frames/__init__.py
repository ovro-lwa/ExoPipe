"""lib/frames - the code behind `exopipe3 frames` (steps/frames.py is the dispatcher; spec plan/specs/frames.md).

Modules (deep tier: unit frames-deep; sub-LST / snapshot: unit frames-tiers)
-------
deep_step.py     tier deep: frame-only rebuilds (search produces the deep frames in the refactor window); items (--cells
                 bulk re-run after a master rebuild; --epochs <LST>_<date> per hour), one unit per (cell, Stokes[, dates]),
                 stages 'residuals' and 'verify' (and --deep-adopt of production attempts), the dry-run plan.
deep_cells.py    read-only resolution of a deep cell's inputs (co-add root table, masters, LOO masters, stage-1 products,
                 build record, gate records), the cell-level pre-gates and the record-based epoch preview.
deep_engine.py   the in-place engine: one child per unit (core.legacy.run) running run_deep_cell(frames_only=True), its
                 environment from search's config sections (engine_values: the ONE reader), the result mapping and the
                 scan of the unit's attempt directory. Generic in cadence: the optional sub-LST searchframes stage can use
                 UnitSpec(tier='sublst') + run_child + collect.
deep_child.py    the child entry point (python -m exopipe.phase3.lib.frames.deep_child SPEC.json).
deep_seam.py     the proposed frames_only keyword of run_deep_cell (anchored edits; never written into lib/search) and the
                 AST proof that the default call is unchanged (python -m exopipe.phase3.lib.frames.deep_seam --check).
tests.py         content tests: search_frame (frames and pbcorr siblings), noise_map, frame_refusals (pbcorr-missing
                 record), epoch_image (= masters' coadd_epoch). Refusal CSVs use search's class refusal_record.
tier_sublst.py,  tiers sublst and snapshot (unit frames-tiers): add_arguments(p) (tier-unique flags --sublst-/--snap-),
tier_snapshot.py run(args, run), optional default_out(args); their tests in tests_<tier>.py.
"""


def select_stages(args, tier):
    """The stages of this invocation for ``tier`` (deep; the sub-LST and snapshot tiers order their own stages from
    frames_tiers), in the tier's order: --stages (default frames.STAGES[tier]; an optional stage,
    frames.OPTIONAL_STAGES[tier], runs only when named), then --from / --until over that order (steps/frames.py adds the
    flags; cli.py does not have them yet, spec B6)."""
    from exopipe.phase3.config import CFG
    default = list(CFG.frames.STAGES[tier])
    order = default + [s for s in CFG.frames.OPTIONAL_STAGES.get(tier, ()) if s not in default]
    if getattr(args, 'stages', None):
        req = [s.strip() for s in args.stages.split(',') if s.strip()]
        unknown = [s for s in req if s not in order]
        if unknown:
            raise ValueError(f'--stages {unknown}: tier {tier} stages are {", ".join(order)}')
        sel = [s for s in order if s in req]
    else:
        sel = list(default)
    frm, until = getattr(args, 'from_stage', None), getattr(args, 'until_stage', None)
    for flag, s in (('--from', frm), ('--until', until)):
        if s and s not in order:
            raise ValueError(f'{flag} {s}: tier {tier} stages are {", ".join(order)}')
    if frm:
        sel = [s for s in sel if order.index(s) >= order.index(frm)]
    if until:
        sel = [s for s in sel if order.index(s) <= order.index(until)]
    if not sel:
        raise ValueError(f'no stage selected for tier {tier} (stages {", ".join(order)})')
    return sel
