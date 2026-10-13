"""exopipe3 dewarp - the final master dewarp of the deep tier, per LST (survey step, intermittent; brief B6, contract step 21).

On the Stokes I Taper Robust -0.75 survey masters of the 19 scopes of one LST (15 subbands + Red, Green, Blue, Full) it
measures a Phase-2 RBF multiquadric warp screen per master: the 73 MHz master anchored to VLSSr (fallbacks 78, 69, 64 MHz),
a cascade outward subband by subband, then each wideband master from the nearest accepted subband
(R-MASTER-DEWARP-RBF-MULTIQUADRIC). A routed cell is ACCEPTED when its final table row is OK - (a) the VLSSr peak median
improved - and (b') median anchor offset - median 5-fold held-out residual >= 5.0 arcsec
(R-MASTER-DEWARP-ACCEPTANCE-A-AND-B-PRIME). An accepted cell gets the SAVED screen applied by cubic-spline resampling to every
FITS product of its generation, pbcorr siblings recomputed (R-MDEWARP-CUBIC-REAPPLY), then content test D1-D6. A routed cell
that is not accepted gets an applied=0 row in the archived screen record - the ONE source of truth for "dewarped cell"
(DESIGN 8a) - and goes to the install un-dewarped. A master already carrying MDWARP is a cascade reference, never re-dewarped.

Use: INTERMITTENT (DESIGN 12.2), after `masters` rebuilt a routed cell and before `install`:
    masters (build..pbmaps) -> dewarp -> install -> masters --stages sources -> frames -> search ...
The per-hour chain never runs it (it reads the installed masters). Runner item = the LST: one measurement per LST is shared by
every routed cell of that LST, so two cells of one LST are never measured in two processes.

    exopipe3 dewarp --lsts 23h --threads 1 --out /data09/parity/dw --work /fast/gh/refactor_scratch      # every stage
    exopipe3 dewarp --cells 23h_23MHz --threads 1 --dry-run                                           # the plan only
    exopipe3 dewarp --lsts 12h --stages stage,measure --threads 1 --out O --work W                    # measurement only
    exopipe3 dewarp --cells 23h_23MHz --stages apply,verify,record --measurement M --out O            # apply a measurement
    exopipe3 dewarp --cells 11h_46MHz --stages verify --measurement M --out O \\
        --waive dewarped_master.D6_product:<file> --ruling "<Gregg's words, UTC, ledger ID>"

Selection: --cells (deep cells <LST>_<scope>; grouped by LST) or --lsts (every ROUTED PROMOTED cell of the LST: routed by
dewarp.ROUTE_SOURCE - 'screen_record' (production since 2026-10-13, chain_cell.sh): the cell's latest row of the archived
screen record says applied=1; 'accepted_list': the 2026-10-07 round's accepted_rebuilt.csv - with a promoted rebuild
<REBUILD_ROOT>/<cell>/coadd_final and <REBUILD_CHECK_ROOT>/<cell>/REBUILD_CONTENT_OK). Neither = every LST with a routed
promoted cell. With --stages stage,measure an LST is measured whether or not it has a routed cell. A re-rooted run (--out)
routes from ITS copy of the record (<out>/archive/...: the parity harness seeds it with the production record).
Production's equivalent of `dewarp_lst.sh <LST>` (which always removes the previous scratch generation, dewarp_lst.sh:57)
is `--lsts <LST> --fresh`: without --fresh an existing <GEN_ROOT>/<cell> refuses (G-A3).

Stages (--stages, pipeline order; default all):
  stage    copy I_deep_Taper_R075_<s>.fits + its pbcorr sibling of the 19 scopes (first of dewarp.REBUILD_ROOT,
           paths.COADD_SB_ROOT holding the plain file, per file) to node-local scratch (--work or the site SCRATCH_ROOT),
           md5-verified; part of measure
  measure  phase3_master_dewarp.run_lst in place on the staged copies, into a fresh <measure root>/<run_id>/
  apply    accepted routed cell: phase3_master_dewarp.apply_saved in place -> <GEN_ROOT>/<cell>/gen_mdewarp_rbf
  verify   accepted routed cell: content test D1-D6 (check_mdewarp_gen.py in place through core.legacy.run; D7 with
           --prior-gen-root) + the install's cubic, applied-screen and product gates; pass = the cell record's
           meta.install {kind mdewarp, gen}
  record   routed cell NOT accepted: applied=0 row in mdewarp_screens.csv; meta.install {kind rebuild, gen <promoted>}

Inputs:  the 19 x 2 measured masters of the LST (dewarp.REBUILD_ROOT before the working tree paths.COADD_SB_ROOT, per
         file); the generation of each routed accepted cell (<REBUILD_ROOT>/<cell>/{coadd_final/**,subtracted/} when that
         holds coadd_final, else the working tree's); <REBUILD_CHECK_ROOT>/<cell>/REBUILD_CONTENT_OK; the screen record
         (routing; the cell's row is kept in its record) or the routing list (paths.OPS_ROOT/dewarp.ROUTE_LIST); VLSSr
         catalogue + peak list (pinned by sha1); data_decisions.csv through core/exclusions; --measurement DIR
         (DIR/tables/table_<LST>.csv, DIR/<LST>/maps/<scope>_warp_screen.fits); --prior-gen-root DIR.
Outputs (product classes):
         dewarp_table          <measure dir>/tables/table_<LST>.csv (24 columns, one row per attempt; the final row decides)
         dewarp_screen         <measure dir>/<LST>/maps/<scope>_warp_screen.fits (SCREEN_X, SCREEN_Y; one per scope reaching
                               the screen fit). Diagnostics beside them (PNG, JSON) are listed in the record, not products.
         dewarped_master       <GEN_ROOT>/<cell>/gen_mdewarp_rbf/ (coadd_final incl. loo/, PB maps, recomputed pbcorr
                               siblings, subtracted/; '_mdewarp' token; MDWARP/MDWMETH/MDWINTP/... cards) with
                               <GEN_ROOT>/<cell>/INSTALL_CONTENT_OK and APPLIED_SCREEN.json (written by D1-D6 on success)
         screen_table          <archive>/mdewarp_screens/<paths.ARCHIVE_RUN>/mdewarp_screens.csv - the archived screen
                               record, install's class (crosscheck 4.2: one class, owned by install); this step writes
                               applied=0 rows only (the applied=1 row and the published screen are the install's) and
                               checks its own cell's row inline ('dewarp.record_row')
         run records           <out>/_records/dewarp/<LST>.json, <cell>.json, _runs/<run_id>.json
         Roots: production (no --out) measure dir <MEASURE_ROOT>/<run_id>, GEN_ROOT, paths.PHASE3_ARCHIVE_ROOT, records under
         GEN_ROOT; --out O re-roots them to O/measure/<run_id>, O/gen, O/archive (a parity harness seeds O/archive with a copy
         of the production mdewarp_screens.csv), records under O.
Failure modes (exit 2 = refused, with the gate named; 3 = no routed promoted cell anywhere; 1 = error):
  G-W   a write root under paths.PRODUCTION_ROOTS while paths.PRODUCTION_WRITES is False (or STAGE_INPUTS False on a
        production tree)          G-D0 a section value differing from the in-place constant (in-place keys are in
        dewarp.PROTECTED_KEYS: config refuses their override), an exopipe module loaded from outside this branch
  G-D0b the production modules check_mdewarp_gen's child imports (phase3_master_dewarp, pb_apply, cascade_dewarp, ...)
        differ from the branch modules the step applies with (only the moved pb_apply import line may differ); after the
        run, the child's code fingerprint differs from what was checked     G-D1 an input root missing on this host, or
        an input / write root node-local to another host (site NODE_LOCAL_ROOTS)
  G-D2  a requested cell not routed / not promoted / without REBUILD_CONTENT_OK   G-D3 VLSSr file sha1 differs from config
  G-D4  VLSSr catalogue fails to load      G-D5 a measured master holds a night data_decisions.csv excludes
  G-D6  a staged copy differs (md5)        G-D7 a routed cell's input master already carries MDWARP
  G-D8  measure without single-thread BLAS (launch with --threads 1; threadpoolctl caps the loaded pools)
  G-A2  no saved screen / NANCHOR differs  G-A3 the generation exists (use --fresh: removes <GEN_ROOT>/<cell>, recorded)
  G-A4/A5 product shape, MDWARP, sibling recompute (raised by the in-place apply)   G-R2 an ACCEPTED cell whose apply refused
  G-A6..A10 content tests dewarped_master.D1_D6 / cubic / applied_screen / products / D4_tested_pixels fail
  G-A11 (--measurement only) the cell's final row predates its rebuild's T1-T8 stamp, or says ALREADY DEWARPED while the
        rebuilt input carries no MDWARP (the measurement is not of this input)
  G-R1  archive record directory not writable or paths.ARCHIVE_RUN empty.  A refusal of the LST unit refuses its cells ('measurement unit <LST> not done').
  Per-scope measurement outcomes (REFUSED ..., REFERENCE ONLY, NOT ATTEMPTED) are table rows, not refusals.
Waivers: --waive dewarped_master.D6_product:<file> --ruling "<words, UTC, ledger ID>" (forwarded to check_mdewarp_gen as a
named per-product D6 waiver; recorded as WAIVED). Every tunable: `exopipe3 config --section dewarp`.
"""
import os
import sys

SUMMARY = 'master dewarp per LST: RBF screen on the rebuilt masters, cubic apply, content test D1-D6'
TIERS = ('deep',)
SELECTORS = ('tier', 'cells', 'lsts')
CONFIG = ('dewarp', 'paths', 'exclusions', 'instrument', 'runner', 'install')   # install: its screen_table test reads it
WRITES = True
STAGES = ('stage', 'measure', 'apply', 'verify', 'record')

# validate.OWNERS maps dewarped_master and dewarp_screen to this module: importing the (light: stdlib + validate) test module
# registers the classes this step owns. The screen record is install's class screen_table (crosscheck 4.2); its tests live
# in lib/install/tests.py, imported here too (light) so the class is registered in this process.
from exopipe.phase3.lib.dewarp import tests as _tests  # noqa: E402,F401
try:
    from exopipe.phase3.lib.install import tests as _install_tests  # noqa: E402,F401
    INSTALL_TESTS_ERROR = None
except Exception as _e:          # the record stage then fails loudly (a class without a registered test raises)
    INSTALL_TESTS_ERROR = f'{type(_e).__name__}: {_e}'


def add_arguments(p):
    p.add_argument('--stages', default=None, metavar='LIST',
                   help='comma list of stage,measure,apply,verify,record (pipeline order; default all). '
                        "'stage' runs with 'measure' only")
    p.add_argument('--measurement', metavar='DIR', default=None,
                   help='use an existing measurement instead of measuring: DIR/tables/table_<LST>.csv and '
                        'DIR/<LST>/maps/<scope>_warp_screen.fits (replaces phase3_master_dewarp --apply-saved)')
    p.add_argument('--prior-gen-root', metavar='DIR', default=None,
                   help='an EXACT re-apply of an earlier measurement: D7 and the two-sided D5 against '
                        'DIR/<cell>/gen_mdewarp_rbf')


def default_out(args):
    """Production record root: the generation root (records under <GEN_ROOT>/_records/dewarp/)."""
    from exopipe.phase3.config import CFG
    return CFG.dewarp.GEN_ROOT


def _done_cell_record(run, cell):
    """Under --resume (not --fresh): this step's own done record of the cell, else None. A cell this step already
    finished may no longer be routed (its applied=0 row, written by the record stage, ends its routing under
    ROUTE_SOURCE 'screen_record' - as dewarp_lst.sh:94 removes the promoted marker); --resume must still skip it."""
    from exopipe.phase3.core import record
    if run is None or not run.resume or run.fresh:
        return None
    rec = record.load_record(run.out_root, 'dewarp', cell)
    return rec if rec and rec.get('status') == 'done' and (rec.get('meta') or {}).get('install') else None


def _plan(args, roots, stages, run=None):
    """{lst: [cells that pass G-D2/G-D7]}, {cell: refusal reason}, routing source path, {cell: G-D2 reason} of the
    planned cells that are no longer routed but have a done record of this step (--resume only: skipped when the
    record is current, refused with the G-D2 reason otherwise)."""
    from exopipe.phase3.core import cells as _cells
    from exopipe.phase3.lib.dewarp import steps_common as SC
    measure_only = not any(s in SC.CELL_STAGES for s in stages)
    cell_list = _cells.expand_cells(args.cells) if args.cells else None
    lsts = None
    if cell_list is not None:
        lsts = sorted({c.split('_', 1)[0] for c in cell_list})
    elif args.lsts:
        lsts = [_cells.parse_lst(x) for x in _cells.expand(args.lsts, universe=_cells.LSTS, kind='LST')]
    if measure_only:
        if lsts is None:
            raise ValueError('--stages stage,measure needs --lsts or --cells')
        return {l: [] for l in lsts}, {}, None, {}
    routed, src = SC.routed_scopes(roots)
    plan, refused, resume_only = {}, {}, {}
    if cell_list is not None:
        for cell in cell_list:
            lst = cell.split('_', 1)[0]
            plan.setdefault(lst, [])
            why = SC.cell_gate(roots, cell, routed)
            if why and why.startswith('G-D2') and 'not routed' in why and _done_cell_record(run, cell):
                resume_only[cell] = why
                plan[lst].append(cell)
            elif why:
                refused[cell] = why
            else:
                plan[lst].append(cell)
    else:
        for lst in (lsts or _cells.LSTS):
            cands = SC.pending_cells(roots, lst, routed)
            if lsts is None and not cands:
                continue
            plan[lst] = []
            for cell in cands:
                why = SC.cell_gate(roots, cell, routed)
                if why:
                    refused[cell] = why
                else:
                    plan[lst].append(cell)
    return plan, refused, src, resume_only


def _mb(p):
    try:
        return f'{os.path.getsize(p) / 1e6:.1f} MB'
    except OSError:
        return 'missing'


def _dry_run(args, run, roots, stages, gate_bad, gw, staging, plan, refused, route_src):
    from exopipe.phase3.lib.dewarp import steps_common as SC, steps_measure as SM
    from exopipe.phase3.lib.dewarp import master_dewarp as MD
    c, CFG = SC.C(), SC.cfg()
    pr = print
    would = []          # gates that would refuse at run time (G-W is reported, not counted: dry runs omit --out)
    pr('exopipe3 dewarp - PLAN (dry run: nothing is written)')
    pr(f'  stages: {",".join(stages)}   tier deep   threads {",".join(f"{v}={os.environ.get(v)}" for v in SC.THREAD_VARS)}')
    pr(f'  roots:  working {roots.working}\n          rebuild {roots.rebuild}\n          rebuild check {roots.check}')
    pr(f'          generation {roots.gen}\n          measure dir {roots.measure_dir if "measure" in stages else args.measurement}')
    pr(f'          screen record {roots.record_csv}\n          records {roots.records}')
    pr(f'          staging {staging or "-"}   (re-rooted under --out: {roots.rerooted})')
    pr('  gates (whole run):')
    pr(f'    G-D0/G-D0b/G-D1: ' + ('WOULD REFUSE - ' + '; '.join(gate_bad) if gate_bad else
                                   'pass (in-place values equal the section; in-place keys protected; the content-test '
                                   'child\'s production modules equal the branch\'s; input roots present)'))
    would += gate_bad
    pr(f'    G-W: ' + ('WOULD REFUSE - ' + '; '.join(gw) if gw else 'pass (no write root under a production root)'))
    if staging and staging.startswith('(refuses'):       # the measure stage needs node-local scratch (--work / site)
        would.append(f'staging: {staging[1:-1]}')
        pr(f'    staging: WOULD REFUSE - {staging[1:-1]}')
    if 'measure' in stages:
        g3 = SC.gate_pinned_inputs()
        pr(f'    G-D3: ' + ('WOULD REFUSE - ' + '; '.join(g3) if g3 else 'pass (VLSSr peak list and catalogue sha1 equal '
                                                                      'the pins)'))
        g8 = SC.gate_threads()
        pr(f'    G-D8: ' + ('WOULD REFUSE - ' + '; '.join(g8) if g8 else 'pass'))
        would += g3 + g8
    if route_src:
        pr(f'  routing: dewarp.ROUTE_SOURCE {c.ROUTE_SOURCE} ({route_src}, sha1 {SC.sha1(route_src)[:12]})')
    if not plan:
        pr('  nothing to do: no routed promoted cell in any LST (exit 3)')
        return 3
    for lst in sorted(plan):
        good = plan[lst]
        bad = sorted(x for x in refused if x.startswith(lst + '_'))
        pr(f'\n  LST {lst}: cells {good or "-"}' + (f'; refused {len(bad)}' if bad else ''))
        for x in bad:
            pr(f'    REFUSED {x}: {refused[x]}')
            would.append(refused[x])
        meas = None
        if 'measure' in stages and (good or not any(s in SC.CELL_STAGES for s in stages)):
            srcs = SM.sources(roots, lst)
            pr(f'    measure inputs ({sum(1 for _, d, _ in srcs if d)} of {len(srcs)} scopes):')
            for s, d, files in srcs:
                if not d:
                    pr(f'      {s:6s} - no master in either root (table row REFUSED no generation root)')
                    continue
                lab = 'REBUILD' if d.startswith(roots.rebuild.rstrip('/') + '/') else 'WORKING'
                md = ' MDWARP (cascade reference)' if SC.carries_mdwarp(files[0]) else ''
                pr(f'      {s:6s} {lab:7s} {d}  plain {_mb(files[0])}, pbcorr {_mb(files[1])}{md}')
            adm, gbad, by, exfp = SM.admitted(lst, srcs)
            would += gbad
            pr(f'    G-D5: ' + ('WOULD REFUSE - ' + '; '.join(gbad) if gbad else
                               f'pass ({sum(by.values())} constituent nights over {len(by)} masters; exclusions '
                               f'{exfp["sha1"][:12]}, {exfp["n_exclude_rows"]} rows)'))
            pr(f'    outputs: {roots.measure_dir}/tables/table_{lst}.csv (dewarp_table); '
               f'{roots.measure_dir}/{lst}/maps/<scope>_warp_screen.fits (dewarp_screen)')
        elif args.measurement:
            meas = SC.Measurement(args.measurement, lst)
            pr(f'    measurement: {meas.table} ({"present" if meas.exists() else "MISSING - cells refused"})')
            if meas.exists() and good:     # the cell units refuse a table failing its own content test (incomplete, ...)
                from exopipe.phase3.lib.dewarp import tests as _T
                ok, v, d = _T.table_rows(meas.table, lst=lst)
                pr(f'    dewarp_table.rows: ' + (f'pass ({v})' if ok else f'WOULD REFUSE every cell - {d}'))
                if not ok:
                    would.append(f'--measurement table {meas.table} fails dewarp_table.rows: {d}')
        for cell in good:
            scope = cell.split('_', 1)[1]
            gen = roots.gen_of(cell)
            groot = MD._gen_root(roots.working, roots.rebuild, lst, scope)
            nsrc = sum(len([f for f in fs if f.endswith('.fits')]) for sub in ('coadd_final', 'subtracted')
                       for _, _, fs in os.walk(os.path.join(groot or '/nonexistent', sub)))
            exists = os.path.isdir(gen) and bool(os.listdir(gen))
            pr(f'    cell {cell}: apply source {groot} ({nsrc} FITS); REBUILD_CONTENT_OK '
               f'"{(open(roots.rebuild_ok(cell)).read().splitlines()[1:] or [""])[0][:80]}"')
            if meas is not None and meas.exists():
                row = meas.final_row(scope)
                acc = SC.accepted(row, c.HELDOUT_MIN_GAIN_AS)
                pr(f"      final row: {row['status'][:60] if row else 'missing'} -> {'ACCEPTED' if acc else 'not accepted'}")
                why = SC.measurement_currency(roots, cell, row)
                if why:
                    pr(f'      WOULD REFUSE - {why}')
                    would.append(why)
            try:
                cmg = SC.content_check_script()
            except Exception as e:
                cmg = f'UNRESOLVED ({e})'
                would.append(str(e))
            pr(f'      if accepted: {gen} (dewarped_master; D1-D6 via {cmg})'
               + (' - EXISTS: G-A3 refuses without --fresh' if exists and 'apply' in stages else ''))
            pr(f'      if not accepted: applied=0 row in {roots.record_csv} (screen_table); install kind rebuild '
               f'{roots.rebuild}/{cell}')
    pr(f'\n  verdict: ' + (f'{len(would)} gate(s) would refuse (exit 2)' if would else 'every gate passes (exit 0)')
       + ('; G-W would refuse these roots without --out' if gw else ''))
    sys.stdout.flush()
    return 2 if would else 0


def run(args, run):
    from exopipe.phase3.core.record import NoInput, Refusal
    from exopipe.phase3.lib.dewarp import steps_common as SC
    try:
        stages = SC.parse_stages(args.stages)
    except ValueError as e:
        print(f'error: {e}', file=sys.stderr)
        return 1
    cell_stages = [s for s in stages if s in SC.CELL_STAGES]
    if 'measure' in stages and args.measurement:
        print('error: --measurement replaces the measure stage: give one or the other', file=sys.stderr)
        return 1
    if cell_stages and 'measure' not in stages and not args.measurement:
        print(f'error: stages {",".join(cell_stages)} need a measurement: include measure or give --measurement DIR',
              file=sys.stderr)
        return 1
    if args.measurement and not os.path.isdir(args.measurement):
        print(f'error: --measurement {args.measurement} is not a directory', file=sys.stderr)
        return 1
    if args.prior_gen_root and 'verify' not in stages:
        print('error: --prior-gen-root is used by the verify stage only', file=sys.stderr)
        return 1
    if args.cells and args.lsts:
        print('error: give --cells or --lsts, not both', file=sys.stderr)
        return 1
    roots = SC.Roots(args, run)
    gate_bad, _ = SC.gate_in_place(import_heavy=True)
    gate_bad += SC.gate_input_roots(roots, stages, args.measurement)
    if 'verify' in stages:
        try:
            gate_bad += SC.gate_child_modules()[0]           # G-D0b (the content-test child's code)
        except Refusal as e:
            gate_bad.append(str(e))
    staging = None
    if 'measure' in stages and SC.C().STAGE_INPUTS:
        try:
            staging = SC.scratch_root(roots.work)
        except Refusal as e:
            if not args.dry_run:
                raise
            staging = f'(refuses at run time: {e})'
    gw = SC.gate_write_roots(roots, stages, staging if staging and not staging.startswith('(') else None)
    if not args.dry_run:
        if gate_bad:
            raise Refusal('; '.join(gate_bad[:6]))
        if gw:
            raise Refusal('G-W: ' + '; '.join(gw))
    try:
        plan, refused, route_src, resume_only = _plan(args, roots, stages, run)
    except ValueError as e:
        print(f'error: {e}', file=sys.stderr)
        return 1
    if args.dry_run:
        return _dry_run(args, run, roots, stages, gate_bad, gw, staging, plan, refused, route_src)
    if not plan:
        raise NoInput('no routed promoted cell in any LST (dewarp.ROUTE_SOURCE '
                      f'{SC.C().ROUTE_SOURCE}; promoted rebuilds under {roots.rebuild})')
    from exopipe.phase3.lib.dewarp import steps_cell as SCELL, steps_measure as SM
    measure_only = not cell_stages
    for lst in sorted(plan):
        good = plan[lst]
        bad = sorted(x for x in refused if x.startswith(lst + '_'))
        meas = None
        if measure_only or good:
            if args.measurement:
                meas = SC.Measurement(args.measurement, lst)
            else:
                try:
                    meas = SM.run_measure(run, args, roots, lst, good, stages)
                except Exception as e:      # recorded as 'error' by the unit; its cells refuse, the next LST goes on
                    print(f'{lst}: measurement unit ERROR {type(e).__name__}: {e}', file=sys.stderr)
                    meas = None
        elif not bad:
            with run.unit(lst, kind='measure') as u:
                raise NoInput(f'{lst}: no routed promoted cell (dewarp.ROUTE_SOURCE {SC.C().ROUTE_SOURCE}, promoted '
                              f'rebuilds under {roots.rebuild})')
        for cell in bad:
            with run.unit(cell, lst=lst, kind='cell') as u:
                u.refuse(refused[cell])
        for cell in good:
            try:
                SCELL.run_cell(run, args, roots, lst, cell, meas, stages, route_src, resume_only.get(cell))
            except Exception as e:          # recorded as 'error' by the unit (exit 1 at the end); the next cell goes on
                print(f'{cell}: unit ERROR {type(e).__name__}: {e}', file=sys.stderr)
    return None
