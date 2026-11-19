"""exopipe3 mosaic - all-sky HEALPix mosaics of the deep masters: co-add, renders, QA, archive staging (survey step).

Mosaics only, one command; every tunable is in config.py (sections mosaic, mosaic_coadd, mosaic_render, mosaic_qa,
mosaic_conform - `exopipe3 config --step mosaic`), overridable per run with --set or a --config file:

    exopipe3 mosaic --out DIR --work /fast/gh/refactor_scratch          # every v2 mosaic (19 scopes x 4 variants), all stages
    exopipe3 mosaic --scopes Red,Green,Blue,Full --variants Taper_R075 --out DIR --work W         # a subset
    exopipe3 mosaic --jobs Blue_Taper_R075 --dry-run                     # the plan: masters found, gates, outputs
    exopipe3 mosaic --config my_mosaic.py --out DIR --work W             # a campaign's overrides in one file
    exopipe3 mosaic --jobs @stale.txt --resume --out DIR --work W        # rebuild only what changed (records decide)
    exopipe3 mosaic --list-jobs --scopes 18MHz,23MHz > jobs.txt          # item list for the runner
    exopipe3 run submit --nodes lwacalim09:10 --items @jobs.txt -- mosaic --jobs {item} --stages inputs,coadd,render \\
        --workers 1 --out DIR --work W
    exopipe3 mosaic --jobs @jobs.txt --stages qa,conform --out DIR       # once, after that job has finished

What it builds (mosaic v2, PHASE3_ARCHITECTURE.md 7bis; Gregg 2026-09-26 'Case A'): for each JOB = (scope, imaging variant),
a PB^2/sigma^2-weighted, PB-rim-feathered (0.15 -> 0.10), bilinear co-add of the per-LST PB-corrected deep masters on a
NSIDE 2048 HEALPix grid, in two domains - Jy/sr with its Omega_eff psfmap (the canonical deliverable) and apparent Jy/beam
with its PB_eff map ('jybeam approved') - then Mollweide (Galactic) and RA/Dec (CAR) renders of the wideband scopes, a
MEASURE-ONLY calibrator flux check and VLSSr astrometry table (the mosaics are never rescaled), and the conformant
archive-staging copies (Taper variants, wideband scopes) that `exopipe3 archive push --class mosaic` transfers.

Unit of work: a mosaic job, item text <scope>_<variant> (Blue_Taper_R075, 27MHz_NoTaper_R0); 19 scopes x 4 variants = 76
jobs, 152 co-adds. An LST is a TILE of every job, never a unit. The LST list is the LST directories of the input master
generation that exist (mosaic.LSTS None; deep chat 927a836); --lsts restricts it. Record units: inputs/<job>, coadd/<job>,
render/<job>, conform/<job> (one per job); qa/_all and conform/_tables (run-wide).

Use: INTERMITTENT only (DESIGN 12.2) - after the masters of a generation are rebuilt, dewarped and installed (e.g. when
the data on disk grew by ~50%), or when masters changed (`--resume` rebuilds exactly the jobs whose masters, admitted nights,
code or config changed). The per-hour chain (runner.CHAINS['epoch']) never runs it; there is no --epochs mode.

Stages (--stages a,b / --from S / --until S; pipeline order; default all; 'coadd' implies 'inputs'):
  inputs   per job: resolve one current pbcorr master per LST tile (both name forms, _shflux_ and _shflux_mdewarp_),
           exclusion gate on every constituent night (and on the weighting-PB reference master), header checks, then the
           node-local stage made by the IN-PLACE exopipe_ops/mosaic_stage_from_lustre.py (symlinks + weighting PB maps of
           both robusts per scope, as production staged them), stage checks, and a census of every tile with the engine's
           own guard arithmetic. Alone (with --keep-stage) it is the input census; --dry-run prints it without staging.
  coadd    per job and domain: lib/mosaic/coadd_jysr.coadd_jysr in place on the stage (same call as run_mosaic_v2*.sh);
           the engine's tile verdicts must equal the census; files placed atomically into <out>/healpix/.
  render   per job of mosaic_render.SCOPES and domain: lib/mosaic/band_products.build in place -> <out>/renders/.
  qa       run-wide, measure only: exopipe_ops/mosaic_v2_flux_anchor_measure.py in place over the jysr maps of the
           selected jobs' coadd records + verify_mosaic.astrometry_check per map -> <out>/qa/.
  conform  per job of mosaic_conform.VARIANTS x SCOPES: the archive-staging copies (port of the 'wb2' branch of
           mosaic_conformance_pass.py: build_name, write_standard_header, append_provenance_table) -> <out>/archive_staging/
           mosaic/; and conform/_tables: the qa tables -> <out>/archive_staging/mosaic/tables/.
Work runs in child processes (lib/mosaic/steps_worker.py, site interpreter 'default') on lanes of mosaic.CORES_PER_JOB cores:
--workers N lanes (default min(mosaic.MAX_JOBS_PER_NODE, usable cores / CORES_PER_JOB)); renders at most
mosaic.RENDER_MAX_PARALLEL at a time. Scratch (the stage) is node-local (--work or the site SCRATCH_ROOT; never Lustre),
removed at the end unless --keep-stage.

Inputs:  the masters <mosaic.INPUT_COADD_RUN>/<LST>/<scope>/<scope>_I_deep_<Taper|NoTaper>_Robust<-0.75|+0.0>_shflux[_mdewarp]_
         pbcorr_dewarped_aligned_coadd_LST<LST>_<d0>-<d1>_N<nnn>.fits (read only; constituent nights from HISTORY
         'COADD constituent'); the PB library through pb_apply (weighting PB, three-factor model); data_decisions.csv via
         core/exclusions; mosaic_qa.VLSSR_CATALOG; exopipe.common.pipeline_config.CALIB_DATA. Later stages read the earlier
         stages' RUN RECORDS (<out>/_records/mosaic/<stage>/<job>.json), never a glob of the build tree.
Outputs (today's mosaic_v2 layout under --out; product classes in brackets):
  <out>/healpix/healpix_I_deep_<V>_<S>_jysr_nside2048.fits, ..._jybeam_nside2048.fits            [mosaic_map]
  <out>/healpix/healpix_PSF_I_deep_<V>_<S>_nside2048.fits                                          [mosaic_psfmap]
  <out>/healpix/healpix_PBeff_I_deep_<V>_<S>_nside2048.fits                                        [mosaic_pbeff]
  <out>/renders/<S>_<V>_<jysr|jybeam>_{Mollweide,RADec}.fits                                       [mosaic_render]
  <out>/qa/flux_anchor_measured_<yyyymmdd>.csv, <out>/qa/vlssr_astrometry_<yyyymmdd>.csv          [mosaic_qa_table]
  <out>/archive_staging/mosaic/wideband/<build_name>.fits (jysr, psfmap, jybeam, pbmap), .../renders/wideband/{mollweide,
         radec}/<build_name>.fits, .../tables/*.csv                                                [mosaic, mosaic_qa_table]
  node-local stage <work>/mosaic_stage_*/<scope>/<LST>/PB_deep[_R075]_<scope>.fits (scratch)       [mosaic_weight_pb]
  <out>/logs/<V>_<S>_<domain>.log (engine output, as run_mosaic_v2.sh), stage_/census_/render_/qa logs
  <out>/_records/mosaic/<stage>/<job>.json, qa/_all.json, conform/_tables.json, _runs/<run_id>.json
  Default --out = mosaic.BUILD_ROOT (a production root: refused while paths.PRODUCTION_WRITES is False).
Failure modes (exit 2 = refused, with the gate named; 3 = nothing to do; 1 = error):
  G0   a config value differs from the literal the in-place code uses (engine, renderer, anchor script, stage script ROOT /
       VARMAP / name forms), or a key with one implementation set to another value - refused before anything runs
  G1   a tile master (or its weighting-PB reference master) contains a night data_decisions.csv excludes, or carries no
       'COADD constituent' HISTORY (not waivable)        G2 exclusion list missing/unreadable (core/exclusions)
  G3   0 or > 1 current-master candidates for a planned tile; a staged link that is not the planned master; the stage
       script refused the scope (a duplicate in any variant of it)
  G4   fewer LST tiles than mosaic.REQUIRE_N_LSTS (waive with --waive mosaic.inputs_complete --ruling "...")
  G5   weighting PB of the variant's robust missing after staging (derivation failed)   G6 PB grid != master grid
  G7   P3_PB_LIBRARY / P3_RESIDUAL_PB / P3_CHAN_AVG_HZ / P3_PB_THREE_FACTOR / P3_RESIDUAL_PB_V2 set in the environment
  G8   a tile header without PBCORR True, PBMODEL/PBCHROM/PBRESID, BMAJ/BMIN, or a frequency above 1 MHz
  G9   tiles the engine guard skips (< 100 valid pixels, CORRUPT-FLAT): recorded and kept out (TILE_GUARD_ACTION
       'skip-and-record', today's behaviour) or refused ('refuse'); the engine's verdicts differing from the census refuses
  G10  no tile contributed       G11 mosaic_qa.APPLY_FLUX_ANCHOR True (the mosaics are never rescaled)
  G12  conform of an explicitly named NoTaper job (push Taper only) unless --waive mosaic.conform_taper_only --ruling
  G13  any staging failure (nothing half-staged survives)   G14 no contributing master with MJD-OBS and DATE-END
  G16  VLSSr median offset above mosaic_qa.VLSSR_GATE_MAX_MEDIAN_OFFSET_ARCMIN (None = measure only, today)
  G15  --out under a production root while paths.PRODUCTION_WRITES is False (cli); no node-local scratch root (core.fitsio)
  A content test failing on any output refuses its unit. A later stage without a done record of its predecessor refuses.

Generation v5 (--generation v5; the hybrid mosaics of mosaic/Run_HYBRID, plan/specs/hybrid.md 2.8, 2.10-2.14): units (stage, scope,
weighting) of the stages stage, coadd, pkcorr, caveat (hybrid.CAVEAT_SCOPES), fill, render, content - see lib/mosaic/v5.py.
--scopes (default mosaic_v5.SCOPES), --weightings, --hybrid-out (the rebuilt tiles), --out (required), --work (node-local link trees):
    exopipe3 mosaic --generation v5 --scopes Blue --weightings R075 --out /fast/gh/v5run --work /fast/gh/scratch
    exopipe3 mosaic --generation v5 --hybrid-out /fast/gh/hybrid_q1 --scopes 18MHz,Red --out /fast/gh/v5run --dry-run

Generation v3 (--generation v3; the FEATHER3 mosaics of mosaic/Run_FEATHER3, plan/specs/hybrid.md 2.9-2.13): units (stage, scope, job)
of the stages stage, coadd, pkcorr (Stokes I), render, content - see lib/mosaic/v3.py. Jobs: the plain Stokes I maps of
mosaic_v3.VARIANTS, the striation-corrected maps of the pairs of hybrid.STRIATION_SELECTION (--striation with|without|both, default
mosaic_v3.STRIATION), the cleaned Stokes V maps (--products I,Vclean). --scopes, --variants, --out (required), --work (node-local link
trees):
    exopipe3 mosaic --generation v3 --scopes Blue --out /fast/gh/v3run --work /fast/gh/scratch
    exopipe3 mosaic --generation v3 --scopes 78MHz --striation with --stages stage --out /fast/gh/v3run --dry-run
"""
import os
import sys

SUMMARY = 'all-sky HEALPix mosaics of the deep masters: co-add, renders, QA, archive staging'
TIERS = ('deep',)
SELECTORS = ('tier', 'scopes', 'lsts')
CONFIG = ('mosaic', 'mosaic_coadd', 'mosaic_render', 'mosaic_qa', 'mosaic_conform', 'mosaic_v5', 'mosaic_caveat', 'mosaic_fill',
          'mosaic_content', 'mosaic_pkcorr', 'mosaic_v3', 'masters_striation',
          'hybrid', 'hybrid_recon', 'hybrid_tilefill', 'paths', 'exclusions', 'instrument', 'runner', 'site')
WRITES = True
STAGES = ('inputs', 'coadd', 'render', 'qa', 'conform')

from exopipe.phase3.lib.mosaic import tests as _tests   # noqa: E402,F401  (registers the content tests; light import)


def add_arguments(p):
    p.add_argument('--jobs', metavar='SPEC',
                   help='mosaic jobs <scope>_<variant>[_<product>]: list, shell patterns over the 76 jobs (*_Taper_R075), '
                        '@FILE (job items, or "variant band domain" lines as run_mosaic_jobs.sh reads them), '
                        'record:STATUS:mosaic@OUT, JOB:STATUS. Default: every scope of --scopes x every variant of '
                        '--variants')
    p.add_argument('--variants', metavar='LIST', help='imaging variants (default mosaic.VARIANTS: Taper_R075,Taper_R0,'
                                                      'NoTaper_R075,NoTaper_R0)')
    p.add_argument('--domains', metavar='LIST', help='co-add domains (default mosaic.DOMAINS: jysr,jybeam)')
    p.add_argument('--products', metavar='LIST', help="products (default mosaic.PRODUCTS: I; any other refuses); v3: I, Vclean (default "
                                                      "mosaic_v3.PRODUCTS)")
    p.add_argument('--stages', metavar='LIST', help='comma list of stages (default all): v2 inputs,coadd,render,qa,conform; v3 stage,'
                                                    'coadd,pkcorr,render,content; v5 stage,coadd,pkcorr,caveat,fill,render,content,conform,'
                                                    'tables')
    p.add_argument('--from', dest='from_stage', metavar='STAGE', help='first stage to run')
    p.add_argument('--until', dest='until_stage', metavar='STAGE', help='last stage to run')
    p.add_argument('--list-jobs', action='store_true', help='print the selected job items, one per line, and exit')
    p.add_argument('--keep-stage', action='store_true', help='keep the node-local stage (symlinks + weighting PB maps)')
    g = p.add_argument_group('generations v3 and v5 (the FEATHER3 mosaics, lib/mosaic/v3.py; the hybrid mosaics, lib/mosaic/v5.py)')
    g.add_argument('--generation', choices=('v2', 'v3', 'v5'), default='v2',
                   help='v2 (default): the mosaics of the deep masters, stages inputs..conform; v3: the FEATHER3 mosaics (plain, '
                        'striation-corrected, Stokes V), stages stage,coadd,pkcorr,render,content (--out required); v5: the hybrid mosaics of the '
                        'unfilled hybrid masters, stages stage,coadd,pkcorr,caveat,fill,render,content (--out required)')
    g.add_argument('--striation', choices=('with', 'without', 'both'),
                   help='v3: the striation-corrected maps of the selected pairs (with), the plain maps (without) or both (default '
                        'mosaic_v3.STRIATION)')
    g.add_argument('--weightings', metavar='LIST', help='v5: R075, R0 or both (default mosaic_v5.WEIGHTINGS)')
    g.add_argument('--hybrid-out', metavar='DIR',
                   help='v5 stage: the `exopipe3 hybrid` work run whose done hybrid records give the rebuilt tiles (work); every other '
                        'LST takes the pinned unfilled source of its installed master (default: all pinned)')


def default_out(args):
    if getattr(args, 'generation', 'v2') == 'v5':
        from exopipe.phase3.config import ConfigRefusal
        raise ConfigRefusal('exopipe3 mosaic --generation v5 needs --out: a work run of its own (node-local; ~2.4 GB per scope and '
                            'weighting), never an archive run or the hybrid work run')
    if getattr(args, 'generation', 'v2') == 'v3':
        from exopipe.phase3.config import ConfigRefusal
        raise ConfigRefusal('exopipe3 mosaic --generation v3 needs --out: a work run of its own (~1 GB of weighting maps per plain job, '
                            '~0.8 GB of maps per job), never an archive run')
    if getattr(args, 'list_jobs', False):
        return '/tmp/exopipe3_mosaic_list_jobs'
    from exopipe.phase3.config import CFG
    return CFG.mosaic.BUILD_ROOT


def select_stages(args):
    st = list(STAGES)
    if args.stages:
        want = [s.strip() for s in args.stages.split(',') if s.strip()]
        bad = [s for s in want if s not in STAGES]
        if bad:
            raise ValueError(f'--stages: unknown stage(s) {bad} (stages: {",".join(STAGES)})')
        st = [s for s in STAGES if s in want]
    for flag, val in (('--from', args.from_stage), ('--until', args.until_stage)):
        if val and val not in STAGES:
            raise ValueError(f'{flag}: unknown stage {val!r} (stages: {",".join(STAGES)})')
    if args.from_stage:
        st = [s for s in st if STAGES.index(s) >= STAGES.index(args.from_stage)]
    if args.until_stage:
        st = [s for s in st if STAGES.index(s) <= STAGES.index(args.until_stage)]
    if 'coadd' in st and 'inputs' not in st:
        st.insert(0, 'inputs')          # the stage is node-local scratch removed at the end of every run
    return st


def _start_gates(run):
    """G0 (in-place literals), G7 (PB environment), G11 (no rescaling): refuse before anything runs."""
    from exopipe.phase3.config import CFG
    from exopipe.phase3.core.record import Refusal
    from exopipe.phase3.lib.mosaic import steps_common as SC
    probs = SC.g0_check()
    if probs:
        raise Refusal('G0: ' + ' | '.join(probs))
    env = [k for k in SC.PB_ENV if os.environ.get(k) not in (None, '')]
    if env:
        raise Refusal(f'G7: {", ".join(env)} set in the environment - the weighting PB must be the ruled three-factor map '
                      f'(LANDED-THREE-FACTOR-PB-MODEL-CHROMATIC-RESIDUAL); unset them')
    if CFG.mosaic_qa.APPLY_FLUX_ANCHOR:
        raise Refusal('G11: mosaic_qa.APPLY_FLUX_ANCHOR is True - the mosaics keep the masters\' scale and are never '
                      'rescaled (R-MOSAIC-V2-FLUX-SCALE-MEASURED-NOT-CORRECTED)')


def _print_plan(args, run, jobs, plans, stages, lsts, universe):
    from exopipe.phase3.config import CFG
    from exopipe.phase3.core import fitsio
    from exopipe.phase3.core.record import Refusal
    from exopipe.phase3.lib.mosaic import steps_common as SC
    out = run.out_root
    print(f'mosaic plan: {len(jobs)} job(s), stages {",".join(stages)}; {len(lsts)} LST tiles '
          f'({lsts[0] if lsts else "-"}..{lsts[-1] if lsts else "-"}; the LST directories of '
          f'{CFG.mosaic.INPUT_COADD_RUN}: {len(universe)})')
    n_ref = 0
    verbose = len(jobs) <= 2
    for job in jobs:
        P = plans.get(job.name)
        hdr = f'job {job.name}: scope {job.scope}, {CFG.mosaic.VARIANT_TOKENS.get(job.variant)}, domains ' \
              f'{",".join(job.domains)}'
        if P is None:
            print(hdr + ' (reads the earlier stages\' records)')
            continue
        if verbose:
            print(hdr)
            for lst in P.lsts:
                t = P.tiles.get(lst) or {}
                h = t.get('hdr') or {}
                if t.get('master'):
                    pr = t.get('pb_ref') or {}
                    print(f"  {lst} {os.path.basename(t['master'])}  nights {len(h.get('dates') or ())} "
                          f"(excluded {len(t.get('excluded') or ())})  PBCORR {h.get('PBCORR')}  "
                          f"freq {(h.get('freq_hz') or 0) / 1e6:.3f} MHz  BMAJ {h.get('BMAJ')}"
                          + (f"  PB ref {os.path.basename(pr['master'])[:48]} (excluded {len(pr['excluded'])})"
                             if pr else ''))
                else:
                    print(f"  {lst} {t.get('n_candidates', 0)} candidates")
        if P.refusals:
            n_ref += 1
            print(f'  {job.name}: WOULD REFUSE - ' + P.reason())
        else:
            nm = sum(1 for t in P.tiles.values() if (t.get('hdr') or {}).get('mdewarp'))
            print(f'  {job.name}: gates G1/G3/G4/G8 pass ({len(P.masters)} tiles, {nm} _mdewarp_, '
                  f'{len(P.admitted)} constituent tile-nights admitted)')
        for n in P.notes:
            print(f'  note: {n}')
        if verbose or not P.refusals:
            from exopipe.phase3.lib.masters.pb_apply import deep_pb_name
            if 'inputs' in stages:
                print(f'    inputs  -> {len(P.lsts)} x <stage>/{job.scope}/<LST>/{deep_pb_name(job.scope, job.robust)} '
                      f'[mosaic_weight_pb, scratch]')
            if 'coadd' in stages:
                for d in job.domains:
                    for name, cls in SC.coadd_products(job, d):
                        print(f'    coadd   -> {os.path.join(SC.layout(out, "healpix"), name)} [{cls}]')
            if 'render' in stages and job.scope in CFG.mosaic_render.SCOPES:
                for d in job.domains:
                    for name, _k in SC.render_products(job, d):
                        print(f'    render  -> {os.path.join(SC.layout(out, "renders"), name)} [mosaic_render]')
            if 'conform' in stages and job.variant in CFG.mosaic_conform.VARIANTS and \
                    job.scope in CFG.mosaic_conform.SCOPES:
                print(f'    conform -> {SC.layout(out, "staging")}/{CFG.mosaic_conform.SUBDIR}/<build_name> x 4 + 4 renders '
                      f'[mosaic]')
    if 'qa' in stages:
        d = SC.utc_date()
        print(f'qa -> {os.path.join(SC.layout(out, "qa"), CFG.mosaic_qa.ANCHOR_TABLE_NAME.format(yyyymmdd=d))}, '
              f'{CFG.mosaic_qa.VLSSR_TABLE_NAME.format(yyyymmdd=d)} [mosaic_qa_table] (measure only)')
    hits = []
    if CFG.production_root_of(out) is not None and not CFG.paths.PRODUCTION_WRITES:
        hits.append(f'--out {out} is under the production root {CFG.production_root_of(out)} (refused while '
                    f'paths.PRODUCTION_WRITES is False)')
    if any(s in stages for s in ('inputs', 'coadd')):
        try:
            print(f'scratch root: {fitsio.scratch_root(args.work)}')
        except Refusal as e:
            hits.append(str(e))
    from exopipe.phase3.lib.mosaic.steps_common import default_workers, lanes
    print(f'lanes: {len(lanes(default_workers(args), CFG.mosaic.CORES_PER_JOB))} x {CFG.mosaic.CORES_PER_JOB} cores')
    for h in hits:
        print(f'a real run would refuse here: {h}')
    print(f'{len(jobs) - n_ref} of {len(jobs)} job(s) pass the input gates' + ('' if not n_ref else
                                                                               f'; {n_ref} would be refused'))
    from exopipe.phase3.core.record import EXIT_REFUSED
    return EXIT_REFUSED if n_ref else 0


def run(args, run):
    import tempfile
    from exopipe.phase3.config import CFG
    from exopipe.phase3.core import exclusions, fitsio
    from exopipe.phase3.core.record import EXIT_NO_INPUT
    from exopipe.phase3.lib.mosaic import steps_common as SC

    if getattr(args, 'generation', 'v2') == 'v3':
        from exopipe.phase3.lib.mosaic import v3
        other = [f for f in ('jobs', 'domains', 'lsts', 'list_jobs', 'keep_stage', 'weightings', 'hybrid_out') if getattr(args, f, None)]
        if other:                             # never ignored: v3 selects by --scopes, --products, --variants, --striation, --stages
            raise ValueError(f'--generation v3 refuses the option(s) {", ".join("--" + f.replace("_", "-") for f in other)} (v3 '
                             f'selects by --scopes, --products, --variants, --striation and --stages; every LST is a tile)')
        if (getattr(args, 'workers', None) or 1) > 1:   # FINDING-20261118-V2-HYBRID-STEP-IGNORES-WORKERS: never silently ignored
            raise ValueError(f'--generation v3 refuses --workers {args.workers}: its units run one after another in one process; run '
                             f'one process per disjoint --scopes group instead (unit records are per unit; tables last, alone: '
                             f'--stages tables)')
        _start_gates(run)
        return v3.run(args, run)
    if getattr(args, 'generation', 'v2') == 'v5':
        from exopipe.phase3.lib.mosaic import v5
        v2only = [f for f in ('jobs', 'variants', 'domains', 'products', 'lsts', 'list_jobs', 'keep_stage', 'striation')
                  if getattr(args, f, None)]
        if v2only:                            # never ignored: v5 selects by --scopes, --weightings, --stages (--dry-run plans)
            raise ValueError(f'--generation v5 refuses the v2 option(s) {", ".join("--" + f.replace("_", "-") for f in v2only)} '
                             f'(v5 selects by --scopes, --weightings and --stages; --dry-run prints the plan)')
        if (getattr(args, 'workers', None) or 1) > 1:   # FINDING-20261118-V2-HYBRID-STEP-IGNORES-WORKERS: never silently ignored
            raise ValueError(f'--generation v5 refuses --workers {args.workers}: its units run one after another in one process; run '
                             f'one process per disjoint --scopes group instead (unit records are per unit; tables last, alone)')
        _start_gates(run)
        return v5.run(args, run)
    for flag in ('weightings', 'hybrid_out'):
        if getattr(args, flag, None):
            raise ValueError(f'--{flag.replace("_", "-")} is a --generation v5 option')
    if getattr(args, 'striation', None):
        raise ValueError('--striation is a --generation v3 option')
    jobs = SC.select_jobs(args)
    if args.list_jobs:
        run.no_records = True
        for j in jobs:
            print(j.name)
        return 0
    if not jobs:
        print('mosaic: no job selected', file=sys.stderr)
        return EXIT_NO_INPUT
    stages = select_stages(args)
    _start_gates(run)
    plans, lsts, universe = {}, [], []
    ex = None
    if any(s in stages for s in ('inputs', 'coadd')):
        lsts, universe = SC.select_lsts(args)
        ex = exclusions.load()
        for job in jobs:
            plans[job.name] = SC.plan_job(job, lsts, universe, ex, run)
    if run.dry_run:
        return _print_plan(args, run, jobs, plans, stages, lsts, universe)

    # code identity: the in-place modules the children run are imported here too (record.code_fingerprint)
    from exopipe.phase3.lib.mosaic import coadd_jysr, band_products, verify_mosaic, steps_worker   # noqa: F401
    from exopipe.phase3.lib.masters import pb_apply                                                 # noqa: F401
    from exopipe.phase3.lib.mosaic import steps_build, steps_qa, steps_conform

    workers = SC.default_workers(args)
    holder = {}

    def stage_root():
        if 'root' not in holder:
            holder['root'] = fitsio.scratch_dir('mosaic_stage', work=args.work)
            print(f'mosaic: node-local stage {holder["root"]}', file=sys.stderr)
        return holder['root']

    task_dir = tempfile.mkdtemp(prefix='exopipe3_mosaic_tasks_')
    ctx = {'stage_root': stage_root, 'task_dir': task_dir, 'exclusions': ex,
           'lanes': lambda n, k: SC.lanes(min(workers, max(1, n)), k)}
    ok = False
    try:
        if 'inputs' in stages:
            steps_build.run_inputs_coadd(args, run, jobs, plans, stages, ctx)
        if 'render' in stages:
            steps_build.run_render(args, run, jobs, ctx)
        if 'qa' in stages:
            steps_qa.run_qa(args, run, jobs, ctx)
        if 'conform' in stages:
            steps_conform.run_conform(args, run, jobs, explicit_jobs=bool(args.jobs))
            steps_conform.run_tables(args, run)
        ok = True
    finally:
        fitsio.cleanup(task_dir)
        if 'root' in holder:
            if args.keep_stage or not ok:
                print(f'mosaic: stage kept at {holder["root"]}' + ('' if ok else ' (run did not finish)'),
                      file=sys.stderr)
            else:
                fitsio.cleanup(holder['root'])
    return None
