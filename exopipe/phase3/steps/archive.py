"""exopipe3 archive - every change to the Phase 3 archive that is not a product's first write: manifest -> checksum transfer ->
content audit -> quarantine of superseded files; retire-excluded; rollback. PLAN-ONLY by default (no --apply = the manifest).

    exopipe3 archive retire-excluded --tier deep --out DIR                        # plan: excluded nights' files in the Run
    exopipe3 archive retire-excluded --apply --manifest PLAN.csv --apply-ruling "<Gregg's words, UTC, ledger ID>" --out DIR
    exopipe3 archive quarantine --cells 08h_59MHz --stokes V --round 20261012 --out DIR    # pre-search frame quarantine plan
    exopipe3 archive sync --cells 08h_59MHz --stokes V --attempt ATTEMPT_DIR --out DIR      # archive frames = attempt frames
    exopipe3 archive retire-refused --cells 08h_59MHz --stokes V --out DIR                  # M10: frames of refused nights
    exopipe3 archive push --epochs 04h_2025-10-14 --staging-root MIRROR --out DIR          # per hour: one epoch's products
    exopipe3 archive push --class mosaic --out DIR                                          # bulk: the mosaic v2 Taper push
    exopipe3 archive audit --class mosaic --manifest ops/MANIFEST_MOSAIC_V2_TAPER_PUSH_20260929.csv --out DIR
    exopipe3 archive rollback --cells 04h_18MHz --stokes I --since 2026-10-08T02:38:14Z --out DIR
    exopipe3 archive supersede --tier snapshot --cells 21h_2025-08-26 --workdir W --out DIR
    exopipe3 archive quarantine-rows --tier snapshot --stores fits_A.csv fits_B.csv --batch-name NAME --out DIR

Actions and units of work (one run record per unit, <out>/_records/archive/<unit>.json):
  quarantine       deep; unit quarantine_<cell>_<S>. Earlier archive search frames ('_msub', mtime before the launch epoch) of
                   the (cell, Stokes) -> <root>/Run_SEARCH_QUARANTINE_<round>/<same path>; membership precheck first (G7).
                   The search step's quarantine stage calls the same function (lib.archive.quarantine.quarantine_for_search).
  sync             deep; unit sync_<cell>_<S>. Plan keep / replace / publish / retire of the archive frames against the
                   published attempt (ops/p3s5_archive_sync.py in place); apply = '.part' copies -> audit -> quarantine +
                   rename -> closing re-plan all 'keep' (G17). Retire rows need --apply-ruling.
  retire-refused   deep; unit retire-refused_<cell>_<S>. The sync's retire rows of gate-REFUSED nights (M10); rows of excluded
                   nights are listed only (retire-excluded owns them).
  retire-excluded  deep: one unit per excluded key retire-excluded_<LST>_<scope>_<date> + summary unit retire-excluded_deep
                   (combined plan + retirement audit). sub-LST / snapshot: REPORT ONLY, unit per night cell.
  push             --class mosaic (unit push_mosaic): the mosaic v2 Taper push. --class epoch / --epochs (unit
                   push_epoch_<LST>_<date>): per hour, every product of one epoch staged in a tree mirroring the archive
                   (--staging-root; in the window the search step's <out>/archive_mirror) -> the archive; bulk = many epochs
                   (--epochs @list, or --lsts / --nights over the staging tree). No staging tree = audit of the hour's archive
                   files. --class qa / run-tables: disabled in config (refused with the reason).
  rollback         deep; unit rollback_<cell>_<S>: quarantined files back from their manifest (rows moved at/after --since).
  supersede        snapshot; unit supersede_<cell>: ops/snapshot_frametime_ship.sh in place (plan = its file listing).
  quarantine-rows  snapshot; unit rows_<name>: ops/snapshot_quarantine_wideband_fits.py in place (plan = its --dry per store).
  audit            unit audit_<class>_<manifest>: re-run the content tests of an earlier manifest (writes only records + an
                   audit table).

Per hour vs intermittent (DESIGN 12): per hour, the epoch chain ends with `archive push --epochs <LST>_<date>` (runner.CHAINS
['epoch']); intermittent: retire-excluded after an exclusion change, sync / retire-refused after a re-search round, bulk pushes
(mosaic, many epochs), rollback, audits. The pre-search quarantine runs inside the per-hour / re-run search.

Inputs:  the archive tree paths.PHASE3_ARCHIVE_ROOT (/lustre/pipeline/exopipe/phase3) <LST>/<date>/<paths.ARCHIVE_RUN>/<scope>/
         <S>/{deep,10min}/ (or --archive-root MIRROR); data_decisions.csv through core.exclusions; the launch epoch
         (archive_quarantine.LAUNCH_EPOCH_FILE) and the fleet claims (archive.WRITER_CLAIM); published attempts (p3s5_state/
         published_<S>.csv via p3s3_sources, or --attempt); quarantine manifests (archive.FRAME_QUARANTINE_MANIFEST_DIR); the
         mosaic staging tree (archive_push.CLASSES['mosaic']); a per-epoch staging tree (--staging-root); snapshot work dirs
         and fit stores. Every path: `exopipe3 config --step archive`.
Outputs (product classes; content tests in lib/archive/tests.py):
         plans and window manifests under <out>/manifests/<kind>/ - never in a production directory:
           quarantine_manifest      frame-quarantine / retire plans and applied manifests (path,dest,bytes,md5,cell,stokes,
                                    moved_at; DRYRUN_/APPLIED_excluded_archive_products_<UTC>.csv; ship MANIFEST.csv)
           archive_manifest         push / publish manifests (src,dst,bytes,md5 or source,dest,...,action,written_at)
           archive_sync_record      sync plan / closing record (JSON)
           retirement_audit         retire-excluded / retire-refused / report audit (JSON; re-scanned by its test)
           restore_manifest         rollback rows (restore_refused_frames_<UTC>.csv form)
           row_quarantine_manifest  QDIR/MANIFEST.csv of quarantine-rows
           archive_plan             plan of an in-place script (supersede, quarantine-rows)
           archive_mosaic_copy      each pushed mosaic file (push_mosaic_v2 audit); pushed per-epoch files keep their own
                                    class (search_frame, search_frame_pbcorr, loo_master, epoch_image)
         applied manifests of moves in the PRODUCTION archive go where production keeps them (p3s5_state/frame_quarantine/
         <cell>_<S>.csv, frame_publish/) - only when paths.PRODUCTION_WRITES is True.
Failure modes (exit 2 = refused, the gate named): G1 data_decisions.csv unreadable; G2 an apply under a production tree while
         paths.PRODUCTION_WRITES is False (the refactor window: plan only, or a mirror --archive-root); G3 cross-filesystem
         move; G4 quarantine destination exists; G5 move not verified (moved back); G6 copy checksum mismatch ('.part'
         removed, nothing replaced); G7 membership precheck failed / no cell inputs; G8 attempt frame without a search-frame
         PRODNAME; G9 push preconditions (non-Taper, intermediate file, existing destination); G10 ruled action without
         --apply-ruling naming a fix_ledger ID; G11 the plan re-derived now differs from --manifest; G12 a live fleet claim
         (concurrent writer) or a fit store written in the last 10 min; G13 fit-store header; G14 supersede approval marker /
         ship not verified; G15 rollback problems; G16 retirement audit (files left); G17 sync not closed on 'keep'; G18 no
         generation Run; G19 free space; a never-waivable test named in --waive. Exit 3 = nothing to do (no attempt, no file,
         no manifest row); exit 1 = error. Nothing is ever deleted (archive.NEVER_DELETE).

Refactor window (DESIGN 11.1): production helpers are called IN PLACE - p3s5_quarantine_frames.py (subprocess), the membership
precheck (production code in a child), p3s5_archive_sync.plan/same_data (in-process), snapshot_frametime_ship.sh and
snapshot_quarantine_wideband_fits.py (subprocesses via core.legacy.run). Gregg's go for an apply is --apply-ruling (the common
--ruling is paired with --waive by cli.py).
NOT YET: `audit --tree` (classification of a whole tree), header repair in place, clean/migrate subcommands; --workers is
ignored (units run one after another).
"""
SUMMARY = 'manifest, checksum transfer, content audit, quarantine; retire excluded nights; per-hour push; rollback'
TIERS = ('deep', 'sublst', 'snapshot')
SELECTORS = ('tier', 'cells', 'lsts', 'nights', 'stokes', 'run_label', 'exclude_cells', 'epochs')
CONFIG = ('archive', 'archive_quarantine', 'archive_sync', 'archive_retire', 'archive_push', 'archive_snapshot',
          'archive_rollback', 'paths', 'exclusions', 'instrument', 'runner', 'site')
WRITES = True

ACTIONS = ('push', 'sync', 'quarantine', 'retire-excluded', 'retire-refused', 'supersede', 'quarantine-rows', 'rollback',
           'audit')
ACTIONS_BY_TIER = {'deep': ('push', 'sync', 'quarantine', 'retire-excluded', 'retire-refused', 'rollback', 'audit'),
                   'sublst': ('retire-excluded', 'push', 'audit'),
                   'snapshot': ('supersede', 'quarantine-rows', 'retire-excluded', 'push', 'audit')}
STAGES = {'push': ('manifest', 'transfer', 'audit'), 'sync': ('manifest', 'transfer', 'audit', 'quarantine'),
          'quarantine': ('manifest', 'quarantine'), 'retire-excluded': ('manifest', 'quarantine', 'audit'),
          'retire-refused': ('manifest', 'quarantine', 'audit'), 'supersede': ('manifest', 'transfer', 'quarantine', 'audit'),
          'quarantine-rows': ('manifest', 'quarantine'), 'rollback': ('manifest', 'quarantine', 'audit'), 'audit': ('audit',)}

# Actions whose apply moves or writes files of the archive tree (--archive-root); G2 is checked on that root before
# anything else. supersede / quarantine-rows check their own roots (ship, photometry, quarantine; fit stores, QDIR).
_WRITES_ARCHIVE_TREE = ('push', 'sync', 'quarantine', 'retire-excluded', 'retire-refused', 'rollback')

from exopipe.phase3.lib.archive import tests as _content_tests  # noqa: E402,F401  (registers the archive content tests)


def add_arguments(p):
    p.add_argument('action', choices=ACTIONS, help='what to do (see the list above)')
    g = p.add_argument_group('archive')
    g.add_argument('--class', dest='push_class', metavar='C',
                   help='push: mosaic, epoch, qa, run-tables (--epochs alone = epoch); audit: mosaic, epoch, quarantine, '
                        'sync, retire, rollback, rows, supersede')
    g.add_argument('--apply', action='store_true',
                   help='perform copies and moves; without it every action stops after its manifest stage (the plan, '
                        'written under --out). Refused under a production tree while paths.PRODUCTION_WRITES is False')
    g.add_argument('--manifest', dest='manifest_list', action='append', metavar='FILE',
                   help='the approved manifest an apply executes (re-derived now and compared, G11); audit: the '
                        'manifest(s) to audit (repeatable, @list); rollback: the quarantine manifest ({cell}/{stokes} '
                        'expanded; default archive.FRAME_QUARANTINE_MANIFEST_DIR/<cell>_<S>.csv)')
    g.add_argument('--apply-ruling', action='append', default=[], metavar='TEXT',
                   help="Gregg's go for a ruled apply (archive.RULING_REQUIRED): \"<Gregg's words, UTC, ledger ID>\"; the "
                        'ledger ID must be a row of paths.FIX_LEDGER (G10)')
    g.add_argument('--stages', help='stages of the action, comma-separated (default: manifest; with --apply all of '
                                    'STAGES[action]); e.g. --stages audit --manifest F re-audits an earlier push')
    g.add_argument('--from', dest='from_stage', metavar='STAGE', help='first stage to run')
    g.add_argument('--until', dest='until_stage', metavar='STAGE', help='last stage to run')
    g.add_argument('--archive-root', metavar='DIR',
                   help='archive root of this run (default paths.PHASE3_ARCHIVE_ROOT); a MIRROR root outside every '
                        'production tree lets an apply run in the refactor window (parity, tests)')
    g.add_argument('--generation-run', metavar='RUN', help='archive generation Run (default paths.ARCHIVE_RUN)')
    g.add_argument('--round', help='quarantine round: Run_SEARCH_QUARANTINE_<round> (default archive_quarantine.ROUND_DEFAULT; '
                                   'production launches pass it, e.g. 20261012)')
    g.add_argument('--launch-epoch', metavar='UTC_S',
                   help='quarantine: frames last modified before this are earlier frames (default: the content of '
                        'archive_quarantine.LAUNCH_EPOCH_FILE)')
    g.add_argument('--no-membership-precheck', action='store_true',
                   help='quarantine: skip the membership precheck (G7); needs --apply-ruling')
    g.add_argument('--attempt', action='append', metavar='DIR',
                   help='sync / retire-refused: attempt directory <cell>_<S>_<att> (repeatable; default the published, '
                        'harvested attempts of the register)')
    g.add_argument('--since', metavar='UTC', help='rollback: manifest rows moved at or after this (YYYY-MM-DDTHH:MM:SSZ or epoch)')
    g.add_argument('--occupant-quarantine', metavar='RUN',
                   help='rollback: Run directory under the archive root that receives a file now occupying an original path '
                        '(written at/after --since)')
    g.add_argument('--cadences', help='retire-excluded: cadence directories scanned (default archive_retire.CADENCES)')
    g.add_argument('--staging-root', metavar='DIR',
                   help='push --class epoch: tree mirroring the archive layout that holds the hour\'s products (default '
                        'archive_push.EPOCH_STAGING_ROOT; none = audit the archive\'s own files of the hour)')
    g.add_argument('--workdir', help='supersede: the frame-time work directory W (W/out, W/cutouts; {cell} expanded)')
    g.add_argument('--ship-root', help='supersede: live per-cell tree (default paths.SNAPSHOT_FLEET_OUT)')
    g.add_argument('--phot-root', help='supersede: photometry cutout root (default archive_snapshot.PHOT_ROOT)')
    g.add_argument('--quar-root', help='supersede: quarantine root (default archive_snapshot.FRAMETIME_QUARANTINE_ROOT)')
    g.add_argument('--stores', nargs='+', metavar='CSV', help='quarantine-rows: fit store files (default the --cells stores '
                                                              'of archive_snapshot.BATCH_ROOT/fits/)')
    g.add_argument('--qdir', help='quarantine-rows: quarantine directory (default archive_snapshot.ROW_QUARANTINE_QDIR)')
    g.add_argument('--quiet-min', type=float, default=None,
                   help='quarantine-rows: refuse a store written less than this many minutes ago (default '
                        'archive_snapshot.ROW_QUARANTINE_QUIET_MIN)')
    g.add_argument('--batch-name', help='quarantine-rows: batch name (QDIR suffix and unit name; default the UTC stamp)')


def default_out(args):
    from exopipe.phase3.config import CFG
    return CFG.archive.RECORDS_ROOT


def resolve_stages(args):
    allst = STAGES[args.action]
    if args.stages:
        want = [s.strip() for s in args.stages.split(',') if s.strip()]
        bad = [s for s in want if s not in allst]
        if bad:
            raise ValueError(f'--stages: {bad} not stages of {args.action} ({", ".join(allst)})')
        sel = [s for s in allst if s in want]
    elif args.apply:
        sel = list(allst)
    else:
        sel = ['manifest'] if 'manifest' in allst else list(allst)
    for flag, val in (('--from', args.from_stage), ('--until', args.until_stage)):
        if val and val not in sel:
            raise ValueError(f'{flag} {val}: not among the stages {sel}')
    if args.from_stage:
        sel = sel[sel.index(args.from_stage):]
    if args.until_stage:
        sel = sel[:sel.index(args.until_stage) + 1]
    moving = [s for s in sel if s in ('transfer', 'quarantine')]
    if moving and not args.apply:
        raise ValueError(f'stage(s) {moving} copy or move files: they need --apply')
    return sel


def _cells(args, required=True):
    from exopipe.phase3.core import cells as CC
    if not args.cells:
        if required:
            raise ValueError(f'{args.action} needs --cells (deep cells, e.g. 08h_59MHz)')
        return None
    out = CC.expand_cells(args.cells)
    if getattr(args, 'exclude_cells', None):
        drop = set(CC.expand_cells(args.exclude_cells))
        out = [c for c in out if c not in drop]
    return out


def run(args, run):
    import sys
    from exopipe.phase3.config import CFG
    from exopipe.phase3.core import record
    from exopipe.phase3.core.record import NoInput, Refusal
    from exopipe.phase3.lib.archive import common as K

    args.manifest = args.manifest_list[0] if args.manifest_list else None
    tier = args.tier or 'deep'
    if args.action not in ACTIONS_BY_TIER[tier]:
        print(f'error: archive {args.action} is not available for --tier {tier} (tier {tier}: '
              f'{", ".join(ACTIONS_BY_TIER[tier])})', file=sys.stderr)
        return record.EXIT_ERROR
    statuses = []

    def unit_fn(name, body, describe=None, **meta):
        if run.resume and not run.fresh and not run.dry_run:
            ok, why = run.resume_check(name, None, None)
            if ok:
                u = run.unit(name, action=args.action, tier=tier, **meta)
                u.status, u.reason = 'skipped-valid', why
                statuses.append(u.status)
                print(f'skipped   {name}: {why}')
                return u
        u = run.unit(name, action=args.action, tier=tier, apply=bool(args.apply), **meta)
        res, finished = None, False
        with u:
            res = body(u)
            finished = True
        if run.dry_run and finished and u.status in ('no-input', 'done'):
            u.status, u.reason = 'planned', 'dry run (nothing written)'
        statuses.append(u.status)
        what = describe(res) if (describe and finished and res is not None) else u.reason
        print(f'{u.status:9s} {name}: {what}')
        for n in u.notes[-6:]:
            if '\n' not in n:
                print(f'          {n}')
        return u

    try:
        never = set(CFG.archive.NEVER_WAIVABLE)
        bad = [t for t, _ in run.waivers if t in never or t.startswith('archive.G')]
        if bad:
            raise Refusal(f'never waivable: {bad} (archive.NEVER_WAIVABLE; DESIGN 11.6)')
        args._stages = resolve_stages(args)
        ctx = K.Ctx(args, run)
        where = (f'archive root {ctx.archive_root}{" (production)" if ctx.production_archive else " (MIRROR)"}, generation '
                 f'Run {ctx.gen_run}, ' if args.action in _WRITES_ARCHIVE_TREE + ('audit',) and tier == 'deep' else '')
        print(f'archive {args.action} --tier {tier}: {where}stages {",".join(args._stages)}'
              f'{", DRY RUN" if run.dry_run else ""}; out {ctx.out}')
        if args.apply and not run.dry_run and args.action in _WRITES_ARCHIVE_TREE:
            ctx.guard_writes([ctx.archive_root], f'archive {args.action} --apply')
        _dispatch(args, run, ctx, tier, unit_fn)
    except ValueError as e:
        print(f'error: {e}', file=sys.stderr)
        return record.EXIT_ERROR
    except (Refusal, NoInput) as e:
        kind = 'refused' if isinstance(e, Refusal) else 'no-input'
        with run.unit(f'{args.action}_{tier}_run', action=args.action, tier=tier) as u:
            raise e
        print(f'{kind:9s} {args.action}_{tier}_run: {e}')
        statuses.append(u.status)
    if not statuses:
        print(f'archive {args.action}: nothing selected', file=sys.stderr)
        return record.EXIT_NO_INPUT
    if run.dry_run:
        if 'error' in statuses:
            return record.EXIT_ERROR
        if 'refused' in statuses:
            return record.EXIT_REFUSED
        if all(s == 'no-input' for s in statuses):
            return record.EXIT_NO_INPUT
        return record.EXIT_DONE
    return None


def _dispatch(args, run, ctx, tier, unit_fn):
    from exopipe.phase3.core.record import NoInput
    from exopipe.phase3.lib.archive import common as K
    a = args.action
    if a == 'quarantine':
        from exopipe.phase3.lib.archive import quarantine as Q
        return Q.run_action(ctx, _cells(args), K.stokes_list(args.stokes), args, unit_fn)
    if a in ('sync', 'retire-refused'):
        from exopipe.phase3.lib.archive import sync as SY
        S = SY.legacy_module()
        atts = SY.select(args, _cells(args, required=False), K.stokes_list(args.stokes))
        if not atts:
            raise NoInput(f'{a}: no published attempt selected')
        for att, cell, st in atts:
            if a == 'sync':
                unit_fn(f'sync_{cell}_{st}', lambda u, att=att, cell=cell, st=st: SY.sync_unit(u, ctx, S, att, cell, st, args),
                        describe=lambda c: ', '.join(f'{k} {v}' for k, v in c.items()))
            else:
                unit_fn(f'retire-refused_{cell}_{st}',
                        lambda u, att=att, cell=cell, st=st: SY.retire_refused_unit(u, ctx, S, att, cell, st, args),
                        describe=lambda n: f'{n} refused-night frame(s)' if not isinstance(n, dict) else
                        ', '.join(f'{k} {v}' for k, v in n.items()))
        return None
    if a == 'retire-excluded':
        from exopipe.phase3.lib.archive import retire as R
        return R.run_action(ctx, args, run, unit_fn)
    if a == 'push':
        from exopipe.phase3.lib.archive import push as P
        return P.run_action(ctx, args, run, unit_fn)
    if a == 'audit':
        from exopipe.phase3.lib.archive import push as P
        return P.run_audit(ctx, args, run, unit_fn)
    if a == 'rollback':
        from exopipe.phase3.lib.archive import rollback as RB
        return RB.run_action(ctx, args, run, unit_fn, _cells(args), K.stokes_list(args.stokes))
    if a == 'supersede':
        from exopipe.phase3.lib.archive import snapshot as SN
        return SN.run_supersede(ctx, args, run, unit_fn)
    if a == 'quarantine-rows':
        from exopipe.phase3.lib.archive import snapshot as SN
        return SN.run_rows(ctx, args, run, unit_fn)
    raise ValueError(f'unknown action {a}')
