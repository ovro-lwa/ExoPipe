"""exopipe3 install - gated install of a new deep cell generation into the working tree and the archive (survey /
intermittent; DESIGN 2 row 4, B10). Deep only (refactor-chat decision 6: the sub-LST hand-over tests are masters/frames
content-test classes, the supersede is the common --fresh --supersede).

Unit of work: a deep CELL <LST>_<scope> (04h_Full, 13h_41MHz); --cells takes any item spec (12h_Red, @list,
record:STATUS:STEP@OUT, JOB:STATUS). Not a per-hour step: it runs after `masters` (and `dewarp`) rebuilt a cell's master;
its enqueue stage emits the per-hour items <LST>_<date> to re-run against the rebuilt master (DESIGN 12.2):
    exopipe3 run submit --chain epoch --items @<out>/_records/install/rerun_epochs/<run id>.txt

    exopipe3 install --cells 12h_Red --kind mdewarp --out /fast/gh/parity/install --dry-run
    exopipe3 install --cells 12h_Red --kind mdewarp --stages plan,install,verify,vclean,cards --out /fast/gh/parity/install
    exopipe3 install --cells 12h_Red --stages vclean,cards        (an installed cell: refresh the cleaned V, check the cards)
    exopipe3 install --cells 12h_Red --stages enqueue,cleanup --out /fast/gh/parity/install     (after masters --stages sources)
    exopipe3 install --cells 12h_Red --kind mdewarp --stages recover [--execute]
    exopipe3 install --stages screen-unapplied --table merged_table.csv

Stages (--stages, comma list; default plan,install,verify,vclean,cards with install.RELEASE1_CARDS True (production, ruled),
plan,install,verify with it False):
  plan      resolve the generation, the kind and the rebuild root; evaluate every read-only gate; choose the quarantine
            round and check EVERY destination before staging; print the plan (with --dry-run the step stops here)
  install   one transaction - production's phase3_install.install called IN PLACE (exopipe/phase3/lib/install/install.py):
            staging + md5 (a0), epoch images and the epoch gate (a1, EG, G4), exclusion conformance on the staging (G0),
            the working-tree swap with md5-verified working quarantine (a), stamping + conformance (b), archive quarantine
            (same-filesystem rename, inode/size verified) and publish (c), audit (d), the applied screen (e, kind mdewarp).
            With install.RELEASE1_CARDS (ruled) b also writes the Release 1 header-repair cards and provenance_nights.csv
            (lib/masters/release1_cards) before c publishes them. Always preceded by plan. Not separable inside (no --from/--until).
            MDWSTAT SOURCE (gate MDW-REC in plan, again on the staged masters before b; lib/install/mdwstat): the cell's
            dewarp record (<install.DEWARP_RECORD_ROOT>/_records/dewarp/<cell>.json, meta.mdwstat of the apply of an
            accepted cell or of the record stage of a not-accepted one; meta.last_done when a later rerun did not finish)
            when one exists, only if CURRENT - every master taken has the md5 the record stamped (the plan pre-checks size
            and mtime; the md5 of the staged copies decides) - else refused; install.MDWSTAT_TABLE only without one. A
            record value that differs from the table wins and is noted with the cell and both values.
  verify    the installed cell against every membership and product ruling (membership.verify + verify_products; today
            `python -m exopipe.phase3.membership verify <cell>`, install_cell.sh:23)
  vclean    refresh the installed cell's cleaned Stokes V pair (install_cell.sh:28-34; FIX-20261113-DEEP-REBUILD-CLEANS-V-BEFORE-
            FLUX-APPLY): vclean_masters.refresh_vclean_cell twice under the quarantine round '<round>vclean' - a pair not
            cleaned from the installed V master (VCSRC) is re-cleaned, the old pair quarantined, the new one stamped, carded,
            archived and audited; with install.RELEASE1_CARDS a current pair that still lacks a sourced Release 1 card is
            carded through the same working-copy, quarantine and publish path (no sv2 run); before anything is written the
            cards stage's checks must already pass on every other file of the cell. The quarantine round is the first of
            '<round>vclean', '<round>vclean2', ... with no destination of the cell taken, <round> the round of the newest
            done install or recover record of the cell; a failed or refused first call is not repeated, and the second call
            only checks (gate VCLEAN refuses unless it reports CURRENT). Implies verify.
  cards     the Release 1 header-repair cards of the installed cell (install_cell.sh:35-54): gate CARDS refuses when a
            sourced card is still to add, an archive copy's header differs from its working file, or provenance_nights.csv
            differs from the one derived; gate CARD-GATE = the sibling card-key gate (sibling_card_gate.py): every archived
            FITS of the cell carries every card key its repaired same-named siblings of the band's other LSTs all hold (a cell
            whose MDWSTAT says not dewarped: names compared without '_mdewarp', the dewarp-only cards not required), and
            MDWSTAT / MDWMED1 agree with the MDWSTAT source the install chose (the dewarp record or install.MDWSTAT_TABLE;
            without install in the same invocation it is chosen again and must be the one the install recorded - a record
            without one means the table - current for the masters it installed, and carried by the installed masters'
            headers: gate MDW-REC). Read only. Implies vclean. Refused when install.RELEASE1_CARDS is False.
  enqueue   hand the cell to the re-search (install_cell.sh:27-42): retire its search markers / policy-skips / parks, assign
            both Stokes (node of the first line of assign_<S>.csv), flow-holder map row, ready.txt; write the request
            record and the per-hour re-run items. Needs a record of the cell with V passed, and VCLEAN, CARDS and CARD-GATE
            when install.RELEASE1_CARDS is True. With --out the live
            p3s5_state is NOT written: a mirror <out>/p3s5_state seeded from it is.
  cleanup   remove this round's node-local rebuild scratch of the cell (install_cell.sh:44) - only after a passing verify
            record, only node-local paths of this host, only when paths.PRODUCTION_WRITES and no --out
  Exclusive modes: recover (continue an install interrupted mid-way, states R1/R2 - reports only unless --execute),
  epochs-repair (an installed cell whose archive lacks its epoch images), screen-unapplied (applied=0 screen rows for
  measured cells that were not applied, --table).
  The production sequence install_cell.sh = plan,install,verify,vclean,cards -> masters --stages sources -> enqueue,cleanup.
  A parity run against the 2026-10-13 baseline (parity/production_20261013.cfg, install.RELEASE1_CARDS False) runs the
  default plan,install,verify.

Inputs (read in place, never written): the generation (--gen; default install.MDEWARP_GEN_FMT for --kind mdewarp,
  install.PROMOTED_GEN_FMT for --kind rebuild) with coadd_final/ (incl. loo/) and subtracted/, and beside it
  INSTALL_CONTENT_OK (D1-D6 / promote record) and, for mdewarp, APPLIED_SCREEN.json + the screen it names; the rebuild
  root (--rebuild-root, default install.REBUILD_ROOT_FMT; REBUILD_CONTENT_OK = T1-T8 of check_rebuild.py or Gregg's
  waiver, weights.csv, night_image_gate.csv, stage2_verdict.csv, residuals_provisional.csv, flow/, coadd_provisional/);
  the current working cell (paths.COADD_SB_ROOT/<cell>) and archive cell (paths.PHASE3_ARCHIVE_ROOT, paths.ARCHIVE_RUN);
  data_decisions.csv (core.exclusions / coadd.load_exclusions); the membership tables (solar audit, named out-of-master,
  the archived screen record); the snapshot freeze marker and link tree; for enqueue the re-search state
  (install_enqueue.SEARCH_STATE_ROOT) and the site key FLOW_HOLDER.
Outputs (product classes): the installed cell <working>/<cell> (installed_cell); the archive cell
  <archive>/Coadd/<run>/<lst>/<scope> + per-night deep files (archive_cell); the attempt manifest
  <out>/_records/install/<cell>.manifest.csv = the plan written before staging + the action-log rows of the attempt
  (install_manifest); the archived screen record <archive>/mdewarp_screens/<run>/mdewarp_screens.csv with the cell's
  applied=1 row and its screen (screen_table, kind mdewarp; dewarp writes the applied=0 rows of the same class); the
  re-search request <records>/<cell>.request.json (research_request); the per-hour re-run items
  <records>/<cell>.rerun_epochs.txt and rerun_epochs/<run id>.txt (rerun_epochs); <records>/<cell>.cleanup.json
  (cleanup_record). Quarantine: working_quarantine_<round>/<cell>/<rel>
  (md5-verified copies) and <archive>/Run_INSTALL_QUARANTINE_<round>/<rel> (renames); the action log
  install.RECORD_CSV (append, unchanged format). With --out every write destination is re-rooted under --out:
  <out>/coadd_sb, <out>/archive, <out>/working_quarantine_<round>, <out>/install_record.csv, <out>/p3s5_state.
Failure modes (exit 2 = refused, the gate named; nothing moves before the last pre-move gate passes):
  H working root not local to this host (site NODE_LOCAL_ROOTS) / --out not writable; R archive run unset; G-1 freeze
  marker absent; G-1b a snapshot_masters link resolves into the archive masters; G1-EXACT / G1 content record absent or
  not naming the generation; CUB not cubic-spline (mdewarp); RB / RB-FILES rebuild record or cell files absent; CELL the
  working cell or its records absent; G-MEM membership violations; G-PROD products lack the ruled flux scale / PB maps /
  cubic dewarp; G1b applied-screen record mismatch; Z no generation file; G2 two files map to one name; G3 space below
  need + install.FREE_FLOOR_GB; STG unfinished staging (use --stages recover); G0-GEN / G0 / b excluded constituent;
  G-DEST a quarantine destination of the named round exists, or (auto) no free round today; a0 staging md5; EG an
  in-master night fails the epoch gate; G4 an epoch fails the epoch content check; V the installed cell violates a
  ruling; VCLEAN the cleaned V pair not CURRENT after the refresh; CARDS / CARD-GATE the Release 1 cards incomplete or
  a sibling's card key missing; Q held (install_enqueue.HOLD_LIST). A failure AFTER the first move is an error (exit 1) naming
  `--stages recover` (rebuilt cells: manual repair). Waivable with --waive install.<GATE> --ruling "...": H, G1-EXACT, Q
  only (the other gates run inside phase3_install during the refactor window, or are never waivable: G0, b, a0, SET,
  AQ, d, e, G-1b, G-DEST, VCLEAN, CARDS, CARD-GATE). Exit 3 = no cell selected / a report-only recover.
"""
import os

SUMMARY = 'gated install of rebuilt deep products into the working tree; quarantine round; re-run list'
TIERS = ('deep',)
SELECTORS = ('tier', 'cells', 'exclude_cells')
CONFIG = ('install', 'install_enqueue', 'dewarp', 'masters', 'masters_admit')    # dewarp.RESAMPLE_METHOD is checked against the in-place
#                                                                  cubic gate; masters: membership rulings (G-MEM, V);
#                                                                  masters_admit: ADMITTED_NIGHTS (D-ADMIT)
WRITES = True

STAGES = ('plan', 'install', 'verify', 'vclean', 'cards', 'enqueue', 'cleanup')
MODES = ('recover', 'epochs-repair', 'screen-unapplied')
DEFAULT_STAGES = ('plan', 'install', 'verify', 'vclean', 'cards')       # install.RELEASE1_CARDS True (production, ruled)
DEFAULT_STAGES_NO_CARDS = ('plan', 'install', 'verify')                 # install.RELEASE1_CARDS False (the parity file)


def default_stages():
    """The default --stages: one switch, install.RELEASE1_CARDS, decides whether vclean and cards are in it (read when the
    step runs, after --config / --set)."""
    from exopipe.phase3.config import CFG
    return DEFAULT_STAGES if CFG.install.RELEASE1_CARDS else DEFAULT_STAGES_NO_CARDS


def add_arguments(p):
    g = p.add_argument_group('install')
    g.add_argument('--stages', default=None,
                   help=f'comma list of {", ".join(STAGES)} (default {",".join(DEFAULT_STAGES)} with install.RELEASE1_CARDS '
                        f'True, {",".join(DEFAULT_STAGES_NO_CARDS)} with it False); or ONE mode of {", ".join(MODES)}. '
                        f'install implies plan.')
    g.add_argument('--kind', choices=('mdewarp', 'rebuild'),
                   help='kind of generation (required for plan, install, recover): mdewarp = the RBF master-dewarp '
                        'generation (names carry _mdewarp); rebuild = a rebuilt cell in canonical names (promoted)')
    g.add_argument('--gen', help='generation directory holding coadd_final/ and subtracted/; may hold {cell} (default '
                                 'install.MDEWARP_GEN_FMT or install.PROMOTED_GEN_FMT)')
    g.add_argument('--rebuild-root', help='the rebuild\'s cell directory (REBUILD_CONTENT_OK, cell files); may hold {cell} '
                                          '(default install.REBUILD_ROOT_FMT)')
    g.add_argument('--not-rebuilt', action='store_true',
                   help='install an mdewarp generation of a cell that was not rebuilt (no rebuild root; membership is '
                        'verified on the installed cell)')
    g.add_argument('--round', help="quarantine round: 'auto' (default install.ROUND: the install chooses "
                                   "'<UTC date><suffix>', the first round whose every destination is free) or a NAME "
                                   "(today's behaviour: refused before staging when any destination of the cell exists)")
    g.add_argument('--omit-vclean', action='store_true',
                   help="do not install the generation's cleaned-V products; the current ones are quarantined "
                        "(R-VCLEAN-REPAIR-DEFERRED)")
    g.add_argument('--execute', action='store_true', help='with --stages recover: act (default: report the state only)')
    g.add_argument('--table', help='with --stages screen-unapplied: the merged dewarp measurement table')


def default_out(args):
    from exopipe.phase3.config import CFG
    return CFG.paths.COADD_SB_ROOT


def parse_stages(text):
    """The ordered stage list, or one exclusive mode. Refuses unknown names and mixtures."""
    from exopipe.phase3.core.record import Refusal
    req = [s.strip() for s in (text or '').split(',') if s.strip()]
    if not req:
        raise Refusal('--stages is empty')
    unknown = [s for s in req if s not in STAGES + MODES]
    if unknown:
        raise Refusal(f'--stages: unknown {unknown}; stages {", ".join(STAGES)}; modes {", ".join(MODES)}')
    modes = [s for s in req if s in MODES]
    if modes:
        if len(req) != 1:
            raise Refusal(f'--stages {text}: a mode ({", ".join(MODES)}) runs alone')
        return modes
    if 'install' in req and 'plan' not in req:
        req.append('plan')                       # the plan always runs first (G-DEST, round, manifest before staging)
    if 'cards' in req and 'vclean' not in req:
        req.append('vclean')                     # production's order: verify, cleaned-V refresh, cards (install_cell.sh:25-53)
    if 'vclean' in req and 'verify' not in req:
        req.append('verify')
    return [s for s in STAGES if s in req]


def run(args, run):
    from exopipe.phase3.core.record import Refusal
    from exopipe.phase3.lib.install import steps_deep
    if getattr(args, 'tier', 'deep') != 'deep':
        raise Refusal(f'install is deep only (refactor-chat decision 6); --tier {args.tier}')
    stages = parse_stages(args.stages or ','.join(default_stages()))
    out = args.out or default_out(args)
    if stages != ['screen-unapplied'] and not args.cells:
        from exopipe.phase3.core.record import NoInput
        raise NoInput('no cells selected: give --cells (e.g. 12h_Red, @list)')
    return steps_deep.run(args, run, stages, os.path.abspath(out))
