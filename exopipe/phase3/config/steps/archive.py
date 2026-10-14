"""Configuration of `exopipe3 archive` - every tunable of the command, one section per stage or tier.

Units folded here: archive. A key in its section's PROTECTED_KEYS is locked
(a mirror of in-place code, a pinned input or a ruling-fixed value: it changes only by a commit); RULED names the
ledger ID of the ruling behind a key. Per run: --set SECTION.KEY=VALUE or --config FILE; `exopipe3 config --step archive` lists the sections.
"""
from exopipe.phase3.config import _Section


# =====================================================================================================================
# unit archive (was exopipe/phase3/_cfgdraft_archive.py)
# Config draft (refactor window) for `exopipe3 archive` (spec plan/specs/archive.md section 3): sections archive,
# archive_quarantine, archive_sync, archive_retire, archive_push, archive_snapshot, archive_rollback.
#
# Loaded by config._load_drafts() until the refactor chat folds these classes into config.py at '# @@STEP-SECTIONS@@'.
# Every value keeps the production value of the baseline and names its source (file:line; 'ops/' = /lustre/gh/main/
# exopipe_ops/) and its ruling (fix_ledger.csv ID). Shared roots are NOT copied (refactor decision 3): the archive is
# CFG.paths.PHASE3_ARCHIVE_ROOT, the generation Run CFG.paths.ARCHIVE_RUN, the ops root CFG.paths.OPS_ROOT, the snapshot
# ship root CFG.paths.SNAPSHOT_FLEET_OUT, the sub-LST / snapshot trees CFG.paths.SUBLST_ROOT / SNAPSHOT_*_ROOT. Templates
# below are expanded by the step: {ops} = paths.OPS_ROOT, {state} = archive.STATE_DIR, {root} = the archive root of the run
# (paths.PHASE3_ARCHIVE_ROOT, or --archive-root for a mirror), {run} = the generation Run (paths.ARCHIVE_RUN or
# --generation-run), {round} = the quarantine round.
#
# PROTECTED_KEYS hold two kinds of key: safety rules no override may relax (NEVER_DELETE, SAME_FILESYSTEM_ONLY, ...), and
# values that MIRROR a literal inside a production helper the step runs in place during the window (an override would make
# the step's plan disagree with what the helper does, so it is refused until absorption).
# =====================================================================================================================

class archive(_Section):
    """`exopipe3 archive` - shared values (brief B17; CLAUDE.md 7 and 8). Archive root = paths.PHASE3_ARCHIVE_ROOT, current
    deep generation = paths.ARCHIVE_RUN (Run_20260909_000000; GEN4-UNDAMAGED-CELLS-STAMPED-AND-ARCHIVED-315, ledger 928)."""
    PROTECTED_KEYS = ('SEARCH_FRAME_TOKEN', 'CHECKSUM', 'SAME_FILESYSTEM_ONLY', 'NEVER_DELETE', 'RULING_REQUIRED',
                      'QUARANTINE_COLUMNS', 'PUBLISH_COLUMNS', 'NEVER_WAIVABLE', 'EPOCH_TREE')
    # Per-night product directory. R-P3-STRUCTURE-RATIFIED (397). pkg/phase3/phase3_deep.py:869; coadd_full_pipeline.py:3207, 3269.
    EPOCH_TREE = '{root}/{lst}/{date}/{run}/{scope}/{stokes}/{cadence}'
    # A search frame (and its pbcorr sibling) carries this token; epoch images and LOO masters do not.
    # ops/p3s5_quarantine_frames.py:51; ops/p3s5_archive_sync.py:60, 63.
    SEARCH_FRAME_TOKEN = '_msub'
    # Manifest checksum per tier (manifest compatibility with today's files; B-59 default (b)). md5:
    # p3s5_quarantine_frames.py:21-26, p3s5_archive_sync.py:29-34, restore_refused_frames.py:23-28, push_mosaic_v2_20260929.py:27-32;
    # sha256: snapshot_frametime_ship.sh:36, snapshot_quarantine_wideband_fits.py:24-29.
    CHECKSUM = {'deep': 'md5', 'sublst': 'md5', 'snapshot': 'sha256'}
    CHECKSUM_BLOCK_BYTES = 4 * 2 ** 20     # bytes per read (p3s5_quarantine_frames.py:24 '1 << 22'); no product effect
    # Quarantine = rename on ONE filesystem, inode + size verified, moved back on mismatch; never copy + delete.
    # INCIDENT-QUARANTINE-CROSS-FS-MOVE-TRUNCATED-TWO-COPIES-RESTORED (948). p3s5_quarantine_frames.py:62-65.
    SAME_FILESYSTEM_ONLY = True
    # No action deletes an archive file. CLAUDE.md 7 ("deletion needs Gregg's explicit 'delete'");
    # R-GENERATIONS-KEPT-DELETION-ONLY-ON-REQUEST-LEGACY-LABELS-REQUESTED (872); R-HOLD-SUPERSEDED-MASTER-DELETION-UNTIL-
    # VALIDATION-PROCESS-DEFINED (1552). The step has no delete code path; a failed copy removes only its own '.part'.
    NEVER_DELETE = True
    # Actions whose apply needs --ruling "<Gregg's words, UTC, ledger ID>" with a ledger ID found in paths.FIX_LEDGER (gate
    # G10). CLAUDE.md 8 ("Gregg's go for anything touching archives"); FINDING-EXCLUDED-NIGHTS-ARCHIVE-PRODUCTS-NOT-RETIRED-
    # 20261012 (2278, "awaiting Gregg go"); FIX-REFACTOR-MEDIUM-DEFECTS-M1-M19-DEEP-20261013 (2309, M10 deferred);
    # R-RESTORE-REFUSED-CELL-FRAMES-APPROVED (2093, one-off scope). 'sync:retire' = a sync whose plan holds retire rows.
    RULING_REQUIRED = ('retire-excluded', 'retire-refused', 'rollback', 'sync:retire')
    # Tests and gates no --waive can cover (DESIGN 11.6: an excluded constituent, a digest mismatch are never waivable).
    NEVER_WAIVABLE = ('retirement_audit.no_excluded_product_left', 'retirement_audit.no_refused_night_frame_left',
                      'quarantine_manifest.rows_quarantined', 'archive_manifest.rows_transferred',
                      'restore_manifest.rows_restored', 'row_quarantine_manifest.rows_moved_with_manifest',
                      'archive_sync_record.archive_equals_attempt')
    # Deep search-fleet state directory: the frame-quarantine / publish manifests live here today (untracked by git).
    # p3s5_archive_sync.py:26 (env P3S5_STATE); p3s5_quarantine_frames.py:18.
    STATE_DIR = '{ops}/p3s5_state'
    FRAME_QUARANTINE_MANIFEST_DIR = '{state}/frame_quarantine'   # p3s5_quarantine_frames.py:18; restore_refused_frames.py:33; p3s5_archive_sync.py:74
    FRAME_PUBLISH_MANIFEST_DIR = '{state}/frame_publish'         # p3s5_archive_sync.py:74 (never created: sync never applied)
    QUARANTINE_COLUMNS = ('path', 'dest', 'bytes', 'md5', 'cell', 'stokes', 'moved_at')   # p3s5_quarantine_frames.py:55; p3s5_archive_sync.py:78
    PUBLISH_COLUMNS = ('source', 'dest', 'bytes', 'md5', 'cell', 'stokes', 'action', 'written_at')   # p3s5_archive_sync.py:79
    # A live fleet claim of the (cell, Stokes) = a concurrent writer: an APPLY refuses (gate G12). p3s5_search_fleet.sh:261-266.
    WRITER_CLAIM = '{state}/claims_{stokes}/{cell}/owner'
    # Default --out (run records, plans). NEW (no production value): under a production root, so the window's write guard
    # refuses it - window runs pass --out to their own root.
    RECORDS_ROOT = '/lustre/gh/main/phase3/archive_ops'
    # GB free on the destination filesystem required before a transfer; 0 = no check (today). Lesson of
    # INCIDENT-EPOCH-REPAIR-FILLED-FAST-ON-LWACALIM07-3087-ERRORS (1699).
    FREE_SPACE_FLOOR_GB = 0.0
    # Working directory of the production helpers run in place (the fleet runs them from the production repo:
    # p3s5_launch.sh:23 'cd /lustre/gh/main/exopipe').
    LEGACY_CWD = '/lustre/gh/main/exopipe'
    RULED = {'RULING_REQUIRED': ('FINDING-EXCLUDED-NIGHTS-ARCHIVE-PRODUCTS-NOT-RETIRED-20261012',
                                 'FIX-REFACTOR-MEDIUM-DEFECTS-M1-M19-DEEP-20261013', 'R-RESTORE-REFUSED-CELL-FRAMES-APPROVED'),
             'SAME_FILESYSTEM_ONLY': 'INCIDENT-QUARANTINE-CROSS-FS-MOVE-TRUNCATED-TWO-COPIES-RESTORED',
             'NEVER_DELETE': 'R-GENERATIONS-KEPT-DELETION-ONLY-ON-REQUEST-LEGACY-LABELS-REQUESTED',
             'EPOCH_TREE': 'R-P3-STRUCTURE-RATIFIED'}


class archive_quarantine(_Section):
    """Pre-search quarantine of a deep (cell, Stokes)'s earlier archive search frames (ops/p3s5_quarantine_frames.py, run in
    place by the search fleet, p3s5_search_fleet.sh:194-197). The values marked MIRROR are literals of that helper."""
    PROTECTED_KEYS = ('RUN_PREFIX', 'SELECT_RULE', 'PRECHECK_CELL_ROOTS', 'PRECHECK_CELL_MARKER', 'HELPER')
    HELPER = '{ops}/p3s5_quarantine_frames.py'   # the production helper (subprocess; an in-process import would resolve the worktree's exopipe)
    RUN_PREFIX = 'Run_SEARCH_QUARANTINE_'        # MIRROR; p3s5_quarantine_frames.py:17
    # Round name: files go to <root>/Run_SEARCH_QUARANTINE_<round>/. p3s5_quarantine_frames.py:17 and p3s5_launch.sh:22 default;
    # production launches pass P3S5_QROUND (20261005 cubic round, R-MDEWARP-CUBIC-REAPPLY ledger 1834; 20261012 at the
    # 2026-10-12 21:24 UTC launch, p3s5_state/launch.log). B-55 default (b): the operator names the round (--round).
    ROUND_DEFAULT = '20261004'
    # Selection: archive '_msub' frames of the (cell, Stokes) with mtime < launch epoch (MIRROR; p3s5_quarantine_frames.py:50-51).
    # TODAY'S RULE (deep defect D2: a re-search leaves frames of nights it no longer searches). Only this value is implemented;
    # 'not-written-by-attempt' waits for the deep chat (search spec Q7).
    SELECT_RULE = 'mtime-before-launch'
    LAUNCH_EPOCH_FILE = '{state}/launch_epoch'   # p3s5_launch.sh:16 (written once, reused; 1791427094 s = 2026-10-08T02:38:14Z)
    MEMBERSHIP_PRECHECK = True                   # p3s5_quarantine_frames.py:34-49 (2026-10-08 incident, R-RESTORE-REFUSED-CELL-FRAMES-APPROVED 2093)
    # MIRROR; p3s5_quarantine_frames.py:42-43: P3_COADD_ROOT default (the fleet runs the helper with it unset,
    # p3s5_search_fleet.sh:197-198) then the shipped inputs; a root holds the cell when <root>/<cell>/<marker> exists.
    PRECHECK_CELL_ROOTS = ('/data07/gh/phase3/coadd_sb', '/fast/gh/main/phase3/s2inputs_p3s5')
    PRECHECK_CELL_MARKER = 'coadd_final/weights.csv'
    RULED = {'MEMBERSHIP_PRECHECK': 'R-RESTORE-REFUSED-CELL-FRAMES-APPROVED', 'ROUND_DEFAULT': 'R-MDEWARP-CUBIC-REAPPLY'}


class archive_sync(_Section):
    """Archive search frames = the published attempt's frames (ops/p3s5_archive_sync.py, imported in place; Gregg
    2026-10-04 'yes to both - patch path approved', R-INSTALL-GATE-AND-SEARCH-PATCH-PATH ledger 1789). MIRROR values are
    literals of that module; the step checks them against its source at start."""
    PROTECTED_KEYS = ('MODULE', 'QUARANTINE_RUN', 'SAME_DATA_RTOL', 'SAME_DATA_ATOL', 'ATTEMPT_FRAME_DIR', 'ARCHIVE_NAME_CARD',
                      'ATTEMPT_DATE_CHARS')
    MODULE = 'p3s5_archive_sync'                 # module name under the exopipe_ops legacy root
    QUARANTINE_RUN = 'Run_SEARCH_QUARANTINE_20261004'   # MIRROR; p3s5_archive_sync.py:25 (fixed, unlike the fleet's per-round name)
    # same_data: identical NaN masks and |x - y| <= RTOL |y| + ATOL elsewhere (float64). Measured 2026-10-04 on 14h 59 MHz V
    # (p3s5_archive_sync.py:38-41). MIRROR; p3s5_archive_sync.py:46.
    SAME_DATA_RTOL = 1e-6                # dimensionless
    SAME_DATA_ATOL = 1e-12               # image unit (Jy/beam)
    ATTEMPT_FRAME_DIR = {'V': 'vresid', 'I': 'iresid'}            # MIRROR; p3s5_archive_sync.py:50
    ARCHIVE_NAME_CARD = 'PRODNAME'       # MIRROR; the attempt frame's header names its archive file; p3s5_archive_sync.py:54-57
    ATTEMPT_DATE_CHARS = 10              # MIRROR; characters of the frame basename that give the date; p3s5_archive_sync.py:58
    ATTEMPTS = 'published-harvested'     # last register entry per (cell, Stokes), harvested only; p3s5_archive_sync.py:120-122 (p3s3_sources.published_attempts)
    RUN_LABEL = 'P3S5'                   # P3S_LABEL set before importing p3s3_sources; p3s5_archive_sync.py:120
    RULED = {'ATTEMPTS': 'R-INSTALL-GATE-AND-SEARCH-PATCH-PATH'}


class archive_retire(_Section):
    """retire-excluded / retire-refused (deep requirement (c), DESIGN.md 8; FINDING-EXCLUDED-NIGHTS-ARCHIVE-PRODUCTS-NOT-
    RETIRED-20261012, ledger 2278)."""
    PROTECTED_KEYS = ('MANIFEST_COLUMNS', 'KIND_RULE')
    # Quarantine Run of retire-excluded: proposed in FINDING 2278 (the quarantine_path column of the 2026-10-12 dry run).
    EXCLUDED_QUARANTINE_RUN = 'Run_EXCLUDED_QUARANTINE_20261012'
    # Quarantine Run of retire-refused (M10 proposal "move to the search quarantine") = archive_sync.QUARANTINE_RUN.
    REFUSED_QUARANTINE_RUN = 'Run_SEARCH_QUARANTINE_20261004'
    CADENCES = ('deep', '10min')         # the 2026-10-12 dry run covered both: 108 deep + 1,103 10min rows of 1,211
    # kind = search_frame if archive.SEARCH_FRAME_TOKEN is in the name else epoch_image (the dry run's 'kind' column; its 8
    # LOO masters were listed as epoch_image).
    KIND_RULE = 'msub-token'
    # DRYRUN_excluded_archive_products.csv header (applied manifests add APPLIED_EXTRA_COLUMNS).
    MANIFEST_COLUMNS = ('lst', 'date', 'scope', 'kind', 'size_bytes', 'path', 'quarantine_path', 'listed_utc')
    APPLIED_EXTRA_COLUMNS = ('md5', 'moved_at')
    # Where the 2026-10-12 dry run (the plan awaiting Gregg's go) lives; read only (an apply passes --manifest).
    MANIFEST_DIR = '{ops}/excluded_quarantine_20261012'
    # Report-only trees per tier (no moves; no ruling covers sub-LST or snapshot archive retirement: critique_rulings.md 7
    # item 7). 'paths.X' entries resolve to that shared root.
    REPORT_TREES = {'sublst': ('paths.SUBLST_ROOT',),
                    'snapshot': ('paths.SNAPSHOT_CUBES_ROOT', 'paths.SNAPSHOT_FLOW_ROOT', 'paths.SNAPSHOT_FLEET_OUT',
                                 '/lustre/gh/main/exopipe_ops/snapshot_lightcurves')}   # snapshot light curves (curves.PRODUCT_ROOT)
    RULED = {'EXCLUDED_QUARANTINE_RUN': 'FINDING-EXCLUDED-NIGHTS-ARCHIVE-PRODUCTS-NOT-RETIRED-20261012',
             'REFUSED_QUARANTINE_RUN': 'FIX-REFACTOR-MEDIUM-DEFECTS-M1-M19-DEEP-20261013',
             'CADENCES': ('FINDING-EXCLUDED-NIGHTS-ARCHIVE-PRODUCTS-NOT-RETIRED-20261012', 'LANDED-TENMIN-ARCHIVE-WRITES-STEPS-61-66')}


class archive_push(_Section):
    """push --class C: manifest -> checksum transfer -> content audit (procedure model ops/push_mosaic_v2_20260929.py;
    RULING-LUSTRE-PUSH-GO-recover-local-class-approved-asap, ledger 335; R-PUSH-WHEN-READY, 489)."""
    PROTECTED_KEYS = ('INTERMEDIATE_PATTERN',)
    # One entry per pushable class. 'existing': what a push does with an archive file already at the destination:
    # 'refuse' (never overwrite; push_mosaic_v2:54-56, 66-67) or 'replace-quarantine' (equal checksum or equal pixels = keep;
    # different = the old file goes to the round quarantine with a manifest row, then the new one is renamed in; adding /
    # replacing the pipeline's own archive products is standing-approved, INSTALLED-PRUNED-CLAUDE-MD-AND-MEMORY-INDEX 1677).
    CLASSES = {
        # mosaic v2 Taper push: R-MOSAIC-V2-PUSH-APPROVED-WHEN-DONE (1601), R-JYBEAM-TOKEN-AND-MOSAIC-V2-PUSH-TAPER-ONLY (1617).
        'mosaic': {'enabled': True,
                   'src_globs': ('/fast/gh/main/phase3/mosaic_v2/archive_staging/mosaic/wideband/*.fits',            # push_mosaic_v2:21, 43
                                 '/fast/gh/main/phase3/mosaic_v2/archive_staging/mosaic/renders/wideband/*/*.fits'),
                   'tables': ('/fast/gh/main/phase3/mosaic_v2/qa/flux_anchor_measured_20260929.csv',),               # push_mosaic_v2:22
                   'run_dir': '{root}/mosaic/Run_FEATHER2_{d0}-{d1}',                                                # push_mosaic_v2:23, 35-38
                   'taper_only': True, 'existing': 'refuse'},                                                        # push_mosaic_v2:44-46, 54-56
        # per-hour push of one epoch's products (DESIGN 12.1) and bulk pushes of many epochs: every file under
        # <staging>/<LST>/<date>/<run>/ (a tree that mirrors the archive layout, e.g. the search step's <out>/archive_mirror)
        # -> <root>/<same relative path>. NEW (no production twin: today search writes its archive copies itself,
        # phase3_deep._archive_frame_publish :850-873, with copy2 + os.replace after the fleet's pre-search quarantine).
        'epoch': {'enabled': True, 'tiers': ('deep',), 'existing': 'replace-quarantine'},
        # QA: ARCHIVED-PHASE3-QA-TO-LUSTRE-AND-INDEXED-THE-PER-NIGHT-PHASE2-QA (1494) - one manual copy, no standing push.
        'qa': {'enabled': False, 'dest_root': '{root}/QA'},
        # run catalogues / pages / verdict journals: B17 gap, nothing pushes them today (ops-deep-serve-archive 9.5); no ruled destination.
        'run-tables': {'enabled': False},
    }
    EPOCH_STAGING_ROOT = None            # default --staging-root of push --class epoch; None = audit-only (the archive's own files of the hour)
    # Quarantine Run prefix of files a push --class epoch replaces (the search round's quarantine, like the fleet's
    # pre-search quarantine: p3s5_quarantine_frames.py:17); the round is --round (archive_quarantine.ROUND_DEFAULT).
    EPOCH_REPLACE_QUARANTINE_PREFIX = 'Run_SEARCH_QUARANTINE_'
    # Product class of an archived per-epoch file, by name tokens, first match wins: (token that must be present, second
    # token or None, class). The archived copy is content-tested with the producing step's class test. A search frame and
    # its pbcorr sibling are ONE class, search_frame (lib/frames/tests.py: pbcorr_matches_plain tests the sibling); every
    # class named here must be registered (review 2026-10-13: 'search_frame_pbcorr' was not -> KeyError on every real hour).
    EPOCH_NAME_CLASSES = (('_msub', None, 'search_frame'), ('_loo', None, 'loo_master'), ('', None, 'epoch_image'))
    REFUSE_CHANGED_SOURCE = True         # source size/checksum must equal the manifest at transfer; push_mosaic_v2:62-64
    # Files refused from any push (classification of the broken one-off exopipe/tools/phase3_archive_manifest.py:29-31).
    INTERMEDIATE_PATTERN = r'(ep_.*_up2|_flow\.npy$|^work_|^tmp|\.claim$|\.pybdsf\.log$|\.lock$|\.casalog$|^_|\.tmp$|\.partial$|\.part$)'
    AUDIT_BUNIT = {'jysr': 'Jy/sr', 'jybeam': 'Jy/beam', 'psfmap': 'sr', 'pbmap': ''}   # push_mosaic_v2:77
    AUDIT_PIXEL_STRIDE = 97              # every 97th row probed for a finite value; push_mosaic_v2:103
    RULED = {'CLASSES': ('R-MOSAIC-V2-PUSH-APPROVED-WHEN-DONE', 'R-JYBEAM-TOKEN-AND-MOSAIC-V2-PUSH-TAPER-ONLY',
                         'INSTALLED-PRUNED-CLAUDE-MD-AND-MEMORY-INDEX')}


class archive_snapshot(_Section):
    """Snapshot-tier archive actions: supersede = ops/snapshot_frametime_ship.sh (run in place); quarantine-rows =
    ops/snapshot_quarantine_wideband_fits.py (run in place)."""
    PROTECTED_KEYS = ('SHIP_SCRIPT', 'ROW_SCRIPT', 'SHIP_NEW_GLOBS', 'SHIP_OLD_GLOBS', 'ROW_QUARANTINE_MARK', 'ROW_QUARANTINE_BANDS',
                      'SUPERSEDE_PRESHIP_TESTS')
    SHIP_SCRIPT = '{ops}/snapshot_frametime_ship.sh'
    # MIRROR of the launcher's sequencing (snapshot_frametime_rerun.sh:223-232): the ship runs only when the kept 10-s
    # inputs are unchanged and content tests A-E of snapshot_frametime_content_test.py passed on the work dir (gate G14).
    SUPERSEDE_PRESHIP_TESTS = ('A', 'B', 'C', 'D', 'E')
    FRAMETIME_QUARANTINE_ROOT = '/lustre/pipeline/exopipe/phase3/Run_SNAP_FRAMETIME_QUARANTINE_20261007'   # snapshot_frametime_rerun.sh:70
    PHOT_ROOT = '/lustre/pipeline/exopipe/phase3/photometry'       # snapshot_frametime_rerun.sh:69
    # Scope approval of the frame-time re-run (RULING-GREGG-20261008-FRAMETIME-RERUN-REJECTED-ACCEPT-ERROR-UNTIL-MAJOR-REDO,
    # 2074; only the cells shipped under the pause). Read as today for production roots (B-52 default (b));
    # snapshot_frametime_rerun.sh:80-81 (a test root, FT_TEST_ROOT, skips it: :76-79).
    SUPERSEDE_APPROVAL_MARKER = '{ops}/snapshot_frametime_rerun_APPROVED'
    # MIRROR of the ship script's file classes (snapshot_frametime_ship.sh:18-25): used only by the plan listing.
    SHIP_NEW_GLOBS = ('noise_*_median_1min.csv', 'candidates_*_median_1min.csv', 'events_*_median_1min.csv', 'runinfo_*_median.json',
                      'candidates_{Red,Green,Blue,Full}_[IV]_median.csv', 'events_{Red,Green,Blue,Full}_[IV]_median.csv',
                      'noise_{Red,Green,Blue,Full}_[IV]_median.csv', 'model_{Red,Green,Blue,Full}_[IV]_median.fits', 'wideband_*.txt',
                      'wideband_members_[IV].csv', 'group_[IV].log', 'rerun1min_*.log', 'frametime_1min_refused.csv')
    SHIP_OLD_GLOBS = ('noise_*MHz_[IV]_median_1min.csv', 'candidates_*MHz_[IV]_median_1min.csv', 'events_*MHz_[IV]_median_1min.csv',
                      'runinfo_*MHz_[IV]_median.json', '*_{Red,Green,Blue,Full}_[IV]_median*.csv',
                      'model_{Red,Green,Blue,Full}_[IV]_median.fits', 'runinfo_{Red,Green,Blue,Full}_[IV]_median.json',
                      'wideband_*_[IV]_*.txt', 'wideband_members_[IV].csv', 'group_*.log', 'rerun1min_*.log', 'frametime_1min_refused.csv')
    ROW_SCRIPT = '{ops}/snapshot_quarantine_wideband_fits.py'
    ROW_PYTHON = 'py36'                  # site PYTHON key (development Python 3.6; snapshot_rechain_batch.sh:50 PY)
    ROW_QUARANTINE_MARK = 'identity screen'                         # MIRROR; snapshot_quarantine_wideband_fits.py:21 (FINDING-SNAP-FIT-WIDEBAND-IMAGES-REGISTERED-TWICE..., 1925, Gregg item 22 "Fine")
    ROW_QUARANTINE_BANDS = ('Red', 'Green', 'Blue', 'Full')         # MIRROR; snapshot_quarantine_wideband_fits.py:20
    ROW_QUARANTINE_QUIET_MIN = 10.0                                 # minutes; snapshot_quarantine_wideband_fits.py:35; snapshot_rechain_batch.sh:182
    BATCH_ROOT = '{ops}/snapshot_batch'                             # snapshot_rechain_batch.sh:51; snapshot_rechain_b2_20261007.sh:40
    ROW_QUARANTINE_QDIR = '{batch_root}/fits_quarantine_wideband_double_registered_{name}'   # snapshot_rechain_batch.sh:52; b2:41
    RULED = {'SUPERSEDE_APPROVAL_MARKER': 'RULING-GREGG-20261008-FRAMETIME-RERUN-REJECTED-ACCEPT-ERROR-UNTIL-MAJOR-REDO',
             'ROW_QUARANTINE_QUIET_MIN': 'FINDING-SNAP-FIT-WIDEBAND-IMAGES-REGISTERED-TWICE-AND-CHAIN-ASYMMETRY-ON-RAW-FRAMES'}


class archive_rollback(_Section):
    """rollback: put quarantined files back from their manifest (ops/membership_rebuild_20261007/restore_refused_frames.py,
    generalised; R-RESTORE-REFUSED-CELL-FRAMES-APPROVED, ledger 2093)."""
    PROTECTED_KEYS = ('RESTORE_COLUMNS',)
    # Run directory that receives a file now occupying an original path (restore_refused_frames.py:16:
    # Run_SEARCH_QUARANTINE_20261008_retired_psfmaster); --occupant-quarantine names it.
    OCCUPANT_QUARANTINE_FORM = '{qrun}_retired_{reason}'
    # restore_refused_frames_<UTC>.csv columns (restore_refused_frames.py:50-51, 63, 69; written there sorted).
    RESTORE_COLUMNS = ('action', 'bytes', 'cell', 'dest', 'md5', 'path', 'restored_at', 'retired_md5', 'retired_to', 'stokes')
