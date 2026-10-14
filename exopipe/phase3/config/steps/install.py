"""Configuration of `exopipe3 install` - every tunable of the command, one section per stage or tier.

Units folded here: install. A key in its section's PROTECTED_KEYS is locked
(a mirror of in-place code, a pinned input or a ruling-fixed value: it changes only by a commit); RULED names the
ledger ID of the ruling behind a key. Per run: --set SECTION.KEY=VALUE or --config FILE; `exopipe3 config --step install` lists the sections.
"""
from exopipe.phase3.config import _Section


# =====================================================================================================================
# unit install (was exopipe/phase3/_cfgdraft_install.py)
# Configuration of `exopipe3 install` (deep only - refactor-chat decision 6, crosscheck group 13):
# sections install and install_enqueue.
#
# Spec plan/specs/install.md section 3. Every value keeps the production value of the baseline and names its source
# (file:line) and ruling. Host names are not here: the flow holder and the node-local roots come from the site file
# (FLOW_HOLDER, NODE_LOCAL_ROOTS).
#
# Shared roots are NOT copied here (decision 3): the working tree is CFG.paths.COADD_SB_ROOT, the archive
# CFG.paths.PHASE3_ARCHIVE_ROOT, the archive generation CFG.paths.ARCHIVE_RUN, the ops root CFG.paths.OPS_ROOT. The step
# refuses an override of those four (an override of the working root would otherwise move the default --out away from the
# production guard while the archive stayed production).
#
# WINDOW NOTE: during the refactor window the install itself is the production module called in place
# (exopipe/phase3/lib/install/install.py = phase3_install.py, "PI"). PI hard-codes some values below (marked 'PI literal');
# the step checks at start that each still equals PI's literal and refuses an override it cannot honour (until absorption),
# so these values are documented and hashed, not yet levers.
# =====================================================================================================================

class install(_Section):
    """exopipe3 install: gated install of a new deep cell generation into the working tree and the archive (B10;
    exopipe/phase3/phase3_install.py, membership_rebuild_20261007/install_cell.sh). Roots: paths.COADD_SB_ROOT,
    paths.PHASE3_ARCHIVE_ROOT, paths.ARCHIVE_RUN. With --out every write destination is re-rooted under --out (parity,
    scratch): <out>/coadd_sb, <out>/archive, <out>/working_quarantine_<round>, <out>/install_record.csv."""
    PROTECTED_KEYS = ('WORKING_QUARANTINE_FMT', 'ARCHIVE_QUARANTINE_FMT', 'RECORD_CSV', 'FREEZE_MARKER',
                      'SNAPSHOT_MASTERS_LINK_TREE', 'PRODUCTION_LOCK')
    # ---- archive layout (under paths.PHASE3_ARCHIVE_ROOT / paths.ARCHIVE_RUN)
    ARCHIVE_MASTERS_FMT = '{archive}/Coadd/{run}/{lst}/{scope}'          # PI literal (phase3_install.py:375); R-MASTERS-TOP-LEVEL-LIKE-MOSAICS (ledger 507), R-MASTERS-BESIDE-MOSAICS (549)
    ARCHIVE_NIGHT_GLOB = '{archive}/{lst}/*/{run}/{scope}/[IV]/deep/*.fits'  # PI literal; per-night deep files; phase3_install.py:393, :515
    # ---- generations (inputs; read only)
    MDEWARP_GEN_FMT = '/lustre/gh/main/phase3/mdewarp_rbf_gen_20261007/{cell}/gen_mdewarp_rbf'  # dewarp_lst.sh:15 (OUT), :68; install_runner.sh:18, :26
    PROMOTED_GEN_FMT = '/fast/gh/main/phase3/rebuild_20261007_promoted/{cell}'                 # chain_cell.sh:9, :44; promote_rebuild.py:11
    REBUILD_ROOT_FMT = '/fast/gh/main/phase3/rebuild_20261007/{cell}'                          # install_cell.sh:7, :19; check_rebuild.py:26
    GEN_TOKENS = {'mdewarp': '_mdewarp', 'rebuild': ''}   # PI literal; name token stripped at staging (first occurrence): canon(), phase3_install.py:89-90
    SHFLUX_TOKEN = '_shflux'                              # a gen_shflux --gen is REFUSED in the window (PI's G1 and canon() cannot take it); promote_rebuild.py:12, :24 (count=1) after absorption
    CONTENT_RECORD = 'INSTALL_CONTENT_OK'                 # G1: <gen>/../INSTALL_CONTENT_OK must name <gen> (exact line in the step; PI substring); phase3_install.py:209-211
    REBUILD_RECORD = 'REBUILD_CONTENT_OK'                 # rebuilt cell: <rebuild root>/REBUILD_CONTENT_OK (T1-T8 of check_rebuild.py); phase3_install.py:223-225; check_rebuild.py:121-123
    SCREEN_RECORD_JSON = 'APPLIED_SCREEN.json'            # G1b: <gen>/../APPLIED_SCREEN.json; phase3_install.py:245-248
    # ---- gates
    FREEZE_MARKER = '/lustre/gh/main/exopipe_ops/mdewarp_rbf_20261004/SNAPSHOT_FREEZE_CONFIRMED'  # PROTECTED; G-1, phase3_install.py:68, :199-200; R-INSTALL-GATE-AND-SEARCH-PATCH-PATH (1789), GATE-SNAPSHOT-FREEZE-CONFIRMED-WRITTEN (1798)
    REQUIRE_FREEZE_MARKER = True                          # B-14: True = today (marker AND link scan); False = B-14 option (a), effective only after absorption (PI checks the marker unconditionally) and Gregg's answer
    SNAPSHOT_MASTERS_LINK_TREE = '/lustre/pipeline/exopipe/phase3/snapshot_masters'  # PROTECTED pinned input; PI literal; G-1b, phase3_install.py:203-207; FREEZE-SNAPSHOT-MASTERS-GEN4 (1786)
    FREE_FLOOR_GB = 40.0                                  # GB (1e9 bytes) left free on the working-tree filesystem after staging; install_cell.sh:11 (P3_INSTALL_FREE_FLOOR_GB=40; code default 0, phase3_install.py:280); APPROVED-RESUME-SPACE-D6-REVISION (1848)
    STAGING_SPACE_FACTOR = 1.05                           # PI literal; x the generation's bytes needed on the working-tree filesystem; phase3_install.py:279
    EPOCH_MASTER_STEM = {'I': 'I_deep_Taper_R075', 'V': 'V_deep_Taper_R0'}   # PI literal; master of the epoch content check per Stokes; phase3_install.py:326, :468, :579
    OMIT_VCLEAN_PATTERN = 'V_deep_Taper_R0_clean'         # PI literal; --omit-vclean selects these names; phase3_install.py:267, :447; R-VCLEAN-REPAIR-DEFERRED (1807)
    # ---- what moves
    CELL_FILES = ('weights.csv', 'night_image_gate.csv', 'stage2_verdict.csv')   # coadd_final/ of a rebuilt cell; phase3_install.py:57 (stage2_verdict.csv since cbe8e77, 2026-10-08); set on PI for the call
    CELL_TOP = ('residuals_provisional.csv', 'outliers.csv', 'residuals.csv')     # cell directory; phase3_install.py:58; set on PI for the call
    CELL_DIRS = ('flow', 'coadd_provisional')                                     # cell directory; phase3_install.py:59; FINDING-GEN4-FLEET-DID-NOT-SHIP-FLOWS-BACK-RECOVERY-PLANNED (921); set on PI for the call
    ARCHIVE_SKIP_TOKENS = ('_msub',)                      # PI literal; per-night search frames: never quarantined or audited by install; phase3_install.py:393, :515
    AUDIT_CELL_FILES = ('weights.csv', 'night_image_gate.csv', 'epoch_gate.csv', 'stage2_verdict.csv')  # PI literal; + PB_deep_<s>, PB_deep_R075_<s>; phase3_install.py:526 (= coadd_full_pipeline.py:3197)
    STAGING_DIRNAME = '.install_staging'                  # PI literal; <cell>/.install_staging; phase3_install.py:288
    # ---- quarantine (one pair per round)
    ROUND = 'auto'                                        # deep requirement (b) (DESIGN 8; RESULT-12HRED-INSTALLED-AFTER-STALE-STAGING-20261012, ledger 2283); production: env P3_INSTALL_ROUND default '20261004' (phase3_install.py:53), launchers 20261008b / P3_INSTALL_ROUND_REINSTALL (install_cell.sh:9-11); --round NAME = today's behaviour
    ROUND_SUFFIXES = ('', 'b', 'c', 'd', 'e', 'f', 'g', 'h')   # auto = '<UTC YYYYMMDD of the attempt><suffix>', the first whose every destination is free (precedents 20261008b, 20261011b)
    WORKING_QUARANTINE_FMT = '/lustre/gh/main/phase3/working_quarantine_{round}'   # PROTECTED; phase3_install.py:54
    ARCHIVE_QUARANTINE_FMT = '{archive}/Run_INSTALL_QUARANTINE_{round}'            # PROTECTED; phase3_install.py:55, :374 (PI: '<archive>/' + QROOT_NAME)
    RECORD_CSV = '/lustre/gh/main/exopipe_ops/mdewarp_rbf_20261004/install_record.csv'  # PROTECTED; append-only action log (178,867 lines on 2026-10-13); phase3_install.py:56; read by recover (:437)
    # ---- screen record (kind mdewarp): class screen_table (owned by install, crosscheck 4.2)
    SCREEN_DIR_FMT = '{archive}/mdewarp_screens/{run}'    # PI literal; phase3_install.py:131-132; S-2 answer, CROSS_CHAT_COORDINATION.md 2026-10-04
    SCREEN_TABLE = 'mdewarp_screens.csv'                  # PI literal; phase3_install.py:137; read by membership.py:255 (DEWARP_ACCEPTED); CORRECTION-11-ACCEPTED-CELLS-NOT-DEWARPED-WAS-WRONG (2022)
    SCREEN_FILE_FMT = '{lst}_{scope}_warp_screen.fits'    # PI literal; phase3_install.py:151
    SCREEN_STATUS_MAXLEN = 120                            # PI literal; characters of the status column; phase3_install.py:185; dewarp_lst.sh:83 (= dewarp.SCREEN_STATUS_MAX_CHARS)
    # ---- concurrency and I/O
    LANES = 1                                             # installs at a time per output root (lane locks under <out>/_records/install/_locks); install_cell.sh:18-20 (flock); 2 approved by APPROVED-RESUME-SPACE-D6-REVISION (1848), 3 in install_runner.sh:14 (superseded)
    PRODUCTION_LOCK = '/lustre/gh/main/exopipe_ops/membership_rebuild_20261007/.install.lock'   # PROTECTED; the production launchers' one-install-at-a-time lock (install_cell.sh:18-20 `exec 9> $D/.install.lock; flock 9`, shared by chain_cell.sh and dewarp_lst.sh): an install on the PRODUCTION roots also holds it (blocking), so it never overlaps an install_cell.sh install of the same working tree (ledger 1845); never taken under --out. None once no production launcher is left (after cutover)
    MD5_BLOCK_BYTES = 4 * 2 ** 20                         # bytes per md5 read in the step's content tests; phase3_install.py:74 (no effect on results)
    MD5_THREADS = 8                                       # parallel md5 reads in the install_manifest / archive_cell content tests (I/O bound; no effect on results)
    RULED = {'FREE_FLOOR_GB': 'APPROVED-RESUME-SPACE-D6-REVISION',
             'LANES': 'APPROVED-RESUME-SPACE-D6-REVISION',
             'REQUIRE_FREEZE_MARKER': ('GATE-SNAPSHOT-FREEZE-CONFIRMED-WRITTEN', 'R-INSTALL-GATE-AND-SEARCH-PATCH-PATH'),
             'OMIT_VCLEAN_PATTERN': 'R-VCLEAN-REPAIR-DEFERRED',
             'CELL_DIRS': 'FINDING-GEN4-FLEET-DID-NOT-SHIP-FLOWS-BACK-RECOVERY-PLANNED'}


class install_enqueue(_Section):
    """exopipe3 install --stages enqueue,cleanup: hand an installed cell to the re-search (install_cell.sh:27-44) and emit
    the per-hour items (<LST>_<date>) to re-run against the rebuilt master (DESIGN 12.2). The state files are the live
    state of p3s5_search_fleet.sh: with --out the step writes a mirror (<out>/p3s5_state) seeded from the live files."""
    PROTECTED_KEYS = ('SEARCH_STATE_ROOT',)
    SEARCH_STATE_ROOT = '/lustre/gh/main/exopipe_ops/p3s5_state'   # PROTECTED; install_cell.sh:7 (S) (= <paths.OPS_ROOT>/p3s5_state)
    STOKES = ('V', 'I')                                   # both Stokes re-searched, in this order; install_cell.sh:33; R-RBF-INSTALL-AND-FULL-DEEP-RESEARCH-DECISIONS (1775)
    READY_LIST = 'ready.txt'                              # install_cell.sh:41; read by p3s5_search_fleet.sh:251
    ASSIGN_FMT = 'assign_{stokes}.csv'                    # install_cell.sh:36; p3s5_search_fleet.sh:5, :269
    ASSIGN_NODE_RULE = 'first-line'                       # the node of the FIRST line of assign_<S>.csv (install_cell.sh:36; implicit choice, B-48); the only rule implemented
    MARKERS_FMT = 'markers_{stokes}'                      # install_cell.sh:34
    PARKED_FMT = 'parked_{stokes}'                        # install_cell.sh:35
    RETIRED_FMT = 'markers_retired_{stokes}'              # install_cell.sh:33-35; names <x>.marker.<UTC>, <cell>.park.<UTC>
    FLOW_HOLDER_MAP = 'cell_node_map.csv'                 # install_cell.sh:40; RESULT-P3S5-I-SEARCH-UNSTALLED-21-STALE-SCRATCH-REMOVED (2145)
    FLOW_WORK_FMT = '/fast/gh/main/phase3/s2work_p3s5/{cell}'   # install_cell.sh:40 (host from the site key FLOW_HOLDER; today lwacalim07)
    HOLD_LIST = None                                      # twopass_hold.csv (install_cell.sh:27) retired by Gregg 2026-10-10 '(a)' (R-GREGG-20261010-SECOND-PASS-A-17H18-T1-23HGREEN-PAGES, 2122); the file holds only its header, so None changes nothing. A path = cells listed there (first field) are held (refused, gate Q)
    REMOVE_ROUND_SCRATCH = True                           # install_cell.sh:44 (rm -rf R/<cell> P/<cell>); only node-local paths of this host (site NODE_LOCAL_ROOTS), only after a passing verify record, only when paths.PRODUCTION_WRITES and no --out
    # per-hour re-run list (new, DESIGN 12.2 'the re-run of every hour of a cell against its rebuilt master'):
    # 'searchable' = nights of the installed weights.csv with a residual, admitted by data_decisions (core.exclusions), not
    # refused by the night-image or epoch gate (the nights the search admits for this cell); 'all' = every night of
    # weights.csv or with a residual, excluded nights still left out. No ruling yet (open question in the impl note).
    RERUN_NIGHTS = 'searchable'
    RULED = {'STOKES': 'R-RBF-INSTALL-AND-FULL-DEEP-RESEARCH-DECISIONS',
             'HOLD_LIST': 'R-GREGG-20261010-SECOND-PASS-A-17H18-T1-23HGREEN-PAGES',
             'FLOW_HOLDER_MAP': 'RESULT-P3S5-I-SEARCH-UNSTALLED-21-STALE-SCRATCH-REMOVED'}
