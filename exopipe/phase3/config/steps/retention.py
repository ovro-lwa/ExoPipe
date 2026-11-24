"""Configuration of `exopipe3 retention` (lib/retention/): the two-week retention of the night's raw visibility data."""
from exopipe.phase3.config import _Section


class retention(_Section):
    """Two-week retention of the raw night data (plan/ORCHESTRATION_DESIGN_20261117.md requirement 4, 5.4 and section 6 step 5;
    R-GREGG-20261117-ORCHESTRATION-DESIGN-COMMENTS item (5), Gregg verbatim: "The night time observing will ony be ~10 -12
    hours a night and the data are averaged by 4 channels, so it's <100 Tb for 2 weeks retention."). Each raw hour expires
    RETENTION_DAYS after its PRODUCTION (the time the hour went through the pipeline, recorded in the retention table); an
    hour no production record covers never expires (the backlog's raw data are deleted as they are processed). A held range
    is never deleted. Every deletion is a dry-run manifest, then Gregg's go (a ledger ruling naming that manifest), then a
    delete that re-checks the manifest against the files."""
    # the raw-data tree: <RAW_ROOT>/<subband>/<UT date>/<UT hour>/<YYYYMMDD_HHMMSS>_<subband>_averaged.ms.tar and
    # <...>_flagmeta.bin (10-s sets averaged by 4 channels; owner pipeline); nights before 2026-11 hold each 10-s set as an
    # MS DIRECTORY <...>_averaged.ms/ instead of a .ms.tar (2024-12-29: 359 per subband hour, 121 entries each). Read by
    # Phase 2's find_archive_files_for_subband (exopipe/phase2/pipeline_utils.py, the same literal root).
    RAW_ROOT = '/lustre/pipeline/night-time/averaged'
    SUBBAND_RE = r'^\d+MHz$'            # subband directories (13MHz ... 82MHz: 16 on 2026-11-24, 13 MHz among them)
    DATE_RE = r'^\d{4}-\d{2}-\d{2}$'    # UT date directories
    HOUR_RE = r'^\d{2}$'                # UT hour directories: one ITEM of the retention table = (subband, date, hour)
    STAMP_RE = r'^(\d{8})_(\d{6})_'     # a data entry's start time (UT) from its name; a name without it holds its item
    MS_DIR_RE = r'\.ms$'                # a directory entry named <stamp>_... and matching this is an MS: a data entry whose
                                        # every file and subdirectory the manifest lists and the delete removes one by one
    RETENTION_DAYS = 14                 # days from production to expiry (Gregg 2026-11-17: "2 weeks retention")
    # default --out: the retention table (production records), the holds, the dry-run and applied manifests, the records.
    # A DELETE runs only with --out at this root (realpath; RT-ROOT), so its records and holds are the production ones; a plan
    # may run elsewhere (a what-if), but a delete refuses its manifest (RT-MANIFEST: planned from another table)
    ROOT = '/lustre/gh/main/phase3/retention'
    TABLE_NAME = 'retention_table.csv'  # <out>/<this>: production records, one row per recorded range (append only)
    HOLDS_NAME = 'retention_holds.csv'  # <out>/<this>: holds and their releases (append only)
    MANIFEST_DIR = 'manifests'          # <out>/<this>/DRYRUN_RETENTION_<UTC>.csv (+ .json) and APPLIED_RETENTION_<UTC>.csv
    PLAN_DIR = 'plans'                  # <out>/<this>/RETENTION_TABLE_<UTC>.csv: every item of the tree with its status
    # one dry-run manifest holds at most this many expired items, oldest expiry first (one night is 16 subbands x 10-12 hours
    # = 160-192 items); the rest are listed in the plan as 'expired' for a later manifest. The manifest stats every entry of
    # its items: one listing per item directory and per directory inside its MS directories, never a walk of the tree.
    # Cost on 2026-11-24 (Lustre under the recorders' load): ~150 ms per os.lstat - ~110 s for a .ms.tar hour of one
    # subband (718 entries), ~16 s per 10-s MS directory (~1.6 h for an MS-directory hour of one subband).
    MAX_ITEMS_PER_MANIFEST = 240
    # a real delete needs --ruling-for <DELETE_RULING_KEY>=<ledger ID>: a row of paths.FIX_LEDGER whose fix_id starts with
    # RULING_PREFIX, whose authorized_by begins with the word RULING_AUTHORIZED_BY, whose status begins with
    # RULING_STATUS_PREFIX (any case), and whose text names the manifest's sha256 - in full or its first RULING_SHA_MIN_HEX hex
    # digits or more (never its file name alone: a manifest regenerated or edited under the same name must not match).
    # Gregg's go is for that manifest, never a standing permission. Code cannot prove who wrote a ledger row: the guard is
    # procedural (no session writes an R-GREGG row without Gregg's words) plus these checks
    DELETE_RULING_KEY = 'retention.DELETE'
    RULING_PREFIX = 'R-GREGG-'
    RULING_AUTHORIZED_BY = 'Gregg'      # the first word of authorized_by (the ledger has 'Gregg', 'Gregg Hallinan',
                                        # 'Gregg 2026-11-17', ... in its 250 R-GREGG rows on 2026-11-24)
    RULING_STATUS_PREFIX = 'RULED'      # status begins with this, any case ('RULED - ...', 'ruled'); not 'RECEIVED, NOT RULED'
    RULING_SHA_MIN_HEX = 12             # hex digits of the manifest's sha256 the row must quote, at least
    CHECKSUM = 'identity'               # 'identity': per file size, mtime (ns) and inode, sha256 per item over them (reads no
                                        # data); 'content': also sha256 of every file's bytes (reads all of it: ~45 GB per
                                        # subband hour of .ms.tar sets; a question for Gregg, see the pull request)
    RULED = {'RETENTION_DAYS': 'R-GREGG-20261117-ORCHESTRATION-DESIGN-COMMENTS'}
