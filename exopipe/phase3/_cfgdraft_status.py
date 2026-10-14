"""Config draft of `exopipe3 status` (refactor window; folded into config.py by the refactor chat).

Sections: status (shared keys), status_report, status_rebuild (NEW: --rebuild-due), status_ledger, status_coverage,
status_conformance, status_qa, status_pages, status_snapshot. Spec: exopipe_ops/refactor_20261012/plan/specs/status.md
section 3. Every value: unit, ruling (ledger ID or document), source file:line of the production value ("ops/" =
/lustre/gh/main/exopipe_ops, "pkg/" = /lustre/gh/main/exopipe/exopipe). Roots shared by several steps are NOT repeated
here (crosscheck decision 3): paths.COADD_SB_ROOT (deep working tree), paths.TRANSIENTS_ROOT, paths.ARCHIVE_RUN,
paths.PHASE1_ROOT, paths.PHASE2_ROOT, paths.PHASE3_ARCHIVE_ROOT, paths.OPS_ROOT, paths.CLV15_ROOT,
paths.SNAPSHOT_FLEET_OUT, paths.SNAPSHOT_REVIEW_ROOT, paths.SUBLST_ROOT; the hub folder is serve.HUB_DIR.
Placeholders expanded by the step: {qa} = status.QA_DIR, {hub} = serve.HUB_DIR, {p2} = paths.PHASE2_ROOT,
{ops} = paths.OPS_ROOT, {root} = paths.COADD_SB_ROOT.
"""
from exopipe.phase3.config import _Section


class status(_Section):
    """`exopipe3 status` - completion, coverage, gaps, exclusion conformance, stale products, rebuild due, served diffs,
    QA tables and hub status pages, for every tier; emits item lists usable as --cells/--epochs @FILE. Writes no science
    product."""
    PROTECTED_KEYS = ('QA_DIR', 'LEDGER', 'PROTECTED_OUTPUTS', 'NEVER_WRITE')
    # Default --out of report / eligibility / batch-list (records, report files, item lists). NEW, no production value;
    # not a production root, so report-only runs pass cli._guard_out in the window.
    REPORT_ROOT = '/lustre/gh/main/exopipe_ops/status_reports'
    # PROTECTED. Phase 3 QA directory. R-PHASE3-QA-CENTRAL-IN-PHASE3-QA-DIR (ledger 1809); ARCHIVED-PHASE3-QA-TO-LUSTRE (1494).
    # ops/build_qa_table.py:24; ops/conformance_check.py:18; ops/solar_contamination_audit.py:29; ops/build_qa_pages.py:23
    QA_DIR = '/lustre/pipeline/exopipe/phase3/QA'
    # PROTECTED. Processing ledger, one row per (LST, night). R-PROCESSING-LEDGER-ON-LUSTRE (ledger 793).
    # pkg/phase3/phase_status.py:9; ops/_build_ledger.py:59; ops/build_qa_pages.py:21
    LEDGER = '/lustre/gh/main/clv15/processing_ledger.csv'
    # Run label of the published deep search (QA table, pages, cell-load) and the sub-LST served cumulative run.
    # ops/build_qa_pages.py:18 and build_data_processing_page.py:12 (setdefault 'P3S5'); build_qa_table.py:155
    SEARCH_LABEL = {'deep': 'P3S5', 'sublst': 'SLSTB'}
    # PROTECTED. Files status writes only through their own stage and only after cutover; in the window every planned
    # output is checked against this list AND paths.PRODUCTION_ROOTS (spec S-G2; cli._guard_out checks --out only).
    PROTECTED_OUTPUTS = ('/lustre/gh/main/clv15/processing_ledger.csv',                    # ops/_build_ledger.py:59
                         '/lustre/gh/main/clv15/RATIFIED_STEPS_REGISTER.csv',             # ops/build_ratified_register.py:14
                         '/fast/gh/main/phase3/qa_assembly_2/.data_decisions.fingerprint')  # phase_status.py:11
    # PROTECTED. Never written by any status stage, window or not: contract 4.2 RATIFIED ("only Gregg writes verdicts",
    # B-38); the installed solar audit is a gate input pinned by masters_build.SOLAR_AUDIT_SHA1 (ledger 2029).
    NEVER_WRITE = ('/lustre/gh/main/clv15/data_decisions.csv',
                   '/lustre/pipeline/exopipe/phase3/QA/solar_contamination_audit.csv')
    # Item kind of --emit lists per tier (DESIGN 11.2; core/cells kinds); --emit-kind overrides per run.
    EMIT_KIND = {'deep': 'cell', 'sublst': 'lst', 'snapshot': 'night_cell'}
    # Exit 2 when a report finds a violation / breach / gap / stale item. Default = today's tools: conformance_check.py
    # exits 0 always (:10); coadd_exclusion_conformance.py:93 exits 1 on a breach (recorded, not inherited).
    STRICT = False
    STALE_CLASSES = ('record', 'search_attempt', 'master', 'frame_time', 'prefit', 'sublst_generation',
                     'qa_night_pages', 'ledger')
    # Roots whose _records/ trees `--stale record` reads, besides REPORT_ROOT, --records and the --out of every job in
    # runner.JOBS_ROOT (job.json step_argv / forward_argv). Production step roots hold no records in the window.
    RECORD_ROOTS = ()
    # Served-diff source statuses (ops/served_change_20261006.py:69 + 'new': a source of B with no member of A).
    SERVED_DIFF_STATUSES = ('unchanged', 'reduced', 'gone', 're-searched cell', 'new')
    # Served catalogue directories of a run per tier ({S} = Stokes). deep: ops/served_change_20261006.py:43;
    # sub-LST: Run_SLSTB tier catalogues (sublst_cumulative.py:107-110).
    SERVED_CATALOG_DIRS = {'deep': ('source_catalogs_{S}',),
                           'sublst': ('source_catalogs_{S}_sb', 'source_catalogs_{S}_wb'),
                           'snapshot': ('source_catalogs_{S}',)}
    # Hub registry tier keys (registry.json rows) whose 'run' is the run served for the tier (serve owns the registry;
    # status only reads it, B-56).
    REGISTRY_KEYS = {'deep': ('1-hour', '1-hour-multiples'), 'sublst': ('10-min', '10-min-multiples'),
                     'snapshot': ('1-min', '10-sec')}
    REGISTRY_FILE = '{hub}/registry.json'          # ops/build_review_hub.py:12 (serve)
    CELL_LOAD_COLUMNS = ('stokes', 'lst', 'date', 'band', 'detections_5sig', 'served_sources')  # ops/p3s3_pass.sh:217
    CELL_LOAD_FILE = 'cell_load_review_{label}.csv'   # in <out>/status; production writes <run>/cell_load_review.csv
    #                                                  (p3s3_pass.sh:216; R-CELL-LOAD-REVIEW-AFTER-FIRST-ITERATION, ledger 836)
    WORKERS = 8                                       # parallel readers (threads) for wide scans (pure)


class status_report(_Section):
    """The session report: phase_status.py in place + canon check + served runs + re-adjudication list + conformance and
    rulings rows (CLAUDE.md section 1 items 1-5)."""
    PROTECTED_KEYS = ('DECISIONS_FINGERPRINT',)
    DECISIONS_FINGERPRINT = '/fast/gh/main/phase3/qa_assembly_2/.data_decisions.fingerprint'   # phase_status.py:11, 55-61
    # 'read' = compare only (window: never consume the production SessionStart hook's "changed" flag - the production file
    # is copied into <out>/status and phase_status writes the copy); 'write' = today (phase_status.py:58-61 rewrites the
    # production file; only after cutover); 'record' = B-60 (a) (compare with the data_decisions sha1 of the last record).
    FINGERPRINT_MODE = 'read'
    # Deep co-add roots in consumer precedence; phase_status.py:15-18 (ledger 475; R-COADD-ROOT-NO-DEFAULTS, 835). The three
    # /fast roots are absent today; kept so the report says so (B-60 default (b)).
    COADD_ROOTS = ('/fast/gh/main/phase3/rebuild_w2', '/fast/gh/main/phase3/rebuild_v2', '/fast/gh/main/phase3/rebuild_13h',
                   '/data07/gh/phase3/coadd_sb')
    RETIRED_COADD = '/fast/gh/main/phase3/coadd'   # warned about if it exists; phase_status.py:19, 120-122
    CANON_CHECK = 'bash /lustre/gh/main/exopipe_ops/canon_check.sh'   # kept in place (MOVE_PLAN keep); clv15/.claude/settings.json:12
    CANON_TIMEOUT_S = 30                           # s (pure)
    REGISTER = '/lustre/gh/main/clv15/RATIFIED_STEPS_REGISTER.csv'   # read for item 5; ops/build_ratified_register.py:14
    REGISTER_PROBLEM = ('NOT-ENFORCED', 'SIDE-SCRIPT-ONLY', 'DATA-FILE-ONLY', 'DOC-CODE-DISCREPANCY', 'UNCLEAR')  # :17
    # Re-adjudication list (CLAUDE.md 1.3; contract 4.4; READJUDICATION-LIST-EMPTY-20260910/-0911/-0916, ledger 942/975/1225):
    # every all- or partial-excluded (LST, night) whose p1_peel or p2 state advanced since the previous report's state
    # (the previous summary record; first run: the newest production QA/readjudication_list_*.csv). Columns of the last
    # list written, QA/readjudication_list_20261007.csv.
    READJUDICATION_COLUMNS = ('lst', 'date', 'dd_status', 'excluded_subbands', 'reason', 'p1_peel', 'p2', 'go_no_go')
    READJUDICATION_GO = 'for Gregg (reprocessed, outcome not measured)'   # go_no_go text of the 2026-10-07 list
    READJUDICATION_GLOB = '{qa}/readjudication_list_*.csv'


class status_rebuild(_Section):
    """`status --rebuild-due` (NEW; PHASE3_ARCHITECTURE.md section 0, C-10: "+50% accumulated observing time since the last
    build"; DESIGN 12.2, Gregg 2026-10-13 "if the amount of images on disk increases substantially (say by 50%)").
    A REPORT, NEVER A TRIGGER: D-ADMIT (R-GREGG-20261013-REFACTOR-SHEET-DADMIT-B01-B02-B03-B05-B30-B32, ledger 2335):
    "the master rebuild is NOT triggered automatically at +50 per cent admitted time - that is a guideline for the cadence;
    the user invokes the rebuild". Nothing reads the table or its item list as an input; the user passes the list to
    `exopipe3 masters --cells @due.txt` when they choose to rebuild.

    HOW ACCUMULATED TIME IS MEASURED (deep cell = <LST>_<scope>; recorded in every row and in the unit record):
      t(night)   = TIME_CARD (INTTIME, s) of the night's Phase 2 deep image of the cell's scope, survey variant IMAGE_GLOB
                   (Stokes I, Taper, Robust -0.75; dewarped, else its plain parent) - the quantity the master's own INTTIME
                   card sums over its constituents (measured 2026-10-13: 04h_Full master 68,220.831 s = the sum over its 19
                   constituent nights; subband masters carry a differently summed INTTIME, e.g. 13h_82MHz 53,756 s against
                   56,304 s from its 16 constituent 82 MHz images, so the master card is shown, not used).
                   A build night with no image of the variant in its sci_dir had no data of the scope at the build: t = 0
                   for the build (listed in build_without_image).
      build set  = the nights the installed master's build ADMITTED: the rows of <COADD_SB_ROOT>/<cell>/coadd_final/
                   BUILD_RECORD (weights.csv: every night the build considered after exclusions, in_master 0 or 1), each
                   measured in the Science run the build used (its sci_dir).
      T_build    = sum of t over the build set (QA-rejected nights included: they were on disk at the build).
      new set    = nights ADMITTED since the build and not in the build set. NEW_SET_SOURCE says where they come from:
                   'on-disk' (the refactor window; D-ADMIT's per-hour admission records do not exist yet): every night
                   with a Phase 2 deep image of the scope (newest Science_* holding one) that core.exclusions does not
                   exclude now; 'admission-record' (once the D-ADMIT admit stage writes ADMISSION_RECORD beside
                   weights.csv, rows date + sci_dir as weights.csv): the record's nights, measured in their sci_dir,
                   that are not excluded now (a cell without the record has none admitted).
      T_new      = sum of t over the new set.   DUE when T_new >= DUE_FRACTION x T_build.
      Nights of the build set that are now excluded are not subtracted (that master is stale by exclusion: --stale master /
      --exclusion-conformance); a cell with no build record and admitted nights is listed 'never-built' (due).
      A night whose image lacks TIME_CARD is 'unmeasured' (listed; the cell row says so; never a default duration)."""
    DUE_FRACTION = 0.5            # dimensionless: +50 per cent of T_build (architecture 0, C-10; ledger DOC 2026-08-11, line 10)
    TIME_CARD = 'INTTIME'         # s; Phase 2 deep image header (TIMEFIX v2-phase1 / v2-subband-solution)
    # Ordered: the first pattern present in the Science run's <scope>/I/deep is the night's image. The dewarped survey
    # image and its plain (not dewarped) parent carry the same INTTIME (one integration set); a night whose Phase 2 dewarp
    # did not run (e.g. 13h 2025-01-08 82 MHz: plain image only) is still observing time on disk.
    IMAGE_GLOB = {'subband': ('{scope}-I-Taper-*s-Robust-0.75-*-image_dewarped.fits',     # Phase 2 subband grammar
                              '{scope}-I-Taper-*s-Robust-0.75-*-image.fits'),
                  'band': ('{scope}_I_deep_Taper_Robust-0.75_dewarped_*_LST{lst}.fits',   # Phase 2 wideband grammar
                           '{scope}_I_deep_Taper_Robust-0.75_*_LST{lst}.fits')}
    BUILD_RECORD = 'weights.csv'  # coadd_final/weights.csv (date, ..., sci_dir, in_master); ops/conformance_check.py:34
    MASTER = 'I_deep_Taper_R075_{scope}.fits'   # reference master (INTTIME, NCOADD cross-check in the content test)
    NEW_RUN_RULE = 'newest'       # Science_* chosen for a new night ('on-disk'): the newest (name order) holding the image
    # HOOK for D-ADMIT (ledger 2335, 'admission record beside weights.csv'): 'on-disk' until the admit stage exists;
    # 'admission-record' reads <cell>/coadd_final/ADMISSION_RECORD (columns ADMISSION_COLUMNS; refused when the key is
    # None or a record lacks a column). No admission record exists on 2026-10-14 (no code writes one yet).
    NEW_SET_SOURCE = 'on-disk'
    ADMISSION_RECORD = None
    ADMISSION_COLUMNS = ('date', 'sci_dir')
    RULED = {'DUE_FRACTION': ('DOC', 'R-GREGG-20261013-REFACTOR-SHEET-DADMIT-B01-B02-B03-B05-B30-B32')}


class status_ledger(_Section):
    """Phase 1/2 inventory and the processing ledger (ops/_inventory_p1p2.py, ops/_build_ledger.py; run in place, with
    their literal output paths redirected under --out)."""
    BATCH_PROGRESS_DIR = '/lustre/gh/main/clv15/batch_progress_rbf'   # <LST>_<date>.done|.timeout; _inventory_p1p2.py:9
    LEGACY_HANDOFF = '/tmp/_inventory_p1p2.pkl'   # node-local hand-off (edge case E22); _inventory_p1p2.py:102; redirected
    P1_COMPLETE_NSUB = 15        # subbands; p1 'complete'; _build_ledger.py:39
    P2_COMPLETE = (15, 4)        # (subbands, widebands); p2 'complete'; _build_ledger.py:40
    PEEL_FULL_MIN = 15           # peeled subbands for p1_peel 'full'; _build_ledger.py:24, 43
    ALL_EXCLUDED_N = 15          # distinct excluded subbands that make dd_status 'all-excluded'; _build_ledger.py:44
    COLUMNS = ('lst', 'date', 'p1', 'p1_nsub', 'p1_peel', 'p1_npeel', 'p2', 'p2_nsub', 'p2_nwide', 'p2_state',
               'dd_excl_subbands', 'dd_status', 'dd_reason', 'qa_verdict', 'qa_basis', 'p1_missing', 'p2_missing')  # :47-57


class status_coverage(_Section):
    """Coverage / gaps per tier (port of ops/completeness_scan_20260907.py and the census classes of
    ops/completeness_census_20260821.py:49-60, on the current roots)."""
    # A Phase 2 deep product = a non-empty <p2>/<LST>/<date>/<Science_*>/<scope>/I/deep; completeness_scan_20260907.py:22-35
    PHASE2_DEEP_DIR = '{p2}/{lst}/{date}/{run}/{scope}/I/deep'
    PHASE2_RUN_PREFIX = 'Science_'
    # A Phase 3 residual of the night in the cell; completeness_scan_20260907.py:14-20
    RESIDUAL = '{root}/{lst}_{scope}/subtracted/{date}_I_sub_{scope}.fits'
    MASTER = '{root}/{lst}_{scope}/coadd_final/I_deep_Taper_R075_{scope}.fits'   # census 'never-attempted' test (:46)
    # Census classes (COMPLETENESS-CENSUS-EXECUTED-manifest-ready, ledger 318); completeness_census_20260821.py:9-14, 49-60
    CENSUS_CLASSES = ('ok', 'recover-local', 'backfill-stageb', 'never-attempted', 'excluded', 'partial-exclusion')
    # 'core' = core/exclusions (per-band rule 4.1b; CORRECTION-COMPLETENESS-495-WAS-MOSTLY-A-SCAN-ARTEFACT, ledger 816, 817).
    # 'exact-row' (completeness_scan_20260907.py:9-12) is not offered: 489 of its 495 gaps were artefacts.
    EXCLUSION_RULE = 'core'
    # Archive columns (epoch image / search frame archived in paths.PHASE3_ARCHIVE_ROOT/<LST>/<date>/<paths.ARCHIVE_RUN>/
    # <scope>/<S>/deep); build_qa_table.py:172-181. False = skip the archive walk (faster; census 'ok' then never assigned).
    ARCHIVE_COLUMNS = True
    # Searched columns from the published attempts of status.SEARCH_LABEL['deep'] (p3s3_sources.published_attempts).
    SEARCHED_COLUMNS = True
    # Report the (LST, night)s whose newest Phase 1 run is newer than the newest Phase 2 Science dir (MEASURED-23-CELLS-
    # PHASE1-RERUN-NEWER-THAN-PHASE2, ledger 2146: "a check the status command could carry"). Report column only.
    P1_NEWER_THAN_P2 = True
    # Sub-LST search: a (cell, Stokes) is DONE by its record, the shipped search log (sublst_search_pass.sh:24 OUT =
    # sublst/search_$ARM, :53 ships <cell>.log only after its acceptance checks :48-52), never by the state dir's done/
    # marker (CLAUDE.md: no marker files); the state dir gives only 'failed' / 'claimed' and the marker shown beside it.
    SUBLST_SEARCH_STATE = '{ops}/sublst_search_state_B'      # done/claims/failed/<LST>_<scope>:<S>; ops-sublst.md
    SUBLST_SEARCH_OUT = '{sublst}/search_B'                  # sublst_search_pass.sh:24 (ARM B)
    SUBLST_SEARCH_RECORD = '{cell}_{S}/{cell}.log'           # the shipped search log; sublst_search_pass.sh:53
    SUBLST_SEARCH_DONE_LINE = 'SUBLST-SEARCH-DONE {cell} {S}'   # printed after run_deep_cell returns; :48, checked :52
    SUBLST_WINDOWS = '{sublst}/tenmin_root_x2'               # <LST>_<scope>/frames.csv (catalog.md 4 preflight)
    SUBLST_REQUIRED_COLUMN = 'pos_kind'           # published LST run without it = pre-fitted generation; sublst_cumulative.py:83-86 (444656a)
    SUBLST_REQUIRED_FILE = 'pertie_detections.csv'   # sublst_cumulative.py:79-81 (window tie)
    SNAPSHOT_WORKLIST = '{ops}/snapshot_fleet_worklist_v9_20261005.txt'   # snapshot_batch_audit.py:139 (v9 fleet worklist)


class status_conformance(_Section):
    """Membership conformance of the deep masters (ops/conformance_check.py in place) and exclusion conformance of every
    tier."""
    LATEST = '{qa}/membership_conformance_latest.csv'     # conformance_check.py:19
    HISTORY = '{qa}/membership_conformance_history.csv'   # conformance_check.py:20, 43-50
    PAGE = '{hub}/conformance.html'                       # conformance_check.py:21
    ROUND_LIST = '/lustre/gh/main/exopipe_ops/membership_rebuild_20261007/rebuild_list.txt'   # conformance_check.py:22
    VIOLATION_CHARS = 300        # characters of first_violation; conformance_check.py:39
    HISTORY_SHOWN = 30           # history rows on the page; conformance_check.py:51
    COLUMNS = ('cell', 'status', 'n_violations', 'in_rebuild_list', 'first_violation', 'checked_utc')   # :38-39
    # Whole-archive master audit (owned by verify; status calls it in place): GATE-BUILT-COADD-EXCLUSION-CONFORMANCE (1535)
    MASTERS_ARCHIVE_ROOT = '/lustre/pipeline/exopipe/phase3/Coadd/Run_20260909_000000'   # coadd_exclusion_conformance.py:26
    MASTERS_EXCL_TABLE = 'coadd_exclusion_conformance.csv'   # under <out>/QA; :27 (env CONF_OUT)
    DD_HISTORY_GLOB = '/lustre/gh/main/clv15/data_decisions.csv.bak_2026*'   # BREACH vs STALE snapshots; :28
    EXCLCONF_TABLE = 'exclusion_conformance_{tier}.csv'     # NEW, under <out>/QA: served / run rows on excluded cell-bands
    # Working-tree masters audited too (NEW): <COADD_SB_ROOT>/<cell>/coadd_final/<WORKING_MASTERS> HISTORY 'COADD
    # constituent' lines, judged with core.exclusions (BREACH / STALE by the DD_HISTORY_GLOB snapshots, same rule as :47-80).
    WORKING_MASTERS = ('I_deep_Taper_R075_{scope}.fits', 'V_deep_Taper_R0_{scope}.fits')
    # S-G6 (window): the production module the branch membership module must equal but for import lines and the
    # NAMED_OUT_OF_MASTER path line (conformance_check.py imports exopipe.phase3.membership -> branch lib/masters).
    PRODUCTION_MEMBERSHIP = '/lustre/gh/main/exopipe/exopipe/phase3/membership.py'
    EXCLCONF_SKIP_DIRS = ('judgements',)  # snapshot_serve_exclusion_gate.py:65-66
    EXCLCONF_SKIP_FILES = ('serve_exclusion_refusals.csv',)   # :65


class status_qa(_Section):
    """QA tables and plots (`status qa`). Each runs the production script in place (section 7.2 of the spec)."""
    STAGES_DEFAULT = ('qa-table', 'solar')
    QA_TABLE = 'qa_all_phases.csv'                         # under <out>/QA; build_qa_table.py:24
    QA_ASSEMBLY = '/fast/gh/main/phase3/qa_assembly'       # read by build_qa_table.py:25 (lwacalim07 /fast)
    ARCHIVE_RUN_ENV = 'P3_ARCHIVE_RUN'                     # build_qa_table.py:134 env name, set from paths.ARCHIVE_RUN
    SEARCH_LABEL_ENV = 'P3_QA_SEARCH_LABEL'                # build_qa_table.py:155 env name, set from status.SEARCH_LABEL
    SOLAR_CANDIDATE = 'solar_contamination_audit.candidate_{utc}.csv'   # NEW: never over the installed table (S-G4)
    SOLAR_INSTALLED = '{qa}/solar_contamination_audit.csv'              # read for the diff (NEVER_WRITE)
    # The solar rule constants below MIRROR the literals of the in-place script (used only by the content test as its
    # oracle; the test refuses when the script's S_WARN / S_BAD / LOW / REF differ from these). solar_contamination_audit.py:
    SOLAR_SUBBANDS_MHZ = (18, 23, 27, 32, 36, 41)          # MHz judged; :26 (LOW)
    SOLAR_REF_MHZ = 46                                     # MHz reference band; :27 (REF)
    SOLAR_S_WARN = 3.0                                     # dimensionless: 'suspect'; :28
    SOLAR_S_BAD = 10.0                                     # dimensionless: 'QUARANTINE'; :28
    SOLAR_REF_WINDOW = (0.5, 2.5)                          # R_46 range for judging on S_f; :68
    SOLAR_CAUSE_DEG = (-12.0, -18.0)                       # deg: 'solar' above -12, 'twilight' above -18; :89
    BEAM_OUT = 'beam_audit'                                # under <out>/QA; beam_audit.py:25 (argv[1])
    BEAM_NPROC = 16                                        # processes; beam_audit.py:26 (pure)
    BEAM_RATIO_FLAG = 1.2                                  # mirrors beam_audit.py:80 (content-test oracle of the flagged file)
    NOISE_PLOT = 'median_noise_vs_freq_all_lsts_v2.png'    # under <out>/QA; as run 2026-09-26 (e52f6c5)
    NOISE_PX = (2040, 900)                                 # px = figsize (17.0, 7.5) in x dpi 120; qa_noise_vs_freq_plot.py:47, 71
    # ---- outliers (pkg/../tools/propagate_date_outliers.py:17-51, selection only; its writer :52-59 is NOT ported) ----
    OUTLIER_MIN_FRAC = 0.60                                # fraction of cells rejecting the date; :12
    OUTLIER_MIN_CELLS = 5                                  # cells with a record of the date; :13
    OUTLIER_SUBBANDS = ('18MHz', '23MHz', '27MHz', '32MHz', '36MHz', '41MHz', '46MHz', '50MHz', '55MHz', '59MHz', '64MHz',
                        '69MHz', '73MHz', '78MHz', '82MHz')   # :16 (rows proposed per leaked LST)
    OUTLIER_PROPOSALS = 'date_outlier_proposals.csv'       # NEW under <out>/QA: proposals for Gregg (S-G3)
    # ---- bandpass (bandpass_phase_warp.py + run_bandpass_phase_qa.sh) ----
    BANDPASS_TABLES_GLOB = '/lustre/pipeline/calibration/results/*/*/successful/*/tables/*.B.flagged'   # run_bandpass_phase_qa.sh:7
    BANDPASS_SHARDS = 16                                   # run_bandpass_phase_qa.sh:9 (pure)
    BANDPASS_SCRIPT = 'bandpass_phase_warp.py'             # ops; run with site PYTHON['casa']
    BANDPASS_QA = 'bandpass_phase_qa.csv'                  # under <out>/QA; run_bandpass_phase_qa.sh:19
    BANDPASS_DECIMALS = 4        # mirrors bandpass_phase_warp.py:112 round(v, 4): the content test's rounding bound on ratio
    # ---- provenance (candidate_provenance_audit.py, perframe_solar_check.py; R-NIGHT-RULE-HORIZON-PLUS-SOLAR-AUDIT 1534) ----
    PROVENANCE_TABLE = 'candidate_provenance_audit.csv'    # candidate_provenance_audit.py:27 (redirected under <out>/QA)
    PERFRAME_TABLE = 'perframe_solar_check.csv'            # perframe_solar_check.py:21 (redirected under <out>/QA)


class status_pages(_Section):
    """Hub status pages (build_qa_pages.py, build_data_processing_page.py, build_ratified_register.py; in place). The hub
    folder is serve.HUB_DIR; status never writes index.html or registry.json (serve)."""
    NIGHT_PAGE = 'nights/{lst}_{date}.html'    # under the hub; build_qa_pages.py:5, 344-345
    FRAGMENT = 'qa_overview_fragment.html'     # build_qa_pages.py:12, 443; read by build_data_processing_page.py:184-186
    DATA_PAGE = 'data_processing.html'         # build_data_processing_page.py:14 (HUB-PAGE-DATA-PROCESSED-PHASES-1-3, 1934)
    REGISTER_PAGE = 'ratified_steps.html'      # build_ratified_register.py:15
    REGISTER_COLUMNS = ('ruling_id', 'date', 'gregg_words', 'step', 'requirement', 'superseded_by', 'enforced_where',
                        'evidence', 'status', 'note', 'source')   # build_ratified_register.py:16
    REGISTER_ORDER = ('NOT-ENFORCED', 'SIDE-SCRIPT-ONLY', 'DATA-FILE-ONLY', 'DOC-CODE-DISCREPANCY', 'UNCLEAR', 'ENFORCED',
                      'SUPERSEDED', 'DRAFT-NOT-RATIFIED', 'OUT-OF-TIER')   # :17-18
    # 'legacy' = today's page text (M17 not applied; build_qa_pages.py:251-256); 'from-record' = M17 after its ruling.
    STAGE3_STATE_TEXT = 'legacy'


class status_snapshot(_Section):
    """Snapshot eligibility audit, batch lists and frame-time staleness (ops/snapshot_batch_audit.py,
    snapshot_batch_lists.py, snapshot_frametime_worklist.py; in place)."""
    FLEET_CLAIMS = '{ops}/snapshot_fleet_claims'           # <cell>/done = shipped; snapshot_batch_audit.py:52, 140
    DEFAULT_WORKLIST = '{ops}/snapshot_fleet_worklist_v9_20261005.txt'   # snapshot_batch_audit.py:139
    ORDER_LIST = '{ops}/snapshot_prefit_worklist_20261008.txt'   # snapshot_batch_lists.py:69
    ELIGIBILITY_TABLE = 'eligibility.csv'                  # under <out>/snapshot (the batch driver writes <L>/eligibility.csv)
    LIST_OK = ('PREFIT-DONE',)                             # snapshot_batch_lists.py:93 (content-test oracle)
    LIST_OK_ALL = ('PREFIT-DONE', 'TO-PREFIT', 'PREFIT-STALE')   # with --all; :93 (content-test oracle)
    # batch-list selection, passed to snapshot_batch_lists.py explicitly (the step's --max-new etc. override one run,
    # recorded as overrides). Values = the script's own argparse defaults, snapshot_batch_lists.py:70-76.
    MAX_NEW = 100                # cells per batch; :70
    MAX_GATE_GB = 300            # GB of chain memory a batch may need (driver model); :71
    LC_CACHE = 4000              # P3_LC_CACHE_MAX the run will use; :72
    BASE_DETECTIONS = 11530      # fit-step detections per base cell without a PREFIT_DONE count (batch 2: 945,417 / 82); :73
    BASE_LOADED = 213270         # loaded detections per base cell without a count (batch 2: 17,487,733 / 82); :74
    NEW_DETECTIONS = 25624       # fit-step detections per new cell without a marker (40-cell sample: 1,024,951 / 40); :75
    NEW_LOADED = 314396          # loaded detections per new cell without a marker (the same sample: 12,575,826 / 40); :76
    STATUS_ORDER = ('NOT-SHIPPED', 'HOLD-DIRTY', 'HOLD-CHAIN-REFUSAL', 'WAIT-FRAMETIME-JOB', 'WAIT-LIGHTCURVES',
                    'WRITER-ACTIVE', 'PREFIT-DONE', 'PREFIT-STALE', 'TO-PREFIT')   # snapshot_batch_audit.py:109-125
    FRAMETIME_THREADS = 16                                 # snapshot_frametime_worklist.py:53 (pure)
