"""Configuration of `exopipe3 serve` - every tunable of the command, one section per stage or tier.

Units folded here: serve. A key in its section's PROTECTED_KEYS is locked
(a mirror of in-place code, a pinned input or a ruling-fixed value: it changes only by a commit); RULED names the
ledger ID of the ruling behind a key. Per run: --set SECTION.KEY=VALUE or --config FILE; `exopipe3 config --step serve` lists the sections.
"""
from exopipe.phase3.config import _Section


# =====================================================================================================================
# unit serve (was exopipe/phase3/_cfgdraft_serve.py)
# Configuration of `exopipe3 serve`.
#
# Every value keeps the production value of 2026-10-13 (exopipe_ops HEAD 0c1580c + 6b8a23a M22) and names its source
# file:line and, where one exists, the ledger ID of the ruling that set it (clv15/fix_ledger.csv). Host names live in the
# site file (REVIEW_HOST, CHAIN_HOST); interpreters are the site PYTHON keys 'default' (servers, every in-place script on the
# review host) and 'py36' (the chain host's gate, the forced AD Leo check); the legacy code root (exopipe_ops: the servers'
# cwd and every in-place script) is core.env.legacy_root('exopipe_ops'). Shared roots come from config.paths
# (TRANSIENTS_ROOT for deep / sub-LST runs, SNAPSHOT_REVIEW_ROOT for snapshot runs, SUBLST_ROOT, OPS_ROOT) - never a copy here
# (refactor-chat decisions 2026-10-13, _CROSSCHECK.md top). The review_dir merge belongs to `pages` (decision 6, group 12):
# its environment key is pages.REVIEW_DIR_MERGE, read by serve for the sub-LST cumulative build.
# =====================================================================================================================

class serve(_Section):
    """exopipe3 serve: review pages (classify_transients.py, run in place) REGISTERED on the two-port server, the review hub,
    and the Jupiter pages' own servers, for every tier. Ports (the pages' identities), run roots, titles and the per-tier
    page environment decide what Gregg and the team see.
    Host names live in the site file (REVIEW_HOST, CHAIN_HOST; the hub page prints no tunnel line since exopipe_ops 4c42f91)."""
    # keys no --set / --config can change (DESIGN 11.4; honoured by config/__init__.py)
    PROTECTED_KEYS = ('RUN_ROOT_KEY', 'HUB_DIR', 'LISTS_DIR', 'INSPECTION_GREENLIT', 'EXCLUSION_GATE_TIERS',
                      'OTHER_SERVED_PORTS', 'JUPITER_ROOTS', 'HUB_TIERS', 'HUB_LINKS', 'TWO_PORT_DIR', 'TWO_PORT_REGISTRY',
                      'TWO_PORT_REGISTERED', 'TWO_PORT_GREENLIT', 'TWO_PORT_LOCK',
                      'FORCED_GENERATIONS',          # the samples each cutout generation measured: V2 of its evaluations
                      'SNAPVIEW_TIERS', 'SNAPVIEW_STOKES', 'SNAPVIEW_PAGE_SUFFIX',   # the combined view's layout: what the page code
                                                                                    # reads (mirrors of snapshot_combined_view.py)
                      'SNAPVIEW_FLOOD_FACTOR', 'SNAPVIEW_FLOOD_MIN', 'SNAPVIEW_ALIGN_FAILED', 'SNAPVIEW_ALIGN_HELD',
                      'SNAPVIEW_FP_ALIGN', 'SNAPVIEW_FP_FLOOD',   # the view's default hides: the snapshot chat's ruled content
                      'RUN_VERSIONS_SUFFIX')                                        # the pointer form (lib/serve/versions.py)

    # ---- served runs (production product roots, from config.paths; PROTECTED) ----------------------------------------
    RUN_ROOT_KEY = {'deep': 'TRANSIENTS_ROOT',        # /data07/gh/phase3/transients; p3s3_serve_stokes.sh:9
                    'sublst': 'TRANSIENTS_ROOT',      # sublst_cumulative.sh:15
                    'snapshot': 'SNAPSHOT_REVIEW_ROOT',   # /data07/gh/phase3/snapshot_search/review; snapshot_serve.sh:22
                    'jupiter': None}                  # the Jupiter pages serve JUPITER_ROOTS (no run directory)
    RUN_NAME = {'deep': 'Run_{label}',      # p3s3_serve_stokes.sh:9 (label required: no P3S3 default, edge case E-6)
                'sublst': 'Run_SLSTB',      # sublst_cumulative.sh:15; R-NEW-CHAIN-TAKES-OVER-10MIN-PAGES-CUMULATIVE (ledger 1637)
                'snapshot': '{label}'}      # snapshot_serve.sh:21 (basename of the chain run)
    SNAPSHOT_RUN_PREFIX = 'Run_SNAP'        # snapshot_rechain_batch.sh:49
    SUBLST_LST_RUN = 'Run_SLST{hh}B'        # sublst_cumulative.py:23; sublst_publish_lst.sh:16
    SUBLST_LST_RUN_GLOB = 'Run_SLST[0-9][0-9]B'   # sublst_publish_lst.sh:42; sublst_catalog_publish.sh:96
    SUBLST_BUILD_SUFFIX = '.build'          # sublst_cumulative.sh:15
    # ---- run versions and the pointer (lib/serve/versions.py; plan/CUTOVER_REMAINING_20261124.md row 8, CUTOVER_READINESS.md:
    # 333-344): a served run U is a symbolic link to <dir>/<name><RUN_VERSIONS_SUFFIX>/<vid>/<name>, switched by one atomic rename
    # once the new version is sealed; the sub-LST swap places its build as a new version (production instead renames U to
    # U.prev_<utc>, sublst_cumulative.sh:102, and keeps every previous build)
    RUN_VERSIONS_SUFFIX = '.versions'       # PROTECTED: the pointer form readers recognize; a new value orphans every version
    RUN_VERSIONS_KEEP = 3                   # retention: the newest 3 complete (sealed) versions are kept, and besides them every
                                            # version the pointer or another link names, a held one, one being built and an
                                            # adopted one without its manifest; the others are retired only by `serve --stages
                                            # retire` (a dry run first, then --retire-plan)
    RUN_VERSIONS_MUTABLE = ('judgements*/judgements.jsonl', 'incremental.log', 'page_index/*', 'page_index_build.log',
                            'PUBLISH_IN_PROGRESS.txt*')   # (fnmatch, relative to the run) the review state appended while a
                                            # version is served: left out of its manifest (classify_transients.py:3147 journal,
                                            # common.inc_log, register.lift_hold) - and the page index, which serve --stages
                                            # page-index rebuilds in place on the served version (flow.build_page_index: its
                                            # DIRTY marker, build_detections_index.mark_run_dirty, and its repair); the seal
                                            # still requires the index current (sublst_build.index_check)
    # what a seal hashes and the content test re-reads (lib/serve/versions.py CONTENT): names, sizes, modification times or
    # markers never prove completion; the content test reads content (Gregg's rule; the coordinator, PR 155 review of finding 9)
    RUN_VERSIONS_HASH_SMALL_BYTES = 16 * 1024 * 1024   # bytes (16 MiB): a file up to this size that is not a binary type, and
    RUN_VERSIONS_HASH_TEXT = ('.csv', '.json', '.jsonl', '.html', '.htm', '.txt', '.tsv', '.js', '.css', '.md', '.xml', '.yaml',
                              '.yml', '.log')   # a file of these types whatever its size (queues, catalogs, page indexes,
                                            # pages, records) is hashed at the seal and re-hashed by every content test
    RUN_VERSIONS_HASH_BINARY = ('.png', '.jpg', '.jpeg', '.gif', '.webp', '.pdf', '.npz', '.npy', '.fits', '.fit', '.fz', '.h5',
                                '.hdf5', '.gz', '.zip', '.tar')   # the binaries - these types of any size, and any other file
                                            # over 16 MiB not of a text type: the content test re-hashes their stable sample.
                                            # Run_SLSTB 2026-11-24: 49,593 .png panels (11.859 GB of its 12.215 GB, none over
                                            # 0.83 MiB) and 1 .npz; the rest 0.35 GB in 48,537 files
    RUN_VERSIONS_HASH_LARGE = 'sample'      # the seal hashes of the binaries: 'sample' (their stable sample only) or 'all'.
                                            # 'sample': hashing all of Run_SLSTB's 12.2 GB read about 9 MB/s from /data07
                                            # (4.0 GB in about 7.5 min), about 25 min - over twice its seal walk (76 s). With
                                            # 'sample' a publish of it read 583.8 MB at the seal (48,537 files not binaries and
                                            # 977 of 49,594 binaries: 241.9 s, cold) and the same 583.8 MB in its content test
                                            # (4.1 s, warm): 1.17 GB of its 12.2 GB
    RUN_VERSIONS_SAMPLE_FRACTION = 0.02     # the stable sample of the binaries: those whose path's sha256 falls in the first
    RUN_VERSIONS_SAMPLE_MIN = 20            # 2% of its range, at least 20 files (all when fewer); plus, in the content test,
                                            # every file whose size or modification time differs from the manifest
    RUN_VERSIONS_LEASE_S = 600              # s: a holder's or builder's lease (lib/serve/versions.py LEASES): one on another node
                                            # counts as alive until its lease, renewed every quarter lease while it runs, expires
    RUN_VERSIONS_COPY_MAX_BYTES = 1 << 30   # bytes (1 GiB): a snapshot of a work directory (versions.clone) is a copy-on-write
                                            # clone (cp --reflink=always; /fast is XFS with reflink); on a file system without
                                            # clones a run up to this size is copied in full and a larger one is REFUSED (Run_P3S5
                                            # 2026-11-24: 188.6 GB in 1,511,521 files on a /fast 96% full - never duplicated)
    WRITER_LOCK_WAIT_S = 3600.0             # s: the longest a run's writer step waits for its writer lock (a snapshot or a
                                            # conversion holds it: minutes), and a snapshot or a conversion for the writer steps
                                            # to finish (lib/serve/versions.py WORK); then gate G32 refuses, nothing written
    # GATE G33 (lib/serve/versions.py seal, served_read): WHAT THE SERVED PAGES AND THE TWO-PORT SERVER READ from a run, by
    # class - (fnmatch on the path relative to the run; '*' matches '/' too). A version in which a file of these classes names
    # the run's work directory, or any link resolves into it, is refused; every other file (logs, summaries, scripts, caches)
    # is left as its writer wrote it (the coordinator, PR 170: a log records truthfully what its writer was given). Sources:
    # classify_transients.py (the page server) and review_filters.py, the files they open in the run
    SERVED_READS = {
        'queue': ('judgements*/queue_*',),  # --queue / --mchain-queue (SERVE_PAGES queue=), the strat queues and their maps
        'merge_links': ('source_catalogs*/*_sources_survivors.csv', 'source_catalogs*/*_pages_index*.csv',
                        'views/*_sources_survivors.csv', 'views/*_pages_index*.csv'),
                                            # the review_dir column the merge writes (p3s3_merge_review_dir.py:55-87: relative
                                            # to the run) and the page indexes it resolves it from
        'page_data': ('source_catalogs*/*.csv', 'source_catalogs*/*.json', 'views/*.csv', 'views/*.json',
                      'candidates/*.csv', 'candidates/*.json', 'localization_figs/*.json', 'page_index/*.json',
                      'fp_study/*.csv', 'fp_study/*.json', 'blind_items/*.csv', 'addendum_20260823/*.csv', 'fp_review/*.csv',
                      'v_deep_images.csv', 'detection_ts.csv', 'master_catalog.csv', 'quality_flags.csv', 'table_filter.csv',
                      'card2_filter.csv', 'tess_coverage.csv', 'tess_coverage_forced.csv', 'fp_filters.csv',
                      'fp_filters.json', 'scintillation_index.csv', 'scintillation_index.json', 'frame_counts.json',
                      'coadd_root.txt'),
                                            # the catalog views (--catalog-dir, --mchain-catalog-dir, localizations.csv;
                                            # classify_transients.py:1788,2228-2360), each page folder's detections.csv,
                                            # vetoed detections and panel sidecars <png>.page.json (candidates/<S>/<page>/;
                                            # :2282-2295,3978,4377), the page index's manifest (:2146-2165), the 10-s frames
                                            # (:1891), v_deep_images.csv (:3841) and detection_ts.csv (:4314); the page
                                            # filters' tables (review_filters.py:92,101,236,269,301-326,389)
        'html': ('*.html', '*.htm'),
    }
    SERVED_READS_NEVER = ('*.log', '*.log.*')   # (fnmatch on the file name) never a served read, whatever SERVED_READS says
    COALESCE_SLOT_DIR = '.coalesce'         # the coalescing builder's slot inside a run's versions root (lib/serve/coalesce.py)
    SUBLST_PAGE_CONTEXT_WORKERS = 16        # page_context_tenmin.py --workers on the placed version (sublst_cumulative.sh:298; the
                                            # 10-minute panels' run context as page text, Gregg 2026-10-22)
    SUBLST_FILTER_TIER = 'tenmin'           # --tier of bright_source_filter_precompute.py and scintillation_index.py on the new build
                                            # (sublst_cumulative.sh:177-180; R-GREGG-20261112-FILTERS-FINALIZE-BRIGHT-AND-SCINTILLATION)
    SNAPSHOT_SOURCE = '/fast/gh/main/phase3/snapshot_rechain_{label}/{label}'   # on site CHAIN_HOST; snapshot_rechain_batch.sh:51
    SNAPSHOT_COPY_EXCLUDE = ('tmp_panels', 'tmp_panel_queue')      # snapshot_serve.sh:28
    SNAPSHOT_SERVE_HOLD = '/lustre/gh/main/exopipe_ops/snapshot_batch/SERVE_HOLD'   # THE SERVE HOLD: while this file exists (any
                                            # file, empty included; its text says why) no snapshot run is served - G29 refuses before
                                            # every stage, G27 included (snapshot_serve.sh, exopipe_ops aa100bf, 2026-10-17:
                                            # FINDING-SNAPSHOT-BATCH3-CHAINED-ON-PRE-MIRROR-FIX-TEMPLATE-FITS-20261017; lifted
                                            # 2026-10-18 as SERVE_HOLD.lifted_20261018T0132) [step]
    SNAPSHOT_CARD2_REJUDGE_APPROVED = '/lustre/gh/main/exopipe_ops/snapshot_batch/SNAPCOMB_CARD2_REJUDGE_SERVE_APPROVED'   # Gregg's go
                                            # on the counts of a combined view built with Card 2 / re-judging (its summary holds
                                            # "card2" or "rejudge"): G27 serves such a run only when this file is non-empty and
                                            # names it (snapshot_serve.sh, exopipe_ops 35aad1a; written by the session that
                                            # receives his go - his words, the time, the view; 2026-11-04T15:30Z for
                                            # Run_SNAPCOMB_20261104). The name must stand as a whole token (production: grep, a
                                            # substring - a marker naming Run_SNAPCOMB_20261104B also passes Run_SNAPCOMB_20261104)
    SNAPSHOT_RESTORE_ONLY_APPROVED = '/lustre/gh/main/exopipe_ops/snapshot_batch/SNAPCOMB_RESTORE_ONLY_SERVE_APPROVED'   # Gregg's go on
                                            # the restore-only wave's totals: G27 serves a combined view holding a run that ran the
                                            # restore-only 3-D gate (its chain_summary.json key gate3d_restore_only) only when every
                                            # such run's test against the run it replaced passed (SNAPSHOT_RESTORE_DIFF status PASS)
                                            # and this file names the view (snapshot_serve.sh, exopipe_ops 8ff3fc8; same name rule)
    SNAPSHOT_RESTORE_DIFF = 'restore_wave_served_diff.json'   # a restore-only run's test result (snapshot_restore_wave_diff.py)
    SNAPSHOT_VIEW_MEMBER_GLOB = 'Run_SNAP{tag}_2026????'   # a view's member run beside it, per tag of its summary's 'batches'
                                            # (snapshot_serve.sh 8ff3fc8: exactly one, else refused) [step]
    SNAPSHOT_COPY_MD5_PNG_SAMPLE = 1000     # panel PNGs in the copy's md5 completion test (G22c); every other file is compared
                                            # (rechain_copy_after_enospc_20261115.sh: md5 of all non-panel files + 1,000 panels)
    SNAPSHOT_COPY_MIN_FREE_FACTOR = 1.0     # x source size; free bytes on the review disk before the copy (owed refusal,
                                            # INCIDENT-REVIEW-HOST-DATA07-FULL-SERVE-FAILED-HUB-PAGE-TRUNCATED-RESTORED, ledger 1208)
    HUB_DIR = '/data07/gh/phase3/transients/review_hub'                # build_review_hub.py:24; served at / on both listeners
                                            # of the two-port server (the per-port inspection hub page, run_inspection_hub.sh,
                                            # was retired 2026-10-27: IMPL-HUB-SAME-ON-BOTH-LISTENERS-LAYOUT-20261027)
    INSPECTION_GREENLIT = '/lustre/gh/main/exopipe_ops/inspection_greenlit.csv'   # build_inspection_hub.py:14 (Gregg's list of
                                            # the retired inspection hub page, read only; tests.greenlit_only)
    LISTS_DIR = '/data07/gh/phase3/transients/review_lists'            # classify_transients.py:2445 (one pair of lists for every
                                                                        # tier, Gregg 2026-09-20); parity servers get <out>/review_lists

    # ---- ports (ruled: R-PORTS-RECLAIMED-CANONICAL-SET-13-PORTS, ledger 1079) -------------------------------------------
    HUB_PORT = 8765                         # classify_transients.py:423; the two-port server's review listener (run_two_port.sh)
    INSPECTION_HUB_PORT = 8795              # classify_transients.py:424 (Gregg 2026-09-26); its inspection listener
    # (tier, page) -> port and what is served on it. serve OWNS every port below and the hub (refactor-chat decision
    # 2026-10-13: "serve owns all ports incl. forced pages 8775/8776/8785/8786 and jupiter 8800/8801 (one owner of ports
    # and the hub)"). Fields:
    #   family   'review' (the tier's transient pages, served by the tier's stage sequence), 'forced' (snapshot forced
    #            photometry pages, each serving its own run), 'deep-forced' (8798/8799: since 2026-10-17 the deep table-flip
    #            sample pages, not forced photometry) / 'strat' (no production launcher: --restart / --stop only), 'jupiter'
    #            (the Jupiter program's pages)
    #   program  'classify' (ops/classify_transients.py), 'jupiter' (ops/jupiter_server.py ROOT PORT), 'http'
    #            (python -m http.server PORT --bind 127.0.0.1 --directory DIR)
    #   python   (jupiter / http pages) the site PYTHON key of the page's interpreter; absent = 'default'
    #   start    'stages' (serve builds and starts it) or 'restart-only' (serve restarts / stops it with its exact argv)
    #   queue, catalog, labels, loc_dir, log: paths relative to the run directory; hub: (Stokes row, hub tier) of the
    #   registry entry (None: no entry, the ports recorded in the registry's 'tunnel_extra' - which the hub page no longer
    #   prints, 4c42f91); twin: the page's former inspection twin port (its entry's inspection_legacy_port: the
    #   inspection listener's filter cookie and link key; the twins themselves are retired)
    #   mchain_queue, mchain_catalog: the page's MULTIPLES CHAIN (Gregg 2026-10-27, ledger R-GREGG-20261027-HUB-BOTH-LISTENERS-
    #   MULTIPLES-CHAIN-FORCED-EVERY-TIER: "the multi-integration sample within each page should be relabeled as 'multiples
    #   chain' and include the full list of multiples that went through the second chain (no sigma-stack)"), paths relative to
    #   the run directory: classify_transients.py --mchain-queue (the tier's multiples page queue, in full: the 'multiples
    #   chain' sample is exactly its identities, those the page's queue does not list join after its own entries) and
    #   --mchain-catalog-dir (deep only: the multiples fork's own view; identities the page's catalogue does not hold are
    #   loaded from it, the page's rows win for the others) - exopipe_ops 4c42f91; production passes them on the deep and
    #   10-minute singles pages (two-port registry p8766/p8768: both, p8787/p8788: the queue;
    #   IMPL-MULTIPLES-CHAIN-SAMPLE-20261027)
    #   (family pinned only) run, code, env, title, extra: see the pinned pages below
    # the grouping design B preview pages p9300 / p9301 (PAGES family pinned): their page code and title, as registered
    GROUPB_PREVIEW_CODE = '/lustre/gh/main/phase3/refactor_analysis/grouping_20261026/preview/page_code/classify_transients.py'
    GROUPB_PREVIEW_TITLE = ("PREVIEW, no production change - 10-minute MULTIPLES under grouping design B, Stokes {S}: one row per "
                            "identity (P >= 0.97), multiples level P >= 0.85, a multiple = detections in two or more distinct "
                            "10-minute windows; second segment: today's multiples that grouping B counts as single-window. Built "
                            "2026-10-26 from Run_SLSTB (the 2026-10-25 21:21 UTC build, 24 LSTs); compare today's p{compare}")
    PAGES = {
        ('deep', 'V'): dict(port=8766, family='review', program='classify', start='stages', stokes='V', kind='single',
                            queue='judgements/queue_p3s3_V.txt', catalog='source_catalogs_V', labels='judgements_V',
                            mchain_queue='judgements/queue_p3s3_V_multi.txt', mchain_catalog='views/source_catalogs_V_multi',
                            hub=('V', '1-hour'), twin=8796, log='server_V.log'),
                            # p3s3_pass.sh:15-16; p3s3_serve_stokes.sh:79,84; restart_inspection_twin.sh:11
        ('deep', 'I'): dict(port=8768, family='review', program='classify', start='stages', stokes='I', kind='single',
                            queue='judgements/queue_p3s3_I.txt', catalog='source_catalogs_I', labels='judgements_I',
                            mchain_queue='judgements/queue_p3s3_I_multi.txt', mchain_catalog='views/source_catalogs_I_multi',
                            hub=('I', '1-hour'), twin=8797, log='server_I.log'),
        ('deep', 'V-multi'): dict(port=8772, family='review', program='classify', start='stages', stokes='V', kind='multi',
                                  queue='judgements/queue_p3s3_V_multi.txt', catalog='views/source_catalogs_V_multi',
                                  labels='judgements_V', hub=('V', '1-hour-multiples'), twin=8793, log='server_V_multi.log'),
                                  # p3s3_serve_multiples.sh:152,158; ledger 696
        ('deep', 'I-multi'): dict(port=8773, family='review', program='classify', start='stages', stokes='I', kind='multi',
                                  queue='judgements/queue_p3s3_I_multi.txt', catalog='views/source_catalogs_I_multi',
                                  labels='judgements_I', hub=('I', '1-hour-multiples'), twin=8794, log='server_I_multi.log'),
        # deep pages without a production launcher (serve restarts / stops them with their exact argv; registry tunnel_extra)
        ('deep', 'strat-I'): dict(port=8791, family='strat', program='classify', start='restart-only', stokes='I',
                                  kind='single', hub=None, twin=None),   # run_deep_repass4_tablemetric_20261001.sh:35-37 (dated)
        ('deep', 'strat-V'): dict(port=8792, family='strat', program='classify', start='restart-only', stokes='V',
                                  kind='single', hub=None, twin=None),
        # 8798 / 8799: the read-only TABLE-FLIP SAMPLES of the deep run Run_P3S5 (/data07/gh/phase3/transients/
        # Sample_TableFlip_{V,I}_20261017: detections the in-search bright-source table removed then and that pass today; ledger
        # SAMPLE-PAGES-TABLE-FLIP-V-I-RUN-P3S5-20261017; exopipe_ops table_flip_judge_run_20261017.py, no launcher). Checked
        # against the live registry /lustre/gh/main/phase3/serve/registry.json 2026-11-17 (p8798 --run ...Sample_TableFlip_V_
        # 20261017, p8799 ...Sample_TableFlip_I_20261017). The deep forced page runs these ports served before (Run_P3S4_forced3
        # / Run_P3S4_forced, deep_forced_panels.py; ledger 1609) were retired 2026-10-17 (RELEASE-DEEP-DATA07-RETIRED-RUNS-
        # BATCH2-20261017). The family id 'deep-forced' is kept: it names the restart-only deep pages on these two ports, and
        # its serve-time exclusion gate key 'deep-deep-forced' (EXCLUSION_GATE_TIERS, RULED) still gates them
        ('deep', 'tableflip-V'): dict(port=8798, family='deep-forced', program='classify', start='restart-only', stokes='V',
                                      kind='single', hub=None, twin=None),
        ('deep', 'tableflip-I'): dict(port=8799, family='deep-forced', program='classify', start='restart-only', stokes='I',
                                      kind='single', hub=None, twin=None),
        ('sublst', 'V'): dict(port=8787, family='review', program='classify', start='stages', stokes='V', kind='single',
                              queue='judgements/queue_p3s3_V.txt', catalog='source_catalogs_V', labels='judgements_V',
                              mchain_queue='judgements/queue_p3s3_V_multi.txt',
                              hub=('V', '10-min'), twin=None, loc_offsets=True, log='server_V.log'),
                              # sublst_cumulative.sh:121-123; sublst_serve.sh:140; ledger 1047
        ('sublst', 'I'): dict(port=8788, family='review', program='classify', start='stages', stokes='I', kind='single',
                              queue='judgements/queue_p3s3_I.txt', catalog='source_catalogs_I', labels='judgements_I',
                              mchain_queue='judgements/queue_p3s3_I_multi.txt',
                              hub=('I', '10-min'), twin=None, loc_offsets=True, log='server_I.log'),
        ('sublst', 'V-multi'): dict(port=8789, family='review', program='classify', start='stages', stokes='V', kind='multi',
                                    queue='judgements/queue_p3s3_V_multi.txt', catalog='source_catalogs_V_multi',
                                    labels='judgements_V_multi', hub=('V', '10-min-multiples'), twin=None, loc_offsets=True,
                                    log='server_V_multi.log'),   # sublst_cumulative.sh:126-142
        ('sublst', 'I-multi'): dict(port=8790, family='review', program='classify', start='stages', stokes='I', kind='multi',
                                    queue='judgements/queue_p3s3_I_multi.txt', catalog='source_catalogs_I_multi',
                                    labels='judgements_I_multi', hub=('I', '10-min-multiples'), twin=None, loc_offsets=True,
                                    log='server_I_multi.log'),
        ('snapshot', 'V-10s'): dict(port=8779, family='review', program='classify', start='stages', stokes='V', kind='10s',
                                    queue='queue_V_10s.txt', catalog='source_catalogs_V_10s', labels='judgements',
                                    loc_dir='source_catalogs_loc_10s', loc_offsets=True, hub=('V', '10-sec'), twin=None,
                                    log='classify_8779.log'),
                                    # snapshot_serve.sh:57-81,100-101; R-SNAP-1MIN-OWN-PAGES-PER-STOKES (ledger 785)
        ('snapshot', 'I-10s'): dict(port=8780, family='review', program='classify', start='stages', stokes='I', kind='10s',
                                    queue='queue_I_10s.txt', catalog='source_catalogs_I_10s', labels='judgements',
                                    loc_dir='source_catalogs_loc_10s', loc_offsets=True, hub=('I', '10-sec'), twin=None,
                                    log='classify_8780.log'),
        ('snapshot', 'V-1min'): dict(port=8781, family='review', program='classify', start='stages', stokes='V', kind='1min',
                                     queue='queue_V_1min.txt', catalog='source_catalogs_V_1min', labels='judgements',
                                     loc_dir='source_catalogs_loc_1min', loc_offsets=True, hub=('V', '1-min'), twin=None,
                                     log='classify_8781.log'),
        ('snapshot', 'I-1min'): dict(port=8782, family='review', program='classify', start='stages', stokes='I', kind='1min',
                                     queue='queue_I_1min.txt', catalog='source_catalogs_I_1min', labels='judgements',
                                     loc_dir='source_catalogs_loc_1min', loc_offsets=True, hub=('I', '1-min'), twin=None,
                                     log='classify_8782.log'),
        # THE COMBINED VIEW's pages (`serve --tier snapshot --family comb`, lib/serve/snapshot_view.py): production serves the
        # view with snapshot_serve.sh SNAP_REGISTER=1 SNAP_PORTS="8986 8987 8988 8989" SNAP_HUB_SUFFIX=comb (snapshot_batch/
        # next_wave_20261113/serve_release1_20261117.sh; the live registry entries p8986-p8989 and hub rows '<tier>-comb' of
        # Run_SNAPCOMB_20261116): the batch serve's four pages on their own ports, in snapshot_serve.sh's order (V 10-s, I 10-s,
        # V 1-min, I 1-min - the --sibling order); one page per tier and Stokes (Gregg 2026-10-26, R-GREGG-20261026-5-LST-REBUILD-
        # AFTER-EVERYTHING-DONE-COMBINED-VIEW-ONE-PAGE-PER-TIER-POLARIZATION)
        ('snapshot', 'comb-V-10s'): dict(port=8986, family='comb', program='classify', start='stages', stokes='V', kind='10s',
                                         queue='queue_V_10s.txt', catalog='source_catalogs_V_10s', labels='judgements',
                                         loc_dir='source_catalogs_loc_10s', loc_offsets=True, hub=('V', '10-sec-comb'),
                                         twin=None, log='classify_8986.log'),
        ('snapshot', 'comb-I-10s'): dict(port=8987, family='comb', program='classify', start='stages', stokes='I', kind='10s',
                                         queue='queue_I_10s.txt', catalog='source_catalogs_I_10s', labels='judgements',
                                         loc_dir='source_catalogs_loc_10s', loc_offsets=True, hub=('I', '10-sec-comb'),
                                         twin=None, log='classify_8987.log'),
        ('snapshot', 'comb-V-1min'): dict(port=8988, family='comb', program='classify', start='stages', stokes='V', kind='1min',
                                          queue='queue_V_1min.txt', catalog='source_catalogs_V_1min', labels='judgements',
                                          loc_dir='source_catalogs_loc_1min', loc_offsets=True, hub=('V', '1-min-comb'),
                                          twin=None, log='classify_8988.log'),
        ('snapshot', 'comb-I-1min'): dict(port=8989, family='comb', program='classify', start='stages', stokes='I', kind='1min',
                                          queue='queue_I_1min.txt', catalog='source_catalogs_I_1min', labels='judgements',
                                          loc_dir='source_catalogs_loc_1min', loc_offsets=True, hub=('I', '1-min-comb'),
                                          twin=None, log='classify_8989.log'),
        # snapshot FORCED photometry pages (snapshot_forced_serve_20261007.sh:29-54, 96-98; ports ruled 2026-10-03, ledger 1765):
        # each page serves its own run FORCED_RUN_NAME (S, T, --forced-tag); queue/catalog per Stokes; journal = <run>/judgements
        ('snapshot', 'forced-V-10s'): dict(port=8786, family='forced', program='classify', start='stages', stokes='V',
                                           kind='10s', queue='queue_V.txt', catalog='source_catalogs_V', labels='judgements',
                                           loc_offsets=True, hub=('V', '10-sec-forced'), twin=None, log='classify_8786.log'),
        ('snapshot', 'forced-I-10s'): dict(port=8785, family='forced', program='classify', start='stages', stokes='I',
                                           kind='10s', queue='queue_I.txt', catalog='source_catalogs_I', labels='judgements',
                                           loc_offsets=True, hub=('I', '10-sec-forced'), twin=None, log='classify_8785.log'),
        ('snapshot', 'forced-V-1min'): dict(port=8775, family='forced', program='classify', start='stages', stokes='V',
                                            kind='1min', queue='queue_V.txt', catalog='source_catalogs_V', labels='judgements',
                                            loc_offsets=True, hub=('V', '1-min-forced'), twin=None, log='classify_8775.log'),
        ('snapshot', 'forced-I-1min'): dict(port=8776, family='forced', program='classify', start='stages', stokes='I',
                                            kind='1min', queue='queue_I.txt', catalog='source_catalogs_I', labels='judgements',
                                            loc_offsets=True, hub=('I', '1-min-forced'), twin=None, log='classify_8776.log'),
        # the 10-MINUTE AND DEEP FORCED-PHOTOMETRY pages (`serve --tier snapshot --family forced-tier`, lib/serve/forced_tier.py):
        # the four review runs forced_tier_pages.RUN_NAME of exopipe3 forced family T (tier_forced_page_20261027.py chain, run
        # in place), registered 2026-10-27 08:04:50 UTC in one call (HWJ10PC-FORCED-10MIN-DEEP-FULL-CHAIN-FITS-LOC-ASSOC-
        # REREGISTERED-20261027, ledger 3080) with the argv of the snapshot forced pages (--loc-offsets, --default-fp-reason,
        # --sibling to the other three in this order). Checked against the live registry 2026-11-25 (p9120-p9123)
        ('snapshot', 'forced-tier-V-10min'): dict(port=9120, family='forced-tier', program='classify', start='stages', stokes='V',
                                                  kind='10min', queue='queue_V.txt', catalog='source_catalogs_V',
                                                  labels='judgements', loc_offsets=True, hub=('V', '10-min-forced'), twin=None,
                                                  log='classify_9120.log'),
        ('snapshot', 'forced-tier-I-10min'): dict(port=9121, family='forced-tier', program='classify', start='stages', stokes='I',
                                                  kind='10min', queue='queue_I.txt', catalog='source_catalogs_I',
                                                  labels='judgements', loc_offsets=True, hub=('I', '10-min-forced'), twin=None,
                                                  log='classify_9121.log'),
        ('snapshot', 'forced-tier-V-deep'): dict(port=9122, family='forced-tier', program='classify', start='stages', stokes='V',
                                                 kind='deep', queue='queue_V.txt', catalog='source_catalogs_V',
                                                 labels='judgements', loc_offsets=True, hub=('V', '1-hour-forced'), twin=None,
                                                 log='classify_9122.log'),
        ('snapshot', 'forced-tier-I-deep'): dict(port=9123, family='forced-tier', program='classify', start='stages', stokes='I',
                                                 kind='deep', queue='queue_I.txt', catalog='source_catalogs_I',
                                                 labels='judgements', loc_offsets=True, hub=('I', '1-hour-forced'), twin=None,
                                                 log='classify_9123.log'),
        # PINNED pages (`serve --tier deep|sublst --family pinned --stages register`): registered once by hand, no launcher - their
        # entry is this configuration, fixed: the run (under the tier's run root), queue, labels, catalogue, the server code
        # ('code': an absolute path; absent = SERVER_ARGV0 of the tier), the environment ('env': a dict; absent = the tier's
        # SERVER_ENV), the title and any trailing arguments ('extra'). Checked against the live registry 2026-11-25.
        # The FALSE-POSITIVE GROUND-TRUTH SAMPLES (Gregg's overnight task 2026-10-22, REFACTOR-FP-GROUNDTRUTH-SAMPLE-20261022,
        # ledger 2782: blind, stratified samples of identities on their own queue and labels, same code as their parents
        # p8766 p8768 p8787 p8788 then); p9411 / p9412 re-registered with --filter-defaults off (R-GREGG-20261023-DEEP-TRACKING-
        # ON-BY-DEFAULT, ledger 2853); p9413 / p9414 serve the frozen tree their parent Run_SLSTB was published from
        ('deep', 'fp-sample-V'): dict(port=9411, family='pinned', program='classify', start='stages', stokes='V', kind='single',
                                      run='Run_P3S5', queue='judgements/queue_fpsample_V_20261022.txt',
                                      labels='judgements_fpsample_V_20261022', catalog='source_catalogs_V', env={},
                                      title='FP ground-truth sample 1 of 4: deep (1-hour) Stokes V, 60 sources (blind: please '
                                            'leave the filters panel closed)',
                                      extra=('--filter-defaults', 'off'), hub=None, twin=None),
        ('deep', 'fp-sample-I'): dict(port=9412, family='pinned', program='classify', start='stages', stokes='I', kind='single',
                                      run='Run_P3S5', queue='judgements/queue_fpsample_I_20261022.txt',
                                      labels='judgements_fpsample_I_20261022', catalog='source_catalogs_I', env={},
                                      title='FP ground-truth sample 2 of 4: deep (1-hour) Stokes I, 82 sources (blind: please '
                                            'leave the filters panel closed)',
                                      extra=('--filter-defaults', 'off'), hub=None, twin=None),
        ('sublst', 'fp-sample-V'): dict(port=9413, family='pinned', program='classify', start='stages', stokes='V', kind='single',
                                        run='Run_SLSTB', queue='judgements/queue_fpsample_V_20261022.txt',
                                        labels='judgements_fpsample_V_20261022', catalog='source_catalogs_V', loc_offsets=True,
                                        code='/lustre/gh/main/phase3/sublst/code_frozen/SLSTB_20261022T072542Z/exopipe_ops/'
                                             'classify_transients.py',
                                        title='FP ground-truth sample 3 of 4: 10-minute Stokes V, 60 sources (blind: please '
                                              'leave the filters panel closed)', hub=None, twin=None),
        ('sublst', 'fp-sample-I'): dict(port=9414, family='pinned', program='classify', start='stages', stokes='I', kind='single',
                                        run='Run_SLSTB', queue='judgements/queue_fpsample_I_20261022.txt',
                                        labels='judgements_fpsample_I_20261022', catalog='source_catalogs_I', loc_offsets=True,
                                        code='/lustre/gh/main/phase3/sublst/code_frozen/SLSTB_20261022T072542Z/exopipe_ops/'
                                             'classify_transients.py',
                                        title='FP ground-truth sample 4 of 4: 10-minute Stokes I, 81 sources (blind: please '
                                              'leave the filters panel closed)', hub=None, twin=None),
        # the GROUPING DESIGN B PREVIEW of the 10-minute multiples (2026-10-26, grouping_20261026/preview; greenlit by Gregg
        # 2026-10-27 "grouping-B preview"): its own run and page code
        ('sublst', 'groupB-preview-I'): dict(port=9300, family='pinned', program='classify', start='stages', stokes='I',
                                             kind='multi', run='Run_SLSTB_GRPB_PREVIEW_20261026',
                                             queue='judgements/queue_groupB_I.txt', labels='judgements_I_groupB',
                                             catalog='source_catalogs_I', loc_dir='source_catalogs_I', loc_offsets=True,
                                             code=GROUPB_PREVIEW_CODE, title=GROUPB_PREVIEW_TITLE.format(S='I', compare=8790),
                                             hub=None, twin=None),
        ('sublst', 'groupB-preview-V'): dict(port=9301, family='pinned', program='classify', start='stages', stokes='V',
                                             kind='multi', run='Run_SLSTB_GRPB_PREVIEW_20261026',
                                             queue='judgements/queue_groupB_V.txt', labels='judgements_V_groupB',
                                             catalog='source_catalogs_V', loc_dir='source_catalogs_V', loc_offsets=True,
                                             code=GROUPB_PREVIEW_CODE, title=GROUPB_PREVIEW_TITLE.format(S='V', compare=8789),
                                             hub=None, twin=None),
        # the Jupiter program's pages (live on 2026-10-13: 8800 = `jupiter_server.py /lustre/gh/main/phase3/jupiter/products
        # 8800` in cwd ops; 8801 = `python -m http.server 8801 --bind 127.0.0.1 --directory <v2 site>`; jupiter_server.py:303;
        # neither has a launcher in ops). Registry: tunnel_extra (no button). 8801's STANDING COMMAND (Jovian chat 2026-11-14,
        # ledger OPS-20261114-JOVIAN-8801-STANDING-COMMAND): py310_orca (site key 'flow'; static files only, the interpreter does
        # not change content), --directory resolved per request (a site swap by rename needs no restart), started in the site's
        # parent directory (Plan.cwd).
        ('jupiter', 'p1'): dict(port=8800, family='jupiter', program='jupiter', start='stages', root='p1', hub=None, twin=None,
                                log='jupiter_server_8800.log'),
        ('jupiter', 'v2-site'): dict(port=8801, family='jupiter', program='http', start='stages', root='v2-site', hub=None,
                                     twin=None, log='http_8801.log', python='flow'),
    }
    JUPITER_ROOTS = {'p1': '/lustre/gh/main/phase3/jupiter/products',     # argv of the live 8800 server (2026-10-13)
                     'v2-site': '/lustre/gh/main/phase3/jupiter/v2/site'}  # --directory of the live 8801 server
    JUPITER_LOG_DIR = '{out}/_records/serve/logs'   # the Jupiter servers' stdout (no log in production: they ran detached)
    # served by something else: serve never binds, stops or re-points these (the registry rows stay as they are)
    OTHER_SERVED_PORTS = {8767: 'Round-2 reference, frozen (R-GREGG-20261011-KEEP-8767-THRESHOLDS-AT-RERUN, ledger 2199)'}
    PARITY_PORT_OFFSET_MIN = 1000           # a parity --port-offset below this is refused (ports 8765-8801 are production)

    # ---- the two-port server: review pages are REGISTERED, never given a server of their own (production since 2026-10-22,
    # R-GREGG-20261017-TWO-PORT-SERVING, "Retire once all pages are transferred and live"; lib/serve/register.py, the port of
    # /lustre/gh/main/phase3/serve/register_page.py). A page keeps its port as its identity: entry id p<port>, legacy_port
    # <port> (the --sibling / cross-tier link key, the filter cookie); a parity serve shifts both by --port-offset and
    # registers in <out>/TWO_PORT_PARITY_DIR, served by its own two-port server on HUB_PORT / INSPECTION_HUB_PORT + offset.
    # The inspection twins are retired: the inspection listener serves a page when Gregg's greenlit list names it.
    TWO_PORT_DIR = '/lustre/gh/main/phase3/serve'   # register_page.py:24 (D); the live server's registry (run_two_port.sh:8)
    TWO_PORT_REGISTRY = 'registry.json'             # register_page.py:25 (REG): what the server reads
    TWO_PORT_REGISTERED = 'registered_pages.json'   # register_page.py:25 (REGD): the registered entries (the re-seed keeps them)
    TWO_PORT_GREENLIT = 'greenlit_pages.json'       # greenlit_overlay.py:16: Gregg's list (read only; applied to every write)
    TWO_PORT_LOCK = '.registry.lock'                # register_page.py LOCK (2026-11-13): the registry lock - an exclusive flock
                                                    # every writer of the registry holds for its whole read-modify-write
                                                    # (register_page.py, reseed_loop_v5.sh, register.registry_lock)
    TWO_PORT_LOCK_WAIT_S = 900.0                    # register_page.py LOCK_WAIT_S: s to wait for another writer; then refused
    TWO_PORT_PARITY_DIR = 'two_port'                # a parity serve's registry directory under --out
    TWO_PORT_ENV_PREFIXES = ('P3_',)                # an entry's env: these variables of the page's server environment
                                                    # (snapshot_register_page.py:50, SNAP_REGISTER_ENV default)
    # THE FITS CUTOUTS BUTTON (2026-11-12; classify_transients.py:4541 P3_CUTOUTS_TIER, :4584 P3_PAGE_ID): a registered page of
    # family 'review' or 'comb' (the combined view's pages, the same snapshot_serve.sh path) carries its tier's cutouts tier,
    # keyed (tier, page kind), and its own registry id; the other families (strat, deep-forced, forced) carry neither. Live registry 2026-11-14: p8766 p8768 p8772 p8773 deep, p8787-p8790 10min,
    # p8986 p8987 snap, p8988 p8989 1min; production snapshot_serve.sh sets them since exopipe_ops 607f7a2, the deep and
    # 10-minute re-registrations keep the current entry's env
    CUTOUTS_TIER = {('deep', 'single'): 'deep', ('deep', 'multi'): 'deep', ('sublst', 'single'): '10min',
                    ('sublst', 'multi'): '10min', ('snapshot', '10s'): 'snap', ('snapshot', '1min'): '1min'}
    TWO_PORT_POLL_S = 5.0                           # s between two reads of the server's build state (/_status); the 5 s
                                                    # of snapshot_serve.sh:199's identity check
    TWO_PORT_STALL_S = 3600.0                       # s with no build started or finished on the server: the page's build
                                                    # makes no progress (G16). Not a budget for the page: it waits as long as
                                                    # builds progress (CUTOVER_READINESS: the snapshot check's 240 x 5 s)
    TWO_PORT_STATUS_TIMEOUT_S = 60.0                # s, one GET /_status (warm_pages.py:21)
    TWO_PORT_MAX_FAILED_REQUESTS = 3                # a page's requests in a row without a 200 (a build that kills its backend,
                                                    # which the next request restarts): its build does not complete (G24)
    TWO_PORT_WARM_TIMEOUT_S = 900.0                 # s, one warm-up request of a page built before the registration
                                                    # (warm_pages.py:69)
    # the two-port server as run_two_port.sh starts it (:5, :7): its environment and niceness - the command serve prints when no
    # server answers (a short socket directory: register.sock_dir)
    TWO_PORT_SERVER_ENV = {'NUMPY_MADVISE_HUGEPAGE': '0', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                           'MKL_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1'}
    TWO_PORT_NICE = 10
    # THE PUBLISH MARKER (two_port.HOLD_FILE in the run directory; production's deep publish marker, exopipe_ops 4a39c6b): the
    # deep views stage writes it while it rewrites the served run in place, the run's registration removes it (renamed
    # .removed_<UTC>, deep_reregister_after_publish_20261031.py:79-81); a page built meanwhile shows this text as a banner
    # (classify_transients.py:1242)
    PUBLISH_MARKER_TEXT = ('This page was built while exopipe3 serve rewrites its run (started {utc}); until the serve registers '
                           'its pages, the list, its panels and filter values may mix the previous serve with the new one.')
    # THE PUBLISH HOLD (an AMENDMENT OPTION, not ruled; FOLD_IN row 10-30 23:3x): True = the two-port server runs with
    # --publish-hold and keeps a marked run's built pages on their previous builds (stale but consistent) instead of rebuilding
    # them with the banner (mixed, labelled). False = production (no hold, the banner only). serve refuses a server whose mode
    # differs (register.check_server)
    TWO_PORT_PUBLISH_HOLD = False

    # ---- server process ---------------------------------------------------------------------------------------------
    # process identity per program: (comm, an argv element it must END with); classify: restart_inspection_twin.sh:26,
    # restart_server_same_argv.sh:7-9; jupiter / http: the argv of the live 8800 / 8801 servers (2026-10-13)
    SERVER_IDENTITY = {'classify': ('python3', 'classify_transients.py'), 'jupiter': ('python3', 'jupiter_server.py'),
                       'http': ('python3', 'http.server')}
    # argv[1] of the server as each tier's launcher writes it ('{ops}' = the exopipe_ops root of the site file):
    # deep and snapshot run the script by name from cwd ops (p3s3_serve_stokes.sh:79; snapshot_serve.sh:72); the sub-LST
    # launchers name it by absolute path $OPSC/classify_transients.py (sublst_serve.sh:140; sublst_cumulative.sh:139)
    SERVER_ARGV0 = {'deep': 'classify_transients.py', 'sublst': '{ops}/classify_transients.py',
                    'snapshot': 'classify_transients.py'}   # forced pages: by name from cwd ops (snapshot_forced_serve_20261007.sh:49)
    PANEL_FORMAT = 'regen_panels7'          # classify_transients.py:2422 (recorded with every label; the server default, not passed)
    DEFAULT_FP_REASON = {'deep': 'bilobe', 'sublst': 'bilobe', 'snapshot': 'VHF/reflection'}   # classify_transients.py:2438;
                                            # snapshot_serve.sh:73; R-SNAP-FPCLASS-VHF-REFLECTION-RENAMED (ledger 1377).
                                            # Passed as --default-fp-reason only where it differs from the server default
                                            # 'bilobe' (today: snapshot only), so deep/sub-LST argv stay as production's
    DEEP_XTIER = 'Run_{label}:V:8766,Run_{label}:I:8768'      # deep pages link their own run; p3s3_serve_stokes.sh:78 (M11, f768b5e)
    OTHER_TIERS_DEEP_XTIER = 'Run_P3S5:V:8766,Run_P3S5:I:8768'   # sublst_serve.sh:140; sublst_cumulative.sh:139; snapshot_serve.sh:72
                                            # (FIX-OUR-PAGES-DEEP-LINKS-TO-RUN-P3S5-AND-TIER-RULE-3-REPIN, ledger 2033)
    TENMIN_XTIER = 'Run_SLSTB:V:8787,Run_SLSTB:I:8788'   # p3s3_serve_stokes.sh:78; sublst_serve.sh:140; snapshot_serve.sh:72 (ledger 1995)
    # environment of each tier's servers (read by classify_transients.py:331,338,354,381,1018 and sublst_xtier.py:42-45, in place)
    SERVER_ENV = {
        'deep': {'P3_DEEP_XTIER': '{deep_xtier}', 'P3_TENMIN_XTIER': '{tenmin_xtier}',        # p3s3_serve_stokes.sh:78; p3s3_serve_multiples.sh:151
                 'P3_LOC_COMMON_LINKS': '1', 'P3_XTIER_FRAME': 'dewarped'},
                 # + the two the deep pages are served with since 2026-10-30 (two-port registry p8766/p8768: "env = the 10-minute
                 # pages'", Gregg 2026-10-30 'yes, pause and fix the deep V page now', IMPL-DEEP-PAGES-CONFORMANCE-SERVED-MEMBERS-
                 # CURRENT-RUN-LINKS-PSR-CREDIBLE-GREY-10S-FRAMES-20261030; p8772/p8773 the same env on 2026-10-31,
                 # OPS-P3S5-DEEP-PUBLISH-REREGISTERED-AND-VERIFIED-20261031): the SIMBAD bar and cross-tier links on the
                 # common-frame columns (Gregg 2026-10-09 item 4) and the cross-tier links in the dewarped frame (Q17)
        'sublst': {'P3_QUEUE_BAND_ALIAS': '1', 'P3_DEEP_XTIER': '{other_deep_xtier}', 'P3_TENMIN_XTIER': '{tenmin_xtier}',
                   'P3_XTIER_FRAME': 'dewarped', 'P3_LOC_COMMON_LINKS': '1', 'NUMPY_MADVISE_HUGEPAGE': '0'},
                   # sublst_serve.sh:140 (singles); sublst_cumulative.sh:139 (multiples, M22 6b8a23a); ledgers 1983, 1995, 2110
        'snapshot': {'P3_LOC_COMMON_LINKS': '1', 'P3_DEEP_XTIER': '{other_deep_xtier}', 'P3_TENMIN_XTIER': '{tenmin_xtier}',
                     'P3_XTIER_FRAME': 'dewarped', 'P3_XTIER_SUBBAND_STACK_PREFIXES': 'Run_SLST,Run_SNAP', 'NUMPY_MADVISE_HUGEPAGE': '0'},
                     # snapshot_serve.sh:74, :86 (the bundle installed 2026-10-14 03:25 UTC, exopipe_ops 82f7102 / clv15 493d860;
                     # P3_LOC_COMMON_LINKS=1: Gregg 2026-10-09 item 4, ledger 2114); ledger 2030
        # the snapshot FORCED pages: their own launcher's five (snapshot_forced_serve_20261007.sh:49), kept apart from
        # 'snapshot' because the staged bundle changes snapshot_serve.sh only (key '<tier>-<family>', plan.server_env_extra)
        'snapshot-forced': {'P3_DEEP_XTIER': '{other_deep_xtier}', 'P3_TENMIN_XTIER': '{tenmin_xtier}',
                            'P3_XTIER_FRAME': 'dewarped', 'P3_XTIER_SUBBAND_STACK_PREFIXES': 'Run_SLST,Run_SNAP',
                            'NUMPY_MADVISE_HUGEPAGE': '0'},
        # the 10-minute and deep forced pages p9120-p9123: the snapshot forced pages' five (their live entries carry the same
        # four P3_* values)
        'snapshot-forced-tier': {'P3_DEEP_XTIER': '{other_deep_xtier}', 'P3_TENMIN_XTIER': '{tenmin_xtier}',
                                 'P3_XTIER_FRAME': 'dewarped', 'P3_XTIER_SUBBAND_STACK_PREFIXES': 'Run_SLST,Run_SNAP',
                                 'NUMPY_MADVISE_HUGEPAGE': '0'},
        'jupiter': {},                      # the live 8800 / 8801 servers carry no P3_* variable
    }
    STOP_TIMEOUT_S = 10.0                   # s, exact-pid SIGTERM then wait; restart_inspection_twin.sh:53-56 (40 x 0.25 s);
                                            # restart_server_same_argv.sh:11 (20 x 0.5 s); never SIGKILL (no launcher does)
    STOP_POLL_S = 0.25                      # s; restart_inspection_twin.sh:56
    START_SETTLE_S = 2.0                    # s between stop and start; p3s3_serve_stokes.sh:71; sublst_serve.sh:131; snapshot_serve.sh:56
    READY_POLL_S = 5.0                      # s; restart_inspection_twin.sh:69
    READY_TIMEOUT_S = {'deep': 300.0, 'sublst': 300.0, 'snapshot': 400.0, 'jupiter': 300.0}   # s, a server of its own (the
                                            # Jupiter pages, --restart): twins 60 x 5 s (restart_inspection_twin.sh:63);
                                            # snapshot and forced 40 x 10 s (snapshot_rechain_batch.sh:416;
                                            # snapshot_forced_serve_20261007.sh:53); jupiter: no production launcher - the
                                            # deep value. A registered page waits on its build state instead (TWO_PORT_*)
    INDEX_READ_TIMEOUT_S = 120.0            # s, GET of an index page; snapshot_rechain_batch.sh:419
    # (servers of their own: the Jupiter pages; the per-port review servers before 2026-10-22, which --restart / --stop
    # still handle) who may be replaced on a page's port: 'port' = any review server on it (the port is the identity); a tuple = only a
    # server whose argv '--run' is <paths root of the tier (RUN_ROOT_KEY)>/<name> with <name> matching one of these shell patterns (else REFUSE).
    # sub-LST: 'Run_SLST*' = sublst_serve.sh:121 grep of '--run .../Run_SLST'; 'Run_P3T4' = SUBLST_TAKEOVER=Run_P3T4
    # (sublst_cumulative.sh:122,132, the archived old-chain pages; exact name, as the trailing space of the grep)
    PORT_OWNERSHIP = {'deep': 'port',       # p3s3_serve_stokes.sh:69 (2026-09-11), p3s3_serve_multiples.sh:145; ledger 1001
                      'sublst': ('Run_SLST*', 'Run_P3T4'),   # sublst_serve.sh:121-122; sublst_cumulative.sh:122,132; ledger 1623
                      'snapshot': 'port',   # snapshot_serve.sh:52-55; forced pages: snapshot_forced_serve_20261007.sh:71
                      'jupiter': 'port'}    # (no launcher) the port is the identity, the program must match SERVER_IDENTITY

    # ---- titles and hub text ----------------------------------------------------------------------------------------
    UTC_FMT = '%Y-%m-%d %H:%M UT'           # p3s3_serve_stokes.sh:74,84; snapshot_serve.sh:101
    TITLE = {('deep', 'single'): '{tier_text} Stokes {S} transients, wideband + subband, Run_{label} (generation Run_20260909_000000) - served {utc}',
             ('deep', 'multi'): '{tier_text} Stokes {S} MULTI-EPOCH transients (multiples chain), Run_{label} (generation Run_20260909_000000) - served {utc}',
                                            # the deep multiples pages' title since 2026-10-31 (two-port registry p8772/p8773;
                                            # deep_reregister_after_publish_20261031.py:70, OPS-P3S5-DEEP-PUBLISH-REREGISTERED-AND-
                                            # VERIFIED-20261031); p3s3_serve_multiples.sh:152 starts its per-port server without
                                            # one (interface divergence D2, closed by the served pages)
             ('sublst', 'single'): ('10-minute transients, new chain, LSTs {lsts} - Stokes {S}: 10-minute sidereal windows stacked from the '
                                    'snapshot cubes (new chain: frames flowed to the deep Master, x2-aligned to the leave-one-out window '
                                    'Master that is subtracted), subbands and wideband groups (Red/Green/Blue), nights after the night rule '
                                    'and exclusions - served {utc}'),   # sublst_serve.sh:133 with prefix sublst_cumulative.sh:123
             ('sublst', 'multi'): '10-minute MULTI-INTEGRATION transients, new chain, LSTs {lsts} - Stokes {S} - served {utc}',   # sublst_cumulative.sh:140
             ('snapshot', 'single'): 'Snapshot transient search: Stokes {S}, {what} ({label})'}   # snapshot_serve.sh:63,75 (Gregg 2026-09-20)
    DEEP_TIER_TEXT = 'deep (1-hour)'        # p3s3_serve_stokes.sh:73
    SNAPSHOT_WHAT = {'10s': '10-second frames', '1min': '1-minute blocks'}   # snapshot_serve.sh:63
    SNAPSHOT_SIBLING_LABEL = 'Stokes {S} {T} page'   # T = '10-s' | '1-min'; snapshot_serve.sh:66-71 (ledger 1414)
    HUB_GENERATION = {'deep': 'generation 4 (Run_20260909_000000)',   # p3s3_serve_stokes.sh:83; p3s3_serve_multiples.sh:157
                      'sublst': 'new cube-based chain (sub-LST), LSTs: {lsts}',   # sublst_cumulative.sh:70
                      'snapshot': '{label}'}           # snapshot_serve.sh:101
    HUB_REMOVE_TIERS = {'sublst': ('10-min-sublst',)}  # sublst_cumulative.sh:145-150 (the 19h sub-LST buttons, retired)
    # THE HUB PAGE (Gregg 2026-10-27, R-GREGG-20261027-HUB-BOTH-LISTENERS-MULTIPLES-CHAIN-FORCED-EVERY-TIER; build_review_hub.py,
    # exopipe_ops 4c42f91, rendered by lib/serve/hub.render): per Stokes row (V on top, I below) one column per tier, left to
    # right, holding the tier's page (its combined view '<tier>-comb' when live) and its HWJ + 10 pc forced button; below both
    # rows these links and no other (no multiples / preview / batch buttons, no Round-2 link, no tunnel line), plus the interim
    # known-LPT page (exopipe_ops 8cb1139, DONE-20261123-LPT-FORCED-ALL-TIERS-SERVED; Gregg 2026-11-24 "keep it up",
    # R-GREGG-20261124-SIX-ANSWERS-RAMPUP-STATUS item 5: until LPT is a sample of the standard forced pages). PROTECTED: a
    # mirror of the in-place page (production hub writes are build_review_hub.py's; the content test compares the page with it)
    HUB_TIERS = ('1-hour', '10-min', '1-min', '10-sec')   # build_review_hub.py:25 (TIERS)
    HUB_LINKS = (('Jovian bursts', '/s/jupiter/'), ('Solar system', '/s/planets/'),
                 ('Known LPTs (all tiers)', '/s/lpt/'),
                 ('Data processed: Phase 1 &rarr; 2 &rarr; 3', 'data_processing.html'))   # build_review_hub.py:26-28 (LINKS;
                                            # the text is HTML: the page prints it as is)

    # ---- queues and views -------------------------------------------------------------------------------------------
    DEEP_SINGLES_TIERS = ('wb', 'sb')       # p3s3_serve_stokes.sh:20; p3s3_stokes_queue.py:21
    DEEP_FORK_TIERS = ('fwb', 'fsb')        # p3s3_serve_multiples.sh:46,93 (R-MULTIPLES-FORK-INTO-PRODUCTION-PREAMBLE-APPROVED, ledger 2081)
    SUBLST_TIERS = ('sb', 'wb')             # sublst_serve.sh:39,57,81
    MULTI_MIN_NDET = 2                      # n_det >= 2 = multi section; p3s3_serve_tier.sh:24; sublst_serve.sh:65 (ledger 629)
    MARGINAL_LABEL = 'marginal'             # pages_input label queued last; p3s3_serve_tier.sh:20 (ledger 730)
    QUEUE_HEADERS = {                       # part of the queue protocol (read by p3s3_stokes_queue.py:38-41 and the server)
        'deep_tier': ('# queue_name: p3s3_{K}_deep',
                      '# selection: MULTI-EPOCH sub-sample - detected in >1 (LST, date) cell of this band, n_det>=2 (SNR-sorted)',
                      '# selection: single-block sources (SNR-sorted)',
                      '# selection: v110 MARGINAL - retained and labelled (Gregg 2026-09-04), SNR-sorted'),   # p3s3_serve_tier.sh:26-29
        'sublst_tier': ('# queue_name: sublst_{lsts}_{K}', '# selection: MULTI-INTEGRATION - n_det >= 2 (SNR-sorted)',
                        '# selection: single-integration sources (SNR-sorted)'),   # sublst_serve.sh:67-68
        'deep_multi': ('# queue_name: p3s3_{S}_multiples',
                       '# selection: FORKED CHAIN v1 - {T} multi-epoch (n_det>=2 under the ratified association; ordered by SNR_rep, the repeat significance): repeat class identified after mstruct/size/ephemeris/A-team and before the table, CUSTOM per-band multiples table, no sigma_stack (SNR-sorted)',
                       '# selection: FORKED CHAIN v1 - {T} repeat-class positions that cluster as SINGLE under the ratified association (partner within one beam width in another cell only; SNR-sorted)',
                       '# tiers: {counts} (one page per Stokes, Gregg 2026-09-06)'),   # p3s3_serve_multiples.sh:64-65,77
        'sublst_multi_marker': '# selection: MULTI-INTEGRATION'}   # sublst_cumulative.sh:79
    # The review_dir merge into the union survivors belongs to `pages` (decision 6, group 12): deep views READ the merged
    # union and refuse when the merge is absent or older than a page index (views.deep_merge_current); the sub-LST cumulative
    # build runs p3s3_merge_review_dir.py on its own build with pages.REVIEW_DIR_MERGE['sublst'] (one config key).
    # where a deep singles view row takes its review_dir: 'union' = the union row's (today, p3s3_serve_stokes.sh:30), 'own-stokes'
    # = resolved in the view's own Stokes page index <band>_pages_index_<S>.csv (exact page id, then (J-name, band, best_date),
    # (J-name, band), the candidates/I/<J-name>_*_<band> directory, J-name) - the D9 residual (FOLD_IN row 09995b5; measured by
    # serve on the 02:04 views: 0 of 1,278 V and 3 of 2,546 I rows change: J2248+4017 36MHz, J0953+0755 82MHz, J1229+7208 Red).
    # Permitted parity diff D9-RESIDUAL (spec serve.md 10, 12.9; Gregg to confirm).
    DEEP_VIEW_REVIEW_DIR = 'own-stokes'

    # ---- sub-LST cumulative build -----------------------------------------------------------------------------------
    SUBLST_LOC_ENV = {'P3_LOC_COMMON_FRAME': '1', 'P3_LOC_CATS': 'source_catalogs_[IV]_[sw]b'}   # sublst_cumulative.sh:28-29 (ruling item 4,
                                            # 2026-10-08, per sublst_cumulative.sh:27-28; ledger ID not located - not in RULED)
    SUBLST_LOC_SPEED_ENV = {'P3_LOC_ROW_INDEX': '1', 'P3_LOC_SHARDS': '6'}   # SPEED ONLY, production's own two switches of
                                            # build_localizations.py (2026-10-17 row index: each catalog file read once, not once
                                            # per identity; 2026-10-27 shards: N processes, the identities in the same sorted
                                            # order) - the tables are byte-identical to the serial run's (proven on the served
                                            # 24-LST run, 19,332 identities); production's 10-minute build sets neither
                                            # (sublst_cumulative.sh:102-103; 110 min on 2026-11-02). {} = as production. 6: the
                                            # Lustre rule (each process reads every Master once)
    SUBLST_PAGE_CONTEXT_CACHE = 'page_context_frames.csv'   # the page context's frame-noise table (path, size, mtime, rms of every
                                            # searched frame; page_context_tenmin.py:117), CARRIED from the served run into the build
                                            # before the swap: the tool re-validates every entry by size and mtime and measures
                                            # only the rest (production starts each build without it and measures every frame:
                                            # 61,926 frames in 2,451 of the step's 2,554 s on 2026-11-02). None = as production
    SUBLST_FIG_THREADS = 4                  # figures drawn at once; sublst_cumulative.sh:57
    SUBLST_FIG_TIMEOUT_S = 600              # s per figure; sublst_cumulative.sh:54
    SUBLST_INDEX_TIERS = ('V_sb', 'I_sb', 'V_wb', 'I_wb')   # p3t_pages_index.py per tier; sublst_cumulative.sh:62-64 (wb optional)
    SUBLST_JOURNALS = ('judgements_V', 'judgements_I', 'judgements_V_multi', 'judgements_I_multi')   # sublst_cumulative.sh:98, 111
    SUBLST_SERVED_TEST_OLD_LSTS = ('19h',)  # LSTs reported apart by test (c); sublst_catalog_publish.sh:19

    # ---- snapshot ---------------------------------------------------------------------------------------------------
    # Serve-time exclusion gate (G17). It RUNS for every tier and page family (tier parsers in lib/serve/exclusion_gate.py)
    # and REFUSES only for the gate keys listed here (key = the tier for its review pages, '<tier>-<family>' otherwise, e.g.
    # 'snapshot-forced'); for the others it records the rows on excluded cell-bands as a report (unit notes, dry-run plan). Snapshot: Gregg 2026-10-13 ruling 2 'Fine.' (RULINGS-GREGG-20261013-DECISIONS-1-7-AND-TENMIN-TABLES,
    # ledger 2297; IMPLEMENTED-SNAPSHOT-SERVE-EXCLUSION-GATE-20261013, ledger 2301; f1996ee). Deep, sub-LST and the snapshot forced
    # pages: Gregg 2026-10-14 approved enforcing it (R-GREGG-20261014-REFACTOR-DECISIONS-LIST-RECOMMENDATIONS); deep / sub-LST
    # refuse PER PAGE on the rows that page's server reads (lib/serve/exclusion_gate.scan_page; 2026-10-14 ~20:3x UTC on the
    # served Run_P3S5: 0 rows on excluded nights in any page - a first folder-wide scan's 34 were 2026-10-07 index files the
    # server never reads). Production ('snapshot' only, no gate on deep / sub-LST / forced pages) is the parity file's value.
    # Also the restart-only deep pages (stratified sample 'deep-strat'; 'deep-deep-forced' = the pages on 8798/8799, the deep
    # forced runs until 2026-10-17 and the table-flip samples since): they are gated
    # when restarted (servers.restart_port). PROTECTED (and RULED): no --set / --config changes it outside a --parity replay
    # into a non-production --out (review 2026-10-14 m6: --ruling-for accepts any ledger ID, so RULED alone was weaker); so the
    # parity file, which sets it, applies only in a --parity run (exopipe3 config --config refuses it, as any protected key).
    # The 10-minute and deep forced pages ('snapshot-forced-tier', the installed run's every CSV keyed per night: lib/serve/
    # exclusion_gate.scan_forced_tier_run) and the pinned deep / sub-LST pages ('deep-pinned', 'sublst-pinned': per page, as
    # the review pages) are forced and deep / sub-LST pages under the same 2026-10-14 approval (no production gate either).
    EXCLUSION_GATE_TIERS = ('snapshot', 'snapshot-forced', 'deep', 'sublst', 'deep-strat', 'deep-deep-forced',
                            'snapshot-forced-tier', 'deep-pinned', 'sublst-pinned')
    HOLD_FP_VERDICTS = ('FP', 'probably-FP', 'lean-FP')   # snapshot_rechain_batch.sh:379 (R-SNAPSHOT-SERVE-HOLD-ONLY-ON-NON-FP-MISSED-JUDGEMENTS, ledger 1698)
    HOLD_NOTE_CHARS = 200                   # carry_missed.csv judge_note cut; snapshot_rechain_batch.sh:397
    # how the HOLD check reads the OLD journal: 'driver' = the batch driver's heredoc, production today
    # (snapshot_rechain_batch.sh:381-384: last record per source_id of scope source, any action); 'carry' = as the carry
    # reads it (action classify, scope source, a retract clears; snapshot_carry_judgements.py:57-69) - the inventory's
    # defect R6 (ops-snapshot-rest.md:879: a retract / skip record can raise a spurious HOLD), NOT ruled: production
    # behaviour until Gregg rules (review 2026-10-13). Measured: no difference on today's journals (every record is action
    # classify, scope source).
    HOLD_JOURNAL_READER = 'driver'
    SNAPSHOT_LOCALIZE = True                # per-tier positions + figures before the servers (snapshot_serve.sh:44-46, ledger 1831)
    SNAPSHOT_FIGURE_HASHSEED = '0'          # PYTHONHASHSEED of snapshot_localize_tiers.sh (every snapshot localize: a batch serve, the
                                            # combined view). Its figure pool forks from a heredoc interpreter; the string-hash seed
                                            # decides the stacking of overlapping markers in source_astrometry_panel.py (V_combview
                                            # review F10: 64 of 4,130 figure v2 PNGs differed between two runs on byte-identical
                                            # tables). Production sets none (a random seed per run: not reproducible, so no parity
                                            # value exists); '' = unset, as production. The refactor's reproducibility setting, not a
                                            # ruling: the figures' content is the same, only the marker stacking is fixed [step]
    # the five steps on the review copy after the gate on the copy and the carry, before any page starts (snapshot_serve.sh:164-198,
    # exopipe_ops 607f7a2; lib/serve/snapshot_copy.py stages scintillation, page-context, tess, bright-filter, page-index)
    SNAPSHOT_FILTER_TIER = 'snapshot'       # --tier of scintillation_index.py and bright_source_filter_precompute.py (:168, :187;
                                            # R-GREGG-20261112-FILTERS-FINALIZE-BRIGHT-AND-SCINTILLATION)
    SNAPSHOT_PAGE_CONTEXT_WORKERS = 16      # page_context_snapshot.py --workers (:176)
    SNAPSHOT_BRIGHT_CADENCES = ('10s', '1min')   # one bright-source npz per Stokes and cadence (:186; the Stokes are the served pages')
    SNAPSHOT_BRIGHT_WORKERS = 6             # bright_source_filter_precompute.py --workers (:187; at most 6 Lustre readers)
    SNAPSHOT_BRIGHT_PEAK_CACHE = '/data07/gh/bright_filter_snapshot_peaks'   # --peak-cache (:187): the batch records' served rows
                                            # kept between runs, on the review host's local disk
    SNAPSHOT_STEP_CWD = '/tmp'              # the working directory of the scintillation index, TESS coverage, bright-source filter and
                                            # page index (:168, :187, :196 `cd /tmp`); the page context runs in the exopipe_ops root
                                            # (:176)
    SERVE_PARITY_CACHE = '{out}/_cache'     # a parity run's tool caches (the parity guard refuses writes outside --out): the
                                            # scintillation index's --cache and the bright-source filter's --peak-cache. Production
                                            # names none of them (the tools' defaults; the peak cache above)
    SNAPSHOT_PCTX_SAMPLE_DIRS = 5000        # content test snapshot_page_context.sidecar_per_panel: the review folders read (a seeded
                                            # sample; every panel in each). The full view holds ~950,000 folders and 2.7M panels
    PAGE_INDEX_FORMAT = 3                   # the page index's format the review server reads (classify_transients.PAGE_INDEX_FORMAT,
                                            # build_detections_index.py FORMAT); content test detections_index.manifest_current
    PAGE_INDEX_RECHECK = 200                # content test detections_index.records_current: folders re-read against their record

    # ---- the COMBINED SNAPSHOT VIEW: the snapshot tier's cumulative step (`serve --tier snapshot --family comb`,
    # lib/serve/snapshot_view.py; spec docs/specs/snapshot_combined_view.md). Port of exopipe_ops snapshot_combined_view.py
    # (1af2dab .. 5b8b8ff) and its driver snapshot_batch/comb_build_20261102.sh; every value is production's.
    SNAPVIEW_TIERS = ('10s', '1min')        # the view's tiers, in production's pass order; snapshot_combined_view.py:57 TIERS
    SNAPVIEW_STOKES = ('V', 'I')            # the Stokes passes, V first (the order of the record rows); :57 STOKES
    SNAPVIEW_PAGE_SUFFIX = {'10s': '', '1min': '_1min'}   # ONE TIER PER PAGE DIRECTORY: candidates/<S>/<name>_<band><suffix>,
                                            # the batches' own layout (phase3_snapshot_pages.py); :204 TSFX (Gregg 2026-10-03 "Having
                                            # the same tiers on a single page is not conformant", SNAP-PAGES-ONE-TIER-PER-PAGE;
                                            # DEFECT-SNAPCOMB-PAGE-DIRS-HOLD-BOTH-TIERS-20261031); the page code reads this layout
    SNAPVIEW_OMIT_DIR = '{root}/deep_omission/{name}'     # the omit product of a view (deep_omission_kept.csv + decisions,
                                            # record, summary): root = the view's run root (production SNAPSHOT_REVIEW_ROOT, a parity
                                            # run <out>/snapshot), name = the view; = deep_omission.default_out's layout. Production
                                            # had none: its builder decided in its step 0 (snapshot_combined_view.py:78-139)
    SNAPVIEW_FIT_STORE = '{batch_dir}/rechain_{run}/fits_store.csv'   # a member batch's fit store (batch_dir =
                                            # catalog_snapshot.BATCH_DIR, run = the batch run's directory name), the filter
                                            # catalogue's input; comb_build_20261102.sh:51 (a missing or empty store refuses)
    SNAPVIEW_PYTHON = 'py36'                # site PYTHON key of the view build (this module's child) and of the filter catalogue:
                                            # production's driver runs both under /opt/devel/pipeline/envs/development/bin/python,
                                            # comb_build_20261102.sh:18, :48, :52 (the chain's merge runs under the interpreter
                                            # production ran it with)
    SNAPVIEW_LOC_ENV = {'P3_LOC_SHARDS': '12'}   # the per-tier localization's environment: comb_build_20261102.sh:82 (speed
                                            # only: build_localizations.py localizes the same identities in N processes)
    SNAPVIEW_BUILD_TIMEOUT_S = 172800       # s, the view build child and the filter catalogue (the 36-batch view of 2026-11-02 took
                                            # 33,492 s; production's driver sets no limit)
    SNAPVIEW_FILTERCAT_PROCS = 6            # readers of the filter catalogue's detections.csv pass, passed as SNAPCAT_PROCS (exopipe_ops
                                            # 1b9475c: chunks of the glob order read in a process pool and merged in that order, so
                                            # the first copy of a key wins as in the serial loop; 1 = serial; fewer than 1000 files
                                            # are read serially whatever the value). Production's default 16; the refactor's 6 is the
                                            # Lustre rule (at most 6 readers on Lustre or the review disk). Speed only: the catalogue
                                            # is byte-identical for every value (production's FILTERCAT-PARALLEL-TEST, 364,909
                                            # lines on the 3-batch test view) [step]
    SNAPVIEW_TEST_SAMPLE_DIRS = 0           # page directories the snapshot_view content test reads back (0 = every one; N = a
                                            # seeded sample of N for a full view's re-test - the build's own T1-T6 cover all) [step]
    # THE VIEW'S FALSE-POSITIVE FILTERS, the two default hides (stage 'hides', lib/serve/fp_filters.py): <view>/fp_filters.csv +
    # fp_filters.json in review_filters.py's format (exopipe_ops review_filters.py:80-92), from the view's own listed detections
    # before its pages are registered. Port of the refactor chat's tools review_items_20261031/hides_20261115/ (parallel_collect.py,
    # newview_hides.py flood + write; run_hides.sh), which wrote the served Run_SNAPCOMB_20261116's files. THE CONTENT IS THE
    # SNAPSHOT CHAT'S: the cells, factors, states, labels and help texts below are production's, unchanged (a filter that hides
    # sources needs Gregg's go). Memory note 2026-10-31 'snapshot hides: alignment + flood': rebuilt from every new view's own
    # detections, never carried from the old view.
    SNAPVIEW_HIDES_READERS = 12             # threads reading the listed sources' detections.csv (run_hides.sh: parallel_collect.py
                                            # ... 12; speed only, the order of the output is the input order) [step]
    SNAPVIEW_FLOOD_FACTOR = 100             # a flood hour: served detections >= this x the median of the other nights at the same
                                            # LST and band (same Stokes and tier) ... (newview_hides.py K; Gregg 2026-10-31
                                            # "recommendations approved", IMPL-SNAPSHOT-FLOOD-HOURS-HIDDEN-BY-DEFAULT-20261031; "100x
                                            # works", R-GREGG-20261108-GEN4-REPLACE-FLOOD-100X-JOVIAN-WAITS-FOR-CUBES)
    SNAPVIEW_FLOOD_MIN = 50                 # ... and >= this many detections (newview_hides.py N_MIN; same rulings)
    # the frame-alignment cells (LST, night) -> bands: FAILED = still on the corrupt September 23h 82 MHz reference (state 0, hidden
    # by default); HELD = re-searched from the cubes 2026-11-01 with counts still above the clean nights (state 1, shown by default:
    # Gregg 2026-11-03 "Only 10-30", R-GREGG-20261103-ALIGNMENT-HIDE-ONLY-1030-SHOW-REPAIRED-NIGHTS). FAILED is empty since
    # 2026-11-15 (newview_hides.py ALIGN_STILL: 23h 2025-10-30 is excluded by data_decisions, lost the sky, 2026-11-13)
    SNAPVIEW_ALIGN_FAILED = {}
    SNAPVIEW_ALIGN_HELD = {('23h', '2025-08-27'): ('82MHz', 'Blue', 'Full'), ('23h', '2025-09-01'): ('82MHz', 'Blue', 'Full'),
                           ('23h', '2025-09-02'): ('82MHz', 'Blue', 'Full')}   # newview_hides.py ALIGN_HELD
    # the two categories of fp_filters.json, production's metadata (the served Run_SNAPCOMB_20261116/fp_filters.json; newview_hides.py
    # write: the live view's states and labels, help texts with this view's counts). default_off: the alignment-failed state only
    # ("Only 10-30"); the flood state's default_off is written as production writes it and ignored by the pages since Gregg 2026-11-12
    # (review_filters.py NO_DEFAULT_CATS, R-GREGG-20261112-TRACKING-AND-FLOOD-DEFAULTS-OFF); neither category has a control on the panel
    # (review_filters.py PANEL_HIDDEN_CATS / NO_DEFAULT_CATS, R-GREGG-20261112-PANEL-ONE-SET-ALL-TIERS-BOTH-PORTS). Help placeholders:
    # {n_held} the held-state detections, {factor} {n_min} the flood rule, {n_hours} {n_det} {n_nights} the view's flood hours.
    SNAPVIEW_FP_ALIGN = dict(
        key='fp_c_align', stokes=None,
        states=['frame alignment failed (corrupt reference Master; re-search queued) - hidden until the re-search',
                're-searched from the cubes 2026-11-01, counts still above the clean nights - held hidden until Gregg looks',
                'frame alignment not in question'],
        default_off=['frame alignment failed (corrupt reference Master; re-search queued) - hidden until the re-search'],
        label='10-s frame alignment failed - hidden until the re-search (Gregg 2026-10-31)',
        help=('The 10-s frames of 23h 82 MHz were aligned (per-frame flow) in September to a corrupt Taper Robust 0 Master; the '
              'alignment moved bright sources by up to ~30 arcmin and made false transients (e.g. 3C 103 and 3C 91). In this view NO '
              'cell remains on the corrupt reference: 23h 2025-10-30 82 MHz (with the Blue and Full wideband images) is excluded by '
              'the data decisions (lost the sky, 2026-11-13) and has no detections here, so nothing is hidden by default. The other '
              'eight nights are repaired: 2025-08-27, 09-01, 09-02, 09-03, 09-06 and 09-13 re-searched from the cubes, 08-23 and '
              '08-26 on the cube flow fields. Of those, 08-27, 09-01 and 09-02 still show more 23h 82 MHz detections than the clean '
              'nights (Stokes I 28, 32, 62 against 0-4 when measured): SHOWN by default (Gregg 2026-11-03: "Only 10-30") for his '
              'review ({n_held:,} detections in this view); untick the second box to hide them.'))
    SNAPVIEW_FP_FLOOD = dict(
        key='fp_c_flood', stokes=None,
        states=['flood hour: >= 100x the usual night at this LST and band (Gregg 2026-10-31) - hidden by default',
                'not a flood hour'],
        default_off=['flood hour: >= 100x the usual night at this LST and band (Gregg 2026-10-31) - hidden by default'],
        label='Flood hour (>= 100x the usual night) - hidden by default (Gregg 2026-10-31)',
        help=('An hour of one night in one band (this tier and Stokes) whose served detections are at least {factor} times the '
              'median of the other nights at the same LST and band, and at least {n_min}: ionospheric storm hours. Measured on THIS '
              'view: {n_hours} flood hours, {n_det:,} detections, {n_nights} nights. A source drops out of the default view only '
              'when all its detections fall in flood hours. Tick the first box to see them. The rule: /s/fp_deep/flood_nights/'))
    SNAPVIEW_FP_NOTE = ("# frame-alignment states and flood hours at {factor}x from this view's own listed detections (exopipe3 serve "
                        "--tier snapshot --family comb, stage hides; lib/serve/fp_filters.py, the port of review_items_20261031/"
                        "hides_20261115/newview_hides.py): {n_failed} alignment failed (hidden by default), {n_held} re-searched "
                        "and held (shown by default), {n_flood} in flood hours (no default since "
                        "R-GREGG-20261112-TRACKING-AND-FLOOD-DEFAULTS-OFF)\n")   # the CSV's first line (the pages skip '#' lines)

    # ---- snapshot forced photometry pages (port of snapshot_forced_gen5_serve_20261026.sh and the 'refresh-siblings' mode of
    # snapshot_forced_serve_20261007.sh; ruled 2026-10-03 / 2026-10-07 Q5/Q7/Q8, ledger 1765) -------------------------------
    FORCED_RUN_NAME = 'Run_SNAPFORCED_{S}{T}_{tag}'          # T = 10s | 1min; snapshot_forced_serve_20261007.sh:30
    FORCED_READY_LOG = 'snapshot_forced_pages_run_{name}.log'   # under paths.OPS_ROOT; its LAST line must hold FORCED_READY_MARK (:63-65)
    FORCED_READY_MARK = 'FT-READY'
    FORCED_READY_RECORD = 'C_verify_{name}'  # a run built by exopipe3 forced --tier snapshot (family C) is ready by this unit's
                                            # record (step forced, under forced_snapshot.WORK_ROOT): done, servable, not partial,
                                            # naming the review copy served; a run without it by its FORCED_READY_LOG
    FORCED_REQUIRED = ('queue_{S}.txt', 'source_catalogs_{S}', 'source_catalogs/localization_offsets.csv',
                       'forced_assoc_identity.csv')          # :66-67
    # THE PAGE INPUTS of the forced pages (lib/serve/forced.page_context): before a page is registered, on the run it will serve,
    # the four steps production's page runner takes before FT-READY (snapshot_forced_pages_run_20261007.sh, exopipe_ops bd07263;
    # by hand on the generation-5 runs 2026-11-12..20), in its order, each refusing: scintillation_index.py RUN --tier
    # SNAPSHOT_FILTER_TIER; bright_source_filter_precompute.py RUN --tier FORCED_BRIGHT_TIER --workers FORCED_BRIGHT_WORKERS;
    # page_context_snapshot.py RUN --served-from VIEW --workers FORCED_PAGE_CONTEXT_WORKERS; tess/tess_coverage.py --run RUN
    # --from-candidates (lib/serve/tess.coverage: site PYTHON[pages_deep.TESS_PYTHON], the shared reader arguments); all from
    # SNAPSHOT_STEP_CWD, as the runner (cd /tmp)
    FORCED_BRIGHT_TIER = 'forced'           # bright_source_filter_precompute.py --tier (bd07263; the tool's forced tier)
    FORCED_BRIGHT_WORKERS = 6               # its --workers (bd07263; at most 6 Lustre readers)
    FORCED_PAGE_CONTEXT_WORKERS = 4         # page_context_snapshot.py --workers on a forced run (bd07263)
    FORCED_PAGE_CONTEXT_VIEW_IDS = ('p8986', 'p8987', 'p8988', 'p8989')   # the production registry entries (read only, a parity
                                            # serve too) of the snapshot transient-search pages, 10 s and 1 min (snapshot_serve.sh
                                            # SNAP_REGISTER=1): their --run is the view the forced pages are paired with (its
                                            # member rows are the panels' 'served detections' and the n_i_epoch / n_v_epoch
                                            # sliders' counts); all four must name ONE run, else the page is refused
    FORCED_PAGE_CONTEXT_REQUIRED = ('scintillation_index.csv', 'scintillation_index.json', 'bright_source_filter.npz',
                                    'frame_counts.json', 'tess_coverage.csv')   # after the four steps; their content by the
                                            # classes tess_coverage (lib/serve/tess.coverage), scintillation_index,
                                            # bright_source_filter and snapshot_forced_page_context (the sidecars)
    FORCED_ADLEO_CHECK = ('test_snapshot_forced_reserve.py', ('adleo', '--absent-ok'))   # ops script + args, site PYTHON 'py36' (:68;
                                            # Gregg 2026-10-08 "don't label AD Leo unless you have a metric defined and measured")
    FORCED_GENERATIONS = {                   # forced run tag = cutout generation -> (the samples text, the generation text) the
                                            # title and hub name, and the samples that generation measured: V2 of its snapshot
                                            # evaluations expects those samples x Stokes x tier (lib/photometry/tests.
                                            # threshold_classes). A tag without an entry is refused (G26; V2: unknown generation)
        'gen5': ('HWJ + 10 pc', 'generation-5 cutouts from the current cubes', ('HWJ', '10pc')),
                                            # snapshot_forced_gen5_serve_20261026.sh:57, :72; generation 5 never measured LPT
        'gen6': ('HWJ + 10 pc + LPT', 'generation-6 cutouts from the rescaled cubes (Master flux scale)', ('HWJ', '10pc', 'LPT')),
                                            # gen6_20261117/gen6_forced_serve_SWITCH_DRAFT.sh:55, :71; LPT measured since 2026-11-18
                                            # (Gregg "Approve gen6 with the LPTs included"; photometry.SAMPLES_MEASURED)
    }
    FORCED_TITLE = ('Snapshot FORCED photometry ({samples}), Stokes {S}, {what} - pixel-distance screen, fitted, reference-tied, '
                    'localized, associated (target / background source / ambiguous) - {generation} '
                    '({run_name}) - served {utc}')   # snapshot_forced_gen5_serve_20261026.sh:57 (the live entries p8775/p8776/p8785/p8786)
    FORCED_SIBLING_LABEL = 'forced Stokes {S} {T} page'      # T = 10-s | 1-min; :34
    FORCED_HUB_GENERATION = ('forced photometry at {samples} targets, {generation}, Stokes {S} '
                             '{kind} events: pixel-distance screen, fitted, reference-tied, localized, associated')   # kind = 10s |
                                            # 1min; snapshot_forced_gen5_serve_20261026.sh:72-74 (the live hub entries)
    # THE 10-MINUTE AND DEEP FORCED PAGES (family forced-tier, lib/serve/forced_tier.py; p9120-p9123): the four review runs
    # forced_tier_pages.RUN_NAME, made by exopipe3 forced family T as <run>.new (tier_runs). Their words, as the live entries
    # (registered 2026-10-27 08:04:50 UTC by hand, ledger 3080) and the live hub buttons (07:07 UTC) give them: the samples text
    # is the run's own (its queue header, FORCED_TIER_QUEUE_HEADER: tier_forced_page_20261027.py samples_short(), :1110), the
    # generation the run name's _gen<N>
    FORCED_TIER_TITLE = ('Snapshot-chain FORCED photometry ({samples}), Stokes {S}, {what} - each event fitted in its own image '
                         '(Gaussian >= 7 sigma, template below), localized, associated (target / background source / '
                         'ambiguous); no reference tie at this tier ({run_name}) - served {utc}')
    FORCED_TIER_WHAT = {'10min': '10-minute windows', 'deep': '1-hour (deep) nightly epochs'}  # planets_forced_page_20261026.py:69
    FORCED_TIER_SIBLING_T = {'10min': '10-minute', 'deep': 'deep (1-hour)'}   # T of FORCED_SIBLING_LABEL (the live --sibling)
    FORCED_TIER_HUB_GENERATION = ('forced photometry at the {samples} targets, {kind} tier, generation-{gen} tier cutouts, '
                                  'Stokes {S} events')
    FORCED_TIER_QUEUE_HEADER = (r'^# queue_name: forced photometry \((?P<samples>[^)]+)\), Stokes (?P<S>[IV]), '
                                r'(?P<tier>10min|deep) tier$')
    FORCED_TIER_REQUIRED = ('queue_{S}.txt', 'source_catalogs_{S}', 'source_catalogs_{S}_loc', 'forced_fits.csv',
                            'forced_assoc.csv', 'forced_assoc_identity.csv', 'cards/lines.json')   # what the chain writes and
                                            # the page reads (tier_forced_page_20261027.py cmd_chain; verify-chain reads cards/)
    # THE INSTALL of <run>.new (production's `swap` :1192-1206 and `swap-stage` :1256-1277): the new tree is renamed to
    # <root>/FORCED_TIER_STAGE/<run> (the tools below recognize a forced run by its own name, scintillation_index.forced_run),
    # gets its page inputs there, then the live run's judgements/ are copied into it (rsync -a --update), the live run is
    # renamed FORCED_TIER_KEEP, the staged run takes its place, and the kept run's judgements/ are copied again (a label
    # written in between). The kept run stays (production kept every one: .superseded_nofit_20261027, .before_epoch_20261110)
    FORCED_TIER_STAGE = '{run}.install'
    FORCED_TIER_KEEP = '{run}.before_{utc}'      # utc = %Y%m%dT%H%M%SZ of the install
    # the page inputs of a tier run (production's interim pushes, ledger PUSH-20261112-FORCED-PAGES-FILTER-FILES and
    # PREAMBLE-20261115-TESS-COVERAGE-INTERIM-SERVED-RUNS; no page_context sidecars on these runs), in this order from
    # SNAPSHOT_STEP_CWD: scintillation_index.py RUN --tier FORCED_TIER_SCINT_TIER[kind]; bright_source_filter_precompute.py RUN
    # --tier FORCED_BRIGHT_TIER --parents <paths.TRANSIENTS_ROOT>/FORCED_TIER_PARENTS[kind] --workers FORCED_BRIGHT_WORKERS (the
    # search run whose Masters the event planes subtracted: the tool refuses another); 10 min: tess/frame_times_tenmin.py
    # --run RUN --image-root <paths.SUBLST_ROOT>/FORCED_TIER_TENMIN_TREE; tess/tess_coverage.py --run RUN --from-candidates
    # (deep: --frame-times <paths.TRANSIENTS_ROOT>/FORCED_TIER_DEEP_FRAME_TIMES)
    FORCED_TIER_SCINT_TIER = {'10min': 'tenmin', 'deep': 'deep'}     # scintillation_index.FORCED_TIER
    FORCED_TIER_PARENTS = {'10min': 'Run_SLSTB', 'deep': 'Run_P3S5'}
    FORCED_TIER_TENMIN_TREE = 'tenmin_root_x2'
    FORCED_TIER_DEEP_FRAME_TIMES = 'Run_P3S5/frame_times.csv'
    FORCED_TIER_INPUTS_REQUIRED = ('scintillation_index.csv', 'scintillation_index.json', 'bright_source_filter.npz',
                                   'tess_coverage.csv')   # + frame_times.csv at 10 min

    # ---- in-place calls ---------------------------------------------------------------------------------------------
    LEGACY_TIMEOUT_S = 7200                 # s, any in-place script (sublst_publish_lst.sh:44 ran the cumulative under 7200 s)
    APP_PROBE_TIMEOUT_S = 1200              # s, the server's own loader + App on one page (sources_listed expectation)

    RULED = {
        'PAGES': ('R-PORTS-RECLAIMED-CANONICAL-SET-13-PORTS', 'R-MULTIPLES-OWN-PAGES-8772-8773', 'R-TENMIN-PORTS-8787-V-8788-I',
                  'R-SNAP-1MIN-OWN-PAGES-PER-STOKES', 'R-GREGG-20261027-HUB-BOTH-LISTENERS-MULTIPLES-CHAIN-FORCED-EVERY-TIER'),
        'HUB_TIERS': 'R-GREGG-20261027-HUB-BOTH-LISTENERS-MULTIPLES-CHAIN-FORCED-EVERY-TIER',
        'HUB_LINKS': ('R-GREGG-20261027-HUB-BOTH-LISTENERS-MULTIPLES-CHAIN-FORCED-EVERY-TIER',
                      'R-GREGG-20261124-SIX-ANSWERS-RAMPUP-STATUS'),
        'HUB_PORT': 'R-PORTS-RECLAIMED-CANONICAL-SET-13-PORTS',
        'INSPECTION_HUB_PORT': 'R-PORTS-RECLAIMED-CANONICAL-SET-13-PORTS',
        'OTHER_SERVED_PORTS': ('R-PORTS-RECLAIMED-CANONICAL-SET-13-PORTS', 'R-GREGG-20261011-KEEP-8767-THRESHOLDS-AT-RERUN'),
        'RUN_NAME': 'R-NEW-CHAIN-TAKES-OVER-10MIN-PAGES-CUMULATIVE',
        'DEFAULT_FP_REASON': 'R-SNAP-FPCLASS-VHF-REFLECTION-RENAMED',
        'DEEP_XTIER': 'FIX-REFACTOR-MEDIUM-DEFECTS-M1-M19-DEEP-20261013',
        'OTHER_TIERS_DEEP_XTIER': 'FIX-OUR-PAGES-DEEP-LINKS-TO-RUN-P3S5-AND-TIER-RULE-3-REPIN',
        'TENMIN_XTIER': 'IMPLEMENTED-Q17-10MIN-CROSS-TIER-LINKS-ONE-FRAME',
        'SERVER_ENV': ('IMPLEMENTED-Q17-10MIN-CROSS-TIER-LINKS-ONE-FRAME', 'IMPLEMENTED-Q17-CROSS-TIER-LINKS-ON-THE-SNAPSHOT-PAGES',
                       'RULINGS-GREGG-20261009-ASTROMETRY-FOLLOWUPS-1-8', 'FIX-REFACTOR-MEDIUM-DEFECTS-M1-M19-DEEP-20261013',
                       'FIX-REFACTOR-M20-M22-M30-M31-M33-20261013',
                       'IMPL-DEEP-PAGES-CONFORMANCE-SERVED-MEMBERS-CURRENT-RUN-LINKS-PSR-CREDIBLE-GREY-10S-FRAMES-20261030'),
        'PORT_OWNERSHIP': 'PRODUCTION-STATUS-PM-PORT-GUARD-PREP-01H-CUBE-PACE',
        'FORCED_ADLEO_CHECK': 'RULINGS-GREGG-20261008-A-D-STACK-FLOWREF-MEMBERSHIP-GEN4-FORCED-I1MIN',
        'MULTI_MIN_NDET': 'CORRECTION-MULTI-EPOCH-SECTION-WAS-MULTIBAND',
        'DEEP_FORK_TIERS': 'R-MULTIPLES-FORK-INTO-PRODUCTION-PREAMBLE-APPROVED',
        'DEEP_VIEW_REVIEW_DIR': 'FIX-REFACTOR-DEFECTS-D2-D6-D7-D9-D10-AND-RULINGS-D1-D5-D8-20261013',
        'HOLD_FP_VERDICTS': 'R-SNAPSHOT-SERVE-HOLD-ONLY-ON-NON-FP-MISSED-JUDGEMENTS',
        'SNAPSHOT_LOCALIZE': 'R-SNAP-EACH-DETECTION-OWN-FIT-AND-PER-TIER-COMBINED-POSITIONS-IMPLEMENTED',
        'SNAPSHOT_FILTER_TIER': 'R-GREGG-20261112-FILTERS-FINALIZE-BRIGHT-AND-SCINTILLATION',
        # the combined snapshot view: one page per tier and Stokes over every batch (Gregg 2026-10-26 "No! I just mean combine
        # the batches for each tier/polarization into one page."), one tier per page directory (Gregg 2026-10-03)
        'SNAPSHOT_CARD2_REJUDGE_APPROVED': 'R-GREGG-20261104-DEEP-OMISSION-SHORT-TIERS-ANSWERS',   # "the count of removed
                                            # sources shown to Gregg before serving" - as a gate (35aad1a)
        'SNAPSHOT_RESTORE_ONLY_APPROVED': 'R-GREGG-20261104-EVERYTHING-AS-RECOMMENDED',   # "Gregg gets the wave totals of restored
                                            # and secondary removals before the new view is served" (8ff3fc8)
        'SNAPVIEW_TIERS': 'R-GREGG-20261026-5-LST-REBUILD-AFTER-EVERYTHING-DONE-COMBINED-VIEW-ONE-PAGE-PER-TIER-POLARIZATION',
        'SNAPVIEW_STOKES': 'R-GREGG-20261026-5-LST-REBUILD-AFTER-EVERYTHING-DONE-COMBINED-VIEW-ONE-PAGE-PER-TIER-POLARIZATION',
        'SNAPVIEW_PAGE_SUFFIX': ('SNAP-PAGES-ONE-TIER-PER-PAGE', 'DEFECT-SNAPCOMB-PAGE-DIRS-HOLD-BOTH-TIERS-20261031'),
        # the view's two default hides (the snapshot chat's content, ported unchanged)
        'SNAPVIEW_FLOOD_FACTOR': ('IMPL-SNAPSHOT-FLOOD-HOURS-HIDDEN-BY-DEFAULT-20261031',
                                  'R-GREGG-20261108-GEN4-REPLACE-FLOOD-100X-JOVIAN-WAITS-FOR-CUBES'),
        'SNAPVIEW_FLOOD_MIN': 'IMPL-SNAPSHOT-FLOOD-HOURS-HIDDEN-BY-DEFAULT-20261031',
        'SNAPVIEW_ALIGN_FAILED': ('IMPL-SNAPSHOT-ALIGNMENT-HIDE-NARROWED-20261031',
                                  'R-GREGG-20261103-ALIGNMENT-HIDE-ONLY-1030-SHOW-REPAIRED-NIGHTS'),
        'SNAPVIEW_ALIGN_HELD': 'R-GREGG-20261103-ALIGNMENT-HIDE-ONLY-1030-SHOW-REPAIRED-NIGHTS',
        'SNAPVIEW_FP_ALIGN': ('R-GREGG-20261103-ALIGNMENT-HIDE-ONLY-1030-SHOW-REPAIRED-NIGHTS',
                              'R-GREGG-20261112-PANEL-ONE-SET-ALL-TIERS-BOTH-PORTS'),
        'SNAPVIEW_FP_FLOOD': ('IMPL-SNAPSHOT-FLOOD-HOURS-HIDDEN-BY-DEFAULT-20261031', 'R-GREGG-20261112-TRACKING-AND-FLOOD-DEFAULTS-OFF',
                              'R-GREGG-20261112-PANEL-ONE-SET-ALL-TIERS-BOTH-PORTS'),
        'EXCLUSION_GATE_TIERS': ('RULINGS-GREGG-20261013-DECISIONS-1-7-AND-TENMIN-TABLES',
                                 'R-GREGG-20261014-REFACTOR-DECISIONS-LIST-RECOMMENDATIONS'),
    }
