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
    """exopipe3 serve: review servers (classify_transients.py, run in place), inspection twins, review hub and inspection
    hub, for every tier. Ports, run roots, titles and the per-tier server environment decide what Gregg and the team see.
    Host names live in the site file (REVIEW_HOST, CHAIN_HOST; the hub tunnel host stays in build_review_hub.py:29 in place)."""
    # keys no --set / --config can change (DESIGN 11.4; honoured by config/__init__.py)
    PROTECTED_KEYS = ('RUN_ROOT_KEY', 'HUB_DIR', 'INSPECTION_HUB_DIR', 'LISTS_DIR', 'INSPECTION_GREENLIT', 'EXCLUSION_GATE_TIERS',
                      'OTHER_SERVED_PORTS', 'JUPITER_ROOTS')

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
    SUBLST_PREV_SUFFIX = '.prev_{utc}'      # utc = %Y%m%dT%H%M%S; sublst_cumulative.sh:102
    SNAPSHOT_SOURCE = '/fast/gh/main/phase3/snapshot_rechain_{label}/{label}'   # on site CHAIN_HOST; snapshot_rechain_batch.sh:51
    SNAPSHOT_COPY_EXCLUDE = ('tmp_panels', 'tmp_panel_queue')      # snapshot_serve.sh:28
    SNAPSHOT_COPY_MIN_FREE_FACTOR = 1.0     # x source size; free bytes on the review disk before the copy (owed refusal,
                                            # INCIDENT-REVIEW-HOST-DATA07-FULL-SERVE-FAILED-HUB-PAGE-TRUNCATED-RESTORED, ledger 1208)
    HUB_DIR = '/data07/gh/phase3/transients/review_hub'                # build_review_hub.py:12
    INSPECTION_HUB_DIR = '/data07/gh/phase3/transients/inspection_hub' # run_inspection_hub.sh:3; build_inspection_hub.py:15
    INSPECTION_GREENLIT = '/lustre/gh/main/exopipe_ops/inspection_greenlit.csv'   # build_inspection_hub.py:14 (Gregg's list, read only)
    LISTS_DIR = '/data07/gh/phase3/transients/review_lists'            # classify_transients.py:2445 (one pair of lists for every
                                                                        # tier, Gregg 2026-09-20); parity servers get <out>/review_lists

    # ---- ports (ruled: R-PORTS-RECLAIMED-CANONICAL-SET-13-PORTS, ledger 1079) -------------------------------------------
    HUB_PORT = 8765                         # classify_transients.py:423; build_review_hub.py:27
    INSPECTION_HUB_PORT = 8795              # classify_transients.py:424; run_inspection_hub.sh:7 (Gregg 2026-09-26)
    # (tier, page) -> port and what is served on it. serve OWNS every port below and the hub (refactor-chat decision
    # 2026-10-13: "serve owns all ports incl. forced pages 8775/8776/8785/8786 and jupiter 8800/8801 (one owner of ports
    # and the hub)"). Fields:
    #   family   'review' (the tier's transient pages, served by the tier's stage sequence), 'forced' (snapshot forced
    #            photometry pages, each serving its own run), 'deep-forced' / 'strat' (no production launcher: --restart /
    #            --stop only), 'jupiter' (the Jupiter program's pages)
    #   program  'classify' (ops/classify_transients.py), 'jupiter' (ops/jupiter_server.py ROOT PORT), 'http'
    #            (python -m http.server PORT --bind 127.0.0.1 --directory DIR)
    #   start    'stages' (serve builds and starts it) or 'restart-only' (serve restarts / stops it with its exact argv)
    #   queue, catalog, labels, loc_dir, log: paths relative to the run directory; hub: (Stokes row, hub tier) of the
    #   registry button (None: tunnel only, registry 'tunnel_extra'); twin: inspection twin port
    PAGES = {
        ('deep', 'V'): dict(port=8766, family='review', program='classify', start='stages', stokes='V', kind='single',
                            queue='judgements/queue_p3s3_V.txt', catalog='source_catalogs_V', labels='judgements_V',
                            hub=('V', '1-hour'), twin=8796, log='server_V.log'),
                            # p3s3_pass.sh:15-16; p3s3_serve_stokes.sh:79,84; restart_inspection_twin.sh:11
        ('deep', 'I'): dict(port=8768, family='review', program='classify', start='stages', stokes='I', kind='single',
                            queue='judgements/queue_p3s3_I.txt', catalog='source_catalogs_I', labels='judgements_I',
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
        ('deep', 'forced-run3'): dict(port=8798, family='deep-forced', program='classify', start='restart-only', stokes='I',
                                      kind='single', hub=None, twin=None),   # Run_P3S4_forced3 (ledger 1609; no launcher)
        ('deep', 'forced-run2'): dict(port=8799, family='deep-forced', program='classify', start='restart-only', stokes='I',
                                      kind='single', hub=None, twin=None),   # Run_P3S4_forced
        ('sublst', 'V'): dict(port=8787, family='review', program='classify', start='stages', stokes='V', kind='single',
                              queue='judgements/queue_p3s3_V.txt', catalog='source_catalogs_V', labels='judgements_V',
                              hub=('V', '10-min'), twin=None, loc_offsets=True, log='server_V.log'),
                              # sublst_cumulative.sh:121-123; sublst_serve.sh:140; ledger 1047
        ('sublst', 'I'): dict(port=8788, family='review', program='classify', start='stages', stokes='I', kind='single',
                              queue='judgements/queue_p3s3_I.txt', catalog='source_catalogs_I', labels='judgements_I',
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
        # the Jupiter program's pages (live on 2026-10-13: 8800 = `jupiter_server.py /lustre/gh/main/phase3/jupiter/products
        # 8800` in cwd ops; 8801 = `python -m http.server 8801 --bind 127.0.0.1 --directory <v2 site>` in cwd the site;
        # jupiter_server.py:303; neither has a launcher in ops). Registry: tunnel_extra (no button)
        ('jupiter', 'p1'): dict(port=8800, family='jupiter', program='jupiter', start='stages', root='p1', hub=None, twin=None,
                                log='jupiter_server_8800.log'),
        ('jupiter', 'v2-site'): dict(port=8801, family='jupiter', program='http', start='stages', root='v2-site', hub=None,
                                     twin=None, log='http_8801.log'),
    }
    JUPITER_ROOTS = {'p1': '/lustre/gh/main/phase3/jupiter/products',     # argv of the live 8800 server (2026-10-13)
                     'v2-site': '/lustre/gh/main/phase3/jupiter/v2/site'}  # --directory of the live 8801 server
    JUPITER_LOG_DIR = '{out}/_records/serve/logs'   # the Jupiter servers' stdout (no log in production: they ran detached)
    TWIN_LOG = 'server_inspect_{port}.log'  # restart_inspection_twin.sh:59 (appended)
    # served by something else: serve never binds, stops or re-points these (the registry rows stay as they are)
    OTHER_SERVED_PORTS = {8767: 'Round-2 reference, frozen (R-GREGG-20261011-KEEP-8767-THRESHOLDS-AT-RERUN, ledger 2199)'}
    PARITY_PORT_OFFSET_MIN = 1000           # a parity --port-offset below this is refused (ports 8765-8801 are production)

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
        'deep': {'P3_DEEP_XTIER': '{deep_xtier}', 'P3_TENMIN_XTIER': '{tenmin_xtier}'},        # p3s3_serve_stokes.sh:78; p3s3_serve_multiples.sh:151
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
        'jupiter': {},                      # the live 8800 / 8801 servers carry no P3_* variable
    }
    STOP_TIMEOUT_S = 10.0                   # s, exact-pid SIGTERM then wait; restart_inspection_twin.sh:53-56 (40 x 0.25 s);
                                            # restart_server_same_argv.sh:11 (20 x 0.5 s); never SIGKILL (no launcher does)
    STOP_POLL_S = 0.25                      # s; restart_inspection_twin.sh:56
    START_SETTLE_S = 2.0                    # s between stop and start; p3s3_serve_stokes.sh:71; sublst_serve.sh:131; snapshot_serve.sh:56
    READY_POLL_S = 5.0                      # s; restart_inspection_twin.sh:69
    READY_TIMEOUT_S = {'deep': 300.0, 'sublst': 300.0, 'snapshot': 400.0, 'jupiter': 300.0}   # s; twins 60 x 5 s
                                            # (restart_inspection_twin.sh:63); snapshot and forced 40 x 10 s
                                            # (snapshot_rechain_batch.sh:416; snapshot_forced_serve_20261007.sh:53);
                                            # jupiter: no production launcher - the deep value
    INDEX_READ_TIMEOUT_S = 120.0            # s, GET of an index page; snapshot_rechain_batch.sh:419
    TWIN_DELAY_S = 3.0                      # s after the judging server starts; p3s3_serve_multiples.sh:155
    # who may be replaced on a page's port: 'port' = any review server on it (the port is the identity); a tuple = only a
    # server whose argv '--run' is <paths root of the tier (RUN_ROOT_KEY)>/<name> with <name> matching one of these shell patterns (else REFUSE).
    # sub-LST: 'Run_SLST*' = sublst_serve.sh:121 grep of '--run .../Run_SLST'; 'Run_P3T4' = SUBLST_TAKEOVER=Run_P3T4
    # (sublst_cumulative.sh:122,132, the archived old-chain pages; exact name, as the trailing space of the grep)
    PORT_OWNERSHIP = {'deep': 'port',       # p3s3_serve_stokes.sh:69 (2026-09-11), p3s3_serve_multiples.sh:145; ledger 1001
                      'sublst': ('Run_SLST*', 'Run_P3T4'),   # sublst_serve.sh:121-122; sublst_cumulative.sh:122,132; ledger 1623
                      'snapshot': 'port',   # snapshot_serve.sh:52-55; forced pages: snapshot_forced_serve_20261007.sh:71
                      'jupiter': 'port'}    # (no launcher) the port is the identity, the program must match SERVER_IDENTITY
    TWIN_TITLE_SUFFIX = ' - INSPECTION (read-only)'   # restart_inspection_twin.sh:43
    TWIN_TITLE_NO_TITLE = '{label_dash}INSPECTION (read-only), {run_name}'   # restart_inspection_twin.sh:49 (judging page
                                            # without --title: the greenlit list's label for the twin port, then this text)

    # ---- titles and hub text ----------------------------------------------------------------------------------------
    UTC_FMT = '%Y-%m-%d %H:%M UT'           # p3s3_serve_stokes.sh:74,84; snapshot_serve.sh:101
    TITLE = {('deep', 'single'): '{tier_text} Stokes {S} transients, wideband + subband, Run_{label} (generation Run_20260909_000000) - served {utc}',
             ('deep', 'multi'): None,       # no --title today (p3s3_serve_multiples.sh:152; interface divergence D2, open)
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
    SUBLST_FIG_THREADS = 4                  # figures drawn at once; sublst_cumulative.sh:57
    SUBLST_FIG_TIMEOUT_S = 600              # s per figure; sublst_cumulative.sh:54
    SUBLST_INDEX_TIERS = ('V_sb', 'I_sb', 'V_wb', 'I_wb')   # p3t_pages_index.py per tier; sublst_cumulative.sh:62-64 (wb optional)
    SUBLST_JOURNALS = ('judgements_V', 'judgements_I', 'judgements_V_multi', 'judgements_I_multi')   # sublst_cumulative.sh:98, 111
    SUBLST_SERVED_TEST_OLD_LSTS = ('19h',)  # LSTs reported apart by test (c); sublst_catalog_publish.sh:19

    # ---- snapshot ---------------------------------------------------------------------------------------------------
    # Serve-time exclusion gate (G17). It RUNS for every tier and page family (tier parsers in lib/serve/exclusion_gate.py)
    # and REFUSES only for the gate keys listed here (key = the tier for its review pages, '<tier>-<family>' otherwise, e.g.
    # 'snapshot-forced'); for the others it records the rows on excluded cell-bands as a report (unit notes, dry-run plan). Snapshot: Gregg 2026-10-13 ruling 2 'Fine.' (RULINGS-GREGG-20261013-DECISIONS-1-7-AND-TENMIN-TABLES,
    # ledger 2297; IMPLEMENTED-SNAPSHOT-SERVE-EXCLUSION-GATE-20261013, ledger 2301; f1996ee). Deep / sub-LST: open (spec 12.8;
    # measured 2026-10-13 ~09:4x UTC on the served Run_P3S5 views + queues: 756 of 311,733 rows on 35 excluded cell-bands,
    # 21h 2025-08-06 the most; sub-LST 0). Production serves deep / sub-LST without a gate: enabling it is Gregg's. PROTECTED.
    EXCLUSION_GATE_TIERS = ('snapshot',)
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

    # ---- snapshot forced photometry pages (port of snapshot_forced_serve_20261007.sh, the 'switch' and 'refresh-siblings'
    # modes; ruled 2026-10-03 / 2026-10-07 Q5/Q7/Q8, ledger 1765) ------------------------------------------------------
    FORCED_RUN_NAME = 'Run_SNAPFORCED_{S}{T}_{tag}'          # T = 10s | 1min; snapshot_forced_serve_20261007.sh:30
    FORCED_READY_LOG = 'snapshot_forced_pages_run_{name}.log'   # under paths.OPS_ROOT; its LAST line must hold FORCED_READY_MARK (:63-65)
    FORCED_READY_MARK = 'FT-READY'
    FORCED_REQUIRED = ('queue_{S}.txt', 'source_catalogs_{S}', 'source_catalogs/localization_offsets.csv',
                       'forced_assoc_identity.csv')          # :66-67
    FORCED_ADLEO_CHECK = ('test_snapshot_forced_reserve.py', ('adleo', '--absent-ok'))   # ops script + args, site PYTHON 'py36' (:68;
                                            # Gregg 2026-10-08 "don't label AD Leo unless you have a metric defined and measured")
    FORCED_TITLE = ('Snapshot FORCED photometry (HWJ + 10 pc), Stokes {S}, {what} - pixel-distance screen, fitted, reference-tied, '
                    'localized, associated (target / background source / ambiguous) - served {utc}')   # :51
    FORCED_SIBLING_LABEL = 'forced Stokes {S} {T} page'      # T = 10-s | 1-min; :34
    FORCED_HUB_GENERATION = ('forced photometry at HWJ + 10 pc targets, Stokes {S} {kind} events: pixel-distance screen, fitted, '
                             'reference-tied, localized, associated')   # kind = 10s | 1min; :97
    FORCED_STOP_TIMEOUT_S = 30.0            # s (30 x 1 s, :57)
    FORCED_RESTORE_LOG = 'classify_{port}_restored.log'   # the old server restarted after a refused switch (:80, 90)

    # ---- in-place calls ---------------------------------------------------------------------------------------------
    LEGACY_TIMEOUT_S = 7200                 # s, any in-place script (sublst_publish_lst.sh:44 ran the cumulative under 7200 s)
    APP_PROBE_TIMEOUT_S = 1200              # s, the server's own loader + App on one page (sources_listed expectation)

    RULED = {
        'PAGES': ('R-PORTS-RECLAIMED-CANONICAL-SET-13-PORTS', 'R-MULTIPLES-OWN-PAGES-8772-8773', 'R-TENMIN-PORTS-8787-V-8788-I',
                  'R-SNAP-1MIN-OWN-PAGES-PER-STOKES'),
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
                       'FIX-REFACTOR-M20-M22-M30-M31-M33-20261013'),
        'PORT_OWNERSHIP': 'PRODUCTION-STATUS-PM-PORT-GUARD-PREP-01H-CUBE-PACE',
        'FORCED_ADLEO_CHECK': 'RULINGS-GREGG-20261008-A-D-STACK-FLOWREF-MEMBERSHIP-GEN4-FORCED-I1MIN',
        'MULTI_MIN_NDET': 'CORRECTION-MULTI-EPOCH-SECTION-WAS-MULTIBAND',
        'DEEP_FORK_TIERS': 'R-MULTIPLES-FORK-INTO-PRODUCTION-PREAMBLE-APPROVED',
        'DEEP_VIEW_REVIEW_DIR': 'FIX-REFACTOR-DEFECTS-D2-D6-D7-D9-D10-AND-RULINGS-D1-D5-D8-20261013',
        'HOLD_FP_VERDICTS': 'R-SNAPSHOT-SERVE-HOLD-ONLY-ON-NON-FP-MISSED-JUDGEMENTS',
        'SNAPSHOT_LOCALIZE': 'R-SNAP-EACH-DETECTION-OWN-FIT-AND-PER-TIER-COMBINED-POSITIONS-IMPLEMENTED',
    }
