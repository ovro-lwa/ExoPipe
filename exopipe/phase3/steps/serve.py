"""exopipe3 serve - review pages and the hub: per-Stokes served queues and views, carried judgements, the pages REGISTERED on
the two-port server (no review page has a server of its own since 2026-10-22), served-page content tests. serve OWNS every
review page (its port is its identity: entry p<port>) and the hub (8765-8801 except the frozen 8767).

    exopipe3 serve --tier deep --run P3S5 --dry-run
    exopipe3 serve --tier deep --run P3S5 --out /data07/gh/refactor_parity/serve --port-offset 10000 --parity
    exopipe3 serve --tier sublst [--lsts 00h,01h,...] --stages build,views,test          (plan a generation, nothing served)
    exopipe3 serve --tier snapshot --run Run_SNAPB2R_20261007 --source lwacalim05:/fast/... --label '...'
    exopipe3 serve --tier snapshot --family forced --forced-tag 20261007 [--pages forced-V-10s]
    exopipe3 serve --tier jupiter                                                         (8800, 8801)
    exopipe3 serve --restart 8800,8801        exopipe3 serve --stop 8801     (servers of their own: the Jupiter pages)

Unit of work: a served PAGE (one unit per page, named <tier>_<run>_<page>) plus one unit per run-level stage (views, test,
build, copy, carry, register, hub, verify ...). Pages (config serve.PAGES; family 'review' unless --family):
  deep      review V 8766, I 8768, V-multi 8772, I-multi 8773 (their former inspection twins 8796, 8797, 8793, 8794 are the
            entries' inspection_legacy_port); strat-I/V 8791/8792 and forced-run3/run2 8798/8799 are restart-only (no
            production launcher defines their argv)
  sublst    review V 8787, I 8788, V-multi 8789, I-multi 8790 (the cumulative Run_SLSTB)
  snapshot  review V-10s 8779, I-10s 8780, V-1min 8781, I-1min 8782; forced pages 8786, 8785, 8775, 8776 (--family forced)
  jupiter   p1 8800 (jupiter_server.py), v2-site 8801 (http.server)
Stages (--stages/--from/--until; default all, in order):
  deep      views, test, carry (only with --carry-from), start, hub, verify
  sublst    build, views, test, carry, swap, start (carry+swap+start only together), hub, verify
  snapshot  gate, copy, localize, gate-copy, carry, test, start, hub, verify
  snapshot --family comb  omit, build, filtercat, localize, ready: THE COMBINED SNAPSHOT VIEW, the snapshot tier's cumulative step
            (lib/serve/snapshot_view.py; spec docs/specs/snapshot_combined_view.md): the deep-omission kept product of the
            --view-runs applied, the batch identities merged with the chain's merge (in place), one page per tier and Stokes
            with hard-linked panels, production's tests T1-T6, the filter catalogue, the per-tier localization; ends
            VIEW-READY-FOR-HIDES (the registration of its pages on p8986-p8989 is not ported: production registers it)
  forced    switch, hub, verify            jupiter   start, hub, verify

Inputs (read, never rewritten unless listed under Outputs):
  deep      Run_<label>/ under paths.TRANSIENTS_ROOT: the tier catalogues source_catalogs_<S>_{wb,sb,fwb,fsb}/, the union
            source_catalogs/<band>_sources_survivors.csv WITH the review_dir column merged by `pages` (refused when absent or
            older than a page index), pages_input_<K>.csv, page indexes, raw queues queue_p3s3_<K>.txt and
            queue_p3s3_<S>_f{wb,sb}_raw.txt, candidates/ (what the server reads)
  sublst    every Run_SLST<hh>B of --lsts (default all found), parent catalogues, the served Run_SLSTB (journals, figures)
  snapshot  the chain run (--source [HOST:]PATH, default site CHAIN_HOST + serve.SNAPSHOT_SOURCE), the served run (journal)
  forced    Run_SNAPFORCED_<S><T>_<TAG> under paths.SNAPSHOT_REVIEW_ROOT and its FT-READY log under paths.OPS_ROOT
  all       data_decisions.csv (read now, the serve-time exclusion gate), inspection_greenlit.csv, the hub registry
Outputs (product classes; content tests run on every write and by `exopipe3 verify <class>`):
  served_queue   sectioned tier queues, queue_p3s3_<S>.txt, queue_p3s3_<S>_multi.txt
  served_view    source_catalogs_<S>/ (deep, sub-LST), views/source_catalogs_<S>_multi/ (deep), source_catalogs_<S>_multi/
  judgement_journal  judgements*/judgements.jsonl after a carry (+ carry_report.csv, carry_missed.csv)
  sublst_cumulative  the Run_SLSTB build (merged CSVs, localizations, figures, page index)
  review_copy    the snapshot review copy;  exclusion_check  the serve-time exclusion gate's pass file;  serve_go  G27's
                 pass file (Gregg's go on a combined view: the go files it names re-read)
  snapshot_view  the combined view (<SNAPSHOT_REVIEW_ROOT>/<view>, parity <out>/snapshot/<view>): V1-V6 read back - queues,
                 page directories, identities against the kept product, the record against the omit stage's;
                 filter_catalog (per detection) and snapshot_card2_union its filter inputs; omission_* its omit product
  hub_page       review hub registry.json / index.html (served on both listeners of the two-port server; the separate
                 inspection hub page and the hub HTTP servers were retired by production on 2026-10-27)
  page_registration  <out>/_records/serve/registrations/<unit>_<UTC>_<hash>.json: the ONE registration of a serve
                 (lib/serve/register.py, the port of /lustre/gh/main/phase3/serve/register_page.py 'register --entry', byte
                 for byte): every page that passed, in one call - registered_pages.json and registry.json written atomically,
                 the server's loader checking the result first, the modification time strictly later - one registry change;
                 the publish marker of each registered run (<run>/PUBLISH_IN_PROGRESS.txt) removed right before it, whatever
                 was refused (as deep_reregister_after_publish_20261031.py); tests: the entries stand as recorded, a change
                 the server sees. Then <unit>_warm.csv (csv_table): the inspection pages built before, requested once
  served_page    <out>/_records/serve/pages/<tier>_<port>.json per served page (its registry entry, the registry and
                 listener, the build state serve waited on, run, queue sha1, view digest, the multiples chain's queue sha1
                 and view digest, expected sources): its tests read the registry and the LIVE two-port server (HTTP 200 at
                 /p/<id>/, the entry and the server's build of it, the index lists the expected sources); a Jupiter page
                 (a server of its own: port, pid, argv, cwd, env);  stopped_page  after --stop
  The deep and 10-minute singles pages carry their MULTIPLES CHAIN (Gregg 2026-10-27; serve.PAGES mchain_queue /
  mchain_catalog -> --mchain-queue / --mchain-catalog-dir): the tier's multiples page queue (deep: and its view), as on disk
  - whatever the multiples page became in the same serve (production's rule: a refused or empty multiples page leaves the
  previous, the refused or an empty chain) - is an input of the page; a page whose chain queue FILE is absent is not served
  (its server would die at start); an absent chain view is noted (the server skips it).
  A parity run (--out outside paths.PRODUCTION_ROOTS) writes everything under --out (deep: a mirror of the served run;
  sub-LST: the build and swap; snapshot / forced: a byte copy) and shifts every port by --port-offset (>= 1000): its pages
  are registered as p<port + offset> in <out>/two_port/registry.json, which a two-port server of its own must serve on
  8765 + offset / 8795 + offset (serve never starts or stops a two-port server; the start stage refuses, printing its
  command line, when none answers); the hub registry and page are written under --out (lib/serve/hub renders the page
  build_review_hub.py writes, exopipe_ops 4c42f91). While paths.PRODUCTION_WRITES is False only parity runs are possible:
  the production registry and a production run's publish hold are never written (G23).
REGISTRATION and BUILD STATE (lib/serve/register.py): the two-port server is checked before anything changes - it must answer,
  report this tree's build state (a server older than this lib/serve/two_port.py reports none and is refused at once, G16: it
  must be redeployed first), read the registry and run with serve.TWO_PORT_PUBLISH_HOLD's mode. After the one registration
  each page is requested once and serve waits on its build state from the server's /_status - done when a build of the
  registered entry that started after the registration began (the time is taken before the registry write) is reported;
  refused at once when that build failed or TWO_PORT_MAX_FAILED_REQUESTS requests in a row got no 200 (G24), the page is held
  or unknown to the server (G16), or another registration replaced its entry (G13); waiting as long as builds start or
  finish on the server, refused (G16) after serve.TWO_PORT_STALL_S without any (not a fixed number of tries). Then every
  inspection page the server had built before the registration is requested once (production's warm-up).
THE PUBLISH MARKER: once the pages merge is current, the deep views stage writes <run>/PUBLISH_IN_PROGRESS.txt (production's deep
  publish marker; a page built meanwhile shows its text) on the run it rewrites in place; the run's registration removes it
  (renamed .removed_<UTC>) whatever was refused; a refused stage leaves it standing. serve.TWO_PORT_PUBLISH_HOLD (an amendment
  option, off = production) runs the server with --publish-hold: it then keeps a marked run's pages on their builds.
Failure modes (exit 2 = refused, the gate named in the record; a refused page keeps its registry entry):
  G1 empty queue; G2 deep union without stokes; G3 deep multiples without the fork; G4 multiples queue != fork catalogues;
  G5-G7 sub-LST Stokes keys / review_dir merge; G8 sub-LST generation mix; G9 localizations / figures; G10 page index;
  G11 nothing swapped in on a refused build; G12 one build at a time (lock); G13 port ownership (a server of its own), the
  two-port server reads another registry, a page's entry replaced while its build was awaited; G14 exact-pid stop failed;
  G16 no HTTP 200 in time (a server of its own), no two-port server answering, one older than this two_port.py (no build
  state), a page unknown to it, held by a publish hold or without build progress for TWO_PORT_STALL_S; G17 serve-time
  exclusion gate (refuses for
  serve.EXCLUSION_GATE_TIERS: a snapshot run; a deep / sub-LST page holding, in what its server reads, a row on an
  excluded night; a snapshot forced page - reports for the keys not listed); G18 carry preconditions; G19 HOLD rule; G20 review
  disk space; G21 snapshot localization; G22 snapshot copy; G23 parity guard (offset, production servers, the production
  registry in the refactor window); G24 the server refused to start / the page's build failed on the two-port server or
  its requests got no 200 TWO_PORT_MAX_FAILED_REQUESTS times in a row;
  G25 not the review host; G26 run label required; G27 the Card 2 / re-judging go (a snapshot run whose combined_view_summary.json
  holds "card2" or "rejudge" is served only when serve.SNAPSHOT_CARD2_REJUDGE_APPROVED names it; before every stage, exopipe_ops
  35aad1a); G28 an earlier stage's record missing or stale; G29 the snapshot serve hold (serve.SNAPSHOT_SERVE_HOLD exists:
  no snapshot run is served; before every stage, exopipe_ops aa100bf);
  the combined view (--family comb): G-V1..G-V6, G-K, T1-T6, F-G1..F-G3 (lib/serve/snapshot_view.py);
  the pages merge absent or stale (deep). Exit 3: nothing to do (no page selected, nothing to carry). Exit 1: error.
  A refused (or errored) run-level stage stops every later stage of the invocation (views/test/carry; build/views/test/
  swap; gate/copy/localize/gate-copy/carry/test): nothing is copied, swapped in or registered after a refusal.
  --resume is not used: a served page is live state - every selected page is re-registered and the verify stage re-tests it.
Per-hour and intermittent use: serve is in the per-hour chain (runner.CHAINS['epoch'], after pages): a page loads its
  catalogues when it is built (classify_transients.py:2493-2510), so adding an hour's sources = re-serving the tier's pages
  from the updated catalogues - one registration, every page rebuilt (--epochs is accepted and recorded; the work is the
  same re-serve). Survey use: after a full publish pass (deep), a cumulative rebuild (sub-LST), a chain batch (snapshot).
"""
import os
import sys

SUMMARY = 'review pages registered on the two-port server and the hub; served-queue and exclusion content tests'
TIERS = ('deep', 'sublst', 'snapshot', 'jupiter')
SELECTORS = ('tier', 'run_label', 'stokes', 'lsts', 'epochs')
CONFIG = ('serve', 'pages', 'catalog_omission', 'catalog_sublst', 'catalog_snapshot', 'exclusions', 'paths', 'runner', 'site')
WRITES = True
FAMILIES = ('review', 'forced', 'strat', 'deep-forced', 'jupiter', 'comb')


def add_arguments(p):
    g = p.add_argument_group('serve')
    g.add_argument('--family', choices=FAMILIES, default=None,
                   help="page family of the tier (default: review; jupiter for --tier jupiter): forced = the snapshot forced "
                        "photometry pages; strat / deep-forced = restart-only deep pages; comb = the COMBINED SNAPSHOT VIEW, the "
                        "snapshot tier's cumulative step (lib/serve/snapshot_view.py: stages omit, build, filtercat, localize, "
                        "ready)")
    g.add_argument('--view-runs', metavar='RUN,RUN|@FILE',
                   help='--family comb: the served batch runs of the view, in order (directories, a comma list or @FILE; bare '
                        'names under paths.SNAPSHOT_REVIEW_ROOT)')
    g.add_argument('--view-omit', metavar='DIR',
                   help='--family comb: an omit product made for exactly these runs (deep_omission_kept.csv with its decisions, '
                        'record and summary; `catalog --tier snapshot --stages omit`) - the omit stage is then not run; default '
                        'the view\'s own omission directory serve.SNAPVIEW_OMIT_DIR, written by the omit stage')
    g.add_argument('--pages', help='comma list of page keys to (re)serve (default every page of the family; see serve.PAGES)')
    g.add_argument('--source', help='snapshot: the chain run to copy, [HOST:]PATH (default site CHAIN_HOST + '
                                    'serve.SNAPSHOT_SOURCE)')
    g.add_argument('--label', help='snapshot: the serve label (page titles, hub generation)')
    g.add_argument('--carry-from', metavar='RUN|served|none',
                   help="judgement carry source: a run directory, 'served' (the run on the tier's first port) or 'none'. "
                        "Default: deep none; sub-LST the served Run_SLSTB (journal copy); snapshot 'served'")
    g.add_argument('--port-offset', type=int, default=0,
                   help='parity: bind every port at port + OFFSET (>= serve.PARITY_PORT_OFFSET_MIN; required outside the '
                        'production roots); link targets inside the cross-tier strings stay on the production ports')
    g.add_argument('--no-hub', action='store_true', help='skip the hub registry and pages')
    g.add_argument('--forced-tag', help='forced family: TAG of the runs Run_SNAPFORCED_<S><T>_<TAG>')
    g.add_argument('--forced-refresh-siblings', action='store_true',
                   help='forced family: restart the pages whose --sibling links differ from what the other ports serve now')
    g.add_argument('--restart', metavar='PORTS', help='restart the servers on these ports with their exact argv, cwd, log '
                                                     'and environment (a server of its own: the Jupiter pages; a registered '
                                                     'page is rebuilt by serving it again); nothing else runs')
    g.add_argument('--stop', metavar='PORTS', help='stop the servers on these ports by exact pid; nothing else runs')
    g.add_argument('--stages', help='comma list of stages (default all, in order; see above)')
    g.add_argument('--from', dest='from_stage', help='first stage')
    g.add_argument('--until', help='last stage')


def default_out(args):
    from exopipe.phase3.config import CFG
    return CFG.paths.SNAPSHOT_REVIEW_ROOT if getattr(args, 'tier', 'deep') == 'snapshot' else CFG.paths.TRANSIENTS_ROOT


def _module(plan):
    from exopipe.phase3.lib.serve import deep, forced, jupiter, snapshot_copy, sublst_build
    if plan.family == 'forced':
        return forced
    return {'deep': deep, 'sublst': sublst_build, 'snapshot': snapshot_copy, 'jupiter': jupiter}[plan.tier]


def select_stages(all_stages, args):
    from exopipe.phase3.core.record import Refusal
    st = list(all_stages)
    if args.stages:
        want = [s.strip() for s in args.stages.split(',') if s.strip()]
        bad = [s for s in want if s not in all_stages]
        if bad:
            raise Refusal(f'--stages {",".join(bad)}: stages of this tier are {", ".join(all_stages)}')
        st = [s for s in all_stages if s in want]
    if args.from_stage:
        if args.from_stage not in all_stages:
            raise Refusal(f'--from {args.from_stage}: stages are {", ".join(all_stages)}')
        st = [s for s in st if all_stages.index(s) >= all_stages.index(args.from_stage)]
    if args.until:
        if args.until not in all_stages:
            raise Refusal(f'--until {args.until}: stages are {", ".join(all_stages)}')
        st = [s for s in st if all_stages.index(s) <= all_stages.index(args.until)]
    return st


def _ports(spec):
    from exopipe.phase3.core.record import Refusal
    try:
        return [int(x) for x in spec.split(',') if x.strip()]
    except ValueError:
        raise Refusal(f'expected a comma list of ports, got {spec!r}')


def restart_or_stop(args, run):
    from exopipe.phase3.lib.serve import flow, servers
    out = os.path.abspath(run.out_root)
    offset = int(args.port_offset or 0)
    action = 'restart' if args.restart else 'stop'
    for port in _ports(args.restart or args.stop):
        if run.dry_run:
            print(f'PLAN --{action} {port}: {flow.describe_port(port)}')
            continue
        with run.unit(f'{action}_{port}', action=action, port=port) as u:
            notes = []
            if action == 'restart':
                for path, rec in servers.restart_port(port, out, notes, offset=offset):
                    u.add_output(path, 'served_page')
            else:
                path, rec = servers.stop_port(port, out, notes, offset=offset)
                u.add_output(path, 'stopped_page')
            for n in notes:
                u.note(n)
    return 0 if run.dry_run else None


def dry_run(plan, mod, stages, args):
    import json
    from exopipe.phase3.lib.serve import common, flow
    flow.print_plan_header(plan, stages)
    utc = common.utc_text()
    for pg in plan.pages:
        if pg.program == 'classify':
            print(f'  page {pg.key}: p{pg.port} [{flow.describe_page(plan, pg)}]; record {pg.record_path()}')
            if pg.family == 'review':
                q = pg.at('queue', plan.U_src)
                print(f'    queue {q}: ' + (f'{len(common.queue_ids(q))} ids' if os.path.exists(q) else 'absent (built by views)'))
            if pg.start == 'stages' and pg.family == 'review':
                from exopipe.phase3.lib.serve import register
                e = register.page_entry(plan, pg, utc)
                print(f'    entry: code {e["code"]}; argv {" ".join(e["argv"])}; env {json.dumps(e["env"])}; '
                      f'legacy_port {e["legacy_port"]}, inspection_legacy_port {e["inspection_legacy_port"]}')
            continue
        print(f'  page {pg.key}: port {pg.port} [{flow.describe_port(pg.port)}]; record {pg.record_path()}')
        if pg.start == 'stages':
            argv = plan.argv(pg, utc)
            print(f'    argv: {" ".join(argv)}')
            print(f'    cwd {pg.cwd()}; log {pg.log_path()}')
    mod.describe(plan, stages, args)
    for n in plan.notes:
        print(f'  note: {n}')
    return 0


def run(args, run):
    from exopipe.phase3.core import injected
    from exopipe.phase3.core.record import Refusal
    from exopipe.phase3.lib.serve import plan as planmod
    # an INJECTED run is never served (core/injected.py; DESIGN 11.2 'search --inject: recorded, never served'): its pages
    # carry synthetic sources - no server, registry, hub, journal or restart, parity ports included
    injected.refuse('serve', 'an injected run is never served (served run, serve registry, hub, judgement journals, restarts)')
    if args.restart or args.stop:
        if args.restart and args.stop:
            raise Refusal('give --restart or --stop, not both')
        return restart_or_stop(args, run)
    if getattr(args, 'family', None) == 'comb':        # the combined snapshot view: a cumulative build, no page of its own yet
        if args.tier != 'snapshot':
            raise Refusal(f'--family comb is the snapshot tier\'s combined view (--tier snapshot), not --tier {args.tier}')
        from exopipe.phase3.lib.serve import snapshot_view
        return snapshot_view.run(args, run)
    plan = planmod.make(args, run.out_root)
    if not plan.pages:
        from exopipe.phase3.core.record import NoInput
        raise NoInput(f'no page selected (tier {plan.tier}, family {plan.family}, --pages {args.pages}, --stokes {args.stokes})')
    mod = _module(plan)
    stages = select_stages(mod.STAGES, args)
    if getattr(args, 'epochs', None):
        from exopipe.phase3.core import cells
        plan.notes.append(f'epochs {cells.expand_epochs(args.epochs)}: a re-serve of the tier pages (the servers load '
                          f'their catalogues at start)')
    if run.dry_run:
        return dry_run(plan, mod, stages, args)
    if plan.gates:
        raise Refusal('; '.join(f'{g}: {m}' for g, m in plan.gates))
    mod.run(args, run, plan, stages)
    return None


if __name__ == '__main__':
    sys.exit('run through exopipe3: python -m exopipe.phase3 serve --help')
