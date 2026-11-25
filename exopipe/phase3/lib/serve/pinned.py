"""serve --tier deep|sublst --family pinned: the PINNED pages - registered once by hand, no launcher and no build of their own, a
fixed entry each (config serve.PAGES family 'pinned': run, queue, labels, catalogue, page code, environment, title, trailing
arguments):
  deep    fp-sample-V 9411, fp-sample-I 9412: the false-positive ground-truth samples 1 and 2 of 4 on Run_P3S5 (blind:
          --filter-defaults off), registered 2026-10-22;
  sublst  fp-sample-V 9413, fp-sample-I 9414: samples 3 and 4 on Run_SLSTB, served by the frozen page code of that build;
          groupB-preview-I 9300, groupB-preview-V 9301: the grouping design B preview Run_SLSTB_GRPB_PREVIEW_20261026, served by
          its own page code (2026-10-26).
register: unit <tier>_pinned_register (flow.register_pages): the two-port server is checked first, then per requested page (a
  page that fails keeps its entry): its run, queue, catalogue view and page code exist; no publish marker stands on its run
  (Run_P3S5 / Run_SLSTB belong to the deep and sub-LST publishers: this family never removes their marker - a marker refuses,
  'publish in progress on <run>, retry after', here and again inside the registry lock, flow.register_pages holds='refuse');
  the serve-time exclusion gate on the page (lib/serve/exclusion_gate.scan_server_page: queue, view, offsets; gate key
  '<tier>-pinned', refusing); the expected index sources (the page's own code: servers.app_probe); then ONE registration of the
  pages that passed (their entries as configured: register.page_entry), one unit per page waiting on its build (class
  served_page). Nothing in the run is written. No hub button (they are linked from no hub tier).
The entry: plan.ReviewHook's argv with the page's own code (absent: serve.SERVER_ARGV0 of the tier), title, trailing arguments
  and environment (absent: the tier's serve.SERVER_ENV) - PinnedHook, this family's PLAN_HOOK.
verify: the served_page records re-tested.
Production only: a parity serve is refused (the pages' label journals are production's; a copy of Run_P3S5 is not made for
them) - test_serve compares the planned entries with the live registry instead.
"""
import os

from exopipe.phase3.core.record import Refusal
from exopipe.phase3.lib.serve import common, exclusion_gate, flow, register, servers
from exopipe.phase3.lib.serve.plan import ReviewHook

STAGES = ('register', 'verify')


class PinnedHook(ReviewHook):
    """PLAN_HOOK of the pinned pages: the review family's argv with the page's own code, title, trailing arguments and
    environment."""

    def code(self, plan, page):
        return page.c.get('code') or super().code(plan, page)

    def env(self, plan, page):
        return page.c['env'] if page is not None and page.c.get('env') is not None else super().env(plan, page)

    def title(self, plan, page, utc, root=None):
        return page.c['title']

    def extra(self, plan, page):
        return list(page.c.get('extra') or ())      # p9411 / p9412: --filter-defaults off

    def hub_words(self, plan, page, root=None):
        raise Refusal(f'pinned page {page.key}: no hub button (linked from no hub tier)')


PLAN_HOOK = PinnedHook()


def hold_refusal(run_dir):
    """The refusal text when a publish marker stands on ``run_dir`` (register.hold_file), else None."""
    h = register.hold_file(run_dir)
    if os.path.lexists(h):
        return f'publish in progress on {run_dir}, retry after (its publish marker {h} stands) - entry kept'
    return None


def check_page(u, plan, pg):
    """The checks of one pinned page; returns (run, expected sources); raises Refusal (the page keeps its entry)."""
    U = pg.root
    code = plan.code(pg)
    miss = [p for p in (U, pg.at('queue', U), pg.at('catalog', U), code) if not os.path.exists(p)]
    if miss:
        raise Refusal(f'{", ".join(miss)} missing - entry kept')
    held = hold_refusal(U)
    if held:
        raise Refusal(held)
    scan = exclusion_gate.scan_server_page(pg, U, tier=plan.tier)
    refuses = exclusion_gate.refuses(plan.tier, 'pinned')
    u.note(f'page {pg.key}: G17 serve-time exclusion gate ({"refusing" if refuses else "report only"}): {scan.summary()}')
    if scan.hits and refuses:
        raise Refusal(f'G17 serve-time exclusion gate: {scan.summary()} - entry kept')
    return U, servers.app_probe(plan, pg, U, unit=u)


def register_stage(run, plan, state, utc):
    def prepare(u):
        flow.excl_note(u, plan.tier)
        items, refused = [], {}
        for pg in plan.pages:
            try:
                U, exp = check_page(u, plan, pg)
                items.append((pg, U, register.page_entry(plan, pg, utc, U), exp, []))
            except Refusal as e:
                refused[pg.key] = str(e)
                u.note(f'page {pg.key} REFUSED: {e}')
        return items, refused
    state['served'].update(flow.register_pages(run, plan, prepare, f'{plan.tier}_pinned_register', stage='register',
                                               inc_log=False, holds='refuse'))


def run(args, run, plan, stages):
    if plan.parity:
        raise Refusal('--family pinned registers the production pages only (their journals and runs are production\'s): no '
                      'parity serve - test_serve compares the planned entries with the live registry')
    state = flow.new_state()
    if 'register' in stages:
        register_stage(run, plan, state, common.utc_text())
    if 'verify' in stages:
        flow.verify_stage(run, plan)


def describe(plan, stages, args):
    """The dry run: per page its run, files, page code, the exclusion gate's scan and the entry it would register. Writes
    nothing."""
    for pg in plan.pages:
        U = pg.root
        code = plan.code(pg)
        miss = [p for p in (U, pg.at('queue', U), pg.at('catalog', U), code) if not os.path.exists(p)]
        print(f'  {pg.key}: p{pg.port} run {U}; code {code}; files {"ok" if not miss else "MISSING " + ", ".join(miss)}; '
              f'registry serves {plan.registered_run(pg) or "-"}' + (f'; {hold_refusal(U)}' if hold_refusal(U) else ''))
        if miss:
            continue
        try:
            scan = exclusion_gate.scan_server_page(pg, U, tier=plan.tier)
            print(f'    G17 exclusion gate: {scan.summary()}'
                  + (' - REFUSED' if scan.hits and exclusion_gate.refuses(plan.tier, 'pinned') else ''))
            e = register.page_entry(plan, pg, common.utc_text(), U)
            print(f'    entry: code {e["code"]}; argv {" ".join(e["argv"])}; env {e["env"]}')
        except Refusal as e:
            print(f'    REFUSED: {e}')
