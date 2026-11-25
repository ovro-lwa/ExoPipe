"""serve --tier deep|sublst --family pinned: the PINNED pages - registered once by hand, no launcher and no build of their own, a
fixed entry each (config serve.PAGES family 'pinned': run, queue, labels, catalogue, page code, environment, title, trailing
arguments):
  deep    fp-sample-V 9411, fp-sample-I 9412: the false-positive ground-truth samples 1 and 2 of 4 on Run_P3S5 (blind:
          --filter-defaults off), registered 2026-10-22;
  sublst  fp-sample-V 9413, fp-sample-I 9414: samples 3 and 4 on Run_SLSTB, served by the frozen page code of that build;
          groupB-preview-I 9300, groupB-preview-V 9301: the grouping design B preview Run_SLSTB_GRPB_PREVIEW_20261026, served by
          its own page code (2026-10-26).
register: unit <tier>_pinned_register (flow.register_pages): the two-port server is checked first, then per requested page (a
  page that fails keeps its entry): its run, queue, catalogue view and page code exist; the serve-time exclusion gate on the
  page (lib/serve/exclusion_gate.scan_server_page: queue, view, offsets; gate key '<tier>-pinned', refusing); the expected
  index sources (the page's own code: servers.app_probe); then ONE registration of the pages that passed (their entries as
  configured: register.page_entry), one unit per page waiting on its build (class served_page). Nothing in the run is
  written. No hub button (they are linked from no hub tier).
verify: the served_page records re-tested.
Production only: a parity serve is refused (the pages' label journals are production's; a copy of Run_P3S5 is not made for
them) - test_serve compares the planned entries with the live registry instead.
"""
import os

from exopipe.phase3.core.record import Refusal
from exopipe.phase3.lib.serve import common, exclusion_gate, flow, register, servers

STAGES = ('register', 'verify')


def check_page(u, plan, pg):
    """The checks of one pinned page; returns (run, expected sources); raises Refusal (the page keeps its entry)."""
    U = pg.root
    code = pg.c.get('code') or os.path.join(plan.ops, 'classify_transients.py')
    miss = [p for p in (U, pg.at('queue', U), pg.at('catalog', U), code) if not os.path.exists(p)]
    if miss:
        raise Refusal(f'{", ".join(miss)} missing - entry kept')
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
                                               inc_log=False))


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
        code = pg.c.get('code') or os.path.join(plan.ops, 'classify_transients.py')
        miss = [p for p in (U, pg.at('queue', U), pg.at('catalog', U), code) if not os.path.exists(p)]
        print(f'  {pg.key}: p{pg.port} run {U}; code {code}; files {"ok" if not miss else "MISSING " + ", ".join(miss)}; '
              f'registry serves {plan.registered_run(pg) or "-"}')
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
