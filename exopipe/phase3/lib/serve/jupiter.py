"""serve --tier jupiter: the Jupiter program's pages 8800 (jupiter_server.py ROOT PORT, cwd ops) and 8801 (the v2 site,
python -m http.server PORT --bind 127.0.0.1 --directory DIR; py310_orca, cwd the site's parent: its standing command since
2026-11-14, ledger OPS-20261114-JOVIAN-8801-STANDING-COMMAND). serve owns these ports and the hub (refactor-chat decision
2026-10-13); the `jupiter` step builds what they show. Neither page has a production launcher: the argv is the live
servers' (2026-10-13 and 2026-11-14, recorded in serve.PAGES / JUPITER_ROOTS). Stages start -> hub -> verify.

start  (one unit per page)  G13 the port holds nothing or a server of the page's program (serve.SERVER_IDENTITY); exact-pid
       stop; start; HTTP 200 (G16); record class served_page (tests live, same_process; sources_listed n/a).
hub    the ports in the registry's tunnel_extra (build_review_hub.py --tunnel-extra; no button, as today - the hub page
       prints no tunnel line since exopipe_ops 4c42f91 and links the Jovian site by its static path, /s/jupiter/).
verify the served_page tests now.
Parity: the offset ports serve the production directories read-only.
"""
from exopipe.phase3.core.record import Refusal
from exopipe.phase3.lib.serve import common, flow, servers

STAGES = ('start', 'hub', 'verify')


def _CFG():
    from exopipe.phase3.config import CFG
    return CFG


class JupiterHook:
    """PLAN_HOOK of the Jupiter program's pages: the live servers' argv (serve.PAGES; module docstring), environment
    serve.SERVER_ENV['jupiter'], no title, no siblings, no hub button."""

    def code(self, plan, page):
        if page.program != 'jupiter':
            raise Refusal(f'page {page.key}: python -m http.server has no code file of its own')
        return 'jupiter_server.py'                   # run from the ops root (Page.cwd)

    def env(self, plan, page):
        return _CFG().serve.SERVER_ENV['jupiter']

    def title(self, plan, page, utc, root=None):
        return None

    def siblings(self, plan, page, root=None):
        return []

    def argv(self, plan, page, utc, root=None):
        py = plan.page_python(page)
        if page.program == 'jupiter':
            return [py, 'jupiter_server.py', page.root, str(page.port)]
        return [py, '-m', 'http.server', str(page.port), '--bind', '127.0.0.1', '--directory', page.root]

    def hub_words(self, plan, page, root=None):
        raise Refusal(f'page {page.key}: the Jupiter pages have no hub button (registry tunnel_extra)')


PLAN_HOOK = JupiterHook()


def start_stage(run, plan, state, utc):
    for pg in plan.pages:
        with run.unit(pg.unit, stage='start', tier='jupiter', page=pg.key, port=pg.port, parity=plan.parity) as u:
            notes = []
            path, rec = servers.replace(plan, pg, utc, notes=notes)
            for n in notes:
                u.note(n)
            u.add_output(path, 'served_page')
            state['served'][pg.key] = (path, rec)


def run(args, run, plan, stages):
    state = flow.new_state()
    utc = common.utc_text()
    if 'start' in stages:
        start_stage(run, plan, state, utc)
    if 'hub' in stages and not getattr(args, 'no_hub', False):
        flow.hub_stage(run, plan, state, utc=utc)
    if 'verify' in stages:
        flow.verify_stage(run, plan)


def describe(plan, stages, args):
    import os
    for pg in plan.pages:
        print(f'  {pg.key}: serves {pg.root}' + ('' if os.path.isdir(pg.root) else '   [ABSENT]'))
