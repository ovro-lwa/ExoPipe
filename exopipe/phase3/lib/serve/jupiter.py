"""serve --tier jupiter: the Jupiter program's pages 8800 (jupiter_server.py ROOT PORT, cwd ops) and 8801 (the v2 site,
python -m http.server PORT --bind 127.0.0.1 --directory DIR). serve owns these ports and the hub (refactor-chat decision
2026-10-13); the `jupiter` step builds what they show. Neither page has a production launcher: the argv is the live
servers' (2026-10-13, recorded in serve.PAGES / JUPITER_ROOTS). Stages start -> hub -> verify.

start  (one unit per page)  G13 the port holds nothing or a server of the page's program (serve.SERVER_IDENTITY); exact-pid
       stop; start; HTTP 200 (G16); record class served_page (tests live, same_process; sources_listed n/a).
hub    the ports in the registry's tunnel_extra (build_review_hub.py --tunnel-extra; no button, as today).
verify the served_page tests now.
Parity: the offset ports serve the production directories read-only.
"""
from exopipe.phase3.lib.serve import common, flow, servers

STAGES = ('start', 'hub', 'verify')


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
