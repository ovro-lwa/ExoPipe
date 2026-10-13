"""serve --tier deep (review pages 8766/8768/8772/8773 + twins 8796/8797/8793/8794): views -> test -> carry -> start ->
twins -> hub -> verify.

views  (unit deep_<run>_views)  the parity mirror (parity only); the pages merge must be current (views.deep_merge_current:
       pages owns the review_dir merge, decision 6 group 12); per selected singles Stokes: section the wb/sb tier queues
       (p3s3_serve_tier.sh:13-32), p3s3_stokes_queue.py in place (p3s3_serve_stokes.sh:11), the per-Stokes view
       (p3s3_serve_stokes.sh:15-55, review_dir per serve.DEEP_VIEW_REVIEW_DIR); per selected multiples Stokes: the fork
       queue and view (p3s3_serve_multiples.sh:14-131; G3 refuses that page only) and multiples_fork.py served <S> in place
       (G4, p3s3_serve_multiples.sh:136-137). Outputs: served_queue, served_view (content tests run on write).
test   (unit deep_<run>_test)   G1, G17 (report only for deep: serve.EXCLUSION_GATE_TIERS), the content tests again, the
       App probe of every page (expected index sources; an unexplained 404 refuses the page).
carry  (unit deep_<run>_carry, only with --carry-from)  deep_carry_judgements.py OLD U in place (G18).
start  (one unit per page)       G13 / G14 / G16 / G24 in servers.replace; record class served_page.
twins  (one unit per twin)       restart_inspection_twin.sh port (G15, M11 environment).
hub    (unit deep_<run>_hub)     build_review_hub.py --set per served page (locked; parity: planned) + inspection hub.
verify (unit deep_<run>_verify)  the served_page tests now.
"""
import os

from exopipe.phase3.core.record import Refusal
from exopipe.phase3.lib.serve import common, exclusion_gate, flow, journals, mirror, queues, scripts, views

STAGES = ('views', 'test', 'carry', 'start', 'twins', 'hub', 'verify')


def _S():
    from exopipe.phase3.config import CFG
    return CFG.serve


def _views_unit(plan):
    return f'deep_{plan.run_name}_views'


def _test_unit(plan):
    return f'deep_{plan.run_name}_test'


def views_stage(run, plan, state):
    S = _S()
    U = plan.U
    with run.unit(_views_unit(plan), stage='views', tier='deep', run=plan.run_name, parity=plan.parity) as u:
        flow.excl_note(u)
        if plan.parity:
            m = mirror.build(plan.U_src, U)
            u.note(f'parity mirror of {plan.U_src}: copied {", ".join(m["copied"])}; {m["linked"]} entries linked')
        ok, detail = views.deep_merge_current(U)
        u.check('pages_merge_current', ok, detail=detail)
        if not ok:
            u.refuse(f'the review_dir merge of `pages` is absent or stale: {detail} - run `exopipe3 pages --tier deep '
                     f'--stages merge --run {plan.label}` first (decision 6: pages writes the merge, serve reads it)')
        u.note(detail)
        singles = [pg.stokes for pg in plan.pages if pg.kind == 'single']
        multis = [pg for pg in plan.pages if pg.kind == 'multi']
        for ST in ('V', 'I'):
            if ST not in singles:
                continue
            for t in S.DEEP_SINGLES_TIERS:
                K = f'{ST}_{t}'
                pin = f'{U}/source_catalogs/pages_input_{K}.csv'
                if not os.path.exists(queues.tier_queue_path(U, K)):
                    u.note(f'no queue for {K} (tier not built) - skipped as p3s3_serve_tier.sh:17')
                    continue
                u.add_input(pin)
                sec = queues.section_tier_queue(U, K, 'deep')
                u.note(f'{K} multi {sec["multi"]} single {sec["single"]} marginal {sec["marginal"]}')
                u.add_output(queues.tier_queue_path(U, K), 'served_queue')
            res = scripts.run('stokes_queue', [U, ST], log=plan.log_path, unit=u)
            if not res.ok:
                u.refuse(f'p3s3_stokes_queue.py {ST} failed: {res.reason or scripts.last_line(res.tail)}')
            u.add_output(f'{U}/judgements/queue_p3s3_{ST}.txt', 'served_queue')
            v = views.deep_stokes_view(U, ST, review_dir=S.DEEP_VIEW_REVIEW_DIR)
            u.note(f'{ST} catalog view: {v["rows"]} survivors rows in {v["dir"]}; review_dir re-resolved in the own-Stokes '
                   f'index (D9 residual): {len(v["review_dir_changed"])} rows changed {v["review_dir_changed"][:3]}')
            u.meta.setdefault('d9_residual_changed', {})[ST] = v['review_dir_changed']
            u.add_output(v['dir'], 'served_view', tier='deep')
        for pg in multis:
            ST = pg.stokes
            try:
                qpath, keep, secs = queues.deep_multi_queue(U, ST)
            except Refusal as e:
                state['refused'][pg.key] = str(e)
                u.note(str(e))
                continue
            v = views.deep_multi_view(U, ST, keep)
            u.note(f'{ST} multiples queue: {sum(len(i) for _, i in secs)} ids ({"; ".join(str(len(i)) for _, i in secs)}); '
                   f'view {v["rows"]} survivors rows in {v["dir"]}')
            if not common.has_j_line(qpath):
                state['refused'][pg.key] = f'G1 empty queue: {qpath} - not served'
                continue
            res = scripts.run('multiples_fork', ['served', ST], log=plan.log_path, unit=u, refusal_codes=(1, 2),
                              env={'P3S_LABEL': plan.label, 'FORK_RUN_DIR': U, 'FORK_OUT_DIR': U})
            u.check(f'G4_fork_served_{ST}', res.ok, detail=scripts.last_line(res.tail))
            if not res.ok:
                state['refused'][pg.key] = (f'G4 served-queue content test FAILED - not served '
                                            f'({scripts.last_line(res.tail)})')
                continue
            u.add_output(qpath, 'served_queue')
            u.add_output(v['dir'], 'served_view', tier='deep')
        dirs = [pg.at('catalog') for pg in plan.pages]
        scan = exclusion_gate.scan_tier_views(U, dirs, [pg.at('queue') for pg in plan.pages])
        u.meta['exclusion_report'] = {' '.join(k): n for k, n in scan.per_key().items()}
        u.note(f'G17 serve-time exclusion gate (report only for deep until Gregg rules, spec 12.8): {scan.summary()}')
        u.meta['refused_pages'] = dict(state['refused'])


def carry_stage(run, plan, args):
    spec = getattr(args, 'carry_from', None)
    if not spec:
        return
    with run.unit(f'deep_{plan.run_name}_carry', stage='carry', tier='deep') as u:
        old = journals.resolve_carry_from(plan, spec)
        u.note(f'carry from {old}')
        lines = journals.deep_carry(old, plan.U, log_path=plan.log_path, unit=u)
        for l in lines:
            u.note(l)
        n = 0
        for ST in ('V', 'I'):
            j = os.path.join(plan.U, f'judgements_{ST}', 'judgements.jsonl')
            if os.path.exists(j):
                u.add_output(j, 'judgement_journal')
                n += 1
        if not n:
            from exopipe.phase3.core.record import NoInput
            raise NoInput(f'no judgement to carry from {old}')


def run(args, run, plan, stages):
    state = flow.new_state()
    if 'views' in stages:
        views_stage(run, plan, state)
    elif any(s in stages for s in ('test', 'start')):
        rec = flow.require_done(run, _views_unit(plan), 'the deep views')
        state['refused'].update((rec.get('meta') or {}).get('refused_pages') or {})
    if 'test' in stages:
        flow.test_stage(run, plan, state, _test_unit(plan))
    elif 'start' in stages:
        exp, refused = flow.expected_from_record(run, _test_unit(plan), 'start')
        state['expected'].update(exp)
        state['refused'].update(refused)
    if 'carry' in stages:
        carry_stage(run, plan, args)
    if 'start' in stages:
        flow.start_stage(run, plan, state)
    if 'twins' in stages and not getattr(args, 'no_twins', False):
        flow.twins_stage(run, plan, state)
    if 'hub' in stages and not getattr(args, 'no_hub', False):
        flow.hub_stage(run, plan, state)
    if 'verify' in stages:
        flow.verify_stage(run, plan)


def describe(plan, stages, args):
    """Dry-run plan of the deep tier (read-only)."""
    U = plan.U_src
    ok, detail = views.deep_merge_current(U) if os.path.isdir(U) else (False, f'{U} absent')
    print(f'  pages merge (written by `pages`): {"current" if ok else "WOULD REFUSE"} - {detail}')
    import csv
    import glob
    nost = []
    for p in sorted(glob.glob(f'{U}/source_catalogs/*_sources_survivors.csv')):
        with open(p) as f:
            if 'stokes' not in (csv.DictReader(f).fieldnames or []):
                nost.append(os.path.basename(p))
    print('  G2 union with a stokes column: ' + ('ok' if not nost else f'WOULD REFUSE - {len(nost)} union files carry no '
                                                  f'stokes column (built before the D9 fix 09995b5), e.g. {nost[0]}'))
    dirs = [pg.at('catalog', U) for pg in plan.pages]
    scan = exclusion_gate.scan_tier_views(U, dirs, [pg.at('queue', U) for pg in plan.pages])
    print(f'  G17 serve-time exclusion gate on the served views (report only for deep): {scan.summary()}')
    for ST in ('V', 'I'):
        for t in _S().DEEP_SINGLES_TIERS:
            q = queues.tier_queue_path(U, f'{ST}_{t}')
            print(f'  tier queue {ST}_{t}: ' + (f'{len(common.queue_ids(q))} ids' if os.path.exists(q) else 'absent'))
        fr = queues.fork_raw_queues(U, ST)
        print(f'  fork raw queues {ST}: ' + (', '.join(os.path.basename(p) for _, p in fr) if fr else
                                             'NONE - the multiples page would refuse (G3)'))
    if getattr(args, 'carry_from', None):
        print(f'  carry from: {getattr(args, "carry_from")}')
