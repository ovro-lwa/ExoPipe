"""serve --tier deep (review pages p8766/p8768/p8772/p8773 on the two-port server; inspection: the greenlit entries on its
inspection listener, the former twins' ports 8796/8797/8793/8794 their inspection_legacy_port): views -> test -> carry ->
start -> hub -> verify (+ page-index alone).

views  (unit deep_<run>_views)  the parity mirror (parity only); the pages merge must be current (views.deep_merge_current:
       pages owns the review_dir merge, decision 6 group 12); then THE PUBLISH MARKER on the served run (register.set_hold,
       <run>/PUBLISH_IN_PROGRESS.txt, production's deep publish marker; one that stands is kept): a page built while this serve
       rewrites the run in place shows its text (with serve.TWO_PORT_PUBLISH_HOLD the server keeps the run's pages on their
       builds instead) until the run's registration removes it; per selected singles Stokes: section the wb/sb tier queues
       (p3s3_serve_tier.sh:13-32), p3s3_stokes_queue.py in place (p3s3_serve_stokes.sh:11), the per-Stokes view
       (p3s3_serve_stokes.sh:15-55, review_dir per serve.DEEP_VIEW_REVIEW_DIR); per selected multiples Stokes: the fork
       queue and view (p3s3_serve_multiples.sh:14-131; G3 refuses that page only) and multiples_fork.py served <S> in place
       (G4, p3s3_serve_multiples.sh:136-137). Outputs: served_queue, served_view (content tests run on write).
test   (unit deep_<run>_test)   G1, G17 per page (serve.EXCLUSION_GATE_TIERS), the content tests again, the
       App probe of every page (expected index sources; an unexplained 404 refuses the page).
carry  (unit deep_<run>_carry, only with --carry-from)  deep_carry_judgements.py OLD U in place (G18).
start  (unit deep_<run>_register, then one unit per page, then deep_<run>_register_warm)  the ONE registration of the pages
       that passed (flow.register_pages: G16 / G13 server check, the publish marker removed, register.write - class
       page_registration), then each page's build state (G24 build failed, G16 no progress / held, G13 replaced) and its
       served_page record, then the inspection pages built before requested once (deep_reregister_after_publish_20261031.py,
       p3s3_serve_stokes.sh with P3S3_NO_SERVER=1: no server of its own).
       Before the registry lock, the run's page index (flow.build_page_index, class detections_index; kept when current; built
       under the standing publish marker, lifted in the lock as before): a failure refuses the registration, the marker stands.
page-index (unit deep_<run>_page_index; only without start)  build_detections_index.py --run U on the served run.
hub    (unit deep_<run>_hub)     build_review_hub.py --set per served page (locked; parity: planned).
verify (unit deep_<run>_verify)  the served_page tests now.

THE VERSIONED DEEP RUN (lib/serve/versions.py, the pointer of the sub-LST runs; plan/CUTOVER_REMAINING_20261124.md row 8):
production rewrites the served deep run in place while its pages read it (the publish marker says so). Once converted, U is
a pointer to a sealed version, R/work/<name> (R = versions.root_of(U)) is the WORK DIRECTORY every writer step of the run
writes (catalog, forced catflux, pages: cli.main and the steps resolve U through versions.writable, holding its writer lock
while they run), and a serve publishes a NEW VERSION:
convert (unit deep_<run>_convert; only when named, once)  the legacy run cloned as a version (versions.snapshot: a
       copy-on-write clone under the writer lock, tested against the run in content), its page index, sealed, then - under
       the journal locks - U switched to it with the legacy directory moved to the work directory (versions.switch
       to_work) and the journal lines written on it since the clone appended. Every later serve publishes as below.
snapshot (unit deep_<run>_snapshot)  the owed catch-up first (sublst_build.catchup_stage); the work directory cloned as a
       new unsealed version (versions.snapshot, under the writer lock: no writer step writes it meanwhile); the served
       version's judgement journals (versions.journal_dirs) copied into it under the journal locks.
views, test, carry  as above on the snapshot (no publish marker: nobody reads it yet).
switch (unit deep_<run>_switch)  the snapshot's page index, the seal (every page's queue and view present, the index
       current, the manifest), then under the journal locks U switched to it (refused when U names another version than
       the one whose journals were copied) and the lines journalled on the old version since the copy appended
       (journals.append_between; a failure is OWED: versions.owe_catchup). A refusal before the switch discards the
       snapshot; U unchanged.
start, hub, verify  as above, on U (the new version).
These run together (views, test, start; carry with --carry-from) through U's coalescing slot (lib/serve/coalesce.py):
one publish at a time, a request during one gets one follow-up. retire (only when named): the retention of U's versions
(sublst_build.retire_stage). A parity serve is never versioned: it serves its mirror in place.
"""
import copy
import os
import sys

from exopipe.phase3.core.record import NoInput, Refusal
from exopipe.phase3.lib.serve import common, exclusion_gate, flow, journals, mirror, queues, register, scripts, versions, views

STAGES = ('convert', 'views', 'test', 'carry', 'page-index', 'start', 'hub', 'verify', 'retire')
DEFAULT_OFF = ('convert', 'retire')      # run only when named (steps/serve.py select_stages)
PUBLISH = ('views', 'test', 'start')     # a versioned run: on one snapshot, together (carry with them when given)


def _S():
    from exopipe.phase3.config import CFG
    return CFG.serve


def _views_unit(plan):
    return f'deep_{plan.run_name}_views'


def _test_unit(plan):
    return f'deep_{plan.run_name}_test'


def views_stage(run, plan, state, marker=True):
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
        # the publish marker, now that the run is rewritten in place (production's deep publish marker; removed at the run's
        # registration): a page built meanwhile says so. A snapshot version is read by nobody before its switch: no marker
        if marker:
            hp, now = register.set_hold(U, S.PUBLISH_MARKER_TEXT.format(utc=common.utc_text()))
            u.note(f'publish marker {hp} ' + ('written: a page built until the run\'s registration shows its text' if now else
                                              'stands (kept with its own text; removed at the run\'s registration)'))
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
        u.meta['refused_pages'] = dict(state['refused'])      # G17 (exclusions) runs per page in the test stage
    return u


def carry_stage(run, plan, args):
    spec = getattr(args, 'carry_from', None)
    if not spec:
        return None
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
    return u


def versioned(plan):
    """True when this serve publishes versions of U: U is a pointer (converted) and the serve is not a parity serve."""
    return not plan.parity and versions.is_pointer(plan.U)


def _note(u):
    return f'serve --tier deep {u.name}'


def convert_stage(run, plan):
    """CONVERSION (module text; unit deep_<run>_convert). NoInput when U is versioned already. Returns the unit."""
    from exopipe.phase3.lib.serve import sublst_build as SB
    U = plan.U
    with run.unit(f'deep_{plan.run_name}_convert', stage='convert', tier='deep', run=plan.run_name, parity=plan.parity) as u:
        if plan.parity:
            u.refuse('convert: a parity serve serves its mirror in place - only the served deep run is versioned')
        if versions.is_pointer(U):
            raise NoInput(f'{U} is versioned already (it names version {versions.served(U).vid}; work directory '
                          f'{versions.work_path(U)})')
        if not os.path.isdir(U):
            u.refuse(f'convert: no run directory {U}')
        versions._guard(U, 'the conversion of')
        legacy = os.path.realpath(U)
        R = versions.root_of(U)
        os.makedirs(R, exist_ok=True)
        versions.hold_writer(R)          # until this serve ends: no writer step writes the run while it is converted
        v, info = versions.snapshot(U, legacy, note=_note(u))
        u.meta.update(version=v.vid, clone=info['clone'], content=info['stats'])
        u.note(f'cloned {legacy} as version {v.vid} ({info["clone"]}); content test: {info["content"][:300]}')
        J = versions.journal_dirs(legacy)
        switched = False
        try:
            flow.build_page_index(u, v.run)
            rec = versions.seal(v, checks=[('page_index_current', SB.index_check)], note=_note(u))
            with versions.journal_locks(U, J):
                sw = versions.switch(U, v, note=_note(u), record=rec, to_work=True)
                switched = True
                W = sw['converted']
                try:
                    app = journals.append_between(W, U, J)
                except Exception as e:
                    raise Refusal(SB._owe(plan, v, None, W, False, {}, e, journal_dirs=J)) from e
        except Exception as e:
            if not switched and os.path.isdir(v.dir):
                _discard(v, e, _note(u))
            raise
        u.meta.update(work=W, journals=J, caught_up=app)
        u.note(f'CONVERTED: {U} -> version {v.vid} (sealed: {rec["n_files"]} files, {rec["n_bytes"]} bytes); the legacy '
               f'directory {legacy} is the work directory {W} (every writer step writes it); journal lines written on it '
               f'since the clone, appended: {app}')
        u.add_output(v.path('VERSION.json'), 'run_version', pointer=U)
        _journal_outputs(u, U, J, W)
    return u


def _discard(v, e, note):
    """Discard the never-served version ``v`` after the refusal ``e`` (a failure is added to e's text, never in its place)."""
    from exopipe.phase3.lib.serve import sublst_build as SB
    try:
        versions.discard(v, note=f'{note}: refused before the switch')
    except Exception as e2:
        SB._add_failure(e, f' - AND version {v.vid} could not be discarded ({e2}): retention retires it')


def snapshot_stage(run, plan, state):
    """The work directory as a new version, seeded with the served journals (module text). state['version'],
    state['seed_version'], state['journals']. Returns the unit."""
    U = plan.U
    with run.unit(f'deep_{plan.run_name}_snapshot', stage='snapshot', tier='deep', run=plan.run_name) as u:
        W = versions.work_path(U)
        if not os.path.isdir(W):
            u.refuse(f'{U} is versioned but has no work directory {W} - not published')
        v, info = versions.snapshot(U, W, note=_note(u))
        try:
            cur = versions.served(U)
            J = versions.journal_dirs(cur.run, v.run)
            with versions.journal_locks(U, J):
                now = versions.served(U)
                if now is None or now.vid != cur.vid:
                    raise Refusal(f'{U} was switched while its journals were copied ({cur.vid} -> '
                                  f'{now.vid if now else "nothing"}) - not published')
                copied = journals.sublst_copy_journals(cur.run, v.run, journals=J)
        except Exception as e:
            _discard(v, e, _note(u))
            raise
        state.update(version=v, seed_version=cur.vid, journals=J)
        u.meta.update(version=v.vid, clone=info['clone'], content=info['stats'], seeded_from=cur.vid, journals=J,
                      journals_copied=len(copied))
        u.note(f'snapshot of {W} as version {v.vid} ({info["clone"]}); content test: {info["content"][:300]}; '
               f'{len(copied)} journal(s) of the served version {cur.vid} copied in under the journal locks')
        u.add_output(v.path('BUILDING.json'), 'run_version_placed')
    return u


def switch_stage(run, plan, state):
    """The seal and the switch of the snapshot (module text). Returns the unit."""
    from exopipe.phase3.lib.serve import sublst_build as SB
    v, J = state.get('version'), state.get('journals') or []
    with run.unit(f'deep_{plan.run_name}_switch', stage='switch', tier='deep', version=getattr(v, 'vid', None)) as u:
        if v is None:
            u.refuse('no snapshot version (the snapshot unit did not make one)')
        switched = False
        try:
            flow.build_page_index(u, v.run)                # no publish marker on a version nobody reads yet
            rec = versions.seal(v, required=SB.seal_required(plan, state, v.run),
                                checks=[('page_index_current', SB.index_check)], note=_note(u))
            u.note(f'version {v.vid} sealed: {rec["n_files"]} files, {rec["n_bytes"]} bytes (manifest walk {rec["walk_s"]} s; '
                   f'{rec["n_hashed"]} files, {rec["hashed_bytes"]} bytes hashed in {rec["hash_s"]} s)')
            with versions.journal_locks(plan.U, J):
                cur = versions.served(plan.U)
                if cur is None or cur.vid != state.get('seed_version'):
                    raise Refusal(f'{plan.U} names {cur.vid if cur else "no version"}, not {state.get("seed_version")} whose '
                                  f'journals the snapshot holds - not switched')
                sw = versions.switch(plan.U, v, note=_note(u), record=rec)
                switched = state['switched'] = True       # served now: never discarded, also when the catch-up fails
                prev = sw['previous']
                try:
                    app = SB.catch_up(prev.run, plan.U, False, {}, journal_dirs=J)
                except Exception as e:
                    raise Refusal(SB._owe(plan, v, prev, prev.run, False, {}, e, journal_dirs=J)) from e
        except Exception as e:
            if not switched and v is not None and os.path.isdir(v.dir):
                _discard(v, e, _note(u))
            raise
        u.meta.update(version=v.vid, previous=prev.vid, caught_up=app)
        u.note(f'switched: {plan.U} -> version {v.vid} (previous {prev.vid}); {sw["verify"]}; journal lines written on '
               f'{prev.vid} since the copy, appended: {app}')
        u.add_output(v.path('VERSION.json'), 'run_version', pointer=plan.U)
        _journal_outputs(u, plan.U, J, prev.run)
    return u


def _journal_outputs(u, U, J, old):
    """Each journal of U after the catch-up: class judgement_journal, starting with every line of the old run's (its seed
    copy, then the lines appended on it since, in order)."""
    for jd in J:
        j, ref = os.path.join(U, jd, 'judgements.jsonl'), os.path.join(old, jd, 'judgements.jsonl')
        if os.path.exists(j):
            u.add_output(j, 'judgement_journal', carried_from=ref if os.path.exists(ref) else None)


def _on(plan, v):
    """The plan with U the snapshot's run (its views, test and carry write there; its log is the version's)."""
    vp = copy.copy(plan)
    vp.U = v.run
    vp.log_path = os.path.join(v.run, 'incremental.log')
    return vp


def publish(args, run, plan, stages):
    """A VERSIONED PUBLISH (module text): snapshot, views, test, [carry], switch on the snapshot, then start, hub, verify on U.
    A refusal before the switch discards the snapshot."""
    from exopipe.phase3.lib.serve import sublst_build as SB
    state = flow.new_state()
    if flow.halts(SB.catchup_stage(run, plan), 'catch-up', need_done=True):
        return
    if flow.halts(snapshot_stage(run, plan, state), 'snapshot', need_done=True):
        return
    v, vp = state['version'], _on(plan, state['version'])
    try:
        if flow.halts(views_stage(run, vp, state, marker=False), 'views'):
            return
        if flow.halts(flow.test_stage(run, vp, state, _test_unit(plan), root=v.run), 'test'):
            return
        if 'carry' in stages and flow.halts(carry_stage(run, vp, args), 'carry'):
            return
        if flow.halts(switch_stage(run, plan, state), 'switch', need_done=True):
            return
    finally:
        if not state.get('switched') and os.path.isdir(v.dir):
            try:
                versions.discard(v, note=f'serve --tier deep {plan.run_name}: a stage refused before the switch')
            except Exception as e:                      # retention retires an unserved version whose builder is gone
                print(f'serve: version {v.vid} not discarded ({e})', file=sys.stderr)
    flow.start_stage(run, plan, state)                 # U's page index: the version's (current, kept)
    if 'hub' in stages and not getattr(args, 'no_hub', False):
        flow.hub_stage(run, plan, state)
    if 'verify' in stages:
        flow.verify_stage(run, plan)


def run(args, run, plan, stages):
    """convert (when named), then a versioned publish (versioned(plan) and any of views, test, carry, start) or the stages in
    place (a legacy or parity run; page-index, hub, verify alone), then retire (when named)."""
    rest = [s for s in stages if s not in DEFAULT_OFF]
    if 'convert' in stages and flow.halts(convert_stage(run, plan), 'convert'):
        return
    if versioned(plan) and set(rest) & {'views', 'test', 'carry', 'start'}:
        missing = [s for s in PUBLISH if s not in rest]
        if missing:
            raise Refusal(f'{plan.U} is a versioned run: views, test and start publish one snapshot together (carry with them) '
                          f'- {", ".join(missing)} not selected; nothing done')
        from exopipe.phase3.lib.serve import sublst_build as SB
        SB.coalesced(args, run, plan, rest, publish)
    elif rest:
        in_place(args, run, plan, rest)
    if 'retire' in stages:
        from exopipe.phase3.lib.serve import sublst_build as SB
        flow.halts(SB.retire_stage(run, plan, args), 'retire')


def in_place(args, run, plan, stages):
    """The stages on U in place (production's deep serve; a parity mirror; page-index, hub, verify of a versioned run)."""
    state = flow.new_state()
    # a refused views / test / carry stage stops the later stages (flow.halts): no server is replaced after a refusal
    if 'views' in stages:
        if flow.halts(views_stage(run, plan, state), 'views'):
            return
    elif any(s in stages for s in ('test', 'start')):
        rec = flow.require_done(run, _views_unit(plan), 'the deep views')
        state['refused'].update((rec.get('meta') or {}).get('refused_pages') or {})
    if 'test' in stages:
        if flow.halts(flow.test_stage(run, plan, state, _test_unit(plan)), 'test'):
            return
    elif 'start' in stages:
        exp, refused = flow.expected_from_record(run, _test_unit(plan), 'start')
        state['expected'].update(exp)
        state['refused'].update(refused)
    if 'carry' in stages:
        if flow.halts(carry_stage(run, plan, args), 'carry'):
            return
    if 'start' in stages:
        flow.start_stage(run, plan, state)                 # builds the page index before the registration
    elif 'page-index' in stages:
        if flow.halts(flow.page_index_stage(run, plan, 'deep'), 'page-index'):
            return
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
    refuses = exclusion_gate.refuses('deep')
    for pg in plan.pages:
        if pg.family != 'review':
            continue
        scan = exclusion_gate.scan_server_page(pg, U, strict=False)
        print(f'  G17 serve-time exclusion gate, page {pg.key} ({"refusing" if refuses else "report only"}; the rows its '
              f'server reads): {scan.summary()}' + (' - WOULD NOT BE SERVED' if scan.hits and refuses else ''))
    for ST in ('V', 'I'):
        for t in _S().DEEP_SINGLES_TIERS:
            q = queues.tier_queue_path(U, f'{ST}_{t}')
            print(f'  tier queue {ST}_{t}: ' + (f'{len(common.queue_ids(q))} ids' if os.path.exists(q) else 'absent'))
        fr = queues.fork_raw_queues(U, ST)
        print(f'  fork raw queues {ST}: ' + (', '.join(os.path.basename(p) for _, p in fr) if fr else
                                             'NONE - the multiples page would refuse (G3)'))
    if getattr(args, 'carry_from', None):
        print(f'  carry from: {getattr(args, "carry_from")}')
    if not plan.parity:
        for line in versions.describe(plan.U):
            print(line)
        if 'convert' in stages:
            print(f'  convert: the run cloned as a version, sealed, U switched to it; the legacy directory becomes the work '
                  f'directory {versions.work_path(plan.U)}')
        elif versioned(plan) and set(stages) & set(PUBLISH):
            print(f'  versioned publish: snapshot of {versions.work_path(plan.U)}, views/test/[carry] on it, seal, switch of '
                  f'{plan.U}, then the registration')
    if os.path.isdir(plan.U or '') and ('start' in stages or 'page-index' in stages):
        print(flow.page_index_describe(plan.U))
