"""serve --tier sublst: the cumulative 10-minute product Run_SLSTB on 8787/8788 (+ multi-integration 8789/8790).
Port of sublst_cumulative.sh (+ the prepare/start halves of sublst_serve.sh it calls); stages
build -> views -> test -> carry -> swap -> start -> hub -> verify. carry, swap and start run only together (the journal
lines written on the old servers between the copy and the swap are appended at the swap, sublst_cumulative.sh:106-120).

build  (unit sublst_<run>_build)  the build lock (G12; sublst_cumulative.sh:23 flock, servers never inherit it); the
       generation-mix gate G8 (every LST run has candidates/, pertie_detections.csv and a pos_kind column - else exit 2;
       today the caller skipped the cumulative and exited 0, sublst_catalog_publish.sh:96-100, batch row B-73);
       rm -rf B; sublst_cumulative.py B LSTs; build_localizations.py with the per-case tie and the common frame (G9);
       the localization figures (md5-keyed reuse from the served run, else drawn, 4 at a time, 600 s each; refuse a
       missing one - port of the PYF heredoc :36-61); p3t_pages_index.py per tier (G10). Output class sublst_cumulative.
views  (unit sublst_<run>_views)  on B: the union with the Stokes key (G5), the review_dir merge with
       pages.REVIEW_DIR_MERGE['sublst'] (the merge's config key is pages'; the call runs here because the union is
       built inside this build), the tier queues sectioned (sb, wb when present), p3s3_stokes_queue.py, the per-Stokes
       views (G6), the multi-integration queues and views (sublst_cumulative.sh:77-95). G11: a refusal here leaves the
       served product and its servers untouched.
test   App probes and content tests on B (expected index sources), G17 report.
carry/swap/start  journals copied into B, B swapped in (U -> U.prev_<utc>), the between lines appended, the four pages
       started from U (G13 PORT_OWNERSHIP sub-LST: Run_SLST* or Run_P3T4 servers only).
hub    10-min / 10-min-multiples set; '10-min-sublst' removed (sublst_cumulative.sh:145-150).
verify served_page tests + sublst_cumulative (with tests (c)+(d), sublst_catalog_publish.sh:101-108) on the served run.
Parity: U, B and the lock live under <out>/sublst/; the build hard-links the LST runs' candidates (cp -al), so a parity
--out must be on the same file system as the runs (/data07: spec 10's parity root /data07/gh/refactor_parity/serve).
"""
import collections
import contextlib
import csv
import fcntl
import hashlib
import os
import shutil
from concurrent.futures import ThreadPoolExecutor

from exopipe.phase3.core.record import NoInput, Refusal
from exopipe.phase3.lib.serve import common, flow, journals, queues, scripts, views

STAGES = ('build', 'views', 'test', 'carry', 'swap', 'start', 'hub', 'verify')


def _CFG():
    from exopipe.phase3.config import CFG
    return CFG


def build_dir(plan):
    return plan.U + _CFG().serve.SUBLST_BUILD_SUFFIX


@contextlib.contextmanager
def build_lock(plan):
    """flock on <runs root>/.sublst_cumulative_build.lock (production path of sublst_cumulative.sh:23); blocking, as the
    production build waits for the other one."""
    d = os.path.dirname(plan.U)
    os.makedirs(d, exist_ok=True)
    fd = os.open(os.path.join(d, '.sublst_cumulative_build.lock'), os.O_RDWR | os.O_CREAT, 0o664)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def lst_run(lst):
    from exopipe.phase3.lib.serve.plan import run_root
    return os.path.join(run_root('sublst'), _CFG().serve.SUBLST_LST_RUN.format(hh=lst[:-1]))


def generation_mix(lsts):
    """G8: the LST runs that cannot be merged ([(lst, reason)]) - sublst_cumulative.py:24-26, 79-86."""
    bad = []
    for l in lsts:
        R = lst_run(l)
        if not os.path.isdir(os.path.join(R, 'candidates')):
            bad.append((l, 'no candidates/ (publish it first)'))
            continue
        if not os.path.exists(os.path.join(R, 'pertie_detections.csv')):
            bad.append((l, 'no pertie_detections.csv (catalogued before the window tie)'))
            continue
        try:
            with open(os.path.join(R, 'master_catalog.csv'), newline='') as f:
                hdr = next(csv.reader(f), [])
        except OSError:
            hdr = []
        if 'pos_kind' not in hdr:
            bad.append((l, 'master_catalog.csv has no pos_kind column (catalogued before fitted positions)'))
    return bad


def figure_key(loc_row, off_rows):
    """md5 of the identity's localization rows (sublst_cumulative.sh:44, verbatim expression)."""
    return hashlib.md5(repr((sorted(loc_row.items()), sorted(sorted(o.items()) for o in off_rows))).encode()).hexdigest()


def figures(B, served, unit=None, log=None):
    """Port of the PYF heredoc (sublst_cumulative.sh:36-61). Returns (n multi, n linked, n drawn, [failures])."""
    S = _CFG().serve
    F, P = f'{B}/localization_figs', f'{served}/localization_figs'
    os.makedirs(F, exist_ok=True)
    loc = {r['identity_id']: r for r in csv.DictReader(open(f'{B}/source_catalogs/localizations.csv'))}
    off = collections.defaultdict(list)
    for r in csv.DictReader(open(f'{B}/source_catalogs/localization_offsets.csv')):
        off[r['identity_id']].append(r)
    multi = sorted(i for i, r in loc.items() if int(r['n_det']) > 1)
    key = {i: figure_key(loc[i], off[i]) for i in multi}
    todo, linked = [], 0
    for i in multi:
        old, ok = f'{P}/{i}.png', f'{P}/{i}.key'
        if os.path.exists(old) and os.path.exists(ok) and open(ok).read().strip() == key[i]:
            try:
                os.link(old, f'{F}/{i}.png')
            except OSError:
                shutil.copy2(old, f'{F}/{i}.png')     # parity --out on another file system: same bytes
            open(f'{F}/{i}.key', 'w').write(key[i])
            linked += 1
        else:
            todo.append(i)
    env = {'P3_LOC_CATS': S.SUBLST_LOC_ENV['P3_LOC_CATS']}

    def draw(i):
        res = scripts.run('astrometry_panel', [i, f'{F}/{i}.png', B], env=env, timeout=S.SUBLST_FIG_TIMEOUT_S,
                          refusal_codes=())
        if res.rc == 0 and os.path.exists(f'{F}/{i}.png'):
            open(f'{F}/{i}.key', 'w').write(key[i])
            return None
        return f'{i}: {scripts.last_line(res.tail)}'
    with ThreadPoolExecutor(S.SUBLST_FIG_THREADS) as ex:
        bad = [e for e in ex.map(draw, todo) if e]
    return len(multi), linked, len(todo) - len(bad), bad


def build_stage(run, plan):
    S = _CFG().serve
    B = build_dir(plan)
    with run.unit(f'sublst_{plan.run_name}_build', stage='build', tier='sublst', lsts=plan.lsts, parity=plan.parity) as u:
        flow.excl_note(u)
        if not plan.lsts:
            raise NoInput('no LST run to merge (no Run_SLST<hh>B under the runs root, or --lsts empty)')
        bad = generation_mix(plan.lsts)
        if bad:
            u.refuse('G8 sub-LST generation mix: ' + '; '.join(f'{l}: {why}' for l, why in bad)
                     + ' - re-catalogue those LSTs first; the served cumulative stays as it is')
        for l in plan.lsts:
            for f in ('master_catalog.csv', 'pertie_detections.csv'):
                u.add_input(os.path.join(lst_run(l), f))
        log = os.path.join(scripts.ops_root(), 'sublst_cumulative.log') if not plan.parity else \
            os.path.join(plan.out, 'sublst', 'sublst_cumulative.log')
        if os.path.isdir(B):
            shutil.rmtree(B)
        res = scripts.run('sublst_cumulative', [B] + list(plan.lsts), log=log, unit=u)
        if not res.ok:
            u.refuse(f'merge failed: {res.reason or scripts.last_line(res.tail)}')
        env = dict(S.SUBLST_LOC_ENV, P3_TENMIN_PERTIE=f'{B}/pertie_detections.csv', P3_LOC_OUT=f'{B}/source_catalogs')
        res = scripts.run('build_localizations', [B], env=env, log=log, unit=u)
        if not res.ok:
            u.refuse(f'G9 localizations failed: {res.reason or scripts.last_line(res.tail)}')
        n, linked, drawn, failed = figures(B, plan.U_src if plan.parity else plan.U, unit=u, log=log)
        u.note(f'localization figures: {n} multi-detection identities, {linked} carried unchanged, {drawn} drawn, '
               f'{len(failed)} failed')
        if failed:
            u.refuse(f'G9 localization figures: {len(failed)} failed, e.g. {failed[:3]}')
        for K in S.SUBLST_INDEX_TIERS:
            pin = f'{B}/source_catalogs/pages_input_{K}.csv'
            nonempty = os.path.exists(pin) and os.path.getsize(pin) > 0 and sum(1 for _ in open(pin)) > 1
            if not nonempty and K.endswith('_wb'):
                continue
            res = scripts.run('pages_index', [B, K], log=log, unit=u, refusal_codes=(1, 2))
            if not res.ok:
                u.refuse(f'G10 pages index {K}: {res.reason or scripts.last_line(res.tail)}')
        u.add_output(B, 'sublst_cumulative', with_served_tests=False)
    return u


def views_stage(run, plan, state):
    S = _CFG().serve
    B = build_dir(plan)
    with run.unit(f'sublst_{plan.run_name}_views', stage='views', tier='sublst', parity=plan.parity) as u:
        if not os.path.isdir(B):
            u.refuse(f'no build directory {B} (run the build stage)')
        log = os.path.join(B, 'incremental.log')
        nb = views.sublst_union(B)
        u.note(f'union survivors: {nb} bands')
        res = scripts.run('merge_review_dir', [B], env=views.pages_merge_env('sublst'), log=log, unit=u,
                          refusal_codes=(1, 2))
        if not res.ok:
            u.refuse(f'G7 Stokes-keyed review_dir merge refused: {scripts.last_line(res.tail)}')
        for ST in ('V', 'I'):
            if not any(pg.stokes == ST for pg in plan.pages):
                continue
            for t in S.SUBLST_TIERS:
                K = f'{ST}_{t}'
                if t == 'wb' and not os.path.isfile(queues.tier_queue_path(B, K)):
                    continue
                sec = queues.section_tier_queue(B, K, 'sublst', lsts_text=plan.lsts_text)
                if sec is None:
                    u.refuse(f'no queue {queues.tier_queue_path(B, K)} (pages index not built)')
                u.note(f'{K} multi {sec["multi"]} single {sec["single"]}')
                u.add_output(queues.tier_queue_path(B, K), 'served_queue')
            res = scripts.run('stokes_queue', [B, ST], log=log, unit=u)
            if not res.ok:
                u.refuse(f'p3s3_stokes_queue.py {ST}: {res.reason or scripts.last_line(res.tail)}')
            v = views.sublst_stokes_view(B, ST)
            u.note(f'{ST} catalog view: {v["rows"]} survivors rows in {v["dir"]}')
            Q = f'{B}/judgements/queue_p3s3_{ST}.txt'
            if not common.has_j_line(Q):
                u.refuse(f'G1 empty queue: {Q} - not served (sublst_serve.sh:112)')
            u.add_output(Q, 'served_queue')
            u.add_output(v['dir'], 'served_view', tier='sublst')
            mp = next((pg for pg in plan.pages if pg.stokes == ST and pg.multi), None)
            if mp is None:
                continue
            QM = f'{B}/judgements/queue_p3s3_{ST}_multi.txt'
            if not queues.sublst_multi_queue(Q, QM):
                state['refused'][mp.key] = f'NO-INPUT: {ST}: no multi-integration sources - multiples page not served'
                u.note(state['refused'][mp.key])
                continue
            mv = views.sublst_multi_view(B, ST, QM)
            u.note(f'{ST} multi view: {mv["sources"]} sources')
            u.add_output(QM, 'served_queue')
            u.add_output(mv['dir'], 'served_view', tier='sublst')
        u.meta['refused_pages'] = dict(state['refused'])
    return u


def carry_swap_stage(run, plan, state):
    """Copy the journals into B, swap B in, append the between lines. Returns the unit. G11 (validate before the swap,
    sublst_cumulative.sh:65-74): a page refused on the build by the views or test stage (anything but a multiples page with
    no multi-integration source, sublst_cumulative.sh:80) refuses the swap - the served product and its servers untouched."""
    S = _CFG().serve
    B = build_dir(plan)
    with run.unit(f'sublst_{plan.run_name}_swap', stage='carry+swap', tier='sublst', parity=plan.parity) as u:
        if not os.path.isdir(B):
            u.refuse(f'no build directory {B}')
        blocking = {k: why for k, why in state['refused'].items() if not str(why).startswith('NO-INPUT: ')}
        if blocking:
            u.refuse('G11 validate before the swap: ' + '; '.join(f'{k}: {why}' for k, why in sorted(blocking.items()))[:800]
                     + ' - nothing swapped in, the served product and its servers untouched')
        src = plan.U_src if plan.parity else plan.U
        copied = journals.sublst_copy_journals(src, B)
        u.note(f'journals copied from {src}: {len(copied)}')
        prev = plan.U + S.SUBLST_PREV_SUFFIX.format(utc=common.stamp())
        moved = journals.sublst_swap(plan.U, B, prev)
        u.note(f'swapped in: {plan.U} (previous {moved or "none"}); {len(os.listdir(os.path.join(plan.U, "candidates", "I")))}'
               f' I pages, {len(os.listdir(os.path.join(plan.U, "candidates", "V")))} V pages')
        between_src = src if plan.parity else moved
        if between_src:
            app = journals.append_between(between_src, plan.U)
            u.note(f'journal lines recorded on the old servers between the copy and the swap, appended: {app}')
        for J in S.SUBLST_JOURNALS:
            j = os.path.join(plan.U, J, 'judgements.jsonl')
            if os.path.exists(j):
                ref = os.path.join(between_src or src, J, 'judgements.jsonl')
                u.add_output(j, 'judgement_journal', carried_from=ref if os.path.exists(ref) else None)
            else:
                os.makedirs(os.path.dirname(j), exist_ok=True)
        if not any(os.path.exists(os.path.join(plan.U, J, 'judgements.jsonl')) for J in S.SUBLST_JOURNALS):
            u.add_output(plan.U, 'sublst_cumulative', with_served_tests=False)
    return u


def run(args, run, plan, stages):
    S = _CFG().serve
    state = flow.new_state()
    together = {'carry', 'swap', 'start'}
    if (together & set(stages)) and not together <= set(stages):
        raise Refusal('sub-LST: carry, swap and start run only together (the journal lines written between the copy and '
                      'the swap are appended at the swap, sublst_cumulative.sh:106-120)')
    with build_lock(plan):
        # each stage runs only after the previous one ended done (flow.halts; G11: a refused build, views or test leaves
        # the served product and its four servers untouched - nothing is swapped in, sublst_cumulative.sh:26-74)
        if 'build' in stages:
            if flow.halts(build_stage(run, plan), 'build', need_done=True):
                return
        if 'views' in stages:
            if flow.halts(views_stage(run, plan, state), 'views', need_done=True):
                return
        elif 'test' in stages or 'start' in stages:
            rec = flow.require_done(run, f'sublst_{plan.run_name}_views', 'the sub-LST views')
            state['refused'].update((rec.get('meta') or {}).get('refused_pages') or {})
        B = build_dir(plan)
        if 'test' in stages:
            if flow.halts(flow.test_stage(run, plan, state, f'sublst_{plan.run_name}_test', root=B), 'test'):
                return
        elif 'start' in stages:
            exp, refused = flow.expected_from_record(run, f'sublst_{plan.run_name}_test', 'start')
            state['expected'].update(exp)
            state['refused'].update(refused)
        if 'start' in stages:
            if flow.halts(carry_swap_stage(run, plan, state), 'carry+swap', need_done=True):
                return
            flow.start_stage(run, plan, state)
    if 'hub' in stages and not getattr(args, 'no_hub', False):
        flow.hub_stage(run, plan, state, remove=S.HUB_REMOVE_TIERS.get('sublst', ()))
    if 'verify' in stages:
        flow.verify_stage(run, plan)
        with run.unit(f'sublst_{plan.run_name}_verify_product', stage='verify', tier='sublst') as u:
            u.add_output(plan.U, 'sublst_cumulative')


def describe(plan, stages, args):
    print(f'  LST runs merged: {", ".join(lst_run(l) for l in plan.lsts) or "(none)"}')
    bad = generation_mix(plan.lsts)
    print('  G8 generation mix: ' + ('ok (every LST carries the window tie and fitted positions)' if not bad else
                                     'WOULD REFUSE - ' + '; '.join(f'{l}: {w}' for l, w in bad)))
    print(f'  build directory: {build_dir(plan)}; journals copied from {plan.U_src}')
