"""serve --tier sublst: the cumulative 10-minute product Run_SLSTB on 8787/8788 (+ multi-integration 8789/8790).
Port of sublst_cumulative.sh (+ the prepare/start halves of sublst_serve.sh it calls); stages
build -> views -> test -> carry -> swap -> start -> hub -> verify. carry, swap and start run only together (the journal
lines written on the old servers between the copy and the swap are appended at the swap, sublst_cumulative.sh:106-120).

build  (unit sublst_<run>_build)  the build lock (G12; sublst_cumulative.sh:23 flock, servers never inherit it); the
       generation-mix gate G8 (every LST run has candidates/, pertie_detections.csv, a verified conform_inputs/ whose
       CONFORM_LOG.txt read lines name it (sublst_cumulative.sh:60, 76-96) and a pos_kind column - else exit 2;
       today the caller skipped the cumulative and exited 0, sublst_catalog_publish.sh:96-100, batch row B-73);
       the DEEP-OMISSION decision table of the LSTs (lib/catalog/deep_omission.tenmin_table; Gregg 2026-10-24,
       sublst_cumulative.sh:37-46, 98-100) written OUTSIDE B, <runs root>/deep_omission/deep_omission_<UTC>.csv (class
       omission_decisions; window-rms cache beside it, production's cache read as a seed in a parity run);
       rm -rf B; SUBLST_DEEP_OMISSION=<table> sublst_cumulative.py B LSTs (the merge drops the OMIT rows per LST in place,
       sublst_deep_omission_drop.py, and records them: B/deep_omission_dropped.csv, class omission_record, with its
       per-instant test; its re-judging of a partial survivor's remaining single takes the builder's own 7-8 sigma gate under
       the configured 10-minute size rule - P3_SIZE_GATE_7_8 = inf under 'envelope', production's 1.7 under 'production',
       deep_omission.tenmin_size_gate); build_localizations.py with the per-case tie and the common frame (G9);
       the localization figures (md5-keyed reuse from the served run, else drawn, 4 at a time, 600 s each; refuse a
       missing one - port of the PYF heredoc :36-61); p3t_pages_index.py per tier (G10). Output class sublst_cumulative.
views  (unit sublst_<run>_views)  on B: the union with the Stokes key (G5), the review_dir merge with
       pages.REVIEW_DIR_MERGE['sublst'] (the merge's config key is pages'; the call runs here because the union is
       built inside this build), the tier queues sectioned (sb, wb when present), p3s3_stokes_queue.py, the per-Stokes
       views (G6), the multi-integration queues and views (sublst_cumulative.sh:77-95). G11: a refusal here leaves the
       served product and its servers untouched.
test   App probes and content tests on B (expected index sources), G17 report.
carry/swap/start  journals copied into B, B swapped in (U -> U.prev_<utc>), the between lines appended, the four pages
       registered in ONE registration and each one's build waited on (flow.start_stage; sublst_cumulative.sh:291-300 ->
       sublst_register_pages.py: no server of its own since 2026-10-22).
hub    10-min / 10-min-multiples set; '10-min-sublst' removed (sublst_cumulative.sh:145-150).
verify served_page tests + sublst_cumulative (with tests (c)+(d), sublst_catalog_publish.sh:101-108) on the served run.
Parity: U, B and the lock live under <out>/sublst/; the build hard-links the LST runs' candidates (cp -al), so a parity
--out must be on the same file system as the runs (/data07: spec 10's parity root /data07/gh/refactor_parity/serve).
"""
import collections
import contextlib
import csv
import fcntl
import glob
import hashlib
import json
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
        # the tie tables the conformance read, recorded and checked (refactor-chat ruling 2026-10-18; sublst_cumulative.sh:60,
        # 76-96): R/conform_inputs verifies and every read line of its CONFORM_LOG.txt names the copy with its sha256
        from exopipe.phase3.lib.catalog import conform_inputs as CI
        ok, why = CI.check(R, os.path.join(R, CI.DIR, CI.LOG_COPY))
        if not ok:
            bad.append((l, f'{why} - re-catalogue {l}'))
            continue
        try:
            with open(os.path.join(R, 'master_catalog.csv'), newline='') as f:
                hdr = next(csv.reader(f), [])
        except OSError:
            hdr = []
        if 'pos_kind' not in hdr:
            bad.append((l, 'master_catalog.csv has no pos_kind column (catalogued before fitted positions)'))
    return bad


def grouping_mix(lsts):
    """Design B (catalog_grouping.GROUPING 'design-b'; WAVE2 GROUPING; review 2026-11-04, F3 - the minimum gate): every LST
    run's catalogues must have been built with ONE grouping table per Stokes - their grouping_table.json (the builder runner's
    provenance, lib/catalog/wraps/builder_grouping.py) names one table sha1 across the LSTs. A regroup renames identities
    (display names follow mean positions; 121 I / 15 V renamed between the 10-25 and 11-03 inputs, the 11-04 review), and the
    cumulative merges the LST runs by name: a mix would show one identity under two names or join two under one. Returns
    ([(lst, reason)], {Stokes: {table sha1: [lsts]}})."""
    from exopipe.phase3.lib.catalog.wraps import builder_grouping as BG
    bad, by = [], {}
    for l in lsts:
        R = lst_run(l)
        dirs = sorted(d for d in glob.glob(os.path.join(R, 'source_catalogs_[IV]_[sw]b')) if os.path.isdir(d))
        if not dirs:
            continue
        for d in dirs:
            S = os.path.basename(d).split('_')[2]
            p = os.path.join(d, BG.PROVENANCE)
            if not os.path.exists(p):
                bad.append((l, f'{os.path.basename(d)} has no {BG.PROVENANCE} (built without the design-B grouping)'))
                continue
            try:
                with open(p) as f:
                    sha = json.load(f)['table_sha1']
            except (OSError, ValueError, KeyError) as e:
                bad.append((l, f'{os.path.basename(d)}/{BG.PROVENANCE} unreadable ({e})'))
                continue
            by.setdefault(S, {}).setdefault(sha, set()).add(l)
    for S, shas in by.items():
        if len(shas) > 1:
            parts = '; '.join(f'{sha[:12]}: {",".join(sorted(ls))}' for sha, ls in sorted(shas.items(), key=lambda kv: -len(kv[1])))
            bad.append(('*', f'Stokes {S}: the LST builds name {len(shas)} grouping tables ({parts}) - rebuild the LSTs of the '
                             f'older tables from stage build (the cumulative joins LST runs by identity name)'))
    return bad, {S: {sha: sorted(ls) for sha, ls in v.items()} for S, v in by.items()}


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


def omission_table(plan, u):
    """The deep-omission decision table of the build's LSTs (sublst_cumulative.sh:37-46: written outside the build, which the
    merge wipes; the window rms cached beside it - in a parity run production's cache is read as a seed, never written).
    Recorded as an input and as an output of class omission_decisions; G6 of the stage applies. Returns its path."""
    from exopipe.phase3.lib.catalog import deep_omission
    dod = os.path.join(os.path.dirname(plan.U), 'deep_omission')
    os.makedirs(dod, exist_ok=True)
    table = os.path.join(dod, f'deep_omission_{common.stamp()}Z.csv')
    cache = os.path.join(dod, 'cache')
    seed = _CFG().catalog_omission.TENMIN_RMS_CACHE_SEED
    seed = None if (not seed or os.path.realpath(seed) == os.path.realpath(cache)) else seed
    rows, recs = deep_omission.tenmin_table(list(plan.lsts), table, cache_dir=cache, seed_dir=seed)
    # WAVE2 G6: catalog_omission.ABSENT_WITH_SERVED 'refuse' (ruled R-REFACTOR-20261104-ABSENT-WITH-SERVED-REFUSE) refuses the whole
    # build on a served detection in an ABSENT key; production proceeds - a parity build passes parity/production_20261013.cfg
    # ('production': kept and counted). 0 on the 24 served LSTs (Q review F5).
    deep_omission.absent_gate(rows, deep_omission.tenmin_served_of(recs))
    u.add_input(table)
    u.add_output(table, 'omission_decisions', tier='sublst')
    om = deep_omission.omitted_keys(rows)
    u.note(f'deep-omission decision table {table}: {len(rows)} decisions, {len(om)} (LST, band, night, Stokes) omitted; '
           f'window rms {dict(recs.n)}')
    return table


def build_stage(run, plan):
    S = _CFG().serve
    B = build_dir(plan)
    with run.unit(f'sublst_{plan.run_name}_build', stage='build', tier='sublst', lsts=plan.lsts, parity=plan.parity) as u:
        flow.excl_note(u, 'sublst')
        if not plan.lsts:
            raise NoInput('no LST run to merge (no Run_SLST<hh>B under the runs root, or --lsts empty)')
        bad = generation_mix(plan.lsts)
        if bad:
            u.refuse('G8 sub-LST generation mix: ' + '; '.join(f'{l}: {why}' for l, why in bad)
                     + ' - re-catalogue those LSTs first; the served cumulative stays as it is')
        from exopipe.phase3.lib.catalog import grouping_stage as _GS
        if _GS.mode().design_b:           # WAVE2 GROUPING (item F.3): one grouping table across the LST builds (production: none)
            bad, tables = grouping_mix(plan.lsts)
            u.meta['grouping_tables'] = tables
            if bad:
                u.refuse('G8 design-B grouping mix: ' + '; '.join(f'{l}: {why}' for l, why in bad)[:1500]
                         + ' - the served cumulative stays as it is')
        for l in plan.lsts:
            for f in ('master_catalog.csv', 'pertie_detections.csv'):
                u.add_input(os.path.join(lst_run(l), f))
        log = os.path.join(scripts.ops_root(), 'sublst_cumulative.log') if not plan.parity else \
            os.path.join(plan.out, 'sublst', 'sublst_cumulative.log')
        table = omission_table(plan, u)
        if os.path.isdir(B):
            shutil.rmtree(B)
        # the drop's RE-JUDGING judges a partial survivor's remaining single with the builder's own 7-8 sigma gate under the
        # configured 10-minute size rule (deep_omission.tenmin_size_gate: P3_SIZE_GATE_7_8 = inf under 'envelope', unset - the
        # drop's 1.7 - under 'production'); the stage's prediction (catalog --stages omit) uses the same value
        from exopipe.phase3.lib.catalog import deep_omission as _DO
        _DO.tenmin_rule_check(list(plan.lsts))           # G12: every LST's search jobs under the configured size rule
        g78, g_env, g_rule = _DO.tenmin_size_gate()
        u.meta['rejudge_size_gate'] = dict(rule=g_rule, gate_beams=str(g78), env=g_env)
        u.note(f're-judging 7-8 sigma gate: {g78} beams (size rule {g_rule!r}; merge environment {g_env or "none"})')
        res = scripts.run('sublst_cumulative', [B] + list(plan.lsts), env=dict(g_env, SUBLST_DEEP_OMISSION=table), log=log,
                          unit=u)
        if not res.ok:
            u.refuse(f'merge failed: {res.reason or scripts.last_line(res.tail)}')
        for f in ('deep_omission.csv', 'deep_omission_dropped.csv', 'deep_omission_summary.txt'):
            if not os.path.exists(os.path.join(B, f)):
                u.refuse(f'the merge applied no deep omission: {B}/{f} not written (sublst_deep_omission_drop.finish)')
        u.note('deep omission: ' + open(os.path.join(B, 'deep_omission_summary.txt')).readline().strip())
        u.add_output(os.path.join(B, 'deep_omission_dropped.csv'), 'omission_record', tier='sublst',
                     decisions=os.path.join(B, 'deep_omission.csv'))
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
            # the multi-integration queue, for every Stokes built (sublst_cumulative.sh:79 writes both): the multiples page's
            # queue and the singles page's multiples chain (--mchain-queue, Gregg 2026-10-27), read after the swap
            QM = f'{B}/judgements/queue_p3s3_{ST}_multi.txt'
            has_multi = queues.sublst_multi_queue(Q, QM)
            if has_multi:
                u.add_output(QM, 'served_queue')
            mp = next((pg for pg in plan.pages if pg.stokes == ST and pg.multi), None)
            if mp is None:
                continue
            if not has_multi:
                state['refused'][mp.key] = f'NO-INPUT: {ST}: no multi-integration sources - multiples page not served'
                u.note(state['refused'][mp.key])
                continue
            mv = views.sublst_multi_view(B, ST, QM)
            u.note(f'{ST} multi view: {mv["sources"]} sources')
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
        # G17 again just before the swap (review 2026-10-14 M1): after the swap every page's server - refused or not -
        # reads the new run by path, so the whole build is gated here, on the build directory
        from exopipe.phase3.lib.serve import exclusion_gate
        if exclusion_gate.refuses('sublst', plan.family):
            for pg in plan.pages:
                if pg.family == 'review' and pg.program == 'classify' and pg.key not in state['refused']:
                    scan = exclusion_gate.scan_server_page(pg, B, tier='sublst')
                    if scan.hits:
                        blocking[pg.key] = f'G17 serve-time exclusion gate at the swap: {scan.summary()}'
        if blocking:
            u.refuse('G11 validate before the swap: ' + '; '.join(f'{k}: {why}' for k, why in sorted(blocking.items()))[:800]
                     + ' - nothing swapped in, the served product and its servers untouched')
        src = plan.U_src if plan.parity else plan.U
        from exopipe.phase3.lib.catalog import grouping_stage as GS
        design_b = GS.mode().design_b     # WAVE2 GROUPING (item F.3): design-B page ids are the grouping's identity names -
        marks = {}                        # the journals are CARRIED by member detections (production: copied, page ids kept)
        if design_b and os.path.isdir(src):
            carried, marks = journals.sublst_carry_journals(src, B)
            u.meta['journal_carry'] = carried
            u.note(f'journals carried from {src} by member detections (design B): {carried}')
        else:
            copied = journals.sublst_copy_journals(src, B)
            u.note(f'journals copied from {src}: {len(copied)}')
        prev = plan.U + S.SUBLST_PREV_SUFFIX.format(utc=common.stamp())
        moved = journals.sublst_swap(plan.U, B, prev)
        u.note(f'swapped in: {plan.U} (previous {moved or "none"}); {len(os.listdir(os.path.join(plan.U, "candidates", "I")))}'
               f' I pages, {len(os.listdir(os.path.join(plan.U, "candidates", "V")))} V pages')
        between_src = src if plan.parity else moved
        if between_src:
            if design_b:
                app = journals.sublst_carry_between(between_src, plan.U, marks)
                u.note(f'journal lines recorded on the old servers between the carry and the swap, carried: {app}')
            else:
                app = journals.append_between(between_src, plan.U)
                u.note(f'journal lines recorded on the old servers between the copy and the swap, appended: {app}')
        for J in S.SUBLST_JOURNALS:
            j = os.path.join(plan.U, J, 'judgements.jsonl')
            if os.path.exists(j):
                ref = os.path.join(between_src or src, J, 'judgements.jsonl')
                # a carried journal (design B) holds re-keyed records, not its source's lines: no prefix test, the carry
                # report's consistency test (judgement_journal carry_consistent) applies
                u.add_output(j, 'judgement_journal', carried_from=(ref if os.path.exists(ref) and not design_b else None))
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
