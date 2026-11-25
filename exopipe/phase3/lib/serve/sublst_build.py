"""serve --tier sublst: the cumulative 10-minute product Run_SLSTB on 8787/8788 (+ multi-integration 8789/8790).
Port of sublst_cumulative.sh (+ the prepare/start halves of sublst_serve.sh it calls); stages
build -> views -> test -> carry -> swap -> start -> hub -> verify (+ page-index alone). carry, swap and start run only together (the journal
lines written on the old servers between the copy and the swap are appended at the swap, sublst_cumulative.sh:106-120).

build  (unit sublst_<run>_build)  the build lock (G12; sublst_cumulative.sh:23 flock, servers never inherit it); the
       version each LST run names held while the build reads it (G31: a placement that switches one during the merge
       refuses the build, whose follow-up reads the new version; lib/serve/versions.pinned); the
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
       deep_omission.tenmin_size_gate); the MASTER IDENTITY (G13; Gregg 2026-11-17, R-GREGG-20261117-TENMIN-MASTER-
       IDENTITY-BEFORE-RELEASE1; sublst_cumulative.sh:102-108, exopipe_ops fa669f0): under production's grouping (the parity
       file) the design-B grouping of exactly B's detections (batch, into B.master_grouping beside B) and
       sublst_master_identity.py B <it>, which merges every served identity that grouping places together (merge-only;
       its tests M1-M4 refuse); under design B the LST builds already name one identity per design-B identity across LSTs
       (G8: one grouping table); in both, the outcome is tested on B and REFUSES: no two served identities hold detections
       of one design-B identity, every served member detection is in the grouping, no (source_id, band) under two
       identities (served_identity_splits); build_localizations.py with the per-case tie and the common frame (G9);
       the localization figures (md5-keyed reuse from the served run, else drawn, 4 at a time, 600 s each; refuse a
       missing one - port of the PYF heredoc :36-61); p3t_pages_index.py per tier (G10). Output class sublst_cumulative.
views  (unit sublst_<run>_views)  on B: the union with the Stokes key (G5), the review_dir merge with
       pages.REVIEW_DIR_MERGE['sublst'] (the merge's config key is pages'; the call runs here because the union is
       built inside this build), the tier queues sectioned (sb, wb when present), p3s3_stokes_queue.py, the per-Stokes
       views (G6), the multi-integration queues and views (sublst_cumulative.sh:77-95), then the bright-source filter and
       the scintillation index on B (bright_source_filter_precompute.py, scintillation_index.py --tier tenmin; refusing,
       sublst_cumulative.sh:172-181), then the TESS coverage on B (tess/frame_times_tenmin.py, tess/tess_coverage.py;
       refusing, sublst_cumulative.sh:192-200; lib/serve/tess.py). G11: a refusal here leaves the served product and its
       servers untouched.
test   App probes and content tests on B (expected index sources), G17 report.
carry/swap/start  (unit sublst_<run>_swap) journals copied into B, then B PLACED as a new unsealed VERSION V of the served
       run U (lib/serve/versions.py: one rename of B into <U>.versions/<vid>/Run_SLSTB; production renames U to U.prev_<utc>
       and B to U, sublst_cumulative.sh:101-105, a window in which U does not exist); the page context on V (unit
       sublst_<run>_page_context: page_context_tenmin.py V --workers serve.SUBLST_PAGE_CONTEXT_WORKERS, the panels'
       <png>.page.json run-context sidecars; page text, not a gate - a failure is recorded and the serve continues,
       sublst_cumulative.sh:295-300); then (unit sublst_<run>_switch) V's page index (flow.build_page_index, class
       detections_index), V SEALED (its completion record: every page's queue and view present, the page index current, the
       manifest of every file) and U SWITCHED to V by one atomic rename of the pointer, the between lines appended (from the
       version U named before) - the switch and that catch-up under the run's journal locks, which the two-port server's
       POSTs take, and a POST answered only by a page of the version U names (two_port Store.get current): no label is left
       in the old version; a refusal before the switch moves V back to B (nothing served changes, the next attempt finds
       the build). Production does the page context and the index AFTER its swap, on the served run: here every file the
       pages read is in V before any reader can reach it. Then the four pages registered in ONE registration and each one's
       build waited on (flow.start_stage; sublst_cumulative.sh:301-307 -> sublst_register_pages.py: no server of its own since
       2026-10-22; its page-index call finds V's index current). page-index without start: the served run's index alone (unit
       sublst_<run>_page_index; it writes into the served version - the repair of an index that went dirty).
retire (unit sublst_<run>_retire; never in the default stages) RETENTION of the run's versions (versions.retire): without
       --retire-plan a dry run (the plan written, nothing deleted); with --retire-plan <that plan> the planned versions
       deleted, each still older than the newest serve.RUN_VERSIONS_KEEP complete ones, not served, not held, not being built.
hub    10-min / 10-min-multiples set; '10-min-sublst' removed (sublst_cumulative.sh:145-150).
THE COALESCING BUILDER (requirement 11 of plan/ORCHESTRATION_DESIGN_20261117.md, approved 2026-11-17: "Publish immutable
       versions behind an atomic pointer. A build pins its inputs at its start, and one coalescing builder"): an invocation
       with build or start goes through the run's slot (lib/serve/coalesce.py, slot_for(U)): it requests a build, then
       builds while a request is pending - its own, and once more for all requests made while it built (one follow-up
       however many; a follow-up makes the plan again at its start, so it reads the LST runs placed by then). An invocation
       that finds another build of the run running records unit sublst_<run>_coalesced, no-input (COALESCED: that builder
       builds once more after its build, which begins after this request) and exits 3; the per-hour chain goes on. The
       sub-LST placement (pages --tier sublst, stage place) requests a build in the slot of the cumulative run beside the
       LST runs it placed. The build lock (G12) stays: production's sublst_cumulative.sh takes the same file.
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
import sys
from concurrent.futures import ThreadPoolExecutor

from exopipe.phase3.core.record import NoInput, Refusal
from exopipe.phase3.lib.serve import common, flow, journals, queues, scripts, tess, versions, views
from exopipe.phase3.lib.serve.plan import ReviewHook

STAGES = ('build', 'views', 'test', 'carry', 'swap', 'page-index', 'start', 'hub', 'verify', 'retire')
DEFAULT_OFF = ('retire',)          # run only when named in --stages (steps/serve.py)
PLAN_HOOK = ReviewHook()      # the sub-LST review pages' words (plan.FAMILY_MODULES)


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


def present_grouping(stokes):
    """(table path, delta_build.read_grouping(table)) of the grouping the LST builds read NOW: the grouping state's own
    table of this Stokes (grouping_stage.recorded_table: the latest group run is done and recorded it), read ONCE - rows,
    sha1 and sha256 are the same bytes, and the sha1 must be the recorded one. Raises StageRefusal."""
    from exopipe.phase3.lib.catalog import delta_build as DB
    from exopipe.phase3.lib.catalog import grouping_stage as GS
    table, sha = GS.recorded_table(GS.sublst_dir(_CFG()), stokes)
    g = DB.read_grouping(table)
    if g[2] != sha:
        raise GS.StageRefusal(f'{table} changed while the cumulative read it (sha1 {g[2][:12]} vs the recorded {sha[:12]}) - '
                              f'REFUSING')
    return table, g


def grouping_mix(lsts, present=present_grouping):
    """Design B (catalog_grouping.GROUPING 'design-b'; WAVE2 GROUPING; review 2026-11-04, F3): every LST run's catalogs
    must stand on ONE grouping table per Stokes, the PRESENT one (``present``: Stokes -> (path, read_grouping result); the
    grouping state's table, never a majority of the directories). A regroup renames identities (display names follow mean
    positions; 121 I / 15 V renamed between the 10-25 and 11-03 inputs, the 11-04 review), and the cumulative merges the LST
    runs by name: a mix would show one identity under two names or join two under one.
      - EVERY catalog directory, whatever table its grouping_table.json (the builder runner's provenance,
        lib/catalog/wraps/builder_grouping.py, or a view's, lib/catalog/delta_build.view_catalog) names, passes only when
        THIS gate finds, from the directory's own record, that it stands under the present table
        (lib/catalog/delta_build.standing: the record bound to the directory by its sha256, every detection grouped as
        recorded, the populations of the hour and key equal but for the detections the record lists as removed before the
        builder). A directory naming the present sha1 is checked too: the table identity is not taken from the directory's
        own provenance (DECISION-20261114-PR51-F3-POPULATION-CHECK-ON-EVERY-DIRECTORY; the carried-root rule,
        DECISION-20261114-PR47-CARRIED-ROOT-READERS-TAKE-TABLE-FROM-INDEX). No stored certificate is trusted;
      - anything else refuses: no provenance, a table that cannot be read, a directory that does not stand, and a MISSING
        directory: an LST of ``lsts`` whose run holds no catalog directory of a Stokes and key class (sb | wb) of which the
        present table has detections of that hour (judged for each Stokes whose present table was read).
    Returns ([(lst, reason)], {Stokes: {table sha1: [lsts]}}, {'present': {Stokes: path, sha1, sha256}, 'certified': [one
    dict per directory passed by the certificate: LST, key, record name/sha1/sha256, both tables' digests, detections]})."""
    from exopipe.phase3.lib.catalog.wraps import builder_grouping as BG
    from exopipe.phase3.lib.catalog import delta_build as DB
    from exopipe.phase3.lib.catalog import grouping_stage as GS
    bad, by, now, certified = [], {}, {}, []
    for l in lsts:
        R = lst_run(l)
        dirs = sorted(d for d in glob.glob(os.path.join(R, 'source_catalogs_[IV]_[sw]b')) if os.path.isdir(d))
        if not dirs:
            continue
        for d in dirs:
            S, kind = os.path.basename(d).split('_')[2:4]
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
            if S not in now:
                try:
                    now[S] = present(S)
                except (GS.StageRefusal, DB.DeltaRefusal, OSError, ValueError, KeyError, csv.Error) as e:
                    now[S] = None
                    bad.append(('*', f'Stokes {S}: the present grouping table cannot be read ({str(e)[:300]})'))
            if now[S] is None:
                continue
            table, g = now[S]
            try:                # EVERY directory, also one naming the present sha1 (its own provenance is not the pass's
                #                 index: DECISION-20261114-PR51-F3-POPULATION-CHECK-ON-EVERY-DIRECTORY)
                st = DB.standing(d, g, table, l, f'{S}_{kind}')
            except (DB.DeltaRefusal, ValueError, KeyError, TypeError) as e:     # a malformed record: a refusal, not a crash
                st = dict(ok=False, why=f'its record could not be certified ({type(e).__name__}: {str(e)[:200]})')
            if st['ok'] is True:
                certified.append(dict(lst=l, key=st['key'], record=st['record'], record_sha1=st['record_sha1'],
                                      record_sha256=st['record_sha256'], table_from=st['table_from'], table_to=st['table_to'],
                                      detections=st['record_detections']))
            else:
                fix = ('make this hour\'s catalog view again in a night pass (catalog --tier sublst; its delta builds), or '
                       'build this LST again from stage build' if st.get('record') == DB.VIEW_ROWS else
                       'build this LST again from stage build')
                named = ('names the present grouping table' if sha == g[2] else
                         f'names the grouping table {sha[:12]}, the present table is')
                bad.append((l, f'{os.path.basename(d)} {named} {g[2][:12]} ({table}), and the directory does not stand under '
                               f'it: {st["why"]} - {fix} (the cumulative joins LST runs by identity name)'))
    for S, v in now.items():            # a key of an hour the present table holds detections of must have its directory
        if not v:
            continue
        cls = {k: DB.key_class(k) for k in ('sb', 'wb')}
        held = {(r.get('lst'), k) for r in v[1][0].values() if r.get('stokes') == S
                for k in ('sb', 'wb') if cls[k](r.get('band') or '')}
        for l in lsts:
            for k in ('sb', 'wb'):
                d = os.path.join(lst_run(l), f'source_catalogs_{S}_{k}')
                if (l, k) in held and not os.path.isdir(d):
                    bad.append((l, f'{os.path.basename(d)} is missing: the present grouping table {v[1][2][:12]} holds '
                                   f'detections of {l} Stokes {S} {k} - build this LST again'))
    return (bad, {S: {sha: sorted(ls) for sha, ls in v.items()} for S, v in by.items()},
            dict(present={S: dict(path=os.path.abspath(v[0]), sha1=v[1][2], sha256=v[1][3]) for S, v in now.items() if v},
                 certified=certified))


def figure_key(loc_row, off_rows):
    """md5 of the identity's localization rows (sublst_cumulative.sh:44, verbatim expression)."""
    return hashlib.md5(repr((sorted(loc_row.items()), sorted(sorted(o.items()) for o in off_rows))).encode()).hexdigest()


def figures(B, served, unit=None, log=None, env=None):
    """Port of the PYF heredoc (sublst_cumulative.sh:36-61); env: extra variables of the figure script (the cross-Stokes
    link's P3_LOC_LINKS: the figure localizes again, as the table did). Returns (n multi, n linked, n drawn, [failures])."""
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
    env = dict({'P3_LOC_CATS': S.SUBLST_LOC_ENV['P3_LOC_CATS']}, **(env or {}))

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


MASTER_GROUPING = '.master_grouping'      # B + this: the design-B grouping of B's detections (production's grouping only)


def served_identity_splits(B, stokes, gid):
    """The outcome of R-GREGG-20261117-TENMIN-MASTER-IDENTITY-BEFORE-RELEASE1 on a cumulative build B, read-only (the union-find
    of production's sublst_master_identity.py, exopipe_ops fa669f0): the served identities of Stokes ``stokes`` (the
    identity_id of every survivors row of source_catalogs_<S>_sb / _wb, as production's script reads them), linked through the
    design-B identity (``gid``: cand_id -> identity_id) of each served member detection (a members row whose (source_id, band)
    has a survivors row). Returns dict(splits=[sorted served identities of one component, for every component of two or more
    - each is one source shown under several identities], members=n served member detections, unlinked=[cand_ids not in
    gid], twice=[(source_id, band) under two identities], blank=[(source_id, band) of survivors rows without an identity_id:
    production's script keys every row by identity_id, so the gate refuses such a row rather than key it by its J-name],
    identities=n served identities)."""
    surv, twice, blank = {}, [], []
    for t in ('sb', 'wb'):
        for p in sorted(glob.glob(os.path.join(B, f'source_catalogs_{stokes}_{t}', '*_sources_survivors.csv'))):
            with open(p, newline='') as f:
                for r in csv.DictReader(f):
                    k, i = (r['source_id'], r['band']), (r.get('identity_id') or '').strip()
                    if not i:
                        blank.append(k)
                        continue
                    if k in surv and surv[k] != i:
                        twice.append(k)
                    surv[k] = i
    parent = {i: i for i in set(surv.values())}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    first, members, unlinked = {}, 0, []
    for t in ('sb', 'wb'):
        for p in sorted(glob.glob(os.path.join(B, f'source_catalogs_{stokes}_{t}', '*_source_members.csv'))):
            with open(p, newline='') as f:
                for r in csv.DictReader(f):
                    k = (r['source_id'], r['band'])
                    if k not in surv:
                        continue                  # a member of a source that did not survive: not served
                    members += 1
                    b = gid.get(r['cand_id'])
                    if b is None:
                        unlinked.append(r['cand_id'])
                        continue
                    if b in first:
                        a, c = find(first[b]), find(surv[k])
                        if a != c:
                            parent[c] = a
                    else:
                        first[b] = surv[k]
    comp = collections.defaultdict(list)
    for i in parent:
        comp[find(i)].append(i)
    return dict(splits=sorted(sorted(m) for m in comp.values() if len(m) > 1), members=members, unlinked=unlinked,
                twice=sorted(set(twice)), blank=sorted(set(blank)), identities=len(parent))


def _served_stokes(B):
    return [S for S in ('I', 'V') if any(os.path.isdir(os.path.join(B, f'source_catalogs_{S}_{t}')) for t in ('sb', 'wb'))]


def batch_grouping(B, G, label, u=None):
    """Design B's grouping (identity level; lib/catalog/grouping.group_catalog, batch: every ID new) of exactly the detections
    of the merged build B (B/master_catalog.csv + B/pertie_detections.csv), each Stokes into G/grouping_<S>/ with its
    content tests (tests_grouping.failures) - the input production's master identity reads (sublst_cumulative.sh:105: 'the
    design-B grouping (exopipe.phase3 catalog --tier sublst --stages group) of the published LST runs', run on exactly the
    build's detections, its test M1). Run under production's grouping too (the parity file), whose catalog step has no group
    stage: the mode is design B's by construction. A Stokes without detections gets a header-only table. Raises
    grouping_stage.StageRefusal. Each table with detections is recorded on ``u`` (class grouping_table: its content test).
    Returns {Stokes: (detections.csv path, counts)}."""
    from exopipe.phase3.lib.catalog import grouping as GR
    from exopipe.phase3.lib.catalog import grouping_stage as GS
    from exopipe.phase3.lib.catalog import tests_grouping as TG
    m = GS.Mode('design-b', 'one-rule', 'multiples-level')      # design B's own values (recorded in summary.json only)
    mc, pt = os.path.join(B, 'master_catalog.csv'), os.path.join(B, 'pertie_detections.csv')
    if os.path.exists(G):
        shutil.rmtree(G)
    out = {}
    for S in ('I', 'V'):
        d = GS.group_dir(G, S)
        if not GS.stokes_rows(mc, S):
            os.makedirs(d, exist_ok=True)
            out[S] = (GS._write_csv(os.path.join(d, GS.DETECTIONS), GS.DET_COLS, []), dict(detections=0, identities=0))
            continue
        try:
            det = GS.load_det('10min', S, mc, pt)
            st = GR.GroupingState.load(G, '10min', S, new=True)
            res = GR.group_catalog(det, '10min', S, state=st, run_label=label, mode='batch')
        except GR.GroupingRefused as e:
            raise GS.StageRefusal(f'{S}: {e}') from None
        paths, summ = GS.write_products(det, res, '10min', S, G, dict(master_catalog=mc, pertie=pt), m, d=d)
        paths['state_next'] = res.state.save(os.path.join(d, 'state_next'))
        bad = TG.failures(paths, tier='10min', stokes=S, input=mc)
        if bad:
            raise GS.StageRefusal(f'{S}: the grouping of the build failed its content tests: ' + '; '.join(bad[:6]))
        out[S] = (paths['detections'], summ['counts'])
        if u is not None:
            u.add_output(paths['detections'], 'grouping_table', tier='10min', stokes=S, input=mc)
    return out


def stokes_link_stage(B, u):
    """G30, the cross-Stokes link of the build (lib/catalog/stokes_link.py; R-GREGG-20261124-CROSS-STOKES-LINK-TEST-APPROVED):
    under design B with both Stokes served, the link the grouping stage computed from the very tables G8 read (their sha1) is
    copied into B/stokes_link/ - the build's own copy, which its localization and figures (P3_LOC_LINKS) and its pages (the
    link at the top of each source page, lib/serve/stokes_banner.py) read. A missing or stale link REFUSES. Returns the
    localization's extra environment ({} under production's grouping or with one Stokes served)."""
    from exopipe.phase3.lib.catalog import grouping_stage as GS
    from exopipe.phase3.lib.catalog import stokes_link as SL
    present = u.meta.get('grouping_present') or {}
    if not GS.mode().design_b or not {'I', 'V'} <= set(_served_stokes(B)):
        u.meta['stokes_link'] = None
        return {}
    if not {'I', 'V'} <= set(present):
        u.refuse(f'G30 cross-Stokes link: both Stokes are served but G8 read the grouping of {sorted(present)} only')
    out = {os.path.dirname(os.path.dirname(present[S]['path'])) for S in ('I', 'V')}
    if len(out) != 1:
        u.refuse(f'G30 cross-Stokes link: the I and V grouping tables are in two directories {sorted(out)}')
    try:
        p = SL.publish(out.pop(), B, {S: present[S]['sha1'] for S in ('I', 'V')})
    except SL.LinkRefused as e:
        u.refuse(f'G30 cross-Stokes link: {e}')
    u.add_output(p, 'stokes_link')
    counts = SL.read_summary(os.path.dirname(p))['counts']
    u.meta['stokes_link'] = counts
    u.note(f"G30 cross-Stokes link: {counts['linked_pairs']} linked I/V pairs, {counts['ambiguous_components']} ambiguous "
           f"components; the localization merges on it")
    return {SL.LOC_ENV: p}


def master_identity(plan, B, u):
    """G13, the master identity of the 10-minute cumulative (R-GREGG-20261117-TENMIN-MASTER-IDENTITY-BEFORE-RELEASE1, Gregg
    2026-11-17: "Why can't you just merge identities into a new Master identity that supersedes the older ones?"; the cross-LST
    half of R-CROSS-TIER-SINGLE-IDENTITY of 2026-09-05). Production (exopipe_ops fa669f0) runs sublst_master_identity.py right
    after the merge, REQUIRED (sublst_cumulative.sh:102-108).
      production's grouping (catalog_grouping.GROUPING 'production', the parity file): the LST runs are grouped per LST and
        joined by J-name; batch_grouping of B into B.master_grouping, then sublst_master_identity.py B B.master_grouping (its
        tests M1-M4; it rewrites only the identity fields of B's catalogs and pages_input; record
        B/source_catalogs/master_identity_<S>.csv + master_identity_summary.txt).
      design B (the default): every LST build stands on ONE grouping table (G8, grouping_mix), whose identities span the LSTs,
        so the merge already holds one identity per design-B identity: nothing is rewritten.
    Then, in both, the outcome on B (served_identity_splits against the grouping used) REFUSES on any split identity, any
    served member detection the grouping does not hold, or a (source_id, band) under two identities."""
    from exopipe.phase3.lib.catalog import delta_build as DB
    from exopipe.phase3.lib.catalog import grouping_stage as GS
    gid = {}
    if GS.mode().design_b:
        present = u.meta.get('grouping_present') or {}
        for S in _served_stokes(B):
            if S not in present:
                u.refuse(f'G13 master identity: Stokes {S} is served but no present design-B grouping table was read (G8)')
            try:
                rows, _mg, sha1, _sha256 = DB.read_grouping(present[S]['path'])
            except (DB.DeltaRefusal, OSError, KeyError, csv.Error) as e:
                u.refuse(f'G13 master identity: the present grouping table of {S} cannot be read ({str(e)[:300]})')
            if sha1 != present[S]['sha1']:
                u.refuse(f'G13 master identity: {present[S]["path"]} changed since G8 read it (sha1 {sha1[:12]} vs '
                         f'{present[S]["sha1"][:12]})')
            gid[S] = {c: r['identity_id'] for c, r in rows.items() if r.get('stokes', S) == S}
        u.meta['master_identity'] = dict(how='design B: one grouping table across the LST builds (G8); nothing rewritten')
    else:
        G = B + MASTER_GROUPING
        try:
            tabs = batch_grouping(B, G, f'{plan.run_name}.master_grouping', u)
        except GS.StageRefusal as e:
            u.refuse(f'G13 master identity: the design-B grouping of the build refused: {str(e)[:1200]}')
        # its own log, fresh with G (batch_grouping removed G): the M-lines and the OK line are read from the log's tail, so a
        # log other runs append to (production's sublst_cumulative.log) could hand this run another run's lines
        res = scripts.run('sublst_master_identity', [B, G], log=os.path.join(G, 'master_identity.log'), unit=u)
        if not res.ok:
            tests = [ln.strip() for ln in (res.tail or '').splitlines() if ln.strip()[:2] in ('M1', 'M2', 'M3', 'M4')]
            u.refuse(f'G13 master identity: sublst_master_identity.py refused (exit {res.rc}; nothing rewritten): '
                     + ('; '.join(tests[:6]) or res.reason or scripts.last_line(res.tail)))
        ok = [ln for ln in (res.tail or '').splitlines() if ln.startswith('MASTER-IDENTITY OK')]
        if not ok:
            u.refuse('G13 master identity: sublst_master_identity.py ended without its MASTER-IDENTITY OK line: '
                     + scripts.last_line(res.tail))
        u.note(ok[-1][:600])
        u.meta['master_identity'] = dict(how='production grouping: sublst_master_identity.py on the design-B grouping of B',
                                         grouping=G, counts={S: c for S, (_p, c) in tabs.items()})
        for S in ('I', 'V'):
            rec = os.path.join(B, 'source_catalogs', f'master_identity_{S}.csv')
            if not os.path.exists(rec):          # the script writes one per Stokes after its OK, header-only when nothing merges
                u.refuse(f'G13 master identity: sublst_master_identity.py printed its OK line but wrote no {rec}')
            u.add_output(rec, 'master_identity_record', tier='sublst', stokes=S, grouping=G)
            with open(tabs[S][0], newline='') as f:
                gid[S] = {r['cand_id']: r['identity_id'] for r in csv.DictReader(f)}
    outcome = {}
    for S in _served_stokes(B):
        r = served_identity_splits(B, S, gid.get(S, {}))
        outcome[S] = dict(identities=r['identities'], members=r['members'], splits=len(r['splits']),
                          unlinked=len(r['unlinked']), twice=len(r['twice']), blank=len(r['blank']))
        why = ([f'{len(r["splits"])} sets of served identities that hold detections of one design-B identity, so one source '
                f'is served under several identities (e.g. {r["splits"][:3]})'] if r['splits'] else []) + \
              ([f'{len(r["unlinked"])} of {r["members"]} served member detections are not in the grouping (e.g. '
                f'{r["unlinked"][:3]})'] if r['unlinked'] else []) + \
              ([f'{len(r["twice"])} (source_id, band) keys under two identities (e.g. {r["twice"][:3]})'] if r['twice'] else []) + \
              ([f'{len(r["blank"])} survivors rows without an identity_id (e.g. {r["blank"][:3]})'] if r['blank'] else [])
        if why:
            u.meta['master_identity']['outcome'] = outcome
            u.refuse(f'G13 master identity, Stokes {S}: ' + '; '.join(why) + ' - the served cumulative stays as it is')
        u.check(f'G13_master_identity_{S}', True, value=outcome[S],
                detail='every design-B identity is one served identity; every served member detection is in the grouping')
    u.meta['master_identity']['outcome'] = outcome
    u.note('master identity: ' + '; '.join(f'{S} {o["identities"]} served identities, {o["members"]} member detections, '
                                           f'0 split' for S, o in outcome.items()))


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


def pin_lsts(plan, u, stack):
    """G31 THE LST RUNS READ ARE PINNED (lib/serve/versions.py): the version each LST run names when the build starts is held
    (retention never deletes it under the merge) and stamped; the merge (production's sublst_cumulative.py, in place) reads the
    runs by their names, so after it the build refuses when any was switched meanwhile (versions.moved) - a placement during
    the build requested the one follow-up build that reads the new version (lib/serve/coalesce.py). Holds the versions until
    ``stack`` closes (the build stage's end); records {LST: version run read} as u.meta['lst_versions']. Returns the pins."""
    runs = {lst_run(l): l for l in plan.lsts}
    pins = stack.enter_context(versions.pinned(list(runs), note=f'serve --tier sublst build of {plan.run_name}'))
    u.meta['lst_versions'] = {runs[U]: d for U, (d, _s) in pins.items()}
    return pins


def build_stage(run, plan):
    S = _CFG().serve
    B = build_dir(plan)
    with run.unit(f'sublst_{plan.run_name}_build', stage='build', tier='sublst', lsts=plan.lsts, parity=plan.parity) as u, \
            contextlib.ExitStack() as held:
        flow.excl_note(u, 'sublst')
        if not plan.lsts:
            raise NoInput('no LST run to merge (no Run_SLST<hh>B under the runs root, or --lsts empty)')
        pins = pin_lsts(plan, u, held)
        bad = generation_mix(plan.lsts)
        if bad:
            u.refuse('G8 sub-LST generation mix: ' + '; '.join(f'{l}: {why}' for l, why in bad)
                     + ' - re-catalogue those LSTs first; the served cumulative stays as it is')
        from exopipe.phase3.lib.catalog import grouping_stage as _GS
        if _GS.mode().design_b:           # WAVE2 GROUPING (item F.3): one grouping table across the LST builds (production: none)
            bad, tables, how = grouping_mix(plan.lsts)
            u.meta['grouping_tables'] = tables
            u.meta['grouping_present'] = how['present']
            u.meta['grouping_certified'] = how['certified']
            u.meta['grouping_refusals'] = [dict(lst=l, why=why) for l, why in bad]   # in full: the reason below is cut
            if how['certified']:
                u.note(f'G8 design-B grouping: {len(how["certified"])} catalog directory(ies) built under an earlier grouping '
                       f'table stand under the present one (certified here from their own records)')
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
        moved = versions.moved(pins)                     # the merge read each LST run by its name: one version of each?
        if moved:
            u.refuse(f'G31 LST runs placed during the build: {", ".join(moved)} switched to a new version while the merge read '
                     f'them - the build may mix two versions of an LST; the served cumulative stays as it is (the placement '
                     f'requested the follow-up build, lib/serve/coalesce.py)')
        for f in ('deep_omission.csv', 'deep_omission_dropped.csv', 'deep_omission_summary.txt'):
            if not os.path.exists(os.path.join(B, f)):
                u.refuse(f'the merge applied no deep omission: {B}/{f} not written (sublst_deep_omission_drop.finish)')
        u.note('deep omission: ' + open(os.path.join(B, 'deep_omission_summary.txt')).readline().strip())
        u.add_output(os.path.join(B, 'deep_omission_dropped.csv'), 'omission_record', tier='sublst',
                     decisions=os.path.join(B, 'deep_omission.csv'))
        master_identity(plan, B, u)                      # G13: right after the merge, before the combined positions
        links = stokes_link_stage(B, u)                  # G30: the cross-Stokes link the localization and the pages merge on
        # SPEED (design 2.3a stage 4): production's own row-index and shard switches - the same tables (serve.SUBLST_LOC_SPEED_ENV)
        env = dict(S.SUBLST_LOC_ENV, **dict(S.SUBLST_LOC_SPEED_ENV or {}), P3_TENMIN_PERTIE=f'{B}/pertie_detections.csv',
                   P3_LOC_OUT=f'{B}/source_catalogs', **links)
        u.meta['localization_speed_env'] = dict(S.SUBLST_LOC_SPEED_ENV or {})
        res = scripts.run('build_localizations', [B], env=env, log=log, unit=u)
        if not res.ok:
            u.refuse(f'G9 localizations failed: {res.reason or scripts.last_line(res.tail)}')
        n, linked, drawn, failed = figures(B, plan.U_src if plan.parity else plan.U, unit=u, log=log, env=links)
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
        filters(u, B, log)
        tess_coverage(u, plan, B, log)
        u.meta['refused_pages'] = dict(state['refused'])
    return u


FILTER_OUTPUTS = (('bright_source_filter', 'bright_source_filter.npz'), ('scintillation_index', 'scintillation_index.csv'))


def filters(u, B, log):
    """BRIGHT-SOURCE AND SCINTILLATION FILTERS (sublst_cumulative.sh:172-181; Gregg 2026-11-12, R-GREGG-20261112-FILTERS-FINALIZE-
    BRIGHT-AND-SCINTILLATION): bright_source_filter_precompute.py and scintillation_index.py, --tier serve.SUBLST_FILTER_TIER, in
    place on the new build B after its views (the bright file reads the survivors views) and before the swap -
    <B>/bright_source_filter.npz (the 2-D bright-source panel) and <B>/scintillation_index.{csv,json} (the 10-minute pages' slider,
    default off). A failure refuses the views unit, so nothing is swapped in (G11; production: the served product and its pages
    untouched)."""
    S = _CFG().serve
    for name, fn in FILTER_OUTPUTS:
        res = scripts.run(name, [B, '--tier', S.SUBLST_FILTER_TIER], log=log, unit=u, env={'NUMPY_MADVISE_HUGEPAGE': '0'})
        if not res.ok:
            u.refuse(f'{scripts.SCRIPTS[name][0]} on the new build: {res.reason or scripts.last_line(res.tail)} - nothing swapped '
                     f'in (sublst_cumulative.sh:177-180)')
        u.note(f'{name}: {scripts.last_line(res.tail)[:200]}')
        u.add_output(os.path.join(B, fn), name, tier=S.SUBLST_FILTER_TIER)


def tess_coverage(u, plan, B, log):
    """TESS COVERAGE (sublst_cumulative.sh:192-200; FIX-20261115-TESS-COVERAGE-PRODUCER-SNAPSHOT-FORCED-10MIN-WIDEBAND), after the
    filters and before the carry and the swap, in place on the new build B (lib/serve/tess.py): tess/frame_times_tenmin.py --run B
    --image-root <the 10-minute tree B> [--reuse <the served run>/frame_times.csv when that file is not empty: a window whose image
    is unchanged keeps its times] - <B>/frame_times.csv, the window times (B has no page_context_frames.csv yet: the page context
    runs after the swap) - then tess/tess_coverage.py --run B - <B>/tess_coverage.csv, which the server reads at page build. A
    failure refuses the views unit, so nothing is swapped in (G11; production: the served product and its pages untouched). A parity
    build reads the orbit table with --no-refresh (never writes the shared table; lib/pages/tess_orbits.reader_args)."""
    cs = _CFG().catalog_sublst
    tree = cs.TENMIN_TREE.format(root=_CFG().paths.SUBLST_ROOT, tree=cs.ARM_TREES['B'])   # sublst_cumulative.sh:197 --image-root
    after = 'nothing swapped in (sublst_cumulative.sh:197-199)'
    tess.tenmin_frame_times(u, B, tree, os.path.join(plan.U, 'frame_times.csv'), log, after=after)
    tess.coverage(u, B, [], log, after=after, parity=bool(getattr(plan, 'parity', False)))


def carry_page_context_cache(src, B, u=None):
    """SPEED (design 2.3a stage 4): the page context's frame-noise table of the served run ``src`` copied into the build ``B``
    before the swap (serve.SUBLST_PAGE_CONTEXT_CACHE), so page_context_tenmin.py - run on the swapped-in run - finds every frame
    it measured before and re-validates it by size and mtime instead of measuring every frame again; a value is stored as
    its repr, so a carried value is the float a new measurement of the unchanged file gives. A table the build already holds is
    left; a served run without one is production's state (every frame measured). Returns a dict for the unit's record."""
    name = _CFG().serve.SUBLST_PAGE_CONTEXT_CACHE
    if not name:
        return dict(carried=False, why='serve.SUBLST_PAGE_CONTEXT_CACHE is not set (as production)')
    a, b = os.path.join(src, name), os.path.join(B, name)
    if os.path.exists(b):
        return dict(carried=False, why=f'the build already holds {name}')
    if not os.path.isfile(a) or os.path.getsize(a) == 0:
        if u is not None:
            u.note(f'page context: no frame-noise table {a} to carry - every frame is measured (as production)')
        return dict(carried=False, why=f'no {a}')
    # only the rows of frames the 10-minute tree still lists (its frames.csv sub_path: the frames page_context_tenmin.py
    # looks up, :111-133): the tool writes back every row it loaded, so a carried table would otherwise keep the rows of
    # frames the rebuild lanes removed or renamed, growing from build to build
    cs = _CFG().catalog_sublst
    tree = cs.TENMIN_TREE.format(root=_CFG().paths.SUBLST_ROOT, tree=cs.ARM_TREES['B'])   # page_context_tenmin.py:48 TROOT
    try:
        keep = set()
        for fl in glob.glob(os.path.join(tree, '*_*', 'frames.csv')):
            with open(fl, newline='') as f:
                keep.update(r['sub_path'] for r in csv.DictReader(f))
        if not keep:
            raise OSError(f'no frames.csv rows under {tree}')
        h, n, dropped = hashlib.sha1(), 0, 0
        with open(a, newline='') as f, open(b + '.part', 'w', newline='') as o:
            head = f.readline()                          # each kept line copied as written (a table with nothing to drop is
            ip = next(csv.reader([head])).index('path')  # carried byte for byte); one row per line (paths hold no newline)
            o.write(head)
            h.update(head.encode())
            for ln in f:
                r = next(csv.reader([ln]), None)
                if r and r[ip] in keep:
                    o.write(ln)
                    h.update(ln.encode())
                    n += 1
                else:
                    dropped += 1
        shutil.copystat(a, b + '.part')
        os.replace(b + '.part', b)
    except (OSError, ValueError, KeyError, StopIteration) as e:   # speed only: never blocks the swap
        with contextlib.suppress(OSError):
            os.remove(b + '.part')
        if u is not None:
            u.note(f'page context: the frame-noise table {a} was not carried ({e}) - every frame is measured (as production)')
        return dict(carried=False, why=f'not carried: {e}')
    if u is not None:
        u.note(f'page context: frame-noise table carried from {a}: {n} frames to re-validate by size and mtime ({dropped} '
               f'rows of frames the 10-minute tree no longer lists dropped)')
    return dict(carried=True, source=a, frames=n, dropped=dropped, sha1=h.hexdigest()[:12])


def carry_swap_stage(run, plan, state):
    """Copy the journals into B, then place B as a new unsealed version of U (versions.place: one rename; nothing is served
    from it before switch_stage). Returns the unit; state['version'] = the Version. G11 (validate before the swap,
    sublst_cumulative.sh:65-74): a page refused on the build by the views or test stage (anything but a multiples page with
    no multi-integration source, sublst_cumulative.sh:80) refuses the swap - nothing placed, the served version and its pages
    untouched."""
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
        # the seed reads the served journals under the run's JOURNAL LOCKS (the two-port POST lock of these journals): a
        # consistent cut, no label write half done in it; the catalogues and member maps are loaded before the locks
        if design_b and os.path.isdir(src):
            maps = journals.sublst_carry_maps(src, B)
            with versions.journal_locks(plan.U, S.SUBLST_JOURNALS):
                snap = journals.sublst_journal_snapshot(src)
            carried, marks = journals.sublst_carry_journals(src, B, snapshot=snap, maps=maps)
            u.meta['journal_carry'] = carried
            u.note(f'journals carried from {src} by member detections (design B; the cut read under the journal locks): '
                   f'{carried}')
        else:
            with versions.journal_locks(plan.U, S.SUBLST_JOURNALS):
                copied = journals.sublst_copy_journals(src, B)
            u.note(f'journals copied from {src} under the journal locks: {len(copied)}')
        u.meta['page_context_cache'] = carry_page_context_cache(src, B, u)
        v = versions.place(plan.U, B, note=f'serve --tier sublst {u.name}')
        state['version'], state['carry_src'], state['design_b'], state['marks'] = v, src, design_b, marks
        u.meta['version'] = v.vid
        u.note(f'build placed as version {v.vid} of {plan.U} ({v.run}; unsealed, not served): '
               f'{len(os.listdir(os.path.join(v.run, "candidates", "I")))} I pages, '
               f'{len(os.listdir(os.path.join(v.run, "candidates", "V")))} V pages')
        u.add_output(v.path('BUILDING.json'), 'run_version_placed', placed_from=B)
    return u


def seal_required(plan, state, root):
    """The paths the version must hold before it is sealed: the queue and the catalog view of every page to be registered
    (not refused by an earlier stage)."""
    out = []
    for pg in plan.pages:
        if pg.key in state['refused']:
            continue
        out.extend(x for x in (pg.at('queue', root), pg.at('catalog', root)) if x)
    return out


def index_check(R):
    """(ok, detail): R's page index passes every detections_index content test (flow._index_current)."""
    why = flow._index_current(R)
    return not why, why or 'current'


def switch_stage(run, plan, state):
    """THE SWITCH (unit sublst_<run>_switch): the placed version V's page index (flow.build_page_index), V sealed
    (versions.seal: the pages' queues and views present, the index current, the manifest) and U switched to V
    (versions.switch with the seal's record: its walk is the switch's; one atomic rename of the pointer; the first switch
    adopts the legacy directory U as a version), then the journal lines recorded on the old pages between the carry and the
    switch carried (design B) or appended from the version U named before. The switch and that catch-up run under the run's
    JOURNAL LOCKS (versions.journal_locks), the locks the two-port server's POSTs to these journals take: a label sent on an
    old page is written before the switch (into the old version, then caught up here) or after the catch-up by a page of V
    (two_port Store.get current) - none is left behind in the old version. An adopted legacy run then gets its manifest
    (versions.seal_adopted). A refusal before the switch - the seal's, or the switch's after a seal - moves V back to B
    (versions.unplace; a sealed, never served V loses its record first): U unchanged; that refusal is the one raised (a
    failure of the move back is added to its text, never in its place). The catalogues and member maps of a design-B
    catch-up are loaded before the locks: under them run only the switch, the read of the old journals' tails and the
    append. A catch-up that fails after the switch is OWED (versions.owe_catchup: the old version and the marks): every later
    build or switch of U refuses and the old version is retained until catchup_stage completes it. Returns the unit."""
    S = _CFG().serve
    v = state.get('version')
    B = build_dir(plan)
    with run.unit(f'sublst_{plan.run_name}_switch', stage='swap', tier='sublst', version=getattr(v, 'vid', None),
                  parity=plan.parity) as u:
        if v is None:
            u.refuse('no placed version (the carry+swap unit did not place the build)')
        switched = False
        src, design_b, marks = state.get('carry_src'), state.get('design_b'), state.get('marks') or {}
        try:
            flow.build_page_index(u, v.run)                # no publish marker on a version nobody reads yet
            rec = versions.seal(v, required=seal_required(plan, state, v.run), checks=[('page_index_current', index_check)],
                                note=f'serve --tier sublst {u.name}')
            u.note(f'version {v.vid} sealed: {rec["n_files"]} files, {rec["n_bytes"]} bytes (manifest walk {rec["walk_s"]} s; '
                   f'{rec["n_hashed"]} files, {rec["hashed_bytes"]} bytes hashed in {rec["hash_s"]} s)')
            # the old run the catch-up reads: the parity source, the version U names, or the legacy directory it adopts
            cur = versions.served(plan.U)
            old = src if plan.parity else (cur.run if cur is not None else
                                           (os.path.realpath(plan.U) if os.path.isdir(plan.U) else None))
            maps = journals.sublst_carry_maps(old, v.run) if design_b and old else None      # before the locks
            with versions.journal_locks(plan.U, S.SUBLST_JOURNALS):
                sw = versions.switch(plan.U, v, note=f'serve --tier sublst {u.name}', record=rec)
                switched = True
                prev = sw['previous']
                between_src = src if plan.parity else (prev.run if prev else None)
                same = plan.parity or (prev is not None and (sw['adopted'] is not None if cur is None else prev.vid == cur.vid))
                if between_src:
                    try:
                        app = catch_up(between_src, plan.U, design_b, marks, maps if same else None)
                    except Exception as e:
                        raise Refusal(_owe(plan, v, prev, between_src, design_b, marks, e)) from e
                    u.note(f'journal lines recorded on the old servers between the '
                           f'{"carry" if design_b else "copy"} and the swap, {"carried" if design_b else "appended"}: {app}')
        except Exception as e:
            if not switched and v is not None and os.path.isdir(v.run):
                try:
                    versions.unplace(v, B, sealed=True)
                    u.note(f'not switched: version {v.vid} moved back to {B} - {plan.U} unchanged')
                except Exception as e2:
                    _add_failure(e, f' - AND version {v.vid} could not be moved back to {B} ({e2}): the next serve moves '
                                    f'it back (recover_placed) or it is retired')
            raise
        u.meta.update(version=v.vid, previous=prev.vid if prev else None, adopted=sw['adopted'].vid if sw['adopted'] else None)
        u.note(f'switched: {plan.U} -> version {v.vid} (previous {prev.vid + " " + prev.run if prev else "none"}'
               f'{"; the legacy run adopted as that version" if sw["adopted"] else ""}); {sw["verify"]}')
        u.add_output(v.path('VERSION.json'), 'run_version', pointer=plan.U)
        if sw['adopted'] is not None:                  # outside the locks: one walk of the old run; U is switched by now, so
            try:                                       # a refusal is recorded, not raised (the version stays 'adopted')
                arec = versions.seal_adopted(sw['adopted'], note=f'serve --tier sublst {u.name}')
            except (Refusal, OSError) as e:
                u.meta['adopted_seal_error'] = str(e)[:400]
                u.note(f'ADOPTED WITHOUT A MANIFEST: {sw["adopted"].vid} ({str(e)[:300]}) - kept by retention, not counted '
                       f'among the newest versions; versions.seal_adopted gives it one')
            else:
                u.note(f'the adopted legacy run {sw["adopted"].vid} given its manifest: {arec["n_files"]} files, '
                       f'{arec["n_bytes"]} bytes (walk {arec["walk_s"]} s; {arec["n_hashed"]} files, '
                       f'{arec["hashed_bytes"]} bytes hashed in {arec["hash_s"]} s) - it can be switched back to')
                u.add_output(sw['adopted'].path('VERSION.json'), 'run_version_adopted')
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


def catch_up(P, U, design_b, marks, maps=None, journal_dirs=None):
    """The journal lines written on the old run P after the seed carry/copy, into U's served version: design B,
    journals.sublst_carry_between from the marks (maps loaded before the locks); else journals.append_between (the exact
    lines U lacks; ``journal_dirs`` its journals, default serve.SUBLST_JOURNALS - the deep run's: versions.journal_dirs). Both
    are safe to repeat (an owed catch-up is retried in full). The caller holds the journal locks."""
    if design_b:
        return journals.sublst_carry_between(P, U, marks, maps=maps)
    return journals.append_between(P, U, journal_dirs)


def _owe(plan, v, prev, P, design_b, marks, e, journal_dirs=None):
    """Record the owed catch-up after a switch (versions.owe_catchup; ``journal_dirs`` the journals it covers, default
    serve.SUBLST_JOURNALS); the refusal text."""
    head = (f'SWITCHED {plan.U} to version {v.vid}, but the catch-up of the journal lines written on '
            f'{prev.vid if prev else P} failed ({type(e).__name__}: {str(e)[:300]})')
    try:
        p = versions.owe_catchup(plan.U, {'from_version': prev.vid if prev else None, 'from_run': P, 'to_version': v.vid,
                                          'design_b': bool(design_b), 'marks': marks, 'error': f'{type(e).__name__}: {e}'[:600],
                                          'journals': list(journal_dirs) if journal_dirs is not None else None,
                                          'note': f'serve --tier {plan.tier} {plan.run_name}'})
    except Exception as e2:
        return (f'{head} - AND the owed catch-up could not be recorded ({e2}): no build or switch of {plan.U} may run until '
                f'the lines of {P} are caught up by hand')
    return (f'{head} - OWED ({p}): every later build or switch of {plan.U} refuses, and {prev.vid if prev else P} is kept, '
            f'until the catch-up completes (the next serve --tier {plan.tier} retries it first)')


_add_failure = versions.add_failure      # a failure added to the refusal's text, never in its place


def catchup_stage(run, plan):
    """THE OWED CATCH-UP (unit sublst_<run>_catchup), before any build or switch: when U owes the catch-up of an old
    version's journal lines (versions.owed_catchup), it is run again (maps loaded first, then under the journal locks the
    tails read and appended) and its record removed (versions.catchup_done). None when nothing is owed; a refusal leaves the
    record (every build and switch keeps refusing). Returns the unit."""
    S = _CFG().serve
    rec = versions.owed_catchup(plan.U)
    if rec is None:
        return None
    J = rec.get('journals') or S.SUBLST_JOURNALS          # the deep run's: the journals its switch caught up
    with run.unit(f'{plan.tier}_{plan.run_name}_catchup', stage='catch-up', tier=plan.tier, parity=plan.parity) as u:
        cur = versions.served(plan.U)
        if cur is None or cur.vid != rec.get('to_version'):
            u.refuse(f'the owed catch-up is into version {rec.get("to_version")}, but {plan.U} names '
                     f'{cur.vid if cur else "no version"} - resolve {versions.OWED_FILE} by hand')
        P = rec.get('from_run')
        if not P or not os.path.isdir(P):
            u.refuse(f'the old run {P} of the owed catch-up is gone - its journal lines cannot be caught up')
        maps = journals.sublst_carry_maps(P, cur.run) if rec.get('design_b') else None
        with versions.journal_locks(plan.U, J):
            app = catch_up(P, plan.U, rec.get('design_b'), rec.get('marks') or {}, maps, journal_dirs=J)
            versions.catchup_done(plan.U, rec)
        u.meta.update(owed=rec, caught_up=app)
        u.note(f'the owed catch-up of {rec.get("from_version")} into {cur.vid} completed: {app}; record removed')
        for jd in J:
            j = os.path.join(plan.U, jd, 'judgements.jsonl')
            if os.path.exists(j):
                u.add_output(j, 'judgement_journal')
    return u


def retire_stage(run, plan, args):
    """RETENTION (unit sublst_<run>_retire): versions.retire on U - a dry run without --retire-plan (the plan written, its
    path in the record; nothing deleted), the planned deletions with it. Returns the unit."""
    plan_path = getattr(args, 'retire_plan', None)
    with run.unit(f'{plan.tier}_{plan.run_name}_retire', stage='retire', tier=plan.tier, parity=plan.parity,
                  apply=bool(plan_path)) as u:
        if not versions.all_versions(plan.U):
            raise NoInput(f'{plan.U} has no versions under {versions.root_of(plan.U)}')
        res = versions.retire(plan.U, plan=plan_path, note=f'serve --tier {plan.tier} {u.name}')
        if plan_path is None:
            u.meta.update(plan=res['plan'], retire=[r['version'] for r in res['retire']])
            u.note(f'DRY RUN: {len(res["retire"])} version(s) would be retired '
                   f'({sum(r["bytes"] for r in res["retire"])} bytes): {", ".join(r["version"] for r in res["retire"]) or "none"}; '
                   f'kept: {", ".join(k["version"] + " (" + k["why"] + ")" for k in res["keep"])}. Apply: --stages retire '
                   f'--retire-plan {res["plan"]}')
            u.add_output(res['plan'], 'retire_plan')
        else:
            u.meta.update(retired=res['retired'], bytes=res['bytes'])
            u.note(f'retired {len(res["retired"])} version(s), {res["bytes"]} bytes: {", ".join(res["retired"]) or "none"}')
            u.add_output(plan_path, 'retire_plan', applied=True)
    return u


def page_context_stage(run, plan, root=None):
    """PAGE CONTEXT (sublst_cumulative.sh:295-300; page_context_tenmin.py, 2026-10-23; Gregg 2026-10-22: "Go ahead with both
    fixes, keep it as page text"): the 10-minute panels' run context - global noise V / I, chain survivors, beam epoch/master,
    windows searched, each with its percentile - written as the <png>.page.json sidecars beside the SERVED panels of the
    placed version (``root``; default the served run U), before it is sealed and switched to, so the pages never read it
    without them (the review server prints them under each panel). Page text, not a gate:
    a failure is recorded and the serve continues, as production (whose exit status it does not change; here a missing or
    malformed sidecar refuses this unit, so the serve exits 2 after registering the pages). Returns the unit."""
    S = _CFG().serve
    R = root or plan.U
    log = os.path.join(R, 'page_context_tenmin.log')
    with run.unit(f'sublst_{plan.run_name}_page_context', stage='page-context', tier='sublst', parity=plan.parity) as u:
        res = scripts.run('page_context_tenmin', [R, '--workers', str(S.SUBLST_PAGE_CONTEXT_WORKERS)], log=log, unit=u,
                          env={'NUMPY_MADVISE_HUGEPAGE': '0'})
        line = scripts.last_line(res.tail)
        u.meta['page_context'] = {'ok': bool(res.ok), 'line': line[:400], 'log': log}
        if res.ok:
            u.note(f'page context: {line[:200]}')
        else:
            u.note(f'PAGE-CONTEXT FAILED (see {log}) - the serve continues without the page text: '
                   f'{(res.reason or line)[:300]}')
        u.add_output(R, 'tenmin_page_context')          # its content tests judge the sidecars either way (refused if missing)
    return u


def recover_placed(plan):
    """A swap killed between its place and its switch left the build as a version whose placing process is gone and that was
    never served (unsealed, or sealed and never switched to): move it back to B (versions.unplace, sealed=True), so the next
    carry+swap finds it where production would. Under the build lock."""
    B = build_dir(plan)
    if os.path.lexists(B):
        return None
    v = versions.placed_abandoned(plan.U, B)
    if v is None:
        return None
    versions.unplace(v, B, sealed=True)
    print(f'serve: version {v.vid} of {plan.U} was placed from {B} by a swap that did not finish - moved back to {B}',
          file=sys.stderr)
    return v


COALESCED_STAGES = ('build', 'start')     # an invocation with either goes through the run's coalescing slot


def _fresh_plan(args, run_):
    """The plan made again (a follow-up build pins its inputs at its own start: the LST runs placed by then)."""
    from exopipe.phase3.lib.serve import plan as planmod
    plan = planmod.make(args, run_.out_root)
    if plan.gates:
        raise Refusal('; '.join(f'{g}: {m}' for g, m in plan.gates))
    return plan


def run(args, run, plan, stages):
    """THE COALESCING BUILDER around build_and_serve (module text). Without build and start (page-index, retire, hub, verify
    alone) the stages run directly."""
    if not set(COALESCED_STAGES) & set(stages):
        return build_and_serve(args, run, plan, stages)
    return coalesced(args, run, plan, stages, build_and_serve)


def coalesced(args, run, plan, stages, body):
    """``body(args, run, plan, stages)`` through the coalescing slot of plan.U (lib/serve/coalesce.py): run now, or - when
    another builder of U runs - left to its follow-up (unit <tier>_<run>_coalesced, no-input: exit 3); each follow-up makes
    the plan again (_fresh_plan). The deep tier's versioned publish takes the same slot (lib/serve/deep.py)."""
    from exopipe.phase3.lib.serve import coalesce
    versions._guard(plan.U, 'the coalescing slot of')
    slot = coalesce.slot_for(plan.U)
    plans, failed = [plan], []

    def build():
        try:
            body(args, run, plans.pop() if plans else _fresh_plan(args, run), stages)
        except Exception as e:                         # re-raised below once the slot is released (drain records it)
            failed.append(e)
            raise
        return 0
    res = slot.submit(build, note=f'serve --tier {plan.tier} {run.run_id}')
    builds = res.get('builds') or []
    if len(builds) > 1:
        print(f'serve: coalescing builder of {plan.U}: {len(builds)} builds (this invocation\'s and {len(builds) - 1} '
              f'follow-up(s) requested while it built); slot {slot.d}', flush=True)
    if failed:
        raise failed[0]
    if res['status'] in ('coalesced', 'covered'):
        with run.unit(f'{plan.tier}_{plan.run_name}_coalesced', stage='coalesce', tier=plan.tier, slot=slot.d) as u:
            b = res.get('builder') or {}
            u.meta['coalesce'] = {'status': res['status'], 'builder': b}
            raise NoInput(f'COALESCED ({res["status"]}): ' + (
                f'a build of {plan.U} runs (pid {b.get("pid")} on {b.get("host")}, since {b.get("utc")}); it builds once more '
                f'after its build ends, beginning after this request (slot {slot.d})' if res['status'] == 'coalesced' else
                f'another builder\'s build of {plan.U}, begun after this request, served it (slot {slot.d})'))
    return None


def build_and_serve(args, run, plan, stages):
    S = _CFG().serve
    state = flow.new_state()
    together = {'carry', 'swap', 'start'}
    if (together & set(stages)) and not together <= set(stages):
        raise Refusal('sub-LST: carry, swap and start run only together (the journal lines written between the copy and '
                      'the swap are appended at the swap, sublst_cumulative.sh:106-120)')
    with build_lock(plan):
        recover_placed(plan)
        if set(stages) & set(COALESCED_STAGES):        # an owed journal catch-up first: no build or switch before it
            if flow.halts(catchup_stage(run, plan), 'catch-up', need_done=True):
                return
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
            try:                                       # page text: never halts the serve (sublst_cumulative.sh:295-300)
                page_context_stage(run, plan, root=state['version'].run)
            except Exception as e:                     # the unit is recorded 'error' (the serve exits 1 at the end); the
                print(f'serve: PAGE-CONTEXT FAILED ({type(e).__name__}: {str(e)[:300]}) - the serve continues without the '
                      f'page text', file=sys.stderr)   # pages are still registered, as production's `|| say` does
            if flow.halts(switch_stage(run, plan, state), 'switch', need_done=True):
                return
            flow.start_stage(run, plan, state)        # U's page index: the version's, built before the switch (kept)
        elif 'page-index' in stages:
            if flow.halts(flow.page_index_stage(run, plan, 'sublst'), 'page-index'):
                return
        if 'retire' in stages:
            if flow.halts(retire_stage(run, plan, args), 'retire'):
                return
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
    for line in versions.describe(plan.U):
        print(line)
    if 'start' in stages:
        print(f'  the swap places the build as a new version under {versions.root_of(plan.U)}; then on that version: page '
              f'context {scripts.SCRIPTS["page_context_tenmin"][0]} --workers {_CFG().serve.SUBLST_PAGE_CONTEXT_WORKERS} (page '
              f'text; a failure does not stop the serve), page index build_detections_index.py (a failure refuses the switch), '
              f'the seal; then {plan.U} switched to it (one atomic rename) and the pages registered')
    if 'retire' in stages:
        print(f'  retire: ' + ('the planned versions deleted (--retire-plan ' + args.retire_plan + ')'
                               if getattr(args, 'retire_plan', None) else 'a DRY RUN (the plan written, nothing deleted)'))
    if os.path.isdir(plan.U or '') and ('start' in stages or 'page-index' in stages):
        print(flow.page_index_describe(plan.U))
