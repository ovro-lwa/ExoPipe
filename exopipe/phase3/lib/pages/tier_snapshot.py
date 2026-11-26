"""Snapshot tier of `exopipe3 pages` (production: steps 6-7 of ops/snapshot_rechain_batch.sh / snapshot_rechain_b2_20261007.sh).

Unit of work: one snapshot chain run (--run Run_SNAP..., required); units <run>:stagein (only with --from-run: a copy of the
chain run as it was before its pages), <run>:build, <run>:filtercat. RUN = <out>/<run> (production
/fast/gh/main/phase3/snapshot_rechain_<run>/<run> on lwacalim05; serve copies it to the review host). Per-hour use: the night
cells of a batch are pages of that batch's chain run - --epochs is accepted only with --run (the run holding those cells).

Stages (core.legacy.run, site PYTHON['py36'] = the production development Python):
  build      gate P23: RUN/served_tiers.csv (run_samples.SERVED_TIERS, the per-tier serving table the catalog snapshot chain's
             stage tierserve writes) present and passing its content test (class served_tiers, lib/catalog/tests_snapshot.py),
             else REFUSED before anything is written - pages are never built without the table (production passes it to
             every pages pass, snapshot_rechain_batch.sh:607-613; rule after Gregg 2026-10-17 "four independent tiers",
             R-GREGG-20261017-SNAPSHOT-MEANS-10S-ONLY-FOUR-TIERS: a source is served in a tier only on that tier's own
             detections). Then clv15/phase3_snapshot_pages.py --run RUN --proto-root <CFG.paths.SNAPSHOT_FLEET_OUT>
             --masters-root <pages_snapshot.MASTERS_ROOT> --fit-store <store> --nproc N [--resume] --served-tiers
             RUN/served_tiers.csv with PYTHONPATH=<ops root> and SNAP_SCALE_MODE (:607-612, :632); then production's per-tier
             hook `pertier_hook full` (:639; pt_fail :565-569, pertier_hook :570-587), in place and in its order:
             ops/apply_served_check.py backup RUN W/pertier/pre_full,
             the views-only rewrite (the builder again with --nproc pages_snapshot.VIEWS_NPROC --served-tiers ... --views-only),
             ops/pertier_views_content_test.py RUN RUN/served_tiers.csv W/pertier/pre_full requiring PERTIER-VIEWS-OK (P24: on
             a failure the views are restored from the backup, apply_served_check.py restore, and the stage refuses; P24 also,
             before the pass, when W/pertier or its pre_full is a symbolic link - never removed, never followed), then the
             Card 2 back-fill clv15/phase3_snapshot_card2.py --run RUN (P25; card2_vleak.csv, card2_vleak_columns.csv,
             card2_filter.csv: an annotation and a page filter off by default, removing nothing); then
             ops/snapshot_rechain_content_test.py RUN --pages (:640): gates P18 (flow mode), P19 (duplicate panel), P20
             (missing panel), P21 (run content test R1-R7)
  filtercat  ops/snapshot_filter_catalog.py RUN <store> (:641)
"""
import glob
import os
import shutil

from exopipe.phase3.lib.pages import common, mirror

TIER = 'snapshot'
PERTIER = 'pertier'            # W/pertier: the per-tier hook's backup (pre_full) and logs (snapshot_rechain_batch.sh:556)
PERTIER_TAG = 'full'           # the one-pass run's hook tag (:639 pertier_hook full; V first is off in production, :621)
PERTIER_OK = 'PERTIER-VIEWS-OK'   # the views test's verdict line (:578-580 grep -q '^PERTIER-VIEWS-OK')
# the Card 2 back-fill's files in RUN (clv15 phase3_snapshot_card2.py backfill; the first is catalog_omission.SNAPSHOT_CARD2,
# the file the combined view and the deep omission read)
CARD2_FILES = ('card2_vleak.csv', 'card2_vleak_columns.csv', 'card2_filter.csv')


def add_arguments(p):
    g = p.add_argument_group('snapshot tier (--tier snapshot)')
    g.add_argument('--snap-fit-store', metavar='CSV', help='fit store of the chain run (default pages_snapshot.FIT_STORE)')
    g.add_argument('--snap-keep-panels', action='store_true',
                   help='keep panel files already on disk (phase3_snapshot_pages.py --resume; an operator decision after an '
                        'interrupted pass; default pages_snapshot.KEEP_PANELS)')


class Ctx:
    pass


def context(args, run):
    from exopipe.phase3.config import CFG
    S = CFG.pages_snapshot
    name = getattr(args, 'run_label', None)
    if not name:
        raise common.Refusal('P1: --run Run_SNAP... (the chain run) is required for the snapshot tier')
    if not name.startswith('Run_SNAP'):
        raise common.Refusal(f'--run {name!r}: a snapshot chain run is named Run_SNAP* (snapshot_rechain_batch.sh:49)')
    c = Ctx()
    c.name, c.out = name, run.out_root
    c.RUN = os.path.join(run.out_root, CFG.pages.RUN_DIR['snapshot'].format(run=name))
    c.W = run.out_root
    c.roots = common.legacy_roots(args.legacy_root)
    c.stages = common.select_stages(args, TIER)
    c.store = args.snap_fit_store or S.FIT_STORE.format(ops=CFG.paths.OPS_ROOT, run=name)
    c.keep = bool(args.snap_keep_panels) or S.KEEP_PANELS
    c.nproc = args.shards or CFG.pages.SNAP_PAGES_NPROC
    c.from_run = args.from_run
    c.epochs = common.epochs_of(args)
    if c.from_run and 'stagein' not in c.stages:
        c.stages = ['stagein'] + c.stages
    return c


def env_for(c):
    S = common.CFG().pages_snapshot
    env = {'PYTHONPATH': common.need_root(c.roots, 'ops'), 'SNAP_SCALE_MODE': S.SCALE_MODE}
    if not S.FLOW_FRAMES:
        env['P3_SNAP_FIT_RAW'] = '1'           # a historical non-flow run only (ruled key)
    return env


def served_tiers_gate(c, u):
    """P23: RUN/served_tiers.csv (run_samples.SERVED_TIERS) present and passing its content test (class served_tiers: production's
    columns, the per-row and per-source rules, and the rows equal to the detection_record.csv.gz beside it by tierserve's own
    definition; ExoPipe PR 185), checked BEFORE anything is written. Production passes the table to every pages pass of a chain
    run (snapshot_rechain_batch.sh:607-613, computed first by pertier_prepare :557-564), so a run without a valid table is
    refused here, never built unfiltered. A failed test refuses unless --waive covers it, as at every gate (u.failed_tests()).
    Returns the table's path (an input of the unit; its test results recorded on it, a waived one with its ruling)."""
    from exopipe.phase3.core import validate
    from exopipe.phase3.lib.catalog import run_samples as RS
    p = os.path.join(c.RUN, RS.SERVED_TIERS)
    if not os.path.isfile(p):
        raise common.Refusal(f'P23: {c.RUN} has no {RS.SERVED_TIERS} (the per-tier serving table; the catalog snapshot chain\'s '
                             f'stage tierserve writes it): the pages of a chain run are built only with it, never unfiltered - '
                             f'nothing built')
    tests = validate.run('served_tiers', p)
    for t in tests:
        u.check(t['name'], t['passed'], t['value'], t['detail'], cls='served_tiers', path=p)
    failed = [t for t in u.failed_tests() if t.get('cls') == 'served_tiers']      # not passed and not waived
    if failed:
        raise common.Refusal('P23: %s fails its content test (%s) - nothing built' % (
            p, '; '.join(('%s: %s' % (t['name'], t['detail']))[:600] for t in failed)))
    waived = [t for t in u.tests if t.get('cls') == 'served_tiers' and t.get('waived')]
    if waived:
        u.note('P23: %s - failed test(s) waived: %s' % (p, '; '.join('%s (%s)' % (t['name'], t['waived']) for t in waived)))
    u.add_input(p)
    return p


def build_argv(c, st, views_only=False):
    """phase3_snapshot_pages.py's arguments: the pages pass (snapshot_rechain_batch.sh:609-610: --nproc N [--resume]
    --served-tiers RUN/served_tiers.csv) or the hook's views-only rewrite (:575-576: --nproc 2 --served-tiers ...
    --views-only)."""
    from exopipe.phase3.config import CFG
    S = CFG.pages_snapshot
    head = ['--run', c.RUN, '--proto-root', CFG.paths.SNAPSHOT_FLEET_OUT, '--masters-root', S.MASTERS_ROOT,
            '--fit-store', c.store]
    if views_only:
        return head + ['--nproc', S.VIEWS_NPROC, '--served-tiers', st, '--views-only']
    return head + ['--nproc', c.nproc] + (['--resume'] if c.keep else []) + ['--served-tiers', st]


def pertier_scripts(c):
    """The in-place scripts of production's per-tier hook (P1-style refusal when one is missing, before the pages pass)."""
    return {'check': common.legacy_script(c.roots, 'ops', 'apply_served_check.py'),
            'views_test': common.legacy_script(c.roots, 'ops', 'pertier_views_content_test.py'),
            'card2': common.legacy_script(c.roots, 'clv15', 'phase3_snapshot_card2.py')}


def card2_argv(c):
    """phase3_snapshot_card2.py's arguments. Production passes --run RUN alone (:582), the module's defaults being the
    production roots and Card 2's ratio 0.05; the step passes the same values from config (the Masters the panels read, the
    snapshot fleet outputs, the chain's light-curve and Master-catalog roots - its frame/Master scale is the chain's
    parent_ratio_test scale - and the ratio the combined view applies, catalog_omission.CARD2_RATIO)."""
    from exopipe.phase3.config import CFG
    S, K = CFG.pages_snapshot, CFG.catalog_snapshot
    return ['--run', c.RUN, '--ratio', '%r' % float(CFG.catalog_omission.CARD2_RATIO), '--masters-root', S.MASTERS_ROOT,
            '--lightcurves-root', K.LIGHTCURVES_ROOT, '--master-catalogs', K.MASTER_CATALOGS,
            '--proto-root', CFG.paths.SNAPSHOT_FLEET_OUT]


def own_output(log):
    """This call's own lines of a log core.legacy.run appends to (after its last '===== ' header line): production's
    `> $PT/<name>.log` gives each call its own file."""
    with open(log, errors='replace') as f:
        lines = f.read().splitlines()
    hdr = [i for i, ln in enumerate(lines) if ln.startswith('===== ')]
    return lines[hdr[-1] + 1:] if hdr else lines


def pertier_backup_dir(c):
    """W/pertier/pre_full, the hook's backup of the views (production `rm -rf $PT/pre_$tag` then the backup, :573-574). P24 when
    W/pertier or pre_full is a symbolic link, or pre_full is not a directory: never removed, never followed (the removal
    would act on what the link points at); checked before the pages pass and again before the removal."""
    PT = os.path.join(c.W, PERTIER)
    pre = os.path.join(PT, f'pre_{PERTIER_TAG}')
    for p in (PT, pre):
        if os.path.islink(p):
            raise common.Refusal(f'P24 per-tier {PERTIER_TAG}: {p} is a symbolic link (to {os.readlink(p)}) - not removed, '
                                 f'not followed; remove the link by hand')
    if os.path.lexists(pre) and not os.path.isdir(pre):
        raise common.Refusal(f'P24 per-tier {PERTIER_TAG}: {pre} is not a directory - not removed; remove it by hand')
    return pre


def pertier_hook(c, u, script, hook, st):
    """Production's per-tier hook after the pages pass (snapshot_rechain_batch.sh:570-587 `pertier_hook full`; pt_fail
    :565-569), in place and in its order:
      1 rm -rf W/pertier/pre_full (P24 when it or W/pertier is a symbolic link: pertier_backup_dir); apply_served_check.py
        backup RUN W/pertier/pre_full (the views: queue_<S>[_<tier>].txt and the source_catalogs_<S>[_<tier>]/ catalogs)
      2 phase3_snapshot_pages.py ... --served-tiers RUN/served_tiers.csv --views-only (catalogs, detections.csv and queues
        rebuilt, nothing rendered; refuses when a kept panel is not on disk)
      3 pertier_views_content_test.py RUN RUN/served_tiers.csv W/pertier/pre_full: exit 0 and a PERTIER-VIEWS-OK line (T1 no
        identity on a tier page that does not qualify there, T2 every qualifying one listed, T3 kept rows' panels on disk,
        T4 union queues = the tier queues' union, T5 the rewrite = the backup minus exactly the non-qualifying identities)
      P24 on a failure of 2 or 3: apply_served_check.py restore W/pertier/pre_full RUN (the pages pass's own views back), refuse
      4 phase3_snapshot_card2.py (card2_argv): card2_vleak.csv, card2_vleak_columns.csv, card2_filter.csv (P25 on a failure).
    Logs W/pertier/backup_full.log (+ the restore), views_full.log, test_full.log, card2.log; their summary lines (production's
    `say "step pertier-full ..."`) in the unit's meta and notes."""
    from exopipe.phase3.config import CFG
    S = CFG.pages_snapshot
    PT = os.path.join(c.W, PERTIER)
    pre = pertier_backup_dir(c)
    blog, vlog = os.path.join(PT, f'backup_{PERTIER_TAG}.log'), os.path.join(PT, f'views_{PERTIER_TAG}.log')
    tlog, clog = os.path.join(PT, f'test_{PERTIER_TAG}.log'), os.path.join(PT, 'card2.log')
    os.makedirs(PT, exist_ok=True)
    if os.path.lexists(pre):
        shutil.rmtree(pre)
    res = common.lrun(u, hook['check'], ['backup', c.RUN, pre], python=S.PYTHON_KEY, cwd=c.W, log=blog)
    if not res.ok:
        raise common.Refusal(f'P24 per-tier {PERTIER_TAG}: backup of the views ({res.reason}; {blog})')

    def fail(why):
        r = common.lrun(u, hook['check'], ['restore', pre, c.RUN], python=S.PYTHON_KEY, cwd=c.W, log=blog)
        raise common.Refusal(f'P24 per-tier {PERTIER_TAG}: {why} - ' + (
            f'views restored from {pre}' if r.ok else f'AND the restore from {pre} failed ({r.reason}; {blog})'))

    res = common.lrun(u, script, build_argv(c, st, views_only=True), python=S.PYTHON_KEY, env=env_for(c), cwd=c.W, log=vlog)
    if not res.ok:
        tail = [ln for ln in own_output(vlog) if 'WARNING' not in ln]
        fail('views-only rewrite (%s)' % (tail[-1] if tail else res.reason)[:160])
    res = common.lrun(u, hook['views_test'], [c.RUN, st, pre], python=S.PYTHON_KEY, cwd=c.W, log=tlog)
    tl = own_output(tlog)
    if not (res.ok and any(ln.startswith(PERTIER_OK) for ln in tl)):
        fail('content test (%s)' % (tl[-1] if tl else res.reason)[:200])
    res = common.lrun(u, hook['card2'], card2_argv(c), python=S.PYTHON_KEY, cwd=c.W, log=clog)
    cl = own_output(clog)
    if not res.ok:
        tail = [ln for ln in cl if 'warn' not in ln.lower()]
        raise common.Refusal('P25 per-tier: Card 2 back-fill (%s)' % (tail[-1] if tail else res.reason)[:160])
    views = [ln for ln in tl if ln[:2] in ('V ', 'I ')]
    card2 = [ln for ln in cl if ln.startswith('CARD2')]
    u.meta['pertier'] = dict(views=views, verdict=PERTIER_OK, card2=card2, backup=pre, logs=[blog, vlog, tlog, clog])
    u.note(f'per-tier {PERTIER_TAG}: ' + '; '.join(views) + f'; {PERTIER_OK}; ' + '; '.join(card2)[:300])


def stage_stagein(c, run, done):
    name = f'{c.name}:stagein'
    src = os.path.abspath(c.from_run)
    with run.unit(name, tier=TIER, stage='stagein', source=src) as u:
        if not os.path.isfile(os.path.join(src, 'chain_summary.json')):
            raise common.Refusal(f'--from-run {src}: not a snapshot chain run (no chain_summary.json)')
        if os.path.lexists(c.RUN):
            if not run.fresh:
                raise common.Refusal(f'{c.RUN} exists; pass --fresh to replace it with the copy of {src}')
            shutil.rmtree(c.RUN)
        shutil.copytree(src, c.RUN, symlinks=True)
        u.add_input(os.path.join(src, 'chain_summary.json'))
        u.add_output(c.RUN, 'pages_run_copy', source=src)
        done[name] = 'done'


def stage_build(c, run, done):
    from exopipe.phase3.config import CFG
    S = CFG.pages_snapshot
    name = f'{c.name}:build'
    with run.unit(name, tier=TIER, stage='build', run=c.name) as u:
        u.meta['test_ctx'] = {'roots': c.roots, 'tier': TIER}
        if c.from_run:
            common.prerequisite(run, done, f'{c.name}:stagein')
        for f in ('chain_summary.json', 'queue.txt'):
            if not os.path.isfile(os.path.join(c.RUN, f)):
                raise common.Refusal(f'{c.RUN} is not a finished chain run (no {f}; the catalog snapshot chain writes it)')
        if not os.path.isfile(c.store):
            raise common.Refusal(f'fit store {c.store} not found (--snap-fit-store)')
        st = served_tiers_gate(c, u)            # P23, before anything is written
        mirror.gate(u, c.roots, TIER, 'build')
        for p in [os.path.join(c.RUN, 'chain_summary.json'), os.path.join(c.RUN, 'queue.txt'), c.store]:
            u.add_input(p)
        tpl = c.store[:-4] + '_template.csv'
        if os.path.exists(tpl):
            u.add_input(tpl)
        for p in sorted(glob.glob(os.path.join(c.RUN, 'source_catalogs', '*_sources_survivors.csv'))):
            u.add_input(p)
        # the record (the Card 2 back-fill's rows and the served_tiers test's reference) and the back-fill's roots (the
        # Masters, the chain's light curves and Master catalogs, card2_argv): a root is fingerprinted as a directory (size +
        # modification time: an entry added, removed or renamed at its top level), its files are not walked
        K = CFG.catalog_snapshot
        for p in (os.path.join(c.RUN, 'detection_record.csv.gz'), S.MASTERS_ROOT, K.LIGHTCURVES_ROOT, K.MASTER_CATALOGS):
            u.add_input(p)
        script = common.legacy_script(c.roots, 'clv15', 'phase3_snapshot_pages.py')
        hook = pertier_scripts(c)               # every in-place script of the hook found before the pass starts
        pertier_backup_dir(c)                   # P24 on a linked backup directory, before the pass
        res = common.lrun(u, script, build_argv(c, st), python=S.PYTHON_KEY, env=env_for(c), cwd=c.W,
                          log=os.path.join(c.W, 'pages.log'))
        if not res.ok:
            raise common.Refusal(f'P18-P20 snapshot pages refused: {res.reason or res.tail[-300:]}')
        pertier_hook(c, u, script, hook, st)    # P24 (views rewritten and tested; restored on failure), P25 (Card 2)
        ct = common.legacy_script(c.roots, 'ops', 'snapshot_rechain_content_test.py')
        res2 = common.lrun(u, ct, [c.RUN, '--pages'], python=S.PYTHON_KEY, cwd=c.W, log=os.path.join(c.W, 'pages_content_test.log'))
        u.meta['run_content_test'] = res2.tail.splitlines()[-1] if res2.tail else ''
        if not res2.ok:
            raise common.Refusal(f'P21 run content test after the pages failed: {u.meta["run_content_test"]}')
        u.add_output(c.RUN, 'snapshot_page_run')
        for f in CARD2_FILES:                   # the Card 2 back-fill's three files, each content-tested (class snapshot_card2)
            u.add_output(os.path.join(c.RUN, f), 'snapshot_card2')
        for q in sorted(glob.glob(os.path.join(c.RUN, 'queue_[IV].txt')) + glob.glob(os.path.join(c.RUN, 'queue_[IV]_*.txt'))):
            u.add_output(q, 'review_queue')
        png_re = common.server_png_re(c.roots)
        for p in sorted(glob.glob(os.path.join(c.RUN, 'candidates', '*', '*', '*.png'))):
            if png_re.match(os.path.basename(p)):          # panels only (astrometry.png is serve's localize product)
                u.add_output(p, 'review_page', tier=TIER, roots=c.roots)
        done[name] = 'done'


def stage_filtercat(c, run, done):
    S = common.CFG().pages_snapshot
    name = f'{c.name}:filtercat'
    with run.unit(name, tier=TIER, stage='filtercat', run=c.name) as u:
        common.prerequisite(run, done, f'{c.name}:build')
        mirror.gate(u, c.roots, TIER, 'filtercat')
        u.add_input(c.store)
        script = common.legacy_script(c.roots, 'ops', 'snapshot_filter_catalog.py')
        from exopipe.phase3.lib.serve.snapshot_view import filtercat_procs
        try:
            procs = filtercat_procs(S.FILTERCAT_PROCS)
        except ValueError as e:
            raise common.Refusal(f'P22 {e}')
        # SNAPCAT_PROCS always passed (exopipe_ops 1b9475c; no inherited value reaches the child): the Lustre rule's reader count
        res = common.lrun(u, script, [c.RUN, c.store], python=S.PYTHON_KEY, env={'SNAPCAT_PROCS': str(procs)}, cwd=c.W,
                          log=os.path.join(c.W, 'filtercat.log'))
        if not res.ok:
            raise common.Refusal(f'P22 filter catalogue failed: {res.reason}')
        u.add_output(os.path.join(c.RUN, 'master_catalog.csv'), 'filter_catalog')
        done[name] = 'done'


def run(args, run_):
    c = context(args, run_)
    if args.dry_run:
        return dry_run(c)
    C = common.CFG().pages
    done = {}
    os.makedirs(c.W, exist_ok=True)
    with common.Lease(run_.out_root, c.RUN, C.LEASE_WAIT_S):
        for st in c.stages:
            {'stagein': stage_stagein, 'build': stage_build, 'filtercat': stage_filtercat}[st](c, run_, done)
            if any(u.status in ('refused', 'error') for u in run_.units):
                break
    return None


def dry_run(c):
    from exopipe.phase3.config import CFG
    S = CFG.pages_snapshot
    refuse = []
    py = common.interpreter_or_none(S.PYTHON_KEY)
    print(f'exopipe3 pages --tier snapshot --run {c.name}: DRY RUN (nothing written); stages {",".join(c.stages)}')
    print(f'  chain run {c.RUN}' + ('' if os.path.isdir(c.RUN) else '  (absent' + (': --from-run copies it)' if c.from_run else ')')))
    print(f"  interpreter PYTHON['{S.PYTHON_KEY}'] = {py or 'NOT SET on this host'}")
    if not py:
        refuse.append(f"no site PYTHON['{S.PYTHON_KEY}']")
    print(f'  fit store {c.store}' + ('' if os.path.isfile(c.store) else '  (MISSING)'))
    if not os.path.isfile(c.store):
        refuse.append(f'fit store {c.store} missing')
    if not c.from_run and not os.path.isfile(os.path.join(c.RUN, 'chain_summary.json')):
        refuse.append(f'{c.RUN} is not a chain run')
    for st in c.stages:
        for l in mirror.plan_lines(c.roots, TIER, st):
            print(l)
            if 'WOULD REFUSE' in l:
                refuse.append(f'{st}: P3 mirror drift')
    if 'build' in c.stages:
        from exopipe.phase3.lib.catalog import run_samples as RS
        st = os.path.join(c.RUN, RS.SERVED_TIERS)
        src = os.path.join(os.path.abspath(c.from_run), RS.SERVED_TIERS) if c.from_run else st   # stagein copies the source run
        has = os.path.isfile(src)
        print(f'  build: P23 {st} ' + (f'from {src} (stagein) ' if c.from_run else '') +
              ('present (its content test, class served_tiers, runs before the pass)' if has else 'MISSING'))
        if not has:
            refuse.append(f'P23 no {RS.SERVED_TIERS} in {os.path.dirname(src)}: the pages are never built without it'
                          + (' (nothing is copied)' if c.from_run else ''))
        pt = os.path.join(c.W, PERTIER)
        print(f'  build: phase3_snapshot_pages.py {" ".join(map(str, build_argv(c, st)))}; env {env_for(c)}')
        print(f'  build: per-tier hook ({PERTIER_TAG}): apply_served_check.py backup {c.RUN} {pt}/pre_{PERTIER_TAG}; '
              f'phase3_snapshot_pages.py {" ".join(map(str, build_argv(c, st, views_only=True)))}; '
              f'pertier_views_content_test.py {c.RUN} {st} {pt}/pre_{PERTIER_TAG} (requires {PERTIER_OK}; restore on a '
              f'failure); phase3_snapshot_card2.py {" ".join(card2_argv(c))} -> {", ".join(CARD2_FILES)} [snapshot_card2]')
        print(f'  build: then snapshot_rechain_content_test.py {c.RUN} --pages')
        from exopipe.phase3.core.record import Refusal      # the class (common.Refusal builds an instance)
        for chk in (pertier_scripts, pertier_backup_dir):
            try:
                chk(c)
            except Refusal as e:
                refuse.append(f'build: {e}')
    if 'filtercat' in c.stages:
        print(f'  filtercat: snapshot_filter_catalog.py {c.RUN} {c.store} (SNAPCAT_PROCS={common.CFG().pages_snapshot.FILTERCAT_PROCS}) '
              f'-> master_catalog.csv, coadd_root.txt [filter_catalog]')
    for r in refuse:
        print(f'  WOULD REFUSE: {r}')
    return 2 if refuse else 0
