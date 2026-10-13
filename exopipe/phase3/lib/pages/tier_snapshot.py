"""Snapshot tier of `exopipe3 pages` (production: steps 6-7 of ops/snapshot_rechain_batch.sh / snapshot_rechain_b2_20261007.sh).

Unit of work: one snapshot chain run (--run Run_SNAP..., required); units <run>:stagein (only with --from-run: a copy of the
chain run as it was before its pages), <run>:build, <run>:filtercat. RUN = <out>/<run> (production
/fast/gh/main/phase3/snapshot_rechain_<run>/<run> on lwacalim05; serve copies it to the review host). Per-hour use: the night
cells of a batch are pages of that batch's chain run - --epochs is accepted only with --run (the run holding those cells).

Stages (core.legacy.run, site PYTHON['py36'] = the production development Python):
  build      clv15/phase3_snapshot_pages.py --run RUN --proto-root <CFG.paths.SNAPSHOT_FLEET_OUT> --masters-root
             <pages_snapshot.MASTERS_ROOT> --fit-store <store> --nproc N [--resume] with PYTHONPATH=<ops root> and
             SNAP_SCALE_MODE (snapshot_rechain_batch.sh:362-363), then ops/snapshot_rechain_content_test.py RUN --pages (:365):
             gates P18 (flow mode), P19 (duplicate panel), P20 (missing panel), P21 (run content test R1-R7)
  filtercat  ops/snapshot_filter_catalog.py RUN <store> (:366)
"""
import glob
import os
import shutil

from exopipe.phase3.lib.pages import common, mirror

TIER = 'snapshot'


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
        mirror.gate(u, c.roots, TIER, 'build')
        for p in [os.path.join(c.RUN, 'chain_summary.json'), os.path.join(c.RUN, 'queue.txt'), c.store]:
            u.add_input(p)
        tpl = c.store[:-4] + '_template.csv'
        if os.path.exists(tpl):
            u.add_input(tpl)
        for p in sorted(glob.glob(os.path.join(c.RUN, 'source_catalogs', '*_sources_survivors.csv'))):
            u.add_input(p)
        script = common.legacy_script(c.roots, 'clv15', 'phase3_snapshot_pages.py')
        argv = ['--run', c.RUN, '--proto-root', CFG.paths.SNAPSHOT_FLEET_OUT, '--masters-root', S.MASTERS_ROOT,
                '--fit-store', c.store, '--nproc', c.nproc] + (['--resume'] if c.keep else [])
        res = common.lrun(u, script, argv, python=S.PYTHON_KEY, env=env_for(c), cwd=c.W, log=os.path.join(c.W, 'pages.log'))
        if not res.ok:
            raise common.Refusal(f'P18-P20 snapshot pages refused: {res.reason or res.tail[-300:]}')
        ct = common.legacy_script(c.roots, 'ops', 'snapshot_rechain_content_test.py')
        res2 = common.lrun(u, ct, [c.RUN, '--pages'], python=S.PYTHON_KEY, cwd=c.W, log=os.path.join(c.W, 'pages_content_test.log'))
        u.meta['run_content_test'] = res2.tail.splitlines()[-1] if res2.tail else ''
        if not res2.ok:
            raise common.Refusal(f'P21 run content test after the pages failed: {u.meta["run_content_test"]}')
        u.add_output(c.RUN, 'snapshot_page_run')
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
        res = common.lrun(u, script, [c.RUN, c.store], python=S.PYTHON_KEY, cwd=c.W, log=os.path.join(c.W, 'filtercat.log'))
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
        print(f'  build: phase3_snapshot_pages.py --run {c.RUN} --proto-root {CFG.paths.SNAPSHOT_FLEET_OUT} --masters-root '
              f'{S.MASTERS_ROOT} --fit-store {c.store} --nproc {c.nproc}' + (' --resume' if c.keep else '')
              + f'; env {env_for(c)}; then snapshot_rechain_content_test.py {c.RUN} --pages')
    if 'filtercat' in c.stages:
        print(f'  filtercat: snapshot_filter_catalog.py {c.RUN} {c.store} -> master_catalog.csv, coadd_root.txt [filter_catalog]')
    for r in refuse:
        print(f'  WOULD REFUSE: {r}')
    return 2 if refuse else 0
