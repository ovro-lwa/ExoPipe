"""Sub-LST tier of `exopipe3 pages` (production: the page part of ops/sublst_publish_lst.sh and ops/sublst_pages_build.sh).

Unit of work: one LST hour (item kind `lst`, core.cells.parse_lst; per-hour use: the LSTs of the --epochs items).
Units <LST>:prep, <LST>:input, <LST>:build, <LST>:index:<K>, <LST>:place. Page run P = <out>/Run_SLST<hh>B (production
/fast/gh/main/phase3/sublst_pages/Run_SLST<hh>B on the publish host, then rsync'ed to the review host). No fingerprint skip in
this tier (every block renders every publish, as today: sublst_pages_build.sh:9 wipes candidates/).

Stages (core.legacy.run, site PYTHON['default'], code from a VERIFIED FROZEN TREE - gate P16, pages_sublst.CODE_PINNED):
  prep   sublst_pages_prep.py <ARM> P (SUBLST_LST=<LST>) after rm -rf P; then P/parent_catalogs -> <root>/parent_catalogs_<LST>
  input  p3t_pages_input.py P
  build  build_tenmin_pages.py P/source_catalogs/tenmin_selection_<K>.csv P <s> <n>, every shard of every set at once
         (SUBLST_SEARCH_ROOT, SUBLST_LOO_MASTER_DIR=P/loo_masters); a set whose selection holds only its header is skipped
  index  p3t_pages_index.py P <K> per set, in pages.SUBLST_INDEX_ORDER (a refusal refuses the LST: P15)
  place  rsync -a --delete P/ <target>/ (a LOCAL target only: the review-host copy is production and is refused in the window)
The review_dir merge of the served cumulative union happens at serve time (serve's sub-LST prepare) through lib/pages/merge.run
with pages.REVIEW_DIR_MERGE['sublst'].
"""
import glob
import hashlib
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor

from exopipe.phase3.lib.pages import common, mirror

TIER = 'sublst'


def add_arguments(p):
    g = p.add_argument_group('sub-LST tier (--tier sublst)')
    g.add_argument('--sublst-place-target', metavar='DIR',
                   help='place stage destination (default pages_sublst.PLACE_TARGET, the review host - refused in the window); '
                        'a parity run passes a local directory')


class Ctx:
    pass


def frozen_tree(roots):
    """(tree, None) when the ops and clv15 roots are inside ONE verified frozen tree (FROZEN.ok + MANIFEST.sha256 whose every
    line verifies; sublst_code_root.sh:15-31), else (None, reason)."""
    ops, clv = roots.get('ops'), roots.get('clv15')
    if not ops or not clv:
        return None, 'ops and clv15 roots are both needed'
    tree = os.path.dirname(ops)
    if os.path.dirname(clv) != tree:
        return None, f'ops ({ops}) and clv15 ({clv}) are not one tree - one run never mixes code trees'
    if not (os.path.isfile(os.path.join(tree, 'FROZEN.ok')) and os.path.isfile(os.path.join(tree, 'MANIFEST.sha256'))):
        return None, f'{tree} is not a frozen code tree (no FROZEN.ok / MANIFEST.sha256)'
    # as `sha256sum -c --quiet --status MANIFEST.sha256` (sublst_code_root.sh:23): a line is '<64 hex>  <path>' (text mode)
    # or '<64 hex> *<path>' (binary mode); other lines are improperly formatted and skipped (no --strict); every formatted
    # line must verify, and a manifest with NO properly formatted line refuses (sha256sum -c exits 1 on it)
    n = 0
    with open(os.path.join(tree, 'MANIFEST.sha256')) as f:
        for line in f:
            line = line.rstrip('\n')
            esc = line.startswith('\\')                  # GNU escaping of a name holding '\' or a newline
            if esc:
                line = line[1:]
            want, sep, rel = line[:64], line[64:66], line[66:]
            if not (re.fullmatch(r'[0-9a-fA-F]{64}', want) and sep in ('  ', ' *') and rel):
                continue
            if esc:
                rel = rel.replace('\\\\', '\x00').replace('\\n', '\n').replace('\x00', '\\')
            want = want.lower()
            p = os.path.join(tree, rel)
            try:
                h = hashlib.sha256(open(p, 'rb').read()).hexdigest()
            except OSError:
                return None, f'{tree}/MANIFEST.sha256 does not verify: {rel} unreadable'
            if h != want:
                return None, f'{tree}/MANIFEST.sha256 does not verify: {rel} changed after the freeze'
            n += 1
    if not n:
        return None, f'{tree}/MANIFEST.sha256 has no properly formatted checksum line (sha256sum -c refuses it)'
    return tree, None


def context(args, run):
    from exopipe.phase3.core import cells
    C, S = common.CFG().pages, common.CFG().pages_sublst
    c = Ctx()
    lsts = []
    if getattr(args, 'lsts', None):
        lsts = [cells.parse_lst(x) for x in cells.expand(args.lsts, universe=cells.LSTS, kind='LST')]
    for lst, _d in common.epochs_of(args):
        if lst not in lsts:
            lsts.append(lst)
    if not lsts:
        raise common.Refusal('sub-LST: give --lsts (e.g. 14h) or --epochs (per-hour items <LST>_<date>)')
    c.lsts = lsts
    c.out = run.out_root
    c.roots = common.legacy_roots(args.legacy_root)
    c.stages = common.select_stages(args, TIER)
    c.sets = common.select_sets(args, TIER)
    c.root = common.CFG().paths.SUBLST_ROOT
    c.shards = args.shards
    c.place_target = getattr(args, 'sublst_place_target', None)
    c.tree, c.tree_err = frozen_tree(c.roots)
    return c


def P_of(c, lst):
    return os.path.join(c.out, common.CFG().pages.RUN_DIR['sublst'].format(hh=lst[:2]))


def code_env(c):
    """The frozen-tree environment (sublst_code_root.sh:28-35): SUBLST_CODE_ROOT and the import guard first on PYTHONPATH."""
    if not c.tree:
        return {}
    return {'SUBLST_CODE_ROOT': c.tree,
            'PYTHONPATH': os.path.join(c.tree, 'exopipe_ops', 'sublst_frozen_guard')}


def gate_code(c, u):
    S = common.CFG().pages_sublst
    if S.CODE_PINNED and not c.tree:
        raise common.Refusal(f'P16 sub-LST code not pinned: {c.tree_err} - pass --legacy-root ops=<frozen tree>/exopipe_ops '
                             f'--legacy-root clv15=<frozen tree>/clv15 (a tree written by sublst_freeze_code.sh)')
    u.meta['code_tree'] = c.tree


def has_sel(P, K):
    return common.n_rows(os.path.join(P, 'source_catalogs', f'tenmin_selection_{K}.csv')) > 0


def stage_prep(c, run, done, lst):
    S = common.CFG().pages_sublst
    name = f'{lst}:prep'
    P = P_of(c, lst)
    cat = S.CATALOG_DIR.format(root=c.root, lst=lst, arm=S.ARM)
    with run.unit(name, tier=TIER, stage='prep', lst=lst) as u:
        gate_code(c, u)
        loc = S.LOC_REQUIRED.format(root=c.root, lst=lst, arm=S.ARM)
        if not (os.path.isfile(loc) and os.path.getsize(loc) > 0):
            raise common.Refusal(f'P11: no localizations for {lst} ({loc}; sublst_publish_lst.sh:26)')
        miss = [f for f in S.REQUIRED_FILES if not os.path.exists(os.path.join(cat, f))]
        if miss:
            raise common.Refusal(f'P12/P13: {cat} lacks {miss} (window-tie table / rejected-source record; sublst_pages_prep.py:39-48)')
        mirror.gate(u, c.roots, TIER, 'prep')
        for f in ('master_catalog.csv', 'v_deep_images.csv') + tuple(S.REQUIRED_FILES):
            p = os.path.join(cat, f)
            if os.path.exists(p):
                u.add_input(p)
        u.add_input(loc)
        if S.FRESH_RUN_DIR and os.path.lexists(P):
            shutil.rmtree(P)                                   # sublst_publish_lst.sh:27
        os.makedirs(os.path.join(P, 'logs'), exist_ok=True)
        script = common.legacy_script(c.roots, 'ops', 'sublst_pages_prep.py')
        env = dict(code_env(c), SUBLST_LST=lst)
        res = common.lrun(u, script, [S.ARM, P], env=env, cwd=P, log=os.path.join(P, 'logs', 'prep.log'))
        if not res.ok:
            raise common.Refusal(f'pages prep failed: {res.reason}')
        link = os.path.join(P, 'parent_catalogs')
        if not os.path.lexists(link):
            os.symlink(S.PARENT_LINK.format(root=c.root, lst=lst), link)      # sublst_publish_lst.sh:29
        u.add_output(P, 'sublst_page_run', cat_dir=cat)
        done[name] = 'done'


def stage_input(c, run, done, lst):
    name = f'{lst}:input'
    P = P_of(c, lst)
    with run.unit(name, tier=TIER, stage='input', lst=lst) as u:
        gate_code(c, u)
        common.prerequisite(run, done, f'{lst}:prep')
        mirror.gate(u, c.roots, TIER, 'input')
        u.add_input(os.path.join(P, 'master_catalog.csv'))
        script = common.legacy_script(c.roots, 'ops', 'p3t_pages_input.py')
        res = common.lrun(u, script, [P], env=code_env(c), cwd=P, log=os.path.join(P, 'logs', 'input.log'))
        if not res.ok:
            raise common.Refusal(f'pages input failed: {res.reason}')
        outs = []
        for K in common.CFG().pages.PAGE_SETS['sublst']:
            for pat in (f'pages_input_{K}.csv', f'tenmin_selection_{K}.csv'):
                p = os.path.join(P, 'source_catalogs', pat)
                if os.path.exists(p) and common.n_rows(p) > 0:
                    u.add_output(p, 'pages_input', tier=TIER)
                    outs.append(p)
        u.meta['output_fp'] = common.output_fps(outs)
        done[name] = 'done'


def stage_build(c, run, done, lst):
    C, S = common.CFG().pages, common.CFG().pages_sublst
    name = f'{lst}:build'
    P = P_of(c, lst)
    with run.unit(name, tier=TIER, stage='build', lst=lst) as u:
        gate_code(c, u)
        u.meta['test_ctx'] = {'roots': c.roots, 'tier': TIER}
        common.prerequisite(run, done, f'{lst}:input')
        mirror.gate(u, c.roots, TIER, 'build')
        sets = [K for K in c.sets if has_sel(P, K)]
        if not sets:
            raise common.NoInput(f'{lst}: no set with a selected source (sublst_pages_build.sh:13)')
        for K in sets:
            u.add_input(os.path.join(P, 'source_catalogs', f'tenmin_selection_{K}.csv'))
        if S.WIPE_CANDIDATES and os.path.isdir(os.path.join(P, 'candidates')):
            shutil.rmtree(os.path.join(P, 'candidates'))       # sublst_pages_build.sh:9
        script = common.legacy_script(c.roots, 'ops', 'build_tenmin_pages.py')
        env = dict(code_env(c), SUBLST_SEARCH_ROOT=S.SEARCH_ROOT.format(root=c.root, arm=S.ARM),
                   SUBLST_LOO_MASTER_DIR=os.path.join(P, 'loo_masters'))
        jobs = []
        for K in sets:
            n = c.shards or C.SUBLST_PAGE_SHARDS[K]
            for s in range(n):
                jobs.append((K, s, n))
        with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
            futs = [(K, s, ex.submit(common.lrun, u, script,
                                     [os.path.join(P, 'source_catalogs', f'tenmin_selection_{K}.csv'), P, s, n], 'default',
                                     env, P, os.path.join(P, 'logs', f'pages_{K}_{s}.log'))) for K, s, n in jobs]
            res = [(K, s, f.result()) for K, s, f in futs]
        bad = [(K, s, r.reason) for K, s, r in res if not r.ok]
        u.meta.update(sets=sets, shards=len(jobs), failed_shards=bad)
        if bad:
            if S.SHARD_FAILURE == 'refuse':
                raise common.Refusal(f'{len(bad)} page shard(s) failed, e.g. {bad[0]}')
            u.note(f'{len(bad)} page shard(s) failed (recorded, as sublst_pages_build.sh:16-19; B-30 open): {bad[:3]}')
        png_re = common.server_png_re(c.roots)
        pngs = sorted(p for p in glob.glob(os.path.join(P, 'candidates', '*', '*', '*.png')) if png_re.match(os.path.basename(p)))
        for p in pngs:
            u.add_output(p, 'review_page', tier=TIER, roots=c.roots)
        u.meta['blocks'] = {S_: sum(1 for p in pngs if f'/candidates/{S_}/' in p) for S_ in ('I', 'V')}
        done[name] = 'done'


def stage_index(c, run, done, lst, K):
    name = f'{lst}:index:{K}'
    P = P_of(c, lst)
    with run.unit(name, tier=TIER, stage='index', lst=lst, set=K) as u:
        gate_code(c, u)
        u.meta['test_ctx'] = {'roots': c.roots, 'tier': TIER}
        if not has_sel(P, K):
            raise common.NoInput(f'{lst} {K}: no selected source')
        common.prerequisite(run, done, f'{lst}:build')
        pin = os.path.join(P, 'source_catalogs', f'pages_input_{K}.csv')
        u.add_input(pin)
        bands = sorted({r['band'] for r in common.read_csv(pin)[1]})
        script = common.legacy_script(c.roots, 'ops', 'p3t_pages_index.py')
        res = common.lrun(u, script, [P, K], env=code_env(c), cwd=P, log=os.path.join(P, 'logs', f'index_{K}.log'))
        if not res.ok:
            raise common.Refusal(f'P15 INDEX REFUSED {K}: {res.reason}')
        for b in bands:
            p = os.path.join(P, 'source_catalogs', f'{b}_pages_index_{K[0]}.csv')
            if os.path.exists(p):
                u.add_output(p, 'page_index', roots=c.roots)
        u.add_output(os.path.join(P, 'judgements', f'queue_p3s3_{K}.txt'), 'review_queue')
        u.add_output(pin, 'pages_input', tier=TIER)
        done[name] = 'done'


def place_target(c, P):
    """(target, None) or (None, reason): --sublst-place-target, else pages_sublst.PLACE_TARGET on the site file's REVIEW_HOST
    (sublst_publish_lst.sh:38; host names live only in the site file - README rule 2)."""
    if c.place_target:
        return c.place_target, None
    from exopipe.phase3.config import CFG
    host = getattr(CFG.site, 'REVIEW_HOST', None)
    if not host:
        return None, 'the site file names no REVIEW_HOST (the host the sub-LST pages are placed on); pass --sublst-place-target'
    return common.CFG().pages_sublst.PLACE_TARGET.format(review_host=host, run=os.path.basename(P)), None


def stage_place(c, run, done, lst):
    from exopipe.phase3.config import CFG
    S = common.CFG().pages_sublst
    name = f'{lst}:place'
    P = P_of(c, lst)
    run_name = os.path.basename(P)
    target, why = place_target(c, P)
    with run.unit(name, tier=TIER, stage='place', lst=lst, target=target) as u:
        if not target:
            raise common.Refusal(f'P17 place: {why}')
        for K in c.sets:
            if has_sel(P, K):
                common.prerequisite(run, done, f'{lst}:index:{K}')
        if ':' in target.split('/')[0]:
            raise common.Refusal(f'P17 place: {target} is the review host copy (production) - refused in the refactor window '
                                 f'(paths.PRODUCTION_WRITES False); a parity run passes --sublst-place-target <local dir>')
        tgt = target.rstrip('/')
        if not CFG.paths.PRODUCTION_WRITES and CFG.production_root_of(tgt):
            raise common.Refusal(f'P17 place: {tgt} is under the production root {CFG.production_root_of(tgt)}')
        if os.path.basename(tgt) != run_name:
            tgt = os.path.join(tgt, run_name)
        os.makedirs(tgt, exist_ok=True)
        cmd = ['rsync', '-a'] + (['--delete'] if S.PLACE_DELETE else []) + [P + '/', tgt + '/']
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        u.meta['rsync'] = dict(cmd=cmd, rc=r.returncode, tail=r.stdout[-500:])
        if r.returncode != 0:
            raise common.Refusal(f'P17 rsync to {tgt} failed (rc {r.returncode}): {r.stdout[-200:]}')
        u.add_output(tgt, 'sublst_page_run')
        done[name] = 'done'


def run(args, run_):
    c = context(args, run_)
    if args.dry_run:
        return dry_run(c, run_)
    C = common.CFG().pages
    done = {}
    for lst in c.lsts:
        P = P_of(c, lst)
        os.makedirs(c.out, exist_ok=True)
        with common.Lease(run_.out_root, P, C.LEASE_WAIT_S):
            for st in c.stages:
                if st == 'prep':
                    stage_prep(c, run_, done, lst)
                elif st == 'input':
                    stage_input(c, run_, done, lst)
                elif st == 'build':
                    stage_build(c, run_, done, lst)
                elif st == 'index':
                    for K in C.SUBLST_INDEX_ORDER:
                        if K in c.sets:
                            stage_index(c, run_, done, lst, K)
                elif st == 'place':
                    stage_place(c, run_, done, lst)
                if any(u.status in ('refused', 'error') and u.meta.get('lst') == lst for u in run_.units):
                    break
    return None


def dry_run(c, run_):
    S = common.CFG().pages_sublst
    refuse = []
    print(f'exopipe3 pages --tier sublst: DRY RUN (nothing written); LSTs {", ".join(c.lsts)}; stages {",".join(c.stages)}')
    print(f'  legacy roots ' + ', '.join(f'{k}={v}' for k, v in sorted(c.roots.items())))
    if S.CODE_PINNED and not c.tree:
        refuse.append(f'P16 code not pinned: {c.tree_err}')
        print(f'  gate P16 frozen code: NOT PINNED ({c.tree_err})')
    else:
        print(f'  gate P16 frozen code: {c.tree or "not required (pages_sublst.CODE_PINNED False)"}')
    for lst in c.lsts:
        P = P_of(c, lst)
        cat = S.CATALOG_DIR.format(root=c.root, lst=lst, arm=S.ARM)
        loc = S.LOC_REQUIRED.format(root=c.root, lst=lst, arm=S.ARM)
        print(f'  {lst}: page run {P}; catalogue {cat}')
        okloc = os.path.isfile(loc) and os.path.getsize(loc) > 0
        print(f'    gate P11 localizations: {"present" if okloc else "MISSING " + loc}')
        if not okloc and 'prep' in c.stages:
            refuse.append(f'{lst}: P11 {loc}')
        miss = [f for f in S.REQUIRED_FILES if not os.path.exists(os.path.join(cat, f))]
        print(f'    gate P12/P13 required files: {"present" if not miss else "MISSING " + str(miss)}')
        if miss and 'prep' in c.stages:
            refuse.append(f'{lst}: P12/P13 {miss}')
        for K in c.sets:
            n = common.CFG().pages.SUBLST_PAGE_SHARDS[K]
            print(f'    build {K}: {c.shards or n} shards of build_tenmin_pages.py (tenmin_selection_{K}.csv written by input)')
        for st in c.stages:
            for l in mirror.plan_lines(c.roots, TIER, st):
                print(l)
                if 'WOULD REFUSE' in l:
                    refuse.append(f'{lst} {st}: P3 mirror drift')
        if 'place' in c.stages:
            tgt, why = place_target(c, P)
            if not tgt:
                print(f'    place: WOULD REFUSE ({why})')
                refuse.append(f'{lst}: P17 {why}')
            else:
                remote = ':' in tgt.split('/')[0]
                print(f'    place: rsync -a{" --delete" if S.PLACE_DELETE else ""} {P}/ {tgt}'
                      + ('  -> WOULD REFUSE (review host = production)' if remote else ''))
                if remote:
                    refuse.append(f'{lst}: P17 place target {tgt} is production')
    for r in refuse:
        print(f'  WOULD REFUSE: {r}')
    return 2 if refuse else 0
