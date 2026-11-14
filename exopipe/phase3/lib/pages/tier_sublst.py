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

THE PAGES OF A NIGHT (design 2.3a of plan/PER_HOUR_GAP_DESIGNS_20261111.md, stage 3; lib/pages/night.py): the same stages
and the same production scripts, with two differences.
  --sublst-night-dir <the catalog directory of a per-hour night pass>: prep reads the hour's catalog VIEW of that pass (the
      index hour_views.json names its root) through a staged directory of links, by the one-line patch of
      lib/pages/wraps/prep_catalog.py (pinned script versions: pages_sublst.PREP_SHA1). An hour the pass left unchanged has
      no page run to make (its earlier run stands); an hour the pass did not view REFUSES.
  --sublst-base-runs <the directory of the earlier page runs>: build renders only the blocks the earlier run of the hour
      cannot vouch for (a changed or new selection row, window-tie row or reference-tie row; a frame whose searched,
      predicted or window-Master files changed; anything, when the code, the all-nights catalog build, E_params.json, the
      search root or the linked directories differ) and carries the others as hard links under their present rank; prep
      carries the earlier run's leave-one-out window Masters whose inputs are unchanged.
Every page run of this tier ends its build with the content test of its blocks (night.test) and the record page_blocks.json a
later run reads to carry blocks; a run that carried blocks REFUSES when the test fails.
"""
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor

from exopipe.phase3.lib.pages import common, mirror, night

TIER = 'sublst'


def add_arguments(p):
    g = p.add_argument_group('sub-LST tier (--tier sublst)')
    g.add_argument('--sublst-place-target', metavar='DIR',
                   help='place stage destination (default pages_sublst.PLACE_TARGET, the review host - refused in the window); '
                        'a parity run passes a local directory')
    g.add_argument('--sublst-night-dir', metavar='DIR',
                   help='THE PAGES OF A NIGHT: the catalog directory of a per-hour night pass (it holds hour_views.json, '
                        'written by the stage view of `catalog --tier sublst`); each LST hour is prepared from its catalog view '
                        'of that pass. Without --lsts: every hour the pass wrote a view root for')
    g.add_argument('--sublst-catalog-dir', metavar='DIR',
                   help='parity/scratch, one LST hour: prep reads THIS catalog directory in place of the production hour\'s '
                        '(through the same pinned one-line patch as the pages of a night)')
    g.add_argument('--sublst-base-runs', metavar='DIR',
                   help='the directory of the EARLIER page runs (Run_SLST<hh>B): a block whose inputs are unchanged since the '
                        'earlier run of its hour is carried from it (a hard link) and not rendered; where the earlier run '
                        'cannot vouch, every block is rendered (lib/pages/night.py)')


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
    c.night_dir, c.views, c.not_viewed = None, {}, []
    if getattr(args, 'sublst_night_dir', None):
        c.night_dir = os.path.abspath(args.sublst_night_dir)
        ixp = os.path.join(c.night_dir, S.NIGHT_INDEX)
        if not os.path.isfile(ixp):
            raise common.Refusal(f'P18 pages of a night: {c.night_dir} holds no {S.NIGHT_INDEX} (the stage view of `catalog '
                                 f'--tier sublst` writes it)')
        with open(ixp) as f:
            ix = json.load(f)
        c.views, c.not_viewed = ix.get('hours') or {}, list(ix.get('not_viewed') or [])
        if not lsts:
            lsts = sorted(L for L, e in c.views.items() if e.get('kind') == 'view' and e.get('written_here'))
            if not lsts:
                raise common.NoInput(f'{c.night_dir}: the night pass wrote no view root (no hour changed): no page run to make')
    c.base_runs = os.path.abspath(args.sublst_base_runs) if getattr(args, 'sublst_base_runs', None) else None
    c.catalog_dir = os.path.abspath(args.sublst_catalog_dir) if getattr(args, 'sublst_catalog_dir', None) else None
    if c.catalog_dir and (c.night_dir or len(lsts) != 1 or not os.path.isdir(c.catalog_dir)):
        raise common.Refusal(f'P18: --sublst-catalog-dir names the catalog directory of ONE LST hour ({len(lsts)} given), is not '
                             f'used with --sublst-night-dir, and must exist ({c.catalog_dir})')
    c.ident = None
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


def night_view(c, lst):
    """None outside the pages of a night; else the hour's entry of the night pass's index (kind view: root, all_nights).
    REFUSES for an hour the pass did not view; an hour the pass left as its all-nights directory has no page run to make."""
    if not c.night_dir:
        return None
    e = c.views.get(lst)
    if e is None:
        raise common.Refusal(f'P18 pages of a night: the night pass {c.night_dir} did not view {lst}'
                             + (' (it lists the hour as not viewed)' if lst in c.not_viewed else ''))
    if e.get('kind') != 'view':
        raise common.NoInput(f'{lst}: the night pass {c.night_dir} changed nothing in this hour (it is viewed as its all-nights '
                             f'catalog directory {e.get("all_nights")}): the hour\'s earlier page run stands')
    return e


def frozen_of(c, lst, P, K):
    """The all-nights catalog build set K's catalogs stand on. A night run takes it from the pass's index entry of the hour
    (DECISION-20261114-PR47-CARRIED-ROOT-READERS-TAKE-TABLE-FROM-INDEX: a carried view root's own catalog_view.json is never
    read for it): the record_sha1 of the served build's key (source_catalogs_<served tag>_<K>, which prep copies as
    source_catalogs_<K>); '' when the entry names none. Otherwise night.frozen_id of the run's copy."""
    e = c.views.get(lst) if c.night_dir else None
    if e is None:
        return night.frozen_id(os.path.join(P, f'source_catalogs_{K}'))
    tag = common.CFG().pages_sublst.SERVED_CATALOG_TAG          # sublst_pages_prep.py:104 copies source_catalogs_<tag>_<K>
    return ((e.get('keys') or {}).get(f'{tag}_{K}') or {}).get('record_sha1') or ''


def base_run(c, lst):
    """The earlier page run of the hour (--sublst-base-runs), or None. REFUSES when it is this pass's own page run (prep
    removes the run directory before anything is read from it)."""
    if not c.base_runs:
        return None
    P = P_of(c, lst)
    b = os.path.join(c.base_runs, os.path.basename(P))
    if os.path.realpath(b) == os.path.realpath(P):
        raise common.Refusal(f'P18: --sublst-base-runs names this pass\'s own page run {P}: the earlier run is another directory')
    return b


def ident(c):
    """What every block of this run is rendered with besides its own rows (night.run_vouches): the code (the frozen tree's
    files, the tree's root normalized), the cross-match sample files and the search root. Computed once per run."""
    if c.ident is None:
        C, S = common.CFG().pages, common.CFG().pages_sublst
        c.ident = dict(code=night.code_digest(c.tree),
                       samples=night.samples_id(c.roots.get('clv15') or C.XMATCH_SAMPLE_DIR, C.XMATCH_SAMPLES[TIER]),
                       search_root=S.SEARCH_ROOT.format(root=c.root, arm=S.ARM))
    return c.ident


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
        view, base = night_view(c, lst), base_run(c, lst)
        if c.catalog_dir:
            loc, cat = os.path.join(c.catalog_dir, os.path.relpath(loc, cat)), c.catalog_dir
            u.note(f'catalog directory {cat} in place of the production hour\'s (--sublst-catalog-dir)')
        if view:
            # THE PAGES OF A NIGHT: the hour's catalog view of the night pass, staged as the directory the script reads
            staging = os.path.join(c.out, S.NIGHT_STAGING.format(lst=lst))
            try:
                st = night.stage_catalog(staging, view['root'], S.NIGHT_ALL_NIGHTS_FILES, S.NIGHT_CONFORM_FILES)
            except night.NightRefusal as e:
                raise common.Refusal(f'P18 pages of a night, {lst}: {e}')
            loc, cat = os.path.join(staging, os.path.relpath(loc, cat)), staging
            u.meta['night'] = dict(night_dir=c.night_dir, view_root=view['root'], staging=staging, **st)
            u.note(f'pages of a night: catalog view {view["root"]} staged in {staging} ({len(st["view"])} entries of the view, '
                   f'{len(st["carried"])} carried from the all-nights catalog directory {st["all_nights"]}'
                   + (f'; not in the all-nights directory: {st["absent"]}' if st['absent'] else '') + ')')
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
        if base:
            # the earlier run's leave-one-out window Masters (the script writes only those the run does not hold)
            lm = night.link_masters(base, P, ident(c), c.root)
            u.meta['loo_masters_carried'] = dict(base=base, **lm)
            u.note(f'leave-one-out window Masters of the earlier page run {base}: '
                   + (f'none carried ({lm["why"]})' if lm['why'] else
                      f'{lm["carried"]} carried, {lm["not_carried"]} not (their window stack or residual changed)'))
        script = common.legacy_script(c.roots, 'ops', 'sublst_pages_prep.py')
        env = dict(code_env(c), SUBLST_LST=lst)
        if view or c.catalog_dir:
            # the tie table of the catalog read: the view's own (staged with it), or beside a catalog directory as production
            # lays it out (<...>/new_<arm>/catalog and <...>/new_<arm>/tie/tie_members.csv)
            tie = (os.path.join(cat, S.NIGHT_TIE) if view else
                   os.path.join(os.path.dirname(os.path.abspath(cat)), 'tie', 'tie_members.csv'))
            env.update(EXOPIPE3_PAGES_CATALOG=cat, EXOPIPE3_PAGES_TIE=tie, EXOPIPE3_PAGES_PREP_SHA1=','.join(S.PREP_SHA1))
            u.meta['tie_table'] = dict(path=tie, present=os.path.isfile(tie))
            wrap = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wraps', 'prep_catalog.py')
            res = common.lrun(u, wrap, [script, S.ARM, P], env=env, cwd=P, log=os.path.join(P, 'logs', 'prep.log'))
            u.meta.setdefault('legacy_code', {})[os.path.abspath(script)] = common.sha1_file(script)
        else:
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
        # THE BLOCKS AN EARLIER PAGE RUN HOLDS (lib/pages/night.py): per set, the rows to render and the images to carry.
        # Without an earlier run, or where it cannot vouch, the set is rendered whole from its selection, as before.
        base = base_run(c, lst) if S.WIPE_CANDIDATES else None
        if c.base_runs and not S.WIPE_CANDIDATES:
            u.note('pages_sublst.WIPE_CANDIDATES is False: no block is carried from the earlier page run (every block is rendered '
                   'over what the run directory holds)')
        idn = ident(c)
        if base and os.path.isfile(os.path.join(base, night.RECORD)):
            u.add_input(os.path.join(base, night.RECORD))
        # the files of every frame, read BEFORE the shards (a file that changes while they run is seen by the next run)
        frames = night.run_frames(P, sets, idn['search_root'], c.root)
        plans, moves, jobs = {}, {}, []
        for K in sets:
            try:
                pl = plans[K] = night.plan(K, P, base, idn, frames)
                sel, moves[K] = night.reduced(pl, P)
            except night.NightRefusal as e:
                raise common.Refusal(f'P18 blocks of {lst} {K}: {e}')
            if sel is None:
                continue                                       # every block of the set is carried
            n = c.shards or C.SUBLST_PAGE_SHARDS[K]
            if sel != pl['selection']:                         # a reduced selection: no more processes than it has frames
                n = max(1, min(n, len({(pl['rows'][i]['lst'], pl['rows'][i]['date'], pl['rows'][i]['band'])
                                       for _rank, i, _rel in (pl['blocks'][pos] for pos in pl['render'])})))
            for s in range(n):
                jobs.append((K, s, n, sel))
        res = []
        if jobs:
            with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
                futs = [(K, s, ex.submit(common.lrun, u, script, [sel, P, s, n], 'default',
                                         env, P, os.path.join(P, 'logs', f'pages_{K}_{s}.log'))) for K, s, n, sel in jobs]
                res = [(K, s, f.result()) for K, s, f in futs]
        bad = [(K, s, r.reason) for K, s, r in res if not r.ok]
        u.meta.update(sets=sets, shards=len(jobs), failed_shards=bad)
        shard_failures(u, bad)
        blocks, carried_any = {}, False
        try:
            for K in sets:
                pl = plans[K]
                moved, none = night.rename(P, moves[K])
                got = night.link_blocks(base, P, pl) if pl['link'] else dict(linked=0, copied=0)
                carried_any = carried_any or bool(pl['link'])
                blocks[K] = dict(rows=len(pl['blocks']), rendered=len(pl['render']), carried=got['linked'] + got['copied'],
                                 carried_as_copies=got['copied'], rendered_whole=pl['whole'], rendered_why=pl['why'],
                                 renamed=moved, frozen=frozen_of(c, lst, P, K))
        except night.NightRefusal as e:
            raise common.Refusal(f'P18 blocks of {lst}: {e}')
        ok, detail, counts = night.test(P, sets, frames)
        for K in sets:
            blocks[K].update(images=counts[K]['images'], without_image=counts[K]['without_image'])
        u.check('blocks.content', ok, value=sum(b['images'] for b in blocks.values()), detail=detail)
        u.meta['page_blocks'] = dict(base=base, sets=blocks, test=detail)
        u.note(f'blocks of {lst}: ' + '; '.join(
            f'{K} {b["rendered"]} rendered + {b["carried"]} carried of {b["rows"]}'
            + (f' (whole set: {b["rendered_whole"]})' if b['rendered_whole'] and base else '') for K, b in blocks.items()))
        if not ok and carried_any:
            raise common.Refusal(f'P18 blocks of {lst}: the page run carries blocks of {base} and fails its content test: {detail}')
        if not ok:
            u.note(f'the blocks do not pass the content test of lib/pages/night.py ({detail}): the record says not complete, and '
                   f'no later page run carries blocks of this one')
        night.write_record(P, dict(version=night.VERSION, complete=bool(ok), lst=lst, base=base, sets=blocks, test=detail,
                                   frames=frames, **idn))
        png_re = common.server_png_re(c.roots)
        pngs = sorted(p for p in glob.glob(os.path.join(P, 'candidates', '*', '*', '*.png')) if png_re.match(os.path.basename(p)))
        for p in pngs:
            u.add_output(p, 'review_page', tier=TIER, roots=c.roots)
        u.meta['blocks'] = {S_: sum(1 for p in pngs if f'/candidates/{S_}/' in p) for S_ in ('I', 'V')}
        done[name] = 'done'


def shard_failures(u, bad):
    """B-30 (a): failed page shards [(set, shard, reason)] refuse the build (pages_sublst.SHARD_FAILURE 'refuse', ruled) or
    are recorded on the unit ('record', production: sublst_pages_build.sh:16-19 waits without checking)."""
    if not bad:
        return
    if common.CFG().pages_sublst.SHARD_FAILURE == 'refuse':
        raise common.Refusal(f'{len(bad)} page shard(s) failed, e.g. {bad[0]}')
    u.note(f'{len(bad)} page shard(s) failed (pages_sublst.SHARD_FAILURE record = production, as sublst_pages_build.sh:16-19): '
           f'{bad[:3]}')


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
        cmd = (['rsync', '-a'] + (['--delete'] if S.PLACE_DELETE else [])   # the reduced selections of a night run are the
               + ['--exclude', night.REDUCED_DIR + '/', P + '/', tgt + '/'])   # renderer's input, not a served selection
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
        if c.night_dir and (c.views.get(lst) or {}).get('kind', 'view') != 'view':
            with run_.unit(f'{lst}:prep', tier=TIER, stage='prep', lst=lst):
                night_view(c, lst)                             # NO-INPUT: the pass changed nothing in the hour
            continue                                           # its earlier page run stands; no stage of the hour runs
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
        e = c.views.get(lst) if c.night_dir else None
        if c.night_dir and (e is None or e.get('kind') != 'view'):
            msg = (f'the night pass {c.night_dir} did not view the hour' if e is None else
                   'the night pass changed nothing in this hour: its earlier page run stands (NO-INPUT)')
            print(f'  {lst}: {msg}')
            if e is None and 'prep' in c.stages:
                refuse.append(f'{lst}: P18 {msg}')
            continue
        if e:
            # prep stages the view root's entries and, from the all-nights directory, the records a night does not write
            vroot, an = e['root'], e.get('all_nights') or ''
            loc = os.path.join(vroot, os.path.relpath(loc, cat))
            print(f'  {lst}: page run {P}; PAGES OF A NIGHT: catalog view {vroot} (all-nights catalog directory {an}), staged '
                  f'in {os.path.join(c.out, S.NIGHT_STAGING.format(lst=lst))}')
        else:
            if c.catalog_dir:
                loc, cat = os.path.join(c.catalog_dir, os.path.relpath(loc, cat)), c.catalog_dir
            print(f'  {lst}: page run {P}; catalogue {cat}')
        if c.base_runs:
            b = base_run(c, lst)
            why = night.run_vouches(b, ident(c))
            print(f'    earlier page run {b}: ' + (f'every block is rendered ({why})' if why else
                                                    'its record is complete and of this code: unchanged blocks are carried'))
        okloc = os.path.isfile(loc) and os.path.getsize(loc) > 0
        print(f'    gate P11 localizations: {"present" if okloc else "MISSING " + loc}')
        if not okloc and 'prep' in c.stages:
            refuse.append(f'{lst}: P11 {loc}')
        if e:                                     # as night.stage_catalog stages them: the view root's entries; from the
            carried = set(S.NIGHT_ALL_NIGHTS_FILES)   # all-nights directory only the records a night does not write and pertie/
            miss = [f for f in S.REQUIRED_FILES if not os.path.exists(os.path.join(vroot, f))
                    and not (an and (f in carried or f.startswith('pertie' + os.sep) or f.startswith('pertie/'))
                             and os.path.exists(os.path.join(an, f)))]
            if not an:
                miss.append('all_nights (the index entry names no all-nights catalog directory)')
            miss += [f'{f} (the view root\'s own conformance read)' for f in S.NIGHT_CONFORM_FILES
                     if not os.path.isfile(os.path.join(vroot, f))]
        else:
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
