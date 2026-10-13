"""THE review_dir merge (crosscheck decision 6, group 12: pages owns it and its one config key pages.REVIEW_DIR_MERGE; serve's
views read the merged files, and serve's sub-LST serve-time merge on the cumulative union calls ``run`` below).

The merge writes the review_dir column of <U>/source_catalogs/<band>_sources_survivors.csv (the union the catalog step - deep -
or serve - sub-LST cumulative - wrote) from the page indexes, in place through .new + rename:
- 'in-place' (default; pages.DEEP_MERGE for deep, always for sub-LST): ops/p3s3_merge_review_dir.py run by core.legacy.run
  with the tier's switches from pages.REVIEW_DIR_MERGE (deep: un-keyed, p3s3_pass.sh:195; sub-LST: INDEX_ONLY + STOKES_KEY,
  sublst_serve.sh:52);
- 'keyed' (deep only, M14 item 2, with pages.PANEL_DIR_LAYOUT 'per-set'): lib/pages/m14.keyed_merge.
Gate P10: the merge refuses (non-zero exit / REFUSED text, or the keyed merge's own refusal) -> the unit is refused.
Product class survivors_review_dir, tested with the pre-merge digest of every other column (only review_dir may change).
"""
import glob
import os

from exopipe.phase3.lib.pages import common


def survivors(U):
    return sorted(glob.glob(os.path.join(U, 'source_catalogs', '*_sources_survivors.csv')))


def run(u, U, tier, roots, mode=None, log=None, dry_run=False):
    """Merge review_dir for tier 'deep' or 'sublst' into the survivors union of run directory U, inside unit ``u``.
    Returns {'mode', 'files', 'pre_digest': {path: sha1}, 'tail'}; raises Refusal on P10."""
    C = common.CFG().pages
    if tier not in C.REVIEW_DIR_MERGE:
        raise common.Refusal(f'review_dir merge: no pages.REVIEW_DIR_MERGE entry for tier {tier!r}')
    mode = mode or (C.DEEP_MERGE if tier == 'deep' else 'in-place')
    files = survivors(U)
    if not files:
        raise common.NoInput(f'no survivors union under {U}/source_catalogs (catalog --stages union writes it)')
    idx = sorted(glob.glob(os.path.join(U, 'source_catalogs', '*_pages_index*.csv')))
    if not idx:
        raise common.Refusal(f'P10: no page index under {U}/source_catalogs - build the pages first')
    for p in files + idx:
        u.add_input(p)
    pre = {os.path.abspath(p): common.digest_without(p, 'review_dir') for p in files}
    out = dict(mode=mode, files=files, pre_digest=pre, tail='')
    if dry_run:
        return out
    if mode == 'keyed':
        if tier != 'deep':
            raise common.Refusal('the keyed (M14) merge is defined for the deep tier only')
        from exopipe.phase3.lib.pages import m14
        res = m14.keyed_merge(U, index_only=False)
        out['tail'] = f"keyed review_dir merge: {res['matched']} rows matched a page, {res['without']} without"
        u.meta['merge'] = dict(mode=mode, matched=res['matched'], without=res['without'])
        return out
    if mode != 'in-place':
        raise common.Refusal(f'pages.DEEP_MERGE {mode!r}: expected in-place or keyed')
    script = common.legacy_script(roots, 'ops', 'p3s3_merge_review_dir.py')
    env = {k: v for k, v in C.REVIEW_DIR_MERGE[tier].items()}
    res = common.lrun(u, script, [U], python='default', env=env, cwd=U, log=log or os.path.join(U, 'pages_merge.log'))
    out['tail'] = res.tail.splitlines()[-1] if res.tail else ''
    u.meta['merge'] = dict(mode=mode, env=env, rc=res.rc, last_line=out['tail'])
    if not res.ok:
        raise common.Refusal(f'P10 review_dir merge refused: {res.reason or res.tail[-300:]}')
    return out


def add_outputs(u, merged):
    """Register every rewritten survivors file with its pre-merge digest (only review_dir may change)."""
    layout = common.CFG().pages.PANEL_DIR_LAYOUT
    for p in merged['files']:
        u.add_output(p, 'survivors_review_dir', pre_digest=merged['pre_digest'].get(os.path.abspath(p)), layout=layout)
