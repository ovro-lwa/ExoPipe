"""`status conformance`, `status pages`, `status register` - the masters conformance table and the hub status pages,
each the production tool IN PLACE (attr: module globals set from config, main() called) under the write guard, outputs
re-rooted under --out (<out>/QA, <out>/review_hub, <out>/clv15). status never writes index.html or registry.json.

conformance  ops/conformance_check.py: ROOT = paths.COADD_SB_ROOT, LATEST/HIST under <out>/QA (the production history is
             copied first so the history continues), PAGE <out>/review_hub/conformance.html, REBUILD = --round (default
             status_conformance.ROUND_LIST). It imports exopipe.phase3.membership, which resolves (core/_shim finder) to the
             branch module lib/masters/membership.py: S-G6 refuses when that module differs from production in anything
             but import lines and the NAMED_OUT_OF_MASTER path line (both tables byte-identical). Whole tree only.
pages        nights: ops/build_qa_pages.py (HUB = <out>/review_hub, LEDGER, DD, ROOT, QA from config; --cells NIGHTS ->
             its --only), then its check() (P1-P4) as the content test of class qa_night_pages (S-G9);
             data-processing: ops/build_qa_release1.py, the page production serves since 2026-11-10 (the Release 1 page:
             every tier's coverage). It imports ops/build_data_processing_page.py (the 10-07 page) as B for its helpers
             and sources; B is imported first with LEDGER, DD, ROOT, QA set from config. The builder writes into
             <out>/status/qa_site (OUT) and reads the QA overview fragment and the night pages of <out>/review_hub (HUB);
             the page and coverage.json are then copied into <out>/review_hub, as production installs them into the hub.
             The builder's dated sources come from config (status_pages.DATA_PAGE_SNAP_VIEW, DATA_PAGE_TENMIN_VIEW,
             DATA_PAGE_SNAP_CELL_LISTS, DATA_PAGE_DEC_PENDING; the roots A, GEN, P1, P2, OPS from paths): S-G18 refuses when
             one is absent (production reads it as empty), S-G19 when the page names a view or archive run other than the
             configured ones (the tier titles are literals in its main()).
register     ops/build_ratified_register.py AUDIT_CSV ...: REG = <out>/clv15/RATIFIED_STEPS_REGISTER.csv, PAGE =
             <out>/review_hub/ratified_steps.html.
"""
import os
import shutil

from exopipe.phase3.lib.status import steps_common as K

HISTORY_COLUMNS = ('checked_utc', 'cells', 'pass', 'fail', 'fail_not_in_rebuild_list')   # conformance_check.py:47-48


def drift_lines(prod_lines, branch_lines):
    """S-G6's comparison: diff lines of the branch module against production other than import lines, the NAMED_OUT_OF_MASTER
    path line and the entry point's no-mmap switch (``_fits_nommap.conf.use_memmap = False``, R-GREGG-20261117-NO-MMAP-OF-
    LUSTRE-FILES-READ-OR-WRITE; tests/test_no_mmap.py check 3 requires it in every entry point) - the switch only where the
    BRANCH has it: a branch without a switch that production has is drift."""
    import difflib
    bad = []
    for ln in difflib.unified_diff(prod_lines, branch_lines, lineterm='', n=0):
        if ln.startswith(('---', '+++', '@@')):
            continue
        body = ln[1:].strip()
        if body.startswith(('import ', 'from ', 'NAMED_OUT_OF_MASTER =')):
            continue
        if ln.startswith('+') and body.startswith('_fits_nommap.conf.use_memmap = False'):
            continue
        bad.append(ln[:160])
    return bad


def membership_drift():
    """S-G6: drift_lines of the branch membership module against production; [] when only the permitted lines differ (and the
    named table files are identical)."""
    import hashlib
    from exopipe.phase3.lib.masters import membership as BM
    prod = K.cfg().status_conformance.PRODUCTION_MEMBERSHIP
    if not os.path.exists(prod):
        return [f'production module {prod} not found']
    bad = drift_lines(open(prod).read().splitlines(), open(BM.__file__).read().splitlines())
    pa = os.path.join(os.path.dirname(prod), 'named_out_of_master.csv')
    pb = BM.NAMED_OUT_OF_MASTER
    if os.path.exists(pa) and os.path.exists(pb):
        if hashlib.sha1(open(pa, 'rb').read()).digest() != hashlib.sha1(open(pb, 'rb').read()).digest():
            bad.append(f'named_out_of_master.csv differs: {pa} vs {pb}')
    return bad


def run_conformance(out, round_list, unit, log, allowed):
    from exopipe.phase3.lib.status import steps_inplace as IP
    C = K.cfg()
    CF = C.status_conformance
    latest = K.check_write(K.place(out, 'QA', os.path.basename(CF.LATEST)))
    hist = K.check_write(K.place(out, 'QA', os.path.basename(CF.HISTORY)))
    page = K.check_write(K.place(out, 'review_hub', os.path.basename(CF.PAGE)))
    for p in (latest, hist, page):
        os.makedirs(os.path.dirname(p), exist_ok=True)
    prod_hist = K.expand(CF.HISTORY)
    if not os.path.exists(hist) and os.path.exists(prod_hist):
        shutil.copyfile(prod_hist, hist)
        unit.add_input(prod_hist)
    attrs = dict(ROOT=C.paths.COADD_SB_ROOT, QA=os.path.dirname(latest), LATEST=latest, HIST=hist, PAGE=page,
                 REBUILD=round_list)
    if os.path.exists(round_list):
        unit.add_input(round_list)
    rc, text, mod = IP.call_attr('conformance_check', attrs, 'main', allowed_roots=allowed, log=log, unit=unit)
    return rc, text, latest, hist, page


def run_nights(out, nights, unit, log, allowed):
    from exopipe.phase3.lib.status import steps_inplace as IP
    C = K.cfg()
    hub = K.place(out, 'review_hub', '')
    K.check_write(os.path.join(hub, 'nights', 'x.html'))
    os.makedirs(hub, exist_ok=True)
    attrs = dict(HUB=hub.rstrip('/'), LEDGER=C.status.LEDGER, DD=C.paths.DATA_DECISIONS, ROOT=C.paths.COADD_SB_ROOT,
                 QA=C.status.QA_DIR)
    argv = ['build_qa_pages.py'] + (['--only'] + list(nights) if nights else [])
    IP.search_sources(C.status.SEARCH_LABEL['deep'])      # the register the page imports, pinned (2026-10-15 review)
    rc, text, mod = IP.call_attr('build_qa_pages', attrs, 'main', allowed_roots=allowed, log=log, unit=unit,
                                 env=IP.search_env(C.status.SEARCH_LABEL['deep']), argv=argv)
    return rc, text, hub.rstrip('/')


# build_qa_release1.py:106, the cell-list glob written inside snap_cov(); replaced by status_pages.DATA_PAGE_SNAP_CELL_LISTS
SNAP_COV_LITERAL = "f'{OPS}/snapshot_rechain_Run_SNAPB*3_20261104.cells'"
# S-G19: run-name families the page prints -> the configured name each must equal (the page's text, its tier titles included)
# build_qa_release1.py:243, the snapshot tier title written inside main(); its view name replaced by DATA_PAGE_SNAP_VIEW's
SNAP_TITLE_LITERAL = '10-second and 1-minute: the served view (Run_SNAPCOMB_20261107)'
# The archive family is the Phase 3 archive generation's run names only (paths.ARCHIVE_RUN, Run_YYYYMMDD_000000: generation
# runs start at midnight; the page also links other Run_* trees, e.g. quarantine runs, which this pattern does not match)
PAGE_RUN_TOKENS = (('snap', r'Run_SNAPCOMB_[A-Za-z0-9_]+'), ('tenmin', r'Run_SLST[A-Za-z0-9_]*'),
                   ('archive', r'(?<![A-Za-z0-9_])Run_20\d{6}_000000(?![A-Za-z0-9_])'))


def data_page_sources(C=None):
    """The builder's dated sources from config: {name: value} with the views and files expanded."""
    C = C or K.cfg()
    P = C.status_pages
    kw = dict(snapreview=C.paths.SNAPSHOT_REVIEW_ROOT, transients=C.paths.TRANSIENTS_ROOT)
    return dict(snap_view=K.expand(P.DATA_PAGE_SNAP_VIEW, **kw), tenmin_view=K.expand(P.DATA_PAGE_TENMIN_VIEW, **kw),
                cell_lists=K.expand(P.DATA_PAGE_SNAP_CELL_LISTS, batch='{batch}', **kw),
                dec_pending=K.expand(P.DATA_PAGE_DEC_PENDING, **kw))


def snap_cell_lists(src):
    """The cell lists of the snapshot view's member batches: the view's combined_view_summary.json 'batches', each
    expanded into src['cell_lists'] ({batch} = the member) with exactly one match. Returns (files, problems)."""
    import glob
    import json
    summ = os.path.join(src['snap_view'], 'combined_view_summary.json')
    if '{batch}' not in src['cell_lists']:
        return sorted(glob.glob(src['cell_lists'])), []
    try:
        batches = json.load(open(summ))['batches']
    except (OSError, ValueError, KeyError, TypeError) as e:
        return [], [f'{summ}: no member batch list ({type(e).__name__})']
    files, bad = [], []
    for b in batches:
        m = sorted(glob.glob(src['cell_lists'].format(batch=b)))
        if len(m) == 1:
            files.append(m[0])
        else:
            bad.append(f'member batch {b}: {len(m)} cell lists match {src["cell_lists"]} (need exactly one)')
    return files, bad + ([] if batches else [f'{summ}: an empty member batch list'])


def data_page_missing(src):
    """S-G18: the sources the builder would read as empty. Returns (missing descriptions, the files read)."""
    import glob
    need = [os.path.join(src['snap_view'], 'deep_omission.csv'), os.path.join(src['tenmin_view'], 'deep_omission.csv'),
            os.path.join(src['tenmin_view'], 'v_deep_images.csv'), src['dec_pending']]
    missing = [p for p in need if not os.path.isfile(p)]
    lists, bad = snap_cell_lists(src)
    missing += bad
    if not lists and not bad:
        missing.append(f'{src["cell_lists"]} (no cell list matches)')
    return missing, [p for p in need if p not in missing] + lists


def snap_cov_reader(builder_file, files):
    """The builder's snap_cov() with its cell-list glob literal (SNAP_COV_LITERAL) replaced by the list ``files``
    (snap_cell_lists); the rest of the body is production's own source, compiled from the builder file. Refusal when the
    literal is not there exactly once."""
    import ast
    import glob
    import re
    from exopipe.phase3.core.record import Refusal
    text = open(builder_file).read()
    fn = [n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == 'snap_cov']
    seg = ast.get_source_segment(text, fn[0]) if len(fn) == 1 else ''
    if seg.count(SNAP_COV_LITERAL) != 1:
        raise Refusal(f'S-G18: {builder_file} snap_cov() no longer reads {SNAP_COV_LITERAL} (exactly once): the configured '
                      f'status_pages.DATA_PAGE_SNAP_CELL_LISTS cannot replace it')
    import types
    ns = dict(glob=types.SimpleNamespace(glob=lambda pat: list(pat)), re=re, DATA_PAGE_SNAP_CELL_LISTS=tuple(files))
    exec(compile(seg.replace(SNAP_COV_LITERAL, 'DATA_PAGE_SNAP_CELL_LISTS'), builder_file, 'exec'), ns)   # noqa: S102
    return ns['snap_cov']


def builder_main(builder_name, builder_file, view):
    """A stand-in for the builder's main() that, called on the imported module, compiles production's main() from the
    builder file with its snapshot tier title literal (SNAP_TITLE_LITERAL) naming ``view`` instead, into the module's own
    namespace, and calls it. Refusal when the literal is not there exactly once or ``view`` is not a Run_SNAPCOMB_* name."""
    import ast
    import re
    import sys
    from exopipe.phase3.core.record import Refusal
    if not re.fullmatch(dict(PAGE_RUN_TOKENS)['snap'], view):
        raise Refusal(f'S-G18: status_pages.DATA_PAGE_SNAP_VIEW names {view!r}, not a Run_SNAPCOMB_* view')
    text = open(builder_file).read()
    fn = [n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == 'main']
    seg = ast.get_source_segment(text, fn[0]) if len(fn) == 1 else ''
    if seg.count(SNAP_TITLE_LITERAL) != 1:
        raise Refusal(f'S-G18: {builder_file} main() no longer titles the snapshot tier {SNAP_TITLE_LITERAL} (exactly once): '
                      f'the configured status_pages.DATA_PAGE_SNAP_VIEW cannot replace it')
    code = compile('\n' * (fn[0].lineno - 1) + seg.replace(SNAP_TITLE_LITERAL, SNAP_TITLE_LITERAL.replace(
        re.search(dict(PAGE_RUN_TOKENS)['snap'], SNAP_TITLE_LITERAL).group(0), view)), builder_file, 'exec')

    def main(*args):
        ns = sys.modules[builder_name].__dict__
        exec(code, ns)                    # noqa: S102 - production's main(), the title literal replaced, in its module
        return ns['main'](*args)
    return main


def page_run_mismatches(page_text, src, archive_run):
    """S-G19: run names of each family the page prints that differ from the configured one ([] = consistent)."""
    import re
    want = dict(snap=os.path.basename(src['snap_view'].rstrip('/')), tenmin=os.path.basename(src['tenmin_view'].rstrip('/')),
                archive=archive_run)
    bad = []
    for fam, pat in PAGE_RUN_TOKENS:
        other = sorted(set(re.findall(pat, page_text)) - {want[fam]})
        if other:
            bad.append(f'{fam}: the page names {other}, config {want[fam]}')
    return bad


def run_data_page(out, unit, log, allowed):
    """build_qa_release1.main in place (module docstring). The installed files are content-tested in the builder's folder
    first (its preview links nights/ and dates/ point at this hub), and installed only when every test passes, both
    staged as .part before either is replaced. Returns (rc, text, {installed path: class}, builder folder, failures).
    Refuses S-G18 (a configured source absent) before the build and S-G19 (the page names another view) after it."""
    import importlib
    import importlib.util
    import sys
    from exopipe.phase3.core import validate
    from exopipe.phase3.core.record import Refusal
    from exopipe.phase3.lib.status import steps_inplace as IP
    C = K.cfg()
    P = C.status_pages
    src = data_page_sources(C)
    missing, reads = data_page_missing(src)
    if missing:
        raise Refusal(f'S-G18: {P.DATA_PAGE_BUILDER} would read these configured sources as empty: {missing} '
                      f'(status_pages.DATA_PAGE_SNAP_VIEW / DATA_PAGE_TENMIN_VIEW / DATA_PAGE_SNAP_CELL_LISTS / DATA_PAGE_DEC_PENDING)')
    for p in reads:
        unit.add_input(p)
    hub = K.place(out, 'review_hub', '').rstrip('/')
    site = K.place(out, 'status', P.DATA_PAGE_SITE)
    dests = {K.check_write(os.path.join(hub, n)): cls for n, cls in P.DATA_PAGE_INSTALL.items()}
    K.check_write(os.path.join(site, P.DATA_PAGE))
    os.makedirs(hub, exist_ok=True)
    env = IP.search_env(C.status.SEARCH_LABEL['deep'])
    IP.search_sources(C.status.SEARCH_LABEL['deep'])      # the register the 10-07 module imports, pinned (2026-10-15 review)
    sys.modules.pop(P.DATA_PAGE_HELPER, None)             # a fresh helper module: the builder imports this one (cached)
    try:
        with IP.redirect({}, allowed, env=env):           # the helper's import-time code runs under the write guard too
            B = importlib.import_module(P.DATA_PAGE_HELPER)
        unit.add_input(B.__file__, extra={'kind': 'code'})
        for k, v in dict(LEDGER=C.status.LEDGER, DD=C.paths.DATA_DECISIONS, ROOT=C.paths.COADD_SB_ROOT,
                         QA=C.status.QA_DIR).items():
            setattr(B, k, v)
        with IP.redirect({}, allowed, env=env):           # the builder's file on the legacy path (found, not imported)
            spec = importlib.util.find_spec(P.DATA_PAGE_BUILDER)
        if spec is None or not spec.origin or not os.path.isfile(spec.origin):
            raise Refusal(f'S-G18: the builder {P.DATA_PAGE_BUILDER} is not found on the legacy path (find_spec: {spec})')
        attrs = dict(OUT=site, HUB=hub, A=C.paths.PHASE3_ARCHIVE_ROOT, GEN=C.paths.ARCHIVE_RUN, OPS=C.paths.OPS_ROOT,
                     P1=C.paths.PHASE1_ROOT, P2=C.paths.PHASE2_ROOT, SNAPV=src['snap_view'], TENV=src['tenmin_view'],
                     TENMIN_VIEW=src['tenmin_view'], DEC=src['dec_pending'],
                     snap_cov=snap_cov_reader(spec.origin, snap_cell_lists(src)[0]),
                     main=builder_main(P.DATA_PAGE_BUILDER, spec.origin, os.path.basename(src['snap_view'].rstrip('/'))))
        rc, text, mod = IP.call_attr(P.DATA_PAGE_BUILDER, attrs, 'main', allowed_roots=allowed, log=log,
                                     unit=unit, env=env, argv=[P.DATA_PAGE_BUILDER + '.py', os.path.dirname(site)])
    finally:
        sys.modules.pop(P.DATA_PAGE_HELPER, None)
    if rc != 0:
        return rc, text, {}, site, []
    page = os.path.join(site, P.DATA_PAGE)
    wrong = page_run_mismatches(open(page).read(), src, C.paths.ARCHIVE_RUN) if os.path.exists(page) else []
    if wrong:
        raise Refusal(f'S-G19: {page} names a run other than the configured sources (a literal in the builder): '
                      + '; '.join(wrong) + ' - not installed')
    bad = [f'{n}: {r["name"]}: {str(r["detail"])[:200]}' for n, cls in P.DATA_PAGE_INSTALL.items()
           for r in validate.run(cls, os.path.join(site, n)) if not r['passed']]
    if bad:
        return rc, text, {}, site, bad
    for d in dests:
        shutil.copyfile(os.path.join(site, os.path.basename(d)), d + '.part')
    for d in dests:
        os.replace(d + '.part', d)
    return rc, text, dests, site, []


def run_register(out, audit_csvs, unit, log, allowed):
    from exopipe.phase3.lib.status import steps_inplace as IP
    C = K.cfg()
    reg = K.check_write(K.place(out, 'clv15', os.path.basename(C.status_report.REGISTER)))
    page = K.check_write(K.place(out, 'review_hub', C.status_pages.REGISTER_PAGE))
    for p in (reg, page):
        os.makedirs(os.path.dirname(p), exist_ok=True)
    for a in audit_csvs:
        unit.add_input(a)
    rc, text, mod = IP.call_attr('build_ratified_register', dict(REG=reg, PAGE=page), 'main', args=(list(audit_csvs),),
                                 allowed_roots=allowed, log=log, unit=unit)
    return rc, text, reg, page
