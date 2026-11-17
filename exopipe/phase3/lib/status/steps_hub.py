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


def run_data_page(out, unit, log, allowed):
    """build_qa_release1.main in place (module docstring). Returns (rc, text, installed paths, builder folder)."""
    import sys
    from exopipe.phase3.lib.status import steps_inplace as IP
    C = K.cfg()
    P = C.status_pages
    hub = K.place(out, 'review_hub', '').rstrip('/')
    site = K.place(out, 'status', P.DATA_PAGE_SITE)
    dests = [K.check_write(os.path.join(hub, n)) for n in P.DATA_PAGE_INSTALL]
    K.check_write(os.path.join(site, P.DATA_PAGE))
    os.makedirs(hub, exist_ok=True)
    env = IP.search_env(C.status.SEARCH_LABEL['deep'])
    IP.search_sources(C.status.SEARCH_LABEL['deep'])      # the register the 10-07 module imports, pinned (2026-10-15 review)
    sys.modules.pop(P.DATA_PAGE_HELPER, None)             # a fresh helper module: the builder imports this one (cached)
    B = IP.import_quiet(P.DATA_PAGE_HELPER, env=env)
    unit.add_input(B.__file__, extra={'kind': 'code'})
    for k, v in dict(LEDGER=C.status.LEDGER, DD=C.paths.DATA_DECISIONS, ROOT=C.paths.COADD_SB_ROOT,
                     QA=C.status.QA_DIR).items():
        setattr(B, k, v)
    try:
        rc, text, mod = IP.call_attr(P.DATA_PAGE_BUILDER, dict(OUT=site, HUB=hub), 'main', allowed_roots=allowed, log=log,
                                     unit=unit, env=env, argv=[P.DATA_PAGE_BUILDER + '.py', os.path.dirname(site)])
    finally:
        sys.modules.pop(P.DATA_PAGE_HELPER, None)
    if rc != 0:
        return rc, text, [], site
    for n, dest in zip(P.DATA_PAGE_INSTALL, dests):
        shutil.copyfile(os.path.join(site, n), dest + '.part')
        os.replace(dest + '.part', dest)
    return rc, text, dests, site


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
