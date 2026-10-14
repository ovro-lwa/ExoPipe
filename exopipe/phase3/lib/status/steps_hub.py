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
             data-processing: ops/build_data_processing_page.py (OUT = <out>/review_hub/data_processing.html; its literal
             fragment path redirected to <out>/review_hub/qa_overview_fragment.html).
register     ops/build_ratified_register.py AUDIT_CSV ...: REG = <out>/clv15/RATIFIED_STEPS_REGISTER.csv, PAGE =
             <out>/review_hub/ratified_steps.html.
"""
import os
import shutil

from exopipe.phase3.lib.status import steps_common as K

HISTORY_COLUMNS = ('checked_utc', 'cells', 'pass', 'fail', 'fail_not_in_rebuild_list')   # conformance_check.py:47-48


def membership_drift():
    """S-G6: lines where the branch membership module differs from production other than import lines and the
    NAMED_OUT_OF_MASTER path line; [] when only those differ (and the named table files are identical)."""
    import difflib
    import hashlib
    from exopipe.phase3.lib.masters import membership as BM
    prod = K.cfg().status_conformance.PRODUCTION_MEMBERSHIP
    if not os.path.exists(prod):
        return [f'production module {prod} not found']
    a = open(prod).read().splitlines()
    b = open(BM.__file__).read().splitlines()
    bad = []
    for ln in difflib.unified_diff(a, b, lineterm='', n=0):
        if ln.startswith(('---', '+++', '@@')):
            continue
        body = ln[1:].strip()
        if body.startswith(('import ', 'from ')) or body.startswith('NAMED_OUT_OF_MASTER ='):
            continue
        bad.append(ln[:160])
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
    rc, text, mod = IP.call_attr('build_qa_pages', attrs, 'main', allowed_roots=allowed, log=log, unit=unit,
                                 env={'P3S_LABEL': C.status.SEARCH_LABEL['deep']}, argv=argv)
    return rc, text, hub.rstrip('/')


def run_data_page(out, unit, log, allowed):
    from exopipe.phase3.lib.status import steps_inplace as IP
    C = K.cfg()
    P = C.status_pages
    dest = K.check_write(K.place(out, 'review_hub', P.DATA_PAGE))
    frag_prod = os.path.join(K.hub_dir(), P.FRAGMENT)
    frag_out = K.place(out, 'review_hub', P.FRAGMENT)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    attrs = dict(OUT=dest, LEDGER=C.status.LEDGER, DD=C.paths.DATA_DECISIONS, ROOT=C.paths.COADD_SB_ROOT, QA=C.status.QA_DIR)
    rc, text, mod = IP.call_attr('build_data_processing_page', attrs, 'main', mapping={frag_prod: frag_out},
                                 allowed_roots=allowed, log=log, unit=unit,
                                 env={'P3S_LABEL': C.status.SEARCH_LABEL['deep']})
    return rc, text, dest


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
