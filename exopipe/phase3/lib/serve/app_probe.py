#!/usr/bin/env python3
"""App probe - run as a SCRIPT (never imported by exopipe): the review server's OWN loader and App on the files one page
will serve, to know how many sources its index will list (served_page test sources_listed, deep and sub-LST pages).

    app_probe.py OPS_ROOT CLV15_ROOT RUN CATALOG_DIR QUEUE [--mchain-queue Q] [--mchain-catalog-dir D]
                                                                -> one JSON line on stdout

Run with the page's environment (classify_transients.py reads P3_QUEUE_BAND_ALIAS and the cross-tier variables at
import and in App.__init__), exactly as sublst_reserve_test_served.py:82-90 builds it: load_sources(run, catalog_dir),
read_queue(queue), App(run, sources, entries, prov, <tmp labels dir>, 'regen_panels7', entry_sel=sel). A page served with
its MULTIPLES CHAIN (exopipe_ops 4c42f91, Gregg 2026-10-27) is probed as classify_transients.main() loads it:
load_sources(..., extra_dir=<--mchain-catalog-dir>) and App.MCHAIN_RAW from read_queue(<--mchain-queue>) before App(), so
the chain identities its own queue does not list - they join the index after its own entries - are counted.
Output: queue_entries, queue_ids (first-presentation identities = the index's source links), dropped (queue ids the server
cannot resolve, 404), explained (dropped ids whose (J-name, band) row is in the view but hidden by a second row of the same
identity and band - the pre-existing case of sublst_reserve_test_served.py:106-110), unexplained.
Writes nothing but a temporary labels directory (removed).
"""
import csv
import glob
import importlib.util
import json
import os
import shutil
import sys
import tempfile


def main():
    ops, clv, run, cat, queue = sys.argv[1:6]
    opt = dict(zip(sys.argv[6::2], sys.argv[7::2]))
    bad = sorted(set(opt) - {'--mchain-queue', '--mchain-catalog-dir'})
    if bad or len(sys.argv[6:]) % 2:
        sys.exit(f'app_probe.py: unknown or incomplete options {sys.argv[6:]}')
    mq, mcat = opt.get('--mchain-queue'), opt.get('--mchain-catalog-dir')
    for p in (clv, ops):
        if p not in sys.path:
            sys.path.insert(0, p)
    spec = importlib.util.spec_from_file_location('ct', os.path.join(ops, 'classify_transients.py'))
    CT = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(CT)
    run = os.path.realpath(run)
    # classify_transients.main(): the multiples chain's own view joins the loader, its queue is read before App()
    src = CT.load_sources(run, catalog_dir=cat, extra_dir=mcat) if mcat else CT.load_sources(run, catalog_dir=cat)
    e, prov, sel = CT.read_queue(queue)
    if mq:
        m_entries, _m_prov, m_sel = CT.read_queue(mq)
        CT.App.MCHAIN_RAW = [(s, p, m_sel.get((s, p), '')) for s, p in m_entries]
    tmp = tempfile.mkdtemp(prefix='exopipe3_app_probe_')
    try:
        app = CT.App(run, src, e, prov, tmp, 'regen_panels7', entry_sel=sel)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    own = set()
    for sid, s in src.items():
        for b_, row in s.bands.items():
            own.add((row.get('source_id', sid), b_))
    vkeys = set()
    for p in glob.glob(os.path.join(cat, '*_sources_survivors.csv')):
        with open(p, newline='') as f:
            for r in csv.DictReader(f):
                vkeys.add((r['source_id'], r['band']))
    drop = [d[0] for d in app.dropped]

    def key(i):
        return (i.split('_')[0], i.rsplit('_', 1)[-1])
    expl = [i for i in drop if key(i) in vkeys and key(i) not in own]
    unexpl = [i for i in drop if i not in expl]
    mchain = getattr(CT.App, 'MCHAIN_IDS', None)
    print(json.dumps({'n': len(app.queue_ids), 'queue_entries': len(e), 'queue_ids': len(app.queue_ids),
                      'sources_loaded': len(src), 'dropped': len(drop), 'explained': expl[:50], 'n_explained': len(expl),
                      'unexplained': unexpl[:50], 'n_unexplained': len(unexpl),
                      'mchain_ids': None if mchain is None else len(mchain)}))


if __name__ == '__main__':
    main()
