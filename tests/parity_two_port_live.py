#!/usr/bin/env python3
"""PARITY of the two-port server against the review servers running today (read-only; run on the review host).

    NUMPY_MADVISE_HUGEPAGE=0 nice taskset -c $(bash /lustre/gh/main/exopipe_ops/node_free_cores.sh) \
        /opt/devel/pipeline/envs/py38_orca_nkosogor/bin/python3 tests/parity_two_port_live.py REGISTRY OUT_JSON [PAGE_ID ...]

For each page (all review entries of the registry when none is named): the page is built in this process exactly as the
server builds it (two_port.Store), and its responses are compared with the live server's on the page's legacy port:
  - the index '/', up to SRC source pages and BANDS band pages linked from it, '/metrics.json', the first panel image;
  - for a greenlit page, the same through the inspection listener against the live inspection twin.
Equal means: the new response is byte-identical to two_port.rewrite_body(live response) for HTML, to the live bytes for
JSON and images. No request writes anything (GET only). Results per page, with the first difference, go to OUT_JSON.
"""
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from exopipe.phase3.lib.serve import two_port as TP   # noqa: E402

SRC, BANDS, TIMEOUT = 12, 2, 300


def live_get(port, path):
    with urllib.request.urlopen(f'http://127.0.0.1:{port}{path}', timeout=TIMEOUT) as r:
        return r.status, r.headers.get('Content-Type', ''), r.read()


def first_diff(a, b, ctx=80):
    n = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), min(len(a), len(b)))
    return {'at': n, 'live_rewritten': a[max(0, n - ctx):n + ctx], 'new': b[max(0, n - ctx):n + ctx],
            'len_live': len(a), 'len_new': len(b)}


def compare_page(store, e, inspect):
    pid = e['id']
    port = int(e.get('inspection_legacy_port') if inspect else e['legacy_port'])
    base, pm = f'/p/{pid}', store.port_map()
    res = {'page': pid, 'inspect': inspect, 'port': port, 'paths': 0, 'equal': 0, 'diffs': []}
    t0 = time.time()
    try:
        store.get(pid, inspect)
    except Exception as exc:
        res['build_error'] = str(exc)
        return res
    res['build_s'] = round(time.time() - t0, 1)

    def one(path, kind):
        try:
            ls, lct, lb = live_get(port, path)
        except Exception as exc:
            res['diffs'].append({'path': path, 'live_error': str(exc)})
            return None
        ns, nh, nb = TP.serve_request(store, inspect, 'GET', base + path, {})
        res['paths'] += 1
        if kind == 'html':
            want = TP.rewrite_body(lb.decode('utf-8'), base, pm, legacy_port=port).encode('utf-8')
        else:
            want = lb
        if ns == ls and nb == want:
            res['equal'] += 1
        else:
            d = first_diff(want.decode('utf-8', 'replace'), nb.decode('utf-8', 'replace')) if kind != 'bin' else \
                {'len_live': len(want), 'len_new': len(nb)}
            res['diffs'].append(dict(path=path, status=(ls, ns), **d))
        return lb
    idx = one('/', 'html')
    if idx is None:
        return res
    text = idx.decode('utf-8', 'replace')
    srcs = list(dict.fromkeys(re.findall(r'href="(/source/[^"]+)"', text)))[:SRC]
    bands = list(dict.fromkeys(re.findall(r'href="(/band/[^"]+)"', text)))[:BANDS]
    img = None
    for p in srcs:
        b = one(html_unescape(p), 'html')
        if b is not None and img is None:
            m = re.search(r'src="(/img\?f=[^"]+)"', b.decode('utf-8', 'replace'))
            img = m and html_unescape(m.group(1))
    for p in bands:
        one(html_unescape(p), 'html')
    one('/metrics.json', 'json')
    if img:
        one(img, 'bin')
    res['s'] = round(time.time() - t0, 1)
    return res


def html_unescape(s):
    import html
    return html.unescape(s)


def main():
    reg, out = sys.argv[1], sys.argv[2]
    want = sys.argv[3:]
    store = TP.Store(reg, max_pages=1000)
    ids = want or [e['id'] for e in store.reg.values() if e['kind'] == 'review']
    results = []
    for pid in ids:
        e = store.entry(pid)
        for inspect in ((False, True) if e.get('greenlit') and e.get('inspection_legacy_port') else (False,)):
            r = compare_page(store, e, inspect)
            results.append(r)
            print(f"{pid}{' (inspection)' if inspect else ''}: {r['equal']}/{r['paths']} equal"
                  + (f", build {r.get('build_s')} s" if 'build_s' in r else '') + (f" BUILD ERROR {r['build_error']}" if r.get('build_error') else '')
                  + (f"; first diff {r['diffs'][0].get('path')}" if r['diffs'] else ''), flush=True)
            json.dump(results, open(out, 'w'), indent=1)
            store.pages.pop((pid, inspect), None)             # one page in memory at a time
    tot = sum(r['paths'] for r in results)
    eq = sum(r['equal'] for r in results)
    print(f'TOTAL {eq}/{tot} responses equal over {len(results)} page-listener pairs')
    return 0 if eq == tot and not any(r.get('build_error') for r in results) else 1


if __name__ == '__main__':
    sys.exit(main())
