#!/usr/bin/env python3
"""PARITY of the two-port server against the review servers running today (read-only; run on the review host).

    NUMPY_MADVISE_HUGEPAGE=0 nice taskset -c $(bash /lustre/gh/main/exopipe_ops/node_free_cores.sh) \
        /opt/devel/pipeline/envs/py38_orca_nkosogor/bin/python3 tests/parity_two_port_live.py REGISTRY OUT_JSON [PAGE_ID ...]

For each page (all review entries of the registry when none is named), its responses - from the RUNNING two-port server
when FRONT=<review port>,<inspection port> is set (the deployed path: front, one backend per code tree), else built in this
process exactly as a backend builds them (two_port.Store; one code tree at a time only) - are compared with the live
server's on the page's legacy port:
  - the index '/', up to SRC source pages and BANDS band pages linked from it, '/metrics.json', the first panel image;
  - for a greenlit page, the same through the inspection listener against the live inspection twin.
Equal means: the new response is byte-identical to two_port.rewrite_body(live response) for HTML, to the live bytes for
JSON and images. No request writes anything (GET only). Results per page, with the first difference, go to OUT_JSON.
With FRONT, also: GET /_status after the run - every built page's code and its review_filters module must have the md5 of the
file its own server runs: for /lustre/gh/main/exopipe_ops, that repository's HEAD; for a frozen tree, the file in the tree.
"""
import hashlib
import subprocess
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.error
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from exopipe.phase3.lib.serve import two_port as TP   # noqa: E402

SRC, BANDS, TIMEOUT = 12, 2, 300


def live_get(port, path):
    with urllib.request.urlopen(f'http://127.0.0.1:{port}{path}', timeout=TIMEOUT) as r:
        return r.status, r.headers.get('Content-Type', ''), r.read()


FRONT = tuple(int(x) for x in os.environ.get('FRONT', '').split(',') if x)


def new_get(store, inspect, path):
    """(status, body) of the new server: the running front when FRONT is set, else this process's Store."""
    if FRONT:
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{FRONT[1 if inspect else 0]}{path}', timeout=TIMEOUT) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()
    ns, nh, nb = TP.serve_request(store, inspect, 'GET', path, {})
    return ns, nb


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
    if not FRONT:
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
        ns, nb = new_get(store, inspect, base + path)
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


def head_md5(path):
    """md5 of a file as its own server runs it: HEAD of /lustre/gh/main/exopipe_ops for that repository, else the file."""
    repo = '/lustre/gh/main/exopipe_ops'
    if os.path.dirname(path) == repo:
        return hashlib.md5(subprocess.check_output(['git', '-C', repo, 'show', 'HEAD:' + os.path.basename(path)])).hexdigest()
    return TP.md5_file(path)


def code_check():
    """[(page, what, served md5, expected md5)] of every mismatch in the front's /_status, and the number checked."""
    import json as _j
    with urllib.request.urlopen(f'http://127.0.0.1:{FRONT[0]}/_status', timeout=TIMEOUT) as r:
        rep = _j.loads(r.read())
    bad, n = [], 0
    for b in rep['backends']:
        for page, r_ in (b.get('pages') or {}).items():
            if not r_:
                continue
            items = [('code', r_['code'])] + [(k, v) for k, v in r_['helpers'].items() if k == 'review_filters']
            for what, (path, md5) in items:
                n += 1
                want = head_md5(path)
                if md5 != want:
                    bad.append((page, what, path, md5, want))
    return bad, n


def main():
    reg, out = sys.argv[1], sys.argv[2]
    want = sys.argv[3:]
    store = TP.Store(reg, max_pages=1000, builds=not FRONT)
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
    bad = []
    if FRONT:
        bad, n = code_check()
        print(f'CODE {n - len(bad)}/{n} page code and review_filters md5 equal to the file each page\'s own server runs')
        for x in bad:
            print('  CODE MISMATCH', x)
        json.dump({'pages': results, 'code_mismatches': bad}, open(out, 'w'), indent=1)
    return 0 if eq == tot and not bad and not any(r.get('build_error') for r in results) else 1


if __name__ == '__main__':
    sys.exit(main())
