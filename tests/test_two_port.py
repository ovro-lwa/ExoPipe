#!/usr/bin/env python3
"""Tests of the two-port review server (lib/serve/two_port.py): plain asserts, runnable with python, under a minute.

    NUMPY_MADVISE_HUGEPAGE=0 /opt/devel/pipeline/envs/py38_orca_nkosogor/bin/python3 tests/test_two_port.py

1. ThreadEnv: a thread's page environment is its own; threads it starts inherit it; the process environment is untouched.
2. Rewriting: the page's routes and index gain /p/<id>; registered legacy-port URLs become page paths, others stay; the hub
   URLs become '/'; '/imgs/...' and file paths are left alone; the filters script is keyed by the legacy port, and a filters
   script the rewrite cannot key is refused.
3. The registry refuses malformed entries, duplicate ids, duplicate legacy ports and argv carrying --port / --inspect.
4. Two synthetic pages (one module file, written to /tmp, shaped like classify_transients.py: module globals, an App,
   a BaseHTTPRequestHandler subclass, main() with argparse and ThreadingHTTPServer) served by one Store: each page renders its
   own title and its own environment (read at import and at request time), also from concurrent threads; a POST writes the
   page's journal and its redirect is rewritten; the inspection listener renders the read-only page, refuses every POST and
   hides pages that are not greenlit; static sites are served with their port links rewritten; a page that fails to build
   gets an error page and the other pages keep serving.
5. Code trees (the 2026-10-18 defect: one process shared helper modules by name across trees): two trees whose page code and
   helper module share their names but not their content, served by the front and one backend process per tree - each page
   renders its own tree's helper whichever tree is built first; /_status reports each backend's tree and the md5 of the code
   and helper each page imported; a POST is forwarded and written; the inspection listener refuses writes in the front; a
   dead backend is restarted on the next request; a tree that leaves the registry has its backend stopped.
6. Rebuilds (WEB SPEED step 2, 2026-10-30): at a registry change a page whose own entry is unchanged keeps answering requests
   from the build before (with the stale header) until its background rebuild - indices included (App.prime) - is done; a fresh
   GET waits for it; a POST goes to the page that showed its form, and labels keep their order; a page whose own entry changed,
   or that left the registry, is dropped at once; a new page requested after a registry change - also a registration's second
   new page - is not queued behind a stale rebuild; a registry change during a rebuild rebuilds again; a build whose entry
   changed while it ran goes only to the request that asked; a failed rebuild drops the old page, leaves no module behind, and
   the next request shows the failure.
"""
import json
import os
import shutil
import sys
import threading
import urllib.parse

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
TMP = '/tmp/exopipe3_tests_two_port'

from exopipe.phase3.lib.serve import two_port as TP   # noqa: E402

FAILS, N = [], [0]


def check(what, ok, detail=''):
    N[0] += 1
    if not ok:
        FAILS.append(f'{what}: {detail}')
        print('FAIL ', what, detail)


PAGE = r'''
import argparse, html, json, os, time, urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
INSPECT = False
FLAG_AT_IMPORT = os.environ.get('P3_TEST_FLAG', 'unset')
_SP = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stamp.txt')
STAMP = open(_SP).read().strip() if os.path.exists(_SP) else ''
HUB_URL = 'http://localhost:8765/'

def hub_url():
    return 'http://localhost:8795/' if INSPECT else HUB_URL

class App:
    TITLE = ''
    PRIMED = 'no'
    def __init__(self, labels):
        self.labels = labels
    def prime(self):
        App.PRIMED = os.environ.get('P3_TEST_FLAG', 'unset')
    def page_index(self):
        return (f'<html><body><h1>{html.escape(self.TITLE)}</h1><a href="/">index</a> <a href="{hub_url()}">hub</a> '
                f'<a href="/source/J1">J1</a> <img src="/img?f=a.png"> <a href="/imgs/x.png">file</a> '
                f'<a href="http://localhost:9901/source/J1">sibling</a> <a href="http://localhost:9999/">unregistered</a> '
                f'flag-at-import={FLAG_AT_IMPORT} flag-now={os.environ.get("P3_TEST_FLAG", "unset")} primed={App.PRIMED} stamp={STAMP} '
                f'<script>var PORT = location.port || \'80\', LS = \'filters_v1\', COOK = \'flt_\' + PORT; '
                f'fetch(\'/metrics.json\');</script></body></html>')
    def page_source(self, sid):
        form = '' if INSPECT else '<form method="POST" action="/label"><input name="source_id" value="J1"></form>'
        return f'<html><body>{sid} {form}</body></html>'
    def handle_label(self, form):
        os.makedirs(self.labels, exist_ok=True)
        with open(os.path.join(self.labels, 'judgements.jsonl'), 'a') as f:
            f.write(json.dumps(dict({k: v[0] for k, v in form.items()}, title=self.TITLE, stamp=STAMP)) + '\n')
        return '/source/' + urllib.parse.quote(form.get('source_id', [''])[0])

class Handler(BaseHTTPRequestHandler):
    app = None
    def _send(self, body, ctype='text/html; charset=utf-8', code=200):
        self.send_response(code); self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        p = urllib.parse.urlparse(self.path)
        if p.path == '/':
            self._send(self.app.page_index().encode())
        elif p.path.startswith('/source/'):
            self._send(self.app.page_source(urllib.parse.unquote(p.path[8:])).encode())
        elif p.path == '/port':
            self._send(str(self.server.server_port).encode(), 'text/plain')
        else:
            self._send(b'not found', 'text/plain', 404)
    def do_POST(self):
        if INSPECT:
            self.send_response(403); self.send_header('Content-Type', 'text/plain'); self.end_headers()
            self.wfile.write(b'read-only'); return
        n = int(self.headers.get('Content-Length', 0))
        form = urllib.parse.parse_qs(self.rfile.read(n).decode())
        dest = self.app.handle_label(form)
        self.send_response(303); self.send_header('Location', dest); self.end_headers()
    def log_message(self, *a):
        pass

def main():
    global INSPECT
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True); ap.add_argument('--labels-dir'); ap.add_argument('--title', default='')
    ap.add_argument('--port', type=int, default=1); ap.add_argument('--inspect', action='store_true')
    ap.add_argument('--fail', action='store_true'); ap.add_argument('--slow', type=float, default=0)
    a = ap.parse_args()
    if a.fail:
        raise SystemExit('REFUSING: this page cannot start')
    time.sleep(a.slow)
    INSPECT = a.inspect
    App.TITLE = a.title
    Handler.app = App(a.labels_dir or os.path.join(a.run, 'judgements'))
    ThreadingHTTPServer(('127.0.0.1', a.port), Handler).serve_forever()
'''


def test_env():
    TP.install_env()
    env = TP._ENV
    base_before = os.environ.get('P3_TEST_FLAG')
    seen = {}

    def worker(name, val):
        with env.use({'P3_TEST_FLAG': val}):
            got = os.environ.get('P3_TEST_FLAG')
            inner = {}
            t = threading.Thread(target=lambda: inner.setdefault('v', os.environ.get('P3_TEST_FLAG')))
            t.start(); t.join()
            seen[name] = (got, inner.get('v'))
    ts = [threading.Thread(target=worker, args=(f'w{i}', f'v{i}')) for i in range(6)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    check('ThreadEnv: every thread reads its own page environment',
          all(seen[f'w{i}'][0] == f'v{i}' for i in range(6)), seen)
    check('ThreadEnv: a thread started inside a page environment inherits it',
          all(seen[f'w{i}'][1] == f'v{i}' for i in range(6)), seen)
    check('ThreadEnv: the process environment is untouched', os.environ.get('P3_TEST_FLAG') == base_before)


def test_rewrite():
    pm = {9901: 'sib', 8766: 'deep_V'}
    src = ('<a href="/">i</a> <a href=\'/\'>i2</a> <a href="/source/J1?b=Red">s</a> <img src="/img?f=a.png"> '
           '<form action="/label"></form> <form action="/list"></form> <a href="/band/xm:25pc">b</a> '
           '<a href="/imgs/x.png">f</a> <a href="/data07/gh/x.png">p</a> <a href="http://localhost:9901/source/J2">sib</a> '
           '<a href="http://localhost:9999/">x</a> <a href="http://localhost:8765/">hub</a> fetch(\'/metrics.json\')')
    out = TP.rewrite_body(src, '/p/me', pm)
    for want in ('href="/p/me/"', "href='/p/me/'", 'href="/p/me/source/J1?b=Red"', 'src="/p/me/img?f=a.png"',
                 'action="/p/me/label"', 'action="/p/me/list"', 'href="/p/me/band/xm:25pc"', 'href="/imgs/x.png"',
                 'href="/data07/gh/x.png"', 'href="/p/sib/source/J2"', 'href="http://localhost:9999/"', 'href="/"',
                 "fetch('/p/me/metrics.json')"):
        check(f'rewrite_body: {want}', want in out, out)
    js = "var PORT = location.port || '80', LS = 'filters_v1', COOK = 'flt_' + PORT;"
    o2 = TP.rewrite_body(js, '/p/me', pm, legacy_port=8766)
    check('rewrite_body: the filters script is keyed by the legacy port',
          "var PORT = '8766', LS = 'filters_v1_8766', COOK = 'flt_' + PORT;" in o2, o2)
    try:
        TP.rewrite_body("COOK = 'flt_' + location.host", '/p/me', pm, legacy_port=8766)
        refused = False
    except TP.RewriteError:
        refused = True
    check('rewrite_body: a filters script it cannot key is refused', refused)
    check('rewrite_location: a path gains the base', TP.rewrite_location('/source/J1', '/p/me', pm) == '/p/me/source/J1')
    check('rewrite_links: a static page keeps its own links, port URLs become page paths',
          TP.rewrite_links('<a href="x.html">x</a> <a href="http://localhost:8766/">d</a>', pm)
          == '<a href="x.html">x</a> <a href="/p/deep_V/">d</a>')


def write_registry(path, pages):
    with open(path, 'w') as f:
        json.dump({'pages': pages}, f)


def test_registry():
    p = os.path.join(TMP, 'bad.json')
    code = os.path.join(TMP, 'page.py')
    cases = {
        'missing kind': [{'id': 'a'}],
        'bad id': [{'id': 'a b', 'kind': 'review', 'code': code, 'argv': ['--run', TMP]}],
        'twice': [{'id': 'a', 'kind': 'review', 'code': code, 'argv': ['--run', TMP]}] * 2,
        'port twice': [{'id': 'a', 'kind': 'review', 'code': code, 'argv': ['--run', TMP], 'legacy_port': 1},
                       {'id': 'b', 'kind': 'review', 'code': code, 'argv': ['--run', TMP], 'legacy_port': 1}],
        'argv --port': [{'id': 'a', 'kind': 'review', 'code': code, 'argv': ['--run', TMP, '--port', '5']}],
        'static not a dir': [{'id': 's', 'kind': 'static', 'root': os.path.join(TMP, 'nope')}],
    }
    for what, pages in cases.items():
        write_registry(p, pages)
        try:
            TP.load_registry(p)
            ok = False
        except TP.RegistryError:
            ok = True
        check(f'registry refuses: {what}', ok)


def test_pages():
    code = os.path.join(TMP, 'page.py')
    open(code, 'w').write(PAGE)
    run_a, run_b = os.path.join(TMP, 'run_a'), os.path.join(TMP, 'run_b')
    for d in (run_a, run_b):
        os.makedirs(d)
    site = os.path.join(TMP, 'site')
    os.makedirs(site)
    open(os.path.join(site, 'index.html'), 'w').write('<a href="http://localhost:9901/">a</a> <a href="q.html">q</a>')
    reg = os.path.join(TMP, 'registry.json')
    write_registry(reg, [
        {'id': 'pa', 'kind': 'review', 'code': code, 'argv': ['--run', run_a, '--title', 'Page A'],
         'env': {'P3_TEST_FLAG': 'A'}, 'legacy_port': 9901, 'greenlit': True},
        {'id': 'pb', 'kind': 'review', 'code': code, 'argv': ['--run', run_b, '--title', 'Page B',
                                                              '--labels-dir', os.path.join(run_a, 'judgements')],
         'env': {'P3_TEST_FLAG': 'B'}, 'legacy_port': 9902, 'greenlit': False},
        {'id': 'broken', 'kind': 'review', 'code': code, 'argv': ['--run', run_a, '--fail'], 'legacy_port': 9903},
        {'id': 'site', 'kind': 'static', 'root': site, 'greenlit': True},
    ])
    st = TP.Store(reg)
    st.REBUILD_GRACE_S = 0.2                         # production waits 30 s after a registry change before rebuilding ...
    st.REBUILD_QUIET_S = 0.0                         # ... and 10 s after the last build a request waited for
    res = {}

    def get(pid, path, inspect=False, key=None):
        r = TP.serve_request(st, inspect, 'GET', f'/p/{pid}{path}', {})
        if key:
            res[key] = r
        return r
    ts = [threading.Thread(target=get, args=(pid, '/', False, f'{pid}{i}')) for i in range(4) for pid in ('pa', 'pb')]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    ba = res['pa0'][2].decode()
    bb = res['pb0'][2].decode()
    check('page A renders its own title and environment (import and request time)',
          'Page A' in ba and 'flag-at-import=A' in ba and 'flag-now=A' in ba, ba[:200])
    check('page B renders its own title and environment', 'Page B' in bb and 'flag-at-import=B' in bb and 'flag-now=B' in bb,
          bb[:200])
    check('concurrent first requests all get the same page', all(res[f'pa{i}'][2] == res['pa0'][2] for i in range(4))
          and all(res[f'pb{i}'][2] == res['pb0'][2] for i in range(4)))
    check('page A: links rewritten (index, source, img, sibling, hub), file links and unregistered ports left',
          'href="/p/pa/"' in ba and 'href="/p/pa/source/J1"' in ba and 'src="/p/pa/img?f=a.png"' in ba
          and 'href="/p/pa/source/J1">sibling' in ba and 'href="/"' in ba and 'href="/imgs/x.png"' in ba
          and 'href="http://localhost:9999/"' in ba, ba)
    check('page A: filters keyed by its legacy port', "var PORT = '9901', LS = 'filters_v1_9901'" in ba, ba)
    s, h, b = get('pa', '/port')
    check("the page's Handler sees its legacy port as server_port", s == 200 and b == b'9901', (s, b))
    s, h, b = TP.serve_request(st, False, 'POST', '/p/pb/label', {'Content-Type': 'application/x-www-form-urlencoded'},
                               b'source_id=J7&confidence=FP')
    loc = dict(h).get('Location')
    jr = os.path.join(run_a, 'judgements', 'judgements.jsonl')
    check('POST: the page writes its journal (its own labels dir) and its redirect is rewritten',
          s == 303 and loc == '/p/pb/source/J7' and os.path.exists(jr) and 'J7' in open(jr).read(), (s, loc))
    check('pages sharing a labels dir share one write lock', st.pages[('pb', False)].write_lock is st.write_locks[
        os.path.realpath(os.path.join(run_a, 'judgements'))])
    s, h, b = get('pa', '/source/J1', inspect=True)
    check('inspection: the read-only page (no form)', s == 200 and b'<form' not in b and b'J1' in b, b)
    s, h, b = get('pa', '/source/J1')
    check('review: the page with its form, action rewritten', b'action="/p/pa/label"' in b, b)
    s, h, b = TP.serve_request(st, True, 'POST', '/p/pa/label', {}, b'source_id=J1')
    check('inspection: a judging POST refused', s == 403, s)
    # Gregg 2026-10-27: the per-person review lists are written from the inspection listener as well - POST /list, /list/user only
    for path, ok in (('/p/pa/list', True), ('/p/pa/list/user', True), ('/p/pa/list?x=1', True), ('/p/pa/listx', False),
                     ('/p/pa/list/user/x', False), ('/p/pa/label', False), ('/list', False)):
        check('inspection_write_ok POST %s -> %s' % (path, ok), TP.inspection_write_ok('POST', path) is ok)
    check('inspection_write_ok: a GET is not a write', TP.inspection_write_ok('GET', '/p/pa/list') is False)
    refusal = b'inspection mode: this view is read-only and records nothing'      # the server's own refusal
    s, h, b = TP.serve_request(st, True, 'POST', '/p/pa/list', {}, b'source_id=J1')
    check('inspection: a list POST reaches the page (the server does not refuse it; this test page then does)',
          b != refusal and b == b'read-only', (s, b[:80]))
    s, h, b = TP.serve_request(st, True, 'POST', '/p/pa/list/user', {}, b'name=Alice')
    check('inspection: a new-user POST reaches the page', b != refusal and b == b'read-only', (s, b[:80]))
    s, h, b = TP.serve_request(st, True, 'POST', '/p/pa/label', {}, b'source_id=J1')
    check('inspection: ... while a judging POST gets the server refusal', s == 403 and b == refusal, (s, b[:80]))
    s, h, b = get('pb', '/', inspect=True)
    check('inspection: a page not greenlit is not served', s == 404, s)
    s, h, b = get('broken', '/')
    check('a page that fails to build gets an error page', s == 500 and b'could not be built' in b, (s, b[:120]))
    s, h, b = get('pa', '/')
    check('... and the other pages keep serving', s == 200 and b'Page A' in b)
    s, h, b = TP.serve_request(st, False, 'GET', '/s/site/', {})
    check('a static site, its port links rewritten', s == 200 and b'href="/p/pa/"' in b and b'href="q.html"' in b, b)
    s, h, b = TP.serve_request(st, False, 'GET', '/s/site/../../etc/passwd', {})
    check('a static site never serves outside its root', s == 404, s)
    s, h, b = TP.serve_request(st, False, 'GET', '/', {})
    check('the hub lists the registry', s == 200 and b'/p/pa/' in b and b'/s/site/' in b, b[:200])
    hubdir = os.path.join(TMP, 'hubsite')
    os.makedirs(os.path.join(hubdir, 'nights'))
    open(os.path.join(hubdir, 'index.html'), 'w').write('<a href="data_processing.html">dp</a>')
    open(os.path.join(hubdir, 'data_processing.html'), 'w').write('<a href="http://localhost:9901/">a</a> report')
    open(os.path.join(hubdir, 'nights', 'n1.html'), 'w').write('night')
    reg_h = os.path.join(TMP, 'registry_hub.json')
    write_registry(reg_h, json.load(open(reg))['pages'] + [{'id': 'hub', 'kind': 'static', 'root': hubdir}])
    sth = TP.Store(reg_h)
    s, h, b = TP.serve_request(sth, False, 'GET', '/data_processing.html', {})
    check('a hub site file beside index.html is served, its port links rewritten', s == 200 and b'report' in b
          and b'href="/p/pa/"' in b, (s, b[:120]))
    s, h, b = TP.serve_request(sth, False, 'GET', '/nights/', {})
    check('a hub folder without an index is listed', s == 200 and b'n1.html' in b, (s, b[:120]))
    write_registry(reg_h, json.load(open(reg_h))['pages'] + [{'id': 'noidx', 'kind': 'static', 'root': os.path.join(hubdir, 'nights')}])
    sth = TP.Store(reg_h)
    s, h, b = TP.serve_request(sth, False, 'GET', '/s/noidx/', {})
    check('a static site whose root has no index is listed at its root', s == 200 and b'n1.html' in b, (s, b[:120]))
    s, h, b = TP.serve_request(sth, False, 'GET', '/../registry.json', {})
    check('the hub site never serves outside its root', s == 404, s)
    outside = os.path.join(TMP, 'outside_assets')            # a site's own symlink to a tree elsewhere (review_hub/qa_assets/*)
    os.makedirs(os.path.join(outside, 'qa'))
    open(os.path.join(outside, 'qa', 'img.png'), 'wb').write(b'PNGDATA')
    os.symlink(outside, os.path.join(hubdir, 'qa_assets'))
    s, h, b = TP.serve_request(sth, False, 'GET', '/qa_assets/qa/img.png', {})
    check('a symlink the site holds is followed (the data-processing QA images)', s == 200 and b == b'PNGDATA', (s, b[:40]))
    reg_b = os.path.join(TMP, 'registry_hub_both.json')     # production since 2026-10-27: both listeners serve the review hub
    write_registry(reg_b, json.load(open(reg_h))['pages'] + [{'id': 'hub_inspection', 'kind': 'static', 'root': hubdir,
                                                               'greenlit': True}])
    s, h, b = TP.serve_request(TP.Store(reg_b), True, 'GET', '/qa_assets/qa/img.png', {})
    check('... on the inspection listener as well (one hub site for both)', s == 200 and b == b'PNGDATA', (s, b[:40]))
    s, h, b = TP.serve_request(sth, False, 'GET', '/qa_assets/../../registry_hub.json', {})
    check('... but a path written to leave the site through it is refused', s == 404, s)
    s, h, b = TP.serve_request(sth, False, 'GET', '/qa_assets/qa/../../../registry_hub.json', {})
    check('... at any depth', s == 404, s)
    s, h, b = TP.serve_request(st, True, 'GET', '/', {})
    check('the inspection hub lists greenlit entries only', b'/p/pa/' in b and b'/p/pb/' not in b, b[:200])
    check('a page is primed (App.prime) under its own environment before it serves', 'primed=A' in ba and 'primed=B' in bb,
          (ba[:300], bb[:300]))
    fresh = {TP.FRESH_HEADER: '1'}

    def fget(pid, path='/'):
        return TP.serve_request(st, False, 'GET', f'/p/{pid}{path}', fresh)
    # a registry edit rebuilds the page; a request that asks for the fresh page waits for it
    reg_now = json.load(open(reg))
    reg_now['pages'][0]['argv'] = ['--run', run_a, '--title', 'Page A2']
    write_registry(reg, reg_now['pages'])
    os.utime(reg, (os.path.getmtime(reg) + 5,) * 2)
    s, h, b = fget('pa')
    check('a changed registry entry is rebuilt (a fresh request waits for it)', b'Page A2' in b and b'primed=A' in b, b[:120])
    settle(st)
    s, h, b = get('pa', '/')
    check('... and every request then gets the rebuilt page', b'Page A2' in b, b[:120])
    # a helper module beside the page code (as review_filters is) is reloaded at a registry change: new page code is deployed
    open(os.path.join(TMP, 'helper_mod.py'), 'w').write('V = 1\n')
    code_now = open(code).read().replace("FLAG_AT_IMPORT = os.environ.get('P3_TEST_FLAG', 'unset')",
                                         "FLAG_AT_IMPORT = os.environ.get('P3_TEST_FLAG', 'unset')\nimport helper_mod")
    code_now = code_now.replace("f'flag-at-import={FLAG_AT_IMPORT} ", "f'helper={helper_mod.V} flag-at-import={FLAG_AT_IMPORT} ")
    open(code, 'w').write(code_now)
    os.utime(reg, (os.path.getmtime(reg) + 10,) * 2)
    s, h, b = fget('pa')
    check('helper module loaded', b'helper=1' in b, b[:160])
    open(os.path.join(TMP, 'helper_mod.py'), 'w').write('V = 2  # a real edit changes the file\n')
    os.utime(reg, (os.path.getmtime(reg) + 10,) * 2)
    s, h, b = fget('pa')
    check('a registry change reloads the helper modules beside the page code (a deploy)', b'helper=2' in b, b[:160])
    settle(st)
    test_rebuilds(st, reg, run_a)


def settle(st, timeout=60):
    """Wait until the store's rebuild thread has nothing left to do."""
    import time
    t0 = time.time()
    while time.time() - t0 < timeout:
        with st.lock:
            idle = not st.rebuild_q and not st.building and (st._rebuilder is None or not st._rebuilder.is_alive())
        if idle:
            return True
        time.sleep(0.05)
    return False


def test_rebuilds(st, reg, run_a):
    """WEB SPEED step 2 (2026-10-30): at a registry change a page whose own entry is unchanged keeps answering GETs from the
    build before until its rebuild (with its indices) is done; a POST and a fresh request wait for the rebuilt page; a page
    whose own entry changed, or that left the registry, is dropped at once; a failed rebuild drops the old page so the next
    request shows the failure; a registry change during a rebuild rebuilds again; a build whose entry changed while it ran is
    given only to the request that asked for it."""
    import time
    pages = json.load(open(reg))['pages']
    pa = next(x for x in pages if x['id'] == 'pa')
    stamp = os.path.join(TMP, 'stamp.txt')

    def write(drop=()):
        write_registry(reg, [x for x in pages if x['id'] not in drop])
        os.utime(reg, (os.path.getmtime(reg) + 20 + time.time() % 7,) * 2)

    def touch():                                     # a registry change that leaves this page's entry as it is
        os.utime(reg, (os.path.getmtime(reg) + 20 + time.time() % 7,) * 2)

    def entry(title, extra=()):
        pa['argv'] = ['--run', run_a, '--title', title] + list(extra)

    def get(path='/', hdr=None):
        return TP.serve_request(st, False, 'GET', '/p/pa' + path, hdr or {})
    open(stamp, 'w').write('S1')
    entry('Page A3', ['--slow', '1.5'])
    write()
    t = time.time()
    s, h, b = get()
    dt = time.time() - t
    check('rebuild: a page whose own entry changed is dropped at once - its next request builds the new entry',
          s == 200 and b'Page A3' in b and b'stamp=S1' in b and dt > 1.2, (s, round(dt, 2), b[:80]))
    open(stamp, 'w').write('S2')
    touch()
    t = time.time()
    s, h, b = get()
    dt = time.time() - t
    check('rebuild: an unchanged entry: during its slow rebuild the page built before answers a GET at once',
          s == 200 and b'Page A3' in b and b'stamp=S1' in b and dt < 1.0, (s, round(dt, 2), b[:80]))
    check('rebuild: ... and says so (the stale header carries its build time)', TP.STALE_HEADER in dict(h), h)
    rep = TP.backend_status(st)
    check('rebuild: /_status reports the page stale and its rebuild under way',
          rep['built'].get('pa', {}).get('stale') is True and ('pa' in rep['building'] or 'pa' in rep['rebuild_queue']), rep['built'])
    t = time.time()
    form = {'Content-Type': 'application/x-www-form-urlencoded'}
    s, h, b = TP.serve_request(st, False, 'POST', '/p/pa/label', form, b'source_id=J8&confidence=FP')
    s1, h1, b1 = TP.serve_request(st, False, 'POST', '/p/pa/label', form, b'source_id=J8&confidence=real')
    dt = time.time() - t
    jl = [json.loads(x) for x in open(os.path.join(run_a, 'judgements', 'judgements.jsonl'))]
    check('rebuild: a POST is answered at once by the page that showed its form, and labels keep their order',
          s == 303 and s1 == 303 and dt < 1.0 and [x.get('confidence') for x in jl[-2:]] == ['FP', 'real']
          and jl[-1].get('stamp') == 'S1', (s, s1, round(dt, 2), jl[-2:]))
    check('rebuild: the rebuild thread finishes', settle(st))
    s2, h2, b2 = get()
    check('rebuild: ... and the next GET gets the rebuilt page', b'stamp=S2' in b2, b2[:80])
    rep = TP.backend_status(st)
    b_ = rep['built'].get('pa', {})
    check('rebuild: /_status then reports it fresh, with its build and index times', b_.get('stale') is False
          and b_.get('build_s') is not None and b_.get('prime_s') is not None, b_)
    s, h, b = get()
    check('rebuild: then every GET gets the rebuilt page (and its indices: primed)', b'stamp=S2' in b and b'primed=A' in b, b[:80])
    check('rebuild: ... without the stale header', TP.STALE_HEADER not in dict(h), h)
    s, h, b = get(hdr={TP.FRESH_HEADER: '1'})
    check('rebuild: a fresh request on a page already rebuilt answers at once', s == 200 and b'stamp=S2' in b, b[:80])
    # the rebuild thread waits REBUILD_GRACE_S after a registry change, and never starts a rebuild while a request's build is
    # pending: a new page requested meanwhile is not queued behind a stale page's slow rebuild
    st.REBUILD_GRACE_S = 1.0
    entry('Page A3', ['--slow', '1.5'])
    open(stamp, 'w').write('S2b')
    pages.append({'id': 'pc', 'kind': 'review', 'code': pa['code'], 'argv': ['--run', run_a, '--title', 'Page C', '--slow', '0.3'],
                  'legacy_port': 9904})
    write()
    t = time.time()
    s, h, b = TP.serve_request(st, False, 'GET', '/p/pa/', {})       # the stale page, at once; its rebuild queued
    s, h, b = TP.serve_request(st, False, 'GET', '/p/pc/', {})       # a new page: built now, not behind pa's 1.5 s rebuild
    dt = time.time() - t
    check('rebuild: a new page requested after a registry change builds first (grace, then stale rebuilds)',
          s == 200 and b'Page C' in b and dt < 1.2, (s, round(dt, 2), b[:60]))
    settle(st)
    # a registration of several pages: its warm-up builds them one after the other; past the grace, the rebuild thread still waits
    # REBUILD_QUIET_S after each, so the second new page is not queued behind the stale page's rebuild either
    st.REBUILD_GRACE_S, st.REBUILD_QUIET_S = 0.2, 1.0
    open(stamp, 'w').write('S2c')
    for pid_, port_ in (('pd', 9905), ('pe', 9906)):
        pages.append({'id': pid_, 'kind': 'review', 'code': pa['code'],
                      'argv': ['--run', run_a, '--title', 'Page ' + pid_, '--slow', '0.3'], 'legacy_port': port_})
    write()
    TP.serve_request(st, False, 'GET', '/p/pa/', {})                  # stale, queued
    time.sleep(0.4)                                                   # past the grace
    TP.serve_request(st, False, 'GET', '/p/pd/', {})                  # the warm-up's first new page
    time.sleep(0.3)
    t = time.time()
    s, h, b = TP.serve_request(st, False, 'GET', '/p/pe/', {})        # its second: not behind pa's 1.5 s rebuild
    dt = time.time() - t
    check('rebuild: a registration\'s second new page is not queued behind a stale rebuild (quiet window)',
          s == 200 and b'Page pe' in b and dt < 1.0, (s, round(dt, 2), b[:60]))
    settle(st)
    st.REBUILD_GRACE_S, st.REBUILD_QUIET_S = 0.2, 0.0
    for pid_ in ('pd', 'pe'):
        pages.remove(next(x for x in pages if x['id'] == pid_))
    pages.remove(next(x for x in pages if x['id'] == 'pc'))
    write()
    TP.serve_request(st, False, 'GET', '/p/pa/', {})
    settle(st)
    check('rebuild: a page that left the registry is dropped', ('pc', False) not in st.pages, list(st.pages))
    # a registry change while the page is being rebuilt: rebuilt again
    touch()
    get()
    time.sleep(0.3)
    open(stamp, 'w').write('S3')
    touch()
    get()
    settle(st)
    s, h, b = get()
    check('rebuild: a registry change during a rebuild rebuilds again', b'stamp=S3' in b, b[:80])
    # an entry changed while its page is being built: the request that asked gets that build, the next request the new entry
    entry('Page A8', ['--slow', '1.0'])
    write()
    res = {}
    th = threading.Thread(target=lambda: res.setdefault('r', get(hdr={TP.FRESH_HEADER: '1'})))
    th.start()
    time.sleep(0.3)
    entry('Page A9', ['--slow', '0.2'])
    write()
    s, h, b = get()
    th.join()
    check('rebuild: an entry changed during its build - the asking request gets that build, the others the new entry',
          b'Page A9' in b and b'Page A8' in res['r'][2], (b[:80], res['r'][2][:80]))
    settle(st)
    s, h, b = get()
    check('rebuild: ... and the store keeps the new entry\'s page', b'Page A9' in b, b[:80])
    # a page whose entry left the registry goes at once
    TP.serve_request(st, False, 'GET', '/p/pb/', {})
    write(drop=('pb',))
    s, h, b = TP.serve_request(st, False, 'GET', '/p/pb/', {})
    check('rebuild: a page that left the registry is not served and is dropped at once',
          s == 404 and ('pb', False) not in st.pages, (s, list(st.pages)))
    # a failed rebuild drops the old page; the next request shows the failure
    open(stamp, 'w').write('FAIL')
    code = pa['code']
    src = open(code).read()
    open(code, 'w').write(src.replace("STAMP = open(_SP).read().strip() if os.path.exists(_SP) else ''",
                                      "STAMP = open(_SP).read().strip() if os.path.exists(_SP) else ''\n"
                                      "if STAMP == 'FAIL':\n    raise SystemExit('REFUSING: stamp says fail')"))
    touch()
    get()
    settle(st)
    s, h, b = get()
    check('rebuild: a failed rebuild drops the old page and the next request shows the failure',
          s == 500 and b'could not be built' in b, (s, b[:120]))
    left = [n for n in sys.modules if n.startswith('_twoport_pa_r_')]
    check('rebuild: failed builds leave no page module behind', left == [], left)
    open(stamp, 'w').write('S4')
    s, h, b = get()
    check('rebuild: ... and the page serves again once the failure is gone', s == 200 and b'stamp=S4' in b, (s, b[:80]))
    open(code, 'w').write(src)

def test_trees():
    trees = {}
    for t in ('treeA', 'treeB'):
        d = os.path.join(TMP, t)
        os.makedirs(d)
        page = PAGE.replace("FLAG_AT_IMPORT = os.environ.get('P3_TEST_FLAG', 'unset')",
                            "FLAG_AT_IMPORT = os.environ.get('P3_TEST_FLAG', 'unset')\nimport helper_mod")
        page = page.replace("f'flag-at-import={FLAG_AT_IMPORT} ", "f'helper={helper_mod.V} flag-at-import={FLAG_AT_IMPORT} ")
        open(os.path.join(d, 'page.py'), 'w').write(page)
        open(os.path.join(d, 'helper_mod.py'), 'w').write(f"V = '{t[-1]}'\n")
        trees[t] = d
    run = os.path.join(TMP, 'run_t')
    os.makedirs(run)
    reg = os.path.join(TMP, 'registry_trees.json')
    pages = [
        {'id': 'ta1', 'kind': 'review', 'code': os.path.join(trees['treeA'], 'page.py'), 'argv': ['--run', run, '--title', 'A1'],
         'legacy_port': 9911, 'greenlit': True},
        {'id': 'ta2', 'kind': 'review', 'code': os.path.join(trees['treeA'], 'page.py'), 'argv': ['--run', run, '--title', 'A2'],
         'legacy_port': 9912, 'greenlit': False},
        {'id': 'tb1', 'kind': 'review', 'code': os.path.join(trees['treeB'], 'page.py'), 'argv': ['--run', run, '--title', 'B1'],
         'legacy_port': 9913, 'greenlit': True},
    ]
    write_registry(reg, pages)
    st = TP.Store(reg, builds=False)
    be = TP.Backends(reg, sock_dir=os.path.join(TMP, 'socks'), python=sys.executable)
    try:
        def go(pid, path='/', inspect=False, method='GET', body=b'', headers=None):
            return TP.front_request(st, be, inspect, method, f'/p/{pid}{path}', headers or {}, body)
        sb, _, bb = go('tb1')                    # tree B first: its helper must not reach tree A's pages
        sa, _, ba = go('ta1')
        s2, _, b2 = go('ta2')
        check('trees: tree B renders its own helper', sb == 200 and b'helper=B' in bb and b'B1' in bb, (sb, bb[:160]))
        check('trees: tree A renders its own helper although tree B was built first', sa == 200 and b'helper=A' in ba, (sa, ba[:160]))
        check('trees: a second page of tree A shares tree A\'s backend and helper', s2 == 200 and b'helper=A' in b2)
        check('trees: one backend process per tree', len(be.procs) == 2 and len({p.pid for p, _ in be.procs.values()}) == 2,
              list(be.procs))
        check('trees: responses are rewritten under /p/<id> as before', b'href="/p/ta1/source/J1"' in ba, ba[:300])
        s, _, b = TP.front_request(st, be, False, 'GET', '/_status', {})
        rep = json.loads(b)
        bytree = {x['tree']: x for x in rep['backends']}
        a = bytree.get(os.path.realpath(trees['treeA']), {})
        pa = (a.get('pages') or {}).get('ta1') or {}
        hm = (pa.get('helpers') or {}).get('helper_mod')
        check('status: each backend reports its tree', set(bytree) == {os.path.realpath(trees[t]) for t in trees}, list(bytree))
        check('status: a page reports the md5 of the code and of the helper it imported',
              pa.get('code', [None, None])[1] == TP.md5_file(os.path.join(trees['treeA'], 'page.py'))
              and hm and hm[1] == TP.md5_file(os.path.join(trees['treeA'], 'helper_mod.py')), pa)
        s, h, b = go('ta1', '/label', method='POST', body=b'source_id=J9&confidence=real',
                     headers={'Content-Type': 'application/x-www-form-urlencoded'})
        jr = os.path.join(run, 'judgements', 'judgements.jsonl')
        check('trees: a POST is forwarded, written and its redirect rewritten',
              s == 303 and dict(h).get('Location') == '/p/ta1/source/J9' and os.path.exists(jr) and 'J9' in open(jr).read(),
              (s, h))
        s, _, b = go('ta1', '/source/J1', inspect=True)
        check('trees: the inspection listener gets the read-only page', s == 200 and b'<form' not in b and b'J1' in b, b[:200])
        s, _, b = go('ta1', '/label', inspect=True, method='POST', body=b'source_id=J1')
        check('trees: the inspection listener refuses a POST in the front', s == 403, s)
        s, _, b = go('ta2', '/', inspect=True)
        check('trees: a page not greenlit is not served on the inspection listener', s == 404, s)
        s, _, b = TP.front_request(st, be, False, 'GET', '/p/ta1/', {TP.LISTENER_HEADER: 'review'})
        check('trees: a browser cannot choose the listener (the front sets the header)', s == 200, s)
        s, _, b = TP.front_request(st, be, True, 'GET', '/p/ta1/source/J1', {TP.LISTENER_HEADER: 'review'})
        check('trees: ... and a forged header on the inspection listener still gets the read-only page',
              s == 200 and b'<form' not in b, b[:200])
        old_pid = be.procs[os.path.realpath(trees['treeA'])][0].pid
        be.procs[os.path.realpath(trees['treeA'])][0].kill()
        be.procs[os.path.realpath(trees['treeA'])][0].wait()
        s, _, b = go('ta1')
        new_pid = be.procs[os.path.realpath(trees['treeA'])][0].pid
        check('trees: a dead backend is restarted on the next request', s == 200 and b'helper=A' in b and new_pid != old_pid,
              (s, old_pid, new_pid))
        write_registry(reg, pages[:2])
        os.utime(reg, (os.path.getmtime(reg) + 5,) * 2)
        bproc = be.procs[os.path.realpath(trees['treeB'])][0]
        s, _, b = go('ta1')
        check('trees: a tree that leaves the registry has its backend stopped',
              os.path.realpath(trees['treeB']) not in be.procs and bproc.poll() is not None, list(be.procs))
    finally:
        procs = [p for p, _ in be.procs.values()]
        be.stop_all()
        check('trees: every backend stops with the front', all(p.poll() is not None for p in procs))


def main():
    shutil.rmtree(TMP, ignore_errors=True)
    os.makedirs(TMP)
    try:
        test_env()
        test_rewrite()
        test_registry()
        test_pages()
        test_trees()
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    print(f'{N[0] - len(FAILS)} of {N[0]} checks passed')
    return 1 if FAILS else 0


def test_all():
    assert main() == 0


if __name__ == '__main__':
    sys.exit(main())
