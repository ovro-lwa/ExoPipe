#!/usr/bin/env python3
"""Tests of lib/serve/label_identity.py through the two-port server's run_request: a label sent from a page shown before a
rebuild lands on the identity it was made on (by group_identity_id), or is refused with nothing recorded.

A synthetic review page shaped like classify_transients.py (an App whose sources hold one catalog row per band, hidden
source_id inputs exactly as production writes them, handle_label refusing an absent page id) is built twice by
two_port.build_page: build 1 from catalog rows before a regroup, build 2 after it. Between the builds GTI-0000001 is renamed,
the name J0200+0200 passes from GTI-0000002 to GTI-0000003 (the case that recorded labels on another identity), GTI-0000006
leaves (merged) and GTI-0000004 is new. Forms rendered by build 1 are posted to build 2:
  stamp     build 1's forms carry the identity key after every hidden source_id input (a page holding two IDs: both, sorted,
            ';'-joined); a page with none is not stamped; a production page (no group_identity_id column) is served byte for
            byte as its module writes it
  carry     a renamed identity's label lands on its new page id; the name now carried by another identity does NOT take the
            label - it lands on its own identity's page; the body rewrite keeps every field (Content-Length follows)
  refuse    an identity on no page of the new build, and an unstamped /label POST to a stamped build, are refused with 409:
            nothing is written, the reason and the label are shown
  pass      a stamped POST of the same build, an unstamped non-label POST, an unstamped label of a page of the build that has no
            stable ID, and every POST to a production page go through as before
  served    serve_request answers a source page with the stamped form

Run: env PYTHONPATH=$PWD NUMPY_MADVISE_HUGEPAGE=0 /opt/devel/pipeline/envs/py38_orca_nkosogor/bin/python3
     tests/test_serve_label_identity.py
Writes only under $TMPDIR/exopipe3_tests_label_identity (removed at the end)."""
import csv
import html
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import urllib.parse

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
TMP = os.path.join(tempfile.gettempdir(), 'exopipe3_tests_label_identity')

from exopipe.phase3.lib.serve import label_identity as LI   # noqa: E402
from exopipe.phase3.lib.serve import two_port as TP          # noqa: E402

FAILS, N = [], [0]


def check(what, ok, detail=''):
    N[0] += 1
    if ok:
        print('PASS ', what)
    else:
        FAILS.append(f'{what}: {detail}')
        print('FAIL ', what, detail)


PAGE = r'''
import argparse, csv, html, json, os, urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
INSPECT = False

class Source:
    def __init__(self, sid):
        self.source_id = sid
        self.bands = {}

class App:
    def __init__(self, run, labels):
        self.labels = labels
        self.sources = {}
        with open(os.path.join(run, 'rows.csv'), newline='') as f:
            for r in csv.DictReader(f):
                sid = (r.get('identity_id') or '').strip() or r['source_id']
                self.sources.setdefault(sid, Source(sid)).bands[r['band']] = r
    def page_source(self, sid):
        e = html.escape(sid)
        return (f'<html><body>{e} <form method="POST" action="/label"><input type="hidden" name="source_id" value="{e}">'
                f'<input type="hidden" name="scope" value="source"><input name="confidence" value="FP">'
                f'<input name="fp_reason" value="leakage"><textarea name="note">a note &amp; more</textarea></form>'
                f'<form method="POST" action="/list"><input type="hidden" name="action" value="cutouts">'
                f'<input type="hidden" name="source_id" value="{e}"></form></body></html>')
    def handle_label(self, form):
        sid = form.get('source_id', [''])[0]
        if sid not in self.sources or not form.get('scope'):
            return None
        os.makedirs(self.labels, exist_ok=True)
        with open(os.path.join(self.labels, 'judgements.jsonl'), 'a') as f:
            f.write(json.dumps({k: v[0] for k, v in form.items()}) + '\n')
        return '/source/' + urllib.parse.quote(sid)

class Handler(BaseHTTPRequestHandler):
    app = None
    def _send(self, body, ctype='text/html; charset=utf-8', code=200):
        self.send_response(code); self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        p = urllib.parse.urlparse(self.path)
        if p.path.startswith('/source/'):
            self._send(self.app.page_source(urllib.parse.unquote(p.path[8:])).encode())
        else:
            self._send(b'not found', 'text/plain', 404)
    def do_POST(self):
        n = int(self.headers.get('Content-Length', 0))
        form = urllib.parse.parse_qs(self.rfile.read(n).decode())
        if self.path == '/list':
            self._send(('list ok ' + form.get('source_id', [''])[0]).encode(), 'text/plain')
            return
        dest = self.app.handle_label(form)
        if dest is None:
            self._send(b'bad request', 'text/plain', 400)
            return
        self.send_response(303); self.send_header('Location', dest); self.end_headers()
    def log_message(self, *a):
        pass

def main():
    global INSPECT
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True); ap.add_argument('--labels-dir')
    ap.add_argument('--port', type=int, default=1); ap.add_argument('--inspect', action='store_true')
    a = ap.parse_args()
    INSPECT = a.inspect
    Handler.app = App(a.run, a.labels_dir or os.path.join(a.run, 'judgements'))
    ThreadingHTTPServer(('127.0.0.1', a.port), Handler).serve_forever()
'''

# (display name, stable ID, band) rows of the two builds
BUILD1 = [('J0100+0100', 'GTI-0000001', '46MHz'), ('J0100+0100', 'GTI-0000001', '55MHz'),
          ('J0200+0200', 'GTI-0000002', '46MHz'), ('J0300+0300', 'GTI-0000003', '46MHz'),
          ('J0600+0600', 'GTI-0000006', '46MHz'), ('J0700+0700', 'GTI-0000007', '46MHz'),
          ('J0700+0700', 'GTI-0000008', '55MHz'), ('J0800+0800', '', '46MHz')]
BUILD2 = [('J0100+0101', 'GTI-0000001', '46MHz'), ('J0100+0101', 'GTI-0000001', '55MHz'),
          ('J0200+0159', 'GTI-0000002', '46MHz'), ('J0200+0200', 'GTI-0000003', '46MHz'),
          ('J0500+0500', 'GTI-0000004', '46MHz'), ('J0900+0900', '', '46MHz')]


def write_run(d, rows, with_id=True):
    os.makedirs(d, exist_ok=True)
    cols = ['source_id', 'identity_id', 'band'] + (['group_identity_id'] if with_id else [])
    with open(os.path.join(d, 'rows.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(cols)
        for name, gid, band in rows:
            w.writerow([f'{name}_{band}', name, band] + ([gid] if with_id else []))
    return d


def build(pid, run, labels):
    code = os.path.join(TMP, 'page.py')
    if not os.path.exists(code):
        open(code, 'w').write(PAGE)
    entry = {'id': pid, 'kind': 'review', 'code': code, 'argv': ['--run', run, '--labels-dir', labels], 'legacy_port': 9911}
    return TP.build_page(entry, False, {}, threading.Lock())


def form_bodies(page_html):
    """[(action, urlencoded body)] of the forms of a served page, as a browser submits them."""
    out = []
    for m in re.finditer(r'<form method="POST" action="([^"]+)">(.*?)</form>', page_html, re.S):
        pairs = [(a, html.unescape(b)) for a, b in re.findall(r'<input (?:type="hidden" )?name="([^"]+)" value="([^"]*)">',
                                                                 m.group(2))]
        pairs += [(a, html.unescape(b)) for a, b in re.findall(r'<textarea name="([^"]+)">(.*?)</textarea>', m.group(2), re.S)]
        out.append((m.group(1), urllib.parse.urlencode(pairs).encode()))
    return out


def get(pg, sid):
    st, hd, body = TP.run_request(pg, 'GET', '/source/' + urllib.parse.quote(sid), {})
    return st, hd, body.decode()


def post(pg, action, body):
    return TP.run_request(pg, 'POST', action, {'Content-Type': 'application/x-www-form-urlencoded',
                                               'Content-Length': str(len(body))}, body)


def journal(labels):
    p = os.path.join(labels, 'judgements.jsonl')
    return [json.loads(x) for x in open(p)] if os.path.exists(p) else []


def test_unit():
    class S:
        def __init__(self, rows):
            self.bands = {r['band']: r for r in rows}

    class A:
        sources = {'X': S([dict(band='46MHz', group_identity_id='GTI-0000001'), dict(band='55MHz', group_identity_id='GTI-0000001')]),
                   'Y': S([dict(band='46MHz', group_identity_id='GTI-0000002'), dict(band='55MHz', group_identity_id='GTI-0000003')]),
                   'Z': S([dict(band='46MHz')]), 'W&1': S([dict(band='46MHz', group_identity_id='GTI-0000004')])}
    by_sid, by_gid, pages = maps = LI.identity_maps(A)
    check('maps: a page keyed by its stable ID (two IDs: sorted, joined); a page with none is in neither map',
          by_sid == {'X': 'GTI-0000001', 'Y': 'GTI-0000002;GTI-0000003', 'W&1': 'GTI-0000004'} and by_gid == {
              'GTI-0000001': ['X'], 'GTI-0000002;GTI-0000003': ['Y'], 'GTI-0000004': ['W&1']} and pages == {'X', 'Y', 'Z', 'W&1'},
          (by_sid, by_gid, pages))
    t = ('<input type="hidden" name="source_id" value="X"><input type="hidden" name="source_id" value="Y">'
         '<input type="hidden" name="source_id" value="W&amp;1">')
    s = LI.stamp(t, by_sid)
    check('stamp: the identity key follows each hidden source_id input of a page that has one (escaped values looked up)',
          s == ('<input type="hidden" name="source_id" value="X"><input type="hidden" name="group_identity_id" '
                'value="GTI-0000001"><input type="hidden" name="source_id" value="Y"><input type="hidden" '
                'name="group_identity_id" value="GTI-0000002;GTI-0000003"><input type="hidden" name="source_id" '
                'value="W&amp;1"><input type="hidden" name="group_identity_id" value="GTI-0000004">'), s)
    check('stamp: no map, no change', LI.stamp(t, {}) is t)
    b, ref, what = LI.resolve('/label', b'source_id=OLD&group_identity_id=GTI-0000001&scope=source&note=a+%26+b', maps)
    check('resolve: a renamed identity is rewritten to its page, every other field kept in order',
          ref is None and what == 'carried OLD -> X' and urllib.parse.parse_qsl(b.decode()) == [
              ('source_id', 'X'), ('group_identity_id', 'GTI-0000001'), ('scope', 'source'), ('note', 'a & b')], (b, what))
    b0 = b'source_id=X&group_identity_id=GTI-0000001&scope=source'
    b, ref, what = LI.resolve('/label', b0, maps)
    check('resolve: the same identity passes the body unchanged', b is b0 and ref is None and what == 'same', what)
    b, ref, what = LI.resolve('/label', b'source_id=X&group_identity_id=GTI-0000009&scope=source&confidence=FP', maps)
    check('resolve: an identity on no page is refused (409) with the label echoed',
          ref is not None and ref[0] == 409 and what == 'refused' and b'Nothing was recorded' in ref[2]
          and b'<td>confidence</td><td>FP</td>' in ref[2], (what, ref and ref[2][:300]))
    b, ref, what = LI.resolve('/label?x=1', b'source_id=X&scope=source', maps)
    check('resolve: an unstamped /label POST to a stamped build is refused', ref is not None and ref[0] == 409, what)
    b, ref, what = LI.resolve('/label', b'source_id=GONE&scope=source', maps)
    check('resolve: an unstamped /label POST naming no page of a stamped build is refused', ref is not None and ref[0] == 409,
          what)
    b, ref, what = LI.resolve('/label', b'source_id=Z&scope=source', maps)
    check('resolve: an unstamped /label POST of a page of the build without a stable ID passes', ref is None, what)
    b, ref, what = LI.resolve('/list', b'source_id=X&which=watch', maps)
    check('resolve: an unstamped POST to another route passes', ref is None and what == 'unstamped', what)
    b, ref, what = LI.resolve('/label', b'source_id=X&scope=source', ({}, {}, frozenset()))
    check('resolve: an unstamped /label POST to a build without stable IDs passes', ref is None and what == 'unstamped', what)


def test_pages():
    run1 = write_run(os.path.join(TMP, 'run1'), BUILD1)
    run2 = write_run(os.path.join(TMP, 'run2'), BUILD2)
    runp = write_run(os.path.join(TMP, 'runp'), BUILD1, with_id=False)
    lab1, lab2, labp = (os.path.join(TMP, x) for x in ('labels1', 'labels2', 'labelsp'))
    p1, p2, pp = build('b1', run1, lab1), build('b2', run2, lab2), build('bp', runp, labp)

    st, hd, h = get(p1, 'J0100+0100')
    raw = p1.mod.Handler.app.page_source('J0100+0100')
    check('stamp: build 1 serves its forms with the stable ID after the hidden source_id input (both forms), Content-Length '
          'follows', h.count('name="source_id" value="J0100+0100"><input type="hidden" name="group_identity_id" '
                             'value="GTI-0000001">') == 2 and h == LI.stamp(raw, LI.page_maps(p1)[0])
          and dict(hd).get('Content-Length') == str(len(h.encode())), h[:400])
    st, hd, h7 = get(p1, 'J0700+0700')
    st, hd, h8 = get(p1, 'J0800+0800')
    check('stamp: a page holding two stable IDs is stamped with both; a page with none is served unstamped',
          'name="group_identity_id" value="GTI-0000007;GTI-0000008"' in h7 and 'group_identity_id' not in h8,
          (h7[:300], h8[:200]))
    st, hd, hp = get(pp, 'J0100+0100')
    check('stamp: a production page (no group_identity_id column) is served byte for byte as its module writes it',
          hp == pp.mod.Handler.app.page_source('J0100+0100'), hp[:200])

    forms1 = {sid: form_bodies(get(p1, sid)[2]) for sid in ('J0100+0100', 'J0200+0200', 'J0300+0300', 'J0600+0600',
                                                            'J0800+0800')}
    before = len(journal(lab2))
    out = {}
    for sid in ('J0100+0100', 'J0200+0200', 'J0300+0300'):
        st, hd, body = post(p2, *forms1[sid][0])
        out[sid] = (st, dict(hd).get('Location'))
    J = journal(lab2)[before:]
    landed = [(r['source_id'], r.get('group_identity_id'), r.get('confidence'), r.get('fp_reason'), r.get('note')) for r in J]
    check('carry: build 1 labels posted to build 2 land on their own identities (renamed GTI-0000001 -> J0100+0101; '
          'GTI-0000002 -> J0200+0159, NOT on J0200+0200 which now names GTI-0000003; GTI-0000003 -> J0200+0200), every field '
          'intact', landed == [('J0100+0101', 'GTI-0000001', 'FP', 'leakage', 'a note & more'),
                               ('J0200+0159', 'GTI-0000002', 'FP', 'leakage', 'a note & more'),
                               ('J0200+0200', 'GTI-0000003', 'FP', 'leakage', 'a note & more')]
          and [out[s] for s in ('J0100+0100', 'J0200+0200', 'J0300+0300')] == [
              (303, '/source/J0100%2B0101'), (303, '/source/J0200%2B0159'), (303, '/source/J0200%2B0200')], (landed, out))

    n = len(journal(lab2))
    st, hd, body = post(p2, *forms1['J0600+0600'][0])
    check('refuse: an identity on no page of build 2 (merged away) is refused with 409, nothing written, the reason and the '
          'label shown', st == 409 and len(journal(lab2)) == n and b'Nothing was recorded' in body and b'GTI-0000006' in body
          and b'<td>confidence</td><td>FP</td>' in body, (st, body[:300]))
    st, hd, body = post(p2, *forms1['J0800+0800'][0])
    check('refuse: an unstamped /label POST (a page without a stable ID) to a stamped build is refused, nothing written',
          st == 409 and len(journal(lab2)) == n, (st, body[:200]))
    st, hd, body = post(p2, '/label', b'source_id=J0200%2B0200&scope=source&confidence=real')
    check('refuse: a production-style form (no stamp) naming a page of the stamped build is refused, nothing written',
          st == 409 and len(journal(lab2)) == n, (st, body[:200]))
    st, hd, body = post(p2, *form_bodies(get(p2, 'J0900+0900')[2])[0])
    check('pass: an unstamped label of a page of build 2 that has no stable ID is recorded as before',
          st == 303 and journal(lab2)[-1]['source_id'] == 'J0900+0900', (st, body[:200]))
    n = len(journal(lab2))
    st, hd, body = post(p2, *forms1['J0100+0100'][1])
    check('carry: a stamped /list POST of a renamed identity reaches its new page', st == 200 and body == b'list ok J0100+0101',
          (st, body))
    st, hd, body = post(p2, '/list', b'action=cutouts&source_id=J0100%2B0101')
    check('pass: an unstamped /list POST passes as before', st == 200 and body == b'list ok J0100+0101', (st, body))
    f2 = form_bodies(get(p2, 'J0200+0200')[2])
    st, hd, body = post(p2, *f2[0])
    J = journal(lab2)
    check('pass: a stamped label of the same build is recorded on its page', st == 303 and J[-1]['source_id'] == 'J0200+0200'
          and J[-1]['group_identity_id'] == 'GTI-0000003', (st, J[-1:]))
    fp = form_bodies(get(pp, 'J0100+0100')[2])
    st, hd, body = post(pp, *fp[0])
    st2, hd2, body2 = post(pp, '/label', b'source_id=J9999%2B9999&scope=source')
    check('pass: a production page records an unstamped label as before, and answers an absent page id as before (400)',
          st == 303 and journal(labp)[-1]['source_id'] == 'J0100+0100' and 'group_identity_id' not in journal(labp)[-1]
          and st2 == 400, (st, st2, journal(labp)))

    reg = os.path.join(TMP, 'registry.json')
    json.dump({'pages': [{'id': 'pg', 'kind': 'review', 'code': os.path.join(TMP, 'page.py'),
                'argv': ['--run', run2, '--labels-dir', os.path.join(TMP, 'labels_s')], 'legacy_port': 9912}]}, open(reg, 'w'))
    store = TP.Store(reg)
    st, hd, body = TP.serve_request(store, False, 'GET', '/p/pg/source/J0100%2B0101', {})
    check('served: serve_request answers a source page with the stamped form',
          st == 200 and b'name="source_id" value="J0100+0101"><input type="hidden" name="group_identity_id" '
                        b'value="GTI-0000001">' in body, body[:300])
    fb = form_bodies(get(p1, 'J0100+0100')[2])[0][1]
    st, hd, body = TP.serve_request(store, False, 'POST', '/p/pg/label', {'Content-Length': str(len(fb))}, fb)
    J = journal(os.path.join(TMP, 'labels_s'))
    check('served: a build 1 label POSTed through serve_request lands on the renamed page, its redirect rewritten',
          st == 303 and J and J[-1]['source_id'] == 'J0100+0101' and dict(hd).get('Location') == '/p/pg/source/J0100%2B0101',
          (st, hd, J))


def main():
    shutil.rmtree(TMP, ignore_errors=True)
    os.makedirs(TMP)
    TP.install_env()
    try:
        test_unit()
        test_pages()
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    print(f'{N[0] - len(FAILS)} of {N[0]} checks passed')
    return 1 if FAILS else 0


def test_all():
    assert main() == 0


if __name__ == '__main__':
    sys.exit(main())
