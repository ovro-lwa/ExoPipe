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
  merged    (R-GREGG-20261124-CROSS-STOKES-LINK-TEST-APPROVED, Q2) build 3 = build 2 with the run's merge table (GTI-0000006
            merged into GTI-0000003): build 1's label of GTI-0000006 follows the merge to J0200+0200; a label of a different
            verdict is NOT made its verdict - it is kept with the verdict it conflicts with (merge_conflicts.jsonl), 409 says
            so, and the page opens AMBIGUOUS with both verdicts until a verdict is recorded on it; the same verdict follows the
            merge; a non-label POST follows it; without the merge table the label is refused as before
  settled   label_identity.conflict_settled: what settles a merge conflict (shared with the journal carry)
  versions  through serve_request on a versioned run (PR 155): a form of the version before the switch whose identity maps
            (merged) lands on the survivor; only an unmappable or unstamped form gets the stale-form 409
  banner    the cross-Stokes link banner (lib/serve/stokes_banner.py) through run_request: after <body>, on a source page of a
            build whose entry names its other-Stokes pages; the forms stamped as before

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
import argparse, csv, datetime, html, json, os, urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
INSPECT = False

class Source:
    def __init__(self, sid):
        self.source_id = sid
        self.bands = {}

class App:
    def __init__(self, run, labels):
        self.labels = labels
        self.journal = os.path.join(labels, 'judgements.jsonl')
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
        rec = {k: v[0] for k, v in form.items()}
        rec['ts'] = datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        with open(self.journal, 'a') as f:
            f.write(json.dumps(rec) + '\n')
        return '/source/' + urllib.parse.quote(sid)
    def state(self):
        st = {}
        if os.path.exists(self.journal):
            for line in open(self.journal):
                r = json.loads(line)
                if r.get('scope') == 'source':
                    st[(r['source_id'], 'p1')] = {'source': dict(r, judge_confidence=r.get('confidence', ''),
                                                                 judge_fp_reason=r.get('fp_reason', ''),
                                                                 judge_fp_reason_2=r.get('fp_reason_2', ''))}
        return st

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


def build(pid, run, labels, **extra):
    code = os.path.join(TMP, 'page.py')
    if not os.path.exists(code):
        open(code, 'w').write(PAGE)
    entry = dict({'id': pid, 'kind': 'review', 'code': code, 'argv': ['--run', run, '--labels-dir', labels], 'legacy_port': 9911},
                 **extra)
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


def with_set(form_body, **kv):
    pairs = [(k, kv.pop(k) if k in kv else v) for k, v in urllib.parse.parse_qsl(form_body.decode(), keep_blank_values=True)]
    return urllib.parse.urlencode(pairs + list(kv.items())).encode()


def conflicts(labels):
    p = os.path.join(labels, LI.MERGE_CONFLICTS)
    return [json.loads(x) for x in open(p)] if os.path.exists(p) else []


def test_merged():
    run1 = os.path.join(TMP, 'run1')
    run3 = write_run(os.path.join(TMP, 'run3'), BUILD2)
    os.makedirs(os.path.join(run3, 'stokes_link'))
    with open(os.path.join(run3, 'stokes_link', 'identity_merges.csv'), 'w') as f:
        f.write('stokes,identity_id,into_id,run\nI,GTI-0000006,GTI-0000003,R2\n')
    lab1, lab3 = os.path.join(TMP, 'labels1m'), os.path.join(TMP, 'labels3')
    p1, p3 = build('m1', run1, lab1), build('m3', run3, lab3)
    f6 = form_bodies(get(p1, 'J0600+0600')[2])
    st, hd, body = post(p3, *f6[0])
    J = journal(lab3)
    check('merged: build 1\'s label of GTI-0000006 (merged into GTI-0000003) lands on J0200+0200, its verdict unchanged',
          st == 303 and len(J) == 1 and (J[0]['source_id'], J[0]['confidence'], J[0]['fp_reason']) ==
          ('J0200+0200', 'FP', 'leakage') and dict(hd).get('Location') == '/source/J0200%2B0200', (st, J))
    st, hd, body = post(p3, '/label', with_set(f6[0][1], confidence='real', fp_reason=''))
    J, C = journal(lab3), conflicts(lab3)
    check('merged: a label of ANOTHER verdict is not made the verdict (journal unchanged), it is kept with the verdict it '
          'conflicts with, and 409 says so',
          st == 409 and len(J) == 1 and b'Kept as a conflicting verdict' in body and len(C) == 1
          and C[0]['source_id'] == 'J0200+0200' and C[0]['group_identity_id'] == 'GTI-0000003'
          and [(v['judge_confidence'], v['source']) for v in C[0]['verdicts']] == [('FP', 'J0200+0200'),
                                                                                   ('real', 'J0600+0600')], (st, body[:300], C))
    st, hd, h = get(p3, 'J0200+0200')
    check('merged: the survivor\'s page opens AMBIGUOUS with both verdicts, its form stamped as before',
          h.startswith('<html><body><div class="merge-conflict"') and 'AMBIGUOUS' in h and '<b>FP / leakage</b>' in h
          and '<b>real</b>' in h and 'name="group_identity_id" value="GTI-0000003"' in h
          and dict(hd).get('Content-Length') == str(len(h.encode())), h[:600])
    st, hd, body = post(p3, *f6[0])
    check('merged: a label of the SAME verdict follows the merge', st == 303 and len(journal(lab3)) == 2
          and journal(lab3)[-1]['source_id'] == 'J0200+0200', (st, journal(lab3)))
    st, hd, body = post(p3, *f6[1])
    check('merged: a stamped /list POST follows the merge', st == 200 and body == b'list ok J0200+0200', (st, body))
    f3 = form_bodies(get(p3, 'J0200+0200')[2])
    st, hd, body = post(p3, '/label', with_set(f3[0][1], confidence='probably-real', fp_reason=''))
    st2, hd2, h = get(p3, 'J0200+0200')
    check('merged: a verdict recorded on the page settles it: no AMBIGUOUS banner', st == 303 and 'AMBIGUOUS' not in h, h[:300])
    p2 = build('m2', os.path.join(TMP, 'run2'), os.path.join(TMP, 'labels2m'))
    st, hd, body = post(p2, *f6[0])
    check('merged: without the run\'s merge table the label is refused as before (409, nothing written)',
          st == 409 and not journal(os.path.join(TMP, 'labels2m')), (st, body[:200]))


def test_banner():
    run3 = os.path.join(TMP, 'run3')
    d = os.path.join(run3, 'stokes_link')
    with open(os.path.join(d, 'stokes_links.csv'), 'w') as f:
        f.write('link_id,link_name,link_size,link_ambiguous,stokes,identity_id,identity_name,linked_ids,linked_names\n'
                'GTI-0000001,J0100+0101,2,0,I,GTI-0000001,J0100+0101,GTV-0000005,J0100+0102\n'
                'GTI-0000001,J0100+0101,2,0,V,GTV-0000005,J0100+0102,GTI-0000001,J0100+0101\n')
    with open(os.path.join(d, 'stokes_link_pairs.csv'), 'w') as f:
        f.write('link_id,link_ambiguous,I_identity_id,I_identity_name,V_identity_id,V_identity_name,offset_arcsec,u,p,'
                'offset_pct,p_chance\nGTI-0000001,0,GTI-0000001,J0100+0101,GTV-0000005,J0100+0102,12.5,0.4,0.991,8.1,0.0012\n')
    pb = build('mb', run3, os.path.join(TMP, 'labelsb'), stokes_partner=[dict(port=8787, stokes='V', kind='single')])
    st, hd, h = get(pb, 'J0100+0101')
    check('banner: a source page of a build naming its other-Stokes pages opens with the link (the V identity, its page, P, '
          'the offset, its percentile, the chance-coincidence probability), its forms stamped as before',
          h.startswith('<html><body><div class="stokes-link"') and 'Stokes V J0100+0102' in h
          and 'http://localhost:8787/source/J0100%2B0102' in h and 'P = 0.991' in h and 'offset percentile 8' in h
          and 'probability 0.0012' in h and 'name="group_identity_id" value="GTI-0000001"' in h, h[:700])
    st, hd, h = get(pb, 'J0500+0500')
    check('banner: an identity of that build without a link says so', 'Not linked to a Stokes V identity' in h, h[:400])
    st, hd, h = get(build('mc', run3, os.path.join(TMP, 'labelsc')), 'J0100+0101')
    check('banner: a build whose entry names no other-Stokes pages has no link banner', 'stokes-link' not in h, h[:300])


def test_settled():
    rec = dict(verdicts=[dict(judge_confidence='FP', judge_fp_reason='leakage', ts='T1'),
                         dict(judge_confidence='real', judge_fp_reason='', ts='T2')])
    fp = dict(judge_confidence='FP', judge_fp_reason='leakage')
    check('settled: unjudged, the listed verdict itself, or a carried copy of a listed verdict (the carry\'s time) settle '
          'nothing; a verdict judged on the page since, or a carried verdict not listed, settles it',
          not LI.conflict_settled(rec, []) and not LI.conflict_settled(rec, [dict(fp, ts='T1')])
          and not LI.conflict_settled(rec, [dict(fp, ts='T9', carried_from_run='R2')])
          and LI.conflict_settled(rec, [dict(fp, ts='T9')])
          and LI.conflict_settled(rec, [dict(judge_confidence='uncertain', ts='T9', carried_from_run='R2')]))


def test_versions():
    """PR 155's stale-form 409 (two_port.serve_request, PAGE_TOKEN_FIELD) comes AFTER the identity mapping: a form of the
    version before the switch whose identity maps onto the served version lands on that identity's page; only an
    unmappable form gets the 409"""
    from exopipe.phase3.lib.serve import versions
    d = os.path.join(TMP, 'versions')
    U, J = os.path.join(d, 'Run_SLSTB'), 'judgements_I'
    os.makedirs(d)
    v1 = versions.begin(U, note='test 1')
    write_run(v1.run, BUILD1)
    os.makedirs(os.path.join(v1.run, J))
    versions.publish(U, v1)
    code = os.path.join(TMP, 'page.py')
    if not os.path.exists(code):
        open(code, 'w').write(PAGE)
    reg = os.path.join(d, 'registry.json')
    with open(reg, 'w') as f:
        json.dump({'pages': [{'id': 'pv', 'kind': 'review', 'code': code, 'legacy_port': 47993,
                              'argv': ['--run', U, '--labels-dir', os.path.join(U, J)]}]}, f)
    st = TP.Store(reg)

    def page(sid):
        r = TP.serve_request(st, False, 'GET', '/p/pv/source/' + urllib.parse.quote(sid), {})
        return [(a, b.decode()) for a, b in form_bodies(r[2].decode()) if a.endswith("/label")][0]
    f6, f7 = page('J0600+0600'), page('J0700+0700')
    check('versions: build 1\'s label forms carry the version token and the identity stamp',
          TP.PAGE_TOKEN_FIELD in f6[1] and 'group_identity_id=GTI-0000006' in f6[1], f6)
    v2 = versions.begin(U, note='test 2')
    write_run(v2.run, BUILD2)
    os.makedirs(os.path.join(v2.run, J))
    os.makedirs(os.path.join(v2.run, 'stokes_link'))
    with open(os.path.join(v2.run, 'stokes_link', 'identity_merges.csv'), 'w') as f:
        f.write('stokes,identity_id,into_id,run\nI,GTI-0000006,GTI-0000003,R2\n')
    with versions.journal_locks(U, [J]):
        versions.switch(U, v2, record=versions.seal(v2))

    def post(body):
        return TP.serve_request(st, False, 'POST', '/p/pv/label', {'Content-Length': str(len(body))}, body.encode())
    r6 = post(f6[1])
    got = journal(os.path.join(U, J))
    check('versions: a form of the version before the switch whose identity was merged lands on the survivor\'s page in the '
          'served journal (the mapping runs before the stale-form 409)',
          r6[0] == 303 and len(got) == 1 and got[0]['source_id'] == 'J0200+0200' and TP.PAGE_TOKEN_FIELD not in got[0]
          and not journal(os.path.join(v1.run, J)), (r6[:1], r6[2][:200], got))
    r7 = post(f7[1])
    r0 = post(re.sub(r'&?group_identity_id=[^&]*', '', f6[1]))
    check('versions: an unmappable form of that version (its identity on no page of the served one) and one without the '
          'stamp get the stale-form 409 "reload and judge again", nothing recorded',
          r7[0] == 409 and b'reload and judge again' in r7[2] and r0[0] == 409 and b'reload and judge again' in r0[2]
          and len(journal(os.path.join(U, J))) == 1, (r7[:1], r7[2][:200], r0[:1]))


def main():
    shutil.rmtree(TMP, ignore_errors=True)
    os.makedirs(TMP)
    TP.install_env()
    try:
        test_unit()
        test_pages()
        test_merged()
        test_banner()
        test_settled()
        test_versions()
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    print(f'{N[0] - len(FAILS)} of {N[0]} checks passed')
    return 1 if FAILS else 0


def test_all():
    assert main() == 0


if __name__ == '__main__':
    sys.exit(main())
