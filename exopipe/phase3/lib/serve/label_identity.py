"""A label lands on the identity it was made on, even when its form was shown before a rebuild renamed the identities.

THE GAP (CUTOVER_READINESS, merged 2026-11-04, cutover blocker): "Labels made after the swap land under OLD page ids and are
not carried. Measured at the first design-B serve: 57% of I / 38% of V stale-page labels show on no page, and 143 I / 16 V show
on a DIFFERENT identity that inherits the old name (as Gregg's own judgement; later carries never flag it). The fix is in serve /
page code." A design-B page id is the identity's DISPLAY name, recomputed from the members' mean position on every grouping run
(lib/catalog/grouping.py display_names); the identity's STABLE ID is the grouping's identity_id, carried on every catalog row
as group_identity_id (lib/catalog/wraps/builder_grouping.py SOURCE_COLS). A browser tab opened before a rebuild still posts the
OLD display name, and the review page (classify_transients.py handle_label) reads that name in the NEW build: an absent name is
"bad request" (the label is lost), a name another identity now carries records the label on that identity.

THE FIX, in the two-port server (lib/serve/two_port.py run_request), carrying labels by group_identity_id; the review page's
own code is unchanged:
  STAMP   every hidden source_id input of a served review page gains a hidden group_identity_id input holding the page's
          identity key: the stable ID its rows carry (group_identity_id; design B serves one identity per page, so one ID - a
          page holding several is keyed by all of them, sorted, ';'-joined). A page whose rows carry none is not stamped
          (production's grouping: every page byte for byte as before).
  RESOLVE a POST that carries the stamp is read against the page build that answers it:
          - its source_id names the stamped identity: sent on unchanged;
          - exactly one other page of this build has the stamped key (the identity was renamed): source_id is rewritten to
            that page, so the label lands on the identity it was made on;
          - no page has it because a grouping run MERGED it into another identity (the build's merge table, <run>/stokes_link/
            identity_merges.csv, lib/catalog/stokes_link.py; R-GREGG-20261124-CROSS-STOKES-LINK-TEST-APPROVED, Q2: "carry
            labels along the grouping's merge events"): the label follows the merge to the survivor's page when exactly one
            page holds the survivor. A source-scope label whose verdict (confidence, reason, second reason) differs from a
            verdict the survivor already holds is NOT made the survivor's verdict ("When the merged identities carry
            conflicting verdicts, show both and mark the source ambiguous; never pick one"): it is kept in the labels
            directory's merge_conflicts.jsonl with the verdicts it conflicts with, HTTP 409 says so, and the survivor's page
            opens with both verdicts, marked AMBIGUOUS (ambiguous_html), until a reviewer judges it on its page;
          - no page or several pages have it otherwise (a regroup issued new IDs, the identity retired, it was merged into an
            identity that retired since - the merge table records that survivor as 'retired' -, or it gained or lost a
            member identity): REFUSED with HTTP 409 - nothing is recorded, the message names the identity and echoes the label,
            the reviewer reloads.
          A POST /label WITHOUT the stamp to a stamped build is REFUSED the same way when its source_id names a stamped page
          or no page of the build (a form shown before stamping, e.g. by a production page before the first design-B serve):
          its source_id cannot be checked against the identity it was made on, and a guess could record the label on another
          identity. A page of the build without a stable ID was served unstamped, so its labels pass as before.
  Labels are never sent to another identity and never dropped silently: either they land on their own identity or the
  reviewer is told that nothing was recorded.

NOT COVERED (stated in the PR): notes, note deletions and the reviewed mark are posted by page scripts from a data-sid attribute
(classify_transients.py:2949, 4875, 4883), not from a form, so they carry no stamp and pass as before; covering them needs the
page code to send the stamp. The per-person lists post their form (FormData) and so carry the stamp."""
import collections
import html
import json
import os
import re
import threading
import time
import urllib.parse

from exopipe.phase3.lib.catalog import stokes_link_tables as SLT

FIELD = 'group_identity_id'
LABEL_PATH = '/label'
REFUSED_STATUS = 409
# the review page's hidden source_id input, exactly as classify_transients.py writes it (:4018, :4181, :4201, :4833, :4925,
# :4940, :4946); the value is html.escape(sid)
_HIDDEN_SID = re.compile(r'<input type="hidden" name="source_id" value="([^"]*)">')
_MAPS_ATTR = '_label_identity_maps'
_MERGES_ATTR = '_label_identity_merges'
MERGE_CONFLICTS = 'merge_conflicts.jsonl'      # in the labels directory, beside judgements.jsonl
VERDICT = ('judge_confidence', 'judge_fp_reason', 'judge_fp_reason_2')
_LOCK = threading.Lock()


def identity_maps(app):
    """({page id: identity key}, {identity key: [page ids]}, {every page id}) of a built review page's App: a page's key is the
    group_identity_id its rows carry (App.sources[sid].bands: one catalog row per band), several sorted and ';'-joined; a page
    whose rows carry none is in neither map (not stamped, never a target)."""
    by_sid, by_gid = {}, collections.defaultdict(list)
    sources = getattr(app, 'sources', None) or {}
    for sid, src in list(sources.items()):
        ids = {(r.get(FIELD) or '').strip() for r in (getattr(src, 'bands', None) or {}).values() if isinstance(r, dict)}
        ids.discard('')
        if ids:
            g = ';'.join(sorted(ids))
            by_sid[sid] = g
            by_gid[g].append(sid)
    return by_sid, {g: sorted(v) for g, v in by_gid.items()}, frozenset(sources)


def page_maps(page):
    """identity_maps of a two-port Page's App, computed once per build (a build's sources never change)."""
    m = getattr(page, _MAPS_ATTR, None)
    if m is None:
        app = getattr(getattr(getattr(page, 'mod', None), 'Handler', None), 'app', None)
        m = identity_maps(app) if app is not None else ({}, {}, frozenset())
        setattr(page, _MAPS_ATTR, m)
    return m


def _argv_value(entry, flag):
    a = (entry or {}).get('argv') or []
    i = a.index(flag) + 1 if flag in a else len(a)
    return a[i] if i < len(a) else None


def merge_map(page):
    """{merged identity ID: the LIVE ID it stands for} of the page's run (<run>/stokes_link/identity_merges.csv through the
    link's one reader, lib/catalog/stokes_link_tables.read_merges, once per version of the file): a merged ID whose chain ends
    at a RETIRED identity is not in it, so a label made on it has no page and is refused (HTTP 409); {} when the run has none
    (production's grouping)."""
    d = _argv_value(getattr(page, 'entry', None), '--run')
    p = os.path.join(SLT.link_dir(d), SLT.MERGES) if d else None
    try:
        st = os.stat(p) if p else None
    except OSError:
        st = None
    if st is None:
        return {}
    key = (st.st_mtime_ns, st.st_size)
    with _LOCK:
        c = getattr(page, _MERGES_ATTR, None)
        if c is None or c[0] != key:
            c = (key, SLT.read_merges(SLT.link_dir(d)))
            setattr(page, _MERGES_ATTR, c)
        return c[1]


def _app(page):
    return getattr(getattr(getattr(page, 'mod', None), 'Handler', None), 'app', None)


def held_verdicts(page, sid):
    """[the source-scope records the page's journal holds now for page id sid, every presentation] (App.state())"""
    app = _app(page)
    if app is None or not hasattr(app, 'state'):
        return []
    return [st['source'] for (s, _p), st in sorted(app.state().items()) if s == sid and st.get('source')]


def conflicts_path(page):
    """<labels directory>/merge_conflicts.jsonl of the page (the directory of its App's journal), or None"""
    j = getattr(_app(page), 'journal', None)
    return os.path.join(os.path.dirname(j), MERGE_CONFLICTS) if j else None


def append_conflict(path, rec):
    with open(path, 'a') as f:
        f.write(json.dumps(rec, sort_keys=True) + '\n')
        f.flush()
        os.fsync(f.fileno())


def _vtext(v):
    t = v.get('judge_confidence') or '?'
    for k in ('judge_fp_reason', 'judge_fp_reason_2'):
        if v.get(k):
            t += f' / {v[k]}'
    return t


def conflict_settled(rec, held):
    """True when a merge-conflict record ``rec`` is settled by the verdicts ``held`` now for its page (source-scope journal
    records): a verdict made on the page itself whose (verdict, time) the record does not list (a reviewer judged it since),
    or a CARRIED verdict (carried_from_run) whose verdict the record does not list. A carried verdict the record lists is the
    conflict itself carried to a rebuilt page (its time is the carry's), so it settles nothing. Shared by the page banner
    (ambiguous_html) and the journal carry, which carries an unsettled record to the next build (lib/catalog/grouping_stage.py
    carry_judgements)."""
    def vk(v):
        return tuple(v.get(k) or '' for k in VERDICT)
    listed = {vk(v) + (str(v.get('ts', '')),) for v in rec.get('verdicts', [])}
    kinds = {vk(v) for v in rec.get('verdicts', [])}
    for v in held:
        if v.get('carried_from_run'):
            if vk(v) not in kinds:
                return True
        elif vk(v) + (str(v.get('ts', '')),) not in listed:
            return True
    return False


def ambiguous_html(page, sid, gids):
    """The AMBIGUOUS banner of page id sid: the newest merge-conflict record of the identity (merge_conflicts.jsonl: written by
    resolve for a stale form, by grouping_stage.carry_judgements for merged pages with conflicting verdicts, and carried by it
    to every later build while unsettled) while it is not settled (conflict_settled: a verdict judged on the page since, or a
    carried verdict the record does not list); '' otherwise."""
    p = conflicts_path(page)
    if not p or not os.path.exists(p):
        return ''
    last = None
    with open(p) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r.get('source_id') == sid or r.get('group_identity_id') in gids:
                    last = r
    if last is None or conflict_settled(last, held_verdicts(page, sid)):
        return ''
    items = ''.join(f'<li><b>{html.escape(_vtext(v))}</b>, made on {html.escape(str(v.get("source", "")))} '
                    f'({html.escape(str(v.get("identity", "")))}) at {html.escape(str(v.get("ts", "")))}'
                    + (f': {html.escape(str(v.get("judge_note")))}' if v.get('judge_note') else '') + '</li>'
                    for v in last.get('verdicts', []))
    how = ('a label from a page shown before the merge' if last.get('origin') == 'post'
           else 'the carry of the earlier pages\' judgements')
    return ('<div class="merge-conflict" style="font-family:sans-serif;font-size:14px;border:2px solid #b00;background:#fee;'
            'padding:6px 10px;margin:4px 0 8px 0"><b>AMBIGUOUS</b>: identities merged into this one carry conflicting '
            f'verdicts ({html.escape(how)}, {html.escape(str(last.get("ts", "")))}); neither was chosen:<ul style="margin:2px 0">'
            f'{items}</ul>Judge the source here to settle it.</div>')


def stamp(text, by_sid):
    """text with a hidden group_identity_id input after every hidden source_id input whose page has a stable ID."""
    if not by_sid or 'name="source_id"' not in text:
        return text

    def _one(m):
        g = by_sid.get(html.unescape(m.group(1)))
        if not g:
            return m.group(0)
        return m.group(0) + f'<input type="hidden" name="{FIELD}" value="{html.escape(g)}">'
    return _HIDDEN_SID.sub(_one, text)


def _echo(form):
    keys = ('scope', 'confidence', 'fp_reason', 'fp_reason_2', 'note', 'which', 'on', 'action', 'text')
    return ''.join(f'<tr><td>{html.escape(k)}</td><td>{html.escape(form[k][0])}</td></tr>' for k in keys if form.get(k))


def refusal(why, form, links=()):
    """(status, headers, body) of a refused POST: nothing was recorded, the reason, what was sent and where to go."""
    go = ''.join(f'<li><a href="/source/{urllib.parse.quote(s)}">{html.escape(s)}</a></li>' for s in links)
    body = (f'<html><head><title>Not recorded</title></head><body style="font-family:sans-serif;max-width:760px;margin:24px">'
            f'<h2>Nothing was recorded</h2><p>{html.escape(why)}</p>'
            + (f'<p>The identity is now on:</p><ul>{go}</ul>' if go else '')
            + '<p>Reload the page and send it again.</p>'
            + (f'<p>What you sent:</p><table border="1" cellpadding="4">{_echo(form)}</table>' if _echo(form) else '')
            + '</body></html>')
    return REFUSED_STATUS, [('Content-Type', 'text/html; charset=utf-8')], body.encode('utf-8')


def ambiguous_response(sid, g, target, live, posted, held):
    """(status, headers, body) of a label kept as a conflicting verdict of a merged identity"""
    vs = ''.join(f'<li>{html.escape(_vtext(v))} (held by {html.escape(target)})</li>' for v in held)
    body = (f'<html><head><title>Kept as a conflicting verdict</title></head><body style="font-family:sans-serif;max-width:760px;'
            f'margin:24px"><h2>Kept as a conflicting verdict, not as the source\'s verdict</h2><p>The identity this label was '
            f'made on ({html.escape(g)}, shown then as {html.escape(sid)!s}) was merged into the identity of page '
            f'<a href="/source/{urllib.parse.quote(target)}">{html.escape(target)}</a> ({html.escape(live)}), which holds a '
            f'different verdict:</p><ul>{vs}</ul><p>Your verdict, <b>{html.escape(_vtext(posted))}</b>, was kept with them. Both '
            f'are shown at the top of that page and the source is marked AMBIGUOUS; neither verdict was chosen. Judge it on '
            f'its page to settle it.</p></body></html>')
    return REFUSED_STATUS, [('Content-Type', 'text/html; charset=utf-8')], body.encode('utf-8')


def _rewrite(text, to):
    pairs = [(k, to if k == 'source_id' else v) for k, v in urllib.parse.parse_qsl(text, keep_blank_values=True)]
    return urllib.parse.urlencode(pairs).encode('utf-8')


def mappable(page, body):
    """True when a POST body carries the stamp (FIELD) and the stamped identity has exactly one page in the page's build:
    its own, the one it was renamed to, or the survivor it was merged into (merge_map). The two-port server asks this BEFORE
    its stale-form check (two_port.serve_request, PAGE_TOKEN_FIELD; PR 155): a form shown by an earlier version whose
    identity maps is resolved here (resolve), and only an unmappable form gets the stale-form 409."""
    by_sid, by_gid, _pages = page_maps(page)
    if not by_gid:
        return False
    try:
        form = urllib.parse.parse_qs((body or b'').decode('utf-8'), keep_blank_values=True)
    except UnicodeDecodeError:
        return False
    g = (form.get(FIELD, [''])[0] or '').strip()
    if not g:
        return False
    if by_sid.get(form.get('source_id', [''])[0]) == g or len(by_gid.get(g, [])) == 1:
        return True
    live = merge_map(page).get(g) if not by_gid.get(g) else None
    return bool(live) and len(by_gid.get(live, [])) == 1


def resolve(path, body, maps, merges=None, held=None, record=None):
    """(body to send on, refusal or None, what happened) of one POST to a review page whose build has ``maps``
    (identity_maps). merges: {merged ID: live ID} of the build (merge_map); held(page id): the source-scope records it holds
    now (held_verdicts); record(rec): keeps a conflicting verdict (append_conflict). what: 'unstamped' | 'same' |
    'carried <old> -> <new>' | 'merged <old> -> <new>' | 'ambiguous <old> -> <new>' | 'refused'."""
    by_sid, by_gid, pages = maps
    route = path.split('?', 1)[0]
    try:
        text = body.decode('utf-8')
    except UnicodeDecodeError:
        return body, None, 'unstamped'
    form = urllib.parse.parse_qs(text, keep_blank_values=True)
    g = (form.get(FIELD, [''])[0] or '').strip()
    sid = form.get('source_id', [''])[0]
    if not g:
        if route == LABEL_PATH and by_sid and (sid in by_sid or sid not in pages):
            return body, refusal(
                f'This label came from a form shown before the page was rebuilt with stable identity IDs (page id {sid!r}), so '
                f'the identity it was made on cannot be checked; it was not recorded, to keep it off another identity.', form), \
                'refused'
        return body, None, 'unstamped'
    if by_sid.get(sid) == g:
        return body, None, 'same'
    there = by_gid.get(g, [])
    if len(there) == 1:
        return _rewrite(text, there[0]), None, f'carried {sid} -> {there[0]}'
    live = (merges or {}).get(g) if not there else None
    target = by_gid.get(live, []) if live else []
    if len(target) == 1:
        to = target[0]
        if route == LABEL_PATH and form.get('scope', [''])[0] == 'source' and form.get('confidence', [''])[0]:
            posted = dict(judge_confidence=form['confidence'][0], judge_fp_reason=form.get('fp_reason', [''])[0],
                          judge_fp_reason_2=form.get('fp_reason_2', [''])[0])
            have = list(held(to)) if held else []
            differ = [v for v in have if tuple(v.get(k) or '' for k in VERDICT) != tuple(posted[k] for k in VERDICT)]
            if differ:
                now = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
                rec = dict(ts=now, origin='post', source_id=to, group_identity_id=live, merged_id=g,
                           verdicts=[dict({k: v.get(k, '') for k in VERDICT + ('judge_note',)}, source=to, identity=live,
                                          ts=v.get('ts', '')) for v in differ]
                           + [dict(posted, judge_note=form.get('note', [''])[0].strip(), source=sid, identity=g, ts=now)])
                if record is None:
                    return body, refusal(f'The identity this label was made on ({g}, shown then as {sid!r}) was merged into '
                                         f'{to!r}, which holds a different verdict; there is nowhere to keep both.', form,
                                         target), 'refused'
                record(rec)
                return body, ambiguous_response(sid, g, to, live, posted, differ), f'ambiguous {sid} -> {to}'
        return _rewrite(text, to), None, f'merged {sid} -> {to}'
    why = (f'The page was rebuilt since this form was shown. The identity it was made on ({g}, shown then as {sid!r}) is '
           + ('on several pages now.' if there else 'on no page of the rebuilt page (merged into another identity, retired, '
                                                    'or given a new ID by a regrouping).'))
    return body, refusal(why, form, there), 'refused'
