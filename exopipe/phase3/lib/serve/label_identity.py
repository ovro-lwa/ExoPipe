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
          - no page or several pages have it (a regroup issued new IDs, the identity merged into another or retired, or
            gained or lost a member identity): REFUSED with HTTP 409 - nothing is recorded, the message names the identity and
            echoes the label, the reviewer reloads.
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
import re
import urllib.parse

FIELD = 'group_identity_id'
LABEL_PATH = '/label'
REFUSED_STATUS = 409
# the review page's hidden source_id input, exactly as classify_transients.py writes it (:4018, :4181, :4201, :4833, :4925,
# :4940, :4946); the value is html.escape(sid)
_HIDDEN_SID = re.compile(r'<input type="hidden" name="source_id" value="([^"]*)">')
_MAPS_ATTR = '_label_identity_maps'


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


def resolve(path, body, maps):
    """(body to send on, refusal or None, what happened) of one POST to a review page whose build has ``maps``
    (identity_maps). what: 'unstamped' | 'same' | 'carried <old> -> <new>' | 'refused'."""
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
        pairs = [(k, there[0] if k == 'source_id' else v)
                 for k, v in urllib.parse.parse_qsl(text, keep_blank_values=True)]
        return urllib.parse.urlencode(pairs).encode('utf-8'), None, f'carried {sid} -> {there[0]}'
    why = (f'The page was rebuilt since this form was shown. The identity it was made on ({g}, shown then as {sid!r}) is '
           + ('on several pages now.' if there else 'on no page of the rebuilt page (merged into another identity, retired, '
                                                    'or given a new ID by a regrouping).'))
    return body, refusal(why, form, there), 'refused'
