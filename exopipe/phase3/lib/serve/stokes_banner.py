"""The cross-Stokes link at the top of a design-B source page (R-GREGG-20261124-CROSS-STOKES-LINK-TEST-APPROVED: "Show the link
at the top of each source page, naming and linking the other Stokes' page").

The link itself is lib/catalog/stokes_link.py: an I identity and a V identity of one tier are linked when the identity-level test
on their mean positions and covariances passes (P >= 0.97 within the 4-sigma cap, the other Stokes' identity density as the
background); one-to-many links form one component, named by its best-localized member and marked ambiguous. Each build writes
its link table beside the pages it serves: <run>/stokes_link/stokes_links.csv and stokes_link_pairs.csv (the 10-minute build:
lib/serve/sublst_build.stokes_link_stage; the deep chain: lib/catalog/deep_chain.py st_identity).

THE BANNER, in the two-port server (lib/serve/two_port.py run_request), for GET /source/<sid> of a review page whose registry
entry names its other-Stokes pages (ENTRY_KEY, written by lib/serve/register.page_entry under design B) and whose rows carry
stable identity IDs (lib/serve/label_identity.py identity_maps):
  linked       the other Stokes' identity (or identities) named and linked to the other Stokes' pages, with the pair's
               probability P, the offset in arcsec and in units of the combined position error, the offset percentile and the
               chance-coincidence probability (core/astrometry.py offset_pct_profile, p_chance_ellipse); "the localization
               combines both Stokes' detections";
  ambiguous    marked AMBIGUOUS, every member of the component listed, "positions are not combined";
  not linked   one line saying so, with the test and the values the link was computed with (its summary.json: config
               catalog_grouping P_IDENTITY and U_CAP);
  no table     one line naming the missing table (the build stages refuse to publish without it, so this is a run served
               from outside them).
The review page's own code is unchanged; pages without stable IDs (production's grouping) are byte for byte as before."""
import html
import os
import re
import threading
import urllib.parse

from exopipe.phase3.lib.catalog import stokes_link_tables as SLT

ENTRY_KEY = 'stokes_partner'           # registry entry: [{"port": legacy port, "stokes": "I", "kind": "single"}, ...]
SOURCE_PATH = '/source/'
_BODY = re.compile(r'<body[^>]*>')
_CACHE_ATTR = '_stokes_banner_cache'
_LOCK = threading.Lock()


def run_dir(entry):
    a = (entry or {}).get('argv') or []
    i = a.index('--run') + 1 if '--run' in a else len(a)
    return a[i] if i < len(a) else None


def tables(page):
    """(links {stokes_identity_id: row}, pairs [row], members {link_id: [row]}, (P threshold, u cap)) of the page's run,
    read through the link's one reader (lib/catalog/stokes_link_tables.py) from ONE version of the link (the directory
    stokes_link points at when read: a switch while reading changes nothing), once per version; the path of the missing
    table when the run has none (or one in another format)."""
    d = run_dir(page.entry)
    if not d:
        return None
    ld = SLT.link_dir(d)
    v = os.path.realpath(ld)
    try:
        key = (v,) + tuple((os.stat(p).st_mtime_ns, os.stat(p).st_size)
                           for p in (os.path.join(v, n) for n in (SLT.LINKS, SLT.PAIRS, SLT.SUMMARY)))
    except OSError:
        return os.path.join(ld, SLT.LINKS)
    with _LOCK:
        c = getattr(page, _CACHE_ATTR, None)
        if c is not None and c[0] == key:
            return c[1]
        try:
            rows = SLT.read_rows(os.path.join(v, SLT.LINKS), SLT.LINK_COLS)
            pairs = SLT.read_pairs(v)
            test = SLT.link_test_values(SLT.read_summary(v))
        except (SLT.TableRefused, KeyError, ValueError) as e:
            return f'{os.path.join(ld, SLT.LINKS)} ({e})'
        t = ({r['identity_id']: r for r in rows}, pairs, SLT.members(rows), test)
        setattr(page, _CACHE_ATTR, (key, t))
        return t


def _num(x, fmt):
    try:
        return format(float(x), fmt)
    except (TypeError, ValueError):
        return '?'


def _pages(name, partners, stokes):
    q = urllib.parse.quote(name)
    return ' '.join(f'<a href="http://localhost:{int(p["port"])}/source/{q}">{html.escape(p.get("kind") or "page")} page</a>'
                    for p in partners if p.get('stokes') == stokes)


def html_for(page, sid, gids):
    """The banner's HTML for source ``sid`` whose rows carry stable IDs ``gids``, or '' (no partner pages registered)."""
    partners = (page.entry or {}).get(ENTRY_KEY) or []
    if not partners or not gids:
        return ''
    other = partners[0]['stokes']
    mine = 'I' if other == 'V' else 'V'
    style = ('style="font-family:sans-serif;font-size:14px;border:2px solid {c};background:{b};padding:6px 10px;margin:4px 0 8px 0"')
    t = tables(page)
    if not isinstance(t, tuple):
        return (f'<div class="stokes-link" {style.format(c="#b00", b="#fee")}>Cross-Stokes link: no link table for this run '
                f'({html.escape(str(t))}: missing or not readable), so no Stokes {other} identity is named here.</div>')
    links, pairs, members, (p_thr, u_cap) = t
    parts = []
    for g in gids:
        r = links.get(g)
        if r is None:
            parts.append(f'Not linked to a Stokes {other} identity (cross-Stokes test: probability P &ge; {p_thr:g} within '
                         f'{u_cap:g} standard deviations of the combined position error).')
            continue
        comp = members.get(r['link_id'], [])
        mine_pairs = [p for p in pairs if p[f'{mine}_identity_id'] == g]
        lines = []
        for p in sorted(mine_pairs, key=lambda p: -float(p['p'] or 0)):
            n = p[f'{other}_identity_name']
            lines.append(f'<b>Stokes {other} {html.escape(n)}</b> ({_pages(n, partners, other)}): P = {_num(p["p"], ".3f")}, '
                         f'offset {_num(p["offset_arcsec"], ".0f")} arcsec = {_num(p["u"], ".2f")} times the combined '
                         f'position error (offset percentile {_num(p["offset_pct"], ".0f")}), chance-coincidence '
                         f'probability {_num(p["p_chance"], ".4f")}')
        if r['link_ambiguous'] == '1':
            names = '; '.join(f'Stokes {m["stokes"]} {html.escape(m["identity_name"])}' for m in
                              sorted(comp, key=lambda m: (m['stokes'], m['identity_name'])))
            head = (f'<b>AMBIGUOUS cross-Stokes link</b>: this Stokes {mine} identity is one of {len(comp)} identities linked '
                    f'one-to-many (component {html.escape(r["link_name"])}: {names}); their positions are not combined.')
        else:
            head = (f'Linked across Stokes (component {html.escape(r["link_name"])}); the localization combines both Stokes\' '
                    f'detections.')
        parts.append(head + ('<br>' + '<br>'.join(lines) if lines else ''))
    amb = any((links.get(g) or {}).get('link_ambiguous') == '1' for g in gids)
    col = ('#b60', '#fff4e0') if amb else (('#06c', '#eef6ff') if any(g in links for g in gids) else ('#888', '#f6f6f6'))
    return f'<div class="stokes-link" {style.format(c=col[0], b=col[1])}>' + '<hr>'.join(parts) + '</div>'


def insert(page, path, text, by_sid, first=()):
    """text of GET ``path`` with the banners after its <body> tag when the path is a source page of an identity with a stable
    ID: first the ones of ``first`` (callables (page, sid, gids) -> HTML; the merge-conflict banner, lib/serve/label_identity.py
    ambiguous_html), then the cross-Stokes link when the entry names its other-Stokes pages; otherwise text unchanged."""
    route = path.split('?', 1)[0]
    if not route.startswith(SOURCE_PATH):
        return text
    sid = urllib.parse.unquote(route[len(SOURCE_PATH):])
    g = by_sid.get(sid)
    if not g:
        return text
    gids = g.split(';')
    b = ''.join(f(page, sid, gids) for f in first) + html_for(page, sid, gids)
    m = _BODY.search(text)
    if not b or m is None:
        return text
    return text[:m.end()] + b + text[m.end():]


def partners_of(page, all_pages):
    """The ENTRY_KEY value of a plan page (lib/serve/plan.Page): the review pages of the same tier showing the other Stokes
    (singles first), [] for a page without them."""
    if (page.family != 'review' or page.stokes not in ('I', 'V') or page.kind not in ('single', 'multi')
            or page.tier not in ('deep', 'sublst')):
        return []
    other = 'V' if page.stokes == 'I' else 'I'
    out = [dict(port=p.port, stokes=other, kind=p.kind) for p in all_pages
           if p.family == 'review' and p.stokes == other and p.kind in ('single', 'multi') and p.tier == page.tier]
    return sorted(out, key=lambda d: (d['kind'] != 'single', d['port']))
