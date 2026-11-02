"""Review hub (registry + page; since 2026-10-27 the page served at '/' on BOTH listeners of the two-port server, 8765 review
and 8795 inspection) and the inspection hub (page from inspection_greenlit.csv, port 8795 before 2026-10-27).

Production (paths.PRODUCTION_WRITES and a production --out): build_review_hub.py --set ... in place for every served page,
then the removal of retired hub tiers (port of sublst_cumulative.sh:145-150, the hand edit of registry.json) and one
no-argument rebuild - all under ONE lock on <HUB_DIR>/.registry.lock (edge case E-33: two tiers serving at once lost hub
updates; the in-place script itself does not lock, so only exopipe3 writers are serialised).
Parity: build_review_hub.py hard-codes /data07/gh/phase3/transients/review_hub (build_review_hub.py:24), so the parity hub
is written here, under <out>/review_hub: registry.json = the production registry with this run's --set entries applied by
the same rule (port of build_review_hub.py:29-34) and the retired tiers removed, and index.html = render() of it, the page
build_review_hub.py writes from that registry.

THE PAGE (render: port of build_review_hub.py:43-90, exopipe_ops 4c42f91; Gregg 2026-10-27, ledger R-GREGG-20261027-HUB-
BOTH-LISTENERS-MULTIPLES-CHAIN-FORCED-EVERY-TIER, verbatim: "I want Stokes V on top; Stokes I below (i.e., as is); I want
1-hour, 10-min, 1-min, 10-sec going from left to right (i.e., as is); there is no need for a "multiples" page on the hub. [...]
The only other buttons under each tier on the Hub page, should be the HWD and 10pc forced photometry, with buttons for every
tier. Then, below both Stokes, a separate set of links for 1. Jovian bursts, 2. solar system and 3. "data processed:
Phase 1->2->3" and no other links! remove the SSH commands"): per Stokes row (V, then I) and per tier (serve.HUB_TIERS, left
to right) the tier's page - its combined view '<tier>-comb' when that entry is live, else '<tier>' - and under it the
'<tier>-forced' button (HWJ + 10 pc forced photometry); a tier without an entry is a greyed placeholder. Then the 1-hour
generation line of each Stokes, the three links (serve.HUB_LINKS) and the 'Page generated' note. No multiples, sub-LST,
preview or batch button (their registry entries stay: the serve scripts still --set them), no Round-2 link, NO TUNNEL LINE.
render() is also the reference of the content test (tests.registry_matches: index.html == render(registry.json) at the
page's own 'Page generated' time) and is drift-tested against the in-place script (tests/test_serve.py test_hub_page).

The registry's 'tunnel_extra' (ports served without a button: the Jupiter pages) is still kept up to date with
--tunnel-extra (build_review_hub.py:36-40 keeps the option and the key "for the serve scripts that still set it"); the page
no longer prints it.
Inspection hub: build_inspection_hub.py <dir> in place (it takes its output directory), production or parity.
Hub HTTP servers: started only when the port does not answer (production only; run_inspection_hub.sh:7; the 8765 hub has
no launcher, interface divergence D21).
"""
import contextlib
import fcntl
import html as _html
import json
import os
import re

from exopipe.phase3.core.record import Refusal
from exopipe.phase3.lib.serve import common, procs, scripts

# registry tier-key suffixes the page reads (build_review_hub.py:54, :63) and the tier of the generation line (:71)
COMB_SUFFIX = '-comb'           # the tier's combined view (every served snapshot batch in one page)
FORCED_SUFFIX = '-forced'       # the forced photometry at the HWJ + 10 pc targets
FORCED_TEXT = 'forced HWJ + 10 pc'
GEN_TIER = '1-hour'
PAGE_NOTE = 'from registry.json (build_review_hub.py)'      # the page's provenance note (:88)


def _S():
    from exopipe.phase3.config import CFG
    return CFG.serve


def apply_set(reg, S_, tier, kvs):
    """build_review_hub.py --set S tier k=v ... (build_review_hub.py:29-34)."""
    e = reg[S_].setdefault(tier, {})
    for kv in kvs:
        k, v = kv.split('=', 1)
        e[k] = (v == '1') if k == 'live' else (int(v) if k == 'port' else v)
    return reg


def remove_tiers(reg, tiers):
    for S_ in 'VI':
        for t in tiers:
            reg[S_].pop(t, None)
    return reg


def _dump(reg, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + '.part', 'w') as f:
        json.dump(reg, f, indent=1)
    os.replace(path + '.part', path)


# ---- the page (port of build_review_hub.py:43-90, exopipe_ops 4c42f91) ------------------------------------------------
def _button(e, txt):
    """build_review_hub.py:43-49."""
    title = f"{e.get('run', '')} {e.get('generation', '')} {e.get('served', '')}".strip()
    if not e.get('live'):
        title += ' (not yet served' + (f"; {e['note']}" if e.get('note') else '') + ')'
    href = e.get('href') or f'http://localhost:{e["port"]}/'
    cls = 'b' if e.get('live') else 'b off'
    return f'<a class="{cls}" href="{_html.escape(href)}" title="{_html.escape(title)}">{txt}</a>'


def tier_page(reg, S_, t):
    """The tier's page: its combined view when it has one (live), else the tier's own entry (build_review_hub.py:52-55)."""
    c = reg[S_].get(t + COMB_SUFFIX)
    return c if (c and c.get('live')) else reg[S_].get(t)


def _row(reg, S_):
    """build_review_hub.py:58-67."""
    cells = []
    for t in _S().HUB_TIERS:
        e = tier_page(reg, S_, t)
        top = _button(e, t) if e else f'<span class="b off" title="not yet served">{t}</span>'
        fo = reg[S_].get(t + FORCED_SUFFIX)
        sub = (_button(fo, FORCED_TEXT) if fo else
               f'<span class="b off" title="not yet served">{FORCED_TEXT}</span>')
        cells.append(f'<div class="col">{top}{sub}</div>')
    return ''.join(cells)


def _gen_text(reg, S_):
    """build_review_hub.py:70-72 (the script fails without a 1-hour entry; refused here)."""
    e = reg[S_].get(GEN_TIER)
    if e is None:
        raise Refusal(f'hub: the registry has no Stokes {S_} {GEN_TIER} entry - build_review_hub.py cannot render the page '
                      f'(its generation line)')
    return f"Stokes {S_} {GEN_TIER}: {e.get('run', '')}, {e.get('generation', '')}, served {e.get('served', '') or 'not yet'}"


def render(reg, generated):
    """The hub page build_review_hub.py writes from the registry ``reg`` at the time ``generated`` ('%Y-%m-%d %H:%M UT',
    serve.UTC_FMT) - byte for byte (build_review_hub.py:75-90)."""
    links = ' '.join(f'<a class="b" href="{href}">{txt}</a>' for txt, href in _S().HUB_LINKS)
    return f'''<!DOCTYPE html><html><head><meta charset="utf-8"><title>ExoPipe transient review hub</title>
<style>body{{font-family:system-ui;background:#0e1117;color:#d7dde3;margin:40px}} h1{{font-size:22px}} h2{{font-size:16px;color:#8b949e;margin:28px 0 6px}}
.row{{display:flex;gap:18px;margin:6px 0 14px}} .col{{display:flex;flex-direction:column;gap:6px;min-width:140px}}
.b{{display:inline-block;background:#161b22;border:1px solid #2d333b;border-radius:10px;padding:14px 22px;margin:2px 0;color:#3b82f6;text-decoration:none;font-size:15px;text-align:center}}
.b:hover{{background:#1c2128}} .b.off{{border-style:dashed;color:#555}} .gen{{color:#d7dde3;font-size:14px;margin:4px 0}}
.links{{margin-top:34px;display:flex;gap:18px;flex-wrap:wrap}}
.note{{color:#8b949e;font-size:13px;margin-top:30px;line-height:1.5;font-family:monospace;white-space:pre-wrap}}</style></head><body>
<h1>ExoPipe transient review</h1>
<h2>Stokes V</h2><div class="row">{_row(reg, 'V')}</div>
<h2>Stokes I</h2><div class="row">{_row(reg, 'I')}</div>
<div class="gen">{_gen_text(reg, 'V')}</div><div class="gen">{_gen_text(reg, 'I')}</div>
<div class="links">{links}</div>
<div class="note">Page generated {generated} {PAGE_NOTE}. Greyed buttons: not yet served.</div>
</body></html>
'''


def page_generated(page):
    """The 'Page generated <time>' of a hub page written from the registry, or None (not such a page)."""
    m = re.search(r'<div class="note">Page generated (\d{4}-\d\d-\d\d \d\d:\d\d UT) ' + re.escape(PAGE_NOTE), page)
    return m.group(1) if m else None


def first_difference(a, b):
    """Where text ``a`` first differs from ``b``: 'line N: <a excerpt> != <b excerpt>' (for the content test's detail)."""
    la, lb = a.split('\n'), b.split('\n')
    for i, (x, y) in enumerate(zip(la, lb)):
        if x != y:
            j = next((k for k in range(min(len(x), len(y))) if x[k] != y[k]), min(len(x), len(y)))
            return f'line {i + 1}: {x[max(0, j - 60):j + 60]!r} != {y[max(0, j - 60):j + 60]!r}'
    return f'{len(la)} lines != {len(lb)} lines'


def _write_page(path, text):
    """The page as build_review_hub.py:91 writes it (a .part file renamed over the page)."""
    with open(path + '.part', 'w') as f:
        f.write(text)
    os.replace(path + '.part', path)


@contextlib.contextmanager
def registry_lock(hub_dir):
    os.makedirs(hub_dir, exist_ok=True)
    fd = os.open(os.path.join(hub_dir, '.registry.lock'), os.O_RDWR | os.O_CREAT, 0o664)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def tunnel_extra_with(reg, ports):
    """The registry's tunnel_extra list with ``ports`` added (build_review_hub.py --tunnel-extra REPLACES the list,
    build_review_hub.py:38-40, so the union is passed), or None when every port is already listed. Since 4c42f91 the page
    prints no tunnel line; the key records the ports served without a button."""
    have = set(reg.get('tunnel_extra', []))
    buttons = {e['port'] for S_ in 'VI' for e in reg.get(S_, {}).values() if 'port' in e} | {reg.get('round2', {}).get('port')}
    need = {int(p) for p in ports} - have - buttons
    return sorted(have | need) if need else None


def update(plan, entries, remove=(), tunnel=(), log_path=None, unit=None):
    """Apply hub entries [(Stokes, hub tier, [k=v ...])], remove ``remove`` tiers and record the button-less ports
    ``tunnel`` in the registry's tunnel_extra. Returns (registry path, index.html path, planned argv lists)."""
    S = _S()
    planned = [['--set', s, t] + list(kvs) for s, t, kvs in entries]
    if plan.parity:
        src = os.path.join(S.HUB_DIR, 'registry.json')
        if not os.path.exists(src):
            raise Refusal(f'hub: no production registry at {src} to plan the parity hub from')
        reg = json.load(open(src))
        for s, t, kvs in entries:
            apply_set(reg, s, t, kvs)
        remove_tiers(reg, remove)
        te = tunnel_extra_with(reg, tunnel)
        if te is not None:
            reg['tunnel_extra'] = te
            planned.append(['--tunnel-extra', ','.join(map(str, te))])
        out = os.path.join(plan.hub_dir, 'registry.json')
        page = render(reg, common.utc_text())         # refuses before anything is written
        _dump(reg, out)
        index = os.path.join(plan.hub_dir, 'index.html')
        _write_page(index, page)
        return out, index, planned
    with registry_lock(plan.hub_dir):
        R = os.path.join(plan.hub_dir, 'registry.json')
        te = tunnel_extra_with(json.load(open(R)), tunnel)
        if te is not None:
            planned.append(['--tunnel-extra', ','.join(map(str, te))])
        for argv in planned:
            res = scripts.run('build_review_hub', argv, log=log_path, unit=unit)
            if not res.ok:
                raise Refusal(f'hub: build_review_hub.py {" ".join(argv[:3])} failed (rc {res.rc}): '
                              f'{scripts.last_line(res.tail)}')
        if remove:
            reg = json.load(open(R))
            _dump(remove_tiers(reg, remove), R)
        res = scripts.run('build_review_hub', [], log=log_path, unit=unit)
        if not res.ok:
            raise Refusal(f'hub: build_review_hub.py rebuild failed (rc {res.rc}): {scripts.last_line(res.tail)}')
    return os.path.join(plan.hub_dir, 'registry.json'), os.path.join(plan.hub_dir, 'index.html'), planned


def inspection_hub(plan, log_path=None, unit=None):
    """build_inspection_hub.py <dir> (run_inspection_hub.sh:5). Returns the index.html path."""
    res = scripts.run('build_inspection_hub', [plan.inspection_hub_dir], log=log_path, unit=unit)
    if not res.ok:
        raise Refusal(f'inspection hub: build_inspection_hub.py failed (rc {res.rc}): {scripts.last_line(res.tail)}')
    return os.path.join(plan.inspection_hub_dir, 'index.html')


def ensure_http_servers(plan):
    """Production only: start `python -m http.server PORT --bind 127.0.0.1` in the hub directories whose port does not
    answer (run_inspection_hub.sh:7). Returns notes."""
    S = _S()
    notes = []
    if plan.parity:
        return ['hub HTTP servers: not started in a parity run (production 8765 / 8795 serve the production hubs)']
    for port, d in ((S.HUB_PORT, plan.hub_dir), (S.INSPECTION_HUB_PORT, plan.inspection_hub_dir)):
        st, _ = procs.http_get(port, timeout=5.0)
        if st == 200:
            notes.append(f'hub HTTP {port}: answering (left as is)')
            continue
        if procs.port_in_use(port):
            raise Refusal(f'G13 hub port {port} is held but does not answer 200 - not replaced')
        p = procs.start([scripts.interpreter('default'), '-m', 'http.server', str(port), '--bind', '127.0.0.1'], d,
                        scripts.server_env({}), os.path.join(d, 'server.log'), append=True)
        ok, detail = procs.wait_ready(port, p, timeout=60, poll=2)
        if not ok:
            raise Refusal(f'G16 hub HTTP {port}: {detail}')
        notes.append(f'hub HTTP {port}: started pid {p.pid} in {d}')
    return notes
