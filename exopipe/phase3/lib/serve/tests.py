"""Content tests of the product classes `exopipe3 serve` writes (spec serve.md 6). Each test reads the product's CONTENT and
is read-only (verify may run it on production files). Registered with @content_test; core/validate discovers this module.

  served_queue       nonempty, protocol, sections, fork_equal
  served_view        stokes_equals_catalogue, unique_keys, complete, review_dir_page, tier_rules
  judgement_journal  parses_append_only, carry_consistent, hold_rule
  hub_page           registry_matches (review hub registry.json / index.html), greenlit_only (inspection hub index.html)
  served_page        live, same_process, sources_listed      (record <out>/_records/serve/pages/<tier>_<port>.json)
  stopped_page       not_listening                           (record of `serve --stop`)
  sublst_cumulative  merge_invariants, figures, served_tests_c_d
  review_copy        byte_copy, no_excluded_rows
  exclusion_check    gate_passed
A test that does not apply to the given file (e.g. fork_equal on a sub-LST queue) passes with detail 'n/a: ...'.
"""
import ast
import csv
import glob
import json
import os
import re
import subprocess
import tempfile
from collections import Counter, defaultdict

from exopipe.phase3.core.validate import content_test

WB = ('Red', 'Green', 'Blue', 'Full')


def _S():
    from exopipe.phase3.config import CFG
    return CFG.serve


def _rows(p):
    with open(p, newline='') as f:
        return list(csv.DictReader(f))


def _body(r):
    """sublst_reserve_test_served.py:34 - every column but review_dir / stokes; an absent column, '' and 'None' are one
    'no value' (ledger SUBLST-RESERVE-23H-SERVED-TEST-FIX)."""
    return frozenset((k, v) for k, v in r.items() if k not in ('review_dir', 'stokes') and v not in (None, '', 'None'))


# =====================================================================================================================
# served_queue
# =====================================================================================================================
_QNAME = re.compile(r'^queue_p3s3_(?P<x>[A-Za-z0-9_]+)\.txt$')


def _queue_kind(path):
    """(U, kind, S, K): kind tier|stokes|multi|other for <U>/judgements/queue_p3s3_<X>.txt."""
    U = os.path.dirname(os.path.dirname(os.path.abspath(path)))
    m = _QNAME.match(os.path.basename(path))
    if not m:
        return U, 'other', None, None
    x = m['x']
    if x in ('V', 'I'):
        return U, 'stokes', x, None
    if re.match(r'^[VI]_multi$', x):
        return U, 'multi', x[0], None
    if re.match(r'^[VI]_(wb|sb)$', x):
        return U, 'tier', x[0], x
    return U, 'other', x[:1], x


def _is_deep_fork_run(U, S_):
    return any(os.path.isdir(f'{U}/source_catalogs_{S_}_{ft}') for ft in _S().DEEP_FORK_TIERS)


def _run_tier(U):
    """'sublst' for a sub-LST run (Run_SLST*, or one carrying pertie_detections.csv), else 'deep'."""
    b = os.path.basename(os.path.normpath(U))
    return 'sublst' if (b.startswith('Run_SLST') or os.path.exists(os.path.join(U, 'pertie_detections.csv'))) else 'deep'


def _parse(path):
    """[(kind, text)] lines: ('hdr', line) for '#' lines, ('id', line) for id lines."""
    out = []
    with open(path) as f:
        for l in f:
            l = l.rstrip('\n')
            if not l.strip():
                continue
            out.append(('hdr' if l.startswith('#') else 'id', l))
    return out


@content_test('served_queue')
def nonempty(path, **ctx):
    """At least one id line starting 'J' (G1; p3s3_serve_stokes.sh:57)."""
    n = sum(1 for k, l in _parse(path) if k == 'id' and l.startswith('J'))
    return n > 0, n, f'{n} id lines' if n else 'no line starting J (empty queue: never served)'


_SEC_CACHE = {}


def _stokes_queue_sections():
    """The section lines p3s3_stokes_queue.py writes (its SEC dict, read from the in-place source with ast)."""
    if 'v' not in _SEC_CACHE:
        try:
            from exopipe.phase3.lib.serve import scripts
            tree = ast.parse(open(scripts.path('stokes_queue')).read())
            val = None
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign) and any(getattr(t, 'id', None) == 'SEC' for t in node.targets):
                    val = ast.literal_eval(node.value)
            _SEC_CACHE['v'] = set(val.values()) if isinstance(val, dict) else None
        except Exception:
            _SEC_CACHE['v'] = None
    return _SEC_CACHE['v']


def _known_selection_lines():
    S = _S()
    H = S.QUEUE_HEADERS
    known = set(H['deep_tier'][1:4]) | set(H['sublst_tier'][1:3])
    for ft in S.DEEP_FORK_TIERS:
        T = ft[1:].upper()
        known |= {H['deep_multi'][1].format(T=T), H['deep_multi'][2].format(T=T)}
    sec = _stokes_queue_sections()
    if sec:
        known |= sec
    return known, sec is not None


def _name_patterns():
    H = _S().QUEUE_HEADERS
    pats = []
    for t in (H['deep_tier'][0], H['sublst_tier'][0], H['deep_multi'][0], '# queue_name: p3s3_{S}_deep'):
        rx = re.escape(t)
        for f in ('K', 'S', 'lsts'):
            rx = rx.replace(re.escape('{' + f + '}'), '.+')
        pats.append(re.compile('^' + rx + '$'))
    return pats


@content_test('served_queue')
def protocol(path, **ctx):
    """First line '# queue_name: ...' (a sub-LST multi-integration extract starts at its MULTI-INTEGRATION section line,
    sublst_cumulative.sh:79); every other '#' line is '# tiers: ...' or a known section line (QUEUE_HEADERS and the
    sections of p3s3_stokes_queue.py); no id line twice."""
    lines = _parse(path)
    if not lines:
        return False, 0, 'empty file'
    U, kind, S_, K = _queue_kind(path)
    first = lines[0][1]
    marker = _S().QUEUE_HEADERS['sublst_multi_marker']
    known, have_sec = _known_selection_lines()
    bad = []
    if first.startswith('# queue_name: '):
        if not any(p.match(first) for p in _name_patterns()):
            bad.append(f'queue_name not one serve writes: {first[:80]}')
        rest = lines[1:]
    elif kind == 'multi' and first.startswith(marker):
        rest = lines
    else:
        return False, None, f'first line is not "# queue_name: ...": {first[:80]}'
    for k, l in rest:
        if k != 'hdr':
            continue
        if l.startswith('# tiers: '):
            continue
        if l.startswith('# selection: '):
            if l not in known and (have_sec or not re.search(r'MULTI-|SINGLE|MARGINAL|RECOVERED', l.upper())):
                bad.append(f'unknown section line: {l[:90]}')
            continue
        bad.append(f'unexpected header line: {l[:80]}')
    ids = Counter(l for k, l in lines if k == 'id')
    dup = [i for i, c in ids.items() if c > 1]
    if dup:
        bad.append(f'{len(dup)} ids listed twice, e.g. {dup[0]}')
    return not bad, len(ids), '; '.join(bad[:3]) if bad else f'{len(ids)} ids, protocol ok'


def _pages_input_files(U, kind, S_, K):
    if kind == 'tier':
        return [f'{U}/source_catalogs/pages_input_{K}.csv']
    if kind == 'stokes':
        return [f'{U}/source_catalogs/pages_input_{S_}_{t}.csv' for t in ('wb', 'sb')]
    if kind == 'multi':
        if _is_deep_fork_run(U, S_):
            return [f'{U}/source_catalogs/pages_input_{S_}_{ft}.csv' for ft in _S().DEEP_FORK_TIERS]
        return [f'{U}/source_catalogs/pages_input_{S_}_{t}.csv' for t in _S().SUBLST_TIERS]
    return []


def _section_of(line):
    u = line.upper()
    if 'MARGINAL' in u:
        return 'marginal'
    if line.startswith('# selection: RECOVERED'):
        return 'recovered'
    if 'MULTI-EPOCH' in u or 'MULTI-INTEGRATION' in u:
        return 'multi'
    if 'SINGLE' in u:
        return 'single'
    return 'other'


@content_test('served_queue')
def sections(path, **ctx):
    """Ids of a multi section have n_det >= MULTI_MIN_NDET in the pages input, ids of a single section fewer (missing row
    = 1, as the launchers' nd.get(i, 1)); marginal ids (label 'marginal') only in the marginal section, which is last."""
    S = _S()
    U, kind, S_, K = _queue_kind(path)
    files = [p for p in _pages_input_files(U, kind, S_, K) if os.path.exists(p)]
    if kind == 'other':
        return True, None, 'n/a: not a queue serve writes'
    deep = (ctx.get('tier') or _run_tier(U)) == 'deep'
    if not files:
        return False, None, f'no pages input for {os.path.basename(path)} ({", ".join(_pages_input_files(U, kind, S_, K))})'
    nd, lab = {}, {}
    for p in files:
        for r in _rows(p):
            pid = f"{r['_sid']}_{r['lst']}_{r['date']}_{r['band']}"
            nd[pid] = int(float(r.get('n_det') or 1))
            lab[pid] = r.get('label') or ''
    cur, order, bad = 'single', [], []
    first = True
    for k, l in _parse(path):
        if k == 'hdr':
            if l.startswith('# selection:'):
                cur = _section_of(l)
                if not order or order[-1] != cur:
                    order.append(cur)
            continue
        if first and kind == 'multi' and not order:
            cur = 'multi'
        first = False
        i = l.split()[0]
        n = nd.get(i, 1)
        if cur == 'multi' and n < S.MULTI_MIN_NDET:
            bad.append(f'{i} in a multi section with n_det {n}')
        elif cur == 'single' and n >= S.MULTI_MIN_NDET:
            bad.append(f'{i} in a single section with n_det {n}')
        if deep and lab.get(i) == S.MARGINAL_LABEL and cur != 'marginal' and kind in ('tier', 'stokes'):
            bad.append(f'{i} is labelled marginal but sits in the {cur} section')
        if cur == 'marginal' and lab.get(i) != S.MARGINAL_LABEL:
            bad.append(f'{i} in the marginal section without the marginal label')
    if 'marginal' in order and order[-1] != 'marginal' and not (order[-1] == 'recovered' and order[-2:-1] == ['marginal']):
        bad.append(f'marginal section not last ({order})')
    return not bad, len(bad), '; '.join(bad[:3]) if bad else f'sections {order} consistent with the pages input'


@content_test('served_queue')
def fork_equal(path, **ctx):
    """Deep *_multi.txt: the (J-name, band) set of the queue equals the fork catalogues' survivors set and is not empty
    (multiples_fork.py:640-654; ALL_ files skipped as there)."""
    U, kind, S_, K = _queue_kind(path)
    if kind != 'multi' or not _is_deep_fork_run(U, S_):
        return True, None, 'n/a: not a deep fork multiples queue'
    qs = {tuple(l.strip().rsplit('_', 3)[i] for i in (0, 3)) for k, l in _parse(path) if k == 'id'}
    cs = set()
    for ft in _S().DEEP_FORK_TIERS:
        for sp in glob.glob(f'{U}/source_catalogs_{S_}_{ft}/*_sources_survivors.csv'):
            if 'ALL_' in sp:
                continue
            cs |= {(r['source_id'], r['band']) for r in _rows(sp)}
    ok = qs == cs and len(qs) > 0
    return ok, len(qs), (f'queue {len(qs)} sources = fork catalogues {len(cs)}' if ok else
                         f'queue {len(qs)}, fork catalogues {len(cs)}; only in queue {len(qs - cs)}, only in catalogues '
                         f'{len(cs - qs)}, e.g. {sorted(qs ^ cs)[:3]}')


# =====================================================================================================================
# served_view
# =====================================================================================================================
_VNAME = re.compile(r'^source_catalogs_(?P<S>[VI])(?P<multi>_multi)?$')


def _view_kind(path, ctx):
    """(U, S, kind single|multi, tier deep|sublst, tiers) of a view directory."""
    d = os.path.abspath(path.rstrip('/'))
    m = _VNAME.match(os.path.basename(d))
    if not m:
        return None
    parent = os.path.dirname(d)
    U = os.path.dirname(parent) if os.path.basename(parent) == 'views' else parent
    S_ = m['S']
    multi = bool(m['multi'])
    tier = ctx.get('tier') or ('deep' if os.path.basename(parent) == 'views' else _run_tier(U))
    if multi and tier == 'deep':
        tiers = _S().DEEP_FORK_TIERS
    elif tier == 'deep':
        tiers = _S().DEEP_SINGLES_TIERS
    else:
        tiers = _S().SUBLST_TIERS
    return U, S_, 'multi' if multi else 'single', tier, tiers


def _tier_survivors(U, S_, tiers):
    out = []
    for t in tiers:
        for p in glob.glob(f'{U}/source_catalogs_{S_}_{t}/*_sources_survivors.csv'):
            if 'ALL_' not in p:
                out.extend(_rows(p))
    return out


def _view_rows(path):
    out = []
    for p in sorted(glob.glob(os.path.join(path, '*_sources_survivors.csv'))):
        out.append((p, _rows(p)))
    return out


@content_test('served_view')
def stokes_equals_catalogue(path, **ctx):
    """Every survivors row equals (all columns but review_dir / stokes) a row of its own Stokes' tier catalogue and of no
    other Stokes (sublst_reserve_test_served.py:63-80, D1)."""
    vk = _view_kind(path, ctx)
    if vk is None:
        return False, None, f'not a view directory: {path}'
    U, S_, kind, tier, tiers = vk
    cat = {X: defaultdict(set) for X in 'VI'}
    for X in 'VI':
        for r in _tier_survivors(U, X, tiers):
            cat[X][(r['source_id'], r['band'])].add(_body(r))
    n = wrong = 0
    ex = []
    for p, rr in _view_rows(path):
        for r in rr:
            n += 1
            k = (r['source_id'], r['band'])
            st = {x for x in 'VI' if _body(r) in cat[x].get(k, ())}
            if st != {S_}:
                wrong += 1
                if len(ex) < 3:
                    ex.append(f'{k} in {sorted(st) or "no catalogue"}')
    return wrong == 0, wrong, (f'{n} rows equal a {S_} tier row ({"/".join(tiers)})' if not wrong else
                               f'{wrong} of {n} rows equal no {S_} row of the tier catalogues ({"/".join(tiers)}), '
                               f'e.g. {"; ".join(ex)}')


@content_test('served_view')
def unique_keys(path, **ctx):
    """No (source_id, band) twice in a survivors file (D2)."""
    dup, n = [], 0
    for p, rr in _view_rows(path):
        kc = Counter((r['source_id'], r['band']) for r in rr)
        n += len(rr)
        dup += [(os.path.basename(p), k) for k, c in kc.items() if c > 1]
    return not dup, len(dup), f'{n} rows, keys unique' if not dup else f'{len(dup)} keys twice, e.g. {dup[0]}'


def _multi_queue_keys(U, S_, tier):
    q = f'{U}/judgements/queue_p3s3_{S_}_multi.txt'
    if not os.path.exists(q):
        return None
    ids = [l.strip() for l in open(q) if l.startswith('J')]
    if tier == 'deep':
        return {(i.rsplit('_', 3)[0], i.rsplit('_', 3)[3]) for i in ids}, 'key'
    return {i.split('_')[0] for i in ids}, 'jname'


@content_test('served_view')
def complete(path, **ctx):
    """Singles view: every survivors row of the view's tier catalogues is in the view exactly once (sublst_serve.sh:93;
    D4). Multi view: every queued source's row is in the view exactly once."""
    vk = _view_kind(path, ctx)
    if vk is None:
        return False, None, f'not a view directory: {path}'
    U, S_, kind, tier, tiers = vk
    vc = Counter((r['source_id'], r['band']) for p, rr in _view_rows(path) for r in rr)
    if kind == 'single':
        want = {(r['source_id'], r['band']) for r in _tier_survivors(U, S_, tiers)}
    else:
        qk = _multi_queue_keys(U, S_, tier)
        if qk is None:
            return False, None, f'no multi queue for {path}'
        keys, how = qk
        if how == 'key':
            want = keys
        else:
            sv = f'{U}/source_catalogs_{S_}'
            want = {(r['source_id'], r['band']) for p, rr in _view_rows(sv) for r in rr if r['source_id'] in keys}
    miss = sorted(k for k in want if vc.get(k, 0) != 1)
    return not miss, len(miss), (f'{len(want)} expected rows, each once' if not miss else
                                 f'{len(miss)} of {len(want)} expected rows not exactly once in the view, e.g. {miss[:3]}')


@content_test('served_view')
def review_dir_page(path, **ctx):
    """review_dir is empty or a page directory under the run that holds PNGs and whose page id carries the row's J-name
    and band (sublst_reserve_test_served.py:76-78, 119-121); sub-LST pages live under candidates/<Stokes>/."""
    vk = _view_kind(path, ctx)
    if vk is None:
        return False, None, f'not a view directory: {path}'
    U, S_, kind, tier, tiers = vk
    bad, n, cache = [], 0, {}
    for p, rr in _view_rows(path):
        for r in rr:
            rd = r.get('review_dir', '')
            if not rd:
                continue
            n += 1
            if rd not in cache:
                d = os.path.join(U, rd)
                try:
                    cache[rd] = any(x.endswith('.png') for x in os.listdir(d))
                except OSError:
                    cache[rd] = False
            pid = rd.rstrip('/').rsplit('/', 1)[-1]
            why = None
            if not rd.startswith('candidates/'):
                why = 'not under candidates/'
            elif tier == 'sublst' and not rd.startswith(f'candidates/{S_}/'):
                why = f'not a Stokes-{S_} page'
            elif not cache[rd]:
                why = 'no PNG'
            elif pid.split('_')[0] != r['source_id'] or pid.rsplit('_', 1)[-1] != r['band']:
                why = "page id does not carry the row's J-name and band"
            if why:
                bad.append(f"{r['source_id']} {r['band']} {rd}: {why}")
    return not bad, len(bad), f'{n} links to page directories with PNGs' if not bad else f'{len(bad)} bad links, e.g. {bad[:2]}'


@content_test('served_view')
def tier_rules(path, **ctx):
    """Sub-LST views: every row's best detection and every member pass tier rules 3 and 4 (phase3_tier_rules, in place;
    sublst_reserve_test_served.py:43-62); LSTs in SUBLST_SERVED_TEST_OLD_LSTS reported apart, not failed."""
    vk = _view_kind(path, ctx)
    if vk is None:
        return False, None, f'not a view directory: {path}'
    U, S_, kind, tier, tiers = vk
    if tier != 'sublst':
        return True, None, 'n/a: tier rules 3 and 4 are tested on the sub-LST views (deep: catalog)'
    from exopipe.phase3.core import env
    env.legacy()
    from phase3_tier_rules import deep_search_refused, flux_scale_failed     # clv15, in place
    old = set(_S().SUBLST_SERVED_TEST_OLD_LSTS)
    mem = defaultdict(list)
    for p in glob.glob(os.path.join(path, '*_source_members.csv')):
        for m in _rows(p):
            mem[(m['source_id'], m['band'])].append(m)
    bad, nold, nd = [], 0, 0
    for p, rr in _view_rows(path):
        for r in rr:
            dets = sorted({(m['cand_id'], m['det_date']) for m in mem.get((r['source_id'], r['band']), [])}
                          | {(r.get('best_cand_id', ''), r.get('best_date', ''))})
            for cid, date in dets:
                c = cid.split('_')
                if len(c) < 3 or not date:
                    continue
                lst, night = c[1], date[:10]
                nd += 1
                why = deep_search_refused(lst, night, r['band'], S_) or flux_scale_failed(lst, night, r['band'])
                if why:
                    if lst in old:
                        nold += 1
                    else:
                        bad.append((r['source_id'], r['band'], cid, why))
    return not bad, len(bad), (f'{nd} detections pass tier rules 3 and 4 ({nold} in not-yet-re-run LSTs {sorted(old)})'
                               if not bad else f'{len(bad)} detections fail tier rule 3 or 4, e.g. {bad[:2]}')


# =====================================================================================================================
# judgement_journal
# =====================================================================================================================
@content_test('judgement_journal')
def parses_append_only(path, carried_from=None, **ctx):
    """Every line is a JSON object; a copied journal (sub-LST carry) starts with every line of its source."""
    n = 0
    with open(path) as f:
        lines = f.read().splitlines()
    for i, l in enumerate(lines, 1):
        if not l.strip():
            continue
        try:
            o = json.loads(l)
        except ValueError as e:
            return False, i, f'line {i} is not JSON ({e})'
        if not isinstance(o, dict):
            return False, i, f'line {i} is not a JSON object'
        n += 1
    if carried_from:
        src = open(carried_from).read().splitlines()
        if lines[:len(src)] != src:
            return False, n, f'does not start with the {len(src)} lines of {carried_from}'
    return True, n, f'{n} records' + (f', starts with its source {carried_from}' if carried_from else '')


@content_test('judgement_journal')
def carry_consistent(path, **ctx):
    """With carry_report.csv beside the journal: every carried record has carried_from_run / carried_from_source /
    shared_detections and the 'CARRIED from' note prefix; their count equals the report's 'carried' rows; no CONFLICT
    row's source was carried (snapshot_carry_judgements.py:101-126; deep_carry_judgements.py:105-127)."""
    rp = os.path.join(os.path.dirname(path), 'carry_report.csv')
    if not os.path.exists(rp):
        return True, None, 'n/a: no carry report (no carry into this journal)'
    rep = _rows(rp)
    recs = [json.loads(l) for l in open(path) if l.strip()]
    carried = [r for r in recs if 'carried_from_run' in r]
    bad = [r.get('source_id') for r in carried if not (r.get('carried_from_source') and r.get('shared_detections')
                                                      and str(r.get('judge_note', '')).startswith('CARRIED from'))]
    n_rep = sum(1 for r in rep if r.get('outcome') == 'carried')
    conflict = {r['new_source'] for r in rep if str(r.get('outcome', '')).startswith('CONFLICT')}
    conf_carried = [r['source_id'] for r in carried if r.get('source_id') in conflict]
    ok = not bad and n_rep == len(carried) and not conf_carried
    return ok, len(carried), (f'{len(carried)} carried records = {n_rep} report rows' if ok else
                              f'carried {len(carried)} vs report {n_rep}; malformed {bad[:3]}; conflicts carried '
                              f'{conf_carried[:3]}')


@content_test('judgement_journal')
def hold_rule(path, **ctx):
    """With carry_missed.csv beside the journal (snapshot): it lists only verdicts in HOLD_FP_VERDICTS (G19)."""
    mp = os.path.join(os.path.dirname(path), 'carry_missed.csv')
    if not os.path.exists(mp):
        return True, None, 'n/a: no carry_missed.csv'
    fpv = set(_S().HOLD_FP_VERDICTS)
    rows = _rows(mp)
    notfp = [r['old_source'] for r in rows if r.get('judge_confidence', '') not in fpv]
    return not notfp, len(notfp), (f'{len(rows)} missed verdicts, all FP' if not notfp else
                                   f'{len(notfp)} missed verdicts are not FP: {notfp[:5]}')


# =====================================================================================================================
# hub_page
# =====================================================================================================================
def _hub_ports(reg):
    return sorted({reg['round2']['port'], _S().HUB_PORT} | {e['port'] for S_ in 'VI' for e in reg[S_].values()}
                  | set(reg.get('tunnel_extra', [])))


@content_test('hub_page')
def registry_matches(path, expected=None, **ctx):
    """Review hub: registry.json well formed (V and I rows, every entry an int port); each served page's entry (ctx
    expected=[(S, tier, {k: v})]) holds its run, port, generation, served, live=1; index.html (when tested) carries a
    button for every registry port and the tunnel line of every port (build_review_hub.py:27-70) and says it was
    generated from registry.json."""
    d = os.path.dirname(os.path.abspath(path))
    reg_p = os.path.join(d, 'registry.json')
    if not os.path.exists(reg_p):
        return True, None, 'n/a: not the review hub (no registry.json beside it)'
    try:
        reg = json.load(open(reg_p))
    except ValueError as e:
        return False, None, f'registry.json unreadable ({e})'
    bad = []
    for S_ in 'VI':
        for t, e in (reg.get(S_) or {}).items():
            if not isinstance(e.get('port'), int):
                bad.append(f'{S_} {t}: port {e.get("port")!r} is not an int')
    for S_, t, want in expected or ():
        e = (reg.get(S_) or {}).get(t)
        if e is None:
            bad.append(f'{S_} {t}: no entry')
            continue
        for k, v in want.items():
            v2 = (v == '1') if k == 'live' and isinstance(v, str) else (int(v) if k == 'port' else v)
            if e.get(k) != v2:
                bad.append(f'{S_} {t}: {k}={e.get(k)!r}, expected {v2!r}')
    if os.path.basename(path) == 'index.html':
        html = open(path).read()
        for p in _hub_ports(reg):
            if f'-L {p}:127.0.0.1:{p}' not in html:
                bad.append(f'tunnel line lacks port {p}')
        for S_ in 'VI':
            for t, e in reg[S_].items():
                if f'href="http://localhost:{e["port"]}/"' not in html:
                    bad.append(f'no button for {S_} {t} port {e["port"]}')
        if 'from registry.json (build_review_hub.py)' not in html:
            bad.append('index.html was not generated from registry.json')
    return not bad, len(bad), '; '.join(bad[:4]) if bad else 'registry and page consistent'


@content_test('hub_page')
def greenlit_only(path, **ctx):
    """Inspection hub: exactly the rows of inspection_greenlit.csv whose status is 'greenlit' appear, each with its port
    (build_inspection_hub.py:19, 31-43)."""
    if os.path.basename(path) != 'index.html' or os.path.exists(os.path.join(os.path.dirname(path), 'registry.json')):
        return True, None, 'n/a: not the inspection hub'
    import html as _html
    page = open(path).read()
    rows = _rows(_S().INSPECTION_GREENLIT)
    bad = []
    green = [r for r in rows if (r.get('status') or '').strip() == 'greenlit']
    for r in green:
        if f'href="http://localhost:{_html.escape(r["port"])}/"' not in page or _html.escape(r['label']) not in page:
            bad.append(f'greenlit {r["port"]} missing')
    for r in rows:
        if r not in green and f'href="http://localhost:{_html.escape(r["port"])}/"' in page:
            bad.append(f'non-greenlit {r["port"]} linked')
    return not bad, len(green), '; '.join(bad[:3]) if bad else f'{len(green)} greenlit pages, nothing else'


# =====================================================================================================================
# served_page / stopped_page
# =====================================================================================================================
def _rec(path):
    with open(path) as f:
        return json.load(f)


@content_test('served_page')
def live(path, **ctx):
    """GET http://127.0.0.1:<port>/ answers 200 within INDEX_READ_TIMEOUT_S."""
    from exopipe.phase3.lib.serve import procs
    rec = _rec(path)
    st, body = procs.http_get(rec['port'], timeout=_S().INDEX_READ_TIMEOUT_S)
    return st == 200, st, f'port {rec["port"]} answers {st}' + ('' if st == 200 else f' ({body[:80]})')


@content_test('served_page')
def same_process(path, **ctx):
    """The pid listening on the port is the recorded pid, its argv is the recorded argv (so its --run is this run), its
    environment carries the recorded SERVER_ENV values; a twin's argv has --inspect."""
    from exopipe.phase3.lib.serve import procs
    rec = _rec(path)
    pids = procs.listeners(rec['port'])
    if pids != [rec['pid']]:
        return False, pids, f'port {rec["port"]} listened by {pids}, recorded pid {rec["pid"]}'
    info = procs.proc_info(rec['pid'])
    bad = []
    if info is None or info['argv'] != rec['argv']:
        bad.append('argv differs from the record')
    if rec.get('program', 'classify') == 'classify' and procs.argv_value(rec['argv'], '--run') != rec.get('run'):
        bad.append(f'--run is not {rec.get("run")}')
    try:
        env = procs.environ(rec['pid'])
    except OSError as e:
        env = {}
        bad.append(f'environ unreadable ({e})')
    for k, v in (rec.get('env') or {}).items():
        if env.get(k) != v:
            bad.append(f'{k}={env.get(k)!r}, recorded {v!r}')
    if rec.get('inspect') and '--inspect' not in rec['argv']:
        bad.append('twin without --inspect')
    return not bad, rec['pid'], '; '.join(bad[:4]) if bad else f'pid {rec["pid"]} serves {rec.get("run")}'


@content_test('served_page')
def sources_listed(path, **ctx):
    """Distinct URL-decoded /source/<id> links on the index = expected_sources (snapshot: queue length,
    snapshot_rechain_batch.sh:419-426; deep / sub-LST: the first-presentation identities of the server's own App)."""
    from exopipe.phase3.lib.serve import procs
    rec = _rec(path)
    if rec.get('program', 'classify') != 'classify':
        return True, None, f'n/a: a {rec.get("program")} page has no source index'
    if rec.get('expected_sources') is None and str(rec.get('expected_kind') or '').startswith('restart'):
        return True, None, 'n/a: a restart with the same argv over unchanged files (restart_server_same_argv.sh)'
    if rec.get('expected_sources') is None:
        return False, None, 'no expected_sources recorded'
    st, body = procs.http_get(rec['port'], timeout=_S().INDEX_READ_TIMEOUT_S)
    if st != 200:
        return False, st, f'index answers {st}'
    n = len(procs.index_source_ids(body))
    ok = n == rec['expected_sources']
    return ok, n, f'{n} sources listed, expected {rec["expected_sources"]} ({rec.get("expected_kind")})'


@content_test('stopped_page')
def not_listening(path, **ctx):
    """Every stopped pid is gone and no review server is on the port."""
    from exopipe.phase3.lib.serve import procs
    rec = _rec(path)
    alive = [s['pid'] for s in rec.get('stopped', []) if procs.alive(s['pid'])]
    on = [s['pid'] for s in procs.review_servers() if s['port'] == str(rec['port'])]
    ok = not alive and not on
    return ok, len(alive) + len(on), (f'port {rec["port"]}: no review server' if ok else
                                      f'still alive {alive}; review servers on the port {on}')


# =====================================================================================================================
# sublst_cumulative
# =====================================================================================================================
def _snr_of(r):
    """sublst_cumulative.py:66-70."""
    for k in ('best_snr', 'snr', 'det_snr'):
        try:
            return abs(float(r.get(k) or 'nan'))
        except ValueError:
            pass
    return 0.0


@content_test('sublst_cumulative')
def merge_invariants(path, lst_runs_root=None, **ctx):
    """Every LST merged (from master_catalog.csv cand_ids) has its page directories in candidates/; pertie_detections.csv
    one row per cand_id; master_catalog.csv carries pos_kind; survivors one row per (source_id, band), the largest |best
    SNR| over the LST runs (sublst_cumulative.py:22-26, 74-104)."""
    S = _S()
    B = os.path.abspath(path)
    from exopipe.phase3.lib.serve.plan import run_root
    root = lst_runs_root or run_root('sublst')
    bad = []
    mc = os.path.join(B, 'master_catalog.csv')
    if not os.path.exists(mc):
        return False, None, 'no master_catalog.csv'
    with open(mc, newline='') as f:
        rd = csv.DictReader(f)
        if 'pos_kind' not in (rd.fieldnames or []):
            bad.append('master_catalog.csv has no pos_kind column')
        lsts = sorted({r['cand_id'].split('_')[1] for r in rd if r.get('cand_id', '').count('_') >= 2})
    runs = [(l, os.path.join(root, S.SUBLST_LST_RUN.format(hh=l[:-1]))) for l in lsts]
    for l, R in runs:
        for S_ in 'IV':
            src = os.path.join(R, 'candidates', S_)
            if not os.path.isdir(src):
                continue
            have = set(os.listdir(os.path.join(B, 'candidates', S_))) if os.path.isdir(os.path.join(B, 'candidates', S_)) else set()
            miss = [x for x in os.listdir(src) if x not in have]
            if miss:
                bad.append(f'{l}: {len(miss)} {S_} page directories not merged, e.g. {miss[0]}')
    pt = os.path.join(B, 'pertie_detections.csv')
    if os.path.exists(pt):
        c = Counter(r['cand_id'] for r in _rows(pt))
        dup = [k for k, v in c.items() if v > 1]
        if dup:
            bad.append(f'pertie_detections.csv: {len(dup)} cand_ids twice, e.g. {dup[0]}')
    else:
        bad.append('no pertie_detections.csv')
    for S_ in 'VI':
        for t in S.SUBLST_TIERS:
            for p in glob.glob(os.path.join(B, f'source_catalogs_{S_}_{t}', '*_sources_survivors.csv')):
                rr = _rows(p)
                kc = Counter((r['source_id'], r['band']) for r in rr)
                if any(v > 1 for v in kc.values()):
                    bad.append(f'{os.path.relpath(p, B)}: a (source_id, band) twice')
                best = {}
                for l, R in runs:
                    q = os.path.join(R, f'source_catalogs_{S_}_{t}', os.path.basename(p))
                    if os.path.exists(q):
                        for r in _rows(q):
                            k = (r.get('source_id', ''), r.get('band', ''))
                            best[k] = max(best.get(k, 0.0), _snr_of(r))
                low = [k for r in rr for k in [(r['source_id'], r['band'])] if k in best and _snr_of(r) < best[k]]
                if low:
                    bad.append(f'{os.path.relpath(p, B)}: {len(low)} rows not the largest |SNR| of their key')
    return not bad, len(bad), '; '.join(bad[:3]) if bad else f'{len(lsts)} LSTs merged ({" ".join(lsts)}), invariants hold'


@content_test('sublst_cumulative')
def figures(path, **ctx):
    """Every identity with n_det > 1 in source_catalogs/localizations.csv has localization_figs/<id>.png and .key
    (sublst_cumulative.sh:43-60; G9)."""
    lp = os.path.join(path, 'source_catalogs', 'localizations.csv')
    if not os.path.exists(lp):
        return False, None, 'no source_catalogs/localizations.csv'
    multi = [r['identity_id'] for r in _rows(lp) if int(r['n_det']) > 1]
    miss = [i for i in multi if not (os.path.exists(os.path.join(path, 'localization_figs', f'{i}.png'))
                                     and os.path.exists(os.path.join(path, 'localization_figs', f'{i}.key')))]
    return not miss, len(miss), f'{len(multi)} multi-detection identities, all with a figure' if not miss else \
        f'{len(miss)} of {len(multi)} without figure/key, e.g. {miss[:3]}'


def candidates_listing(run):
    """`find candidates -mindepth 2 -maxdepth 3` from the run directory (sublst_catalog_publish.sh:104), made in process."""
    out = []
    base = os.path.join(run, 'candidates')
    for S_ in sorted(os.listdir(base)) if os.path.isdir(base) else []:
        d1 = os.path.join(base, S_)
        if not os.path.isdir(d1):
            continue
        for pid in sorted(os.listdir(d1)):
            out.append(f'candidates/{S_}/{pid}')
            d2 = os.path.join(d1, pid)
            if os.path.isdir(d2):
                out.extend(f'candidates/{S_}/{pid}/{f}' for f in sorted(os.listdir(d2)))
    return out


@content_test('sublst_cumulative')
def served_tests_c_d(path, with_served_tests=True, **ctx):
    """sublst_reserve_test_served.py (in place) on the run with a listing made in process: SERVED-TEST PASS."""
    if not with_served_tests:
        return True, None, 'n/a: tests (c)+(d) run on the swapped-in run (verify), as production (sublst_catalog_publish.sh:101-108)'
    from exopipe.phase3.lib.serve import scripts
    S = _S()
    fd, lst = tempfile.mkstemp(prefix='exopipe3_sublst_listing_', suffix='.txt')
    try:
        with os.fdopen(fd, 'w') as f:
            f.write('\n'.join(candidates_listing(path)) + '\n')
        res = scripts.run('sublst_served_test', [path, lst, '--old-lsts', ' '.join(S.SUBLST_SERVED_TEST_OLD_LSTS)],
                          refusal_codes=(1, 2))
    finally:
        os.unlink(lst)
    line = next((l for l in res.tail.splitlines() if l.startswith('SERVED-TEST')), scripts.last_line(res.tail))
    return res.rc == 0 and 'PASS' in line, res.rc, line


# =====================================================================================================================
# review_copy
# =====================================================================================================================
# files serve itself writes into the review copy after the copy (not part of the byte comparison): the exclusion gate's
# outputs (snapshot_serve_exclusion_gate.py:86-104), the localization outputs (snapshot_localize_tiers.sh: per-tier dirs and
# astrometry.png in page dirs), the journal / carry files and the server logs
DST_WRITES = ('serve_exclusion_check.txt', 'serve_exclusion_refusals.csv', 'source_catalogs_loc_*', 'source_catalogs_*_locin_*',
              'astrometry.png', 'judgements', 'classify_*.log')


@content_test('review_copy')
def byte_copy(path, source=None, **ctx):
    """`rsync -anc` from the source to the copy lists nothing (SNAPSHOT_COPY_EXCLUDE and the files serve writes into the
    copy aside; defect R7: the copy was unchecked)."""
    if not source:
        return False, None, 'no source given (the step passes source=<chain run>; verify: --set is not enough)'
    ex = []
    for x in tuple(_S().SNAPSHOT_COPY_EXCLUDE) + DST_WRITES:
        ex += ['--exclude', x]
    p = subprocess.run(['rsync', '-anc', '--out-format=%n'] + ex + [source.rstrip('/') + '/', path.rstrip('/') + '/'],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=_S().LEGACY_TIMEOUT_S)
    diff = [l for l in p.stdout.decode('utf-8', 'replace').splitlines() if l.strip() and not l.endswith('/')]
    ok = p.returncode == 0 and not diff
    return ok, len(diff), (f'byte-identical to {source}' if ok else f'rsync rc {p.returncode}; {len(diff)} files differ, '
                                                                     f'e.g. {diff[:3]}')


@content_test('review_copy')
def no_excluded_rows(path, **ctx):
    """No CSV row of the run (judgements excepted) lies on a cell-band excluded by data_decisions.csv read now
    (core.exclusions; the same keying as snapshot_serve_exclusion_gate.py). Read-only."""
    from exopipe.phase3.lib.serve import exclusion_gate
    scan = exclusion_gate.scan_snapshot_run(path)
    return not scan.hits, len(scan.hits), scan.summary()


# =====================================================================================================================
# exclusion_check (the serve-time exclusion gate's pass file: <run>/serve_exclusion_check.txt of the in-place gate, or
# serve's own <out>/_records/serve/gates/<run>_<what>.txt)
# =====================================================================================================================
@content_test('exclusion_check')
def gate_passed(path, run=None, **ctx):
    """The file says EXCLUSION-GATE PASS, and - when the run directory is known (ctx run=) - core.exclusions, reading
    data_decisions.csv NOW, finds no row of the run on an excluded cell-band (snapshot keying)."""
    with open(path) as f:
        first = f.readline().strip()
    if not first.startswith('EXCLUSION-GATE PASS'):
        return False, None, first[:160] or 'empty file'
    if run:
        from exopipe.phase3.lib.serve import exclusion_gate
        scan = exclusion_gate.scan_snapshot_run(run)
        if scan.hits:
            return False, len(scan.hits), f're-read now: {scan.summary()}'
        return True, 0, f'{first[:100]}; re-read now: 0 rows on excluded cell-bands'
    return True, None, first[:160]
