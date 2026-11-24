"""Content tests of the grouping products of the catalog step (item F.3; lib/catalog/grouping_stage.py).

  grouping_table   grouping_<S>/detections.csv with its siblings identities.csv, multiples.csv, events.jsonl and (when the run
                   had one) state_previous/: ctx tier ('deep' | '10min'), stokes, input (the catalogue grouped)
    T1 the table's columns, one Stokes, cand_ids unique
    T2 EVERY detection of the input's Stokes exactly once, with the input's own fingerprint (band, LST, date, ra, dec, S/N)
    T3 every identity's members: identities.csv = the identities of the table, each with its count, windows, nights, LSTs,
       bands, best detection and name; one multiples-level group per identity (nesting)
    T4 the multiples links: multiples.csv = the groups of the table, each with exactly the identities that link to it, its
       counts, its multiplicity (10-minute windows / deep nights - catalog_grouping.MULTIPLE_MIN) and the multiple flag
       (multiplicity >= MULTIPLE_MIN) on the group and on every member detection; a tier and Stokes without a multiples level
       (deep V, catalog_grouping.P_MULTIPLES None) holds one identity per group
    T5 IDs and names: every ID of its level's format for the tier and Stokes (catalog_grouping.ID_FORMAT), identity names unique,
       a group named after one of its identities
    T6 stable IDs carried per the module's rules (with state_previous/): an ID of the table is live in the previous state or new
       to it (never a retired or merged one), and every detection of the previous state that kept its cand_id and fingerprint
       (or that a 'remeasured' event matched) and changed identity or multiples ID has its 'move' event
  grouping_state   <state dir>/grouping_<tier>_<S>_state.json: ctx tier, stokes, table (the detections.csv of the same run)
    S1 it loads (format, fields, every assignment to a live ID, next_seq above every issued ID: grouping.GroupingState.load)
    S2 its assignments are the table's: the same cand_ids, each with the table's identity and multiples IDs and fingerprint
    S3 every live ID of the state is in the table, and every ID of the table is live
  stokes_link      stokes_link/stokes_links.csv with its siblings (stokes_link_pairs.csv, identity_merges.csv, summary.json;
                   lib/catalog/stokes_link.py), the grouping's own or a served run's copy
    L1 the two grouping tables the summary names are the present ones (sha1) and no display name is held in both Stokes
    L2 every linked (Stokes, identity) is an identity of its Stokes' table with that name; every pair passes the test it records
       (u <= U_CAP, P >= the identity level), with an offset percentile in 0-100 and a chance-coincidence probability in 0-1
    L3 the components are the pairs' connected components: each holds both Stokes, link_size members, link_ambiguous exactly
       when it holds three or more, and is named (link_id, link_name) by one of its members, the best-localized one
    L4 identity_merges.csv: every merged ID stands for a live identity of its Stokes' table and is not live itself
``failures(paths, tier, stokes, input)`` runs both on a stage's products before the stage saves the state (run_group's check).
"""
import collections
import csv
import json
import os
import re

from exopipe.phase3.core.validate import content_test

csv.field_size_limit(1 << 30)


def _cfg():
    from exopipe.phase3.config import CFG
    return CFG


def _rows(path):
    with open(path, newline='') as f:
        rd = csv.DictReader(f)
        return list(rd.fieldnames or []), list(rd)


def _id_re(level, tier, stokes):
    g = _cfg().catalog_grouping
    pat = re.escape(g.ID_FORMAT.format(level=g.ID_LEVEL_CODE[level], tier=g.ID_TIER_CODE[tier], stokes=stokes, seq=0))
    pat = pat.replace(re.escape('0' * 7), r'\d{7}')
    return re.compile('^' + pat + '$')


def _seq(i):
    return int(str(i).rsplit('-', 1)[-1])


def table_failures(path, tier, stokes, input=None):
    """[failure texts] of a grouping_<S>/detections.csv and its siblings (T1-T6)."""
    from exopipe.phase3.lib.catalog import grouping_stage as GS
    from exopipe.phase3.lib.catalog import grouping as G
    bad = []
    d = os.path.dirname(os.path.abspath(path))
    hdr, rows = _rows(path)
    if tuple(hdr) != GS.DET_COLS:
        return [f'T1 columns {hdr} are not {list(GS.DET_COLS)}']
    if not rows:
        return ['T1 the table holds no detection']
    by = {}
    for r in rows:
        if r['cand_id'] in by:
            bad.append(f'T1 cand_id {r["cand_id"]} twice')
        by[r['cand_id']] = r
        if r['stokes'] != stokes:
            bad.append(f'T1 {r["cand_id"]}: Stokes {r["stokes"]}, the table is {stokes}')
    # T2 every detection of the input exactly once, with its own fingerprint
    if input:
        src = dict(GS.stokes_rows(input, stokes))
        miss = sorted(set(src) - set(by))
        extra = sorted(set(by) - set(src))
        if miss:
            bad.append(f'T2 {len(miss)} detection(s) of the input in no identity, e.g. {miss[:3]}')
        if extra:
            bad.append(f'T2 {len(extra)} detection(s) of the table not in the input, e.g. {extra[:3]}')
        nf = []
        for c, r in by.items():
            s = src.get(c)
            if s is None:
                continue
            if (s['band'], s['lst'], s['date']) != (r['band'], r['lst'], r['date']) or r['key'] != f"{s['lst']}|{s['date']}" \
                    or any(float(s[k]) != float(r[k]) for k in ('ra', 'dec', 'snr')):
                nf.append(c)
        if nf:
            bad.append(f'T2 {len(nf)} detection(s) carry another fingerprint than the input\'s, e.g. {nf[:3]}')
    # T3 identities
    mem = collections.defaultdict(list)
    for r in rows:
        mem[r['identity_id']].append(r)
    ih, irows = _rows(os.path.join(d, GS.IDENTITIES))
    if tuple(ih) != GS.IDENT_COLS:
        bad.append(f'T3 identities.csv columns {ih}')
        irows = []
    ib = {r['identity_id']: r for r in irows}
    if len(ib) != len(irows):
        bad.append('T3 an identity twice in identities.csv')
    if set(ib) != set(mem):
        bad.append(f'T3 identities.csv holds {len(ib)} identities, the table {len(mem)} '
                   f'(e.g. {sorted(set(ib) ^ set(mem))[:3]})')
    names = collections.Counter(r['identity_name'] for r in irows)
    dup = [n for n, k in names.items() if k > 1]
    if dup:
        bad.append(f'T5 {len(dup)} identity name(s) shared, e.g. {dup[:3]}')
    nbad = []
    for i, ms in mem.items():
        e = ib.get(i)
        if e is None:
            continue
        best = max(ms, key=lambda r: abs(float(r['snr'])))
        want = dict(n_det=str(len(ms)), n_windows=str(len({r['key'] for r in ms})), n_nights=str(len({r['night'] for r in ms})),
                    n_lsts=str(len({r['lst'][:3] for r in ms})), bands='+'.join(sorted({r['band'] for r in ms})),
                    multiples_id=ms[0]['multiples_id'])
        got = {k: e[k] for k in want}
        if got != want or abs(float(e['best_snr'])) != abs(float(best['snr'])):
            nbad.append(f'{i}: {got} != {want}')
        if len({r['multiples_id'] for r in ms}) != 1:
            nbad.append(f'{i}: its detections lie in {len({r["multiples_id"] for r in ms})} multiples-level groups (nesting)')
        if {r['identity_name'] for r in ms} != {e['identity_name']}:
            nbad.append(f'{i}: detections name it {sorted({r["identity_name"] for r in ms})[:2]}, identities.csv {e["identity_name"]}')
    if nbad:
        bad.append(f'T3 {len(nbad)} identity row(s) do not match their members, e.g. {nbad[:2]}')
    # T4 multiples links
    mm = collections.defaultdict(list)
    for r in rows:
        mm[r['multiples_id']].append(r)
    mh, mrows = _rows(os.path.join(d, GS.MULTIPLES))
    if tuple(mh) != GS.MULT_COLS:
        bad.append(f'T4 multiples.csv columns {mh}')
        mrows = []
    mb = {r['multiples_id']: r for r in mrows}
    if set(mb) != set(mm):
        bad.append(f'T4 multiples.csv holds {len(mb)} groups, the table {len(mm)} (e.g. {sorted(set(mb) ^ set(mm))[:3]})')
    what, nmin = _cfg().catalog_grouping.MULTIPLE_MIN[tier]
    col = {'windows': 'key', 'nights': 'night'}[what]
    level_off = G.operating_point(tier, stokes)[1] is None
    lbad = []
    for m, ms in mm.items():
        e = mb.get(m)
        if e is None:
            continue
        ids = sorted({r['identity_id'] for r in ms}, key=_seq)
        mult = len({r[col] for r in ms})
        flag = int(mult >= int(nmin))
        want = dict(identity_ids=';'.join(ids), n_identities=str(len(ids)), n_det=str(len(ms)),
                    n_windows=str(len({r['key'] for r in ms})), n_nights=str(len({r['night'] for r in ms})),
                    n_lsts=str(len({r['lst'][:3] for r in ms})), multiplicity=str(mult), multiple=str(flag))
        got = {k: e[k] for k in want}
        if got != want:
            lbad.append(f'{m}: {got} != {want}')
        if any((r['multiplicity'], r['multiple']) != (str(mult), str(flag)) for r in ms):
            lbad.append(f'{m}: a member detection carries another multiplicity / flag than its group ({mult}, {flag})')
        if e['multiples_name'] not in {ib[i]['identity_name'] for i in ids if i in ib}:
            lbad.append(f'{m}: named {e["multiples_name"]}, the name of none of its identities')
        if level_off and len(ids) != 1:
            lbad.append(f'{m}: {len(ids)} identities although {tier} {stokes} has no multiples level (P_MULTIPLES None)')
    if lbad:
        bad.append(f'T4 {len(lbad)} multiples link(s) inconsistent, e.g. {lbad[:2]}')
    ri, rm = _id_re('identity', tier, stokes), _id_re('multiples', tier, stokes)
    fid = [i for i in mem if not ri.match(i)] + [m for m in mm if not rm.match(m)]
    if fid:
        bad.append(f'T5 {len(fid)} ID(s) not of their level\'s format for {tier} {stokes}, e.g. {fid[:3]}')
    # T6 stable IDs carried (when the run had a previous state)
    prev = os.path.join(d, 'state_previous', f'grouping_{tier}_{stokes}_state.json')
    if os.path.exists(prev):
        bad += carried_failures(prev, rows, os.path.join(d, GS.EVENTS))
    return bad


def carried_failures(prev_path, rows, events_path):
    """T6 against the previous state: IDs never re-issued; every changed ID of a detection the previous state held (same
    cand_id and fingerprint, or matched by a 'remeasured' event) recorded by a 'move' event of its level."""
    with open(prev_path) as f:
        P = json.load(f)
    pids, pas = P.get('ids', {}), P.get('assign', {})
    bad = []
    reused = sorted({x for r in rows for x in (r['identity_id'], r['multiples_id'])
                     if x in pids and pids[x].get('status') != 'live'})
    if reused:
        bad.append(f'T6 {len(reused)} ID(s) of the table were retired or merged in the previous state (IDs are never reused), '
                   f'e.g. {reused[:3]}')
    moves, remeas = set(), {}
    if os.path.exists(events_path):
        with open(events_path) as f:
            for line in f:
                e = json.loads(line)
                if e.get('event') == 'move':
                    moves.add((e.get('level'), e.get('cand_id'), e.get('from_id'), e.get('to_id')))
                elif e.get('event') == 'remeasured':
                    remeas[e['cand_id']] = e['cand_id_old']
    nfail = []
    for r in rows:
        old_c = remeas.get(r['cand_id'], r['cand_id'])
        a = pas.get(old_c)
        if a is None:
            continue
        if old_c == r['cand_id']:
            fp_old = tuple(a[2:7])
            fp_new = (r['key'], r['band'], repr(float(r['ra'])), repr(float(r['dec'])), repr(float(r['snr'])))
            if fp_old != fp_new:
                continue                          # the cand_id names another detection now ('reused'): not carried
        for lv, old, new in (('identity', a[0], r['identity_id']), ('multiples', a[1], r['multiples_id'])):
            if old != new and (lv, r['cand_id'], old, new) not in moves:
                nfail.append(f'{r["cand_id"]} {lv} {old} -> {new}')
    if nfail:
        bad.append(f'T6 {len(nfail)} carried detection(s) changed ID without a move event, e.g. {nfail[:3]}')
    return bad


def state_failures(path, tier, stokes, table):
    from exopipe.phase3.lib.catalog import grouping as G
    bad = []
    try:
        st = G.GroupingState.load(os.path.dirname(os.path.abspath(path)), tier, stokes)
    except G.GroupingRefused as e:
        return [f'S1 {e}']
    if os.path.abspath(st.path(os.path.dirname(os.path.abspath(path)))) != os.path.abspath(path):
        return [f'S1 {path} is not the state file of {tier} {stokes}']
    _, rows = _rows(table)
    by = {r['cand_id']: r for r in rows}
    if set(by) != set(st.assign):
        bad.append(f'S2 the state assigns {len(st.assign)} detections, the table holds {len(by)} '
                   f'(e.g. {sorted(set(by) ^ set(st.assign))[:3]})')
    nb = []
    for c, (i, m, fp, _e, _w) in st.assign.items():
        r = by.get(c)
        if r is None:
            continue
        want = (r['key'], r['band'], repr(float(r['ra'])), repr(float(r['dec'])), repr(float(r['snr'])))
        if (i, m) != (r['identity_id'], r['multiples_id']) or tuple(fp) != want:
            nb.append(c)
    if nb:
        bad.append(f'S2 {len(nb)} assignment(s) differ from the table (IDs or fingerprint), e.g. {nb[:3]}')
    live = {i for i, e in st.ids.items() if e.get('status') == 'live'}
    used = {x for r in rows for x in (r['identity_id'], r['multiples_id'])}
    if live != used:
        bad.append(f'S3 live IDs of the state {len(live)} != IDs of the table {len(used)} (e.g. {sorted(live ^ used)[:3]})')
    return bad


def failures(paths, tier, stokes, input=None):
    """Both tests on the products of one run_group (paths: 'detections', 'state_next')."""
    out = table_failures(paths['detections'], tier, stokes, input)
    out += state_failures(paths['state_next'], tier, stokes, paths['detections'])
    return out


@content_test('grouping_table')
def grouping_table(path, **ctx):
    tier, stokes = ctx.get('tier'), ctx.get('stokes')
    if tier not in ('deep', '10min') or stokes not in ('I', 'V'):
        return False, None, f'grouping_table needs ctx tier (deep / 10min) and stokes (I / V), got {tier!r} {stokes!r}'
    bad = table_failures(path, tier, stokes, ctx.get('input'))
    with open(path) as f:
        n = max(0, sum(1 for _ in f) - 1)
    return (not bad), n, ('; '.join(bad[:4]) if bad else
                          f'{n} detections, each in one identity; identities, multiples links and IDs consistent'
                          + (' (input checked)' if ctx.get('input') else ''))


@content_test('grouping_state')
def grouping_state(path, **ctx):
    tier, stokes, table = ctx.get('tier'), ctx.get('stokes'), ctx.get('table')
    if not table:
        return False, None, 'grouping_state needs ctx table (the detections.csv of the same run)'
    bad = state_failures(path, tier, stokes, table)
    return (not bad), None, ('; '.join(bad[:4]) if bad else 'the state = the table (assignments, live IDs, next_seq)')


def link_failures(path):
    """L1-L4 on stokes_link/stokes_links.csv (path) and its siblings: [] or what fails. The link directory is the grouping's
    own (beside grouping_I/ and grouping_V/) or a served run's copy (lib/serve/sublst_build.stokes_link_stage): the tables
    are the ones its summary names, in one grouping directory, and the grouping's own when one stands beside it."""
    from exopipe.phase3.lib.catalog import grouping_stage as GS, stokes_link as SL
    d = os.path.dirname(os.path.abspath(path))
    bad = []
    with open(os.path.join(d, SL.SUMMARY)) as f:
        summ = json.load(f)
    outs = {os.path.dirname(os.path.dirname(os.path.abspath(summ['tables'][S]['path']))) for S in ('I', 'V')}
    out = sorted(outs)[0]
    if len(outs) != 1:
        bad.append(f'L1 the two Stokes\' tables are in two grouping directories {sorted(outs)}')
    beside = os.path.dirname(d)
    if os.path.isdir(GS.group_dir(beside, 'I')) and os.path.abspath(beside) != out:
        bad.append(f'L1 the link read the grouping under {out}, not the one beside it ({beside})')
    names = {}
    for S in ('I', 'V'):
        t = summ['tables'][S]
        if os.path.abspath(t['path']) != os.path.abspath(os.path.join(GS.group_dir(out, S), GS.DETECTIONS)):
            bad.append(f'L1 {S}: the link read {t["path"]}, not a grouping detections table')
        elif SL._sha1(t['path']) != t['sha1']:
            bad.append(f'L1 {S}: {t["path"]} changed since the link read it')
        names[S] = {r['identity_id']: r['identity_name'] for r in _rows(os.path.join(GS.group_dir(out, S), GS.IDENTITIES))[1]}
    shared = SL.check_names(out)
    if shared:
        bad.append(f'L1 {len(shared)} display name(s) held in both Stokes, e.g. {shared[:3]}')
    _h, links = _rows(path)
    _h, pairs = _rows(os.path.join(d, SL.PAIRS))
    p_thr, u_cap = float(summ['counts']['p_threshold']), float(summ['counts']['u_cap'])
    for r in links:
        if names.get(r['stokes'], {}).get(r['identity_id']) != r['identity_name']:
            bad.append(f'L2 {r["stokes"]} {r["identity_id"]} {r["identity_name"]}: not an identity of that name in its table')
    for p in pairs:
        if not (float(p['u']) <= u_cap and float(p['p']) >= p_thr):
            bad.append(f'L2 pair {p["I_identity_id"]}-{p["V_identity_id"]}: u {p["u"]}, P {p["p"]} fail the test')
        if not (0 <= float(p['offset_pct']) <= 100 and 0 <= float(p['p_chance']) <= 1):
            bad.append(f'L2 pair {p["I_identity_id"]}-{p["V_identity_id"]}: offset percentile / chance probability out of range')
    # L3 the components from the pairs alone
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            x = parent[x]
        return x
    for p in pairs:
        a, b = find(('I', p['I_identity_id'])), find(('V', p['V_identity_id']))
        if a != b:
            parent[max(a, b)] = min(a, b)
    comp = collections.defaultdict(set)
    for x in list(parent):
        comp[find(x)].add(x)
    by_link = collections.defaultdict(list)
    for r in links:
        by_link[r['link_id']].append(r)
    got = sorted(sorted((r['stokes'], r['identity_id']) for r in rs) for rs in by_link.values())
    want = sorted(sorted(c) for c in comp.values())
    if got != want:
        bad.append(f'L3 the link table holds {len(got)} components, the pairs make {len(want)} (or their members differ)')
    for lid, rs in by_link.items():
        n = len(rs)
        if {r['stokes'] for r in rs} != {'I', 'V'}:
            bad.append(f'L3 {lid}: a component without both Stokes')
        if any(int(r['link_size']) != n or int(r['link_ambiguous']) != int(n > 2) for r in rs):
            bad.append(f'L3 {lid}: link_size / link_ambiguous disagree with its {n} members')
        head = [r for r in rs if r['identity_id'] == lid]
        if len(head) != 1 or any(r['link_name'] != head[0]['identity_name'] for r in rs):
            bad.append(f'L3 {lid}: not named by one of its members')
        elif float(head[0]['pos_err_arcsec']) > min(float(r['pos_err_arcsec']) for r in rs):
            bad.append(f'L3 {lid}: named by a member that is not the best-localized')
    # L4 the merge table: every merged ID stands for a live identity of its Stokes' table and is none itself
    mp = os.path.join(d, SL.MERGES)
    if not os.path.exists(mp):
        bad.append(f'L4 {mp} is missing')
    else:
        for m in _rows(mp)[1]:
            live = names.get(m['stokes'], {})
            if m['into_id'] not in live or m['identity_id'] in live:
                bad.append(f'L4 {m["stokes"]} {m["identity_id"]} -> {m["into_id"]}: not a merged ID of a live identity')
    return bad


@content_test('stokes_link')
def stokes_link(path, **ctx):
    bad = link_failures(path)
    _h, rows = _rows(path)
    return (not bad), len(rows), ('; '.join(bad[:4]) if bad else
                                  f'{len(rows)} linked identities; tables current, names unique across Stokes, pairs pass the '
                                  f'test, components named by their best-localized member, merged IDs stand for live ones')
