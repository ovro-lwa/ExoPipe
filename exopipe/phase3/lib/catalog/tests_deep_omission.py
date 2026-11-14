"""Content tests of the deep-omission stage (lib/catalog/deep_omission.py), registered with core/validate.

Classes:
  omission_decisions  deep_omission.csv: production's header for the tier; every verdict follows from the row's own values -
                      a decision row names only propagating gates and its attempt; ABSENT has no statistics; KEEP exactly when
                      rms_pass and rate_pass, each recomputed from the row's numbers (4 significant digits as written; a nan
                      threshold never passes the noise test and always passes the rate test); the reason is the verdict's; a
                      wideband-member row is OMIT, of a band group, and every member it names is a decision row of the same
                      LST, night, Stokes and tier with that verdict (one of WIDEBAND_MEMBER_VERDICTS), a member of the group
                      and not WIDEBAND_EXCEPT; one row per (key, tier) and kind.
  omission_record     deep_omission_dropped.csv (with the decisions beside, or ctx decisions=): production's columns for the
                      tier; every removed detection, member row and panel has an OMIT key and the 'why' of exactly that key's
                      decisions; every dropped identity (10-s / 1-minute) has a recorded detection; the summary's page line and
                      json count the record's detections; a removal on a per-image key (dde1760) has its own image holding
                      an omitted member (the member table read here); a 'card2 detection' row is a 10-s Stokes V one. The per-instant
                      test of this class is in tests_per_instant.py.
  omission_kept       deep_omission_kept.csv (10-s / 1-minute; the record, decisions and summary beside, or ctx decisions=):
                      the drop the snapshot tier's cumulative step applies (Q review F2). K1 the writer's columns and kinds,
                      every row of a served run of the summary (its batch tag); K2 every 'detection kept' row has no OMIT key, is
                      no detection the record omits, of no identity it drops; their number = the summary's kept = served -
                      omitted - Card 2 - 1-min elevation floor, their identities = identities_kept, kept + dropped identities = identities; no detection key kept
                      twice, the kept rows' multiset digest = the summary's and kept + omitted = the served set's digest, the
                      record's detection rows = the omitted digest (Q fix verification N2); K5 the served counts beside
                      (deep_omission_served.csv) = the kept + omitted detections of each key (V_combview review F5); K3 the per-tier rows
                      are the record's one to one ('source dropped' / 'source re-pointed' = 'source partial' / 'member row
                      dropped' = 'member row' / 'panel dropped' = 'panel'), each on its tier's catalogue file; K4 a re-pointed
                      source's fields name its new best, a kept detection of its own identity, with production's note; a row
                      with a key has an OMIT key, one without names its identity's drop. Streamed (two passes over the file).
                      Since exopipe_ops dde1760 / dbc40a6: a row on a per-image key must have its image hold an omitted member and a
                      kept detection's must not, read back here (ImageCheck, independently of deep_omission.Omission - review T4 /
                      T8); a kept detection Card 2 flags refuses (each run's card2_vleak.csv read here); K5 counts the served set
                      without its Card 2 detections; the digests add up kept + omitted + Card 2 = served; K6 the re-judged-out
                      (batch identity, tier) of the re-judging's table and the kept product's 'source re-judged' / 'member row
                      re-judged' rows are one another's, their notes and a still-qualifying source's rejudged_routes the table's.
  omission_rejudged   deep_omission_rejudged.csv (10-s / 1-minute; production's rejudged.csv, dbc40a6): columns, one row per (batch,
                      identity, tier), qualifies 0 / 1, the removal counts, 'every member removed' exactly when nothing is left, the
                      summary's re-judging counts.
  omission_served     deep_omission_served.csv (10-s / 1-minute): the served counts the decisions were made with - columns, one row per
                      key, positive counts of the snapshot tiers summing to the summary's served (V_combview review F5).
Every test reads content and returns (passed, value, detail).
"""
import collections
import csv
import hashlib
import json
import math
import os
import re

from exopipe.phase3.core.validate import content_test

csv.field_size_limit(10 ** 9)


def _DO():
    from exopipe.phase3.lib.catalog import deep_omission as DO
    return DO


def _rows(path):
    with open(path, newline='') as f:
        rd = csv.DictReader(f)
        return list(rd.fieldnames or []), list(rd)


def _f(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _fail(msgs, value, what):
    if msgs:
        return False, len(msgs), '; '.join(msgs[:6]) + (f' (+{len(msgs) - 6} more)' if len(msgs) > 6 else '') + ' | ' + what
    return True, value, what


def tier_of(cols, ctx):
    t = ctx.get('tier')
    if t in ('snapshot', 'sublst'):
        return t
    return 'snapshot' if 'tier' in cols else 'sublst'


def _key(r, tier):
    return (r['lst'], r['band'], r['night'], r['stokes'], r.get('tier') if tier == 'snapshot' else '10min')


TOL = 1e-3      # the statistics are written with 4 significant digits (relative rounding <= 5e-4)


def _le(a, b):
    return a <= b * (1 + TOL) if b >= 0 else a <= b * (1 - TOL)


def _ge(a, b):
    return a >= b * (1 - TOL) if b >= 0 else a >= b * (1 + TOL)


def check_decisions(path, tier=None):
    """(passed, value, detail) of a decision table (module docstring)."""
    DO = _DO()
    from exopipe.phase3.config import CFG
    c = CFG.catalog_omission
    G = DO.groups()
    with open(path, newline='') as f:
        text = f.read()
    cols, rows = _rows(path)
    tier = tier or ('snapshot' if 'tier' in cols else 'sublst')
    if not rows:
        want = DO.EMPTY_SNAPSHOT_HEADER if tier == 'snapshot' else DO.EMPTY_TENMIN_HEADER
        return (text == want, 0, 'no decision (no deep refusal in these LSTs): the header-only table'
                + ('' if text == want else f' - but the file is {text[:80]!r}, not {want!r}'))
    want = DO.SNAPSHOT_COLS if tier == 'snapshot' else DO.TENMIN_COLS
    if tier == 'snapshot' and any(r['gates'].startswith('member ') for r in rows):
        want = want + [DO.SNAPSHOT_WB_COL]         # dde1760: the union header ends with it once a member row is written
    if cols != want:
        return False, None, f'columns {cols} are not production\'s {want}'
    msgs, n = [], {'KEEP': 0, 'OMIT': 0, 'ABSENT': 0, 'member': 0}
    by_key = {}
    seen = set()
    for i, r in enumerate(rows):
        member = r['gates'].startswith('member ')
        k = _key(r, tier)
        kk = k + ('member' if member else 'decision',)
        if kk in seen:
            msgs.append(f'row {i + 2}: a second {kk[-1]} row of {k}')
        seen.add(kk)
        if not member:
            by_key[k] = r
    for i, r in enumerate(rows):
        where = f'row {i + 2} ({r["lst"]} {r["band"]} {r["night"]} {r["stokes"]}{" " + r["tier"] if tier == "snapshot" else ""})'
        v = r['verdict']
        if v not in ('KEEP', 'OMIT', 'ABSENT'):
            msgs.append(f'{where}: verdict {v!r}')
            continue
        if tier == 'snapshot' and r['tier'] not in c.SNAPSHOT_TIERS:
            msgs.append(f'{where}: tier {r["tier"]!r}')
            continue
        if r['gates'].startswith('member '):
            n['member'] += 1
            m = r['gates'][len('member '):]
            if tier == 'snapshot':      # dde1760: per image (wb_omitted_members = the members), or fail closed (none)
                wbm = r.get(DO.SNAPSHOT_WB_COL) or ''
                want_reason = (DO.snapshot_member_image_reason(m, r['band']) if wbm else
                               DO.snapshot_member_failclosed_reason(m, r['band']))
                words = [w for w in m.split(', ')]
                if wbm and wbm != DO.member_list(words):
                    msgs.append(f'{where}: wb_omitted_members {wbm!r} is not its members {words} by frequency')
                    continue
            else:
                want_reason = DO.tenmin_member_reason(m)
            if r['band'] not in G or v != 'OMIT' or r['reason'] != want_reason:
                msgs.append(f'{where}: a wideband-member row must be OMIT, of a band group, with the member reason '
                            f'(band {r["band"]}, verdict {v}, reason {r["reason"][:60]!r})')
                continue
            for word in m.split(', '):
                sb, _, mv = word.rpartition(' ')
                src = by_key.get((r['lst'], sb, r['night'], r['stokes'], r.get('tier') if tier == 'snapshot' else '10min'))
                if sb not in G.get(r['band'], ()) or sb in c.WIDEBAND_EXCEPT or mv not in DO.member_verdicts():
                    msgs.append(f'{where}: member {word!r} is not a non-excepted member of {r["band"]} with a verdict of '
                                f'catalog_omission.WIDEBAND_MEMBER_VERDICTS {DO.member_verdicts()}')
                elif src is None or src['verdict'] != mv:
                    msgs.append(f'{where}: member {word!r} has no decision row of its own night with that verdict '
                                f'({"none" if src is None else src["verdict"]})')
            continue
        gates = r['gates'].split(';')
        if not r['gates'] or any(g not in c.PROPAGATE for g in gates) or not r['deep_attempt']:
            msgs.append(f'{where}: gates {r["gates"]!r} / attempt {r["deep_attempt"]!r} not of the propagating gates')
            continue
        n[v] += 1
        if v == 'ABSENT':
            if r['rms_pass'] or r['rate_pass']:
                msgs.append(f'{where}: ABSENT with a test result')
            ok = (r['reason'] == DO.snapshot_absent_reason(r['tier']) if tier == 'snapshot'
                  else r['reason'] in (DO.TENMIN_NO_WINDOWS, DO.TENMIN_NIGHT_ABSENT))
            if not ok:
                msgs.append(f'{where}: ABSENT reason {r["reason"]!r}')
            continue
        rp, ap = r['rms_pass'], r['rate_pass']
        if rp not in ('0', '1') or ap not in ('0', '1'):
            msgs.append(f'{where}: rms_pass {rp!r} rate_pass {ap!r}')
            continue
        med, thr, rate, rthr = (_f(r[x]) for x in ('tier_median_rms', 'tier_rms_threshold', 'rate', 'rate_threshold'))
        want_rp = '1' if (med is not None and thr is not None and _le(med, thr)) else '0'
        if rp != want_rp and not (rp == '0' and med is not None and thr is not None and _ge(med, thr)):
            msgs.append(f'{where}: rms_pass {rp} but median {r["tier_median_rms"]} vs threshold {r["tier_rms_threshold"]}')
        # dbc40a6: a night outside the view's searched cells has no served rate (blank) and passes the rate condition
        want_ap = rthr is None or (r['rate'] == '' and tier == 'snapshot') or (rate is not None and _le(rate, rthr))
        if (ap == '1') != want_ap and not (ap == '0' and rate is not None and rthr is not None and _ge(rate, rthr)):
            msgs.append(f'{where}: rate_pass {ap} but rate {r["rate"]} vs threshold {r["rate_threshold"]}')
        want_v, want_reason = ('KEEP', DO.PASSES) if (rp, ap) == ('1', '1') else (
            'OMIT', DO.RMS_ABOVE if rp == '0' else DO.RATE_HIGH)
        if (v, r['reason']) != (want_v, want_reason):
            msgs.append(f'{where}: {v} / {r["reason"]!r} with rms_pass {rp} rate_pass {ap}')
    return _fail(msgs, len(rows), f'{len(rows)} decisions ({tier}): KEEP {n["KEEP"]}, OMIT {n["OMIT"]}, ABSENT {n["ABSENT"]}, '
                                  f'wideband-member {n["member"]}')


@content_test('omission_decisions')
def decisions_follow_their_values(path, tier=None, **ctx):
    """deep_omission.csv: production's header, every verdict recomputed from its own row, every member row's members."""
    return check_decisions(path, tier if tier in ('snapshot', 'sublst') else None)


def decisions_path(path, ctx):
    return ctx.get('decisions') or os.path.join(os.path.dirname(os.path.abspath(path)), 'deep_omission.csv')


class ImageCheck:
    """THE PER-IMAGE PREDICATE READ BACK INDEPENDENTLY (V_combview review T4 / T8: the kept product and the build share
    deep_omission's Omission; a mistake there would make them agree on the wrong detections): the cell's wideband member table
    parsed here, a 10-s image by its frame ts, a 1-minute image by its block (clv15 phase3_snapshot_search.integration_minute_key
    of each frame's DATE-OBS + '00', in place - the search's own key). holds(key, image, members) -> True when the image holds
    one of the omitted members, False when not, None when the table has no such image (the stage refuses those, G7)."""

    def __init__(self, proto, members_name):
        self.proto, self.name = proto, members_name
        self._t = {}
        self._mk = None

    def _table(self, lst, night, S, T):
        k = (lst, night, S, T)
        if k not in self._t:
            out = collections.defaultdict(set)
            p = os.path.join(self.proto, '%s_%s' % (lst, night), self.name.format(stokes=S))
            if os.path.exists(p):
                if T != '10s' and self._mk is None:
                    import sys
                    from exopipe.phase3.core import env
                    root = env.legacy_root('clv15')
                    if root not in sys.path:
                        sys.path.insert(0, root)
                    from phase3_snapshot_search import integration_minute_key
                    self._mk = integration_minute_key
                with open(p, newline='') as f:
                    for r in csv.DictReader(f):
                        out[(r['group'], r['ts'] if T == '10s' else self._mk(r['date_obs']) + '00')].add(r['subband'])
            self._t[k] = out
        return self._t[k]

    def holds(self, key, image, members):
        lst, g, night, S, T = key
        mem = self._table(lst, night, S, T).get((g, image))
        return None if mem is None else bool(mem & set(members))


def image_of(r):
    """The image id of a record or kept-product row (a detection or member row by its cand_id, a panel by its file name), or
    None."""
    DO = _DO()
    if r['kind'] in ('panel', 'panel dropped', 'card2 panel', 'card2 panel dropped', 'elevation-floor panel',
                     'elevation-floor panel dropped'):
        m = re.match(r'^(?:[A-Za-z0-9]+_)?rank\d+_\d\dh_\d{4}-\d\d-\d\d_[A-Za-z0-9]+?(\d{14})', r.get('note') or '')
        return (m.group(1)[:8] + '_' + m.group(1)[8:]) if m else None
    m = DO.SNAP_CID.match(r.get('cand_id') or '')
    return r['cand_id'][m.end():] if m else None


def card2_flagged(runs, name, tag_of):
    """{batch tag: {(cand_id, ra_peak 5 decimals, dec_peak 5 decimals)}} of each run's 10-s Card 2 flags, read here (the file's own
    flag column; the stage checks the file against itself, G8)."""
    out = collections.defaultdict(set)
    for r in runs:
        p = os.path.join(r, name)
        if not os.path.exists(p):
            continue
        with open(p, newline='') as f:
            for x in csv.DictReader(f):
                if x['tier'] == '10s' and x['card2_would_remove'] == '1':
                    out[tag_of(r)].add((x['cand_id'], '%.5f' % float(x['ra_peak']), '%.5f' % float(x['dec_peak'])))
    return out


def removed_key(r, tier):
    """The OMIT key a record row names, or None for a consequence row (identity / source / page)."""
    if tier == 'snapshot':
        if r['kind'] in ('detection', 'panel') or (r['kind'] == 'member row' and r.get('lst')):
            return (r['lst'], r['band'], r['night'], r['stokes'], r['tier'])
        return None
    return (r['lst'], r['band'], r['night'], r['stokes'], '10min') if r['kind'] == 'detection' else None


def check_record(path, tier=None, decisions=None):
    """(passed, value, detail) of an omission record against its decisions (module docstring)."""
    DO = _DO()
    cols, rows = _rows(path)
    tier = tier or ('snapshot' if 'batch' in cols else 'sublst')
    want = DO.RECORD_COLS if tier == 'snapshot' else DO.TENMIN_RECORD_COLS
    if cols != want:
        return False, None, f'columns {cols} are not the {tier} record\'s {want}'
    dp = decisions or os.path.join(os.path.dirname(os.path.abspath(path)), 'deep_omission.csv')
    if not os.path.exists(dp):
        return False, None, f'no decision table {dp} beside: the removals cannot be traced to their decisions'
    _c, dec = _rows(dp)
    why = DO.why_of(dec)
    OMWB = DO.omitted_wb(dec) if tier == 'snapshot' else {}
    OM = DO.omitted_keys(dec)
    sp_json = os.path.join(os.path.dirname(os.path.abspath(path)), 'deep_omission_summary.json')
    summ_ = json.load(open(sp_json)) if os.path.exists(sp_json) else {}
    from exopipe.phase3.config import CFG
    ic = ImageCheck(summ_.get('proto') or CFG.paths.SNAPSHOT_FLEET_OUT, CFG.catalog_omission.SNAPSHOT_MEMBERS) if OMWB else None
    msgs, n = [], {}
    det_ids = set()
    n_img = 0
    for i, r in enumerate(rows):
        n[r['kind']] = n.get(r['kind'], 0) + 1
        k = removed_key(r, tier)
        if k is None:
            if tier == 'snapshot' and r['kind'] == 'card2 detection':
                det_ids.add((r['batch'], r['identity']))
                if r['tier'] != '10s' or r['stokes'] != 'V' or r['why'] != DO.CARD2_WHY:
                    msgs.append(f'row {i + 2} card2 detection {r["cand_id"]}: tier {r["tier"]} Stokes {r["stokes"]} (Card 2 acts on '
                                f'10-s Stokes V only) / why {r["why"][:60]!r}')
            continue
        if k not in why:
            msgs.append(f'row {i + 2} {r["kind"]} {r.get("cand_id") or r.get("note")}: {k} is not an OMIT key of the decisions')
        elif r['why'] != why[k]:
            msgs.append(f'row {i + 2} {r["kind"]}: why {r["why"][:80]!r} is not its key\'s decisions {why[k][:80]!r}')
        elif k not in OM and k in OMWB:     # a per-image key: its own image must hold an omitted member (read back here, T4)
            n_img += 1
            img = image_of(r)
            h = ic.holds(k, img, OMWB[k]) if img else None
            if h is not True:
                msgs.append(f'row {i + 2} {r["kind"]} {r.get("cand_id") or r.get("note")}: removed on the per-image key {k}, but its '
                            f'image {img} ' + ('is not in the cell\'s member table' if h is None else
                                               f'holds none of the omitted members {sorted(OMWB[k])}'))
        if r['kind'] == 'detection' and tier == 'snapshot':
            det_ids.add((r['batch'], r['identity']))
    if tier == 'snapshot':
        for i, r in enumerate(rows):
            if r['kind'] == 'identity dropped' and (r['batch'], r['identity']) not in det_ids:
                msgs.append(f'row {i + 2}: identity {r["batch"]} {r["identity"]} dropped with no omitted detection recorded')
    n_det = n.get('detection', 0)
    sp = os.path.join(os.path.dirname(os.path.abspath(path)), 'deep_omission_summary.txt')
    if os.path.exists(sp):
        lines = open(sp).read().splitlines()
        line = lines[0] if lines else ''
        jp = sp[:-len('.txt')] + '.json'
        j = json.load(open(jp)) if os.path.exists(jp) else None
        # 10-minute: production's page line also counts the member rows of the sources re-judged out (sublst_deep_omission_drop.
        # finish: its members_rejudged_<S>_<T> counts, listed in the summary; the stage's json 'rejudged_members')
        n_rjm = 0
        if tier == 'sublst':
            cnt = [ln.split() for ln in lines[2:] if ln.startswith('members_rejudged_')]
            n_rjm = sum(int(x[1]) for x in cnt if len(x) == 2 and x[1].isdigit()) if cnt else \
                int((j or {}).get('rejudged_members') or 0)
            if n.get('source re-judged') and not n_rjm:
                msgs.append(f"{n['source re-judged']} sources re-judged out and no count of their member rows in the summary")
        if line != DO.page_line(n_det + n_rjm):
            msgs.append(f'summary line {line!r} is not the page line of {n_det} detections'
                        + (f' + {n_rjm} member rows of re-judged sources' if n_rjm else ''))
        if j is not None and j.get('omitted') != n_det:
            msgs.append(f'summary json omitted {j.get("omitted")} != {n_det} recorded detections')
    return _fail(msgs, n_det, f'{len(rows)} record rows ({tier}): ' + ', '.join(f'{k} {v}' for k, v in sorted(n.items()))
                 + f'; every removal traced to its OMIT key in {os.path.basename(dp)}'
                 + (f', {n_img} on per-image keys each in an image holding an omitted member' if n_img else ''))


@content_test('omission_record')
def removals_follow_decisions(path, tier=None, decisions=None, **ctx):
    """deep_omission_dropped.csv: every removal traced to its own OMIT key and decision text; counts."""
    return check_record(path, tier if tier in ('snapshot', 'sublst') else None, decisions)


def check_kept(path, decisions=None):
    """(passed, value, detail) of the 10-s / 1-minute kept product (module docstring, K1-K6)."""
    DO = _DO()
    from exopipe.phase3.config import CFG
    d = os.path.dirname(os.path.abspath(path))
    dp = decisions or os.path.join(d, 'deep_omission.csv')
    rp, sp = os.path.join(d, 'deep_omission_dropped.csv'), os.path.join(d, 'deep_omission_summary.json')
    for p in (dp, rp, sp):
        if not os.path.exists(p):
            return False, None, f'no {p} beside: the kept product cannot be traced to its decisions, record and summary'
    with open(path, newline='') as f:
        cols = next(csv.reader(f), [])
    if cols != DO.KEPT_COLS:
        return False, None, f'columns {cols} are not the kept product\'s {DO.KEPT_COLS}'
    dec = _rows(dp)[1]
    OM, OMWB = DO.omitted_keys(dec), DO.omitted_wb(dec)
    _rc, rec = _rows(rp)
    summ = json.load(open(sp))
    runs = {os.path.abspath(r) for r in summ.get('runs') or []}
    ic = ImageCheck(summ.get('proto') or CFG.paths.SNAPSHOT_FLEET_OUT, CFG.catalog_omission.SNAPSHOT_MEMBERS)
    flagged = card2_flagged(sorted(runs), CFG.catalog_omission.SNAPSHOT_CARD2, DO.batch_tag)
    om_det = {(r['batch'], r['identity'], r['cand_id']) for r in rec
              if r['kind'] in ('detection', 'card2 detection', 'elevation-floor 1min detection')}
    dropped_ids = {(r['batch'], r['identity']) for r in rec if r['kind'] == 'identity dropped'}
    # the re-judging's table beside (the omit product's copy of rejudged.csv): its re-judged-out (batch, identity, tier) and notes
    rjp = os.path.join(d, DO.REJUDGED_FILE)
    rj_out, rj_routes = {}, {}
    if os.path.exists(rjp):
        for x in _rows(rjp)[1]:
            if x['qualifies_before'] == '1' and x['qualifies_after'] == '0' and x['reason_after'] != 'every member removed':
                nm, nr = int(x['n_members']), int(x['n_removed'])
                rj_out[(x['batch'], x['identity_jname'], x['tier'])] = 're-judged on %d of %d members (%s removed): %s' % (
                    nm - nr, nm, x['removed_by'], x['reason_after'])
            elif x['qualifies_before'] == '1' and x['qualifies_after'] == '1':
                rj_routes[(x['batch'], x['identity_jname'], x['tier'])] = x['routes_after']
    else:
        return False, None, f'no {DO.REJUDGED_FILE} beside: the re-judged sources cannot be traced'
    cat_rx = {'sources': re.compile(r'^source_catalogs_([IV])_(10s|1min)/.+_sources_survivors\.csv$'),
              'members': re.compile(r'^source_catalogs_([IV])_(10s|1min)/.+_source_members\.csv$')}
    SOURCE_KINDS = ('source dropped', 'source re-pointed', 'source routes', 'source re-judged')
    MEMBER_KINDS = ('member row dropped', 'member row re-judged', 'card2 member row dropped', 'elevation-floor member row dropped')
    msgs, n = [], collections.Counter()
    # pass 1: the per-tier rows (few), checked against the record; the re-pointed sources' new bests to find in pass 2
    per_tier = collections.defaultdict(collections.Counter)
    targets = {}
    rj_seen = collections.Counter()
    with open(path, newline='') as f:
        for i, r in enumerate(csv.DictReader(f)):
            kd = r['kind']
            n[kd] += 1
            where = f'row {i + 2} {kd} {r.get("source_id") or r.get("cand_id") or ""}'
            if kd not in DO.KEPT_KINDS:
                msgs.append(f'K1 {where}: kind {kd!r} is not one of {DO.KEPT_KINDS}')
                continue
            if r['run'] not in runs or DO.batch_tag(r['run']) != r['batch']:
                msgs.append(f'K1 {where}: run {r["run"]} / batch {r["batch"]} is not a served run of the summary')
            if kd == 'detection kept':
                continue
            if kd in SOURCE_KINDS or kd in MEMBER_KINDS:
                m = cat_rx['members' if kd in MEMBER_KINDS else 'sources'].match(r['catalog'])
                if not m or (m.group(1), m.group(2)) != (r['stokes'], r['tier']):
                    msgs.append(f'K3 {where}: catalogue {r["catalog"]!r} is not its tier\'s ({r["stokes"]} {r["tier"]})')
            key = (r['lst'], r['band'], r['night'], r['stokes'], r['tier']) if r['lst'] else None
            if kd in ('member row dropped', 'panel dropped'):
                if key is not None and key not in OM and not (key in OMWB and ic.holds(key, image_of(r), OMWB[key]) is True):
                    msgs.append(f'K4 {where}: its key {key} is not an OMIT key (nor a per-image one holding its image)')
                elif key is None and not (r['note'] == DO.IDENTITY_EMPTY_NOTE and (r['batch'], r['identity']) in dropped_ids):
                    msgs.append(f'K4 {where}: neither an OMIT key nor its dropped identity')
            if kd in ('card2 member row dropped', 'card2 panel dropped') and (r['stokes'], r['tier']) != ('V', '10s'):
                msgs.append(f'K4 {where}: a Card 2 row of {r["stokes"]} {r["tier"]} (Card 2 acts on 10-s Stokes V only)')
            if kd in ('elevation-floor member row dropped', 'elevation-floor panel dropped') and r['tier'] != '1min':
                msgs.append(f'K4 {where}: an elevation-floor row of the {r["tier"]} tier (the floor acts on 1-min rows only)')
            if kd == 'source dropped':
                ok = ((r['batch'], r['identity']) in dropped_ids if r['note'] == DO.IDENTITY_EMPTY_NOTE
                      else r['note'] in (DO.SOURCE_EMPTY_NOTE, DO.SOURCE_EMPTY_CARD2_NOTE)
                      and (r['batch'], r['identity']) not in dropped_ids)
                if not ok:
                    msgs.append(f'K4 {where}: note {r["note"]!r} does not follow its identity\'s drop')
            if kd in ('source re-judged', 'member row re-judged'):
                t = (r['batch'], r['identity'], r['tier'])
                if t not in rj_out:
                    msgs.append(f'K6 {where}: its (batch identity, tier) {t} is not re-judged out in {DO.REJUDGED_FILE}')
                elif kd == 'source re-judged':
                    rj_seen[t] += 1
                    if r['note'] != rj_out[t]:
                        msgs.append(f'K6 {where}: note {r["note"][:80]!r} is not the re-judging\'s {rj_out[t][:80]!r}')
            if kd in ('source re-pointed', 'source routes'):
                try:
                    fl = json.loads(r['fields'], object_pairs_hook=collections.OrderedDict)
                except ValueError:
                    fl = {}
                t = (r['batch'], r['identity'], r['tier'])
                if 'rejudged_routes' in fl and fl['rejudged_routes'] != rj_routes.get(t):
                    msgs.append(f'K6 {where}: rejudged_routes {fl["rejudged_routes"]!r} is not the re-judging\'s {rj_routes.get(t)!r}')
                if kd == 'source routes':
                    if list(fl) != ['rejudged_routes']:
                        msgs.append(f'K4 {where}: fields {r["fields"][:120]!r} are not the routes of a source still qualifying')
                else:
                    want = [x for x in DO.REPOINT_FIELDS] + (['rejudged_routes'] if 'rejudged_routes' in fl else [])
                    lst_, date_ = (r['cand_id'].split('_')[0:2] + ['', ''])[:2] if r['cand_id'] else ('', '')
                    band = r['catalog'].rsplit('/', 1)[-1][:-len('_sources_survivors.csv')] if r['catalog'] else ''
                    if list(fl) != want or fl.get('best_snr') != r['snr'] or fl.get('deep_omission') != DO.REPOINT_NOTE \
                            or fl.get('best_cand_id') != '%s_%s_%s_%s' % (r['identity'], lst_, date_, band):
                        msgs.append(f'K4 {where}: fields {r["fields"][:160]!r} do not name its new best {r["cand_id"]} '
                                    f'(production\'s re-point, dbc40a6)')
                    targets[(r['batch'], r['identity'], r['cand_id'])] = where
            src = r['source_id']
            kk = {'source dropped': (r['batch'], r['identity'], src, r['stokes'], r['tier'], r['note']),
                  'source re-pointed': (r['batch'], r['identity'], src, r['stokes'], r['tier'], r['note']),
                  'source routes': (r['batch'], r['identity'], src, r['stokes'], r['tier']),
                  'source re-judged': (r['batch'], r['identity'], src, r['stokes'], r['tier'], r['note']),
                  'member row dropped': (r['batch'], src, r['cand_id'], r['stokes'], r['tier']),
                  'member row re-judged': (r['batch'], src, r['cand_id'], r['stokes'], r['tier']),
                  'card2 member row dropped': (r['batch'], src, r['cand_id'], r['stokes'], r['tier']),
                  'panel dropped': (r['batch'], src, r['stokes'], r['tier'], r['note']),
                  'card2 panel dropped': (r['batch'], src, r['stokes'], r['tier'], r['note']),
                  'elevation-floor member row dropped': (r['batch'], src, r['cand_id'], r['stokes'], r['tier']),
                  'elevation-floor panel dropped': (r['batch'], src, r['stokes'], r['tier'], r['note'])}[kd]
            per_tier[kd][kk] += 1
    miss_rj = sorted(t for t in rj_out if not rj_seen[t])
    if miss_rj:
        msgs.append(f'K6 {len(miss_rj)} re-judged-out (batch identity, tier) of {DO.REJUDGED_FILE} drop no per-tier source row '
                    f'(e.g. {miss_rj[:2]}) - production\'s T7')
    want = {'source dropped': ('source dropped', lambda x: (x['batch'], x['identity'], x['source'], x['stokes'], x['tier'],
                                                            x['note'])),
            'source re-pointed': ('source partial', lambda x: (x['batch'], x['identity'], x['source'], x['stokes'], x['tier'],
                                                               x['note'])),
            'source re-judged': ('source re-judged', lambda x: (x['batch'], x['identity'], x['source'], x['stokes'], x['tier'],
                                                                x['note'])),
            'member row dropped': ('member row', lambda x: (x['batch'], x['source'], x['cand_id'], x['stokes'], x['tier'])),
            'member row re-judged': ('member row re-judged', lambda x: (x['batch'], x['source'], x['cand_id'], x['stokes'],
                                                                        x['tier'])),
            'card2 member row dropped': ('card2 member row', lambda x: (x['batch'], x['source'], x['cand_id'], x['stokes'],
                                                                       x['tier'])),
            'panel dropped': ('panel', lambda x: (x['batch'], x['source'], x['stokes'], x['tier'], x['note'])),
            'card2 panel dropped': ('card2 panel', lambda x: (x['batch'], x['source'], x['stokes'], x['tier'], x['note'])),
            'elevation-floor member row dropped': ('elevation-floor member row', lambda x: (x['batch'], x['source'], x['cand_id'],
                                                                                           x['stokes'], x['tier'])),
            'elevation-floor panel dropped': ('elevation-floor panel', lambda x: (x['batch'], x['source'], x['stokes'], x['tier'],
                                                                                 x['note']))}
    for kd, (rk, kf) in want.items():
        exp = collections.Counter(kf(x) for x in rec if x['kind'] == rk)
        if per_tier[kd] != exp:
            extra, miss = per_tier[kd] - exp, exp - per_tier[kd]
            msgs.append(f'K3 {kd}: {sum(extra.values())} rows not in the record\'s {rk!r} rows (e.g. {list(extra)[:1]}), '
                        f'{sum(miss.values())} of the record\'s missing (e.g. {list(miss)[:1]})')
    # pass 2: the kept detections, streamed. N2 (Q fix verification): a detection key (cand_id, ra_peak, dec_peak) kept twice
    # refuses (G3's key: one detection in two places), and the kept rows' digest must be the summary's and, with the omitted
    # and the Card 2 detections' digests, add up to the served set's - so a kept row substituted, duplicated or moved to another
    # identity refuses; the record's detection rows must be the omitted ones (their (batch, identity, cand_id) digest). The
    # per-image predicate and Card 2 are read back here (T4): a kept detection whose image holds an omitted member, or whose
    # key Card 2 flags, refuses
    dg = summ.get('digests') or {}
    if not all(dg.get(k) for k in ('served', 'kept', 'omitted', 'omitted_cid', 'card2', 'elev1min')):
        msgs.append('K2 the summary has no served / kept / omitted / card2 / elev1min digests (deep_omission_summary.json written '
                    'before the served-set fingerprint, Q fix verification N2, before Card 2 or before the 1-min elevation floor, '
                    'exopipe_ops b114ff3): re-run the omit stage')
    seen_keys, n_dup = set(), 0
    D_kept = DO.MultisetDigest()
    kept_ids, found = set(), set()
    kept_per_key = collections.Counter()          # K5: the served counts = kept + omitted (not Card 2), per decision key
    n_img = 0
    with open(path, newline='') as f:
        for i, r in enumerate(csv.DictReader(f)):
            if r['kind'] != 'detection kept':
                continue
            idk = (r['batch'], r['identity'])
            k = (r['lst'], r['band'], r['night'], r['stokes'], r['tier'])
            if k in OM:
                msgs.append(f'K2 row {i + 2} detection kept {r["cand_id"]}: its key {k} is an OMIT key')
            elif k in OMWB:
                n_img += 1
                if ic.holds(k, image_of(r), OMWB[k]) is not False:
                    msgs.append(f'K2 row {i + 2} detection kept {r["cand_id"]}: its image holds an omitted member of {k} (or is '
                                f'not in the member table)')
            if r['tier'] == '10s' and r['stokes'] == 'V' and (r['cand_id'], '%.5f' % float(r['ra_peak']),
                                                              '%.5f' % float(r['dec_peak'])) in flagged.get(r['batch'], ()):
                msgs.append(f'K2 row {i + 2} detection kept {r["cand_id"]}: Card 2 flags it ({CFG.catalog_omission.SNAPSHOT_CARD2})')
            if (r['batch'], r['identity'], r['cand_id']) in om_det:
                msgs.append(f'K2 row {i + 2} detection kept {r["cand_id"]}: the record removes it')
            if idk in dropped_ids:
                msgs.append(f'K2 row {i + 2} detection kept {r["cand_id"]}: its identity {idk} is dropped in the record')
            dk = (r['cand_id'], r['ra_peak'], r['dec_peak'])
            h = hashlib.blake2b('\x1f'.join(dk).encode('utf-8'), digest_size=16).digest()
            if h in seen_keys:
                n_dup += 1
                if n_dup <= 3:
                    msgs.append(f'K2 row {i + 2} detection kept {r["cand_id"]} ({r["ra_peak"]}, {r["dec_peak"]}): this detection '
                                f'key is kept twice')
            seen_keys.add(h)
            D_kept.add(DO.detection_item(r['batch'], r['identity'], dk))
            kept_per_key[(r['lst'], r['night'], r['band'], r['stokes'], r['tier'])] += 1
            kept_ids.add(idk)
            t = (r['batch'], r['identity'], r['cand_id'])
            if t in targets:
                found.add(t)
    if n_dup > 3:
        msgs.append(f'K2 {n_dup} detection keys kept twice in all')
    if dg.get('kept') and dg.get('omitted') and dg.get('served') and dg.get('card2') and dg.get('elev1min'):
        if D_kept.text() != dg['kept']:
            msgs.append(f'K2 the kept rows\' digest {D_kept.text()[:24]}... is not the summary\'s kept digest '
                        f'{dg["kept"][:24]}... (a kept detection substituted, duplicated, lost or moved to another identity)')
        if D_kept + DO.MultisetDigest(dg['omitted']) + DO.MultisetDigest(dg['card2']) + DO.MultisetDigest(dg['elev1min']) != \
                DO.MultisetDigest(dg['served']):
            msgs.append('K2 the kept rows + the omitted + the Card 2 + the 1-min elevation-floor detections do not reproduce the '
                        'served set\'s digest')
    if dg.get('omitted_cid'):
        D_rec = DO.MultisetDigest()
        for r in rec:
            if r['kind'] == 'detection':
                D_rec.add('\x1f'.join((r['batch'], r['identity'], r['cand_id'])))
        if D_rec.text() != dg['omitted_cid']:
            msgs.append('K2 the record\'s detection rows are not the omitted detections of the summary\'s digest')
    for t, where in targets.items():
        if t not in found:
            msgs.append(f'K4 {where}: its new best {t[2]} is no kept detection of its identity {t[0]} {t[1]}')
    # K5 (V_combview review F5): the served counts the decisions were made with (deep_omission_served.csv beside) are the kept
    # plus the omitted detections of each (LST, night, band, Stokes, tier), the Card 2 ones (dbc40a6) and the 1-min ones below the
    # elevation floor (b114ff3) not counted
    svp = os.path.join(d, 'deep_omission_served.csv')
    if not os.path.exists(svp):
        msgs.append('K5 no deep_omission_served.csv beside: the counts the decisions were made with are not recorded')
    else:
        served = DO.read_served(svp)
        omitted = collections.Counter((r['lst'], r['night'], r['band'], r['stokes'], r['tier']) for r in rec
                                      if r['kind'] == 'detection' and r.get('note') not in (DO.ALSO_CARD2, DO.ALSO_ELEV1MIN))
        want_c = kept_per_key + omitted
        if served != dict(want_c):
            bad = sorted(k for k in set(served) | set(want_c) if served.get(k, 0) != want_c.get(k, 0))
            msgs.append(f'K5 {len(bad)} served counts are not the kept + omitted detections of their key (e.g. {bad[:2]})')
    nk = n['detection kept']
    if not (nk == summ.get('kept') == summ.get('served', -1) - summ.get('omitted', 0) - summ.get('card2', 0)
            - summ.get('elev1min', 0)):
        msgs.append(f'K2 {nk} kept detections != the summary\'s kept {summ.get("kept")} (served {summ.get("served")} - omitted '
                    f'{summ.get("omitted")} - Card 2 {summ.get("card2")} - 1-min elevation floor {summ.get("elev1min")})')
    if len(kept_ids) != summ.get('identities_kept') or len(kept_ids) + len(dropped_ids) != summ.get('identities'):
        msgs.append(f'K2 identities: {len(kept_ids)} kept + {len(dropped_ids)} dropped vs the summary\'s '
                    f'{summ.get("identities_kept")} kept of {summ.get("identities")}')
    return _fail(msgs, nk, f'kept product: ' + ', '.join(f'{k} {v}' for k, v in sorted(n.items()))
                 + f'; traced to the decisions, the record, the re-judging and the summary beside ({n_img} kept detections on '
                   f'per-image keys read back)')


def check_rejudged(path):
    """(passed, value, detail) of deep_omission_rejudged.csv (production's rejudged.csv, dbc40a6): its columns, each (batch,
    identity, tier) once, qualifies 0 / 1, the removal counts (n_removed >= 1, at most n_members, removed_by adding up), 'every
    member removed' exactly when nothing is left, and the summary's re-judging counts recomputed from the rows."""
    DO = _DO()
    cols, rows = _rows(path)
    want = ['batch', 'identity_jname', 'tier', 'src_index', 'qualifies_before', 'qualifies_after', 'reason_after',
            'routes_after', 'n_members', 'n_removed', 'removed_by']
    if rows and cols != want or not rows and cols not in (['batch'], want):
        return False, None, f'columns {cols} are not production\'s rejudged.csv {want}'
    msgs, seen = [], set()
    cnt = collections.Counter()
    for i, r in enumerate(rows):
        k = (r['batch'], r['identity_jname'], r['tier'])
        if k in seen:
            msgs.append(f'row {i + 2}: {k} twice')
        seen.add(k)
        try:
            q0, q1, nm, nr = int(r['qualifies_before']), int(r['qualifies_after']), int(r['n_members']), int(r['n_removed'])
            parts = [p.rsplit(' ', 1) for p in r['removed_by'].split(';') if p]
            nrb = sum(int(x[1]) for x in parts)
        except (ValueError, IndexError):
            msgs.append(f'row {i + 2}: unreadable counts {r}')
            continue
        if q0 not in (0, 1) or q1 not in (0, 1) or not 1 <= nr <= nm or nrb != nr or \
                any(not set(x[0].split(';')) <= {'omit', 'card2', 'elev1min'} for x in parts):
            msgs.append(f'row {i + 2} {k}: qualifies {q0}/{q1}, removed {nr} of {nm}, removed_by {r["removed_by"]!r}')
        if (r['reason_after'] == 'every member removed') != (q1 == 0 and nr == nm and r['routes_after'] == ''):
            msgs.append(f'row {i + 2} {k}: reason {r["reason_after"]!r} with {nr} of {nm} removed')
        if q1 == 1 and not r['routes_after']:
            msgs.append(f'row {i + 2} {k}: qualifies after with no route')
        if q0 == 1:
            cnt['touched'] += 1
            cnt['still_qualify' if q1 == 1 else ('every_member_removed' if r['reason_after'] == 'every member removed'
                                                 else 'rejudged_out')] += 1
    sp = os.path.join(os.path.dirname(os.path.abspath(path)), 'deep_omission_summary.json')
    if os.path.exists(sp):
        rj = json.load(open(sp)).get('rejudge') or {}
        for k in ('touched', 'still_qualify', 'rejudged_out', 'every_member_removed'):
            if rj.get(k) != cnt[k]:
                msgs.append(f'the summary\'s re-judging {k} {rj.get(k)} is not the table\'s {cnt[k]}')
    else:
        msgs.append('no deep_omission_summary.json beside')
    return _fail(msgs, len(rows), f'{len(rows)} touched (source, tier): ' + ', '.join(f'{k} {v}' for k, v in sorted(cnt.items())))


@content_test('omission_rejudged')
def rejudged_table(path, **ctx):
    """deep_omission_rejudged.csv: the re-judging's table (production's rejudged.csv), its counts."""
    return check_rejudged(path)


def check_served(path):
    """(passed, value, detail) of deep_omission_served.csv: production's key columns, one row per key, positive counts of the
    snapshot tiers, their sum = the summary's served less its Card 2 detections (dbc40a6: not counted as served in the rate test;
    b114ff3: nor the 1-min ones below the elevation floor;
    the per-key tie to the kept and omitted detections is K5 of omission_kept)."""
    DO = _DO()
    cols, rows = _rows(path)
    if cols != DO.SERVED_COLS:
        return False, None, f'columns {cols} are not {DO.SERVED_COLS}'
    from exopipe.phase3.config import CFG
    msgs, seen, tot = [], set(), 0
    for i, r in enumerate(rows):
        k = tuple(r[c] for c in DO.SERVED_COLS[:-1])
        if k in seen:
            msgs.append(f'row {i + 2}: key {k} twice')
        seen.add(k)
        if r['tier'] not in CFG.catalog_omission.SNAPSHOT_TIERS or not r['n'].isdigit() or int(r['n']) < 1:
            msgs.append(f'row {i + 2}: tier {r["tier"]!r} / n {r["n"]!r}')
            continue
        tot += int(r['n'])
    sp = os.path.join(os.path.dirname(os.path.abspath(path)), 'deep_omission_summary.json')
    if not os.path.exists(sp):
        msgs.append('no deep_omission_summary.json beside')
    else:
        sj = json.load(open(sp))
        # dbc40a6 / b114ff3: the Card 2 detections and the 1-min ones below the elevation floor are not counted as served
        want = sj.get('served', -1) - sj.get('card2_served', 0) - sj.get('elev1min_served', 0)
        if want != tot:
            msgs.append(f'the counts sum to {tot}, the summary\'s served less its Card 2 and elevation-floor detections is {want}')
    return _fail(msgs, tot, f'{len(rows)} (LST, night, band, Stokes, tier) keys, {tot} served detections')


@content_test('omission_served')
def served_counts_add_up(path, **ctx):
    """deep_omission_served.csv: the served counts the decisions were made with."""
    return check_served(path)


@content_test('omission_kept')
def kept_product_follows_the_drop(path, tier=None, decisions=None, **ctx):
    """deep_omission_kept.csv: what the snapshot tier's cumulative step applies, traced to the decisions, record and summary."""
    return check_kept(path, decisions)
