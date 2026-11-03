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
                      json count the record's detections. The per-instant test of this class is in tests_per_instant.py.
  omission_kept       deep_omission_kept.csv (10-s / 1-minute; the record, decisions and summary beside, or ctx decisions=):
                      the drop the snapshot tier's cumulative step applies (Q review F2). K1 the writer's columns and kinds,
                      every row of a served run of the summary (its batch tag); K2 every 'detection kept' row has no OMIT key, is
                      no detection the record omits, of no identity it drops; their number = the summary's kept = served -
                      omitted, their identities = identities_kept, kept + dropped identities = identities; no detection key kept
                      twice, the kept rows' multiset digest = the summary's and kept + omitted = the served set's digest, the
                      record's detection rows = the omitted digest (Q fix verification N2); K3 the per-tier rows
                      are the record's one to one ('source dropped' / 'source re-pointed' = 'source partial' / 'member row
                      dropped' = 'member row' / 'panel dropped' = 'panel'), each on its tier's catalogue file; K4 a re-pointed
                      source's fields name its new best, a kept detection of its own identity, with production's note; a row
                      with a key has an OMIT key, one without names its identity's drop. Streamed (two passes over the file).
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
            want_reason = (DO.snapshot_member_reason(m, r['band']) if tier == 'snapshot' else DO.tenmin_member_reason(m))
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
        want_ap = rthr is None or (rate is not None and _le(rate, rthr))
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
    msgs, n = [], {}
    det_ids = set()
    for i, r in enumerate(rows):
        n[r['kind']] = n.get(r['kind'], 0) + 1
        k = removed_key(r, tier)
        if k is None:
            continue
        if k not in why:
            msgs.append(f'row {i + 2} {r["kind"]} {r.get("cand_id") or r.get("note")}: {k} is not an OMIT key of the decisions')
        elif r['why'] != why[k]:
            msgs.append(f'row {i + 2} {r["kind"]}: why {r["why"][:80]!r} is not its key\'s decisions {why[k][:80]!r}')
        if r['kind'] == 'detection' and tier == 'snapshot':
            det_ids.add((r['batch'], r['identity']))
    if tier == 'snapshot':
        for i, r in enumerate(rows):
            if r['kind'] == 'identity dropped' and (r['batch'], r['identity']) not in det_ids:
                msgs.append(f'row {i + 2}: identity {r["batch"]} {r["identity"]} dropped with no omitted detection recorded')
    n_det = n.get('detection', 0)
    sp = os.path.join(os.path.dirname(os.path.abspath(path)), 'deep_omission_summary.txt')
    if os.path.exists(sp):
        line = open(sp).readline().rstrip('\n')
        if line != DO.page_line(n_det):
            msgs.append(f'summary line {line!r} is not the page line of {n_det} detections')
        jp = sp[:-len('.txt')] + '.json'
        if os.path.exists(jp):
            j = json.load(open(jp))
            if j.get('omitted') != n_det:
                msgs.append(f'summary json omitted {j.get("omitted")} != {n_det} recorded detections')
    return _fail(msgs, n_det, f'{len(rows)} record rows ({tier}): ' + ', '.join(f'{k} {v}' for k, v in sorted(n.items()))
                 + f'; every removal traced to its OMIT key in {os.path.basename(dp)}')


@content_test('omission_record')
def removals_follow_decisions(path, tier=None, decisions=None, **ctx):
    """deep_omission_dropped.csv: every removal traced to its own OMIT key and decision text; counts."""
    return check_record(path, tier if tier in ('snapshot', 'sublst') else None, decisions)


def check_kept(path, decisions=None):
    """(passed, value, detail) of the 10-s / 1-minute kept product (module docstring, K1-K4)."""
    DO = _DO()
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
    OM = DO.omitted_keys(_rows(dp)[1])
    _rc, rec = _rows(rp)
    summ = json.load(open(sp))
    runs = {os.path.abspath(r) for r in summ.get('runs') or []}
    om_det = {(r['batch'], r['identity'], r['cand_id']) for r in rec if r['kind'] == 'detection'}
    dropped_ids = {(r['batch'], r['identity']) for r in rec if r['kind'] == 'identity dropped'}
    cat_rx = {'sources': re.compile(r'^source_catalogs_([IV])_(10s|1min)/.+_sources_survivors\.csv$'),
              'members': re.compile(r'^source_catalogs_([IV])_(10s|1min)/.+_source_members\.csv$')}
    msgs, n = [], collections.Counter()
    # pass 1: the per-tier rows (few), checked against the record; the re-pointed sources' new bests to find in pass 2
    per_tier = collections.defaultdict(collections.Counter)
    targets = {}
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
            if kd in ('source dropped', 'source re-pointed', 'member row dropped'):
                m = cat_rx['members' if kd == 'member row dropped' else 'sources'].match(r['catalog'])
                if not m or (m.group(1), m.group(2)) != (r['stokes'], r['tier']):
                    msgs.append(f'K3 {where}: catalogue {r["catalog"]!r} is not its tier\'s ({r["stokes"]} {r["tier"]})')
            key = (r['lst'], r['band'], r['night'], r['stokes'], r['tier']) if r['lst'] else None
            if kd in ('member row dropped', 'panel dropped'):
                if key is not None and key not in OM:
                    msgs.append(f'K4 {where}: its key {key} is not an OMIT key')
                elif key is None and not (r['note'] == 'identity left with no detection'
                                          and (r['batch'], r['identity']) in dropped_ids):
                    msgs.append(f'K4 {where}: neither an OMIT key nor its dropped identity')
            if kd == 'source dropped':
                ok = ((r['batch'], r['identity']) in dropped_ids if r['note'] == 'identity left with no detection'
                      else r['note'] == 'every detection omitted' and (r['batch'], r['identity']) not in dropped_ids)
                if not ok:
                    msgs.append(f'K4 {where}: note {r["note"]!r} does not follow its identity\'s drop')
            if kd == 'source re-pointed':
                try:
                    fl = json.loads(r['fields'])
                except ValueError:
                    fl = {}
                if (fl.get('best_cand_id'), fl.get('best_snr'), fl.get('deep_omission')) != (r['cand_id'], r['snr'],
                                                                                           DO.SNAPSHOT_PARTIAL_NOTE):
                    msgs.append(f'K4 {where}: fields {r["fields"][:120]!r} do not name its new best {r["cand_id"]}')
                targets[(r['batch'], r['identity'], r['cand_id'])] = where
            src = r['source_id']
            kk = {'source dropped': (r['batch'], r['identity'], src, r['stokes'], r['tier'], r['note']),
                  'source re-pointed': (r['batch'], r['identity'], src, r['stokes'], r['tier'], r['note']),
                  'member row dropped': (r['batch'], src, r['cand_id'], r['stokes'], r['tier']),
                  'panel dropped': (r['batch'], src, r['stokes'], r['tier'], r['note'])}[kd]
            per_tier[kd][kk] += 1
    want = {'source dropped': ('source dropped', lambda x: (x['batch'], x['identity'], x['source'], x['stokes'], x['tier'],
                                                            x['note'])),
            'source re-pointed': ('source partial', lambda x: (x['batch'], x['identity'], x['source'], x['stokes'], x['tier'],
                                                               x['note'])),
            'member row dropped': ('member row', lambda x: (x['batch'], x['source'], x['cand_id'], x['stokes'], x['tier'])),
            'panel dropped': ('panel', lambda x: (x['batch'], x['source'], x['stokes'], x['tier'], x['note']))}
    for kd, (rk, kf) in want.items():
        exp = collections.Counter(kf(x) for x in rec if x['kind'] == rk)
        if per_tier[kd] != exp:
            extra, miss = per_tier[kd] - exp, exp - per_tier[kd]
            msgs.append(f'K3 {kd}: {sum(extra.values())} rows not in the record\'s {rk!r} rows (e.g. {list(extra)[:1]}), '
                        f'{sum(miss.values())} of the record\'s missing (e.g. {list(miss)[:1]})')
    # pass 2: the kept detections, streamed. N2 (Q fix verification): a detection key (cand_id, ra_peak, dec_peak) kept twice
    # refuses (G3's key: one detection in two places), and the kept rows' digest must be the summary's and, with the omitted
    # detections' digest, add up to the served set's - so a kept row substituted, duplicated or moved to another identity
    # refuses; the record's detection rows must be the omitted ones (their (batch, identity, cand_id) digest)
    dg = summ.get('digests') or {}
    if not all(dg.get(k) for k in ('served', 'kept', 'omitted', 'omitted_cid')):
        msgs.append('K2 the summary has no served / kept / omitted digests (deep_omission_summary.json written before the '
                    'served-set fingerprint, Q fix verification N2): re-run the omit stage')
    seen_keys, n_dup = set(), 0
    D_kept = DO.MultisetDigest()
    kept_ids, found = set(), set()
    with open(path, newline='') as f:
        for i, r in enumerate(csv.DictReader(f)):
            if r['kind'] != 'detection kept':
                continue
            idk = (r['batch'], r['identity'])
            k = (r['lst'], r['band'], r['night'], r['stokes'], r['tier'])
            if k in OM:
                msgs.append(f'K2 row {i + 2} detection kept {r["cand_id"]}: its key {k} is an OMIT key')
            if (r['batch'], r['identity'], r['cand_id']) in om_det:
                msgs.append(f'K2 row {i + 2} detection kept {r["cand_id"]}: the record omits it')
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
            kept_ids.add(idk)
            t = (r['batch'], r['identity'], r['cand_id'])
            if t in targets:
                found.add(t)
    if n_dup > 3:
        msgs.append(f'K2 {n_dup} detection keys kept twice in all')
    if dg.get('kept') and dg.get('omitted') and dg.get('served'):
        if D_kept.text() != dg['kept']:
            msgs.append(f'K2 the kept rows\' digest {D_kept.text()[:24]}... is not the summary\'s kept digest '
                        f'{dg["kept"][:24]}... (a kept detection substituted, duplicated, lost or moved to another identity)')
        if D_kept + DO.MultisetDigest(dg['omitted']) != DO.MultisetDigest(dg['served']):
            msgs.append('K2 the kept rows + the omitted detections do not reproduce the served set\'s digest')
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
    nk = n['detection kept']
    if not (nk == summ.get('kept') == summ.get('served', -1) - summ.get('omitted', 0)):
        msgs.append(f'K2 {nk} kept detections != the summary\'s kept {summ.get("kept")} (served {summ.get("served")} - omitted '
                    f'{summ.get("omitted")})')
    if len(kept_ids) != summ.get('identities_kept') or len(kept_ids) + len(dropped_ids) != summ.get('identities'):
        msgs.append(f'K2 identities: {len(kept_ids)} kept + {len(dropped_ids)} dropped vs the summary\'s '
                    f'{summ.get("identities_kept")} kept of {summ.get("identities")}')
    return _fail(msgs, nk, f'kept product: ' + ', '.join(f'{k} {v}' for k, v in sorted(n.items()))
                 + '; traced to the decisions, the record and the summary beside')


@content_test('omission_kept')
def kept_product_follows_the_drop(path, tier=None, decisions=None, **ctx):
    """deep_omission_kept.csv: what the snapshot tier's cumulative step applies, traced to the decisions, record and summary."""
    return check_kept(path, decisions)
