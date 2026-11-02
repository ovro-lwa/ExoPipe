"""Content tests of the deep-omission stage (lib/catalog/deep_omission.py), registered with core/validate.

Classes:
  omission_decisions  deep_omission.csv: production's header for the tier; every verdict follows from the row's own values -
                      a decision row names only propagating gates and its attempt; ABSENT has no statistics; KEEP exactly when
                      rms_pass and rate_pass, each recomputed from the row's numbers (4 significant digits as written; a nan
                      threshold never passes the noise test and always passes the rate test); the reason is the verdict's; a
                      wideband-member row is OMIT, of a band group, and every member it names is a decision row of the same
                      LST, night, Stokes and tier with that verdict, a member of the group and not WIDEBAND_EXCEPT; one row per
                      (key, tier) and kind.
  omission_record     deep_omission_dropped.csv (with the decisions beside, or ctx decisions=): production's columns for the
                      tier; every removed detection, member row and panel has an OMIT key and the 'why' of exactly that key's
                      decisions; every dropped identity (10-s / 1-minute) has a recorded detection; the summary's page line and
                      json count the record's detections. The per-instant test of this class is in tests_per_instant.py.
Every test reads content and returns (passed, value, detail).
"""
import csv
import json
import math
import os

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
                if sb not in G.get(r['band'], ()) or sb in c.WIDEBAND_EXCEPT or mv not in ('OMIT', 'ABSENT'):
                    msgs.append(f'{where}: member {word!r} is not a non-excepted member of {r["band"]} omitted or absent')
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
