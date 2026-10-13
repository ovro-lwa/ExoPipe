"""Content tests of the snapshot-tier `exopipe3 search` product classes (core/validate discovers this module).

Classes (spec search.md section 6, snapshot rows; files under <out>/<cell>/ = today's snapshot_fleet_out/<cell>/):
  snapshot_search    runinfo_<tag>.json, tested against its own tables (noise, candidates, events, admission)
  snapshot_1min      noise_<tag>_1min.csv: 1-minute block membership by integration midpoint (snapshot_frametime_content_test
                     test B, from the shipped 10-s noise table instead of the frames)
  snapshot_wideband  wideband_outcome_<S>.json: every group accounted for (v9:246-264) + member pairing by integration
                     (test A, from wideband_members_<S>.csv)
  snapshot_model     model_<tag>.fits: 2-D, celestial WCS, finite fraction >= 0.5
  night_refusal      nightrefused_<tag>.json (clv15 769778b record): status, counts = frames, no runinfo for the same tag
  snapshot_table     every other shipped CSV (admission, candidates, events, noise, members, ...): parses, rectangular, a
                     0-byte candidates/events table only where its runinfo counts 0 rows (S-M1 today)
  injection_book     injected_<sb>_<S>.csv: one row per injection inside the frame
tag = <subband|group>_<Stokes>_<mode>. A test returns (passed, value, detail) and reads only the file and its siblings.
"""
import csv
import datetime
import json
import math
import os
import re

from exopipe.phase3.core.validate import content_test

csv.field_size_limit(100000000)
_TAG = re.compile(r'^(?P<kind>runinfo|nightrefused|model|noise|candidates|events)_(?P<band>\d+MHz|Red|Green|Blue|Full)_'
                  r'(?P<stokes>[IV])_(?P<mode>median|master)(?P<one>_1min)?\.(?:json|fits|csv)$')
_GROUPS = ('Red', 'Green', 'Blue', 'Full')


def _cfg():
    from exopipe.phase3.config import CFG
    return CFG.search_snapshot


def _tag(path):
    m = _TAG.match(os.path.basename(path))
    return m.groupdict() if m else None


def _rows(path):
    if os.path.getsize(path) == 0:
        return None, []
    with open(path, newline='') as f:
        rd = csv.DictReader(f)
        return list(rd.fieldnames or []), list(rd)


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float('nan')


def _fail(probs, value):
    return False, value, '; '.join(probs[:6]) + (f' (+{len(probs) - 6} more)' if len(probs) > 6 else '')


def _dateobs_unix(s):
    s = str(s).strip().replace('T', ' ')
    fmt = '%Y-%m-%d %H:%M:%S.%f' if '.' in s else '%Y-%m-%d %H:%M:%S'
    return (datetime.datetime.strptime(s, fmt) - datetime.datetime(1970, 1, 1)).total_seconds()


# ---- snapshot_search ----------------------------------------------------------------------------------------------------
@content_test('snapshot_search')
def runinfo_and_tables(path, **ctx):
    """runinfo parses and names its own (band, Stokes, mode); flow_registered true (P3_SNAP_FLOW=1 production); the search
    domain floor = search_snapshot.MIN_ELEVATION_DEG; noise_<tag>.csv rows = n_frames; candidates_<tag>.csv rows =
    n_candidates, every |snr| >= thresh; events rows = n_events; a subband's admission_<sb>.csv rows = its V frame count;
    1-minute tier: 'paused' <=> no 1-minute file, else frame_time_1min = the integration-midpoint basis and the 1-minute
    candidate count matches (snapshot_frametime_content_test test D)."""
    s = _cfg()
    t = _tag(path)
    if not t or t['kind'] != 'runinfo':
        return False, None, f'not a runinfo_<band>_<S>_<mode>.json name: {os.path.basename(path)}'
    with open(path) as f:
        ri = json.load(f)
    probs = []
    d = os.path.dirname(path)
    tag = f"{t['band']}_{t['stokes']}_{t['mode']}"
    for k, want in (('subband', t['band']), ('stokes', t['stokes']), ('mode', t['mode'])):
        if ri.get(k) != want:
            probs.append(f'{k} {ri.get(k)!r} != the file name {want!r}')
    for k in ('lst', 'date', 'n_frames', 'n_admitted', 'n_candidates', 'n_events', 'thresh'):
        if k not in ri:
            probs.append(f'no {k}')
    if s.FLOW_REQUIRED and ri.get('flow_registered') is not True:
        probs.append(f"flow_registered {ri.get('flow_registered')!r} (P3_SNAP_FLOW=1 searches flow-registered frames)")
    if ri.get('min_elevation_deg') is not None and float(ri['min_elevation_deg']) != float(s.MIN_ELEVATION_DEG):
        probs.append(f"min_elevation_deg {ri['min_elevation_deg']} != {s.MIN_ELEVATION_DEG}")
    if os.path.exists(os.path.join(d, f'nightrefused_{tag}.json')):
        probs.append('a night-refusal record exists for the same search')
    nz = os.path.join(d, f'noise_{tag}.csv')
    if not os.path.exists(nz):
        probs.append('no noise table')
    else:
        _c, nrows = _rows(nz)
        if len(nrows) != int(ri.get('n_frames', -1)):
            probs.append(f"noise rows {len(nrows)} != n_frames {ri.get('n_frames')}")
    thr = float(ri.get('thresh', s.THRESH_SIGMA))
    for kind, key in (('candidates', 'n_candidates'), ('events', 'n_events')):
        p = os.path.join(d, f'{kind}_{tag}.csv')
        if not os.path.exists(p):
            probs.append(f'no {kind} table')
            continue
        cols, rows = _rows(p)
        if len(rows) != int(ri.get(key, -1)):
            probs.append(f'{kind} rows {len(rows)} != {key} {ri.get(key)}')
        if kind == 'candidates':
            low = [r for r in rows if not abs(_f(r.get('snr'))) >= thr - 1e-6]
            if low:
                probs.append(f'{len(low)} candidates with |snr| < {thr:g}')
    if t['band'] not in _GROUPS and t['stokes'] == 'V':
        ad = os.path.join(d, f"admission_{t['band']}.csv")
        if os.path.exists(ad):
            _c, arows = _rows(ad)
            if len(arows) != int(ri.get('n_frames', -1)):
                probs.append(f"admission rows {len(arows)} != V n_frames {ri.get('n_frames')}")
        else:
            probs.append('no admission table')
    one = os.path.join(d, f'candidates_{tag}_1min.csv')
    if ri.get('tier_1min') == 'paused':
        if os.path.exists(one):
            probs.append('tier_1min paused but a 1-minute candidates table exists')
    else:
        if ri.get('frame_time_1min') != s.TIER1MIN_TIME_BASIS:
            probs.append(f"frame_time_1min {ri.get('frame_time_1min')!r} != {s.TIER1MIN_TIME_BASIS!r}")
        if not os.path.exists(one):
            probs.append('no 1-minute candidates table (tier not paused)')
        else:
            _c, r1 = _rows(one)
            if 'n_candidates_1min' in ri and len(r1) != int(ri['n_candidates_1min']):
                probs.append(f"1-minute candidates {len(r1)} != n_candidates_1min {ri['n_candidates_1min']}")
    if probs:
        return _fail(probs, ri.get('n_candidates'))
    return True, ri.get('n_candidates'), (f"{tag}: {ri.get('n_frames')} frames ({ri.get('n_admitted')} admitted), "
                                          f"{ri.get('n_candidates')} candidates, tables consistent")


# ---- snapshot_1min ------------------------------------------------------------------------------------------------------
@content_test('snapshot_1min')
def blocks_by_integration(path, **ctx):
    """noise_<tag>_1min.csv: every block_ts is YYYYMMDD_HHMM00; an admitted block uses >= MIN_BLOCK_FRAMES frames (else
    refused lt3-frames); every member frame's integration midpoint (its 10-s noise-table date_obs + INTEGRATION_S / 2) lies
    in the block's UTC minute; block counts = runinfo n_blocks_1min / n_blocks_admitted_1min (test B)."""
    s = _cfg()
    t = _tag(path)
    if not t or t['kind'] != 'noise' or not t['one']:
        return False, None, f'not a noise_<band>_<S>_<mode>_1min.csv name: {os.path.basename(path)}'
    d = os.path.dirname(path)
    tag = f"{t['band']}_{t['stokes']}_{t['mode']}"
    _c, blocks = _rows(path)
    _c, ten = _rows(os.path.join(d, f'noise_{tag}.csv')) if os.path.exists(os.path.join(d, f'noise_{tag}.csv')) else (None, [])
    dob = {r['ts']: r.get('date_obs', '') for r in ten}
    probs = []
    n_adm = 0
    for b in blocks:
        bt = b.get('block_ts', '')
        if not re.match(r'^\d{8}_\d{4}00$', bt):
            probs.append(f'block_ts {bt!r}')
            continue
        mem = [m for m in (b.get('member_ts') or '').split(';') if m]
        adm = str(b.get('admitted')) == '1'
        n_adm += adm
        if adm and len(mem) < s.MIN_BLOCK_FRAMES:
            probs.append(f'{bt}: admitted with {len(mem)} frames < {s.MIN_BLOCK_FRAMES}')
        if not adm and len(mem) >= s.MIN_BLOCK_FRAMES and int(_f(b.get('flood_refused')) or 0) == 0 \
                and not b.get('refuse_reason'):
            probs.append(f'{bt}: {len(mem)} frames but not admitted and no refusal reason')
        minute = datetime.datetime.strptime(bt[:13], '%Y%m%d_%H%M')
        for m in mem:
            if m not in dob or not dob[m]:
                probs.append(f'{bt}: member {m} not in the 10-s noise table (or no date_obs)')
                continue
            mid = datetime.datetime(1970, 1, 1) + datetime.timedelta(seconds=_dateobs_unix(dob[m]) + s.INTEGRATION_S / 2.0)
            if mid.replace(second=0, microsecond=0) != minute:
                probs.append(f'{bt}: member {m} integration midpoint {mid:%H:%M:%S} outside the minute')
    rp = os.path.join(d, f'runinfo_{tag}.json')
    if os.path.exists(rp):
        with open(rp) as f:
            ri = json.load(f)
        if 'n_blocks_1min' in ri and len(blocks) != int(ri['n_blocks_1min']):
            probs.append(f"{len(blocks)} blocks != runinfo n_blocks_1min {ri['n_blocks_1min']}")
        if 'n_blocks_admitted_1min' in ri and n_adm != int(ri['n_blocks_admitted_1min']):
            probs.append(f"{n_adm} admitted blocks != runinfo n_blocks_admitted_1min {ri['n_blocks_admitted_1min']}")
    if probs:
        return _fail(probs, len(blocks))
    return True, len(blocks), f'{len(blocks)} blocks ({n_adm} admitted), every member in its integration minute'


# ---- snapshot_wideband --------------------------------------------------------------------------------------------------
@content_test('snapshot_wideband')
def members_and_outcome(path, **ctx):
    """wideband_outcome_<S>.json: Red/Green/Blue each searched (runinfo present), night-refused (record present), excluded
    (REFUSED marker present), no-members or too-few-frames - never not-run/missing (v9:246-264); an error is admitted only
    when it is the recorded night-rule refusal; wideband_members_<S>.csv: every member's DATE-OBS within PAIR_TOL_S of its
    wideband frame's (one integration, test A)."""
    s = _cfg()
    m = re.match(r'^wideband_outcome_([IV])\.json$', os.path.basename(path))
    if not m:
        return False, None, f'not a wideband_outcome_<S>.json name: {os.path.basename(path)}'
    st = m.group(1)
    d = os.path.dirname(path)
    with open(path) as f:
        rec = json.load(f)
    probs = []
    if rec.get('stokes') != st:
        probs.append(f"stokes {rec.get('stokes')!r} != {st}")
    groups = rec.get('groups') or {}
    for g in ('Red', 'Green', 'Blue'):
        v = groups.get(g, 'missing')
        if v == 'searched' and not os.path.exists(os.path.join(d, f'runinfo_{g}_{st}_{s.MODE}.json')):
            probs.append(f'{g}: searched but no runinfo')
        elif v == 'night-refused' and not os.path.exists(os.path.join(d, f'nightrefused_{g}_{st}_{s.MODE}.json')):
            probs.append(f'{g}: night-refused but no record')
        elif v == 'excluded' and not os.path.exists(os.path.join(d, f'wideband_{g}_{st}_REFUSED.txt')):
            probs.append(f'{g}: excluded but no REFUSED marker')
        elif v not in ('searched', 'night-refused', 'excluded', 'no-members', 'too-few-frames'):
            probs.append(f'{g}: {v}')
    err = rec.get('error')
    if err and not ('night rule' in err and all(groups.get(g) != 'not-run' for g in ('Red', 'Green', 'Blue'))):
        if not (str(err).startswith('SystemExit') and all(groups.get(g) == 'excluded' for g in ('Red', 'Green', 'Blue'))):
            probs.append(f'the group search ended in an error that is not a recorded refusal: {str(err)[:120]}')
    mp = os.path.join(d, f'wideband_members_{st}.csv')
    n_mem = 0
    if any(groups.get(g) == 'searched' for g in ('Red', 'Green', 'Blue')):
        if not os.path.exists(mp):
            probs.append('a group was searched but there is no wideband_members table')
        else:
            _c, rows = _rows(mp)
            n_mem = len(rows)
            far = 0
            for r in rows:
                try:
                    if abs(_dateobs_unix(r['member_date_obs']) - _dateobs_unix(r['date_obs'])) > s.PAIR_TOL_S:
                        far += 1
                except (KeyError, ValueError):
                    far += 1
            if far:
                probs.append(f'{far} of {len(rows)} members not of their wideband frame\'s integration (> {s.PAIR_TOL_S} s)')
    if probs:
        return _fail(probs, n_mem)
    return True, n_mem, ', '.join(f'{g} {groups.get(g)}' for g in ('Red', 'Green', 'Blue')) + f'; {n_mem} members paired'


# ---- snapshot_model -----------------------------------------------------------------------------------------------------
@content_test('snapshot_model')
def grid(path, **ctx):
    """model_<tag>.fits: a 2-D image with a celestial WCS, finite fraction >= 0.5, and the runinfo of the same search
    beside it (save_map, phase3_snapshot_search.py)."""
    import numpy as np
    from astropy.io import fits
    from astropy.wcs import WCS
    t = _tag(path)
    if not t or t['kind'] != 'model':
        return False, None, f'not a model_<band>_<S>_<mode>.fits name: {os.path.basename(path)}'
    with fits.open(path, memmap=True) as h:
        d = np.squeeze(h[0].data)
        w = WCS(h[0].header).celestial
        frac = float(np.isfinite(d).mean()) if d is not None and d.size else 0.0
        shape = None if d is None else d.shape
    probs = []
    if shape is None or len(shape) != 2:
        probs.append(f'data shape {shape}')
    if not w.has_celestial:
        probs.append('no celestial WCS')
    if frac < 0.5:
        probs.append(f'finite fraction {frac:.3f} < 0.5')
    if not os.path.exists(os.path.join(os.path.dirname(path), f"runinfo_{t['band']}_{t['stokes']}_{t['mode']}.json")):
        probs.append('no runinfo of the same search')
    if probs:
        return _fail(probs, round(frac, 4))
    return True, round(frac, 4), f'{shape[1]}x{shape[0]} px, finite {frac:.3f}'


# ---- night_refusal ------------------------------------------------------------------------------------------------------
@content_test('night_refusal')
def record(path, **ctx):
    """nightrefused_<tag>.json (night_refusal_record, clv15 769778b): status refused-night-rule; names its own (band, Stokes,
    mode); rule, solar audit, verdict counts summing to n_frames, Sun elevation min <= max, first_ts <= last_ts; and no
    runinfo for the same search (a refusal is an outcome in place of the runinfo, never beside it)."""
    s = _cfg()
    t = _tag(path)
    if not t or t['kind'] != 'nightrefused':
        return False, None, f'not a nightrefused_<band>_<S>_<mode>.json name: {os.path.basename(path)}'
    with open(path) as f:
        r = json.load(f)
    probs = []
    if r.get('status') != s.NIGHT_RULE_STATUS:
        probs.append(f"status {r.get('status')!r}")
    for k, want in (('subband', t['band']), ('stokes', t['stokes']), ('mode', t['mode'])):
        if r.get(k) != want:
            probs.append(f'{k} {r.get(k)!r} != the file name {want!r}')
    for k in ('rule', 'solar_audit', 'first_ts', 'last_ts', 'lst', 'date'):
        if not r.get(k):
            probs.append(f'no {k}')
    counts = r.get('verdict_counts') or {}
    n = int(r.get('n_frames') or 0)
    if n <= 0 or sum(int(v) for v in counts.values()) != n:
        probs.append(f'verdict counts {counts} do not sum to n_frames {n}')
    if '' in counts:
        probs.append('a frame the night rule passed is counted (the record is written only when every frame is refused)')
    if not (_f(r.get('sun_elevation_deg_min')) <= _f(r.get('sun_elevation_deg_max'))):
        probs.append('sun elevation min > max')
    if str(r.get('first_ts', '')) > str(r.get('last_ts', '')):
        probs.append('first_ts > last_ts')
    if os.path.exists(os.path.join(os.path.dirname(path), f"runinfo_{t['band']}_{t['stokes']}_{t['mode']}.json")):
        probs.append('a runinfo exists for the same search')
    if probs:
        return _fail(probs, n)
    return True, n, f"{t['band']} {t['stokes']}: {n} frames refused by the night rule {counts}"


# ---- snapshot_table -----------------------------------------------------------------------------------------------------
@content_test('snapshot_table')
def parses(path, **ctx):
    """Every other shipped CSV: a header and rectangular rows; a 0-byte candidates/events table is accepted only where its
    runinfo counts 0 such rows (production writes no header for an empty table, S-M1); admission tables carry ts and
    admitted; wideband member tables carry the integration columns."""
    name = os.path.basename(path)
    d = os.path.dirname(path)
    if os.path.getsize(path) == 0:
        t = _tag(path)
        if t and t['kind'] in ('candidates', 'events'):
            rp = os.path.join(d, f"runinfo_{t['band']}_{t['stokes']}_{t['mode']}.json")
            if os.path.exists(rp):
                with open(rp) as f:
                    ri = json.load(f)
                key = f"n_{t['kind']}" + ('_1min' if t['one'] else '')
                if int(ri.get(key, -1)) == 0:
                    return True, 0, f'empty table, runinfo {key} = 0'
                return False, 0, f'0-byte table but runinfo {key} = {ri.get(key)}'
        return False, 0, '0-byte table'
    with open(path, newline='') as f:
        rd = csv.reader(f)
        hdr = next(rd, None)
        bad = n = 0
        for row in rd:
            n += 1
            if len(row) != len(hdr):
                bad += 1
    probs = []
    if not hdr:
        probs.append('no header')
    if bad:
        probs.append(f'{bad} rows whose length is not the header\'s')
    need = {'admission_': ('ts', 'admitted'), 'wideband_members_': ('group', 'ts', 'date_obs', 'member_ts', 'member_date_obs')}
    for pre, cols in need.items():
        if name.startswith(pre):
            miss = [c for c in cols if c not in (hdr or [])]
            if miss:
                probs.append(f'missing columns {miss}')
    if probs:
        return _fail(probs, n)
    return True, n, f'{n} rows, {len(hdr)} columns'


# ---- injection_book -----------------------------------------------------------------------------------------------------
@content_test('injection_book')
def complete(path, **ctx):
    """injected_<sb>_<S>.csv (phase3_snapshot_search.run_search --inject): columns inj_id, px, py, t0, t1; inj_id unique;
    px/py finite and inside the frame grid of the injected search's model (when shipped beside it); t0 <= t1."""
    m = re.match(r'^injected_(\d+MHz)_([IV])\.csv$', os.path.basename(path))
    if not m:
        return False, None, f'not an injected_<sb>_<S>.csv name: {os.path.basename(path)}'
    cols, rows = _rows(path)
    probs = []
    if cols is not None and [c for c in ('inj_id', 'px', 'py', 't0', 't1') if c not in cols]:
        probs.append(f'columns {cols}')
    ids = [r.get('inj_id') for r in rows]
    if len(set(ids)) != len(ids):
        probs.append('duplicate inj_id')
    shape = None
    mp = os.path.join(os.path.dirname(path), f'model_{m.group(1)}_{m.group(2)}_median.fits')
    if os.path.exists(mp):
        from astropy.io import fits
        h = fits.getheader(mp)
        shape = (int(h['NAXIS2']), int(h['NAXIS1']))
    for r in rows:
        px, py = _f(r.get('px')), _f(r.get('py'))
        if not (math.isfinite(px) and math.isfinite(py)):
            probs.append(f"{r.get('inj_id')}: px/py not finite")
        elif shape and not (0 <= px < shape[1] and 0 <= py < shape[0]):
            probs.append(f"{r.get('inj_id')}: ({px:.1f}, {py:.1f}) outside the {shape[1]}x{shape[0]} grid")
        try:
            if int(r.get('t0')) > int(r.get('t1')):
                probs.append(f"{r.get('inj_id')}: t0 > t1")
        except (TypeError, ValueError):
            pass
    if probs:
        return _fail(probs, len(rows))
    return True, len(rows), f'{len(rows)} injections' + (f' inside the {shape[1]}x{shape[0]} grid' if shape else '')
