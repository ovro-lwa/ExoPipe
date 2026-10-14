"""The conformance table step replayed on a detection list - the bright-parent diff of `exopipe3 calibrate` (installed vs
candidate tables, on the served list).

Ported (logic only; the gate itself is the shared function parent_curve_table.removals, called IN PROCESS, and the gate
inputs are gate_inputs.parents / deep_target_beam / target_beam / phase_centre, called IN PLACE):
  - deep_tables_elev_20261006.py `_cuts` (:96-113), `dryrun` (:116-183): grouping by (LST, band, Stokes, night), rms per
    (LST, band, Stokes) = nanmedian |peak_jy / global_snr| over the list, mid-hour elevation at the legacy latitude,
    served identities (survivors -> members) and judgements, changed rows, tallies by Stokes / change / elevation band,
    in-search table removals of the published gate ledgers the candidate gate would keep;
  - p3s2_conform.py:254-300 for the 10-minute branch (window mid-time elevation from the key <night>T<kk>, each detection's
    frame beam gate_inputs.target_beam) and the multiples table class (<Stokes>-multi, TABLE_MODE multi).
Nothing here writes: the caller writes the rows and the summary.
"""
import collections
import json
import os

import numpy as np

ELEV_LABEL_BANDS = ((20, 25), (25, 35), (35, 50), (50, 90))   # tally bands of the 2026-10-06 dry run (:139-140)


def _f(x):
    return float(x) if x not in ('', None) else float('nan')


def elev_mid_hour(lst, ra, dec, lat_deg):
    """Elevation at the epoch's mid-hour (LST hour + 0.5 h): deep_tables_elev_20261006._elev (:90-93)."""
    lstdeg = 15.0 * (int(lst[:-1]) + 0.5)
    lat = np.radians(lat_deg)
    hr = np.radians(lstdeg - np.asarray(ra, float))
    dr = np.radians(np.asarray(dec, float))
    return np.degrees(np.arcsin(np.sin(dr) * np.sin(lat) + np.cos(dr) * np.cos(lat) * np.cos(hr)))


def elev_window(lst, key, ra, dec, lat_deg):
    """Elevation at a 10-minute window's mid-time (LST hour + (k + 0.5) x 10 min, key <night>T<kk>): p3s2_conform.py:277-279."""
    lstdeg = 15.0 * (int(lst.rstrip('h')) + (int(str(key).split('T')[1]) + 0.5) / 6.0)
    lat = np.radians(lat_deg)
    hr = np.radians(lstdeg - np.asarray(ra, float))
    dr = np.radians(np.asarray(dec, float))
    return np.degrees(np.arcsin(np.sin(dr) * np.sin(lat) + np.cos(dr) * np.cos(lat) * np.cos(hr)))


def rms_by_cell(rows):
    """{(lst, band, stokes): nanmedian |peak_jy / global_snr|} (deep_tables_elev_20261006.py:121-125)."""
    acc = collections.defaultdict(list)
    for r in rows:
        try:
            acc[(r['lst'], r['band'], r['stokes'])].append(abs(_f(r['peak_jy'])) / abs(_f(r['global_snr'])))
        except Exception:     # noqa: BLE001 - production skips unparseable rows (:123-124)
            pass
    return {k: float(np.nanmedian(v)) for k, v in acc.items()}


def cuts(rows, rms, tables, P, G, cadence='deep', elev_on=True, table_class_suffix='', prefix=None, lat_deg=37.2398):
    """{row index: (removed, note)} of the conformance table step on ``rows`` with the table set ``tables`` (the merged
    {key: record} dict parent_curve_table.tables() would return). P = parent_curve_table, G = gate_inputs (in place).
    cadence 'deep': target beam of the searched deep epoch frame, elevation at mid-hour; '10min': each detection's frame
    beam, elevation at its window's mid-time. elev_on False = no elevation passed (all-elevation tables only;
    P3_TABLE_ELEV=0, the 'installed' side of the 2026-10-06 dry run). table_class_suffix '-multi' = the multiples tables."""
    from exopipe.phase3.lib.calibrate.common import legacy_env
    env = {'P3_CADENCE': cadence, 'P3_TABLE_PREFIX': prefix, 'P3_TABLE_ELEV': '1' if elev_on else '0'}
    saved = P._TABLES
    P._TABLES = dict(tables)
    try:
        with legacy_env(env):
            grp = collections.defaultdict(list)
            for i, r in enumerate(rows):
                grp[(r['lst'], r['band'], r['stokes'], r['date'])].append(i)
            out = {}
            for (l, b, s, d), idx in grp.items():
                ra = np.array([_f(rows[i]['ra']) for i in idx])
                de = np.array([_f(rows[i]['dec']) for i in idx])
                if cadence == '10min':
                    el = elev_window(l, d, ra, de, lat_deg) if (elev_on and 'T' in str(d)) else None
                    beam = G.target_beam(l, b, d, s)
                else:
                    el = elev_mid_hour(l, ra, de, lat_deg) if elev_on else None
                    beam = G.deep_target_beam(l, b, str(d)[:10], s)
                c, n = P.removals(s + table_class_suffix, b, G.parents(l, b), rms.get((l, b, s), 0.0), beam, ra, de,
                                  elev=el, phase_centre=G.phase_centre(l, b))
                for i, x, nn in zip(idx, c, n):
                    out[i] = (bool(x), nn)
            return out
    finally:
        P._TABLES = saved


def served_map(run_dir, tiers, stokes=('I', 'V')):
    """{cand_id: (Stokes, served identity)} from <run>/source_catalogs_<S><tier>/*_sources_survivors.csv + members
    (deep_tables_elev_20261006.py:128-132; every served tier, spec Q7 default)."""
    import csv
    import glob
    served = {}
    for S in stokes:
        for t in tiers:
            for p in sorted(glob.glob(os.path.join(run_dir, f'source_catalogs_{S}{t}', '*_sources_survivors.csv'))):
                with open(p, newline='') as f:
                    ids = {r['source_id']: (r.get('identity_id') or r['source_id']) for r in csv.DictReader(f)}
                mp = p.replace('_sources_survivors.csv', '_source_members.csv')
                if not os.path.exists(mp):
                    continue
                with open(mp, newline='') as f:
                    for r in csv.DictReader(f):
                        if r.get('source_id') in ids:
                            served[r['cand_id']] = (S, ids[r['source_id']])
    return served


def judgements(run_dir, pattern, stokes=('I', 'V')):
    """{(Stokes, source_id): judge_confidence or 'judged'} of the classify actions (deep_tables_elev_20261006.py:133-138)."""
    judg = {}
    for S in stokes:
        jp = pattern.format(run=run_dir, S=S)
        if not os.path.exists(jp):
            continue
        for ln in open(jp):
            try:
                j = json.loads(ln)
            except ValueError:
                continue
            if j.get('action') == 'classify':
                judg[(S, j['source_id'])] = j.get('judge_confidence', '') or 'judged'
    return judg


def eband(e):
    return next((f'{lo}-{hi}' for lo, hi in ELEV_LABEL_BANDS if lo <= e < hi), '<20')


def compare(rows, old, new, served, judg, cadence='deep', lat_deg=37.2398):
    """(changed rows, tally Counter[(Stokes, change, elevation band)]) - deep_tables_elev_20261006.py:141-151."""
    out, tally = [], collections.Counter()
    for i, r in enumerate(rows):
        o, n = old.get(i, (False, '')), new.get(i, (False, ''))
        if o[0] == n[0]:
            continue
        if cadence == '10min' and 'T' in str(r['date']):
            e = float(elev_window(r['lst'], r['date'], float(r['ra']), float(r['dec']), lat_deg))
        else:
            e = float(elev_mid_hour(r['lst'], float(r['ra']), float(r['dec']), lat_deg))
        ch = 'newly removed' if n[0] else 'newly kept'
        sv = served.get(r['cand_id'])
        out.append(dict(cand_id=r['cand_id'], stokes=r['stokes'], lst=r['lst'], band=r['band'], date=r['date'], ra=r['ra'],
                        dec=r['dec'], snr=r.get('snr', ''), elev_deg=f'{e:.1f}', elev_band=eband(e), change=ch,
                        served_identity=(sv[1] if sv else ''), judged=(judg.get(sv, '') if sv else ''),
                        installed_note=o[1][:120], band_note=n[1][:120]))
        tally[(r['stokes'], ch, eband(e))] += 1
    return out, tally


def judged_labels(run_dir, pattern, stokes=('I', 'V')):
    """{(Stokes, source_id): label or judge_confidence or 'judged'} of the classify actions - the label served_change_20261006.py
    reports (:47-52; the 2026-10-06 dry run's `judgements` keeps judge_confidence only)."""
    out = {}
    for S in stokes:
        jp = pattern.format(run=run_dir, S=S)
        if not os.path.exists(jp):
            continue
        for ln in open(jp):
            try:
                j = json.loads(ln)
            except ValueError:
                continue
            if j.get('action') == 'classify':
                out[(S, j['source_id'])] = j.get('label') or j.get('judge_confidence') or 'judged'
    return out


def source_fates(rows, old, new, served, labels):
    """Per served source (served_change_20261006.py:53-75, the band-gate part): its member detections in ``rows`` that the
    candidate removes where the installed gate kept them LEAVE it; a source is 'gone' when every member leaves, 'reduced'
    when some do, else 'unchanged'. Returns (status Counter[(Stokes, status)], judged Counter[(Stokes, status)],
    leaving Counter[Stokes], rows of the judged sources)."""
    members = collections.defaultdict(list)
    for i, r in enumerate(rows):
        sv = served.get(r['cand_id'])
        if sv:
            members[sv].append(i)
    src, jud, leave, out = collections.Counter(), collections.Counter(), collections.Counter(), []
    for (S, sid), idx in sorted(members.items()):
        n_leave = sum(1 for i in idx if new.get(i, (False, ''))[0] and not old.get(i, (False, ''))[0])
        status = 'gone' if n_leave == len(idx) else ('reduced' if n_leave else 'unchanged')
        src[(S, status)] += 1
        leave[S] += n_leave
        if (S, sid) in labels:
            jud[(S, status)] += 1
            out.append(dict(stokes=S, source=sid, label=labels[(S, sid)], status=status, detections=len(idx),
                            band_removed=n_leave))
    return src, jud, leave, out


def table_notes(rows, cut):
    """The TABLE-NOTES content test of the 10-minute chain (sublst_catalogs.sh:187-201) on the notes of the rows a gate
    removes: (n bright-parent removals, n without an elevation band and the image-pixel metric, first bad note)."""
    n = bad = 0
    ex = ''
    for i, (c, note) in cut.items():
        t = note or ''
        if not c or 'table:' not in t or 'parent S=' not in t:
            continue
        n += 1
        if '[el' not in t or '(image-pixel)' not in t:
            bad += 1
            ex = ex or t[:200]
    return n, bad, ex
