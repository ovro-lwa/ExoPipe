"""Every removal rule of a deep detection, once - B-02 (a), R-GREGG-20261013-REFACTOR-SHEET-DADMIT-B01-B02-B03-B05-B30-B32;
R-GREGG-20261014-REFACTOR-DECISIONS-LIST-RECOMMENDATIONS items 5-6; plan/wave2/B02_DESIGN.md.

Production applies these rules a second time at publish (exopipe_ops/p3s2_conform.py, the conformance) to the merged rows,
from recorded columns, frame cards and tables. Each rule is a function here, so that the search applies it once and the
conformance check evaluates the same function again. conform() composes them in production's order with production's
inputs: under production's conventions it is p3s2_conform.py (deep tier, main chain) - the same rows removed with the
same words, the same conformant rows, the same printed lines (tests/test_removal_rules.py, on Run_P3S5).

Conventions (catalog_conform): DEEP_ELEV_EPOCH 'mid-hour' (production: LST + 0.5 h) or 'frame-mid' (the frame's own
mid-time, B02_DESIGN question 2). The inputs a rule reads are passed in; nothing here writes outside the caller's files.
"""
import collections
import csv
import glob
import math
import os
import re

import numpy as np

from exopipe.phase3.config import CFG

ELEV_EPOCHS = ('mid-hour', 'frame-mid')
SEARCH_RULES = ('once-in-search', 'production')   # search_detect.REMOVAL_RULES
RULES_ENV = 'EXOPIPE3_REMOVAL_RULES'              # how the search step hands it to the engine child (deep_step.engine_env)
RULES_VERSION = 'b02-3a'   # what a 'once-in-search' attempt applies: the daytime gate per frame, the Jupiter-track veto
AUDIT_ENV = 'EXOPIPE3_SOLAR_AUDIT'                # the solar audit the search's daytime gate reads (search_tables.SOLAR_AUDIT)
RULES_RECORD = 'search_rules.json'                # in each attempt directory: the rules it was searched under (deep_step)


def _cc():
    return CFG.catalog_conform


def search_rules(value=None):
    """search_detect.REMOVAL_RULES - ``value`` checked, else as the search step passed it to the engine child
    (EXOPIPE3_REMOVAL_RULES), else (a process no step started) the config. An unknown or empty value REFUSES - a typo never
    selects either (deep_search._eph_mode's reading); a RuntimeError, so the engine child maps it to a refusal."""
    if value is None:
        value = os.environ.get(RULES_ENV)
        if value is None:
            value = CFG.search_detect.REMOVAL_RULES
    if value not in SEARCH_RULES:
        raise RuntimeError(f'search_detect.REMOVAL_RULES = {value!r} is not one of {SEARCH_RULES} - REFUSING')
    return value


def rules_record(rules):
    """The record of a search under ``rules``: {rules, version - RULES_VERSION under 'once-in-search', else None}."""
    rules = search_rules(rules)
    return dict(rules=rules, version=RULES_VERSION if rules == 'once-in-search' else None)


def write_attempt_rules(attempt_dir, rules):
    """Record in an attempt directory the rules it is searched under (the search step, before the engine child starts;
    published with the attempt, deep_step.stage_publish); returns the record (rules_record)."""
    import json
    rec = rules_record(rules)
    with open(os.path.join(attempt_dir, RULES_RECORD), 'w') as fh:
        json.dump(rec, fh)
    return rec


def attempt_rules(attempt_dir):
    """The record ({rules, version}) an attempt directory was searched under: its RULES_RECORD, else {'rules': 'production',
    'version': None} - every attempt without one was searched without the rules (made by a fleet, or by the step before
    B-02 step 3a). Two commits applied the rules without a published record - be86762 (none written) and 872e305 (written
    in the staged attempt only) - and neither made an attempt outside the tests. An unreadable or unknown record REFUSES
    (RuntimeError): its attempt's rules cannot be told."""
    import json
    p = os.path.join(attempt_dir, RULES_RECORD)
    if not os.path.exists(p):
        return dict(rules='production', version=None)
    try:
        rec = json.load(open(p))
    except (OSError, ValueError) as e:
        rec = f'{type(e).__name__}: {e}'
    ok = (isinstance(rec, dict) and set(rec) == {'rules', 'version'} and rec['rules'] in SEARCH_RULES   # 'production' has no
          and (rec['version'] is None if rec['rules'] == 'production' else isinstance(rec['version'], str)))   # version
    if not ok:
        raise RuntimeError(f'{p}: not a search rules record ({str(rec)[:120]}) - REFUSING (the attempt\'s rules cannot be told)')
    return rec


def f(x):
    """p3s2_conform.f: float, or nan."""
    try:
        return float(x)
    except Exception:
        return np.nan


def band_class(band):
    """p3s2_conform.bclass: a subband's wideband class (Red, Green, Blue); any other name is returned as given."""
    b = band.replace('MHz', '')
    for cls, subs in CFG.catalog.BAND_CLASS.items():
        if f'{b}MHz' in subs:
            return cls
    return band


# ---------------------------------------------------------------------------------------------------------------------
# elevation floor (p3s2_conform.py:_deep_mid_el, :500-504; Gregg 2026-10-06 "3. (c)")
# ---------------------------------------------------------------------------------------------------------------------

def _elevation(lstdeg, ra, dec):
    """Elevation (deg) of (RA, Dec) at local sidereal angle `lstdeg` (scalars or arrays; arrays give an array)."""
    hr = np.radians(lstdeg - ra); dr = np.radians(dec); lat = np.radians(CFG.catalog.SITE_LAT_DEG['conform_mid_el'])
    return np.degrees(np.arcsin(np.sin(dr) * np.sin(lat) + np.cos(dr) * np.cos(lat) * np.cos(hr)))


def mid_hour_elevation(lst, ra, dec):
    """Elevation (deg) at the LST hour's mid-point (LST + 0.5 h) - production's convention, the deep tables' too."""
    return float(_elevation(15.0 * (int(lst.rstrip('h')) + 0.5), ra, dec))


def mid_hour_elevations(lst, ra, dec):
    """mid_hour_elevation over arrays (the elevation that chooses a detection's elevation-band table)."""
    return _elevation(15.0 * (int(lst.rstrip('h')) + 0.5), np.asarray(ra), np.asarray(dec))


def frame_mid_lst_deg(date_obs, date_end):
    """Apparent local sidereal time (deg) at OVRO at the frame's mid-time."""
    from astropy.time import Time
    import astropy.units as u
    t0, t1 = Time(date_obs), Time(date_end)
    tm = t0 + (t1 - t0) * 0.5
    return float(tm.sidereal_time('apparent', longitude=CFG.catalog.SITE_LON_DEG * u.deg).deg)


def deep_elevation(lst, ra, dec, epoch=None, frame_time=None):
    """(elevation deg, the words naming the time) under catalog_conform.DEEP_ELEV_EPOCH (or `epoch`). 'frame-mid' needs the
    frame's (DATE-OBS, DATE-END); without them it refuses - an elevation it could not judge is never a pass."""
    epoch = epoch or _cc().DEEP_ELEV_EPOCH
    if epoch == 'mid-hour':
        return mid_hour_elevation(lst, ra, dec), 'at mid-hour'
    if epoch == 'frame-mid':
        if not frame_time:
            raise ValueError(f'{lst}: the frame-mid elevation needs the frame\'s DATE-OBS/DATE-END')
        return float(_elevation(frame_mid_lst_deg(*frame_time), ra, dec)), 'at the frame mid-time'
    raise ValueError(f'catalog_conform.DEEP_ELEV_EPOCH {epoch!r}: one of {ELEV_EPOCHS}')


def elevation_reason(el, words, floor=None):
    floor = _cc().DEEP_ELEV_FLOOR_DEG if floor is None else floor
    if el < floor:
        return f'elevation {el:.1f} deg {words} < the deep floor {floor:g} deg (Gregg 2026-10-06)'
    return ''


# ---------------------------------------------------------------------------------------------------------------------
# daytime gate (p3s2_conform.py:309-358, :505-511; snapshot chat ruling 2026-09-26, Gregg 2026-09-27)
# ---------------------------------------------------------------------------------------------------------------------

def _ovro():
    from astropy.coordinates import EarthLocation
    import astropy.units as u
    return EarthLocation(lat=CFG.catalog.SITE_LAT_DEG['conform'] * u.deg, lon=CFG.catalog.SITE_LON_DEG * u.deg,
                         height=CFG.catalog.SITE_HEIGHT_M['conform'] * u.m)


def solar_quarantine(path):
    """{(lst, date): {subband MHz numbers}} the solar contamination audit QUARANTINEs; refuses without the audit."""
    if not os.path.exists(path):
        raise SystemExit(f'conform: daytime gate has no solar audit {path} - REFUSING')
    q = collections.defaultdict(set)
    for r in csv.DictReader(open(path)):
        if r['verdict'] == 'QUARANTINE':
            q[(r['lst'], r['date'])].add(r['subband'].replace('MHz', ''))
    return q


def quarantined(quar, lst, band, date):
    """A subband frame is flagged when its own subband is quarantined, a wideband frame when any constituent is (the
    is_excluded rule)."""
    q = quar.get((lst, date), set()); b = band.replace('MHz', '')
    if f'{b}MHz' in {s for subs in CFG.catalog.BAND_CLASS.values() for s in subs}:
        return b in q
    return any(band_class(s) == b or b == 'Full' for s in q)


def daytime_verdicts(frame_times, quar):
    """{frame key (stokes, lst, band, date): '' admitted, else the refusal} for every key of `frame_times`
    ({key: (DATE-OBS, DATE-END)}). The Sun's elevation at the frame's start, 1/4, mid, 3/4 and end: above the horizon at
    any sample refuses in every band; twilight (to -12 deg rising, -18 deg setting) refuses in a quarantined band/night."""
    from astropy.coordinates import get_sun, AltAz
    from astropy.time import Time
    c = _cc()
    out = {}
    fk = sorted(frame_times)
    if not fk:
        return out
    ovro = _ovro()
    t0 = Time([frame_times[k][0] for k in fk]); t1 = Time([frame_times[k][1] for k in fk])
    fr = np.array(c.DAYTIME_SAMPLES)
    ts = t0[:, None] + (t1 - t0)[:, None] * fr[None, :]
    el = get_sun(ts).transform_to(AltAz(obstime=ts, location=ovro)).alt.deg
    for k, e in zip(fk, el):
        thr = c.TWILIGHT_RISING_DEG if e[-1] > e[0] else c.TWILIGHT_SETTING_DEG
        if e.max() > 0:
            out[k] = f'daytime: Sun {e.min():.1f} to {e.max():.1f} deg over the frame (solar gate 2026-09-26/27)'
        elif e.max() > thr and quarantined(quar, k[1], k[2], k[3]):
            out[k] = f'twilight in a solar-audit QUARANTINE band: Sun to {e.max():.1f} deg (solar gate 2026-09-26/27)'
        else:
            out[k] = ''
    return out


DAYTIME_UNREADABLE = 'daytime gate: frame DATE-OBS/DATE-END unreadable - cannot be judged, refused'


def daytime_candidate_frames(lst, scope, ebase, dates, resid, skip):
    """{(Stokes, lst, scope, date): Stage-1 product path} - the frames the search's daytime gate judges: each Stokes'
    searched nights (``dates`` with a residual of that Stokes in ``resid[Stokes]`` - not the other nights an admitted
    night's stack needs) that no other gate refused (``skip[Stokes]`` - a frame refused there is not searched anyway)."""
    return {(st, lst, scope, d): f'{ebase}/subtracted/{d}_{st}_sub_{scope}.fits'
            for st in ('I', 'V') for d in dates if d in resid[st] and d not in skip[st]}


def product_times(path):
    """(DATE-OBS, DATE-END) of an epoch's Stage-1 product - the time block its archived search frame carries
    (deep_search._archive_frame_header: DATE-END, else DATE-OBS + DURATION) - or None when it cannot be read."""
    from astropy.io import fits
    from astropy.time import Time, TimeDelta
    try:
        h = fits.getheader(path)
        t0, t1 = h.get('DATE-OBS'), h.get('DATE-END')
        if t1 is None and t0 is not None and h.get('DURATION'):
            t1 = (Time(t0) + TimeDelta(float(h['DURATION']), format='sec')).isot
        if t0 is None or t1 is None:
            return None
        Time(str(t0)); Time(str(t1))
        return str(t0), str(t1)
    except Exception:
        return None


def daytime_frames(frames, quar=None):
    """{(Stokes, date): the refusal} of the frames ({(Stokes, lst, band, date): Stage-1 product path}) the daytime gate refuses:
    the Sun up at any of the frame's five samples, twilight in a quarantined band, or times that cannot be read - the
    conformance's rule (daytime_verdicts), applied by the search to each frame before it is searched (B-02 step 3a). The
    quarantine bands from the solar audit the search step names (EXOPIPE3_SOLAR_AUDIT = search_tables.SOLAR_AUDIT, pinned by
    md5 at the step's gate G2), else - a process no step started - the config's."""
    if quar is None:
        audit = os.environ.get(AUDIT_ENV)
        if audit is None:
            audit = CFG.search_tables.SOLAR_AUDIT
        try:
            quar = solar_quarantine(audit)
        except SystemExit as e:                   # the conformance's words; in the search a refusal of the cell
            raise RuntimeError(str(e))
    times, out = {}, {}
    for k, p in frames.items():
        t = product_times(p)
        if t is None:
            out[(k[0], k[3])] = DAYTIME_UNREADABLE
        else:
            times[k] = t
    for k, why in daytime_verdicts(times, quar).items():
        if why:
            out[(k[0], k[3])] = why
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Jupiter track (p3s2_conform.py:279-307, :512-518; Gregg 2026-09-18)
# ---------------------------------------------------------------------------------------------------------------------

def jupiter_track(lst, weights_path, nights=None, skipped=None):
    """[(RA, Dec, night)] of Jupiter at the cell's mid-hour on every night its master co-adds (deep: in_master 1;
    'all weighted': every row of weights.csv) when Jupiter is above JUPITER_TRACK_MIN_ALT_DEG then. Production's conformance
    leaves out, silently, a night whose position fails and every night when weights.csv is missing; a list given as
    ``skipped`` receives them (the search names them: B-30 site fallbacks.SEARCH_JUPITER_TRACK_FALLBACK)."""
    from astropy.coordinates import get_body, SkyCoord, AltAz
    from astropy.time import Time
    import astropy.units as u
    c = _cc()
    nights = nights or c.JUPITER_TRACK_NIGHTS['deep']
    pts = []
    if not (weights_path and os.path.exists(weights_path)):
        if skipped is not None:
            skipped.append(f'every night (no {weights_path})')
        return pts
    ovro = _ovro()
    lh = int(lst.rstrip('h'))
    for r_ in csv.DictReader(open(weights_path)):
        if nights == 'in_master' and r_.get('in_master') != '1':
            continue
        try:
            t = Time(f"{r_['date']}T00:00:00") + np.arange(0, 24, c.JUPITER_TRACK_STEP_H) * u.hour
            lst_t = t.sidereal_time('apparent', longitude=ovro.lon).hour
            i = int(np.argmin(np.abs(((lst_t - (lh + 0.5) + 12) % 24) - 12))); tm = t[i]
            jb = get_body('jupiter', tm, ovro); j = SkyCoord(jb.ra, jb.dec)
            if j.transform_to(AltAz(obstime=tm, location=ovro)).alt.deg < c.JUPITER_TRACK_MIN_ALT_DEG:
                continue
            pts.append((float(j.ra.deg), float(j.dec.deg), r_['date']))
        except Exception as e:
            if skipped is not None:
                skipped.append(f"{r_.get('date')} ({type(e).__name__}: {e})")
            continue
    return pts


def jupiter_track_radius(band):
    return _cc().JUPITER_TRACK_RADIUS_DEG.get(band_class(band), 0.5)


def jupiter_track_hit(ra, dec, band, pts):
    """The first co-added night whose Jupiter position lies within the band's radius of (ra, dec) (flat-sky separation), or
    None."""
    jr = jupiter_track_radius(band)
    return next((d_ for (jra, jdec, d_) in pts
                 if np.hypot(((jra - ra + 180.0) % 360.0 - 180.0) * np.cos(np.radians(dec)), jdec - dec) < jr), None)


def jupiter_track_reason(ra, dec, band, pts):
    """jupiter_track_hit as the conformance's words ('' when none)."""
    hit = jupiter_track_hit(ra, dec, band, pts)
    if hit:
        return f'jupiter-track: within {jupiter_track_radius(band)} deg of Jupiter on co-added night {hit} (Gregg 2026-09-18)'
    return ''


# ---------------------------------------------------------------------------------------------------------------------
# 3x per-Stokes noise epochs (p3s2_conform.py:125-174, :522-525; Gregg 2026-09-05)
# ---------------------------------------------------------------------------------------------------------------------

def noise_outliers(values, x=None, min_n=None):
    """The dates whose value exceeds x times the median over `values` ({date: value}); none with fewer than min_n."""
    c = _cc()
    x = c.SANITY_X if x is None else x
    min_n = c.SANITY_MIN_NIGHTS if min_n is None else min_n
    if len(values) < min_n:
        return set()
    med = float(np.median(list(values.values())))
    return {d for d, v in values.items() if v > x * med}


def weights_qa3(weights_path):
    """{date: qa3_rms_mjy} of a cell's weights.csv (every row whose value parses)."""
    vals = {}
    for r in csv.DictReader(open(weights_path)):
        try:
            vals[r['date']] = float(r['qa3_rms_mjy'])
        except (TypeError, ValueError, KeyError):
            pass
    return vals


def noise_reason(stokes):
    return f'epoch refused: {stokes} frame noise > {_cc().SANITY_X:.0f}x cell median (Gregg 2026-09-05)'


# ---------------------------------------------------------------------------------------------------------------------
# row-line (p3s2_conform.py:176-190, :526-529; Gregg 2026-09-05) and the backing test (:83-103; 2026-09-05)
# ---------------------------------------------------------------------------------------------------------------------

def rowline_members(points, min_n=None):
    """[(RA, Dec)] of the finals ([(pix_y, RA, Dec)] of one Stokes, cell and epoch) that lie min_n or more on one pixel
    row +/-1."""
    min_n = _cc().ROWLINE_MIN if min_n is None else min_n
    if len(points) < min_n:
        return []
    ys = np.array([p[0] for p in points]); hit = set()
    for y in set(ys.tolist()):
        m = np.where(np.abs(ys - y) <= 1)[0]
        if len(m) >= min_n:
            hit.update(m.tolist())
    return [(points[i][1], points[i][2]) for i in hit]


def rowline_reason(ra, dec, members):
    if members:
        a = np.array(members); dra = (a[:, 0] - ra + 180.0) % 360.0 - 180.0
        if (np.sqrt(dra ** 2 * np.cos(np.radians(dec)) ** 2 + (a[:, 1] - dec) ** 2) < _cc().ROWLINE_MATCH_DEG).any():
            return f'row-line: >= {_cc().ROWLINE_MIN} finals of this epoch on one pixel row (Gregg 2026-09-05)'
    return ''


def backed_rows(rows_by_key, finals_by_key, cap=None):
    """Indices of the catalogue rows ({key: [(index, RA, Dec)]}) matched one-to-one to a final ledger row of their key by
    nearest position within `cap` deg (RA-wrapped) - the merge gate's rule."""
    cap = CFG.catalog_merge.BACKING_CAP_DEG if cap is None else cap
    backed = set()
    for key, crows in rows_by_key.items():
        fl = finals_by_key.get(key, [])
        if not fl:
            continue
        a = np.array(fl); pairs = []
        for i, ra, dec in crows:
            dra = (a[:, 0] - ra + 180.0) % 360.0 - 180.0
            d = np.sqrt(dra ** 2 * np.cos(np.radians(dec)) ** 2 + (a[:, 1] - dec) ** 2)
            for fi in np.where(d < cap)[0]:
                pairs.append((d[fi], i, int(fi)))
        pairs.sort(); uc = set(); uf = set()
        for d, i, fi in pairs:
            if i in uc or fi in uf:
                continue
            uc.add(i); uf.add(fi); backed.add(i)
    return backed


UNBACKED = 'no final gate-ledger verdict (crashed attempt, audit R-08)'
DUPLICATE = 'duplicate (cell executed twice, audit R-07)'
EXCLUDED = 'excluded epoch (data_decisions at publish time; CLAUDE.md S2)'


# ---------------------------------------------------------------------------------------------------------------------
# size and axis ratio above 8 sigma (p3s2_conform.py:536-566; Gregg 2026-09-04/05, D8 2026-10-13)
# ---------------------------------------------------------------------------------------------------------------------

def size_axis(r, bmaj_deg=None, beam_ratio=None, size_step=None):
    """(the row's conformed size/axis columns, the removal words or ''): keeps the search's own size_beams / axis_norm
    (D8); a row without them gets FWHM / its searched frame's BMAJ and axis ratio / the frame beam's BMAJ/BMIN."""
    c = _cc()
    size_step = c.SIZE_STEP['deep'] if size_step is None else size_step
    cols = {}
    ratio = f(r.get('size_beams'))
    if np.isnan(ratio):
        ratio = f(r.get('fwhm_maj_arcmin')) / (bmaj_deg * 60.0) if bmaj_deg else float('nan')
    cols['c50_mjy'] = ''
    cols['size_beams'] = '' if np.isnan(ratio) else round(ratio, 2)
    cols['size_flag'] = ''
    if size_step and abs(f(r['snr'])) > c.SIZE_SNR and not np.isnan(ratio):
        if ratio > c.SIZE_REMOVE_BEAMS:
            return cols, (f'gaussfit size {ratio:.2f} beams > {c.SIZE_REMOVE_BEAMS} above {c.SIZE_SNR} sigma '
                          f'(Gregg 2026-09-04)')
        if ratio > c.SIZE_FLAG_BEAMS:
            cols['size_flag'] = f'size {c.SIZE_FLAG_BEAMS}-{c.SIZE_REMOVE_BEAMS} beams'
    ar = f(r.get('axis_ratio'))
    an = f(r.get('axis_norm'))
    cols['axis_norm'] = round(an, 2) if not np.isnan(an) else ('' if (not beam_ratio or np.isnan(ar)) else round(ar / beam_ratio, 2))
    cols['axis_flag'] = ''
    if cols['axis_norm'] != '' and abs(f(r['snr'])) > c.SIZE_SNR:
        if cols['axis_norm'] > c.AXIS_REMOVE:
            return cols, f'axis ratio {cols["axis_norm"]} x beam > {c.AXIS_REMOVE} above {c.SIZE_SNR} sigma (Gregg 2026-09-05)'
        if cols['axis_norm'] > c.AXIS_FLAG:
            cols['axis_flag'] = f'axis {c.AXIS_FLAG}-{c.AXIS_REMOVE} x beam'
    return cols, ''


# ---------------------------------------------------------------------------------------------------------------------
# bright-source table (p3s2_conform.py:191-276; Gregg 2026-09-05/06, 2026-09-23/24, 2026-10-01, 2026-10-04)
# ---------------------------------------------------------------------------------------------------------------------

def cell_frame_noise(raw):
    """{(lst, band, Stokes): the median over its rows of |peak_jy / global_snr|} - the cell's median frame noise, the
    denominator of S the deep tables were derived with (p3s3_curve_derive.py:10-11)."""
    rms = {}
    for r in raw:
        try:
            rms.setdefault((r['lst'], r['band'], r['stokes']), []).append(abs(f(r['peak_jy'])) / abs(f(r['global_snr'])))
        except Exception:
            pass
    return {k: float(np.nanmedian(v)) for k, v in rms.items()}


def table_cuts(raw, G, PCT, noise, elev_bands=None):
    """{row index: (cut, note)} under the per-band, per-Stokes table of each (cell, Stokes, night): parents from the Robust
    -0.75 master catalogue, S over `noise` ({(lst, band, Stokes): Jy}), distances in the searched epoch frame's beam about
    the master's phase centre, the elevation-band table of each row's mid-hour elevation. G = gate_inputs, PCT =
    parent_curve_table (both refuse, never fall back)."""
    c = _cc()
    elev_bands = c.TABLE_ELEV['deep'] if elev_bands is None else elev_bands
    groups = collections.defaultdict(list)
    for i, r in enumerate(raw):
        groups[(r['lst'], r['band'], r['stokes'], r['date'])].append(i)
    cut = {}
    for gk, idx in groups.items():
        l, b, s = gk[:3]
        ra = np.array([f(raw[i]['ra']) for i in idx]); de = np.array([f(raw[i]['dec']) for i in idx])
        el = mid_hour_elevations(l, ra, de) if elev_bands else None
        cuts, notes = PCT.removals(s, b, G.parents(l, b), noise.get((l, b, s), 0.0), G.deep_target_beam(l, b, gk[3][:10], s),
                                   ra, de, elev=el, phase_centre=G.phase_centre(l, b))
        for i, x, nn in zip(idx, cuts, notes):
            cut[i] = (bool(x), nn)
    return cut


def table_reason(note):
    return f'per-band per-Stokes table: {note} (Gregg 2026-09-05/06)'


# ---------------------------------------------------------------------------------------------------------------------
# pulsar-source consistency (p3s2_conform.py:359-488; Gregg 2026-09-28) - cross-cell: B-02 makes it one named stage
# ---------------------------------------------------------------------------------------------------------------------

_PSR_NOTE = re.compile(r'ATNF pulsar (.+?) (?:is the parent|within 1 beam NOT exempting)')


def pulsar_positions(path=None):
    """{ATNF common name: (RA, Dec) at the current epoch}."""
    pos = {}
    for r in csv.DictReader(open(path or CFG.catalog.PULSARS_CSV, encoding='utf-8-sig')):
        try:
            pos[(r.get('common_name') or '').strip()] = (float(r['ra_current']), float(r['dec_current']))
        except (KeyError, TypeError, ValueError):
            pass
    return pos


def _unit(ra_, de_):
    return np.array([math.cos(math.radians(de_)) * math.cos(math.radians(ra_)),
                     math.cos(math.radians(de_)) * math.sin(math.radians(ra_)), math.sin(math.radians(de_))])


def _model_error(r, LOC):
    """A row's 1-sigma position error (arcsec): source_localization.detections()'s model-error branch (C.8c, Gregg
    2026-10-14) - the merged rows carry no fitted error."""
    el = LOC.elev(r['lst'], f(r['ra']), f(r['dec']))
    stat = _cc().PULSAR_STAT_COEFF * LOC.beam_as(r['band']) / max(abs(f(r['snr'])), _cc().PULSAR_SNR_FLOOR)
    sc = LOC.SYS_SCALE.get((r['stokes'], LOC.A.band_class(r['band']) if r['band'] != 'Full' else 'Full'), 1.0)
    return math.hypot(stat, LOC.A.sys_arcmin(r['band'], el) * 60.0 * sc)


def pulsar_consistency(raw, cut, day, excluded, G, LOC, psr_pos):
    """Groups of 2+ detections within 1 beam of a pulsar that a table decision named (kept as 'the pulsar is the parent',
    or cut because a triggering parent is not the pulsar), the daytime gate and the exclusions passed, linked by the
    association criterion (3.034 x quadrature errors) and combined with the served localization's estimator: the pulsar
    inside the combined 99 % radius keeps the group's table cuts; outside, the group's pulsar-exempted detections are cut.
    Changes `cut` in place; returns (kept, removed, [(pulsar, n, sep arcsec, r99 arcsec)])."""
    c = _cc()
    if LOC._SP.active():
        raise RuntimeError('removal_rules.pulsar_consistency: the 10-minute tie (P3_TENMIN_PERTIE) is set - this port is the '
                           'deep tier only (B-02); REFUSING')
    psr_rows = collections.defaultdict(list)
    for i, (_c, nt) in cut.items():
        m = _PSR_NOTE.search(nt or '')
        if m:
            psr_rows[m.group(1).strip()].append(i)
    n_keep = n_drop = 0
    log = []
    RA_ = np.array([f(r['ra']) for r in raw]); DE_ = np.array([f(r['dec']) for r in raw])
    RAWU = np.nan_to_num(np.stack([np.cos(np.radians(DE_)) * np.cos(np.radians(RA_)),
                                   np.cos(np.radians(DE_)) * np.sin(np.radians(RA_)), np.sin(np.radians(DE_))], -1))
    for pn in psr_rows:
        if pn not in psr_pos:
            continue
        pra, pdec = psr_pos[pn]; pu = _unit(pra, pdec)
        near = []
        for i in np.where(RAWU @ pu > math.cos(math.radians(c.PULSAR_PRECUT_DEG)))[0]:
            r = raw[i]; ra_, de_ = f(r['ra']), f(r['dec'])
            sep = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(pu, _unit(ra_, de_)))))))
            try:
                bm = G.deep_target_beam(r['lst'], r['band'], r['date'][:10], r['stokes'])
            except Exception:
                continue
            if sep > bm:
                continue
            if excluded(r) or day.get((r['stokes'], r['lst'], r['band'], r['date']), 'x'):
                continue
            near.append(i)
        if len(near) < 2:
            continue
        dets = {i: dict(ra=f(raw[i]['ra']), dec=f(raw[i]['dec']), sig=_model_error(raw[i], LOC),
                        epoch=(raw[i]['lst'], raw[i]['date'][:10]), band=raw[i]['band'], snr=f(raw[i]['snr'])) for i in near}
        par = {i: i for i in near}

        def _fd(i):
            while par[i] != i:
                par[i] = par[par[i]]; i = par[i]
            return i
        for a_ in near:
            for b_ in near:
                if a_ < b_:
                    da, db = dets[a_], dets[b_]
                    sep_as = math.hypot((da['ra'] - db['ra']) * math.cos(math.radians(da['dec'])), da['dec'] - db['dec']) * 3600
                    if sep_as < c.PULSAR_ASSOC_K * math.hypot(da['sig'], db['sig']):
                        par[_fd(b_)] = _fd(a_)
        groups = collections.defaultdict(list)
        for i in near:
            groups[_fd(i)].append(i)
        for g in groups.values():
            if len(g) < 2:
                continue
            dl = [dets[i] for i in g]
            cra, cdec, err, _ne = LOC.combine(dl, frame=False)
            chi2 = LOC.consistency(dl, cra, cdec, frame=False)
            if chi2 and chi2 > 1.0:
                err *= math.sqrt(chi2)
            sep = math.hypot((pra - cra) * math.cos(math.radians(cdec)), pdec - cdec) * 3600
            r99 = c.PULSAR_ASSOC_K * err
            for i in g:
                if i not in cut:
                    continue
                _c, nt = cut[i]
                if 'ATNF pulsar' not in (nt or ''):
                    continue
                if sep <= r99 and _c:
                    cut[i] = (False, f'table-exempt: pulsar-consistent multi-detection source - {pn} {sep:.0f} arcsec from its '
                                     f'combined position, inside the 99% radius {r99:.0f} arcsec (n={len(g)}; Gregg 2026-09-28; '
                                     f'would remove: {nt})')
                    n_keep += 1
                elif sep > r99 and not _c:
                    cut[i] = (True, f'table: multi-detection source inconsistent with {pn} - {sep:.0f} arcsec from its combined '
                                    f'position, outside the 99% radius {r99:.0f} arcsec (n={len(g)}): background (Gregg '
                                    f'2026-09-28; was: {nt})')
                    n_drop += 1
            log.append((pn, len(g), round(sep), round(r99)))
    return n_keep, n_drop, log, len(psr_rows)


# ---------------------------------------------------------------------------------------------------------------------
# the recorded inputs of a run (p3s2_conform.py's reads, in its order; its three caches in the run dir)
# ---------------------------------------------------------------------------------------------------------------------

_RESID = re.compile(r'(\d{4}-\d\d-\d\d)_[IV]_redo_resid_(\w+)\.fits$')


def ledger_finals(U):
    """(finals {(lst, scope, Stokes, date): [(RA, Dec)]}, row-line points {(Stokes, lst, scope, date): [(pix_y, RA, Dec)]},
    v110 marginal rows {(lst, scope, Stokes, date): [row]}) of the run's gate ledger, exact-duplicate rows collapsed (a
    cell executed twice). Only the rows these use are kept: a final verdict, or a v110 marginal veto."""
    finals = collections.defaultdict(list); rows_by = collections.defaultdict(list); marg = collections.defaultdict(list)
    seen = set()
    for r in csv.DictReader(open(f'{U}/v_deep_ledger.csv')):
        if r['final'] != '1' and r['veto_reason'] != 'v110:marginal':
            continue
        k = tuple(r.values())
        if k in seen:
            continue
        seen.add(k)
        key = (r['lst'], r['scope'], r['stokes'], r['date'])
        if r['final'] == '1':
            finals[key].append((f(r['ra']), f(r['dec'])))
        if r['veto_reason'] == 'v110:marginal':
            marg[key].append(r)
        if r.get('final') == '1':
            try:
                rows_by[(r['stokes'], r['lst'], r['scope'], r['date'])].append((int(float(r['pix_y'])), f(r['ra']), f(r['dec'])))
            except (TypeError, ValueError):
                pass
    return finals, rows_by, marg


def v110_labels(U):
    """{(lst, band, date, RA 2 dp, Dec 2 dp): fp_prediction} of the run's v110 verdict files (a later file wins)."""
    vlab = {}
    for p in glob.glob(f'{U}/v110_products/*/V_top100_fp_gate_p3.csv'):
        for r in csv.DictReader(open(p)):
            try:
                vlab[(r['lst'], r['band'], r['date'], round(f(r['ra']), 2), round(f(r['dec']), 2))] = r.get('fp_prediction', '')
            except Exception:
                pass
    return vlab


def _resid_frames(U, stokes):
    """[(path, (Stokes, lst, band, date))] of the run's archived search frames of one Stokes."""
    out = []
    for p in glob.glob(f'{U}/{"vresid" if stokes == "V" else "iresid"}/*/*_{stokes}_redo_resid_*.fits'):
        m = _RESID.match(os.path.basename(p))
        if m:
            out.append((p, (stokes, os.path.basename(os.path.dirname(p)).split('_')[0], m.group(2), m.group(1))))
    return out


def frame_beam_ratios(U):
    """{(Stokes, lst, band, date): BMAJ/BMIN} of the run's search frames: frame_beam_ratio.csv, then every frame it lacks
    (the cache is written again, rounded to 4 places, when any was added)."""
    from astropy.io import fits
    brp = f'{U}/frame_beam_ratio.csv'
    BR = {}
    if os.path.exists(brp):
        for r in csv.DictReader(open(brp)):
            BR[(r['stokes'], r['lst'], r['band'], r['date'])] = float(r['ratio'])
    new = []
    for st in ('V', 'I'):
        for p, k in _resid_frames(U, st):
            if k in BR:
                continue
            try:
                h = fits.getheader(p); BR[k] = float(h['BMAJ']) / float(h['BMIN']); new.append(k)
            except Exception:
                pass
    if new:
        with open(brp, 'w', newline='') as fh:
            w = csv.writer(fh); w.writerow(['stokes', 'lst', 'band', 'date', 'ratio'])
            for k, v in BR.items():
                w.writerow([*k, round(v, 4)])
    return BR


def frame_bmaj(U):
    """{(Stokes, lst, band, date): BMAJ deg} of the run's search frames (not the PB-corrected ones): frame_beam_bmaj.csv,
    then every frame it lacks (the cache is replaced, full precision, when any was added)."""
    from astropy.io import fits
    bmp = f'{U}/frame_beam_bmaj.csv'
    BM = {}
    if os.path.exists(bmp):
        for r in csv.DictReader(open(bmp)):
            BM[(r['stokes'], r['lst'], r['band'], r['date'])] = float(r['bmaj_deg'])
    new = []
    for st in ('V', 'I'):
        for p, k in _resid_frames(U, st):
            if k[2].startswith('pbcorr') or k in BM:
                continue
            try:
                BM[k] = float(fits.getheader(p)['BMAJ']); new.append(k)
            except Exception:
                pass
    if new:
        with open(bmp + '.part', 'w', newline='') as fh:
            w = csv.writer(fh); w.writerow(['stokes', 'lst', 'band', 'date', 'bmaj_deg'])
            for k, v in BM.items():
                w.writerow([*k, repr(v)])
        os.replace(bmp + '.part', bmp)
    return BM


def i_noise_epochs(roots_csv=None):
    """{(lst, band, date)} whose I flux-alignment QA rms (weights.csv qa3_rms_mjy) exceeds SANITY_X x its cell's median, over
    every cell of the search fleet's roots map."""
    roots = {l.split(',')[0]: l.strip().split(',')[2] for l in open(roots_csv or CFG.catalog.DEEP_ROOTS_CSV) if l.strip()}
    bad = set()
    for cell, root in roots.items():
        wp = f'{root}/{cell}/coadd_final/weights.csv'
        if not os.path.exists(wp):
            continue
        l, b = cell.split('_')
        bad |= {(l, b, d) for d in noise_outliers(weights_qa3(wp))}
    return bad


def v_noise_epochs(U):
    """{(lst, band, date)} whose archived V search frame's THERMNOI exceeds SANITY_X x its cell's median (frames with a
    positive THERMNOI)."""
    from astropy.io import fits
    vn = {}
    for p, k in _resid_frames(U, 'V'):
        try:
            vn[k[1:]] = float(fits.getheader(p).get('THERMNOI') or 0)
        except Exception:
            pass
    bad = set()
    for (l, b) in {(k[0], k[1]) for k in vn}:
        bad |= {(l, b, d) for d in noise_outliers({k[2]: v for k, v in vn.items() if k[0] == l and k[1] == b and v > 0})}
    return bad


def frame_times(U, keys):
    """{(Stokes, lst, band, date): (DATE-OBS, DATE-END)}: frame_times.csv, then the header of every frame of `keys` it
    lacks (the cache is written again, sorted, when any key was missing - found or not)."""
    from astropy.io import fits
    ftp = f'{U}/frame_times.csv'
    FT = {}
    if os.path.exists(ftp):
        for r in csv.DictReader(open(ftp)):
            FT[(r['stokes'], r['lst'], r['band'], r['date'])] = (r['date_obs'], r['date_end'])
    need = sorted(set(keys) - set(FT))
    for (st, l, b, d) in need:
        p = f"{U}/{'vresid' if st == 'V' else 'iresid'}/{l}_{b}/{d}_{st}_redo_resid_{b}.fits"
        try:
            h = fits.getheader(p); FT[(st, l, b, d)] = (str(h['DATE-OBS']), str(h['DATE-END']))
        except Exception:
            pass
    if need:
        with open(ftp, 'w', newline='') as fh:
            w = csv.writer(fh); w.writerow(['stokes', 'lst', 'band', 'date', 'date_obs', 'date_end'])
            for k, v in sorted(FT.items()):
                w.writerow([*k, *v])
    return FT


# ---------------------------------------------------------------------------------------------------------------------
# the conformance of a run, composed in production's order (p3s2_conform.py, deep tier, main chain)
# ---------------------------------------------------------------------------------------------------------------------

def _refuse_other_modes(size_step):
    """The port is the deep tier's main chain (B-02: deep tier only); the fork's, the 10-minute tier's and the comparison
    modes stay production's script. Each environment switch p3s2_conform.py and parent_curve_table read must be the deep
    main chain's (they are what the chain passes), and a P3_CONFORM_SIZE that is set must be the size step asked for -
    else REFUSING, never another mode silently."""
    c = _cc()
    bad = []
    want = {'P3_CONFORM_FORK': ('', '0'), 'P3_CONFORM_TABLE': ('', c.TABLE_MODE['deep']), 'P3_CADENCE': ('', 'deep'),
            'P3_CONFORM_SIZE': ('', '1' if size_step else '0'),
            'P3_GATE_TARGET_BEAM': ('', '1') if c.TABLE_TARGET_BEAM else ('0',),
            'P3_TABLE_ELEV': ('', '1') if c.TABLE_ELEV['deep'] else ('0',), 'P3_TABLE_PREFIX': ('', c.TABLE_PREFIX['deep']),
            'P3_TABLE_RECORD_ONLY': ('', '0'), 'P3_TABLE_XM_EXEMPT': ('', '1') if c.TABLE_XM_EXEMPT else ('0',),
            'P3_TABLE_ELEV_BANDS': ('', c.TABLE_ELEV_BANDS), 'P3_TABLE_EXTRA': ('',), 'P3_TENMIN_PERTIE': ('',),
            'P3_DEEP_MIN_ELEVATION': ('', f'{c.DEEP_ELEV_FLOOR_DEG}', f'{c.DEEP_ELEV_FLOOR_DEG:g}')}
    for k, ok in want.items():
        if os.environ.get(k, '') not in ok:
            bad.append(f'{k}={os.environ.get(k)!r}')
    if c.TABLE_MODE['deep'] != 'singles' or not c.TABLE_TARGET_BEAM:
        bad.append(f'catalog_conform.TABLE_MODE[deep]={c.TABLE_MODE["deep"]!r}, TABLE_TARGET_BEAM={c.TABLE_TARGET_BEAM!r}')
    if bad:
        raise SystemExit('conform (removal_rules): not the deep main chain - ' + ', '.join(bad) + ' - REFUSING')


def conform(U, catalog_dir=None, size_step=None, out=print):
    """p3s2_conform.py <U> (deep tier, main chain), from the run's recorded inputs. Writes, as it does: <CD>/
    master_catalog_asrun.csv (the merged rows, untouched), master_catalog.csv (conformant), master_catalog_removed.csv
    (every removed row and its reason), and the run's frame caches. Returns the counts. Refuses where it refuses."""
    import shutil
    import gate_inputs as G
    import p3s2_frames as F
    import source_localization as LOC
    from exopipe.phase3.lib.masters.coadd import load_exclusions, is_excluded
    from exopipe.phase3.lib.search import parent_curve_table as PCT
    c = _cc()
    size_step = c.SIZE_STEP['deep'] if size_step is None else size_step
    _refuse_other_modes(size_step)
    CD = catalog_dir or U
    raw = list(csv.DictReader(open(f'{CD}/master_catalog.csv')))
    if not raw:
        raise SystemExit(f'conform: {CD}/master_catalog.csv has no rows - REFUSING')
    cols = list(raw[0].keys())
    shutil.copy(f'{CD}/master_catalog.csv', f'{CD}/master_catalog_asrun.csv')
    finals, rows_by, marg = ledger_finals(U)
    vlab = v110_labels(U)

    def v110_label(r):
        v = vlab.get((r['lst'], r['band'], r['date'], round(f(r['ra']), 2), round(f(r['dec']), 2)), '')
        return v if v else 'in-chain pass (verdict file not found)'
    groups = collections.defaultdict(list)
    for i, r in enumerate(raw):
        groups[(r['lst'], r['band'], r['stokes'], r['date'])].append((i, f(r['ra']), f(r['dec'])))
    backed = backed_rows(groups, finals)
    BR = frame_beam_ratios(U)
    BMAJ = frame_bmaj(U)
    bad_i = i_noise_epochs()
    bad_v = v_noise_epochs(U)
    rowline = {}
    for k, pts in rows_by.items():
        m = rowline_members(pts)
        if m:
            rowline[k] = m
    EXCL = load_exclusions()
    noise = cell_frame_noise(raw)
    cut = table_cuts(raw, G, PCT, noise)
    jtrack = {}

    def track(lst, band):
        if (lst, band) not in jtrack:
            root = F.cell_root(lst, band)
            jtrack[(lst, band)] = jupiter_track(lst, f'{root}/{lst}_{band}/coadd_final/weights.csv' if root else None)
        return jtrack[(lst, band)]
    quar = solar_quarantine(CFG.catalog.SOLAR_AUDIT)
    fkeys = {(r['stokes'], r['lst'], r['band'], r['date']) for r in raw}
    FT = frame_times(U, fkeys)
    day = daytime_verdicts({k: FT[k] for k in fkeys if k in FT}, quar)
    n_keep, n_drop, plog, n_psr = pulsar_consistency(raw, cut, day, lambda r: is_excluded(r['lst'], r['band'], r['date'], EXCL),
                                                     G, LOC, pulsar_positions())
    out(f'conform: pulsar-source consistency - {n_psr} pulsars in table decisions; kept {n_keep} table-cut detections of '
        f'pulsar-consistent sources, removed {n_drop} pulsar-exempted detections of inconsistent sources', flush=True)
    for x in plog:
        out(f'conform: pulsar group {x[0]}: n={x[1]} combined position {x[2]} arcsec from the pulsar, 99% radius {x[3]} arcsec '
            f'-> {"consistent" if x[2] <= x[3] else "inconsistent"}', flush=True)
    n = collections.Counter()
    removed, kept, seen = [], [], set()
    for i, r in enumerate(raw):
        key = (r['lst'], r['band'], r['stokes'], r['date'])
        ra, dec = f(r['ra']), f(r['dec'])
        dk = key + (round(ra, 4), round(dec, 4))
        fk = (r['stokes'], r['lst'], r['band'], r['date'])
        why = ''
        if dk in seen:
            why, kind = DUPLICATE, 'dup'
        elif is_excluded(r['lst'], r['band'], r['date'], EXCL):
            why, kind = EXCLUDED, 'excl'
        elif c.DEEP_ELEV_EPOCH != 'frame-mid' or fk in FT:   # frame-mid without the frame's times: the daytime gate refuses it
            why, kind = elevation_reason(*deep_elevation(r['lst'], ra, dec, frame_time=FT.get(fk))), 'elf'
        if not why:
            if fk not in day:
                why, kind = DAYTIME_UNREADABLE, 'dayx'
            elif day[fk]:
                why, kind = day[fk], ('day' if day[fk].startswith('daytime') else 'dayq')
        if not why:
            why, kind = jupiter_track_reason(ra, dec, r['band'], track(r['lst'], r['band'])), 'jup'
        note = ''
        if not why:
            x, note = cut.get(i, (False, ''))
            if x:
                why, kind = table_reason(note), 'par'
        if why:
            n[kind] += 1; removed.append({**r, 'removal_reason': why})
            continue
        seen.add(dk)
        if (r['stokes'] == 'I' and (r['lst'], r['band'], r['date']) in bad_i) or \
                (r['stokes'] == 'V' and (r['lst'], r['band'], r['date']) in bad_v):
            why, kind = noise_reason(r['stokes']), 'san'
        else:
            why, kind = rowline_reason(ra, dec, rowline.get(fk)), 'row'
        if not why and i not in backed:
            why, kind = UNBACKED, 'noled'
        if why:
            n[kind] += 1; removed.append({**r, 'removal_reason': why})
            continue
        r = dict(r)
        r['fp_prediction'] = (r.get('fp_prediction') or (v110_label(r) if r['stokes'] == 'V' else ''))
        r['conform_note'] = note or ''
        cols_, why = size_axis(r, BMAJ.get(fk), BR.get(fk), size_step=size_step)
        r.update(cols_)
        if why:
            n['size' if why.startswith('gaussfit') else 'axis'] += 1; removed.append({**r, 'removal_reason': why})
            continue
        if r['size_flag']:
            n['flag'] += 1
        if r['axis_flag']:
            n['aflag'] += 1
        kept.append(r)
    for key, rows in marg.items():
        lst, band, stokes, date = key
        for j, m in enumerate(rows):
            dk = key + (round(f(m['ra']), 4), round(f(m['dec']), 4))
            if dk in seen:
                continue
            seen.add(dk); n['marg'] += 1
            kept.append({'cand_id': f'VM_{lst}_{date}_{band}_{j:04d}', 'lst': lst, 'date': date, 'band': band,
                         'stokes': stokes, 'cadence': 'deep', 'interval': '', 'ra': m['ra'], 'dec': m['dec'],
                         'peak_jy': m['peak_jy'], 'int_jy': '', 'snr': m['snr'], 'axis_ratio': '', 'fwhm_maj_arcmin': '',
                         'c50_mjy': '', 'mstruct': m['mstruct'], 'mstruct_raw': m['mstruct_raw'], 'sstack': m['sstack'],
                         'jupiter_deg': m['jupiter_deg'], 'n_v_epoch': m['n_v_epoch'], 'body': '', 'repeat_group': '',
                         'epoch_qa': '', 'fp_prediction': 'marginal', 'size_beams': '', 'size_flag': '', 'axis_norm': '',
                         'axis_flag': '', 'conform_note': 'v110 marginal RE-ADMITTED from gate ledger (Gregg 2026-09-04: retain + label)'})
    ocols = cols + ['fp_prediction', 'size_beams', 'size_flag', 'axis_norm', 'axis_flag', 'conform_note']
    with open(f'{CD}/master_catalog.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=ocols, extrasaction='ignore'); w.writeheader(); w.writerows(kept)
    with open(f'{CD}/master_catalog_removed.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=ocols + ['removal_reason'], extrasaction='ignore'); w.writeheader(); w.writerows(removed)
    out(f'conform: {len(raw)} raw -> {len(kept)} conformant | removed: {n["excl"]} on excluded epochs, {n["elf"]} below the deep '
        f'elevation floor {c.DEEP_ELEV_FLOOR_DEG:g} deg, {n["day"]} daytime (Sun above the horizon in the frame), {n["dayq"]} '
        f'twilight in a solar-audit quarantine band, {n["dayx"]} frame times unreadable, {n["jup"]} jupiter-track (Gregg '
        f'2026-09-18), {n["par"]} per-band table (both Stokes), {n["dup"]} duplicates, {n["noled"]} without ledger verdict, '
        f'{n["size"]} gaussfit size >{c.SIZE_REMOVE_BEAMS} beams above {c.SIZE_SNR} sigma | flagged {n["flag"]} at '
        f'{c.SIZE_FLAG_BEAMS}-{c.SIZE_REMOVE_BEAMS} beams | {n["axis"]} axis ratio >{c.AXIS_REMOVE} x beam above {c.SIZE_SNR} '
        f'sigma | flagged {n["aflag"]} at {c.AXIS_FLAG}-{c.AXIS_REMOVE} | {n["san"]} on epochs refused by the 3x per-Stokes '
        f'noise gate | {n["row"]} row-line | {n["marg"]} v110 marginal re-admitted (labelled)')
    if len(kept) == 0:
        raise SystemExit('conform: ZERO rows would be served - REFUSING (gate, not warning)')
    return dict(n, raw=len(raw), kept=len(kept), removed=len(removed))


def main(argv=None):
    """python -m exopipe.phase3.lib.search.removal_rules conform <run dir> [<catalogue dir>] - p3s2_conform.py's call."""
    import sys
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) < 2 or argv[0] != 'conform':
        raise SystemExit('usage: removal_rules conform <run dir> [<catalogue dir>]')
    from exopipe.phase3.core import env
    env.legacy()
    conform(argv[1], argv[2] if len(argv) > 2 else None)


if __name__ == '__main__':
    main()
