"""The JOBS of the forced targets' wide-area burst test (family W, stage wide_jobs) and the forced events the analysis weighs.

Port of the analysis scripts (phase3/refactor_analysis/review_items_20261031/forced_wide/): forced_ambiguous.py (2026-11-01,
the ambiguous-target list), forced_wide_jobs.py (the base jobs; Gregg 2026-11-01 "a wide search area that allows the burst
tapering to be identified", "Go"), v3/jobs_ext.py (the extension jobs after Gregg's KELT-24 note, 2026-11-02) and
v3/forced_wide_v3.py load_events. Every computation is the scripts' own, in their order (a target's first row wins); the
values come from config (forced_wide).

A job is one (Stokes, target) of the four forced runs (forced_wide.RUNS, in that order):
  AMBIGUOUS  each listed identity of a run (queue_<S>.txt order) at its current list position (snapshot_forced_assoc.
             build_targets, in place): the nearest cataloged entry >= AMB_MIN_S74_JY within AMB_SEARCH_BEAMS Master beams, the
             beam at the median frequency of its forced events (MASTER_BEAM_ARCMIN, nearest key; WIDEBAND_MHZ for a wideband);
  BASE       ambiguous targets of the 10-s and 1-minute runs with that entry within BASE_MAX_BEAMS beams: the background S = the
             nearest physical source (entries grouped within GROUP_ARCMIN, the flux VLSSr's where listed) >= BASE_BG_MIN_S74_JY
             within BASE_BG_RADIUS_BEAMS beams, the bands = the subbands with forced events of that Stokes (10 s or 1 min);
  EXTENSION  the other targets with forced events that have a physical source >= EXT_MIN_S74_JY within EXT_RADIUS_FWHM x the
             snapshot FWHM of their lowest event band (sorted by Stokes, identity).
The base jobs' v1 same-cutout control pairs (forced_wide_jobs.py 'controls', 'mid', 'r_max_arcmin', 'bg_not_isolated') are
not carried: no stage reads them (v2 pooled its controls; v3 calibrates on control lines).
EVENTS: every forced event of a run's subbands - the candidates/<dir>/<dir>/detections.csv rows with a frame sigma, one per
(identity, cand_id) - with its S/N and frame sigma as written (the analysis' value = S/N x sigma, Jy/beam)."""
import collections
import csv
import glob
import json
import math
import os

import numpy as np

EVENT_COLUMNS = ('stokes', 'tier', 'identity', 'cand_id', 'band', 'ts', 'snr', 'sigma_frame_jyb')


def cfg():
    from exopipe.phase3.config import CFG
    return CFG


def inplace(name):
    """A clv15 / exopipe_ops module imported in place (core/env.legacy put their roots on sys.path). The sys.path entries a
    module inserts at import (snapshot_forced_eval, snapshot_forced_assoc put the production roots first) are taken out again,
    so later imports of this process resolve as before."""
    import importlib
    import sys
    before = list(sys.path)
    try:
        return importlib.import_module(name)
    finally:
        sys.path[:] = before


def FA():
    return inplace('snapshot_forced_assoc')


def beam_fwhm_arcmin(band):
    """The snapshot frame beam FWHM of a band (phase3_snapshot_astro.beam_fwhm_arcmin, in place)."""
    return inplace('phase3_snapshot_astro').beam_fwhm_arcmin(band)


def band_mhz(band):
    return float(band[:-3])


# ---------------------------------------------------------------------------------------------------------------------
# geometry and the cataloged sources (forced_wide_jobs.py:30-73, verbatim)
# ---------------------------------------------------------------------------------------------------------------------

def tan_off(ra, dec, ra0, dec0):
    """(east, north) arcmin of (ra, dec) in the tangent plane at (ra0, dec0) (gnomonic)"""
    r, d, r0, d0 = map(np.radians, (ra, dec, ra0, dec0))
    cosc = np.sin(d0) * np.sin(d) + np.cos(d0) * np.cos(d) * np.cos(r - r0)
    x = np.cos(d) * np.sin(r - r0) / cosc
    y = (np.cos(d0) * np.sin(d) - np.sin(d0) * np.cos(d) * np.cos(r - r0)) / cosc
    return np.degrees(x) * 60.0, np.degrees(y) * 60.0


def tan_pos(e, n, ra0, dec0):
    """inverse of tan_off: the sky position (ra, dec) of tangent-plane offsets (east, north arcmin) at (ra0, dec0)"""
    x, y = np.radians(np.asarray(e, float) / 60.0), np.radians(np.asarray(n, float) / 60.0)
    r0, d0 = math.radians(ra0), math.radians(dec0)
    rho = np.hypot(x, y)
    c = np.arctan(rho)
    with np.errstate(invalid='ignore', divide='ignore'):
        dec = np.where(rho > 0, np.arcsin(np.cos(c) * math.sin(d0) + y * np.sin(c) * math.cos(d0) / np.where(rho > 0, rho, 1)), d0)
        ra = r0 + np.arctan2(x * np.sin(c), rho * math.cos(d0) * np.cos(c) - y * math.sin(d0) * np.sin(c))
    return float(np.degrees(ra) % 360.0), float(np.degrees(dec))


def physical_sources(ra0, dec0, rmax_arcmin, smin, group_arcmin=None):
    """cataloged entries >= smin Jy (74 MHz) within rmax of (ra0, dec0), grouped into physical sources (entries within
    group_arcmin, default forced_wide.GROUP_ARCMIN): [(ra, dec, S74, name)], the flux = VLSSr's where listed, else the
    brightest entry's (forced_wide_jobs.physical_sources)"""
    fa = FA()
    cats = fa.catalogues()
    g_arcmin = cfg().forced_wide.GROUP_ARCMIN if group_arcmin is None else group_arcmin
    ents = []
    u0 = fa.uvec(ra0, dec0)
    for k, a in cats.items():
        d = np.degrees(np.arccos(np.clip(a['xyz'] @ u0, -1, 1))) * 60
        for j in np.where((d < rmax_arcmin) & np.isfinite(a['S74']) & (a['S74'] >= smin))[0]:
            ents.append((float(a['ra'][j]), float(a['dec'][j]), float(a['S74'][j]), k, str(a['name'][j])))
    ents.sort(key=lambda e: -e[2])
    groups = []
    for e in ents:
        for g in groups:
            if float(fa.sep_arcsec(e[0], e[1], g[0][0], g[0][1])) / 60.0 <= g_arcmin:
                g.append(e)
                break
        else:
            groups.append([e])
    out = []
    for g in groups:
        v = [e for e in g if e[3] == 'VLSSr']
        ref = v[0] if v else g[0]
        out.append((ref[0], ref[1], (v[0][2] if v else max(e[2] for e in g)), f'{ref[3]} {ref[4]}'))
    return out


# ---------------------------------------------------------------------------------------------------------------------
# the forced runs
# ---------------------------------------------------------------------------------------------------------------------

def run_label(stokes, cadence):
    return f'{stokes}{cadence}'


def run_dirs(root):
    """[(label, Stokes, cadence, run dir)] of forced_wide.RUNS under ``root`` (forced_wide.RUN_NAME)."""
    w = cfg().forced_wide
    return [(run_label(S, cad), S, cad, os.path.join(root, w.RUN_NAME.format(S=S, cadence=cad))) for S, cad in w.RUNS]


def queue_ids(run, stokes):
    """The listed identities of a run, in queue order, once each (forced_ambiguous.py:31-37)."""
    q = []
    with open(os.path.join(run, f'queue_{stokes}.txt')) as f:
        for ln in f:
            ln = ln.strip()
            if ln and not ln.startswith('#'):
                i = ln.split()[0].split(',')[0]
                if i not in q:
                    q.append(i)
    return q


def event_rows(run):
    """forced_assoc_events.csv of a run (identity_id, band, ...)."""
    with open(os.path.join(run, 'forced_assoc_events.csv'), newline='') as f:
        return list(csv.DictReader(f))


def run_inputs(run, stokes):
    """Every file of a run the jobs read: the queue, the events table, the build tree's and the served tree's detections."""
    out = [os.path.join(run, f'queue_{stokes}.txt'), os.path.join(run, 'forced_assoc_events.csv')]
    out += sorted(glob.glob(os.path.join(run, 'candidates', '*', '*', 'detections.csv')))
    return out


# ---------------------------------------------------------------------------------------------------------------------
# ambiguous targets (forced_ambiguous.py)
# ---------------------------------------------------------------------------------------------------------------------

def event_mhz(band):
    w = cfg().forced_wide
    return float(band[:-3]) if band.endswith('MHz') else w.WIDEBAND_MHZ.get(band, w.DEFAULT_MHZ)


def master_beam(f):
    """The Master beam (arcmin) at the nearest tabulated frequency (ties to the first key, as min() takes it)."""
    tab = cfg().forced_wide.MASTER_BEAM_ARCMIN
    k = min(tab, key=lambda x: abs(x - f))
    return tab[k]


def nearest_entries(ra, de, rmax):
    """[(sep arcmin, S74, catalogue, name)] of every catalogue entry >= AMB_MIN_S74_JY within rmax, sorted (nearest first)."""
    fa = FA()
    smin = cfg().forced_wide.AMB_MIN_S74_JY
    out = []
    for k, a in fa.catalogues().items():
        d = np.degrees(np.arccos(np.clip(a['xyz'] @ fa.uvec(ra, de), -1, 1))) * 60
        for j in np.where((d < rmax) & np.isfinite(a['S74']) & (a['S74'] >= smin))[0]:
            out.append((float(d[j]), float(a['S74'][j]), k, str(a['name'][j])))
    return sorted(out)


def target_member(ident, tg, rule):
    """The member target of a (merged) identity whose list position the job takes: tg = build_targets' {(sample, target):
    (ra, dec, name, offset)} of the identity. 'first' = the first in build_targets' order - production: the run tree's glob
    order, which differs between /data07 and a copy (forced_ambiguous.py ran on /data07); 'identity' = the member the identity
    is named after, else the first in (sample, target) order - the same on any copy."""
    items = list(tg.items())
    if rule == 'first':
        return items[0]
    if rule != 'identity':
        raise ValueError(f'forced_wide.TARGET_MEMBER {rule!r}: identity or first')
    own = [it for it in items if it[0][1] == ident]
    return own[0] if own else min(items, key=lambda it: it[0])


def ambiguous_rows(runs):
    """forced_ambiguous.py's rows for ``runs`` = [(label, Stokes, cadence, run dir)]: every listed identity with a target, its
    list position, its events' median frequency and beam, and its nearest cataloged background (the label column of the
    script is not carried: no job reads it)."""
    fa = FA()
    w = cfg().forced_wide
    rows = []
    for label, st, _cad, run in runs:
        q = queue_ids(run, st)
        T = fa.build_targets(run)
        ev = collections.defaultdict(list)
        for e in event_rows(run):
            ev[e['identity_id']].append(e['band'])
        for ident in q:
            tg = T.get(ident) or {}
            if not tg:
                continue
            (smp, _tname), (tra, tde, disp, _o) = target_member(ident, tg, w.TARGET_MEMBER)
            bands = ev.get(ident, [])
            f = float(np.median([event_mhz(b) for b in bands])) if bands else w.DEFAULT_MHZ
            bm = master_beam(f)
            nb = nearest_entries(tra, tde, w.AMB_SEARCH_BEAMS * bm)
            d1 = nb[0] if nb else None
            rows.append(dict(run=label, identity=ident, target=disp, sample=smp, ra=tra, dec=tde, n_events=len(bands),
                             median_mhz=f, beam_arcmin=bm, nearest_bg=(f'{d1[2]} {d1[3]}' if d1 else ''),
                             bg_s74=(d1[1] if d1 else ''), bg_sep_arcmin=(round(d1[0], 2) if d1 else ''),
                             bg_sep_beams=(round(d1[0] / bm, 3) if d1 else '')))
    return rows


# ---------------------------------------------------------------------------------------------------------------------
# the jobs (forced_wide_jobs.py main, jobs_ext.py)
# ---------------------------------------------------------------------------------------------------------------------

def event_bands(runs):
    """{(Stokes, identity): {subband}} of the runs' forced events (forced_wide_jobs.py:79-84)."""
    out = collections.defaultdict(set)
    for _label, st, _cad, run in runs:
        for e in event_rows(run):
            if e['band'].endswith('MHz'):
                out[(st, e['identity_id'])].add(e['band'])
    return out


def build_jobs(runs, log=print):
    """(base jobs, extension jobs, ambiguous rows) for ``runs`` (module notes)."""
    fa = FA()
    w = cfg().forced_wide
    amb = ambiguous_rows(runs)
    labels = {lb for lb, _s, _c, _r in runs}
    ev_bands = event_bands(runs)
    pairs = collections.OrderedDict()
    for r in amb:
        if r['bg_sep_beams'] != '' and float(r['bg_sep_beams']) <= w.BASE_MAX_BEAMS and r['run'] in labels:
            pairs.setdefault((r['run'][0], r['identity']), r)
    base = []
    for (st, ident), r in pairs.items():
        tra, tde = float(r['ra']), float(r['dec'])
        bm = float(r['beam_arcmin'])
        near = physical_sources(tra, tde, bm * w.BASE_BG_RADIUS_BEAMS, w.BASE_BG_MIN_S74_JY)
        if not near:
            log(f'  {st} {ident}: no background source found again (skipped)')
            continue
        near.sort(key=lambda s: float(fa.sep_arcsec(s[0], s[1], tra, tde)))
        S = near[0]
        sep = float(fa.sep_arcsec(S[0], S[1], tra, tde)) / 60.0
        bands = sorted(ev_bands.get((st, ident), set()), key=band_mhz)
        if not bands:
            log(f'  {st} {ident}: no subband events (skipped)')
            continue
        base.append(dict(stokes=st, identity=ident, sample=r['sample'], target=r['target'], T=[tra, tde], S=[S[0], S[1]],
                         s74=S[2], bg_name=S[3], sep_arcmin=sep, bands=bands, median_mhz=float(r['median_mhz'])))
    have = {(j['stokes'], j['identity']) for j in base}
    rows = {}
    for r in amb:
        rows.setdefault((r['run'][0], r['identity']), r)
    ext = []
    for (st, ident), r in sorted(rows.items()):
        if (st, ident) in have or r.get('ra') in (None, ''):
            continue
        bands = sorted(ev_bands.get((st, ident), set()), key=band_mhz)
        if not bands:
            continue
        ra, de = float(r['ra']), float(r['dec'])
        fw = beam_fwhm_arcmin(bands[0])
        br = physical_sources(ra, de, w.EXT_RADIUS_FWHM * fw, w.EXT_MIN_S74_JY)
        if not br:
            continue
        br.sort(key=lambda s: -s[2])
        ext.append(dict(stokes=st, identity=ident, target=r['target'], sample=r['sample'], T=[ra, de], bands=bands,
                        bright=[[s[0], s[1], s[2], s[3]] for s in br]))
    return base, ext, amb


BASE_KEYS = ('stokes', 'identity', 'sample', 'target', 'T', 'S', 's74', 'bg_name', 'sep_arcmin', 'bands', 'median_mhz')
EXT_KEYS = ('stokes', 'identity', 'target', 'sample', 'T', 'bands', 'bright')


def adopt_jobs(paths):
    """(base, ext) from job lists written by the analysis (forced_wide_jobs.py jobs.json, then v3 jobs_ext.json: lists of job
    dicts; a job with 'bright' is an extension job): each job's carried fields only, in the files' order."""
    base, ext = [], []
    for p in paths:
        with open(p) as f:
            for j in json.load(f):
                if 'bright' in j:
                    ext.append({k: j[k] for k in EXT_KEYS})
                else:
                    base.append({k: j[k] for k in BASE_KEYS})
    return base, ext


def all_jobs(doc):
    """The job list of a jobs.json document in the analysis' order: the base jobs, then the extension jobs."""
    return list(doc['base']) + list(doc['ext'])


def write_jobs(path, base, ext, runs, adopted=None):
    doc = {'base': base, 'ext': ext, 'runs': [{'label': lb, 'stokes': s, 'cadence': c, 'dir': d} for lb, s, c, d in runs],
           'adopted': adopted or []}
    tmp = '%s.tmp.%d' % (path, os.getpid())
    with open(tmp, 'w') as f:
        json.dump(doc, f, indent=1)
        f.write('\n')
    os.replace(tmp, path)


def load_jobs(path):
    with open(path) as f:
        return json.load(f)


def job_key(job):
    return f"{job['stokes']}_{job['identity']}"


# ---------------------------------------------------------------------------------------------------------------------
# the forced events (forced_wide_v3.py load_events)
# ---------------------------------------------------------------------------------------------------------------------

def listing(path, order):
    """os.listdir in forced_wide.EVENT_ORDER: 'sorted', or 'directory' (the file system's own order, as the analysis read)."""
    names = os.listdir(path)
    if order == 'sorted':
        return sorted(names)
    if order != 'directory':
        raise ValueError(f'forced_wide.EVENT_ORDER {order!r}: sorted or directory')
    return names


def event_table(runs, order=None):
    """[row] of EVENT_COLUMNS: per run (Stokes, cadence) every candidates/<b>/<d>/detections.csv row of a subband with a frame
    sigma, one per (identity, cand_id) - the last one read (a cand_id the build tree and the served tree both hold carries the
    same values in both: event_conflicts). The rows' order is the order the analysis meets a target's bands in - it lists and
    sums them so - read in forced_wide.EVENT_ORDER."""
    order = order or cfg().forced_wide.EVENT_ORDER
    out = []
    for _label, st, cad, run in runs:
        by = collections.OrderedDict()
        root = os.path.join(run, 'candidates')
        for b in listing(root, order):
            for d in listing(os.path.join(root, b), order):
                p = os.path.join(root, b, d, 'detections.csv')
                if not os.path.exists(p):
                    continue
                with open(p, newline='') as f:
                    for r in csv.DictReader(f):
                        if not r['band'].endswith('MHz') or not r.get('sigma_frame_jyb'):
                            continue
                        by[(r['identity'], r['cand_id'])] = dict(stokes=st, tier=cad, identity=r['identity'],
                                                                 cand_id=r['cand_id'], band=r['band'], ts=r['ts'],
                                                                 snr=r['snr'], sigma_frame_jyb=r['sigma_frame_jyb'])
        out.extend(by.values())
    return out


def event_conflicts(runs):
    """[(Stokes, cadence, identity, cand_id)] of events read twice with different values (the build tree and the served tree
    of a run disagree): the order of reading would then decide the value - the stage refuses (W-EVENTS)."""
    bad = []
    for _label, st, cad, run in runs:
        seen = {}
        root = os.path.join(run, 'candidates')
        for p in sorted(glob.glob(os.path.join(root, '*', '*', 'detections.csv'))):
            with open(p, newline='') as f:
                for r in csv.DictReader(f):
                    if not r['band'].endswith('MHz') or not r.get('sigma_frame_jyb'):
                        continue
                    k = (r['identity'], r['cand_id'])
                    v = (r['band'], r['ts'], r['snr'], r['sigma_frame_jyb'])
                    if k in seen and seen[k] != v:
                        bad.append((st, cad) + k)
                    seen.setdefault(k, v)
    return bad


def write_events(path, rows):
    tmp = '%s.tmp.%d' % (path, os.getpid())
    with open(tmp, 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(EVENT_COLUMNS))
        wr.writeheader()
        wr.writerows(rows)
    os.replace(tmp, path)


def load_events(path):
    """{(Stokes, tier): {identity: [(band, ts, value Jy/beam)]}} - value = float(S/N) x float(frame sigma), as the analysis."""
    out = {}
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            out.setdefault((r['stokes'], r['tier']), collections.defaultdict(list))[r['identity']].append(
                (r['band'], r['ts'], float(r['snr']) * float(r['sigma_frame_jyb'])))
    return {k: dict(v) for k, v in out.items()}
