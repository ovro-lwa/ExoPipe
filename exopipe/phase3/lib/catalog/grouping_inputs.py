"""Grouping inputs: design B's detection table (det, lib/catalog/grouping.py) from a run's catalogs, with the approved error
model (refactor item F; the research loaders harness.py v2b load() and harness_deep.py load(), ported).

10-MINUTE TIER (load_tenmin): the Stokes rows of master_catalog.csv, in file order, and the window tie of pertie_detections.csv.
  sigma = hypot(pos_stat_maj_arcsec, sys_arcmin x 60) - the fitted statistical error and the window tie's per-case systematic:
  the VALIDATED core (FINDINGS_20261026 items 1-6), not sublst_pertie.sys_arcsec (its window-position term makes it 2.3x too
  large for pair offsets); beam = pertie beam_arcmin.
DEEP TIER (load_deep): the Stokes rows of master_catalog.csv and the frames' beams (frame_beam_bmaj.csv, bmaj_deg x 60 per
  (Stokes, LST, band, date)). sigma = hypot(1.6 x pos_stat_maj_arcsec, 0 x pos_sys_arcsec) (config DEEP_STAT_SCALE /
  DEEP_SYS_SCALE: the table systematic is the scatter against the sky, ~3x too large for night-to-night pairs, FINDINGS item 10).
BOTH: positions mapped to the COMMON frame (Gregg 2026-10-08, RULING-GREGG-20261008-ASTROMETRY-MEASURE-DONT-MOVE-BOTH-TIERS
  item 4: cross-LST matches go through the dewarp screens, plus the screens' remaining scatter as one term per (LST, band)) by
  the in-place production astrom_frame (refactor window, DESIGN 11.1): to_common_many per (LST, band) - 10-minute members='auto'
  with the tier's measured frame scale (P3_TENMIN_PERTIE set for the call, as every 10-minute consumer runs it), deep
  members=None and no tier scale - and fsig = term(...).sigma_arcsec; then the elevation term (grouping.covariance).
  A detection without its fitted statistical error, its window-tie systematic (10-minute), its beam, or with a blank S/N
  REFUSES, naming it (the research loaders filled such values with medians; none of the approved inputs needed it).
The det keeps ra_raw / dec_raw (the catalog positions: the detections' fingerprints, grouping.fingerprints), frame_shift
(arcsec; NaN where the screen did not converge) and the astrom_frame generation tag (frame_generation) of the mapping, which
grouping.group_catalog records in its run summary.
SINGLE-THREAD USE: the mapping switches a process-wide environment flag for astrom_frame (P3_TENMIN_PERTIE); this module's
mappings are serialized (_FRAME_LOCK) and hold astrom_frame's lock, but another thread calling astrom_frame meanwhile could
see the flag.
"""
import collections
import contextlib
import csv
import os
import threading

import numpy as np

from exopipe.phase3.lib.catalog import grouping as G

csv.field_size_limit(10 ** 9)
# The common-frame mapping switches astrom_frame's 10-minute tier through a PROCESS-WIDE environment flag (P3_TENMIN_PERTIE), which
# astrom_frame reads outside its own lock (tier_scale_active). The mappings of this module are therefore serialized by this lock
# and hold astrom_frame's lock while the flag is switched; a FOREIGN thread calling astrom_frame meanwhile could still read the
# flag between its locked calls, so the loaders are for single-thread use (the catalog step loads one tier and Stokes at a time).
_FRAME_LOCK = threading.Lock()


def _rows(path, stokes):
    with open(path, newline='') as f:
        return [r for r in csv.DictReader(f) if r['stokes'] == stokes]


def _num(r, col, what, path):
    v = r.get(col)
    try:
        x = float(v)
    except (TypeError, ValueError):
        x = float('nan')
    if not np.isfinite(x):
        raise G.GroupingRefused(f'grouping inputs: {r.get("cand_id")} has no {what} ({col}={v!r}) in {path} - REFUSING')
    return x


def _opt(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float('nan')


def load_tenmin(master_catalog, pertie, stokes, common=True):
    """det of the 10-minute tier (harness.py v2b load()): master_catalog.csv rows of Stokes `stokes` + pertie_detections.csv"""
    sysd, beamd = {}, {}
    with open(pertie, newline='') as f:
        for r in csv.DictReader(f):
            if r.get('sys_arcmin') not in ('', None):
                sysd[r['cand_id']] = float(r['sys_arcmin']) * 60
            if r.get('beam_arcmin') not in ('', None):
                beamd[r['cand_id']] = float(r['beam_arcmin'])
    D = _rows(master_catalog, stokes)
    if not D:
        raise G.GroupingRefused(f'grouping inputs: no Stokes {stokes} rows in {master_catalog} - REFUSING')
    st = np.array([_num(r, 'pos_stat_maj_arcsec', 'fitted statistical error', master_catalog) for r in D])
    miss = [r['cand_id'] for r in D if r['cand_id'] not in sysd]
    if miss:
        raise G.GroupingRefused(f'grouping inputs: {len(miss)} detection(s) without a window-tie systematic in {pertie} '
                                f'(e.g. {miss[0]}) - REFUSING')
    miss = [r['cand_id'] for r in D if r['cand_id'] not in beamd]
    if miss:
        raise G.GroupingRefused(f'grouping inputs: {len(miss)} detection(s) without a beam in {pertie} (e.g. {miss[0]}) - REFUSING')
    sy = np.array([sysd[r['cand_id']] for r in D])
    det = dict(cand_id=np.array([r['cand_id'] for r in D]), ra=np.array([float(r['ra']) for r in D]),
               dec=np.array([float(r['dec']) for r in D]),
               snr=np.array([_num(r, 'snr', 'S/N', master_catalog) for r in D]), band=np.array([r['band'] for r in D]),
               lst=np.array([r['lst'] for r in D]), key=np.array([r['lst'] + '|' + r['date'] for r in D]),
               night=np.array([r['date'][:10] for r in D]), stat=st, sys=sy, sig=G.detection_sigma(st, sy, '10min'),
               stokes=stokes, tier='10min')
    det['beam'] = np.array([beamd[r['cand_id']] for r in D])
    return _finish(det, '10min', common, pertie=pertie)


def load_deep(master_catalog, frame_beam, stokes, common=True):
    """det of the deep tier (harness_deep.py load()): master_catalog.csv rows of Stokes `stokes` + frame_beam_bmaj.csv"""
    P = G.Params()
    beam = {}
    with open(frame_beam, newline='') as f:
        for r in csv.DictReader(f):
            if r['bmaj_deg'] not in ('', 'nan'):
                beam[(r['stokes'], r['lst'], r['band'], r['date'])] = float(r['bmaj_deg']) * 60
    D = _rows(master_catalog, stokes)
    if not D:
        raise G.GroupingRefused(f'grouping inputs: no Stokes {stokes} rows in {master_catalog} - REFUSING')
    st = np.array([_num(r, 'pos_stat_maj_arcsec', 'fitted statistical error', master_catalog) for r in D])
    if P.deep_k != 0.0:
        sy = np.array([_num(r, 'pos_sys_arcsec', 'systematic error', master_catalog) for r in D])
        sy_sig = sy
    else:                                             # K = 0: the table systematic does not enter (hypot(x, 0 x sys) = |x|);
        sy = np.array([_opt(r.get('pos_sys_arcsec')) for r in D])          # kept for information only
        sy_sig = np.where(np.isfinite(sy), sy, 0.0)
    miss = [r['cand_id'] for r in D if (stokes, r['lst'], r['band'], r['date']) not in beam]
    if miss:
        raise G.GroupingRefused(f'grouping inputs: {len(miss)} detection(s) without a frame beam in {frame_beam} '
                                f'(e.g. {miss[0]}) - REFUSING')
    det = dict(cand_id=np.array([r['cand_id'] for r in D]), ra=np.array([float(r['ra']) for r in D]),
               dec=np.array([float(r['dec']) for r in D]),
               snr=np.array([_num(r, 'snr', 'S/N', master_catalog) for r in D]), band=np.array([r['band'] for r in D]),
               lst=np.array([r['lst'] for r in D]), key=np.array([r['lst'] + '|' + r['date'] for r in D]),
               night=np.array([r['date'][:10] for r in D]), stat=st, sys=sy, sig=G.detection_sigma(st, sy_sig, 'deep', P),
               stokes=stokes, tier='deep')
    det['beam'] = np.array([beam[(stokes, r['lst'], r['band'], r['date'])] for r in D])
    return _finish(det, 'deep', common)


def _finish(det, tier, common, pertie=None):
    det['ra_raw'], det['dec_raw'] = det['ra'].copy(), det['dec'].copy()
    if common:
        det['ra'], det['dec'], det['frame_shift'], det['fsig'], det['frame_generation'] = common_frame(det, tier, pertie)
    det['el'], det['q'] = G.elevation_pa(det, tier)
    det['C'] = G.covariance(det, tier)
    return det


@contextlib.contextmanager
def _tier_env(tier, pertie):
    """the 10-minute tier's process flag for astrom_frame (P3_TENMIN_PERTIE = its window-tie table: the measured frame scale
    applies), set for the 10-minute mapping and unset for deep, restored afterwards. Process-wide: callers hold _FRAME_LOCK."""
    old = os.environ.get('P3_TENMIN_PERTIE')
    if tier == '10min':
        if not pertie:
            raise G.GroupingRefused('grouping inputs: the 10-minute common frame needs the window-tie table (pertie) - REFUSING')
        os.environ['P3_TENMIN_PERTIE'] = pertie
    else:
        os.environ.pop('P3_TENMIN_PERTIE', None)
    try:
        yield
    finally:
        if old is None:
            os.environ.pop('P3_TENMIN_PERTIE', None)
        else:
            os.environ['P3_TENMIN_PERTIE'] = old


def common_frame(det, tier, pertie=None):
    """(ra, dec, shift arcsec, per-axis Master-frame sigma arcsec, generation tag) of every detection in the COMMON frame:
    astrom_frame.to_common_many per (LST, band) and term(...).sigma_arcsec (10-minute: members='auto' and the tier's measured
    frame scale; deep: members=None, no tier scale). Single-thread use: see _FRAME_LOCK."""
    from exopipe.phase3.core import env
    env.legacy()
    import astrom_frame as AF
    ra, de = det['ra'].copy(), det['dec'].copy(); sh = np.zeros(len(ra)); fs = np.zeros(len(ra))
    keys = collections.defaultdict(list)
    for i, (l, b) in enumerate(zip(det['lst'], det['band'])):
        keys[(l[:3], b)].append(i)
    af_lock = getattr(AF, '_LOCK', None) or contextlib.nullcontext()     # astrom_frame's own (re-entrant) lock, when it has one
    with _FRAME_LOCK, af_lock, _tier_env(tier, pertie):
        for (l, b), idx in keys.items():
            idx = np.array(idx)
            if tier == '10min':
                r, d, s_, mp, note = AF.to_common_many(l, b, det['ra'][idx], det['dec'][idx])
                fs[idx] = AF.term(l, b).sigma_arcsec
            else:
                r, d, s_, mp, note = AF.to_common_many(l, b, det['ra'][idx], det['dec'][idx], members=None)
                fs[idx] = AF.term(l, b, members=None).sigma_arcsec
            ra[idx], de[idx], sh[idx] = r, d, s_
        gen = AF.generation()
    return ra, de, sh, fs, gen
