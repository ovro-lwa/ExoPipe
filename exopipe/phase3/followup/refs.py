#!/usr/bin/env python3
"""candinv/refs.py — reference catalogs (VLSS, A-team, target/contaminant lists)."""
import csv
import os
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

from . import config as C

# A-team / bright low-frequency sources (search_v_transients_v2 CONFIG).
ATEAM = {
    'CasA': SkyCoord('23h23m24s', '+58d48m54s'),
    'CygA': SkyCoord('19h59m28s', '+40d44m02s'),
    'TauA': SkyCoord('05h34m31s', '+22d00m52s'),
    'VirA': SkyCoord('12h30m49s', '+12d23m28s'),
    'HydA': SkyCoord('09h18m05.7s', '-12d05m44s'),
}

_VLSS = None
_TARGETS = {}


def vlss_unresolved():
    """SkyCoord array of unresolved VLSS sources (astrometry references)."""
    global _VLSS
    if _VLSS is None:
        ra, dec, flux = [], [], []
        with open(C.VLSS_UNRESOLVED) as f:
            for line in f:
                p = line.split()
                if len(p) < 3:
                    continue
                try:
                    ra.append(float(p[0])); dec.append(float(p[1]))
                    flux.append(float(p[2]))
                except ValueError:
                    continue
        _VLSS = (SkyCoord(np.array(ra) * u.deg, np.array(dec) * u.deg),
                 np.array(flux))
    return _VLSS


def nearest_ateam(coord):
    best, bsep = None, 1e9
    for name, c in ATEAM.items():
        s = coord.separation(c).deg
        if s < bsep:
            best, bsep = name, s
    return best, float(bsep)


def _load_target_csv(path):
    """Load a target/source list. Flexible columns: name = common_name|name,
    RA = ra_current|ra|RA, Dec = dec_current|dec|Dec (decimal degrees), optional
    distance. utf-8-sig to tolerate a BOM (the *pc lists have one)."""
    names, ra, dec, dist = [], [], [], []
    with open(path, encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            rk = row.get('ra_current', row.get('ra', row.get('RA')))
            dk = row.get('dec_current', row.get('dec', row.get('Dec')))
            try:
                ra.append(float(rk)); dec.append(float(dk))
            except (TypeError, ValueError):
                continue
            names.append((row.get('common_name') or row.get('name') or '').strip())
            try:
                dist.append(float(row.get('distance', 'nan')))
            except (TypeError, ValueError):
                dist.append(float('nan'))
    return (names, SkyCoord(np.array(ra) * u.deg, np.array(dec) * u.deg),
            np.array(dist))


def target_catalog(key):
    """key in TARGET_CSVS / FLARE_STAR / ATNF. Cached; None if file absent."""
    if key in _TARGETS:
        return _TARGETS[key]
    path = C.TARGET_CSVS.get(key)
    if path is None:
        path = {'flare': C.FLARE_STAR_CSV, 'atnf': C.ATNF_CSV}.get(key)
    if not path or not os.path.exists(path):
        _TARGETS[key] = None
        return None
    try:
        _TARGETS[key] = _load_target_csv(path)
    except Exception:
        _TARGETS[key] = None
    return _TARGETS[key]


def crossmatch(coord, key, radius_arcmin):
    """Nearest catalog entry within radius. Returns (name, sep_arcmin, dist_pc)
    or None. Nearest-by-distance within the circle (caller decides ranking)."""
    cat = target_catalog(key)
    if cat is None:
        return None
    names, coords, dist = cat
    seps = coord.separation(coords).arcmin
    j = int(np.argmin(seps))
    if seps[j] <= radius_arcmin:
        return (names[j], float(seps[j]),
                float(dist[j]) if j < len(dist) else float('nan'))
    return None
