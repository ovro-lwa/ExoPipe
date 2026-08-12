#!/usr/bin/env python3
"""
phase3_source_ids.py — STABLE position-based source IDs (Gregg 2026-07-10).

Ranks reshuffle on every list rebuild; IDs must not. Design: a persistent
REGISTRY (/data07/gh/phase3/transients/source_registry.csv). At catalog build
time each source is matched to the registry (within MATCH_ARCMIN, band-blind);
a match INHERITS the existing ID, otherwise a new ID is minted from the
discovery position (OVRO-LWA Jhhmm±ddmm; 'b','c'... suffix on collision) and
registered. IDs are shared between the Stokes I and V catalogs, so the same
sky source carries one name everywhere. Batch implementation (vectorized
match against the registry, then incremental numpy match among the new ones).
"""
import os, csv, fcntl
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

REGISTRY = '/data07/gh/phase3/transients/source_registry.csv'
MATCH_ARCMIN = 5.0      # ~2-3x the survey astrometric scatter (worst band)


def _mint(ra, dec, taken):
    c = SkyCoord(ra * u.deg, dec * u.deg)
    h = c.ra.hms
    d = c.dec.signed_dms
    base = (f'J{int(h.h):02d}{int(h.m):02d}'
            f'{"+" if d.sign >= 0 else "-"}{int(d.d):02d}{int(d.m):02d}')
    if base not in taken:
        return base
    for suf in 'bcdefghijk':
        if base + suf not in taken:
            return base + suf
    return base + 'x'


def assign_ids(sources, origin=''):
    """sources: list of dicts with 'ra'/'dec' (deg). Adds 'source_id' to each
    IN PLACE, updates the registry (flock-guarded). Sources within
    MATCH_ARCMIN of each other in one batch share one ID (first wins).
    Returns the number of newly minted IDs."""
    if not sources:
        return 0
    lk = open(REGISTRY + '.lock', 'w')
    fcntl.flock(lk, fcntl.LOCK_EX)
    try:
        reg = list(csv.DictReader(open(REGISTRY))) if os.path.exists(REGISTRY) else []
        taken = {r['source_id'] for r in reg}
        sra = np.array([float(s['ra']) for s in sources])
        sdc = np.array([float(s['dec']) for s in sources])
        sc = SkyCoord(sra * u.deg, sdc * u.deg)
        sid = [None] * len(sources)
        if reg:
            rc = SkyCoord([float(r['ra']) for r in reg] * u.deg,
                          [float(r['dec']) for r in reg] * u.deg)
            idx, d2d, _ = sc.match_to_catalog_sky(rc)
            ok = d2d.arcmin <= MATCH_ARCMIN
            for i in np.where(ok)[0]:
                sid[i] = reg[int(idx[i])]['source_id']
        # mint for the rest, clustering within the batch via flat-sky numpy
        new_rows = []
        nra, ndc, nid = [], [], []
        cosd = np.cos(np.radians(sdc))
        order = np.argsort(-np.array([float(s.get('snr', 0) or 0)
                                      for s in sources]))
        for i in order:
            if sid[i] is not None:
                continue
            if nra:
                d = np.hypot((np.array(nra) - sra[i]) * cosd[i],
                             np.array(ndc) - sdc[i]) * 60.0
                j = int(np.argmin(d))
                if d[j] <= MATCH_ARCMIN:
                    sid[i] = nid[j]
                    continue
            s_id = _mint(sra[i], sdc[i], taken)
            taken.add(s_id)
            sid[i] = s_id
            nra.append(sra[i]); ndc.append(sdc[i]); nid.append(s_id)
            new_rows.append({'source_id': s_id, 'ra': f'{sra[i]:.5f}',
                             'dec': f'{sdc[i]:.5f}', 'origin': origin})
        for s, i_d in zip(sources, sid):
            s['source_id'] = i_d
        if new_rows or not os.path.exists(REGISTRY):
            with open(REGISTRY, 'w', newline='') as fh:
                w = csv.DictWriter(fh, fieldnames=['source_id', 'ra', 'dec',
                                                   'origin'])
                w.writeheader()
                w.writerows(reg + new_rows)
        return len(new_rows)
    finally:
        fcntl.flock(lk, fcntl.LOCK_UN)
        lk.close()
