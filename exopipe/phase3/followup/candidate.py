#!/usr/bin/env python3
"""candinv/candidate.py — the Candidate record (one row of the top50 list)."""
import csv
from dataclasses import dataclass, field

from astropy.coordinates import SkyCoord
import astropy.units as u

from . import config as C


def iau_name(ra_deg, dec_deg):
    """IAU-style J-name JHHMMSS+DDMMSS (truncated, sign explicit)."""
    sc = SkyCoord(ra_deg * u.deg, dec_deg * u.deg)
    r = sc.ra.hms
    d = sc.dec.signed_dms
    sign = '+' if d.sign >= 0 else '-'
    return (f'J{int(r.h):02d}{int(r.m):02d}{int(r.s):02d}'
            f'{sign}{int(d.d):02d}{int(d.m):02d}{int(d.s):02d}')


@dataclass
class Candidate:
    rank: int
    ra: float
    dec: float
    band: str
    date: str
    cadence: str
    lst: str
    iv: object = None              # 10-min bin index (None for deep)
    peak_snr: float = float('nan')
    ndet: int = 0
    ndates: int = 0
    bands: str = ''
    nearest_Ateam: str = ''
    warning: str = ''
    raw: dict = field(default_factory=dict)

    @property
    def cand_id(self):
        # rank FIRST so every dossier dir + file sorts/groups by rank (Gregg
        # 2026-06-28). 2-digit zero-pad keeps lexical == numeric order.
        return f'rank{int(self.rank):02d}_{iau_name(self.ra, self.dec)}_LST{self.lst}'

    @property
    def skycoord(self):
        return SkyCoord(self.ra * u.deg, self.dec * u.deg)


def _to_int(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def from_row(row):
    iv = _to_int(row.get('iv')) if str(row.get('iv', '')).strip() != '' else None
    return Candidate(
        rank=int(float(row['rank'])), ra=float(row['ra']), dec=float(row['dec']),
        band=row['band'].strip(), date=row['date'].strip(),
        cadence=row['cadence'].strip(), lst=row['lst'].strip(), iv=iv,
        peak_snr=float(row.get('peak_snr', 'nan') or 'nan'),
        ndet=_to_int(row.get('ndet')) or 0,
        ndates=_to_int(row.get('ndates')) or 0,
        bands=row.get('bands', '').strip(),
        nearest_Ateam=row.get('nearest_Ateam', '').strip(),
        warning=row.get('WARNING', '').strip(), raw=dict(row))


def load_top50(path=None):
    path = path or C.TOP50_CSV
    with open(path) as f:
        return [from_row(r) for r in csv.DictReader(f)]


def get_rank(rank, path=None):
    for c in load_top50(path):
        if c.rank == int(rank):
            return c
    raise KeyError(f'rank {rank} not in {path or C.TOP50_CSV}')
