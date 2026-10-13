#!/usr/bin/env python3
"""
phase3_c50.py — MEASURED V 50%-completeness, no hard-wired sigma numbers.

The catalog's c50_mjy must come from a real blind injection-recovery measurement,
not a hand-typed constant (Gregg 2026-06-30). This module is the single read point:
it loads the band -> measured-C50 (in LOCAL-rms sigma, final gated chain) table that
phase3_v_c50.py writes, and returns None for any band that has NOT been measured so
the catalog leaves c50_mjy blank rather than guessing.

Measured table (JSON), written by phase3_v_c50.py:
  {"Blue": {"sigma_local": 6.70, "cell": "10h_2024-12-27", "n": 600,
            "method": "blind, mstruct-sole-gate, raw 50% crossing", "stamp": "..."}, ...}
"""
import os, json

MEASURED_FILE = '/data07/gh/phase3/transients/c50_measured.json'


def _load():
    try:
        with open(MEASURED_FILE) as fh:
            return json.load(fh)
    except Exception:
        return {}


def c50_sigma(band):
    """Measured final-gate C50 for `band` in LOCAL-rms sigma, or None if unmeasured.
    None -> the catalog writes a blank c50_mjy (never a hard-wired fallback)."""
    # 2026-10-13 (Gregg on M6: 'Fine' - blank now): the measured table predates the production gate chain (no v110, bright-parent table,
    # size or row-line gates), lives on one node's local disk (M4) and was applied to Stokes I from Stokes V injections (M5): every band
    # is UNMEASURED until a production-chain re-measurement replaces it, so the catalogue writes a blank c50_mjy
    return None
    rec = _load().get(band)
    if not rec:
        return None
    try:
        return float(rec['sigma_local'])
    except Exception:
        return None


def record(band, sigma_local, cell, n, method, stamp):
    """Persist a measured C50 for `band` (called by phase3_v_c50.py)."""
    os.makedirs(os.path.dirname(MEASURED_FILE), exist_ok=True)
    tbl = _load()
    tbl[band] = dict(sigma_local=round(float(sigma_local), 3), cell=cell,
                     n=int(n), method=method, stamp=stamp)
    tmp = MEASURED_FILE + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(tbl, fh, indent=2, sort_keys=True)
    os.replace(tmp, MEASURED_FILE)
    return tbl
