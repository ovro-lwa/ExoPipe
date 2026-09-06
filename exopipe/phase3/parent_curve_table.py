"""Per-Stokes, per-band bright-parent exclusion TABLE R(S) (R-2D-GATE-PER-POLARIZATION-PER-BAND-DEEP,
Gregg 2026-09-05; validated CARD_DEEP_2D_VALIDATION_20260905). The chain and the publish pass both
consume exopipe/common/parent_curve_tables.json: for key deep_<Stokes>_<band>, a list of
[S_lo, S_hi, R_beam_widths]; a candidate is removed when a parent of brightness S in [S_lo, S_hi)
lies within R beam widths. S = parent peak flux / the cell's frame global rms. No power-law
extrapolation: brightness beyond the last bin uses the last bin's radius (held constant); below the
faintest bin with an entry nothing is removed. Gregg 2026-09-06: removal of REPEATING sources by
this curve is fine (it catches the FPs of the repeating class); sigma_stack removal is not."""
import json, os
import numpy as np
from scipy.spatial import cKDTree
_TABLES = None
PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'common', 'parent_curve_tables.json')
def tables():
    global _TABLES
    if _TABLES is None:
        _TABLES = json.load(open(PATH))['tables']
    return _TABLES
def table_for(stokes, band):
    rec = tables().get(f'deep_{stokes}_{band}')
    return [tuple(t) for t in rec['table']] if rec else []
def _uv(ra, dec):
    r, d = np.radians(ra), np.radians(dec)
    return np.stack([np.cos(d) * np.cos(r), np.cos(d) * np.sin(r), np.sin(d)], -1)
def removals(stokes, band, parents_radec_flux, frame_rms_jy, beam_deg, ra, dec):
    """(removed: bool array, note: list[str]) for candidate positions ra/dec (deg).
    parents_radec_flux: array (N,3) of RA, Dec, peak flux (Jy) of the cell's I master catalog."""
    ra = np.atleast_1d(np.asarray(ra, float)); dec = np.atleast_1d(np.asarray(dec, float))
    cut = np.zeros(len(ra), bool); note = [''] * len(ra)
    tab = table_for(stokes, band)
    par = np.asarray(parents_radec_flux, float).reshape(-1, 3)
    if not tab or not len(par) or not (frame_rms_jy and frame_rms_jy > 0) or not (beam_deg and beam_deg > 0):
        return cut, note
    S = par[:, 2] / frame_rms_jy
    pos = _uv(ra, dec)
    s_top = max(t[1] for t in tab)
    for s_lo, s_hi, R in tab:
        m = (S >= s_lo) & ((S < s_hi) if s_hi < s_top else np.ones(len(S), bool))
        if not m.any():
            continue
        d, i = cKDTree(_uv(par[m, 0], par[m, 1])).query(pos, k=1)
        sep_bw = (2 * np.arcsin(np.clip(d / 2, 0, 1)) * 180 / np.pi) / beam_deg
        hit = sep_bw < R
        for j in np.where(hit & ~cut)[0]:
            note[j] = f'table: parent S={S[m][i[j]]:.0f} at {sep_bw[j]:.2f} bw < {R:.2f} (bin {s_lo:g}-{s_hi:g})'
        cut |= hit
    return cut, note
