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
_RECORD_ONLY_SAID = set()
def table_for(stokes, band):
    """The table for THIS cadence (P3_CADENCE: deep default, or 10min). Contract step 74 (Gregg 2026-09-09 (d),
    2026-09-11 (c)): at 10 minutes the deep-calibrated table is NEVER applied (it over-excludes 10-15 per cent of
    bright I injections at that cadence); a missing 10-minute table REFUSES unless P3_TABLE_RECORD_ONLY=1 declares
    a record-only pass (the first production pass and the battery, whose tables are derived from the harvest
    afterwards and re-gated offline) - then nothing is removed and the choice is printed once per band."""
    cad = os.environ.get('P3_CADENCE', 'deep')
    key = f'{cad}_{stokes}_{band}'
    rec = tables().get(key)
    if rec:
        return [tuple(t) for t in rec['table']]
    if cad == 'deep':
        return []
    if os.environ.get('P3_TABLE_RECORD_ONLY') == '1':
        if key not in _RECORD_ONLY_SAID:
            _RECORD_ONLY_SAID.add(key)
            print(f'  [table] {key}: no 10-minute table - RECORD-ONLY pass (P3_TABLE_RECORD_ONLY=1): nothing removed; '
                  f'the table is derived from this harvest and applied offline', flush=True)
        return []
    raise RuntimeError(f'[table] {key}: no 10-minute bright-parent table and P3_TABLE_RECORD_ONLY is not set - '
                       f'REFUSING (contract step 74: the deep table is never applied at 10 minutes)')
def _uv(ra, dec):
    r, d = np.radians(ra), np.radians(dec)
    return np.stack([np.cos(d) * np.cos(r), np.cos(d) * np.sin(r), np.sin(d)], -1)
# CROSS-MATCH EXEMPTION (Gregg 2026-09-06 ~11:1x UT, 'Agreed - implement the cross-match exemption'): the
# table is a sky-area cut for the BLIND search; a position of a standing cross-match sample (Hot/Warm
# Jupiters, 10 pc, 25 pc, flare/UCD, ATNF pulsars, Sydney stars) inside a parent's exclusion zone is NOT
# removed - it is kept with the note 'table-exempt: xm <sample>:<name>' and judged by the cross-match
# E statistic and forced photometry instead. Radius: one beam FWHM of the cell. P3_TABLE_XM_EXEMPT=0 disables.
_XM = None
_XM_SAMPLES = ('Hot_Warm_Jupiters', '10pc', '25pc', 'flare_ucd', 'pulsars_atnf', 'sydney_stars')
_XM_DIR = '/lustre/gh/main/clv15'
def _xm_targets():
    global _XM
    if _XM is None:
        import csv
        pts, names = [], []
        for smp in _XM_SAMPLES:
            fp = os.path.join(_XM_DIR, f'{smp}.csv')
            if not os.path.exists(fp):
                continue
            with open(fp, encoding='utf-8-sig') as fh:
                for r in csv.DictReader(fh):
                    try:
                        pts.append((float(r['ra_current']), float(r['dec_current']))); names.append(f"{smp}:{r.get('common_name', '')}")
                    except (KeyError, TypeError, ValueError):
                        continue
        arr = np.asarray(pts, float).reshape(-1, 2)
        _XM = (cKDTree(_uv(arr[:, 0], arr[:, 1])) if len(arr) else None, names)
    return _XM
def xm_exempt(ra, dec, beam_deg):
    """(mask, names) of candidate positions within one beam FWHM of a standing-sample target."""
    ra = np.atleast_1d(np.asarray(ra, float)); dec = np.atleast_1d(np.asarray(dec, float))
    m = np.zeros(len(ra), bool); nm = [''] * len(ra)
    if os.environ.get('P3_TABLE_XM_EXEMPT', '1') == '0' or not (beam_deg and beam_deg > 0):
        return m, nm
    tree, names = _xm_targets()
    if tree is None:
        return m, nm
    chord = 2 * np.sin(np.radians(beam_deg) / 2)
    d, i = tree.query(_uv(ra, dec), k=1, distance_upper_bound=chord)
    for j in np.where(np.isfinite(d))[0]:
        m[j] = True; nm[j] = names[int(i[j])]
    return m, nm
import re as _re_mod
_re_S = _re_mod.compile(r'parent S=(\d+)')
XM_EXEMPT_SAMPLES = {'Hot_Warm_Jupiters', '10pc'}   # Gregg 2026-09-20
XM_EXEMPT_MAX_S = 20.0                              # Gregg 2026-09-20: master source must be below 20 sigma
def removals(stokes, band, parents_radec_flux, frame_rms_jy, beam_deg, ra, dec):
    """(removed: bool array, note: list[str]) for candidate positions ra/dec (deg).
    parents_radec_flux: array (N,3) of RA, Dec, peak flux (Jy) of the cell's I master catalog."""
    ra = np.atleast_1d(np.asarray(ra, float)); dec = np.atleast_1d(np.asarray(dec, float))
    cut = np.zeros(len(ra), bool); note = [''] * len(ra)
    tab = table_for(stokes, band)
    par = np.asarray(parents_radec_flux, float).reshape(-1, 3)
    # 2026-09-20 INCIDENT FIX (Gregg's 3C 219 / 3C 327 cases): PyBDSF parent lists carry Gaussians with NaN positions;
    # a cKDTree built on points containing NaN returns WRONG nearest neighbours silently, so every flux bin holding such a
    # row missed its real parents and the gate let candidates on bright sources through. Non-finite parents cannot be
    # used and are dropped here; the count is exposed for the caller's log.
    _fin = np.isfinite(par).all(axis=1)
    removals.n_nonfinite_parents = int((~_fin).sum())
    par = par[_fin]
    if not tab or not len(par) or not (frame_rms_jy and frame_rms_jy > 0) or not (beam_deg and beam_deg > 0):
        return cut, note
    S = par[:, 2] / frame_rms_jy
    okc = np.isfinite(ra) & np.isfinite(dec)
    pos = _uv(np.where(okc, ra, 0.0), np.where(okc, dec, 0.0))
    s_top = max(t[1] for t in tab)
    for s_lo, s_hi, R in tab:
        m = (S >= s_lo) & ((S < s_hi) if s_hi < s_top else np.ones(len(S), bool))
        if not m.any():
            continue
        d, i = cKDTree(_uv(par[m, 0], par[m, 1])).query(pos, k=1)
        sep_bw = (2 * np.arcsin(np.clip(d / 2, 0, 1)) * 180 / np.pi) / beam_deg
        hit = (sep_bw < R) & okc
        for j in np.where(hit & ~cut)[0]:
            note[j] = f'table: parent S={S[m][i[j]]:.0f} at {sep_bw[j]:.2f} bw < {R:.2f} (bin {s_lo:g}-{s_hi:g})'
        cut |= hit
    # R-XMATCH-EXEMPT-HWJ-10PC-ONLY-BELOW-20SIGMA (Gregg 2026-09-20, 'Proposed replacement approved'): a cross-match can
    # exempt a candidate from the table only when the parent that triggered the cut is below 20 sigma (S < 20, parent peak
    # over the frame rms) and only for the Hot/Warm Jupiter and 10 pc samples. Supersedes R-XMATCH-SAMPLE-POSITIONS-EXEMPT-
    # FROM-TABLE (2026-09-06: any of six samples, any parent brightness).
    xm, xn = xm_exempt(ra, dec, beam_deg)
    _S_hit = np.full(len(ra), np.nan)
    for j in np.where(cut)[0]:
        _m = _re_S.search(note[j]); _S_hit[j] = float(_m.group(1)) if _m else np.nan
    for j in np.where(cut & xm)[0]:
        _smp = xn[j].split(':', 1)[0]
        if _smp in XM_EXEMPT_SAMPLES and np.isfinite(_S_hit[j]) and _S_hit[j] < XM_EXEMPT_MAX_S:
            note[j] = f'table-exempt: xm {xn[j]}, parent S={_S_hit[j]:.0f} < {XM_EXEMPT_MAX_S:g} (would remove: {note[j]})'; cut[j] = False
        else:
            note[j] = f'{note[j]}; xm {xn[j]} NOT exempting (sample/parent-brightness rule 2026-09-20)'
    return cut, note
