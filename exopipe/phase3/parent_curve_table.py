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
        # SUB-LST TABLES (Gregg 2026-09-27: derived for the cube-based sub-LST chain, with elevation bands) live in their
        # own file, named by P3_TABLE_EXTRA and merged here; a named file that is missing REFUSES.  Unset = unchanged.
        _extra = os.environ.get('P3_TABLE_EXTRA', '')
        if _extra:
            if not os.path.exists(_extra):
                raise RuntimeError(f'[table] P3_TABLE_EXTRA={_extra} does not exist - REFUSING')
            _TABLES = dict(_TABLES); _TABLES.update(json.load(open(_extra))['tables'])
    return _TABLES
_RECORD_ONLY_SAID = set()
def table_for(stokes, band):
    """The table for THIS cadence (P3_CADENCE: deep default, or 10min). Contract step 74 (Gregg 2026-09-09 (d),
    2026-09-11 (c)): at 10 minutes the deep-calibrated table is NEVER applied (it over-excludes 10-15 per cent of
    bright I injections at that cadence); a missing 10-minute table REFUSES unless P3_TABLE_RECORD_ONLY=1 declares
    a record-only pass (the first production pass and the battery, whose tables are derived from the harvest
    afterwards and re-gated offline) - then nothing is removed and the choice is printed once per band."""
    cad = os.environ.get('P3_CADENCE', 'deep')
    key = f"{os.environ.get('P3_TABLE_PREFIX') or cad}_{stokes}_{band}"   # P3_TABLE_PREFIX=sublst (2026-09-27); unset = cadence
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
    pos = _uv(ra, dec)
    d, i = tree.query(pos, k=1, distance_upper_bound=chord)
    for j in np.where(np.isfinite(d))[0]:
        m[j] = True; nm[j] = names[int(i[j])]
        # 2026-09-27 TIE FIX: the same star sits at an IDENTICAL position in several samples (10 pc stars are also 25 pc
        # rows), and k=1 returned whichever copy the tree yielded - so rebuilding Hot_Warm_Jupiters.csv (2026-09-26)
        # silently flipped Struve 2398 A and Gaia DR3 2091177593123254016 from their 10 pc copy (exempting) to their
        # 25 pc copy (not exempting). Among targets at the nearest distance, an exempting sample wins; a farther target
        # never displaces the nearest one (that would widen the 2026-09-20 rule).
        ties = [k for k in tree.query_ball_point(pos[j], d[j] + 1e-9)
                if names[k].split(':', 1)[0] in XM_EXEMPT_SAMPLES]
        if ties:
            nm[j] = names[ties[0]]
    return m, nm
import re as _re_mod
_re_S = _re_mod.compile(r'parent S=(\d+)')
# Gregg 2026-09-28: "No need to have the bright source exemption for any sample aside from pulsars" - the HWJ and 10 pc
# exemptions of 2026-09-20 are withdrawn (those samples are measured by forced photometry instead); pulsars keep their
# own exemption below. Cross-matches are still NOTED on the removal ('NOT exempting').
XM_EXEMPT_SAMPLES = set()
XM_EXEMPT_MAX_S = 20.0                              # Gregg 2026-09-20: master source must be below 20 sigma
ELEV_BANDS = [tuple(float(x) for x in b.split('-')) for b in os.environ.get('P3_TABLE_ELEV_BANDS', '25-35,35-50,50-90').split(',')]


def removals(stokes, band, parents_radec_flux, frame_rms_jy, beam_deg, ra, dec, elev=None, _tab=None):
    """(removed: bool array, note: list[str]) for candidate positions ra/dec (deg).
    parents_radec_flux: array (N,3) of RA, Dec, peak flux (Jy) of the cell's I master catalog.
    ELEVATION BANDS (Gregg 2026-09-27, sub-LST: 25-35, 35-50, >50 deg): with P3_TABLE_ELEV=1 and elev (deg, one per
    candidate) given, each candidate uses the table <prefix>_<Stokes>_<band>_el<lo>-<hi> of its elevation band when
    that table was installed, else the all-elevation table (the install step leaves out an elevation table that
    measured no bin above the 20-candidate excess guard - the fallback Gregg approved).  Default: unchanged."""
    ra = np.atleast_1d(np.asarray(ra, float)); dec = np.atleast_1d(np.asarray(dec, float))
    if _tab is None and elev is not None and os.environ.get('P3_TABLE_ELEV') == '1':
        elev = np.atleast_1d(np.asarray(elev, float))
        cut = np.zeros(len(ra), bool); note = [''] * len(ra)
        pre = os.environ.get('P3_TABLE_PREFIX') or os.environ.get('P3_CADENCE', 'deep')
        base = table_for(stokes, band)
        lab = np.array(['all'] * len(ra), dtype=object)
        for lo, hi in ELEV_BANDS:
            lab[(elev >= lo) & (elev < hi)] = 'el%g-%g' % (lo, hi)
        for L in sorted(set(lab)):
            idx = np.where(lab == L)[0]
            rec = tables().get(f'{pre}_{stokes}_{band}_{L}') if L != 'all' else None
            tab = [tuple(t) for t in rec['table']] if rec else base
            c_, n_ = removals(stokes, band, parents_radec_flux, frame_rms_jy, beam_deg, ra[idx], dec[idx], _tab=tab)
            for j, cj, nj in zip(idx, c_, n_):
                cut[j] = bool(cj); note[j] = (f'[{L if rec else "all-elevation"}] ' + nj) if nj else ''
        return cut, note
    cut = np.zeros(len(ra), bool); note = [''] * len(ra)
    tab = _tab if _tab is not None else table_for(stokes, band)
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
            note[j] = f'{note[j]}; xm {xn[j]} NOT exempting (only pulsars exempt, Gregg 2026-09-28)'
    # PULSARS ARE KEPT (Gregg 2026-09-24: parent scintillation and intrinsic parent variability may be excluded, "although
    # pulsars in the latter category are welcome to include"). A candidate the table would remove is kept when an ATNF
    # pulsar lies within one beam of the target image - at ANY parent brightness, since the pulsar is the parent. Searched
    # separately from xm_exempt, which reports only the nearest standing-sample target.
    if cut.any():
        pt, pn = _psr_targets()
        if pt is not None:
            chord = 2 * np.sin(np.radians(beam_deg) / 2)
            for j in np.where(cut & okc)[0]:
                k = pt.query_ball_point(pos[j], chord)
                if k:
                    note[j] = f'table-exempt: ATNF pulsar {pn[k[0]]} within 1 beam (Gregg 2026-09-24; would remove: {note[j]})'
                    cut[j] = False
    return cut, note


_PSR = None
def _psr_targets():
    """(cKDTree of unit vectors, names) of the ATNF pulsar sample (clv15/pulsars_atnf.csv)."""
    global _PSR
    if _PSR is None:
        import csv
        fp = os.path.join(_XM_DIR, 'pulsars_atnf.csv')
        pts, names = [], []
        if os.path.exists(fp):
            for r in csv.DictReader(open(fp)):
                try:
                    pts.append((float(r['ra_current']), float(r['dec_current']))); names.append(r['common_name'])
                except (KeyError, ValueError):
                    pass
        _PSR = (cKDTree(_uv(np.array([p[0] for p in pts]), np.array([p[1] for p in pts]))), names) if pts else (None, [])
    return _PSR
