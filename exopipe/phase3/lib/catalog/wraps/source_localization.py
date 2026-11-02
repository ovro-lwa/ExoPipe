"""B-32 (a), Gregg 2026-10-13 (R-GREGG-20261013-REFACTOR-SHEET-DADMIT-B01-B02-B03-B05-B30-B32): "one position estimator
(inverse-variance mean with sandwich error, refusing a non-positive variance) and one error source (each detection's fitted
error) in every tier; a missing master beam or failing localization refuses."

The import wrap of the PRODUCTION module exopipe_ops/source_localization.py in every child that imports it (core/wrap.py;
core/parity_sitecustomize/sitecustomize.py) - build_localizations.py, p3s2_conform.py's pulsar groups, the astrometry panel,
clv15 phase3_snapshot_anchor.py and sublst_pertie.py all get the same estimator. SELF-CONTAINED: a child's `exopipe` can be
the production package (source_localization.py puts /lustre/gh/main/exopipe first on sys.path), so this file imports nothing
from the branch; core/fallback.py arrives as an argument (loaded by path by the finder).

Pin: the sha1 of the whole production FILE (its bytes) at exopipe_ops 70654c1 (source_localization.py, 2026-10-31 20:02; re-ported
2026-11-02 from 940bc26: RA ACROSS 0h - combine() places members that straddle 0h within 180 deg of the first and returns the RA in
[0, 360) only then; consistency / crossmatch / master_context / locate take every RA difference across 0h (_dra); crossmatch strips
the '(7.8 arcmin, ...)' statistics after the name.  The wrapped estimator below takes the same placement; locate is production's,
wrapped.  Before: re-ported 2026-10-30 from 43db654 - production's detections() reads the catalogs through survivor_paths /
ident_rows / member_rows, a row index that is off unless a caller enables it, the same rows in the same order) -
any edit refuses at import, so no decorator, second definition or module-level rebinding of a replaced function can slip
past (review MINOR 2); the per-function text pins below only name what changed in the refusal. A mismatch: production
changed - re-port this file (even a comment edit asks for that). Design: exopipe_ops/refactor_20261012/plan/wave2/B32_DESIGN.md.
"""
import hashlib
import math

FILE_SHA1 = 'c975d68c9f510e74f9bf3db3aafb34562b720773'               # source_localization.py at exopipe_ops 70654c1
PINS = {'combine': '93f9d36c5ee582b86f2dc0b9cdf5cbd0b56e570e',       # source_localization.py:291-329 (70654c1: across 0h)
        'detections': '55e147a0bfb38a5a9206c3b2302b39a3747f3804',    # :160-227 (940bc26: survivor_paths / ident_rows / member_rows)
        'beam_as': 'c3ba730f3a5a4bd898a2e2a6361b50df81956b35',       # :76-88
        'locate': '7479aeb7580b7ed222d7a3bf8e385f5ebff2a0ad'}        # :427-476 (70654c1: across 0h)


def fn_text(src, name):
    """The source text of top-level function ``name``: its `def` line to the line before the next top-level statement,
    trailing blank and comment lines dropped; None when absent."""
    lines = src.split('\n')
    start = next((i for i, ln in enumerate(lines) if ln.startswith(f'def {name}(')), None)
    if start is None:
        return None
    end = start + 1
    while end < len(lines) and (not lines[end] or lines[end][0] in ' \t#'):
        end += 1
    seg = lines[start:end]
    while seg and (not seg[-1].strip() or seg[-1].lstrip().startswith('#')):
        seg.pop()
    return '\n'.join(seg)


def fn_sha1(src, name):
    t = fn_text(src, name)
    return None if t is None else hashlib.sha1(t.encode()).hexdigest()


def check_pins(src, pins=None):
    """[names whose production source differs from the pin]."""
    pins = PINS if pins is None else pins
    return sorted(n for n, want in pins.items() if fn_sha1(src, n) != want)


def ivw_combine(mod):
    """combine() in every tier = the production 10-minute branch (source_localization.py combine, the _SP.active() form):
    w_i = (1 / s_i^2) / sum(1 / s^2) with s_i the detection's own error (+ its Master-frame term when frame=True), the
    position the w-weighted mean, the error sqrt(w^T V w) under the measured correlations; a non-positive variance REFUSES."""
    np = mod.np

    def combine(dets, rho_epoch=mod.RHO_EPOCH, rho_far=mod.RHO_FAR, frame=False):
        ras, moved = mod._ras(dets, frame)           # across 0h (production 70654c1): straddling members within 180 deg of the first
        ra0 = float(np.mean(ras))
        dec0 = float(np.mean([mod._pos(d, frame)[1] for d in dets]))
        cosd = math.cos(math.radians(dec0))
        x = np.array([(r - ra0) * cosd * 3600.0 for r in ras])
        y = np.array([(mod._pos(d, frame)[1] - dec0) * 3600.0 for d in dets])
        sig = np.array([d['sig'] for d in dets])
        V = mod._cov(dets, rho_epoch, rho_far, frame)
        s2 = sig ** 2 + (np.array([d['frame_sig'] for d in dets]) ** 2 if frame else 0.0)
        w = (1.0 / s2) / float(np.sum(1.0 / s2))
        var = float(w @ V @ w)
        if not (math.isfinite(var) and var > 0):
            raise RuntimeError(f'combined position: w^T V w = {var} is not a positive variance - REFUSING (B-32)')
        return (mod._ra360(ra0 + float(w @ x) / (cosd * 3600.0), moved), dec0 + float(w @ y) / 3600.0, math.sqrt(var),
                float((np.median(sig) ** 2) / var))
    combine.__doc__ = 'B-32 (a): the inverse-variance mean with the sandwich error in every tier (wrapped).'
    return combine


def fitted_detections(mod, fallback):
    """detections() with each detection's FITTED error: per cand_id the member row carrying a finite positive
    det_stat_err_arcsec (singles tiers) wins over a row without one (the multiples-fork tiers carry peak positions and no
    error - production keeps the FIRST row seen over the sorted globs). A detection with no fitted row anywhere: listed ONLY
    in multiples-fork tiers (_fsb / _fwb) it is left out of the combined position, counted once per identity
    (fallbacks.LOC_FORK_ONLY_FALLBACK; Gregg 2026-10-14, C.8b (b): until the fork builder fits its detections); with a
    singles-tier row it REFUSES (a missing fitted error). Otherwise production's code (pinned), step for step."""
    import os
    import re
    fork_dir = re.compile(r'_f(sb|wb)$')

    def detections(ident, U):
        srcs = []
        for p in mod.survivor_paths(U):              # production's own catalog readers (940bc26): its row index when enabled
            st = 'I' if '_I_' in os.path.basename(os.path.dirname(p)) + '_' else 'V'
            for r in mod.ident_rows(p, ident):
                r['_st'] = st
                r['_dir'] = os.path.dirname(p)
                srcs.append(r)
        rows = {}                                   # cand_id -> [(src, member row)] in production's order
        for s in srcs:
            mp = f"{s['_dir']}/{s['band']}_source_members.csv"
            if not os.path.exists(mp):
                continue
            for m in mod.member_rows(mp, s['source_id']):
                if m['band'] != s['band']:
                    continue
                rows.setdefault(m['cand_id'], []).append((s, m))

        def own_err(m):
            try:
                v = float(m.get('det_stat_err_arcsec') or 'nan')
            except (TypeError, ValueError):
                v = float('nan')
            return v if (math.isfinite(v) and v > 0) else None
        dets, fork_only = [], []
        for cid, cands in rows.items():
            pick = next(((s, m) for s, m in cands if own_err(m) is not None), None)
            if pick is None:
                if all(fork_dir.search(os.path.basename(s['_dir'])) for s, _ in cands):
                    fork_only.append(cid)                    # C.8b (b): left out, counted per identity below
                    continue
                raise RuntimeError(f'{ident}: detection {cid} has no fitted statistical error in any tier '
                                   f'({len(cands)} member row(s)) - REFUSING (B-32: each detection\'s fitted error)')
            s, m = pick
            ra, dec, snr = float(m['det_ra']), float(m['det_dec']), float(m['det_snr'])
            lst = m['cand_id'].split('_')[1]
            dd = m['det_date']
            win = int(dd.split('T')[1]) if 'T' in dd else None
            el = mod.elev(lst, ra, dec, win)
            stat, stat_src = own_err(m), 'own'
            try:
                own_sys = float(m.get('det_sys_err_arcsec') or 'nan')
            except (TypeError, ValueError):
                own_sys = float('nan')
            if math.isfinite(own_sys) and own_sys > 0:
                sysd, sys_src = own_sys, 'own'
            elif mod._SP.active():
                sysd, sys_src = mod._SP.sys_arcsec(m['cand_id']), 'window-tie'
            else:
                sc = mod.SYS_SCALE.get((s['_st'], mod.A.band_class(s['band']) if s['band'] != 'Full' else 'Full'), 1.0)
                sysd, sys_src = mod.A.sys_arcmin(s['band'], el) * 60.0 * sc, 'table'
            fr = {}
            if mod.COMMON_FRAME:
                fp = mod._AF.to_common(lst, s['band'], ra, dec)
                ft = mod._AF.term(lst, s['band'])
                fr = dict(ra_c=fp.ra, dec_c=fp.dec, frame_shift=fp.shift_arcsec, frame_mapped=fp.mapped,
                          frame_sig=ft.sigma_arcsec, frame_key=ft.key, frame_basis=ft.basis)
            dets.append(dict(cand_id=m['cand_id'], ra=ra, dec=dec, snr=snr, band=s['band'], stokes=s['_st'],
                             lst=lst, date=m['det_date'], el=el, stat=stat, stat_source=stat_src, sys=sysd,
                             sys_source=sys_src, sig=math.hypot(stat, sysd), beam=mod.beam_as(s['band']), **fr,
                             epoch=(lst, m['det_date'][:10])))
        if fork_only:
            fallback.site('fallbacks.LOC_FORK_ONLY_FALLBACK', f'{ident}: {len(fork_only)} detection(s) only in multiples-fork '
                          f'tiers, without a fitted error (e.g. {fork_only[:3]}), left out of the combined position '
                          f'({len(dets)} kept) - Gregg 2026-10-14 C.8b (b), until the fork fits its detections', n=1)
        return srcs, dets
    detections.__doc__ = 'B-32 (a): each detection with its fitted error (wrapped).'
    return detections


def beam_as_site(mod, fallback):
    """beam_as() with B-30's site where production takes 900 arcsec for a band without a Master (outside the strict tiers)."""
    def beam_as(band):
        if band not in mod._BEAM:
            for lst in [f'{h:02d}h' for h in range(24)]:
                p = mod.F.master_path(lst, band, 'I')
                if p and mod.os.path.exists(p):
                    mod._BEAM[band] = float(mod.fits.getheader(p)['BMAJ']) * 3600.0
                    break
            else:
                if mod.beam_strict():
                    raise RuntimeError(f'{band}: no Master found for its beam (roots {mod.F.ROOTS_CSV}) - REFUSING (no 900-arcsec '
                                       f'default in the 10-minute or snapshot tier; set P3S2_ROOTS_CSV to the run\'s Master roots)')
                fallback.site('fallbacks.LOC_MASTER_BEAM_FALLBACK', f'{band}: no Master found for its beam (roots '
                              f'{mod.F.ROOTS_CSV}) - the beam would be 900 arcsec')
                mod._BEAM[band] = 900.0
        return mod._BEAM[band]
    return beam_as


def locate_site(mod, fallback):
    """locate() recording a failing identity as B-30's LOCALIZATION site: sticky, so build_localizations.py's own except
    cannot hide it (the child's dump carries it to the parent's unit)."""
    orig = mod.locate

    def locate(*a, **k):
        try:
            return orig(*a, **k)
        except fallback.FallbackRefused:
            raise
        except Exception as e:                       # noqa: BLE001 - recorded, then raised as production raises it
            fallback.site('fallbacks.LOCALIZATION_FALLBACK', f'identity {a[0] if a else k.get("ident")}: '
                          f'{type(e).__name__}: {str(e)[:300]}')
            raise
    return locate


def check_file(path):
    """[] when the production file is the one this patch was written for (FILE_SHA1, its bytes - no locale decoding,
    review MINOR 3), else what changed: the pinned functions whose text differs, or 'the file' (an edit elsewhere)."""
    with open(path, 'rb') as f:
        raw = f.read()
    if hashlib.sha1(raw).hexdigest() == FILE_SHA1:
        return []
    return check_pins(raw.decode('utf-8', 'replace')) or ['the file (outside the pinned functions)']


def apply(mod, cfg, fallback):
    """Wrap ``mod`` (the production source_localization) under ``cfg`` ({key: value}); ``fallback`` = core/fallback.py."""
    bad = check_file(mod.__file__)
    if bad:
        raise RuntimeError(f'B-32 wrap of {mod.__file__}: production changed {bad} since the patch was written - REFUSING '
                           f'(re-port lib/catalog/wraps/source_localization.py)')
    if cfg.get('astrometry.LOC_ESTIMATOR') == 'ivw-sandwich':
        mod.combine = ivw_combine(mod)
    if cfg.get('astrometry.LOC_ERROR_SOURCE') == 'fitted':
        mod.detections = fitted_detections(mod, fallback)
    mod.beam_as = beam_as_site(mod, fallback)
    mod.locate = locate_site(mod, fallback)
    mod.B32_WRAPPED = dict(cfg)
    return mod
