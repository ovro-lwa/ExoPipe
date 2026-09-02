#!/usr/bin/env python3
"""candinv/resolve.py — locate the FITS a candidate needs, across three trees.

  bundle   /data07/gh/top50_allsky_tmp/rank{NN}_{lst}_{band}_{date}_{cadence}
           [_bin{iv}]_{Vresidual,Iresidual,Imaster}.fits   (discovery image —
           the ONLY surviving copy of deep wideband _clean residuals)
  PROD     /data07/gh/phase3/{lst}/{date}/P3_PROD/{scope}/{V,I}/{deep,10min}/
           {scope}_{S}_{cadence}_Taper_Robust+0.0_dewarped_{aligned,msub}_...fits
           (per-subband + 10-min spectra)
  phase2   /lustre/pipeline/exopipe/phase2/{lst}/{date}/Science_*/{scope}/{S}/
           {cadence}/...image[.pbcorr]_dewarped.fits   (sky images + epoch times)

All lookups are globs (timestamps vary) and return None when absent.
"""
import glob
import os

from astropy.io import fits

from . import config as C


# --- bundle (discovery products) -------------------------------------------
def _variant_token(path):
    """'Robust-0.75' or 'Robust-0' as written in a Phase-2 filename.

    Substring tests are unsafe here: 'Robust-0' is a prefix of
    'Robust-0.75', so `'Robust-0' in name` matches BOTH variants.
    """
    b = os.path.basename(path)
    return 'Robust-0.75' if 'Robust-0.75' in b else (
        'Robust-0' if 'Robust-0' in b else None)


def bundle_path(rank, lst, band, date, cadence, iv=None, kind='Vresidual'):
    binpart = f'_bin{int(iv)}' if (cadence == '10min' and iv not in (None, '')) else ''
    name = f'rank{int(rank):02d}_{lst}_{band}_{date}_{cadence}{binpart}_{kind}.fits'
    p = os.path.join(C.BUNDLE_DIR, name)
    return p if os.path.exists(p) else None


# --- PROD per-epoch residuals ----------------------------------------------
def prod_epoch_dir(lst, date, scope, stokes, cadence, label=None):
    return os.path.join(C.PROD_ROOT, lst, date, f'P3_{label or C.PROD_LABEL}',
                        scope, stokes, cadence)


def _labels_for(scope, cadence):
    """Which P3 label(s) to try, in order. WIDEBAND DEEP lives in REDO1 (PROD
    lacks it; project_rank48_and_prod_redo1); everything else in PROD. Fall back
    to the other label so a partial run still resolves what it has."""
    if scope in C.WIDEBANDS and cadence == 'deep':
        return [C.REDO1_LABEL, C.PROD_LABEL]
    return [C.PROD_LABEL, C.REDO1_LABEL]


def prod_residual(lst, date, scope, stokes, cadence, iv=None, kind='msub'):
    """kind in {'msub','aligned','nmsub'}. Returns one path or None. Wideband
    deep is sourced from REDO1, subbands + 10-min from PROD (auto-selected)."""
    if cadence == '10min' and iv not in (None, ''):
        pat = f'{scope}_{stokes}_10min_Taper_*_{kind}_LST{lst}_bin{int(iv)}_*.fits'
    elif cadence == 'deep':
        pat = f'{scope}_{stokes}_deep_Taper_*_{kind}_LST{lst}_*.fits'
    else:
        pat = f'{scope}_{stokes}_{cadence}_Taper_*_{kind}_LST{lst}_*.fits'
    for label in _labels_for(scope, cadence):
        d = prod_epoch_dir(lst, date, scope, stokes, cadence, label=label)
        hits = sorted(glob.glob(os.path.join(d, pat)))
        if hits:
            return hits[0]
    return None


def prod_tenmin_bins(lst, date, scope, stokes='V', kind='msub'):
    """Sorted list of (bin_index, path) for the 10-min residuals present."""
    d = prod_epoch_dir(lst, date, scope, stokes, '10min')
    out = []
    for p in glob.glob(os.path.join(
            d, f'{scope}_{stokes}_10min_Taper_*_{kind}_LST{lst}_bin*_*.fits')):
        base = os.path.basename(p)
        for tok in base.split('_'):
            if tok.startswith('bin') and tok[3:].isdigit():
                out.append((int(tok[3:]), p))
                break
    return sorted(out)


def prod_dates_for_lst(lst):
    """All dates with a PROD per-epoch tree at this LST (for the T4 escalation)."""
    base = os.path.join(C.PROD_ROOT, lst)
    out = []
    for d in sorted(glob.glob(os.path.join(base, '20*'))):
        if os.path.isdir(os.path.join(d, f'P3_{C.PROD_LABEL}')):
            out.append(os.path.basename(d))
    return out


# --- phase-2 sky images + epoch times --------------------------------------
def phase2_image(lst, date, scope, stokes, cadence, prefer_pbcorr=False):
    """A dewarped Phase-2 sky image (Taper, Robust-0) for the I counterpart and
    for sourcing conformant epoch times. Returns one path or None."""
    sci = sorted(glob.glob(os.path.join(C.PHASE2_ARCHIVE, lst, date,
                                        'Science_*', scope, stokes, cadence)))
    if not sci:
        sci = sorted(glob.glob(os.path.join(C.PHASE2_FAST, lst, date,
                                            'Science_*', scope, stokes, cadence)))
    for d in sci:
        # legacy subband: {scope}-{S}-Taper-...-image[.pbcorr]_dewarped.fits
        # wideband: {scope}_{S}_{cadence}_Taper_Robust-0_..._dewarped_...fits
        pats = []
        if prefer_pbcorr:
            pats += [f'{scope}*Taper*Robust-0*pbcorr*dewarped*.fits',
                     f'{scope}*Taper*Robust*0*pbcorr*dewarped*.fits']
        pats += [f'{scope}*Taper*Robust-0*dewarped*.fits',
                 f'{scope}*Taper*Robust*0*dewarped*.fits']
        for pat in pats:
            hits = [h for h in sorted(glob.glob(os.path.join(d, pat)))
                    if 'pbcorr' in h or not prefer_pbcorr]
            if hits:
                return hits[0]
    return None


def phase2_subband_image(lst, date, scope, stokes, cadence, iv=None):
    """Dewarped Phase-2 sky image for ONE subband/cadence/bin (gap-fill when a
    Phase-3 residual is absent). 10-min bins are chronological -> index by iv
    (lexical sort = chronological, timestamps zero-padded). Returns one path/None.
    Phase-2 images are NOT master-subtracted; safe for Stokes V (the V sky is
    sparse), use with care for I."""
    for base in (C.PHASE2_ARCHIVE, C.PHASE2_FAST):
        for d in sorted(glob.glob(os.path.join(base, lst, date, 'Science_*',
                                               scope, stokes, cadence))):
            imgs = [h for h in sorted(glob.glob(os.path.join(d, '*image_dewarped.fits')))
                    if 'pbcorr' not in os.path.basename(h)]
            # VARIANT SELECTION (fixed 2026-09-02). This returned imgs[0],
            # and lexical sort puts NoTaper-Robust-0 FIRST for Stokes I
            # ('NoTaper' < 'Taper') — never the survey variant, and the
            # variant most likely to hold a diverged CLEAN (values to 1e19
            # Jy/beam; see R-READMIT-CLEAN-DIVERGENCE and
            # R-SEARCH-SANITY-GATE-10X). Reading the wrong variant produced
            # three false alarms in one session before it was caught.
            # The transient survey uses I = Taper Robust-0.75 and
            # V = Taper Robust-0 (R-VARIANT-DOC; Gregg 2026-09-02) — select
            # that explicitly and fall back only if it is genuinely absent.
            _want = ('Robust-0.75' if stokes == 'I' else 'Robust-0')
            _survey = [h for h in imgs
                       if 'NoTaper' not in os.path.basename(h)
                       and _variant_token(h) == _want]
            if _survey:
                imgs = _survey
            if not imgs:
                continue
            if cadence == '10min' and iv is not None:
                return imgs[int(iv)] if 0 <= int(iv) < len(imgs) else None
            return imgs[0]
    return None


def v_spectrum_image(lst, date, scope, cadence, iv=None):
    """Stokes-V image for the spectrum/photometry: PROD/REDO1 master-subtracted
    residual if present, else the Phase-2 dewarped V sky image (V sky is sparse,
    so a sound fallback; project_rank48_and_prod_redo1). Returns (path, source)
    where source in {'prod_msub','phase2_v_sky', None}."""
    p = prod_residual(lst, date, scope, 'V', cadence, iv, 'msub')
    if p:
        return p, 'prod_msub'
    p = phase2_subband_image(lst, date, scope, 'V', cadence, iv)
    if p:
        return p, 'phase2_v_sky'
    return None, None


def epoch_times(lst, date, scope, cadence):
    """(start_iso, end_iso, inttime_s) from a Phase-2 V image's header, for
    conformant per-epoch header stamping. The _clean/master/bundle headers are
    incomplete and carry the co-add date, so time MUST come from Phase-2."""
    for s in ('V', 'I'):
        p = phase2_image(lst, date, scope, s, cadence)
        if p is None:
            continue
        try:
            h = fits.getheader(p)
            start = h.get('DATE-OBS')
            end = h.get('DATE-END')
            it = h.get('INTTIME', h.get('DURATION'))
            if start:
                return start, end, (float(it) if it is not None else None)
        except Exception:
            pass
    return None, None, None


# --- warp screens (T5 snapshot dewarping) ----------------------------------
def warp_screen(lst, date, scope):
    for base in (C.PHASE2_FAST, C.PHASE2_ARCHIVE):
        hits = sorted(glob.glob(os.path.join(
            base, lst, date, 'Science_*', 'Dewarp_Diagnostics', 'maps',
            f'{date}_{lst}_{scope}_warp_screen.fits')))
        if hits:
            return hits[0]
    return None


# --- scope iteration helpers -----------------------------------------------
def spectrum_scopes(cadence):
    """Scopes to sweep for the spectrum: subbands always (richest), widebands
    where present. Deep wideband per-epoch may be absent (only bundle has it)."""
    return C.SUBBANDS + C.WIDEBANDS
