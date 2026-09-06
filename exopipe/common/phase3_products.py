#!/usr/bin/env python3
"""
phase3_products.py — Single source of truth for Phase 3 product naming,
FITS headers, and provenance.

WHY THIS MODULE EXISTS
----------------------
Phase 2 left three naming grammars in the archive and incomplete headers
on stacked products (see project_header_standard memory). Phase 3 products
feed period-folding searches for exoplanet radio emission, where a
half-duration timestamp error smears the science (the v1 header bug).
Everything Phase 3 writes therefore goes through this module:

  build_name()/parse_name()      the Phase 3 naming grammar (plus parsers
                                 for both legacy Phase 2 grammars)
  write_standard_header()        the full header standard + write-time
                                 validation gate
  append_provenance_table()      binary-table provenance, one row per
                                 constituent file of a co-add

THE PHASE 3 GRAMMAR
-------------------
    {SCOPE}_{STOKES}_{CADENCE}_{TAPER}_{ROBUST}[_{PROC}...]_{TIMEID}.fits

  SCOPE    18MHz..82MHz (subbands), Red, Green, Blue, Full
  STOKES   I, V (Q, U accepted for completeness)
  CADENCE  deep, 10min, snap
  TAPER    Taper, NoTaper
  ROBUST   'Robust+0.0' or 'Robust-0.75' — the sign is ALWAYS explicit.
           Underscore is the ONLY field delimiter; a hyphen only ever
           means minus (this removes the legacy ambiguity where
           'Robust-0' meant robust=0 but 'Robust-0.75' meant -0.75).
  PROC     tokens in processing order, present only when applied:
           pbcorr, dewarped, aligned, coadd, msub[-cNN], ssub, sigstack
           ('msub-c01' ties a subtraction to co-add generation 01;
           'sigstack' marks the ensemble per-pixel noise-structure map,
           added beyond the original token list as a first-class product)
  TIMEID   single epoch:        LST10h_20241229T081959
                                (UTC START of first integration, NOT the
                                 midpoint — legacy timestamps are midpoints)
           10-min interval:     LST10h_bin2_20241229T082951
                                (binN is the LST sub-hour bin computed from
                                 LST arithmetic, NOT file order)
           multi-date co-add:   LST10h_20241218-20250105_N017
           10-min bin master:   LST10h_bin2_20241218-20250105_N017
  Cutouts prefix the target name and append '_cutout':
           GJ486_Green_I_deep_Taper_Robust+0.0_..._cutout.fits

LEGACY GRAMMARS (parse-only, plus exact rebuilders for round-trip tests)
  Phase 2 subband (hyphen-delimited, MIDPOINT timestamps):
    FREQ-STOKES-TAPER-DURATIONs-ROBUST-TIMESTAMP-image[.pbcorr][_dewarped].fits
    e.g. 55MHz-I-Taper-3571s-Robust-0.75-20241229_114918-image.pbcorr_dewarped.fits
  Phase 2 wideband:
    BAND_STOKES_CADENCE_TAPER_ROBUST[_pbcorr]_dewarped_YYYYMMDD_LSTxxh[_tNNNN].fits
    e.g. Full_I_deep_Taper_Robust-0_dewarped_20241229_LST07h.fits
"""
import os
import re
import numpy as np
from astropy.io import fits
from astropy.time import Time

from exopipe.common import phase2_config as cfg

# ---------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------
SCOPES = list(cfg.ALL_SUBBANDS) + ['Red', 'Green', 'Blue', 'Full']
STOKES_VALUES = ['I', 'V', 'Q', 'U']
CADENCES = ['deep', '10min', 'snap']
TAPERS = ['Taper', 'NoTaper']
# Canonical processing order. Build emits tokens in this order; parse
# accepts them in any order but preserves what it saw.
# 'loo' (leave-one-out master, tag = excluded date) and tagged 'ssub'
# (tag = striation window) added per ratified structure spec S15
# (R-P3-STRUCTURE-RATIFIED, 2026-08-26).
# 'shflux' (SH12 absolute flux scale baked in) and 'mdewarp' (final master dewarp) added per
# MASTER_SHFLUX_VLSSR_DEWARP_SPEC_20260828 (ratified 2026-08-28; implemented 2026-09-06).
PROC_ORDER = ['vclean', 'mstruct', 'loo', 'shflux', 'mdewarp', 'pbcorr', 'dewarped', 'aligned',
              'psfmatch', 'coadd', 'msub', 'nmsub', 'ssub', 'sigstack',
              'mosaic', 'jysr', 'feather', 'tcorr', 'lsfill',
              'psfmap', 'rgbcomp', 'molrender', 'carrender', 'pbmap']
# Mosaic extension (agreed with the mosaic thread 2026-07-21): vclean/
# mstruct = input-basis modifiers (lead); mosaic..lsfill = the HEALPix
# all-sky chain; psfmap/rgbcomp/molrender/carrender = first-class mosaic
# products (sigstack-style). Mosaic TIMEID uses the 'allsky' cell token.
# psfmatch = symmetric cross-convolution subtraction (D = G_m(x)epoch −
# G_e(x)master, per-pair dilution renorm) — ratified into the production
# Stokes-I chain 2026-07-05 (§5.14e); sits between alignment and msub.

# New-grammar robust REQUIRES a decimal point ('Robust+0.0'), which
# makes it impossible to confuse with the legacy 'Robust-0' (where the
# hyphen was a delimiter and meant robust = +0).
_RE_ROBUST = re.compile(r'^Robust([+-]\d+\.\d+)$')
# nmsub = native-beam msub: epoch at NATIVE resolution minus a
# per-epoch master built from other epochs matched DOWN to it —
# maximum point-source sensitivity (canonical msub trades SNR for
# PSF uniformity; see 2026-06-11 sensitivity discussion).
_RE_MSUB = re.compile(r'^n?msub(?:-c(\d+))?$')
_RE_LST = re.compile(r'^LST(\d{1,2}h)$')
_RE_BIN = re.compile(r'^bin(\d+)$')
_RE_EPOCH = re.compile(r'^(\d{8}T\d{6})$')
_RE_RANGE = re.compile(r'^(\d{8})-(\d{8})$')
_RE_NEP = re.compile(r'^N(\d{3,})$')

# Legacy subband: FREQ-STOKES-TAPER-DURs-ROBUST-TIMESTAMP-PRODUCT...
_RE_LEGACY_SUBBAND = re.compile(
    r'^(\d{2}MHz)-([IQUV])-(NoTaper|Taper)-(\d+)s-Robust-([\d.]+)-'
    r'(\d{8}_\d{6})-(image|dirty|residual|model|psf)'
    r'(\.pbcorr)?(_dewarped)?\.fits$')

# Legacy wideband: BAND_STOKES_CADENCE_TAPER_Robust-X[_pbcorr]_dewarped_
#                  YYYYMMDD_LSTxxh[_tNNNN].fits
_RE_LEGACY_WIDEBAND = re.compile(
    r'^(Red|Green|Blue|Full)_([IQUV])_(deep|10min|snap)_(NoTaper|Taper)_'
    r'Robust-([\d.]+)(_pbcorr)?(_dewarped)?_(\d{8})_LST(\d{1,2}h)'
    r'(?:_t(\d{4}))?\.fits$')


def _fmt_robust(value):
    """'Robust+0.0' / 'Robust-0.75' — explicit sign, >=1 decimal place."""
    s = f"{float(value):+.2f}".rstrip('0')
    if s.endswith('.'):
        s += '0'
    return f"Robust{s}"


def _legacy_robust_to_float(magnitude_str):
    """Map a legacy 'Robust-X' magnitude to a signed robust value.

    In the legacy grammars the hyphen after 'Robust' is a DELIMITER, and
    the magnitude carries an implicit sign: 'Robust-0' means briggs 0,
    'Robust-0.75' means briggs -0.75 (the only negative value used).
    Phase 1 only ever imaged at robust 0 and -0.75, so: zero stays zero,
    anything non-zero is negative.
    """
    mag = float(magnitude_str)
    return 0.0 if mag == 0.0 else -mag


def _epoch_token(start):
    """UTC start time -> 'YYYYMMDDTHHMMSS' (truncated to whole seconds)."""
    if isinstance(start, str):
        if re.match(r'^\d{8}T\d{6}$', start):
            return start
        start = Time(start, scale='utc')
    isot = start.isot  # YYYY-MM-DDTHH:MM:SS.sss
    return isot[:19].replace('-', '').replace(':', '')


# ---------------------------------------------------------------------
# build_name
# ---------------------------------------------------------------------
def build_name(scope, stokes, cadence, taper, robust, proc=(),
               lst=None, lst_bin=None, epoch=None,
               date_start=None, date_end=None, n_epochs=None,
               target=None, cutout=False, ext='.fits', **_ignored):
    """Build a Phase 3 product filename.

    Parameters
    ----------
    scope : str        One of SCOPES ('18MHz'..'82MHz', Red/Green/Blue/Full).
    stokes : str       'I' or 'V'.
    cadence : str      'deep', '10min', 'snap'.
    taper : str        'Taper' or 'NoTaper'.
    robust : float     Briggs robust value; sign rendered explicitly.
    proc : sequence    Processing tokens actually applied, e.g.
                       ['pbcorr', 'dewarped', 'aligned', 'msub-c01'].
                       Emitted in canonical PROC_ORDER regardless of the
                       order given (filenames must be deterministic).
    lst : str          'LST10h' or '10h'.
    lst_bin : int      Optional LST sub-hour bin (10-min products). The
                       bin index must come from LST arithmetic on the
                       start time, never from file ordering.
    epoch : Time/str   UTC START of the first integration (single-epoch
                       products). Accepts astropy Time, ISO string, or a
                       ready 'YYYYMMDDTHHMMSS' token.
    date_start/date_end : str  'YYYYMMDD' or 'YYYY-MM-DD' for multi-date
                       co-adds (mutually exclusive with epoch).
    n_epochs : int     Number of constituent epochs in a co-add.
    target : str       Cutout target name prefix (sanitized: no '_').
    cutout : bool      Append '_cutout' before the extension.

    Extra keys (e.g. 'grammar' from parse_name output) are ignored so
    that ``build_name(**parse_name(n)) == n`` round-trips directly.
    """
    if scope not in SCOPES:
        raise ValueError(f"Unknown scope '{scope}' (expected one of {SCOPES})")
    if stokes not in STOKES_VALUES:
        raise ValueError(f"Unknown stokes '{stokes}'")
    if cadence not in CADENCES:
        raise ValueError(f"Unknown cadence '{cadence}'")
    if taper not in TAPERS:
        raise ValueError(f"Unknown taper '{taper}'")

    # Processing tokens in canonical order (msub may carry a generation tag)
    def _proc_key(tok):
        base = tok.split('-')[0]
        if base not in PROC_ORDER:
            raise ValueError(f"Unknown proc token '{tok}'")
        if base not in ('msub', 'nmsub', 'ssub', 'loo') and tok != base:
            raise ValueError(f"Only msub/nmsub may carry a tag, got '{tok}'")
        return PROC_ORDER.index(base)
    proc_sorted = sorted(proc, key=_proc_key)

    # TIMEID
    if lst is None:
        raise ValueError("lst is required")
    if lst == 'allsky':
        pass                       # mosaic footprint token, used verbatim
    else:
        lst = lst if lst.startswith('LST') else f'LST{lst}'
        if not _RE_LST.match(lst):
            raise ValueError(f"Bad LST token '{lst}'")
    timeid = [lst]
    if lst_bin is not None:
        timeid.append(f'bin{int(lst_bin)}')
    if epoch is not None and (date_start or date_end):
        raise ValueError("Give either epoch or date_start/date_end, not both")
    if epoch is not None:
        timeid.append(_epoch_token(epoch))
    elif date_start and date_end:
        d0 = str(date_start).replace('-', '')
        d1 = str(date_end).replace('-', '')
        if not (re.match(r'^\d{8}$', d0) and re.match(r'^\d{8}$', d1)):
            raise ValueError(f"Bad date range '{date_start}'..'{date_end}'")
        if n_epochs is None:
            raise ValueError("Multi-date co-adds require n_epochs")
        timeid.append(f'{d0}-{d1}')
        timeid.append(f'N{int(n_epochs):03d}')
    else:
        raise ValueError("TIMEID needs epoch or (date_start, date_end)")

    parts = [scope, stokes, cadence, taper, _fmt_robust(robust)]
    parts += proc_sorted
    parts += timeid
    name = '_'.join(parts)
    if target:
        clean = re.sub(r'[\s_]+', '', str(target))
        name = f'{clean}_{name}'
    if cutout:
        name += '_cutout'
    return name + ext


# ---------------------------------------------------------------------
# parse_name
# ---------------------------------------------------------------------
def parse_name(filename):
    """Parse a product filename in any of the three known grammars.

    Returns a dict with 'grammar' in {'phase3', 'phase2_subband',
    'phase2_wideband'} plus the grammar's fields, or raises ValueError.

    For 'phase3' the dict feeds straight back into build_name(**fields).
    Legacy results carry enough to rebuild the original name exactly
    (build_legacy_*_name) and to translate forward; legacy timestamps
    are interval MIDPOINTS and are returned as 'timestamp_mid' — never
    use them as start times (project_time_semantics).
    """
    base = os.path.basename(filename)

    m = _RE_LEGACY_SUBBAND.match(base)
    if m:
        dur = int(m.group(4))
        if dur >= 1800:
            cadence = 'deep'
        elif dur >= 60:
            cadence = '10min'
        else:
            cadence = 'snap'
        return {
            'grammar': 'phase2_subband',
            'scope': m.group(1), 'stokes': m.group(2), 'taper': m.group(3),
            'duration_s': dur, 'cadence': cadence,
            'robust': _legacy_robust_to_float(m.group(5)),
            'robust_token': f'Robust-{m.group(5)}',
            'timestamp_mid': m.group(6),       # MIDPOINT, not start!
            'product': m.group(7),
            'pbcorr': m.group(8) is not None,
            'dewarped': m.group(9) is not None,
        }

    m = _RE_LEGACY_WIDEBAND.match(base)
    if m:
        return {
            'grammar': 'phase2_wideband',
            'scope': m.group(1), 'stokes': m.group(2), 'cadence': m.group(3),
            'taper': m.group(4),
            'robust': _legacy_robust_to_float(m.group(5)),
            'robust_token': f'Robust-{m.group(5)}',
            'pbcorr': m.group(6) is not None,
            'dewarped': m.group(7) is not None,
            'date': m.group(8),                # observing date (UTC day)
            'lst': m.group(9),
            'tindex': int(m.group(10)) if m.group(10) else None,
        }

    # --- Phase 3 grammar ---
    if not base.endswith('.fits'):
        raise ValueError(f"Not a .fits filename: '{base}'")
    stem = base[:-5]
    cutout = stem.endswith('_cutout')
    if cutout:
        stem = stem[:-len('_cutout')]
    tokens = stem.split('_')

    # Optional cutout target prefix: tokens before the first valid SCOPE
    target = None
    if tokens and tokens[0] not in SCOPES:
        try:
            iscope = next(i for i, t in enumerate(tokens) if t in SCOPES)
        except StopIteration:
            raise ValueError(f"No SCOPE token in '{base}'")
        target = ''.join(tokens[:iscope])
        tokens = tokens[iscope:]

    if len(tokens) < 7:
        raise ValueError(f"Too few fields in '{base}'")
    scope, stokes, cadence, taper, robust_tok = tokens[:5]
    if scope not in SCOPES:
        raise ValueError(f"Bad scope '{scope}' in '{base}'")
    if stokes not in STOKES_VALUES:
        raise ValueError(f"Bad stokes '{stokes}' in '{base}'")
    if cadence not in CADENCES:
        raise ValueError(f"Bad cadence '{cadence}' in '{base}'")
    if taper not in TAPERS:
        raise ValueError(f"Bad taper '{taper}' in '{base}'")
    mr = _RE_ROBUST.match(robust_tok)
    if not mr:
        raise ValueError(f"Bad robust token '{robust_tok}' in '{base}' "
                         f"(sign must be explicit, e.g. Robust+0.0)")
    robust = float(mr.group(1))
    if robust == 0.0 and mr.group(1).startswith('-'):
        raise ValueError(f"Negative-zero robust '{robust_tok}' in '{base}' "
                         f"(write Robust+0.0)")

    # PROC tokens until the LST token
    rest = tokens[5:]
    proc = []
    i = 0
    while i < len(rest) and not _RE_LST.match(rest[i]) \
            and rest[i] != 'allsky':
        tok = rest[i]
        _base = tok.split('-')[0]
        if _RE_MSUB.match(tok) or tok in PROC_ORDER or \
                (_base in ('ssub', 'loo') and _base in PROC_ORDER):
            proc.append(tok)
        else:
            raise ValueError(f"Unknown proc token '{tok}' in '{base}'")
        i += 1
    if i >= len(rest):
        raise ValueError(f"No LSTxxh/allsky token in '{base}'")

    out = {
        'grammar': 'phase3', 'scope': scope, 'stokes': stokes,
        'cadence': cadence, 'taper': taper, 'robust': robust,
        'proc': proc, 'target': target, 'cutout': cutout,
        'lst': ('allsky' if rest[i] == 'allsky'
                else _RE_LST.match(rest[i]).group(1)), 'lst_bin': None,
        'epoch': None, 'date_start': None, 'date_end': None,
        'n_epochs': None,
    }
    i += 1
    if i < len(rest):
        mb = _RE_BIN.match(rest[i])
        if mb:
            out['lst_bin'] = int(mb.group(1))
            i += 1
    if i >= len(rest):
        raise ValueError(f"Missing time token in '{base}'")
    me = _RE_EPOCH.match(rest[i])
    mrange = _RE_RANGE.match(rest[i])
    if me:
        out['epoch'] = me.group(1)
        i += 1
    elif mrange:
        out['date_start'], out['date_end'] = mrange.group(1), mrange.group(2)
        i += 1
        if i >= len(rest) or not _RE_NEP.match(rest[i]):
            raise ValueError(f"Date-range name missing Nnnn count: '{base}'")
        out['n_epochs'] = int(_RE_NEP.match(rest[i]).group(1))
        i += 1
    else:
        raise ValueError(f"Bad time token '{rest[i]}' in '{base}'")
    if i != len(rest):
        raise ValueError(f"Trailing tokens {rest[i:]} in '{base}'")
    return out


# ---------------------------------------------------------------------
# Legacy rebuilders (prove the parse is lossless; used by tests)
# ---------------------------------------------------------------------
def build_legacy_subband_name(p):
    """Reconstruct the exact Phase 2 subband filename from parse_name output."""
    name = (f"{p['scope']}-{p['stokes']}-{p['taper']}-{p['duration_s']}s-"
            f"{p['robust_token']}-{p['timestamp_mid']}-{p['product']}")
    if p['pbcorr']:
        name += '.pbcorr'
    if p['dewarped']:
        name += '_dewarped'
    return name + '.fits'


def build_legacy_wideband_name(p):
    """Reconstruct the exact Phase 2 wideband filename from parse_name output."""
    parts = [p['scope'], p['stokes'], p['cadence'], p['taper'],
             p['robust_token']]
    if p['pbcorr']:
        parts.append('pbcorr')
    if p['dewarped']:
        parts.append('dewarped')
    parts += [p['date'], f"LST{p['lst']}"]
    if p['tindex'] is not None:
        parts.append(f"t{p['tindex']:04d}")
    return '_'.join(parts) + '.fits'


# ---------------------------------------------------------------------
# Header standard
# ---------------------------------------------------------------------
def _as_time(t):
    if t is None:
        return None
    if isinstance(t, Time):
        return t
    return Time(t, scale='utc')


def frequency_block(scope):
    """Exact correlator-edge frequency keywords for a scope.

    Uses phase2_config's channel arithmetic (3072 x 23.926 kHz from
    13.398 MHz) rather than any value found in image headers, so every
    Phase 3 product carries identical, exact band edges.
    Returns dict of header keyword -> (value, comment).
    """
    if scope in cfg.SUBBAND_FREQ_HZ:
        info = cfg.SUBBAND_FREQ_HZ[scope]
        lo, hi, center = info['lo'], info['hi'], info['center']
    elif scope in cfg.WIDEBAND_BANDS_HZ:
        lo, hi = cfg.WIDEBAND_BANDS_HZ[scope]
        center = (lo + hi) / 2.0
    else:
        raise ValueError(f"Unknown scope '{scope}' for frequency block")
    return {
        # FITS WCS: declaring axes 3/4 on a 2-D image REQUIRES
        # WCSAXES=4 — ds9 honors WCSAXES literally and discards the
        # entire WCS otherwise (2026-06-11: msub showed no coordinates).
        'WCSAXES': (4, 'Coordinate axes incl. degenerate FREQ/STOKES'),
        'CTYPE3': ('FREQ', 'Frequency axis'),
        'CRVAL3': (center, '[Hz] Band center frequency'),
        'CDELT3': (hi - lo, '[Hz] Total bandwidth'),
        'CRPIX3': (1.0, 'Reference pixel'),
        'CUNIT3': ('Hz', 'Frequency unit'),
        'RESTFRQ': (center, '[Hz] Band center frequency'),
        'FREQ_MIN': (lo, '[Hz] Lower band edge (exact correlator edge)'),
        'FREQ_MAX': (hi, '[Hz] Upper band edge (exact correlator edge)'),
        'BANDWDTH': (hi - lo, '[Hz] Total bandwidth'),
    }


def write_standard_header(target, *, scope, stokes,
                          start=None, end=None, inttime_s=None,
                          time_basis='phase2',
                          beam=None, ebeam=None,
                          thermnoi=None, imgrms=None,
                          pbcorr=None, dewarped=None,
                          aligned=None, alignref=None,
                          msub=None, msubfile=None,
                          ssub=None, leaksub=None,
                          fluxref=None, astref=None, scale_a=None,
                          lstbin=None, calbp=None, p1run=None, p2run=None,
                          extra=None, validate=True, allow_no_time=False):
    """Stamp the full Phase 3 header standard.

    R-07 (Gregg 2026-08-11): an incomplete time block (start or end
    missing/None) is a HARD FAILURE unless allow_no_time=True is passed
    explicitly. The old behavior — silently leaving the template header's
    time block in place — is the P8 mechanism (epoch residuals inheriting
    their master's times).

    Parameters
    ----------
    target : str or fits.Header
        Path to a FITS file (stamped in place, then validated as a
        write-time gate) or a Header object (stamped and returned —
        validation is then the caller's responsibility at write time).
    scope, stokes : product identity; drives the frequency block.
    start, end : Time/ISO str
        START of the first integration and END of the last. DURATION is
        derived as the wall-clock span (days for multi-night stacks).
    inttime_s : float
        Summed integration time actually included [s]. INDEPENDENT of
        DURATION: a 17-night LST stack has DURATION ~18 days but INTTIME
        ~17 hours. Defaults to the wall-clock span only when omitted
        (correct for contiguous single-epoch images).
    time_basis : str
        Provenance of the start time; written as TIMEFIX='v2-<basis>'.
    beam : (bmaj_deg, bmin_deg, bpa_deg)   Header/nominal beam.
    ebeam : (bmaj_deg, bmin_deg, bpa_deg)  EMPIRICAL beam (Gaussian fits
        of unresolved sources — what the data actually delivers, which
        differs from the WSClean beam under ionospheric smearing).
    thermnoi : float  [Jy/beam] V-based thermal noise.
    imgrms : float    [Jy/beam] This image's own measured central rms.
    pbcorr/dewarped/aligned/msub/ssub/leaksub : bool or None
        Processing-state flags; written only when not None.
    alignref : str   Alignment reference (file or date) when aligned.
    msubfile : str   Master co-add subtracted, when msub.
    fluxref/astref : str  Flux / astrometric reference date ('2024-12-29').
    scale_a : float  Per-epoch mean flux scale applied (SCALE_A).
    lstbin : int     LST sub-hour bin index where applicable.
    calbp/p1run/p2run : str  Calibration + Phase 1/2 run provenance.
    extra : dict     Additional keyword -> value (or (value, comment)).
    validate : bool  Run validate_fits_headers.validate_header and raise
        RuntimeError on critical failures (path targets only).

    Returns the stamped fits.Header.
    """
    path = None
    if isinstance(target, fits.Header):
        hdr = target
    else:
        path = str(target)
        hdul = fits.open(path, mode='update')
        hdr = hdul[0].header

    try:
        # --- Time block (semantics per validate_fits_headers._stamp_times) ---
        t0, t1 = _as_time(start), _as_time(end)
        if (t0 is None or t1 is None) and not allow_no_time:
            raise RuntimeError(
                f"write_standard_header: incomplete time block "
                f"(start={start!r}, end={end!r}) for "
                f"{path or 'header object'} — REFUSING to write. "
                f"Silent template-time inheritance is the P8 mechanism "
                f"(R-07 ruling 2026-08-11). Pass allow_no_time=True only "
                f"for products that legitimately carry no epoch time.")
        if t0 is not None and t1 is not None:
            duration_s = float((t1 - t0).sec)
            itime = float(inttime_s) if inttime_s is not None else duration_s
            hdr['DATE-OBS'] = (t0.isot, 'Start of first integration (UTC)')
            hdr['MJD-OBS'] = (t0.mjd, 'Start MJD (UTC)')
            hdr['DATE-END'] = (t1.isot, 'End of last integration (UTC)')
            hdr['DURATION'] = (duration_s,
                               '[s] Wall-clock span (DATE-END - DATE-OBS)')
            hdr['INTTIME'] = (itime, '[s] Summed integration time included')
            hdr['MJD-MID'] = ((t0.mjd + t1.mjd) / 2.0, 'Midpoint MJD (UTC)')
            hdr['TIMESYS'] = ('UTC', 'Time system')
            hdr['TIMEFIX'] = (f'v2-{time_basis}',
                              'Time correction version-basis')
        elif inttime_s is not None:
            hdr['INTTIME'] = (float(inttime_s),
                              '[s] Summed integration time included')

        # --- Frequency block (exact correlator edges) ---
        for key, (val, comment) in frequency_block(scope).items():
            hdr[key] = (val, comment)

        # --- Stokes axis ---
        stokes_code = {'I': 1.0, 'Q': 2.0, 'U': 3.0, 'V': 4.0}[stokes]
        hdr['CTYPE4'] = ('STOKES', 'Stokes axis')
        hdr['CRVAL4'] = (stokes_code, f'Stokes {stokes}')
        hdr['CDELT4'] = (1.0, 'Stokes increment')
        hdr['CRPIX4'] = (1.0, 'Reference pixel')
        hdr['STOKES'] = (stokes, 'Stokes parameter')

        # --- Beams ---
        if beam is not None:
            hdr['BMAJ'] = (float(beam[0]), '[deg] Beam major axis')
            hdr['BMIN'] = (float(beam[1]), '[deg] Beam minor axis')
            hdr['BPA'] = (float(beam[2]), '[deg] Beam position angle')
        if ebeam is not None:
            hdr['EBMAJ'] = (float(ebeam[0]),
                            '[deg] Empirical beam major (unresolved fits)')
            hdr['EBMIN'] = (float(ebeam[1]), '[deg] Empirical beam minor')
            hdr['EBPA'] = (float(ebeam[2]), '[deg] Empirical beam PA')

        # --- Noise ---
        # Blank/fully-flagged images yield NaN statistics; FITS headers
        # cannot hold NaN, and one bad image must never kill a run —
        # omit the keyword instead.
        if thermnoi is not None and np.isfinite(thermnoi):
            hdr['THERMNOI'] = (float(thermnoi),
                               '[Jy/beam] Thermal noise (Stokes V based)')
        if imgrms is not None and np.isfinite(imgrms):
            hdr['IMGRMS'] = (float(imgrms),
                             '[Jy/beam] Measured central rms of this image')

        # --- Processing-state flags ---
        if pbcorr is not None:
            hdr['PBCORR'] = (bool(pbcorr), 'Primary-beam corrected')
        if dewarped is not None:
            hdr['DEWARPED'] = (bool(dewarped), 'Ionospheric dewarp applied')
        if aligned is not None:
            hdr['ALIGNED'] = (bool(aligned), 'Flow-aligned to reference')
            if aligned and alignref:
                hdr['ALIGNREF'] = (str(alignref)[:68], 'Alignment reference')
        if msub is not None:
            hdr['MSUB'] = (bool(msub), 'Master co-add subtracted')
            if msub and msubfile:
                hdr['MSUBFILE'] = (os.path.basename(str(msubfile))[:68],
                                   'Subtracted master co-add')
        if ssub is not None:
            hdr['SSUB'] = (bool(ssub), 'Striation model subtracted')
        if leaksub is not None:
            hdr['LEAKSUB'] = (bool(leaksub),
                              'Low-order V leakage screen subtracted')

        # --- Provenance scalars ---
        hdr['GITTAG'] = (cfg.PIPELINE_GITTAG, 'Pipeline code version')
        if fluxref is not None:
            hdr['FLUXREF'] = (str(fluxref), 'Flux scale reference date')
        if astref is not None:
            hdr['ASTREF'] = (str(astref), 'Astrometric reference date')
        if scale_a is not None:
            hdr['SCALE_A'] = (float(scale_a),
                              'Mean flux scale applied (target/reference)')
        if lstbin is not None:
            hdr['LSTBIN'] = (int(lstbin), 'LST sub-hour bin index')
        if calbp is not None:
            hdr['CALBP'] = (str(calbp)[:68], 'Bandpass calibration table')
        if p1run is not None:
            hdr['P1RUN'] = (str(p1run)[:68], 'Phase 1 run label')
        if p2run is not None:
            hdr['P2RUN'] = (str(p2run)[:68], 'Phase 2 run label')

        if extra:
            for key, val in extra.items():
                hdr[key] = val
    finally:
        if path is not None:
            hdul.close()  # flushes the update

    # --- Write-time validation gate ---
    if path is not None and validate:
        from exopipe.common.validate_fits_headers import validate_header
        report = validate_header(path, level='standard')
        if not report.ok:
            raise RuntimeError(
                f"Header validation FAILED for {path}:\n" +
                "\n".join(f"  ERROR: {e}" for e in report.errors))
    return hdr


# ---------------------------------------------------------------------
# Provenance table
# ---------------------------------------------------------------------
# One row per constituent of a co-add (or per parent of a subtraction):
# enough to reweight the stack offline and to trace any pixel back to
# raw visibilities (FILE -> Phase 2 run -> P1RUN -> MS files).
PROVENANCE_COLUMNS = [
    ('FILE',      'A256', 'Constituent file (archive-relative)'),
    ('MJD_OBS',   'D',    'Start MJD (UTC)'),
    ('MJD_MID',   'D',    'Midpoint MJD (UTC)'),
    ('INTTIME',   'D',    '[s] Summed integration time'),
    ('WEIGHT',    'D',    'Co-add weight (inverse variance)'),
    ('THERMNOI',  'D',    '[Jy/beam] Thermal noise'),
    ('N_ANCHORS', 'J',    'Dewarp anchor count'),
    ('FLUXSCALE', 'D',    'Flux scale applied'),
    ('P1RUN',     'A64',  'Phase 1 run label'),
]


def append_provenance_table(path, rows):
    """Append (or replace) the PROVENANCE binary table HDU.

    rows: list of dicts keyed by PROVENANCE_COLUMNS names. Missing
    numeric fields become NaN (-1 for N_ANCHORS), missing strings ''.
    """
    cols = []
    for name, fmt, _comment in PROVENANCE_COLUMNS:
        if fmt.startswith('A'):
            vals = [str(r.get(name, '') or '') for r in rows]
        elif fmt == 'J':
            vals = [int(r.get(name, -1) if r.get(name) is not None else -1)
                    for r in rows]
        else:
            vals = [float(r.get(name, np.nan)
                          if r.get(name) is not None else np.nan)
                    for r in rows]
        cols.append(fits.Column(name=name, format=fmt, array=np.array(vals)))
    tbl = fits.BinTableHDU.from_columns(cols, name='PROVENANCE')
    tbl.header['NPROV'] = (len(rows), 'Number of constituent files')
    for name, _fmt, comment in PROVENANCE_COLUMNS:
        tbl.header['COMMENT'] = f'{name}: {comment}'

    with fits.open(path, mode='update') as hdul:
        existing = [i for i, h in enumerate(hdul)
                    if h.name == 'PROVENANCE']
        for i in reversed(existing):
            hdul.pop(i)
        hdul.append(tbl)


def read_provenance(path):
    """Read the PROVENANCE table; returns list of dicts (or [])."""
    out = []
    with fits.open(path) as hdul:
        if 'PROVENANCE' not in [h.name for h in hdul]:
            return out
        data = hdul['PROVENANCE'].data
        names = data.columns.names
        for row in data:
            d = {}
            for n in names:
                v = row[n]
                if isinstance(v, bytes):
                    v = v.decode()
                elif isinstance(v, np.integer):
                    v = int(v)
                elif isinstance(v, np.floating):
                    v = float(v)
                d[n] = v
            out.append(d)
    return out


# ---------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------
def _selftest_naming():
    """Round-trip the new grammar and both legacy grammars."""
    failures = []

    def check(label, cond, detail=''):
        status = 'PASS' if cond else 'FAIL'
        print(f"  [{status}] {label}" + (f"  ({detail})" if detail else ''))
        if not cond:
            failures.append(label)

    # --- New grammar: build -> parse -> build identity ---
    cases = [
        dict(scope='Green', stokes='I', cadence='deep', taper='Taper',
             robust=0.0, proc=['dewarped', 'aligned'],
             lst='10h', epoch='20241229T081959'),
        dict(scope='73MHz', stokes='V', cadence='10min', taper='Taper',
             robust=-0.75, proc=['pbcorr', 'dewarped'],
             lst='10h', lst_bin=2, epoch='20241229T082951'),
        dict(scope='Full', stokes='I', cadence='deep', taper='NoTaper',
             robust=-0.75, proc=['dewarped', 'aligned', 'coadd'],
             lst='10h', date_start='20241218', date_end='20250105',
             n_epochs=17),
        dict(scope='Red', stokes='V', cadence='10min', taper='Taper',
             robust=0.0, proc=['dewarped', 'aligned', 'coadd'],
             lst='10h', lst_bin=2, date_start='20241218',
             date_end='20250105', n_epochs=17),
        dict(scope='Green', stokes='I', cadence='deep', taper='Taper',
             robust=0.0, proc=['dewarped', 'aligned', 'msub-c01', 'ssub'],
             lst='10h', epoch='20241229T081959'),
        dict(scope='Blue', stokes='I', cadence='deep', taper='Taper',
             robust=0.0, proc=['dewarped', 'aligned', 'coadd', 'sigstack'],
             lst='07h', date_start='20241218', date_end='20250105',
             n_epochs=19),
        dict(scope='Green', stokes='I', cadence='deep', taper='Taper',
             robust=0.0, proc=['pbcorr', 'dewarped', 'aligned', 'msub-c02'],
             lst='10h', epoch='20241229T081959',
             target='GJ486', cutout=True),
    ]
    for c in cases:
        name = build_name(**c)
        parsed = parse_name(name)
        rebuilt = build_name(**parsed)
        check(f"new-grammar round-trip: {name}", rebuilt == name,
              '' if rebuilt == name else f'rebuilt={rebuilt}')

    # Spot-check exact rendering against the agreed grammar examples
    n1 = build_name(scope='Green', stokes='I', cadence='deep',
                    taper='Taper', robust=0.0, proc=['dewarped'],
                    lst='10h', epoch='2024-12-29T08:19:59.123')
    check("robust sign + epoch truncation",
          n1 == 'Green_I_deep_Taper_Robust+0.0_dewarped_LST10h_'
                '20241229T081959.fits', n1)
    n2 = build_name(scope='18MHz', stokes='I', cadence='deep',
                    taper='Taper', robust=-0.75, proc=[],
                    lst='10h', date_start='2024-12-18',
                    date_end='2025-01-05', n_epochs=17)
    check("date-range TIMEID",
          n2 == '18MHz_I_deep_Taper_Robust-0.75_LST10h_'
                '20241218-20250105_N017.fits', n2)
    # proc canonical ordering is enforced regardless of input order
    n3 = build_name(scope='Green', stokes='V', cadence='deep',
                    taper='Taper', robust=0.0,
                    proc=['msub-c01', 'aligned', 'dewarped'],
                    lst='10h', epoch='20241229T081959')
    check("proc token canonical ordering",
          '_dewarped_aligned_msub-c01_' in n3, n3)

    # --- Legacy subband grammar (real archive filenames) ---
    legacy_sub = [
        '55MHz-I-Taper-3571s-Robust-0-20241229_114918-image_dewarped.fits',
        '55MHz-I-NoTaper-3571s-Robust-0.75-20241229_114918-'
        'image.pbcorr_dewarped.fits',
        '69MHz-I-Taper-592s-Robust-0-20241218_120740-image.fits',
        '73MHz-V-Taper-10s-Robust-0-20241229_114918-image.fits',
    ]
    expect_sub = [
        ('55MHz', 'I', 'deep', 0.0, False, True),
        ('55MHz', 'I', 'deep', -0.75, True, True),
        ('69MHz', 'I', '10min', 0.0, False, False),
        ('73MHz', 'V', 'snap', 0.0, False, False),
    ]
    for name, (sc, st, cad, rob, pb, dw) in zip(legacy_sub, expect_sub):
        p = parse_name(name)
        ok = (p['grammar'] == 'phase2_subband' and p['scope'] == sc
              and p['stokes'] == st and p['cadence'] == cad
              and p['robust'] == rob and p['pbcorr'] == pb
              and p['dewarped'] == dw
              and build_legacy_subband_name(p) == name)
        check(f"legacy subband round-trip: {name}", ok,
              '' if ok else str(p))

    # --- Legacy wideband grammar ---
    legacy_wb = [
        'Full_I_deep_Taper_Robust-0_dewarped_20241229_LST07h.fits',
        'Green_I_deep_NoTaper_Robust-0.75_dewarped_20241229_LST10h.fits',
        'Green_I_10min_Taper_Robust-0_pbcorr_dewarped_20241229_'
        'LST10h_t0003.fits',
        'Green_V_deep_Taper_Robust-0_pbcorr_dewarped_20241229_LST10h.fits',
    ]
    expect_wb = [
        ('Full', 'I', 'deep', 0.0, False, None, '07h'),
        ('Green', 'I', 'deep', -0.75, False, None, '10h'),
        ('Green', 'I', '10min', 0.0, True, 3, '10h'),
        ('Green', 'V', 'deep', 0.0, True, None, '10h'),
    ]
    for name, (sc, st, cad, rob, pb, tidx, lst) in zip(legacy_wb, expect_wb):
        p = parse_name(name)
        ok = (p['grammar'] == 'phase2_wideband' and p['scope'] == sc
              and p['stokes'] == st and p['cadence'] == cad
              and p['robust'] == rob and p['pbcorr'] == pb
              and p['tindex'] == tidx and p['lst'] == lst
              and build_legacy_wideband_name(p) == name)
        check(f"legacy wideband round-trip: {name}", ok,
              '' if ok else str(p))

    # --- Error cases: malformed names must raise ---
    bad = [
        'Green_I_deep_Taper_Robust-0_dewarped_LST10h_20241229T081959.fits',
        # ^ new-grammar position but legacy implicit-sign robust
        'Green_I_deep_Taper_Robust+0.0_dewarped_LST10h.fits',  # no time
        'Green_I_deep_Taper_Robust+0.0_LST10h_20241218-20250105.fits',
        # ^ range without Nnnn
        'Purple_I_deep_Taper_Robust+0.0_LST10h_20241229T081959.fits',
    ]
    for name in bad:
        try:
            parse_name(name)
            check(f"rejects malformed: {name}", False, 'parsed without error')
        except ValueError:
            check(f"rejects malformed: {name}", True)

    return failures


def _selftest_header(src, workdir):
    """Stamp + provenance on a copy of a real archived file."""
    import shutil
    failures = []
    os.makedirs(workdir, exist_ok=True)
    dst = os.path.join(workdir, build_name(
        scope='Green', stokes='I', cadence='deep', taper='Taper',
        robust=0.0, proc=['dewarped'], lst='10h',
        epoch=fits.getheader(src).get('DATE-OBS', '2024-12-29T11:19:33')))
    shutil.copy(src, dst)
    print(f"  Test copy: {dst}")

    h0 = fits.getheader(src)
    write_standard_header(
        dst, scope='Green', stokes='I',
        start=h0['DATE-OBS'], end=h0['DATE-END'],
        inttime_s=h0.get('INTTIME'), time_basis='phase2',
        beam=(h0['BMAJ'], h0['BMIN'], h0['BPA']),
        ebeam=(0.231, 0.173, 88.0),
        thermnoi=h0.get('THERMNOI'), imgrms=0.0123,
        pbcorr=False, dewarped=True, aligned=True, alignref='2024-12-29',
        fluxref='2024-12-29', astref='2024-12-29', scale_a=1.0,
        p2run='Science_20260525_220814')
    rows = [
        dict(FILE='Green/I/deep/Green_I_deep_Taper_Robust-0_dewarped_'
                  '20241229_LST10h.fits',
             MJD_OBS=60673.4719, MJD_MID=60673.4926, INTTIME=3581.07,
             WEIGHT=1.0, THERMNOI=0.0178, N_ANCHORS=161, FLUXSCALE=1.0,
             P1RUN='Run_20260331_171539'),
        dict(FILE='second/file.fits', MJD_OBS=60674.47, MJD_MID=60674.49,
             INTTIME=3581.0, WEIGHT=0.8, THERMNOI=0.019, N_ANCHORS=140,
             FLUXSCALE=0.97, P1RUN='Run_20260331_171539'),
    ]
    append_provenance_table(dst, rows)
    # second call must REPLACE, not duplicate
    append_provenance_table(dst, rows)

    def check(label, cond, detail=''):
        status = 'PASS' if cond else 'FAIL'
        print(f"  [{status}] {label}" + (f"  ({detail})" if detail else ''))
        if not cond:
            failures.append(label)

    h = fits.getheader(dst)
    check("DATE-OBS preserved (start, not midpoint)",
          h['DATE-OBS'] == h0['DATE-OBS'], h['DATE-OBS'])
    check("INTTIME independent of DURATION",
          abs(h['INTTIME'] - h0['INTTIME']) < 0.01
          and 'DURATION' in h)
    check("TIMEFIX v2", str(h.get('TIMEFIX', '')).startswith('v2-'),
          h.get('TIMEFIX'))
    check("frequency axis exact edges",
          h['CTYPE3'] == 'FREQ'
          and abs(h['FREQ_MIN'] - cfg.WIDEBAND_BANDS_HZ['Green'][0]) < 1
          and abs(h['FREQ_MAX'] - cfg.WIDEBAND_BANDS_HZ['Green'][1]) < 1,
          f"{h.get('FREQ_MIN')}..{h.get('FREQ_MAX')}")
    check("empirical beam stamped", 'EBMAJ' in h and 'EBPA' in h)
    check("state flags + provenance scalars",
          h.get('ALIGNED') is True and h.get('ALIGNREF') == '2024-12-29'
          and h.get('FLUXREF') == '2024-12-29'
          and h.get('GITTAG') == cfg.PIPELINE_GITTAG
          and h.get('IMGRMS') == 0.0123)
    with fits.open(dst) as hdul:
        n_prov_hdus = sum(1 for x in hdul if x.name == 'PROVENANCE')
    check("single PROVENANCE HDU after re-append", n_prov_hdus == 1,
          f"{n_prov_hdus} HDUs")
    back = read_provenance(dst)
    check("provenance read-back",
          len(back) == 2 and back[0]['N_ANCHORS'] == 161
          and back[1]['P1RUN'] == 'Run_20260331_171539'
          and abs(back[0]['MJD_OBS'] - 60673.4719) < 1e-6)

    from exopipe.common.validate_fits_headers import validate_header
    rep = validate_header(dst, level='full')
    check("validate_header passes (no errors)", rep.ok,
          '; '.join(rep.errors) if rep.errors else rep.summary())
    return failures


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--stamp_test_src',
                    help='Archived FITS to copy + stamp for the header test')
    ap.add_argument('--stamp_test_dir',
                    default='/fast/gh/main/phase3_v2/selftest',
                    help='Scratch dir for the header test copy')
    args = ap.parse_args()

    print("=== phase3_products self-test: naming grammars ===")
    fails = _selftest_naming()
    if args.stamp_test_src:
        print("\n=== phase3_products self-test: header + provenance ===")
        fails += _selftest_header(args.stamp_test_src, args.stamp_test_dir)
    print(f"\n{'ALL TESTS PASSED' if not fails else 'FAILURES: ' + str(fails)}")
    raise SystemExit(0 if not fails else 1)
