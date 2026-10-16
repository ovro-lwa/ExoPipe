"""THERMNOI of a night and band, measured at Stage 1 - the output standard's V-based thermal noise (PIPELINE_GOALS_DRAFT.md
6.4; plan/wave2/OUTPUT_WRITERS_DESIGN.md "Where the values come from", writers step 1).

Phase 2 stamps the card (exopipe/phase2/phase2_science_agg.py, _stamp_subband_thermnoi / _stamp_wideband_thermnoi): the
inner rms (get_inner_rms: the MAD std of the central 1024^2) of the night's Stokes V deep image. A subband takes the first
of its dewarped V deep Taper images (the raw ones when none is dewarped; _find_subband_image) and stamps that value on
every image of the subband; a wideband measures each V deep image, and its I images take the preferred one's value (not
PB-corrected first, then dewarped, among the images that measure). Stage 1 measures it again, with those functions, on the
archived images of every night, rather than copying a card: the 2026-10-16 survey of the 456 installed cells (8,765
nights) found 43 nights whose images carry no card - 33 of them with a V image to measure - and 27 whose I card is stale
(their I images were re-made later; the V card is the measurement). The measurement equals Phase 2's card on every
sampled night (300 subband, 100 wideband) and the V card on the 27. A night with no V deep image that measures (4 have
none, 6 only an empty or truncated file) has no THERMNOI: measure() says why.
"""
import functools
import glob
import os
import re

import numpy as np

# Phase 2's card comments (_write_thermnoi): a subband's images (_stamp_subband_thermnoi) and a wideband's I images
# (_stamp_wideband_thermnoi; every Phase 3 product of the night takes the I images' form - it is not 'this V image')
COMMENT_SUBBAND = '[Jy/beam] Noise (matching V Taper inner rms)'
COMMENT_WIDEBAND = '[Jy/beam] Noise (matching V product inner rms)'
_INTERVAL = re.compile(r'_t(\d{4})')                         # a wideband interval product's tag (Phase 2's tkey)
# the wideband bands a Phase 2 run did not form (phase2_controller / run_wideband_stacking log lines; Gregg 2026-07-09 and
# 2026-09-05: no wideband product from an incomplete or excluded subband set, Full not formed when a band is not)
_SKIP = re.compile(r'Wideband bands (?:SKIPPED \(incomplete subband set\)|NOT FORMED \(excluded constituent\)|'
                   r'to SKIP \(incomplete\)): \[([^\]]*)\]')
_P2_LOADED = []


def comment_for(band):
    """The THERMNOI card comment Phase 2 writes for ``band``'s products."""
    return COMMENT_SUBBAND if band.endswith('MHz') else COMMENT_WIDEBAND


def _p2():
    """Imports Phase 2's module once. Its import switches astropy's solar-system ephemeris process-wide (to de432s, measured
    2026-10-16); the ephemeris in force before is restored, so a measurement never changes the positions the rest of the
    process computes. (It also selects matplotlib's Agg backend, which every Phase 3 plotting path uses.)"""
    if not _P2_LOADED:
        from astropy.coordinates import solar_system_ephemeris
        prev = solar_system_ephemeris.get()
        try:
            from exopipe.phase2 import phase2_science_agg            # noqa: F401 (Phase 2's own functions)
        finally:
            solar_system_ephemeris.set(prev)
        _P2_LOADED.append(True)


@functools.lru_cache(maxsize=4096)
def phase2_skipped(sci_dir):
    """The wideband bands the Phase 2 run in ``sci_dir`` did not form, from its log (phase2_*.log): Phase 2 stamps no
    THERMNOI on them, whatever image a bug left (the Full band before the 2026-09-05 fix)."""
    out = set()
    for lg in sorted(glob.glob(os.path.join(sci_dir, 'phase2_*.log'))):
        with open(lg, errors='replace') as fh:
            for ln in fh:
                m = _SKIP.search(ln)
                if m:
                    out |= {x.strip(" '\"") for x in m.group(1).split(',') if x.strip(" '\"")}
    return frozenset(out)


def _measures(rms):
    return rms is not None and bool(np.isfinite(rms)) and rms > 0


def candidates(sci_dir, band):
    """The V deep images measure(sci_dir, band) may read, from the directory listing alone (no pixels read): a subband's one
    image (_find_subband_image's first), a wideband's in the order measure() walks them (none where Phase 2 did not form the
    band). They are inputs of every product of the night that carries the card (a unit's record and --resume identity)."""
    _p2()
    from exopipe.phase2.phase2_science_agg import _find_subband_image
    if band.endswith('MHz'):
        sb = os.path.join(sci_dir, band)
        v = (_find_subband_image(sb, 'V', 'deep', 'Taper', '', require_dewarped=True)
             or _find_subband_image(sb, 'V', 'deep', 'Taper', '', require_dewarped=False))
        return [v[0]] if v else []
    if band in phase2_skipped(sci_dir):
        return []
    vdir = os.path.join(sci_dir, band, 'V', 'deep')
    cands = [f for f in sorted(glob.glob(os.path.join(vdir, '*.fits'))) if not _INTERVAL.search(os.path.basename(f))]
    return sorted(cands, key=lambda f: ('pbcorr' in os.path.basename(f), '_dewarped' not in os.path.basename(f)))


def measure(sci_dir, band):
    """(THERMNOI in Jy/beam rounded as Phase 2 stamps it, the V image measured, '') of the night whose Phase 2 Science
    directory is ``sci_dir``, in ``band`` ('73MHz' or 'Red' / 'Green' / 'Blue' / 'Full'); (None, the image or '', why)
    when the night has no V deep image that measures, or Phase 2 did not form the band (its log)."""
    _p2()
    from exopipe.phase2.phase2_science_agg import _find_subband_image, get_inner_rms
    if band.endswith('MHz'):
        sb = os.path.join(sci_dir, band)
        v = (_find_subband_image(sb, 'V', 'deep', 'Taper', '', require_dewarped=True)
             or _find_subband_image(sb, 'V', 'deep', 'Taper', '', require_dewarped=False))
        if not v:
            return None, '', f'no Stokes V deep Taper image in {sb}/V/deep'
        rms = get_inner_rms(v[0])
        if not _measures(rms):
            return None, v[0], f'{os.path.basename(v[0])} does not measure (inner rms {rms})'
        return float(round(rms, 9)), v[0], ''
    if band in phase2_skipped(sci_dir):
        return None, '', (f'Phase 2 did not form {band} on this night (its log: an incomplete or excluded subband set; '
                          f'Gregg 2026-07-09, 2026-09-05) - it stamped no THERMNOI there')
    # Phase 2's preference (_stamp_wideband_thermnoi): not PB-corrected first, then dewarped; the first in sorted order among
    # equals. Walked in that order (candidates), the first image that measures is the one Phase 2's value came from.
    for f in candidates(sci_dir, band):
        rms = get_inner_rms(f)
        if _measures(rms):
            return float(round(rms, 9)), f, ''
    return None, '', f'no Stokes V deep image in {os.path.join(sci_dir, band, "V", "deep")} that measures'
