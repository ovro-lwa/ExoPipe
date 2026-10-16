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
import glob
import os
import re

import numpy as np

COMMENT = '[Jy/beam] Noise (matching V Taper inner rms)'   # Phase 2's card comment (_stamp_subband_thermnoi)
_INTERVAL = re.compile(r'_t(\d{4})')                         # a wideband interval product's tag (Phase 2's tkey)


def _p2():
    from exopipe.phase2 import phase2_science_agg as P2      # Phase 2's own functions (imported when first measured)
    return P2


def _measures(rms):
    return rms is not None and bool(np.isfinite(rms)) and rms > 0


def measure(sci_dir, band):
    """(THERMNOI in Jy/beam rounded as Phase 2 stamps it, the V image measured, '') of the night whose Phase 2 Science
    directory is ``sci_dir``, in ``band`` ('73MHz' or 'Red' / 'Green' / 'Blue' / 'Full'); (None, the image or '', why)
    when the night has no V deep image that measures."""
    P2 = _p2()
    if band.endswith('MHz'):
        sb = os.path.join(sci_dir, band)
        v = (P2._find_subband_image(sb, 'V', 'deep', 'Taper', '', require_dewarped=True)
             or P2._find_subband_image(sb, 'V', 'deep', 'Taper', '', require_dewarped=False))
        if not v:
            return None, '', f'no Stokes V deep Taper image in {sb}/V/deep'
        rms = P2.get_inner_rms(v[0])
        if not _measures(rms):
            return None, v[0], f'{os.path.basename(v[0])} does not measure (inner rms {rms})'
        return float(round(rms, 9)), v[0], ''
    vdir = os.path.join(sci_dir, band, 'V', 'deep')
    best = None
    for f in sorted(glob.glob(os.path.join(vdir, '*.fits'))):
        bn = os.path.basename(f)
        if _INTERVAL.search(bn):            # an interval product: not the night's deep image
            continue
        rms = P2.get_inner_rms(f)
        if not _measures(rms):
            continue
        pref = ('pbcorr' not in bn, '_dewarped' in bn)
        if best is None or pref > best[0]:
            best = (pref, rms, f)
    if best is None:
        return None, '', f'no Stokes V deep image in {vdir} that measures'
    return float(round(best[1], 9)), best[2], ''
