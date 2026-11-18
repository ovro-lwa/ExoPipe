"""Warp screens of the hybrid stacks. Port of notaper_largescale.py (NL) at exopipe 4c7098c: _peak_offsets 207, derive_warp_screen
222. The acceptance limits are config hybrid_stack.WARP_MEDIAN_PX / WARP_P90_PX / WARP_MIN_SOURCES (NL 289, 350, 414).

Change from production (no pixel changes): a cached screen is reused only while the raw / dewarped pair it was derived from is
unchanged (size and mtime of both files, recorded in the cache's JSON); production reused any cached screen of the night and
subband. Arrays are read with ordinary reads (memmap=False)."""
import json
import os
import shutil
import tempfile
import time

import numpy as np


class WarpRefusal(RuntimeError):
    """A night cannot be put on the master's frame (the stack stage records it, or refuses past hybrid_stack.MAX_UNDERIVABLE)."""


def offsets_pass(d):
    """The offsets pass the acceptance limits (hybrid_stack.WARP_*)."""
    from exopipe.phase3.config import CFG
    S = CFG.hybrid_stack
    return bool(d.size >= S.WARP_MIN_SOURCES and np.median(d) <= S.WARP_MEDIAN_PX and np.percentile(d, 90) <= S.WARP_P90_PX)


def offsets_summary(d):
    return dict(n=int(d.size), median_px=float(np.median(d)) if d.size else None,
                p90_px=float(np.percentile(d, 90)) if d.size else None)


def peak_offsets(A, B, nsig=40.0, size=21, box=25, inner=0.7):
    """Pixel distance from each bright compact source of B (local maxima above nsig x the MAD noise, size-px neighborhood, within
    inner x the half-width of the image from its center: beam-corrected images put noise peaks above any threshold near the
    horizon) to the brightest pixel of A within box px: ~0 when A and B are on the same frame."""
    from scipy.ndimage import maximum_filter
    f = np.isfinite(B)
    sig = 1.4826 * np.median(np.abs(B[f] - np.median(B[f])))
    bb = np.where(f, B, 0.0)
    yy, xx = np.nonzero((bb == maximum_filter(bb, size)) & (bb > nsig * sig))
    d = []
    cy, cx = (B.shape[0] - 1) / 2, (B.shape[1] - 1) / 2
    for y, x in zip(yy, xx):
        if np.hypot(y - cy, x - cx) < inner * min(cy, cx) and box < y < A.shape[0] - box and box < x < A.shape[1] - box:
            w = np.nan_to_num(A[y - box:y + box + 1, x - box:x + box + 1])
            j, i = np.unravel_index(np.argmax(w), w.shape)
            d.append(np.hypot(j - box, i - box))
    return np.array(d)


def _pair_fp(paths):
    out = {}
    for p in paths:
        st = os.stat(p)
        out[p] = [st.st_size, st.st_mtime_ns]
    return out


def _read_screen(out):
    from astropy.io import fits
    with fits.open(out, memmap=False) as hs:
        return hs['SCREEN_X'].data.astype(np.float64), hs['SCREEN_Y'].data.astype(np.float64)


def derive_warp_screen(sci_dir, date, lst, band, cache_dir):
    """The night's warp screen where Phase 2 dewarped the deep images (flow cascade) but saved no screen for this subband
    (R-GREGG-20261101-DERIVE-MISSING-WARP-SCREENS-REBUILD-HB-MEASURE-18-59-13H, Gregg 2026-11-01 "All approved"): derived exactly as
    Phase 2 recovers it for the night's snapshots (cascade_dewarp._dewarp_remaining_snapshots): PyBDSF sources of the night's raw and
    dewarped deep Taper Robust -0.75 beam-corrected images (the uncorrected pair when absent), cross-matched
    (CASCADE_MATCH_RADIUS_BASE, CASCADE_FLUX_RATIO window), RBF screen (WARP_RBF_*), original -> dewarped frame. PyBDSF runs on
    node-local copies (it writes beside its input; the Phase 2 directories are never written). Content test: the raw image of the
    pair warped with the screen lines up with the dewarped one (offsets_pass), else WarpRefusal. Cached per night and subband in
    cache_dir (node-local; deterministic, re-derivable; reused only while its pair is unchanged). Returns (screen_x, screen_y,
    record)."""
    import logging
    from astropy import units as u
    from astropy.io import fits
    from astropy.wcs import WCS
    from exopipe.common import phase2_config as P2C
    from exopipe.phase2 import cascade_dewarp as CD
    from exopipe.phase2.cascade_dewarp import apply_warp
    from exopipe.phase3.config import CFG
    os.makedirs(cache_dir, exist_ok=True)
    out = f'{cache_dir}/{date}_{lst}_{band}_warp_screen_derived.fits'
    lg = logging.getLogger('exopipe.phase3.lib.hybrid.warp')
    pairs = [(CD.find_deep_pb_image(sci_dir, band, lg), CD.find_deep_pb_image_dewarped(sci_dir, band, lg)),
             (CD.find_deep_raw_image(sci_dir, band, lg), CD.find_deep_raw_image_dewarped(sci_dir, band, lg))]
    pairs = [(o, d) for o, d in pairs if o and d]
    if not pairs:
        raise WarpRefusal(f'{date} {lst} {band}: no raw + dewarped deep image pair in {sci_dir} to derive its warp screen - REFUSING')
    fp_now = _pair_fp(sorted({p for pr in pairs for p in pr}))
    try:                                              # one derivation per night and subband: the cell's other stacks wait for it
        os.close(os.open(out + '.lock', os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        mine = True
    except FileExistsError:
        mine = False
        t0 = time.time()
        while (os.path.exists(out + '.lock') and not os.path.exists(out + '.json')
               and time.time() - t0 < CFG.hybrid_stack.SCREEN_LOCK_WAIT_S):
            time.sleep(20)
    try:
        if os.path.exists(out) and os.path.exists(out + '.json'):
            with open(out + '.json') as f:
                rec = json.load(f)
            if rec.get('pair_fingerprint') == fp_now:
                sx, sy = _read_screen(out)
                return sx, sy, rec
            if not mine:
                raise WarpRefusal(f'{date} {lst} {band}: the cached warp screen {out} was derived from another raw / dewarped pair '
                                  f'and another process holds its lock - REFUSING')
        elif not mine:
            raise WarpRefusal(f'{date} {lst} {band}: waited {CFG.hybrid_stack.SCREEN_LOCK_WAIT_S} s for another process deriving '
                              f'{out} - REFUSING')
        tmp = tempfile.mkdtemp(prefix=f'screen_{date}_{lst}_{band}_', dir=cache_dir)
        try:
            # Phase 2's pair (the beam-corrected one), then its fallback pair (uncorrected) - Phase 2 falls back only when the corrected
            # files are missing; here also when PyBDSF fails on them ("unphysical rms" at 13h 2025-04-21 64-78 MHz, 2026-11-03) or
            # they give no anchors
            tried = []
            for orig, dew in pairs:
                to, td = f'{tmp}/{os.path.basename(orig)}', f'{tmp}/{os.path.basename(dew)}'
                shutil.copy2(orig, to)
                shutil.copy2(dew, td)
                with fits.open(td, mode='update', memmap=False) as hd_, fits.open(to, memmap=False) as ho_:
                    # as cascade_dewarp._ensure_freq_header (flow strips the frequency cards PyBDSF needs), on the copy, without
                    # its NAXIS3/4 cards (invalid in a 2-D image; current astropy refuses to write them)
                    if 'CRVAL3' not in hd_[0].header:
                        for k in ('CRVAL3', 'CTYPE3', 'CDELT3', 'CRPIX3', 'CUNIT3', 'BMAJ', 'BMIN', 'BPA'):
                            if k in ho_[0].header and k not in hd_[0].header:
                                hd_[0].header[k] = ho_[0].header[k]
                        hd_.flush(output_verify='silentfix')
                do, dd = CD.extract_sources(to, lg), CD.extract_sources(td, lg)
                if do is None or do.empty or dd is None or dd.empty:
                    tried.append(f'{os.path.basename(orig)}: source extraction failed')
                    continue
                anchors = CD.cross_match_subbands(do, dd, P2C.CASCADE_MATCH_RADIUS_BASE * u.arcmin, P2C.CASCADE_FLUX_RATIO_MIN,
                                                  P2C.CASCADE_FLUX_RATIO_MAX, lg, beam_deg=fits.getheader(orig).get('BMAJ', 0.1))
                if anchors is None:
                    tried.append(f'{os.path.basename(orig)}: too few raw / dewarped matches')
                    continue
                break
            else:
                raise WarpRefusal(f'{date} {lst} {band}: the warp screen cannot be derived ({"; ".join(tried)}) - REFUSING')
            h = fits.getheader(orig)
            R0 = np.squeeze(fits.getdata(orig, memmap=False)).astype(np.float64)
            sx, sy = CD.compute_warp_screen(anchors, WCS(h).celestial, R0.shape, lg, method='rbf', rbf_kernel=P2C.WARP_RBF_KERNEL,
                                            rbf_smoothing=P2C.WARP_RBF_SMOOTHING, rbf_epsilon=P2C.WARP_RBF_EPSILON)
            if sx is None:
                raise WarpRefusal(f'{date} {lst} {band}: warp screen computation failed - REFUSING')
            # the test on the night's uncorrected (apparent) pair when it exists: uniform noise, so 40-sigma peaks are sources
            # (beam-corrected images put artifact and noise peaks above any threshold - 13h 64 MHz 2025-05-02: 90th percentile 17 px
            # on the corrected pair)
            ta, tb = CD.find_deep_raw_image(sci_dir, band, lg), CD.find_deep_raw_image_dewarped(sci_dir, band, lg)
            if not (ta and tb):
                ta, tb = orig, dew
            d = peak_offsets(apply_warp(np.squeeze(fits.getdata(ta, memmap=False)).astype(np.float64), sx, sy),
                             np.squeeze(fits.getdata(tb, memmap=False)).astype(np.float64))
            t = dict(offsets_summary(d), pair=[os.path.basename(ta), os.path.basename(tb)])
            if not offsets_pass(d):
                raise WarpRefusal(f'{date} {lst} {band}: the derived screen does not reproduce the Phase 2 dewarp {t} - REFUSING')
            rec = dict(method='rbf from the raw / dewarped deep pair (cascade_dewarp._dewarp_remaining_snapshots)', raw=orig,
                       dewarped=dew, pairs_failed=tried, n_anchors=int(len(anchors['ra_lwa'])),
                       flux_ratio_median=float(anchors['flux_ratio_median']), test_raw_warped_vs_dewarped=t, pair_fingerprint=fp_now)
            hl = fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(sx.astype(np.float32), name='SCREEN_X'),
                               fits.ImageHDU(sy.astype(np.float32), name='SCREEN_Y')])
            hl.writeto(out + '.part', overwrite=True)
            os.replace(out + '.part', out)
            with open(out + '.json.part', 'w') as f:
                json.dump(rec, f, indent=1)
            os.replace(out + '.json.part', out + '.json')
            sx, sy = _read_screen(out)                  # the cached (float32) screen is the one every stack applies
            return sx, sy, rec
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    finally:
        if mine and os.path.exists(out + '.lock'):
            os.remove(out + '.lock')
