#!/usr/bin/env python3
"""
cascade_dewarp.py  Cascade Ionospheric Dewarping for OVRO-LWA Phase 2

Bootstrap dewarping from a 73 MHz anchor (cross-matched against VLSSr at
74 MHz with strict flux tolerance), then cascade outward to all other
subbands by cross-matching each subband against its already-dewarped
neighbor.

Improvements over the per-subband independent dewarping:
  - No spectral index extrapolation (each step bridges ~45 MHz)
  - Iterative sigma-clip outlier rejection on warp anchors
  - Thin-plate spline interpolation (globally smooth, no edge artifacts)
  - Re-extraction from dewarped images prevents error accumulation
"""
import os
import glob
import warnings
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import map_coordinates
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.stats import sigma_clip, mad_std
from astropy.utils.exceptions import AstropyWarning

warnings.simplefilter('ignore', category=AstropyWarning)

import bdsf
from exopipe.common import phase2_config as cfg

# Reuse VLSSr parser from existing module
from exopipe.phase2.extractor_pb_75 import parse_vlssr_text, load_catalog as load_vlssr_catalog


# =====================================================================
# DIAGNOSTIC FILENAME PREFIX
# =====================================================================

def _diag_prefix_from_workdir(work_dir):
    """Extract date_LST prefix from work directory path.

    work_dir is structured as .../lst/date/Science_*, e.g.
    /fast/gh/main/phase2/10h/2024-12-29/Science_20260404_191323

    Returns a prefix string like '2024-12-29_10h_', or '' if the
    path doesn't match the expected structure.
    """
    import re
    path = work_dir.rstrip(os.sep)
    m = re.search(r'/(\d+h)/(\d{4}-\d{2}-\d{2})/', path + '/')
    if m:
        return f"{m.group(2)}_{m.group(1)}_"
    return ""


# =====================================================================
# SOURCE EXTRACTION
# =====================================================================

# Elevation floor for anchor extraction (degrees), set per-LST by
# run_cascade_dewarping from cfg.ANCHOR_MIN_ELEVATION_DEG. Galactic-
# plane LSTs (13h family) wedge BDSF for hours on low-elevation diffuse
# emission that can never yield anchors — blanking it BEFORE extraction
# fixes both the runtime and the "0 anchors" failures (2026-06-11).
ANCHOR_EL_CUT = None
# East-side elevation floor (Gregg 2026-08-28): see
# cfg.ANCHOR_EAST_MIN_ELEVATION_DEG — blanks east-of-meridian pixels below
# the listed elevation at plane-rising LSTs. Set alongside ANCHOR_EL_CUT.
ANCHOR_EAST_EL_CUT = None
# Island size cap in BEAM AREAS for anchor extraction (Gregg approved
# 2026-08-28): BDSF maxpix_isl computed per image from the header beam.
# Discards the enormous diffuse plane islands BEFORE fitting (the measured
# ~24-min-per-interval sink at 13h); anchors are unresolved by contract,
# so no anchor and no sky is lost. Set per-LST via
# cfg.ANCHOR_MAX_ISLAND_BEAM_AREAS in run_cascade_dewarping.
ANCHOR_MAXPIX_BEAM_AREAS = None


def _anchor_maxpix(image_path):
    """maxpix_isl for this image: ANCHOR_MAXPIX_BEAM_AREAS x the header
    beam area in pixels, or None when the cap is not configured."""
    if ANCHOR_MAXPIX_BEAM_AREAS is None:
        return None
    try:
        hdr = fits.getheader(image_path)
        cd = abs(hdr.get('CDELT1', 0.037))
        bmaj = float(hdr.get('BMAJ', 5 * cd))
        bmin = float(hdr.get('BMIN', bmaj))
        beam_px = np.pi * bmaj * bmin / (4.0 * np.log(2.0)) / cd ** 2
        return int(max(ANCHOR_MAXPIX_BEAM_AREAS * beam_px, 50))
    except Exception:
        return None

# Anchor fallback order: when the primary 73 MHz anchor is missing (or yields
# too few VLSSr anchors), fall back to the next band nearest the VLSSr reference
# (74 MHz), staying within the high-frequency (Blue) arm so the spectral
# extrapolation to VLSSr stays small. Order: 73 -> 78 -> 69 -> 82 -> 64.
_VLSSR_REF_MHZ = 74.0
# Bands eligible as fallback anchors: the high-frequency (Blue) arm centred on
# 73 MHz, EXCLUDING 82 MHz (typically lower quality). Ordered by nearness to the
# VLSSr reference (74 MHz): 73 -> 78 -> 69 -> 64. (82 is still dewarped as a
# normal cascade subband; it is just not used as the anchor.)
_ANCHOR_EXCLUDE = {'82MHz'}
_anchor_idx = cfg.ALL_SUBBANDS.index(cfg.ANCHOR_SUBBAND)
_blue_arm = cfg.ALL_SUBBANDS[max(0, _anchor_idx - 2):_anchor_idx + 3]
ANCHOR_FALLBACK = sorted(
    [sb for sb in _blue_arm if sb not in _ANCHOR_EXCLUDE],
    key=lambda sb: abs(int(str(sb).replace('MHz', '')) - _VLSSR_REF_MHZ))


def _elevation_masked_copy(image_path, min_el_deg, logger,
                           east_min_el_deg=None):
    """Write a copy with pixels below min_el_deg blanked; optionally also
    blank EAST-of-meridian pixels below east_min_el_deg (Gregg 2026-08-28:
    plane-rising LSTs — the Galactic emission climbs the eastern sky above
    the ring floor and wedges BDSF).

    Images are zenith-centered SIN projections, so elevation is purely
    radial: sin(zenith_angle) = r_pix * cdelt (in radians). East = the
    rising side: hour angle < 0, i.e. RA east of the zenith meridian
    (CRVAL1 = zenith RA = LST for these images).
    """
    with fits.open(image_path) as h:
        data = h[0].data.copy()
        hdr = h[0].header
    cdelt = abs(hdr.get('CDELT1', 0.037))
    img2d = data.squeeze()
    ny, nx = img2d.shape[-2], img2d.shape[-1]
    yy, xx = np.mgrid[0:ny, 0:nx]
    r_deg = np.sqrt((xx - nx / 2) ** 2 + (yy - ny / 2) ** 2) * cdelt
    r_max = np.degrees(np.sin(np.radians(90.0 - min_el_deg)))
    mask = r_deg > r_max
    tag = f'.elcut{int(min_el_deg)}'
    if east_min_el_deg is not None:
        try:
            from astropy.wcs import WCS as _W
            w = _W(hdr).celestial
            ra_grid, _dec = w.wcs_pix2world(xx, yy, 0)
            lst_ra = float(hdr.get('CRVAL1', 0.0))
            east = np.sin(np.radians(ra_grid - lst_ra)) > 0
            r_max_e = np.degrees(np.sin(np.radians(90.0 - east_min_el_deg)))
            mask |= east & (r_deg > r_max_e)
            tag += f'e{int(east_min_el_deg)}'
        except Exception as e:
            logger.warning(f"  [BDSF] east elevation cut failed ({e}) — "
                           f"ring cut only")
    data[..., mask] = np.nan
    out = image_path.replace('.fits', f'{tag}.fits')
    fits.writeto(out, data, hdr, overwrite=True)
    logger.info(f"  [BDSF] elevation cut {min_el_deg:.0f} deg"
                + (f" + east<{east_min_el_deg:.0f} deg"
                   if east_min_el_deg is not None else "")
                + f": blanked {mask.mean():.0%} of field before extraction")
    return out


def extract_sources(image_path, logger):
    """Run BDSF on a single image. Returns DataFrame with ra, dec,
    flux_peak, maj, min columns, or empty DataFrame on failure."""
    logger.info(f"  [BDSF] Extracting from {os.path.basename(image_path)} "
                f"(thresh_pix={cfg.BDSF_THRESH_PIX:.0f})...")
    bdsf_input = image_path
    tmp_cut = None
    if ANCHOR_EL_CUT is not None:
        try:
            tmp_cut = _elevation_masked_copy(image_path, ANCHOR_EL_CUT,
                                             logger, ANCHOR_EAST_EL_CUT)
            bdsf_input = tmp_cut
        except Exception as e:
            logger.warning(f"  [BDSF] elevation mask failed ({e}) — "
                           f"using full image")
    try:
        _mp = _anchor_maxpix(bdsf_input)
        _mp_kw = {'maxpix_isl': _mp} if _mp else {}
        # PyBDSF sizes its worker pool from the machine's CPU count, ignoring the process's core pinning; many
        # parallel extractions then oversubscribe the pinned cores. BDSF_NCORES (a runtime setting) caps it; unset =
        # PyBDSF's default (unchanged behaviour). Results do not depend on it.
        if os.environ.get('BDSF_NCORES'):
            _mp_kw['ncores'] = int(os.environ['BDSF_NCORES'])
        if _mp:
            logger.info(f"  [BDSF] island cap: maxpix_isl={_mp} "
                        f"({ANCHOR_MAXPIX_BEAM_AREAS:.0f} beam areas)")
        img = bdsf.process_image(
            bdsf_input,
            thresh_pix=cfg.BDSF_THRESH_PIX,
            thresh_isl=cfg.BDSF_THRESH_ISL,
            adaptive_rms_box=True,
            quiet=True, **_mp_kw
        )
        sources = []
        for s in img.sources:
            if not np.isnan(s.posn_sky_max[0]):
                sources.append({
                    'ra': s.posn_sky_max[0],
                    'dec': s.posn_sky_max[1],
                    'flux_peak': s.peak_flux_max,
                    'maj': getattr(s, 'maj_axis', 0.0),
                    'min': getattr(s, 'min_axis', 0.0),
                })
        if not sources:
            logger.warning(f"  [BDSF] No sources found in {os.path.basename(image_path)}")
            return pd.DataFrame()
        df = pd.DataFrame(sources)
        logger.info(f"  [BDSF] Found {len(df)} sources")
        return df
    except Exception as e:
        logger.error(f"  [BDSF] Extraction failed: {e}")
        return pd.DataFrame()
    finally:
        if tmp_cut and os.path.exists(tmp_cut):
            try:
                os.remove(tmp_cut)
            except OSError:
                pass


# =====================================================================
# CROSS-MATCHING
# =====================================================================

def cross_match_vlssr(df_lwa, vlssr_cat, match_radius, flux_min, flux_max,
                      logger, beam_deg=None):
    """Cross-match LWA extraction against VLSSr catalog (anchor step).

    No spectral extrapolation needed: 73 MHz vs 74 MHz.
    Filters both catalogs to unresolved sources before matching.

    Returns dict with ra_lwa, dec_lwa, ra_cat, dec_cat arrays,
    or None if fewer than MIN_WARP_ANCHORS matched.
    """
    if df_lwa.empty or vlssr_cat is None:
        return None

    # Filter resolved VLSSr sources
    cat_mask = vlssr_cat['maj'] < cfg.VLSS_MAX_SIZE_DEG
    cat_ra_filt = vlssr_cat['ra'][cat_mask]
    cat_dec_filt = vlssr_cat['dec'][cat_mask]
    cat_flux_filt = vlssr_cat['flux'][cat_mask]
    cat_coords_filt = SkyCoord(cat_ra_filt, cat_dec_filt, unit='deg')

    n_cat_rejected = np.sum(~cat_mask)
    logger.info(f"  [VLSSr] Catalog: {len(vlssr_cat['ra'])} total, "
                f"{n_cat_rejected} resolved rejected, "
                f"{np.sum(cat_mask)} unresolved")

    # Filter resolved LWA sources  reject sources with fitted major axis
    # larger than 1.5× the synthesized beam
    lwa_unresolved = df_lwa.copy()
    if beam_deg is not None and beam_deg > 0 and 'maj' in df_lwa.columns:
        size_limit = beam_deg * 1.5
        size_mask = df_lwa['maj'] < size_limit
        lwa_unresolved = df_lwa[size_mask].reset_index(drop=True)
        n_lwa_rejected = np.sum(~size_mask)
        logger.info(f"  [LWA] Sources: {len(df_lwa)} total, "
                    f"{n_lwa_rejected} resolved rejected "
                    f"(maj > {size_limit*3600:.0f}\"), "
                    f"{len(lwa_unresolved)} unresolved")

    if lwa_unresolved.empty:
        return None

    # Cross-match
    lwa_coords = SkyCoord(lwa_unresolved['ra'].values,
                          lwa_unresolved['dec'].values, unit='deg')
    idx, d2d, _ = lwa_coords.match_to_catalog_sky(cat_coords_filt)

    # Position filter
    mask_pos = d2d < match_radius

    # Flux ratio filter (no spectral correction  frequencies nearly identical)
    lwa_flux = lwa_unresolved['flux_peak'].values
    cat_flux_matched = cat_flux_filt[idx]
    flux_ratio = lwa_flux / cat_flux_matched
    mask_flux = (flux_ratio > flux_min) & (flux_ratio < flux_max)

    mask = mask_pos & mask_flux
    n_matched = np.sum(mask)

    logger.info(f"  [VLSSr Match] Position matches: {np.sum(mask_pos)}, "
                f"Flux-filtered: {n_matched} "
                f"(ratio window [{flux_min:.2f}, {flux_max:.2f}])")

    if n_matched < cfg.MIN_WARP_ANCHORS:
        logger.warning(f"  [VLSSr Match] Only {n_matched} anchors "
                       f"(need {cfg.MIN_WARP_ANCHORS})  dewarping will fail")
        return None

    return {
        'ra_lwa': lwa_unresolved['ra'].values[mask],
        'dec_lwa': lwa_unresolved['dec'].values[mask],
        'ra_cat': cat_ra_filt[idx[mask]],
        'dec_cat': cat_dec_filt[idx[mask]],
        'flux_peak': lwa_unresolved['flux_peak'].values[mask],
    }


def cross_match_subbands(df_target, df_ref, match_radius, flux_min, flux_max,
                         logger, beam_deg=None, flux_center=None):
    """Cross-match target subband extraction against reference (dewarped
    neighbor) extraction.

    No spectral extrapolation: frequency gap is ~45 MHz.
    Filters both catalogs to unresolved sources before matching.

    flux_center='measured-median' (cfg.CASCADE_FLUX_CENTER, Gregg 2026-10-01):
    the flux window is [flux_min, flux_max] x the median target/reference
    peak-flux ratio of the position matches, instead of x 1.0 - the subband
    flux scales differ until the absolute flux scale is applied later.

    Returns dict with ra_target, dec_target, ra_ref, dec_ref arrays (and
    flux_ratio_median, the window centre used), or None if fewer than
    MIN_WARP_ANCHORS matched.
    """
    if df_target.empty or df_ref.empty:
        return None

    # Filter resolved sources from both target and reference
    tgt_filt = df_target.copy()
    ref_filt = df_ref.copy()

    if beam_deg is not None and beam_deg > 0:
        size_limit = beam_deg * 1.5
        if 'maj' in df_target.columns:
            tgt_mask = df_target['maj'] < size_limit
            tgt_filt = df_target[tgt_mask].reset_index(drop=True)
            logger.info(f"  [Target] {len(df_target)} ? {len(tgt_filt)} unresolved "
                        f"(maj < {size_limit*3600:.0f}\")")
        if 'maj' in df_ref.columns:
            ref_mask = df_ref['maj'] < size_limit
            ref_filt = df_ref[ref_mask].reset_index(drop=True)
            logger.info(f"  [Ref]    {len(df_ref)} ? {len(ref_filt)} unresolved")

    if tgt_filt.empty or ref_filt.empty:
        return None

    tgt_coords = SkyCoord(tgt_filt['ra'].values, tgt_filt['dec'].values,
                          unit='deg')
    ref_coords = SkyCoord(ref_filt['ra'].values, ref_filt['dec'].values,
                          unit='deg')

    idx, d2d, _ = tgt_coords.match_to_catalog_sky(ref_coords)

    mask_pos = d2d < match_radius

    # Flux ratio filter
    tgt_flux = tgt_filt['flux_peak'].values
    ref_flux = ref_filt['flux_peak'].values[idx]
    flux_ratio = tgt_flux / ref_flux
    centre = 1.0
    if flux_center == 'measured-median':
        if not np.any(mask_pos):
            logger.warning("  [Cascade Match] no position matches")
            return None
        centre = float(np.median(flux_ratio[mask_pos]))
        logger.info(f"  [Cascade Match] flux window centred on the median ratio "
                    f"{centre:.3f} of {np.sum(mask_pos)} position matches: "
                    f"[{flux_min * centre:.3f}, {flux_max * centre:.3f}]")
    elif flux_center is not None:
        raise ValueError(f"unknown flux_center {flux_center!r}")
    mask_flux = (flux_ratio > flux_min * centre) & (flux_ratio < flux_max * centre)

    mask = mask_pos & mask_flux
    n_matched = np.sum(mask)

    logger.info(f"  [Cascade Match] Position matches: {np.sum(mask_pos)}, "
                f"Flux-filtered: {n_matched}")

    if n_matched < cfg.MIN_WARP_ANCHORS:
        logger.warning(f"  [Cascade Match] Only {n_matched} anchors "
                       f"(need {cfg.MIN_WARP_ANCHORS})")
        return None

    return {
        'ra_lwa': tgt_filt['ra'].values[mask],
        'dec_lwa': tgt_filt['dec'].values[mask],
        'ra_cat': ref_filt['ra'].values[idx[mask]],
        'dec_cat': ref_filt['dec'].values[idx[mask]],
        'flux_peak': tgt_filt['flux_peak'].values[mask],
        'flux_ratio_median': centre,
    }


# =====================================================================
# OUTLIER REJECTION
# =====================================================================

def reject_outlier_anchors(anchors, wcs, shape, logger):
    """Iterative sigma-clipping of warp anchor offsets.

    Fits a preliminary TPS warp screen, computes residuals at anchor
    positions, and sigma-clips outliers. Repeats for OUTLIER_CLIP_ITERS.

    Returns cleaned anchors dict (same format as input), or None if
    too few anchors survive.
    """
    ra_lwa = anchors['ra_lwa'].copy()
    dec_lwa = anchors['dec_lwa'].copy()
    ra_cat = anchors['ra_cat'].copy()
    dec_cat = anchors['dec_cat'].copy()

    n_start = len(ra_lwa)

    for iteration in range(cfg.OUTLIER_CLIP_ITERS):
        n_current = len(ra_lwa)
        if n_current < cfg.MIN_WARP_ANCHORS:
            break

        # Convert to pixel coordinates
        mx, my = wcs.all_world2pix(ra_lwa, dec_lwa, 0)
        tx, ty = wcs.all_world2pix(ra_cat, dec_cat, 0)

        dx = mx - tx  # LWA measured - catalog truth
        dy = my - ty

        # Fit preliminary TPS to current anchors
        try:
            points = np.column_stack([ty, tx])
            rbf_x = RBFInterpolator(
                points, dx, kernel='thin_plate_spline',
                smoothing=cfg.WARP_RBF_SMOOTHING * 2  # extra smoothing for outlier step
            )
            rbf_y = RBFInterpolator(
                points, dy, kernel='thin_plate_spline',
                smoothing=cfg.WARP_RBF_SMOOTHING * 2
            )

            # Predicted offsets at anchor positions
            pred_dx = rbf_x(points)
            pred_dy = rbf_y(points)

            # Residuals
            resid_x = dx - pred_dx
            resid_y = dy - pred_dy
            resid_mag = np.sqrt(resid_x**2 + resid_y**2)

            # Sigma clip on residual magnitude
            clipped = sigma_clip(resid_mag, sigma=cfg.OUTLIER_CLIP_SIGMA,
                                 maxiters=1, stdfunc=mad_std)
            keep = ~clipped.mask

            n_rejected = np.sum(~keep)
            if n_rejected == 0:
                break

            logger.info(f"  [Outlier] Iter {iteration+1}: rejected {n_rejected} "
                        f"anchors (median resid = {np.median(resid_mag):.2f} pix)")

            ra_lwa = ra_lwa[keep]
            dec_lwa = dec_lwa[keep]
            ra_cat = ra_cat[keep]
            dec_cat = dec_cat[keep]

        except Exception as e:
            logger.warning(f"  [Outlier] TPS fit failed in iter {iteration+1}: {e}")
            break

    n_final = len(ra_lwa)
    n_rejected_total = n_start - n_final
    if n_rejected_total > 0:
        logger.info(f"  [Outlier] Total rejected: {n_rejected_total}/{n_start} "
                    f"? {n_final} anchors remain")

    if n_final < cfg.MIN_WARP_ANCHORS:
        logger.warning(f"  [Outlier] Only {n_final} anchors survive "
                       f"(need {cfg.MIN_WARP_ANCHORS})")
        return None

    return {
        'ra_lwa': ra_lwa,
        'dec_lwa': dec_lwa,
        'ra_cat': ra_cat,
        'dec_cat': dec_cat,
    }


# =====================================================================
# WARP SCREEN COMPUTATION
# =====================================================================

def compute_warp_screen(anchors, wcs, shape, logger,
                        method='rbf', rbf_kernel='thin_plate_spline',
                        rbf_smoothing=1.5, rbf_epsilon=None,
                        snr_weighting=False):
    """Build 2D warp screen from cleaned anchors.

    Supports RBF (configurable kernel), Zernike polynomial, or linear
    interpolation.

    Parameters
    ----------
    method : str
        'rbf', 'zernike', or 'linear'.
    rbf_kernel : str
        Kernel for RBFInterpolator (only used when method='rbf').
    rbf_smoothing : float
        Smoothing parameter for RBFInterpolator (only used when method='rbf').
    rbf_epsilon : float or None
        Shape parameter for kernels that require it (multiquadric,
        inverse_multiquadric, gaussian). None = auto-compute from mean
        nearest-neighbor distance between anchors.
    snr_weighting : bool
        If True, use per-anchor smoothing proportional to 1/flux².
        Bright sources (small position error) get low smoothing;
        faint sources (large position error) get high smoothing.

    Returns (screen_x, screen_y) arrays of shape `shape`, or (None, None)
    on failure.
    """
    mx, my = wcs.all_world2pix(anchors['ra_lwa'], anchors['dec_lwa'], 0)
    tx, ty = wcs.all_world2pix(anchors['ra_cat'], anchors['dec_cat'], 0)

    dx = mx - tx
    dy = my - ty

    ny, nx = shape
    method_label = f"{method}" if method != 'rbf' else f"rbf/{rbf_kernel}"
    logger.info(f"  [Warp] Building {method_label} screen from "
                f"{len(dx)} anchors for {nx}×{ny} image...")

    try:
        points = np.column_stack([ty, tx])

        if method == 'zernike':
            raise RuntimeError('zernike engine retired to attic (§12.1)')
            model = fit_zernike_warp(
                tx, ty, dx, dy,
                image_shape=shape,
                max_order=cfg.WARP_ZERNIKE_ORDER,
            )

            grid_y, grid_x = np.mgrid[0:ny, 0:nx]
            screen_x, screen_y = evaluate_zernike_warp(
                model, grid_x, grid_y)

            logger.info(f"  [Warp] Zernike screen computed. Fit residual: "
                        f"{model['residual_median']:.3f} pix "
                        f"(MAD={model['residual_mad']:.3f} pix)")
            logger.info(f"  [Warp] Median shift: "
                        f"dx={np.median(screen_x):.2f}, "
                        f"dy={np.median(screen_y):.2f} pix")

            return screen_x, screen_y

        elif method == 'rbf':
            # Kernels that require an epsilon shape parameter
            needs_epsilon = rbf_kernel in ('multiquadric',
                                           'inverse_multiquadric', 'gaussian')
            rbf_kwargs = dict(kernel=rbf_kernel)

            # Smoothing: scalar or per-point array
            if snr_weighting and 'flux_peak' in anchors:
                flux = np.asarray(anchors['flux_peak'], dtype=float)
                flux = np.clip(flux, 1e-6, None)  # avoid division by zero
                flux_ref = np.median(flux)
                # Per-point smoothing ∝ 1/flux² (= ∝ σ²_pos)
                # Median-flux source gets base_smoothing; bright sources
                # get less (screen passes closer); faint sources get more
                # (screen not forced to noisy positions).
                smoothing_array = rbf_smoothing * (flux_ref / flux)**2
                smoothing_array = np.clip(smoothing_array,
                                          rbf_smoothing * 0.01,
                                          rbf_smoothing * 100)
                rbf_kwargs['smoothing'] = smoothing_array

                bright_sm = np.min(smoothing_array)
                faint_sm = np.max(smoothing_array)
                logger.info(f"  [Warp] SNR-weighted smoothing: "
                            f"bright={bright_sm:.3f}, "
                            f"median={rbf_smoothing:.3f}, "
                            f"faint={faint_sm:.1f} "
                            f"(flux range {flux.min():.1f}"
                            f"-{flux.max():.1f} Jy)")
            else:
                rbf_kwargs['smoothing'] = rbf_smoothing

            if needs_epsilon:
                if rbf_epsilon is None:
                    # Auto-compute: mean nearest-neighbor distance
                    from scipy.spatial import cKDTree
                    tree = cKDTree(points)
                    dists, _ = tree.query(points, k=2)
                    rbf_epsilon = float(np.mean(dists[:, 1]))
                    logger.info(f"  [Warp] Auto epsilon={rbf_epsilon:.2f} "
                                f"(mean nearest-neighbor distance)")
                else:
                    logger.info(f"  [Warp] Using epsilon={rbf_epsilon:.2f}")
                rbf_kwargs['epsilon'] = rbf_epsilon

            rbf_x = RBFInterpolator(points, dx, **rbf_kwargs)
            rbf_y = RBFInterpolator(points, dy, **rbf_kwargs)
        else:
            # Fallback: linear griddata (legacy behavior)
            from scipy.interpolate import griddata, NearestNDInterpolator
            grid_y, grid_x = np.mgrid[0:ny, 0:nx]
            screen_x = griddata(
                (ty, tx), dx, (grid_y, grid_x),
                method='linear', fill_value=0.0
            )
            screen_y = griddata(
                (ty, tx), dy, (grid_y, grid_x),
                method='linear', fill_value=0.0
            )
            mask_nan = np.isnan(screen_x)
            if np.any(mask_nan):
                nn_x = NearestNDInterpolator(list(zip(ty, tx)), dx)
                nn_y = NearestNDInterpolator(list(zip(ty, tx)), dy)
                screen_x[mask_nan] = nn_x(grid_y[mask_nan], grid_x[mask_nan])
                screen_y[mask_nan] = nn_y(grid_y[mask_nan], grid_x[mask_nan])
            return screen_x, screen_y

        # Evaluate RBF on coarse grid then interpolate to full resolution.
        # Direct evaluation on 16M points with >2000 anchors is very slow.
        # A 512×512 coarse grid captures the smooth ionospheric structure
        # and bilinear upsampling preserves it with negligible error.
        coarse_factor = max(1, min(ny, nx) // 512)
        cy_coarse = np.arange(0, ny, coarse_factor)
        cx_coarse = np.arange(0, nx, coarse_factor)
        cg_y, cg_x = np.meshgrid(cy_coarse, cx_coarse, indexing='ij')
        eval_pts = np.column_stack([cg_y.ravel(), cg_x.ravel()])

        logger.info(f"  [Warp] Evaluating RBF ({rbf_kernel}) on "
                    f"{len(cy_coarse)}×{len(cx_coarse)} "
                    f"coarse grid (factor {coarse_factor})...")

        screen_x_coarse = rbf_x(eval_pts).reshape(cg_y.shape)
        screen_y_coarse = rbf_y(eval_pts).reshape(cg_y.shape)

        # Bilinear interpolation to full resolution
        from scipy.ndimage import zoom as ndimage_zoom
        zoom_y = ny / screen_x_coarse.shape[0]
        zoom_x = nx / screen_x_coarse.shape[1]
        screen_x = ndimage_zoom(screen_x_coarse, (zoom_y, zoom_x), order=1)
        screen_y = ndimage_zoom(screen_y_coarse, (zoom_y, zoom_x), order=1)

        # Trim to exact shape (zoom can overshoot by 1 pixel)
        screen_x = screen_x[:ny, :nx]
        screen_y = screen_y[:ny, :nx]

        logger.info(f"  [Warp] Screen computed. Median shift: "
                    f"dx={np.median(screen_x):.2f}, dy={np.median(screen_y):.2f} pix")

        return screen_x, screen_y

    except Exception as e:
        logger.error(f"  [Warp] Screen computation failed: {e}")
        return None, None


# =====================================================================
# WARP APPLICATION
# =====================================================================

def _save_warp_screen(screen_x, screen_y, wcs, subband, diag_dir, method,
                      kernel, n_anchors, logger, work_dir=None):
    """Save warp screen as a multi-extension FITS for later reuse.

    Saves to QA/Dewarp_Diagnostics/maps/{prefix}{subband}_warp_screen.fits
    with two extensions: SCREEN_X and SCREEN_Y (pixel offsets).
    """
    prefix = _diag_prefix_from_workdir(work_dir) if work_dir else ""
    maps_dir = os.path.join(diag_dir, "maps")
    os.makedirs(maps_dir, exist_ok=True)
    out_path = os.path.join(maps_dir, f"{prefix}{subband}_warp_screen.fits")

    hdr = fits.Header()
    hdr['SUBBAND'] = (subband, 'Subband label')
    hdr['METHOD'] = (method, 'Dewarping method')
    if kernel:
        hdr['KERNEL'] = (kernel, 'RBF kernel')
    hdr['NANCHOR'] = (n_anchors, 'Number of warp anchors')
    hdr['NAXIS1'] = screen_x.shape[1]
    hdr['NAXIS2'] = screen_x.shape[0]
    hdr.update(wcs.to_header())

    hdu_x = fits.ImageHDU(data=screen_x.astype(np.float32), name='SCREEN_X')
    hdu_x.header.update(hdr)
    hdu_y = fits.ImageHDU(data=screen_y.astype(np.float32), name='SCREEN_Y')
    hdu_y.header.update(hdr)

    hdul = fits.HDUList([fits.PrimaryHDU(), hdu_x, hdu_y])
    hdul.writeto(out_path, overwrite=True)
    logger.info(f"  [Warp] Saved warp screen: {os.path.basename(out_path)} "
                f"({screen_x.shape[0]}×{screen_x.shape[1]})")


def apply_warp(data, screen_x, screen_y):
    """Resample a 2D image using the warp screen.

    For each output pixel (x, y), samples the input at
    (x + screen_x[y,x], y + screen_y[y,x]).

    Out-of-bounds pixels are set to NaN (not zero) so that downstream
    tools (BDSF, photometry) naturally exclude them as blanked pixels.

    Returns warped 2D array, or None on shape mismatch.
    """
    ny, nx = data.shape
    sy, sx = screen_x.shape
    if ny != sy or nx != sx:
        return None
    grid_y, grid_x = np.mgrid[0:ny, 0:nx]
    return map_coordinates(
        data, [grid_y + screen_y, grid_x + screen_x],
        order=1, mode='constant', cval=np.nan
    )


# =====================================================================
# DIAGNOSTICS
# =====================================================================

def save_diagnostics(anchors, screen_x, screen_y, wcs, shape, subband,
                     diag_dir, ref_label, logger, work_dir=None):
    """Save diagnostic plots and warp statistics JSON."""
    os.makedirs(diag_dir, exist_ok=True)
    prefix = _diag_prefix_from_workdir(work_dir) if work_dir else ""
    base_name = f"{prefix}{subband}_warp"

    mx, my = wcs.all_world2pix(anchors['ra_lwa'], anchors['dec_lwa'], 0)
    tx, ty = wcs.all_world2pix(anchors['ra_cat'], anchors['dec_cat'], 0)
    dx = mx - tx
    dy = my - ty

    # 1. Distortion magnitude map
    try:
        scales = wcs.proj_plane_pixel_scales()
        scale_deg = np.mean([s.value if hasattr(s, 'value') else s for s in scales])
        scale_arcmin = scale_deg * 60.0
    except Exception:
        scale_arcmin = 0.03125 * 60.0  # fallback

    shift_mag = np.sqrt(screen_x**2 + screen_y**2) * scale_arcmin

    # Mask outside anchor coverage (TPS extrapolation is unreliable there)
    ny, nx = shape
    cy_img, cx_img = ny / 2, nx / 2
    yy, xx = np.mgrid[0:ny, 0:nx]
    dist_from_center = np.sqrt((xx - cx_img)**2 + (yy - cy_img)**2)
    if len(tx) > 10:
        anchor_r = np.percentile(
            np.sqrt((tx - cx_img)**2 + (ty - cy_img)**2), 95)
    else:
        anchor_r = min(ny, nx) * 0.45
    horizon_mask = dist_from_center > anchor_r * 1.1

    shift_display = shift_mag.copy()
    shift_display[horizon_mask] = np.nan

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(shift_display, origin='lower', cmap='inferno',
                   vmin=0, vmax=np.nanpercentile(shift_display, 99))
    plt.colorbar(im, ax=ax, label='Shift (arcmin)')
    ax.set_title(f'Ionospheric Distortion: {subband} (ref: {ref_label})')
    ax.plot(tx, ty, 'c+', ms=4, markeredgewidth=0.3, alpha=0.4)

    plt.savefig(os.path.join(diag_dir, f"distortion_map_{base_name}.png"),
                dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Compute ionospheric disturbance metric
    #    Fit a 2nd-order polynomial (6 terms) to the displacement field
    #    to capture the bulk TEC gradient. The RMS of the residuals in
    #    arcmin measures small-scale structure: smooth ionosphere -> small
    #    residuals, structured ionosphere -> large residuals.
    disturbance_rms_arcmin = None
    disturbance_quality = None
    try:
        if len(tx) >= 10:
            # Normalize positions to [-1, 1] for numerical stability
            cx_norm = (tx - cx_img) / (nx / 2)
            cy_norm = (ty - cy_img) / (ny / 2)

            # Vandermonde matrix for 2nd-order: 1, x, y, x², xy, y²
            V = np.column_stack([
                np.ones(len(cx_norm)),
                cx_norm, cy_norm,
                cx_norm**2, cx_norm * cy_norm, cy_norm**2
            ])

            # Least-squares fit
            coeffs_x, _, _, _ = np.linalg.lstsq(V, dx, rcond=None)
            coeffs_y, _, _, _ = np.linalg.lstsq(V, dy, rcond=None)

            # Residuals
            resid_x = dx - V @ coeffs_x
            resid_y = dy - V @ coeffs_y
            resid_mag_arcmin = np.sqrt(resid_x**2 + resid_y**2) * scale_arcmin

            disturbance_rms_arcmin = float(np.sqrt(np.mean(resid_mag_arcmin**2)))

            if disturbance_rms_arcmin < 0.10:
                disturbance_quality = "Excellent (smooth)"
            elif disturbance_rms_arcmin < 0.20:
                disturbance_quality = "Good"
            elif disturbance_rms_arcmin < 0.40:
                disturbance_quality = "Moderate (structured)"
            else:
                disturbance_quality = "Poor (highly structured)"

            logger.info(f"  [Diag] {subband}: disturbance RMS = "
                        f"{disturbance_rms_arcmin:.3f}' "
                        f"({disturbance_quality})")
    except Exception as e:
        logger.warning(f"  [Diag] Disturbance metric failed: {e}")

    # 3. Anchor offset quiver plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor('black')
    q = ax.quiver(tx, ty, dx, dy, shift_mag[ty.astype(int) % shape[0],
                  tx.astype(int) % shape[1]],
                  cmap='plasma', scale=None, alpha=0.8)
    plt.colorbar(q, ax=ax, label='Shift (arcmin)')
    ax.set_xlim(0, shape[1])
    ax.set_ylim(0, shape[0])
    ax.set_aspect('equal')
    title_str = f'Warp Anchors: {subband} vs {ref_label} ({len(dx)} sources)'
    if disturbance_rms_arcmin is not None:
        title_str += f'\nDisturbance: {disturbance_quality} (RMS {disturbance_rms_arcmin:.2f}\')'
    ax.set_title(title_str, fontsize=11)
    plt.savefig(os.path.join(diag_dir, f"anchor_vectors_{base_name}.png"),
                dpi=150, bbox_inches='tight')
    plt.close()

    # 4. Warp statistics JSON
    ny, nx = screen_x.shape
    q = 4
    sy1, sy2 = ny // q, ny - ny // q
    sx1, sx2 = nx // q, nx - nx // q
    inner_sx = screen_x[sy1:sy2, sx1:sx2]
    inner_sy = screen_y[sy1:sy2, sx1:sx2]
    inner_mag = np.sqrt(inner_sx**2 + inner_sy**2)
    valid = inner_mag > 0

    freq_str = subband.replace('MHz', '')
    try:
        freq_mhz = float(freq_str)
    except ValueError:
        freq_mhz = 0.0

    if np.any(valid):
        stats = {
            'freq_mhz': freq_mhz,
            'ref_label': ref_label,
            'n_anchors': len(dx),
            'median_dx_pix': float(np.median(inner_sx[valid])),
            'median_dy_pix': float(np.median(inner_sy[valid])),
            'median_shift_pix': float(np.median(inner_mag[valid])),
            'std_shift_pix': float(np.std(inner_mag[valid])),
            'max_shift_pix': float(np.max(inner_mag[valid])),
            'median_shift_arcmin': float(np.median(inner_mag[valid]) * scale_arcmin),
            'std_shift_arcmin': float(np.std(inner_mag[valid]) * scale_arcmin),
            'max_shift_arcmin': float(np.max(inner_mag[valid]) * scale_arcmin),
            'pix_scale_arcmin': float(scale_arcmin),
            'n_valid_pixels': int(np.sum(valid)),
        }
        if disturbance_rms_arcmin is not None:
            stats['disturbance_rms_arcmin'] = disturbance_rms_arcmin
            stats['disturbance_quality'] = disturbance_quality
    else:
        stats = {'freq_mhz': freq_mhz, 'ref_label': ref_label,
                 'n_anchors': len(dx), 'error': 'no valid inner pixels'}

    stats_path = os.path.join(diag_dir, f"warp_stats_{prefix}{subband}.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    if 'median_shift_arcmin' in stats:
        logger.info(f"  [Diag] {subband}: median shift = "
                    f"{stats['median_shift_arcmin']:.2f}' "
                    f"(max {stats['max_shift_arcmin']:.2f}'), "
                    f"{stats['n_anchors']} anchors")


# =====================================================================
# FIND DEEP IMAGE
# =====================================================================

def find_deep_pb_image(work_dir, subband, logger):
    """Find the deep tapered PB-corrected Stokes I image for a subband.

    Used for the anchor step (73 MHz vs VLSSr) where VLSSr has true fluxes.

    Searches: {work_dir}/{subband}/I/deep/*Taper*pbcorr*.fits
    Excludes NoTaper, dewarped, and already-processed variants.
    Returns path or None.
    """
    pat = os.path.join(work_dir, subband, "I", "deep", "*Taper*pbcorr*.fits")
    candidates = [f for f in glob.glob(pat)
                  if "NoTaper" not in os.path.basename(f)
                  and "_dewarped" not in os.path.basename(f)
                  and ".pybdsf" not in os.path.basename(f)]
    if not candidates:
        logger.warning(f"  No deep tapered pbcorr image found for {subband}")
        return None
    # Prefer Robust-0.75 (primary science image)
    robust075 = [f for f in candidates if "Robust-0.75" in os.path.basename(f)]
    if robust075:
        return sorted(robust075)[0]
    return sorted(candidates)[0]


def find_deep_raw_image(work_dir, subband, logger):
    """Find the deep tapered non-PB-corrected Stokes I image for a subband.

    Used for cascade steps where both target and reference must have
    consistent beam attenuation (so beam effects cancel in the flux ratio).

    Searches: {work_dir}/{subband}/I/deep/*Taper*image*.fits
    Excludes NoTaper, pbcorr, dewarped.
    Returns path or None.
    """
    pat = os.path.join(work_dir, subband, "I", "deep", "*Taper*image*.fits")
    candidates = [f for f in glob.glob(pat)
                  if "NoTaper" not in os.path.basename(f)
                  and "pbcorr" not in os.path.basename(f)
                  and "_dewarped" not in os.path.basename(f)
                  and ".pybdsf" not in os.path.basename(f)]
    if not candidates:
        logger.warning(f"  No deep tapered raw image found for {subband}")
        return None
    robust075 = [f for f in candidates if "Robust-0.75" in os.path.basename(f)]
    if robust075:
        return sorted(robust075)[0]
    return sorted(candidates)[0]


def find_all_images_to_warp(work_dir, subband, categories=None):
    """Find all FITS images that need dewarping for a subband.

    Returns list of paths. Only includes *image* and *pbcorr* files
    (not dirty, model, psf, residual, mask, or horizon-mask).
    Excludes already-dewarped files.

    Parameters
    ----------
    work_dir : str
    subband : str
    categories : list of str, optional
        Image categories to include, e.g. ['deep'] or ['deep', '10min'].
        If None, includes all categories.
    """
    base = os.path.join(work_dir, subband)
    all_fits = []
    cats = categories or ['deep', '10min']
    for pol in ['I', 'V']:
        for cat in cats:
            d = os.path.join(base, pol, cat)
            if not os.path.isdir(d):
                continue
            for f in glob.glob(os.path.join(d, "*.fits")):
                bn = os.path.basename(f)
                if "_dewarped" in bn:
                    continue
                if ".pybdsf" in bn:
                    continue
                if '-image' not in bn:
                    continue
                all_fits.append(f)
    return all_fits


# =====================================================================
# DEWARP A SINGLE SUBBAND
# =====================================================================

def dewarp_subband(work_dir, subband, ref_catalog, ref_label, is_anchor,
                   diag_dir, logger, method='rbf', rbf_kernel='thin_plate_spline',
                   rbf_smoothing=1.5, rbf_epsilon=None, snr_weighting=False,
                   warp_categories=None):
    """Dewarp a single subband against a reference catalog.

    Parameters
    ----------
    work_dir : str
        Phase 2 working directory (contains all subband dirs).
    subband : str
        Subband label, e.g. '73MHz'.
    ref_catalog : dict or DataFrame
        For anchor: VLSSr dict with 'ra', 'dec', 'flux', 'maj' arrays.
        For cascade: DataFrame from BDSF extraction of dewarped neighbor.
    ref_label : str
        Label for diagnostics, e.g. 'VLSSr' or '73MHz_dewarped'.
    is_anchor : bool
        True for the 73 MHz vs VLSSr step.
    diag_dir : str
        Path to Dewarp_Diagnostics/ directory.
    logger : logging.Logger
    method : str
        Interpolation method: 'rbf', 'zernike', or 'linear'.
    rbf_kernel : str
        Kernel for RBFInterpolator (only used when method='rbf').
    rbf_smoothing : float
        Smoothing parameter for RBFInterpolator.

    Returns
    -------
    (dewarped_catalog, success, anchor_data) : (DataFrame, bool, dict or None)
        dewarped_catalog is the BDSF extraction from the dewarped image
        (used as reference for the next cascade step).
        success is False if dewarping failed.
        anchor_data contains pixel-space anchor measurements for the
        global TEC fit: {freq_mhz, px_lwa, py_lwa, px_cat, py_cat, dx, dy, wcs, shape}
    """
    logger.info(f"--- Dewarping {subband} (ref: {ref_label}) ---")

    # 1. Find the extraction image  always PB-corrected so flux ratios
    #    reflect true source fluxes with the beam divided out. This matters
    #    for cascade steps where the primary beam is frequency-dependent.
    calc_img = find_deep_pb_image(work_dir, subband, logger)
    if calc_img is None:
        return pd.DataFrame(), False, None

    logger.info(f"  Extraction image: {os.path.basename(calc_img)}")

    # 2. Read image metadata (WCS, beam, shape)  used for extraction,
    #    cross-matching, and warp screen computation
    with fits.open(calc_img) as hdul:
        header = hdul[0].header
        wcs = WCS(header).celestial
        shape = hdul[0].data.squeeze().shape
        beam_deg = header.get('BMAJ', None)
        freq_hz = header.get('CRVAL3', None)
        if beam_deg:
            logger.info(f"  Beam: BMAJ = {beam_deg*60:.1f}' "
                        f"({beam_deg*3600:.0f}\")")

    # Parse frequency
    try:
        freq_mhz = float(subband.replace('MHz', ''))
    except ValueError:
        freq_mhz = freq_hz / 1e6 if freq_hz else cfg.ANCHOR_FREQ_MHZ

    # 3. Extract sources
    df_lwa = extract_sources(calc_img, logger)
    if df_lwa.empty:
        return pd.DataFrame(), False, None

    # 4. Cross-match (filtering both catalogs to unresolved sources)
    #    Match radius scales as (?_ref/?)² for cascade steps
    if is_anchor:
        match_radius = cfg.ANCHOR_MATCH_RADIUS
        anchors = cross_match_vlssr(
            df_lwa, ref_catalog,
            match_radius,
            cfg.ANCHOR_FLUX_RATIO_MIN,
            cfg.ANCHOR_FLUX_RATIO_MAX,
            logger,
            beam_deg=beam_deg
        )
    else:
        freq_scale = (cfg.ANCHOR_FREQ_MHZ / freq_mhz) ** cfg.CASCADE_MATCH_RADIUS_EXPONENT
        match_radius_arcmin = cfg.CASCADE_MATCH_RADIUS_BASE * freq_scale
        match_radius = match_radius_arcmin * u.arcmin
        logger.info(f"  Match radius: {match_radius_arcmin:.1f}' "
                    f"(base {cfg.CASCADE_MATCH_RADIUS_BASE}' × "
                    f"(73/{freq_mhz:.0f})^{cfg.CASCADE_MATCH_RADIUS_EXPONENT} "
                    f"= ×{freq_scale:.2f})")
        anchors = cross_match_subbands(
            df_lwa, ref_catalog,
            match_radius,
            cfg.CASCADE_FLUX_RATIO_MIN,
            cfg.CASCADE_FLUX_RATIO_MAX,
            logger,
            beam_deg=beam_deg,
            flux_center=cfg.CASCADE_FLUX_CENTER
        )

    if anchors is None:
        return pd.DataFrame(), False, None

    # 5. Reject outlier anchors
    anchors = reject_outlier_anchors(anchors, wcs, shape, logger)
    if anchors is None:
        return pd.DataFrame(), False, None

    # 6. Compute warp screen
    screen_x, screen_y = compute_warp_screen(anchors, wcs, shape, logger,
                                              method=method,
                                              rbf_kernel=rbf_kernel,
                                              rbf_smoothing=rbf_smoothing,
                                              rbf_epsilon=rbf_epsilon,
                                              snr_weighting=snr_weighting)
    if screen_x is None:
        return pd.DataFrame(), False, None

    # 6b. Store anchor pixel-space data for global TEC fit
    mx, my = wcs.all_world2pix(anchors['ra_lwa'], anchors['dec_lwa'], 0)
    tx, ty = wcs.all_world2pix(anchors['ra_cat'], anchors['dec_cat'], 0)
    anchor_data = {
        'freq_mhz': freq_mhz,
        'px_lwa': mx, 'py_lwa': my,
        'px_cat': tx, 'py_cat': ty,
        'dx': mx - tx, 'dy': my - ty,
        'wcs': wcs, 'shape': shape,
    }

    # 7. Save diagnostics
    save_diagnostics(anchors, screen_x, screen_y, wcs, shape, subband,
                     diag_dir, ref_label, logger, work_dir=work_dir)

    # 7b. Save warp screen for later use (e.g., standalone snapshot dewarping)
    _save_warp_screen(screen_x, screen_y, wcs, subband, diag_dir, method,
                      rbf_kernel, len(anchors), logger, work_dir=work_dir)

    # 8. Apply warp to deep images (+ snapshots if requested)
    cats = warp_categories or ['deep']
    files_to_warp = find_all_images_to_warp(work_dir, subband, categories=cats)
    n_warped = 0
    for fpath in files_to_warp:
        out_path = fpath.replace('.fits', '_dewarped.fits')
        if os.path.exists(out_path):
            n_warped += 1
            continue
        try:
            with fits.open(fpath) as hf:
                data = hf[0].data.squeeze()
                if data.shape == screen_x.shape:
                    warped = apply_warp(data, screen_x, screen_y)
                    if warped is not None:
                        fits.writeto(out_path, warped, hf[0].header,
                                     overwrite=True)
                        n_warped += 1
                else:
                    logger.warning(f"  Shape mismatch: {os.path.basename(fpath)} "
                                   f"{data.shape} vs screen {screen_x.shape}")
        except Exception as e:
            logger.warning(f"  Warp failed for {os.path.basename(fpath)}: {e}")

    logger.info(f"  Dewarped {n_warped}/{len(files_to_warp)} images for {subband}")

    # 9. Re-extract sources from the dewarped PB-corrected image
    #    (for next cascade step  pbcorr so flux ratios are true).
    #    Edge pixels are NaN from apply_warp(cval=NaN), which BDSF
    #    handles natively as blanked pixels.
    dewarped_pbcorr = calc_img.replace('.fits', '_dewarped.fits')
    if os.path.exists(dewarped_pbcorr):
        dewarped_catalog = extract_sources(dewarped_pbcorr, logger)
    else:
        dewarped_catalog = pd.DataFrame()

    # If re-extraction failed, fall back to original extraction
    if dewarped_catalog.empty:
        logger.warning(f"  Re-extraction failed  falling back to original "
                       f"extraction ({len(df_lwa)} sources)")
        dewarped_catalog = df_lwa

    return dewarped_catalog, True, anchor_data


# =====================================================================
# BAND BOUNDARY DETECTION (new image sizes post-cutover)
# =====================================================================

def crosses_band_boundary(subband, ref_subband):
    """Check if a cascade step crosses a band boundary.

    After the Phase 1 image size cutover (2024-12-27), subbands in
    different bands have different pixel scales and image sizes.
    Flow alignment requires same-size images, so these steps must
    fall back to source-based (RBF) dewarping.

    Returns True if the two subbands are in different bands.
    """
    band_tgt = cfg.SUBBAND_TO_BAND.get(subband)
    band_ref = cfg.SUBBAND_TO_BAND.get(ref_subband)
    if band_tgt is None or band_ref is None:
        return False  # unknown subband, don't block
    return band_tgt != band_ref


def is_post_cutover(work_dir):
    """Check if data in this work_dir is after the image size cutover.

    Reads the date from the work_dir path (e.g., .../10h/2024-12-28/...).
    Returns True if the date is on or after IMAGE_SIZE_CUTOVER_DATE.
    """
    import re
    m = re.search(r'(\d{4}-\d{2}-\d{2})', work_dir)
    if m:
        return m.group(1) >= cfg.IMAGE_SIZE_CUTOVER_DATE
    # Can't determine — assume post-cutover to be safe
    return True


def _regrid_reference_for_flow(work_dir, ref_subband, target_subband,
                                diag_dir, logger):
    """Regrid a dewarped reference image to the target subband's pixel grid.

    Used at band boundaries where image sizes differ. The regridded
    reference is saved as a temporary file for the flow subprocess.

    Returns path to the regridded FITS file, or None on failure.
    """
    from reproject import reproject_interp

    # Find the dewarped reference (prefer PBcorr for Robust-0.75)
    ref_path = find_deep_pb_image_dewarped(work_dir, ref_subband, logger)
    if not ref_path:
        ref_path = find_deep_raw_image_dewarped(work_dir, ref_subband, logger)
    if not ref_path:
        logger.error(f"  No dewarped reference found for {ref_subband}")
        return None

    # Find any target image for its WCS and shape
    tgt_path = find_deep_raw_image(work_dir, target_subband, logger)
    if not tgt_path:
        tgt_path = find_deep_pb_image(work_dir, target_subband, logger)
    if not tgt_path:
        logger.error(f"  No target image found for {target_subband}")
        return None

    try:
        with fits.open(ref_path) as hdul:
            ref_data = hdul[0].data.squeeze()
            ref_hdr = hdul[0].header
            ref_wcs = WCS(ref_hdr).celestial

        with fits.open(tgt_path) as hdul:
            tgt_data = hdul[0].data.squeeze()
            tgt_hdr = hdul[0].header
            tgt_wcs = WCS(tgt_hdr).celestial
            tgt_shape = tgt_data.shape

        logger.info(f"  Regridding {ref_subband} ({ref_data.shape[0]}px) -> "
                    f"{target_subband} ({tgt_shape[0]}px)")

        reprojected, footprint = reproject_interp(
            (ref_data, ref_wcs), tgt_wcs, shape_out=tgt_shape)
        reprojected[footprint < 0.5] = 0.0

        # Save regridded reference
        maps_dir = os.path.join(diag_dir, "maps")
        os.makedirs(maps_dir, exist_ok=True)
        out_path = os.path.join(maps_dir,
                                f"{ref_subband}_regridded_to_{target_subband}.fits")
        out_hdr = tgt_hdr.copy()
        out_hdr['HISTORY'] = (f'Regridded from {ref_subband} '
                              f'({ref_data.shape[0]}px) to {target_subband} '
                              f'({tgt_shape[0]}px)')
        fits.writeto(out_path, reprojected.astype(np.float32),
                     out_hdr, overwrite=True)

        logger.info(f"  Saved regridded reference: {os.path.basename(out_path)}")
        return out_path

    except Exception as e:
        logger.error(f"  Regridding failed: {e}")
        return None


# =====================================================================
# FLOW DISPATCH HELPERS (subprocess calls to py310_orca)
# =====================================================================

def _run_flow_anchor(work_dir, diag_dir, logger):
    """Run optical flow anchor step via subprocess in py310_orca.

    Calls flow_cascade_dewarp.py --anchor_only, which handles the
    synthetic sky generation and Jy/beam correction via calcflow.
    (Previous monkey patches for image_plane_correction bugs removed
    as of May 2026 upstream fixes.)

    Returns True on success.
    """
    import subprocess

    script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'flow_cascade_dewarp.py')
    cmd = [
        "conda", "run", "-n", cfg.FLOW_CONDA_ENV, "--no-capture-output",
        "python", script,
        "--work_dir", work_dir,
        "--anchor_only",
    ]

    logger.info(f"  [Flow] Running anchor via subprocess in {cfg.FLOW_CONDA_ENV}")
    logger.info(f"  [Flow] {' '.join(cmd[-6:])}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=1200)
        if result.stdout:
            for line in result.stdout.strip().split('\n'):
                logger.info(f"  [flow] {line}")
        if result.returncode != 0:
            logger.error(f"  [Flow] Anchor failed (rc={result.returncode})")
            if result.stderr:
                for line in result.stderr.strip().split('\n')[-10:]:
                    logger.error(f"  [flow-err] {line}")
            return False
        return True
    except subprocess.TimeoutExpired:
        logger.error("  [Flow] Anchor subprocess timed out (1200s)")
        return False
    except Exception as e:
        logger.error(f"  [Flow] Anchor subprocess error: {e}")
        return False


def _run_flow_cascade_step(work_dir, subband, ref_subband, diag_dir, logger,
                           ref_image=None):
    """Run optical flow cascade step via flow_cascade_dewarp.py subprocess.

    Calls flow_cascade_dewarp.py --cascade_step which handles file discovery,
    flow computation, application to all I/V variants, header fixup, and
    writes _dewarped.fits alongside originals.

    Parameters
    ----------
    ref_image : str, optional
        Override reference image path (e.g., regridded for band boundary).

    Returns True on success.
    """
    import subprocess

    script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'flow_cascade_dewarp.py')
    cmd = [
        "conda", "run", "-n", cfg.FLOW_CONDA_ENV, "--no-capture-output",
        "python", script,
        "--work_dir", work_dir,
        "--cascade_step",
        "--subband", subband,
        "--ref_subband", ref_subband,
    ]
    if ref_image:
        cmd.extend(["--ref_image", ref_image])

    logger.info(f"  [Flow] {subband} (ref: {ref_subband})")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=600)
        if result.stdout:
            for line in result.stdout.strip().split('\n'):
                logger.info(f"  [flow] {line}")
        if result.returncode != 0:
            logger.error(f"  [Flow] Cascade step {subband} failed "
                         f"(rc={result.returncode})")
            if result.stderr:
                for line in result.stderr.strip().split('\n')[-5:]:
                    logger.error(f"  [flow-err] {line}")
            return False
        return True
    except subprocess.TimeoutExpired:
        logger.error(f"  [Flow] {subband} subprocess timed out (600s)")
        return False
    except Exception as e:
        logger.error(f"  [Flow] {subband} subprocess error: {e}")
        return False


def _run_flow_10min_step(work_dir, subband, ref_image_path, tgt_image_path,
                          timestamp, diag_dir, logger):
    """Run optical flow for a single 10-min interval via flow_cascade_dewarp.py.

    Returns True on success.
    """
    import subprocess

    script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'flow_cascade_dewarp.py')
    cmd = [
        "conda", "run", "-n", cfg.FLOW_CONDA_ENV, "--no-capture-output",
        "python", script,
        "--work_dir", work_dir,
        "--tenmin_step",
        "--subband", subband,
        "--target", tgt_image_path,
        "--reference", ref_image_path,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=600)
        return result.returncode == 0
    except Exception:
        return False


def find_deep_raw_image_dewarped(work_dir, subband, logger):
    """Find the dewarped deep tapered non-PBcorr Stokes I image.

    Used as flow reference for the next cascade step.
    """
    pat = os.path.join(work_dir, subband, "I", "deep",
                       "*Taper*image*_dewarped.fits")
    candidates = [f for f in glob.glob(pat)
                  if "NoTaper" not in os.path.basename(f)
                  and "pbcorr" not in os.path.basename(f)
                  and "_dewarped_dewarped" not in os.path.basename(f)
                  and ".pybdsf" not in os.path.basename(f)]
    if not candidates:
        return None
    robust075 = [f for f in candidates if "Robust-0.75" in os.path.basename(f)]
    if robust075:
        return sorted(robust075)[0]
    return sorted(candidates)[0]


# =====================================================================
# MAIN ENTRY POINT: CASCADE DEWARPING
# =====================================================================

def run_cascade_dewarping(work_dir, logger, anchor_method=None,
                          cascade_method=None, rbf_kernel=None,
                          rbf_smoothing=None, rbf_epsilon=None,
                          snr_weighting=False, anchor_only=False,
                          warp_categories=None):
    global ANCHOR_EL_CUT, ANCHOR_EAST_EL_CUT, ANCHOR_MAXPIX_BEAM_AREAS
    # Per-LST elevation floor (e.g. 13h galactic plane); LST inferred
    # from the work_dir path .../{lst}/{date}/Science_*
    try:
        parts = os.path.normpath(work_dir).split(os.sep)
        lst = next(p for p in parts if len(p) in (3, 4) and p.endswith('h')
                   and p[:-1].isdigit())
        ANCHOR_EL_CUT = getattr(cfg, 'ANCHOR_MIN_ELEVATION_DEG',
                                {}).get(lst)
        # east-side floor for plane-rising LSTs (Gregg 2026-08-28)
        ANCHOR_EAST_EL_CUT = getattr(cfg, 'ANCHOR_EAST_MIN_ELEVATION_DEG',
                                     {}).get(lst)
        # island cap: per-LST value, '*' = all LSTs (Gregg 2026-08-28)
        _cap_cfg = getattr(cfg, 'ANCHOR_MAX_ISLAND_BEAM_AREAS', {})
        ANCHOR_MAXPIX_BEAM_AREAS = _cap_cfg.get(lst, _cap_cfg.get('*'))
        if ANCHOR_MAXPIX_BEAM_AREAS:
            logger.info(f"Anchor extraction island cap for {lst}: "
                        f"{ANCHOR_MAXPIX_BEAM_AREAS:.0f} beam areas")
        if ANCHOR_EL_CUT:
            logger.info(f"Anchor extraction elevation floor for {lst}: "
                        f"{ANCHOR_EL_CUT:.0f} deg"
                        + (f" + east<{ANCHOR_EAST_EL_CUT:.0f} deg"
                           if ANCHOR_EAST_EL_CUT else ""))
    except StopIteration:
        ANCHOR_EL_CUT = None
        ANCHOR_EAST_EL_CUT = None
        ANCHOR_MAXPIX_BEAM_AREAS = None
    """Run the full cascade dewarping pipeline.

    Parameters
    ----------
    work_dir : str
        Phase 2 working directory containing all subband directories.
    logger : logging.Logger
    anchor_method : str
        Method for 73 MHz anchor: 'rbf', 'zernike', or 'flow'.
    cascade_method : str
        Method for cascade and 10-min: 'rbf', 'zernike', or 'flow'.
    rbf_kernel : str
        Kernel for RBFInterpolator (rbf method only).
    rbf_smoothing : float
        Smoothing for RBFInterpolator (rbf method only).
    anchor_only : bool
        If True, run only the 73 MHz anchor step and return.
    warp_categories : list of str, optional
        Image categories to dewarp, e.g. ['deep'] or ['deep', 'snapshots'].
        Default: ['deep'].

    Returns
    -------
    dict : {subband: bool} indicating success/failure per subband.
    """
    # Apply defaults from config
    if anchor_method is None:
        anchor_method = cfg.ANCHOR_METHOD
    if cascade_method is None:
        cascade_method = cfg.CASCADE_METHOD
    if rbf_kernel is None:
        rbf_kernel = cfg.WARP_RBF_KERNEL
    if rbf_smoothing is None:
        rbf_smoothing = cfg.WARP_RBF_SMOOTHING
    if rbf_epsilon is None and cfg.WARP_RBF_EPSILON is not None:
        rbf_epsilon = cfg.WARP_RBF_EPSILON
    logger.info("=" * 60)
    logger.info("CASCADE DEWARPING")
    logger.info(f"  anchor_method={anchor_method}, cascade_method={cascade_method}")
    if anchor_method == 'rbf' or cascade_method == 'rbf':
        logger.info(f"  rbf_kernel={rbf_kernel}, rbf_smoothing={rbf_smoothing}")
    logger.info("=" * 60)

    diag_dir = os.path.join(work_dir, "Dewarp_Diagnostics")
    os.makedirs(diag_dir, exist_ok=True)

    results = {}

    # -- Phase 1a: Anchor selection with fallback --
    # Try candidate anchors nearest the VLSSr reference (74 MHz) until one
    # succeeds; recovers nights where 73 MHz is missing or yields too few
    # anchors. The cascade arms are then re-rooted at the chosen anchor.
    candidates = [sb for sb in ANCHOR_FALLBACK
                  if os.path.isdir(os.path.join(work_dir, sb))]
    # EXPLICIT ANCHOR OVERRIDE (Gregg 2026-09-01, "64 MHz anchor is fine").
    # The fallback walk only triggers when a candidate is MISSING or yields
    # too few VLSSr anchors — not when it yields plenty of anchors and a
    # badly wrong solution. At 14h 2025-05-03 the primary 73MHz anchor and
    # the 78MHz and 69MHz fallbacks all produce warp solutions that measure
    # 24.7, 32.1 and 18.8 arcmin median VLSSr separation against a 3.0
    # threshold, while 64MHz measures 0.44 — so the anchor must be named
    # explicitly. Default behaviour is unchanged when the variable is unset.
    _forced = os.environ.get('P2_FORCE_ANCHOR_SUBBAND')
    if _forced:
        if os.path.isdir(os.path.join(work_dir, _forced)):
            logger.info(f"ANCHOR FORCED to {_forced} via "
                        f"P2_FORCE_ANCHOR_SUBBAND (overriding "
                        f"{ANCHOR_FALLBACK})")
            candidates = [_forced]
        else:
            logger.error(f"P2_FORCE_ANCHOR_SUBBAND={_forced} but "
                         f"{work_dir}/{_forced} does not exist — REFUSING "
                         f"to silently fall back to the default anchor")
            return results
    if not candidates:
        logger.error(f"No anchor candidate subbands {ANCHOR_FALLBACK} found in "
                     f"{work_dir}  aborting dewarping")
        return results

    vlssr = None
    if anchor_method != 'flow':
        vlssr = load_vlssr_catalog(cfg.VLSSR_CATALOG, "VLSSr")
        if vlssr is None:
            logger.error("Failed to load VLSSr catalog  aborting dewarping")
            return results

    anchor_sb = None
    anchor_catalog = None
    anchor_ok = False
    for cand in candidates:
        is_fallback = (cand != cfg.ANCHOR_SUBBAND)
        logger.info(f"\n{'='*40}")
        logger.info(f"ANCHOR: {cand} vs VLSSr ({anchor_method})"
                    + (f"  [FALLBACK from {cfg.ANCHOR_SUBBAND}]" if is_fallback else ""))
        logger.info(f"{'='*40}")
        if anchor_method == 'flow':
            if is_fallback:
                logger.warning("Flow anchor fallback not supported  "
                               "skipping non-primary candidate")
                continue
            cat, ok = None, _run_flow_anchor(work_dir, diag_dir, logger)
        else:
            cat, ok, _ = dewarp_subband(
                work_dir, cand, vlssr, 'VLSSr',
                is_anchor=True, diag_dir=diag_dir, logger=logger,
                method=anchor_method, rbf_kernel=rbf_kernel,
                rbf_smoothing=rbf_smoothing, rbf_epsilon=rbf_epsilon,
                snr_weighting=snr_weighting,
                warp_categories=warp_categories,
            )
        results[cand] = ok
        if ok:
            anchor_sb, anchor_catalog, anchor_ok = cand, cat, True
            if is_fallback:
                logger.warning(f"Using FALLBACK anchor {cand} "
                               f"(primary {cfg.ANCHOR_SUBBAND} unavailable/failed)")
            break
        logger.warning(f"Anchor {cand} failed  trying next fallback band")

    if not anchor_ok:
        logger.error(f"All anchor candidates {candidates} failed  aborting cascade")
        return results

    if anchor_only:
        logger.info(f"\n{'='*40}")
        logger.info(f"ANCHOR ONLY ({anchor_sb})  skipping cascade and 10-min dewarping")
        logger.info(f"{'='*40}")
        return results

    # If cascade uses source-based method but anchor was flow, extract catalog
    if cascade_method != 'flow' and anchor_catalog is None:
        dewarped_pb = find_deep_pb_image_dewarped(work_dir, anchor_sb, logger)
        if dewarped_pb:
            anchor_catalog = extract_sources(dewarped_pb, logger)
        if anchor_catalog is None or anchor_catalog.empty:
            logger.error("Cannot extract catalog from flow-dewarped anchor "
                         " aborting source-based cascade")
            return results

    # Re-root the cascade arms at the chosen anchor (outward in both directions)
    a_idx = cfg.ALL_SUBBANDS.index(anchor_sb)
    cascade_down = cfg.ALL_SUBBANDS[a_idx-1::-1] if a_idx > 0 else []
    cascade_up = cfg.ALL_SUBBANDS[a_idx+1:]
    if anchor_sb != cfg.ANCHOR_SUBBAND:
        logger.info(f"Cascade arms re-rooted at {anchor_sb}: "
                    f"DOWN={cascade_down}  UP={cascade_up}")

    # -- Phase 1b: Cascade Down --
    logger.info(f"\n{'='*40}")
    logger.info(f"CASCADE DOWN ({cascade_method})")
    logger.info(f"{'='*40}")

    if cascade_method == 'flow':
        _run_cascade_arm_flow(work_dir, cascade_down,
                              anchor_sb, diag_dir, results, logger,
                              warp_categories=warp_categories)
    else:
        _run_cascade_arm(work_dir, cascade_down, anchor_catalog,
                         anchor_sb, diag_dir, results, logger,
                         method=cascade_method, rbf_kernel=rbf_kernel,
                         rbf_smoothing=rbf_smoothing, rbf_epsilon=rbf_epsilon,
                         snr_weighting=snr_weighting,
                         warp_categories=warp_categories)

    # -- Phase 1c: Cascade Up --
    logger.info(f"\n{'='*40}")
    logger.info(f"CASCADE UP ({cascade_method})")
    logger.info(f"{'='*40}")

    if cascade_method == 'flow':
        _run_cascade_arm_flow(work_dir, cascade_up,
                              anchor_sb, diag_dir, results, logger,
                              warp_categories=warp_categories)
    else:
        _run_cascade_arm(work_dir, cascade_up, anchor_catalog,
                         anchor_sb, diag_dir, results, logger,
                         method=cascade_method, rbf_kernel=rbf_kernel,
                         rbf_smoothing=rbf_smoothing, rbf_epsilon=rbf_epsilon,
                         snr_weighting=snr_weighting,
                         warp_categories=warp_categories)

    # -- Summary --
    n_ok = sum(1 for v in results.values() if v)
    n_fail = sum(1 for v in results.values() if not v)
    n_missing = len(cfg.ALL_SUBBANDS) - len(results)
    logger.info(f"\n{'='*40}")
    logger.info(f"CASCADE SUMMARY: {n_ok} OK, {n_fail} failed, "
                f"{n_missing} not attempted")
    logger.info(f"{'='*40}")

    # -- Phase 2: Per-interval 10-minute dewarping --
    successful_subbands = [sb for sb, ok in results.items() if ok]
    if successful_subbands:
        if cascade_method == 'flow':
            _dewarp_10min_flow(work_dir, successful_subbands, diag_dir, logger)
        else:
            dewarp_10min_intervals(work_dir, successful_subbands, diag_dir,
                                   logger, method=cascade_method,
                                   rbf_kernel=rbf_kernel,
                                   rbf_smoothing=rbf_smoothing,
                                   rbf_epsilon=rbf_epsilon,
                                   snr_weighting=snr_weighting)

    # -- Phase 3: Snapshot dewarping (post-cascade) --
    cats = warp_categories or ['deep']
    if 'snapshots' in cats and successful_subbands:
        _dewarp_remaining_snapshots(work_dir, successful_subbands,
                                     diag_dir, logger)

    return results


def _run_cascade_arm(work_dir, subband_list, start_catalog, start_subband,
                     diag_dir, results, logger, method='rbf',
                     rbf_kernel='thin_plate_spline', rbf_smoothing=1.5,
                     rbf_epsilon=None, snr_weighting=False,
                     warp_categories=None):
    """Run one arm of the cascade (down or up) using source-based methods."""
    ref_catalog = start_catalog
    ref_subband = start_subband
    consecutive_failures = 0

    for subband in subband_list:
        if not os.path.isdir(os.path.join(work_dir, subband)):
            logger.warning(f"  {subband}: directory not found  skipping")
            results[subband] = False
            consecutive_failures += 1
            if consecutive_failures >= cfg.MAX_CONSECUTIVE_FAILURES:
                logger.error(f"  {cfg.MAX_CONSECUTIVE_FAILURES} consecutive failures "
                             f" halting cascade arm")
                break
            continue

        ref_label = f"{ref_subband}_dewarped"

        catalog, ok, _ = dewarp_subband(
            work_dir, subband, ref_catalog, ref_label,
            is_anchor=False, diag_dir=diag_dir, logger=logger,
            method=method, rbf_kernel=rbf_kernel, rbf_smoothing=rbf_smoothing,
            rbf_epsilon=rbf_epsilon, snr_weighting=snr_weighting,
            warp_categories=warp_categories,
        )
        results[subband] = ok

        if ok:
            ref_catalog = catalog
            ref_subband = subband
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            logger.warning(f"  {subband} failed  next subband will bridge "
                           f"to {ref_subband} (gap = {consecutive_failures})")
            if consecutive_failures >= cfg.MAX_CONSECUTIVE_FAILURES:
                logger.error(f"  {cfg.MAX_CONSECUTIVE_FAILURES} consecutive failures "
                             f" halting cascade arm")
                break


def _run_cascade_arm_flow(work_dir, subband_list, start_subband,
                          diag_dir, results, logger,
                          warp_categories=None):
    """Run one arm of the cascade using optical flow (subprocess).

    At band boundaries (where image sizes change post-cutover), the
    dewarped reference is regridded onto the target's pixel grid and
    the flow step runs against it; the cascade then continues with flow.
    There is no RBF fallback: if the regrid fails, that subband is
    skipped (ratified 2026-10-16, RATIFY-CASCADE-DEWARP-DOCSTRING-...).
    """
    ref_subband = start_subband
    consecutive_failures = 0
    post_cutover = is_post_cutover(work_dir)

    for subband in subband_list:
        if not os.path.isdir(os.path.join(work_dir, subband)):
            logger.warning(f"  {subband}: directory not found  skipping")
            results[subband] = False
            consecutive_failures += 1
            if consecutive_failures >= cfg.MAX_CONSECUTIVE_FAILURES:
                logger.error(f"  {cfg.MAX_CONSECUTIVE_FAILURES} consecutive failures "
                             f" halting cascade arm")
                break
            continue

        # Check for band boundary (different image sizes post-cutover)
        if post_cutover and crosses_band_boundary(subband, ref_subband):
            logger.info(f"  [Band boundary] {ref_subband} -> {subband} "
                        f"({cfg.SUBBAND_TO_BAND.get(ref_subband)} -> "
                        f"{cfg.SUBBAND_TO_BAND.get(subband)}): "
                        f"regridding reference for flow")

            # Regrid dewarped reference to target's pixel grid, then use flow
            regridded_path = _regrid_reference_for_flow(
                work_dir, ref_subband, subband, diag_dir, logger)

            if regridded_path:
                ok = _run_flow_cascade_step(work_dir, subband, ref_subband,
                                             diag_dir, logger,
                                             ref_image=regridded_path)
            else:
                logger.warning(f"  Regridding failed — skipping {subband}")
                ok = False
        else:
            ok = _run_flow_cascade_step(work_dir, subband, ref_subband,
                                         diag_dir, logger)

        results[subband] = ok

        if ok:
            ref_subband = subband
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            logger.warning(f"  {subband} failed  next subband will bridge "
                           f"to {ref_subband} (gap = {consecutive_failures})")
            if consecutive_failures >= cfg.MAX_CONSECUTIVE_FAILURES:
                logger.error(f"  {cfg.MAX_CONSECUTIVE_FAILURES} consecutive failures "
                             f" halting cascade arm")
                break


def _get_dewarped_catalog(work_dir, subband, logger):
    """Extract source catalog from a dewarped subband image.

    Used as the reference catalog when falling back to RBF at band
    boundaries. Prefers PB-corrected image for accurate flux ratios.
    Ensures frequency keywords are present (flow may strip them).
    """
    dewarped_pb = find_deep_pb_image_dewarped(work_dir, subband, logger)
    if dewarped_pb and os.path.exists(dewarped_pb):
        # Ensure frequency keywords exist (flow strips them)
        orig_pb = find_deep_pb_image(work_dir, subband, logger)
        if orig_pb:
            _ensure_freq_header(dewarped_pb, orig_pb, logger)
        return extract_sources(dewarped_pb, logger)

    # Fall back to non-PBcorr dewarped
    dewarped_raw = find_deep_raw_image_dewarped(work_dir, subband, logger)
    if dewarped_raw and os.path.exists(dewarped_raw):
        logger.warning(f"  Using non-PBcorr image for {subband} reference catalog")
        orig_raw = find_deep_raw_image(work_dir, subband, logger)
        if orig_raw:
            _ensure_freq_header(dewarped_raw, orig_raw, logger)
        return extract_sources(dewarped_raw, logger)

    return None


def _make_ionosphere_movie_from_flow(work_dir, diag_dir, logger):
    """Create ionosphere GIF from saved 73 MHz 10-min flow offset files.

    Reads flow_offsets_73MHz_10min_{timestamp}.npz files saved by the
    flow subprocess, renders distortion map frames, computes variability
    metric, and assembles into a GIF.
    """
    try:
        import imageio
    except ImportError:
        logger.warning("  imageio not installed — skipping ionosphere movie")
        return None

    prefix = _diag_prefix_from_workdir(work_dir)

    # Find saved offset files
    npz_files = sorted(glob.glob(
        os.path.join(diag_dir, "flow_offsets_73MHz_10min_*.npz")))

    if len(npz_files) < 3:
        logger.info(f"  Ionosphere movie: only {len(npz_files)} offset files "
                    f"(need ≥3)")
        return None

    logger.info(f"\n--- Ionosphere Evolution Movie (flow, {len(npz_files)} "
                f"frames) ---")

    # Load all offset arrays
    import re
    frame_data = []
    for npz_path in npz_files:
        m = re.search(r'_(\d{8}_\d{6})\.npz$', npz_path)
        timestamp = m.group(1) if m else os.path.basename(npz_path)
        d = np.load(npz_path)
        offset_x = d['offset_x']
        offset_y = d['offset_y']
        magnitude = np.sqrt(offset_x**2 + offset_y**2)
        frame_data.append({
            'timestamp': timestamp,
            'offset_x': offset_x,
            'offset_y': offset_y,
            'magnitude': magnitude,
        })

    # Horizon mask
    h, w = frame_data[0]['magnitude'].shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    horizon = dist > min(h, w) * 0.45
    visible = ~horizon

    # Global color scale
    all_mags = np.concatenate([f['magnitude'][visible].ravel()
                                for f in frame_data])
    vmax = float(np.percentile(all_mags, 99))

    # Variability metric from consecutive frames
    variabilities = [0.0]
    global_median = float(np.median(all_mags))
    for i in range(1, len(frame_data)):
        delta = np.sqrt(
            (frame_data[i]['offset_x'] - frame_data[i-1]['offset_x'])**2
            + (frame_data[i]['offset_y'] - frame_data[i-1]['offset_y'])**2
        )
        if global_median > 0:
            variabilities.append(
                float(np.median(delta[visible]) / global_median))
        else:
            variabilities.append(0.0)

    mean_var = float(np.mean(variabilities[1:])) if len(variabilities) > 1 else 0.0
    max_var = float(np.max(variabilities[1:])) if len(variabilities) > 1 else 0.0

    if mean_var < 0.10:
        iono_quality = "Excellent (stable)"
    elif mean_var < 0.25:
        iono_quality = "Good (slow evolution)"
    elif mean_var < 0.50:
        iono_quality = "Moderate (active)"
    else:
        iono_quality = "Poor (rapid evolution)"

    logger.info(f"  Ionosphere quality: {iono_quality} "
                f"(mean variability = {mean_var:.3f}, "
                f"median shift = {global_median:.2f} pix)")

    # Render frames
    frames = []
    for i, fd in enumerate(frame_data):
        fig, ax = plt.subplots(figsize=(8, 8))
        display = fd['magnitude'].copy()
        display[horizon] = np.nan

        im = ax.imshow(display, origin='lower', cmap='inferno',
                       vmin=0, vmax=vmax)
        plt.colorbar(im, ax=ax, label='Shift (pixels)', shrink=0.8)

        ts = fd['timestamp']
        time_label = (f"{ts[:4]}-{ts[4:6]}-{ts[6:8]} "
                      f"{ts[9:11]}:{ts[11:13]}:{ts[13:15]} UTC")
        med_shift = float(np.nanmedian(display))
        ax.set_title(
            f'Ionosphere: 73MHz  {time_label}\n'
            f'median shift = {med_shift:.2f} pix, '
            f'variability = {variabilities[i]:.3f}',
            fontsize=11)
        ax.set_xlabel('Pixel X')
        ax.set_ylabel('Pixel Y')

        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)
        plt.close(fig)

    # Save GIF (1 second per frame)
    gif_path = os.path.join(
        diag_dir, f"{prefix}ionosphere_evolution_73MHz.gif")
    try:
        imageio.mimsave(gif_path, frames, duration=1000, loop=0)
    except TypeError:
        imageio.mimsave(gif_path, frames, duration=1.0, loop=0)
    logger.info(f"  Saved ionosphere movie: {os.path.basename(gif_path)} "
                f"({len(frames)} frames)")

    # Save variability JSON
    variability_summary = {
        'subband': '73MHz',
        'method': 'flow',
        'n_frames': len(frame_data),
        'mean_variability': mean_var,
        'max_variability': max_var,
        'median_shift_pix': float(global_median),
        'quality': iono_quality,
        'per_frame': [
            {'timestamp': fd['timestamp'],
             'median_shift_pix': float(np.nanmedian(fd['magnitude'][visible])),
             'variability': variabilities[i]}
            for i, fd in enumerate(frame_data)
        ],
    }
    json_path = os.path.join(
        diag_dir, f"{prefix}ionosphere_variability_73MHz.json")
    with open(json_path, 'w') as f:
        json.dump(variability_summary, f, indent=2)
    logger.info(f"  Saved variability summary: {os.path.basename(json_path)}")

    return variability_summary


def _dewarp_10min_flow(work_dir, subband_list, diag_dir, logger):
    """Dewarp 10-min intervals using optical flow (sequential, subprocess).

    Sequential because each call requires the GPU and runs
    flow_cascade_dewarp.py in py310_orca via subprocess.
    """
    import re

    logger.info(f"\n{'='*60}")
    logger.info("10-MINUTE INTERVAL DEWARPING (flow, sequential)")
    logger.info(f"{'='*60}")

    n_total = 0
    n_ok = 0
    n_fail = 0

    for subband in sorted(subband_list,
                          key=lambda x: int(x.replace('MHz', ''))):
        # Find dewarped deep reference
        ref_image = find_deep_raw_image_dewarped(work_dir, subband, logger)
        if ref_image is None:
            logger.warning(f"  {subband}: no dewarped deep reference  skipping")
            continue

        # Find 10-min snapshots
        i_10min_dir = os.path.join(work_dir, subband, "I", "10min")
        if not os.path.isdir(i_10min_dir):
            continue

        snapshots = sorted([
            f for f in glob.glob(os.path.join(i_10min_dir, "*Taper*image*.fits"))
            if "_dewarped" not in os.path.basename(f)
            and ".pybdsf" not in os.path.basename(f)
            and "pbcorr" not in os.path.basename(f)
            and "NoTaper" not in os.path.basename(f)
        ])

        if not snapshots:
            continue

        logger.info(f"\n  {subband}: {len(snapshots)} intervals "
                    f"(ref: {os.path.basename(ref_image)})")

        for tgt_path in snapshots:
            n_total += 1
            m = re.search(r'(\d{8}_\d{6})', os.path.basename(tgt_path))
            timestamp = m.group(1) if m else os.path.basename(tgt_path)

            ok = _run_flow_10min_step(work_dir, subband, ref_image,
                                       tgt_path, timestamp, diag_dir, logger)
            if ok:
                n_ok += 1
                logger.info(f"    {timestamp}: OK")
            else:
                n_fail += 1
                logger.warning(f"    {timestamp}: FAILED")

    logger.info(f"\n  10-min flow summary: {n_ok} OK, {n_fail} failed "
                f"out of {n_total}")

    # Create ionosphere evolution GIF from 73 MHz flow distortion maps
    _make_ionosphere_movie_from_flow(work_dir, diag_dir, logger)


# =====================================================================
# 10-MINUTE INTERVAL DEWARPING (PARALLEL)
# =====================================================================


# =====================================================================
# POST-CASCADE SNAPSHOT DEWARPING
# =====================================================================

def _ensure_freq_header(fits_path, reference_path, logger):
    """Ensure a FITS file has frequency keywords, copying from reference if missing.

    Flow dewarping can strip CRVAL3/CTYPE3 from the header, causing BDSF
    to fail. This copies the essential frequency and beam keywords from the
    reference (original) image if they're missing.
    """
    freq_keys = ['CRVAL3', 'CTYPE3', 'CDELT3', 'CRPIX3', 'CUNIT3',
                 'NAXIS3', 'NAXIS4', 'BMAJ', 'BMIN', 'BPA']
    with fits.open(fits_path, mode='update') as hdul:
        hdr = hdul[0].header
        if 'CRVAL3' in hdr:
            return  # already has frequency info
        # Copy from reference
        with fits.open(reference_path) as ref:
            ref_hdr = ref[0].header
            n_copied = 0
            for key in freq_keys:
                if key in ref_hdr and key not in hdr:
                    hdr[key] = ref_hdr[key]
                    if ref_hdr.comments[key]:
                        hdr.comments[key] = ref_hdr.comments[key]
                    n_copied += 1
            if n_copied > 0:
                hdul.flush()
                logger.info(f"  Copied {n_copied} header keywords to "
                            f"{os.path.basename(fits_path)}")


def _dewarp_remaining_snapshots(work_dir, subband_list, diag_dir, logger):
    """Dewarp snapshots for subbands where the deep was dewarped by flow.

    The RBF anchor step dewarps snapshots directly (via warp_categories).
    Flow cascade steps only dewarp the deep. This function handles the
    remaining subbands by:
      1. Extracting sources from the original and dewarped deep
      2. Cross-matching to find the displacement field
      3. Computing an RBF warp screen from the displacements
      4. Applying that screen to all un-dewarped snapshot files
    """
    logger.info(f"\n{'='*40}")
    logger.info(f"SNAPSHOT DEWARPING (post-cascade)")
    logger.info(f"{'='*40}")

    n_subbands = 0
    n_total = 0

    for subband in subband_list:
        # Find un-dewarped snapshots
        snap_files = []
        for pol in ['I', 'V']:
            snap_dir = os.path.join(work_dir, subband, pol, "snapshots")
            if not os.path.isdir(snap_dir):
                continue
            for f in sorted(glob.glob(os.path.join(snap_dir, "*-image.fits"))):
                bn = os.path.basename(f)
                if '_dewarped' in bn:
                    continue
                dewarped = f.replace('.fits', '_dewarped.fits')
                if not os.path.exists(dewarped):
                    snap_files.append(f)

        if not snap_files:
            continue

        # Find original and dewarped deep images
        orig_pb = find_deep_pb_image(work_dir, subband, logger)
        dewarped_pb = find_deep_pb_image_dewarped(work_dir, subband, logger)

        if not orig_pb or not dewarped_pb:
            # Fall back to non-PBcorr
            orig_raw = find_deep_raw_image(work_dir, subband, logger)
            dewarped_raw = find_deep_raw_image_dewarped(work_dir, subband, logger)
            if not orig_raw or not dewarped_raw:
                logger.warning(f"  {subband}: cannot find original+dewarped deep "
                               f" skipping {len(snap_files)} snapshots")
                continue
            orig_img = orig_raw
            dewarped_img = dewarped_raw
        else:
            orig_img = orig_pb
            dewarped_img = dewarped_pb

        # Extract sources from both
        logger.info(f"  {subband}: extracting sources from original deep...")
        df_orig = extract_sources(orig_img, logger)

        # Ensure dewarped image has frequency keywords (flow may strip them)
        _ensure_freq_header(dewarped_img, orig_img, logger)

        logger.info(f"  {subband}: extracting sources from dewarped deep...")
        df_dewarped = extract_sources(dewarped_img, logger)

        if df_orig is None or df_orig.empty or df_dewarped is None or df_dewarped.empty:
            logger.warning(f"  {subband}: source extraction failed "
                           f" skipping {len(snap_files)} snapshots")
            continue

        # Get beam size for filtering
        with fits.open(orig_img) as hdul:
            hdr = hdul[0].header
            beam_deg = hdr.get('BMAJ', 0.1)

        # Cross-match: original positions (target) vs dewarped positions (ref)
        match_radius = cfg.CASCADE_MATCH_RADIUS_BASE * u.arcmin
        anchors = cross_match_subbands(
            df_orig, df_dewarped, match_radius,
            cfg.CASCADE_FLUX_RATIO_MIN, cfg.CASCADE_FLUX_RATIO_MAX,
            logger, beam_deg=beam_deg)

        if anchors is None:
            logger.warning(f"  {subband}: cross-match failed "
                           f" skipping {len(snap_files)} snapshots")
            continue

        # Compute warp screen (original → dewarped frame)
        with fits.open(orig_img) as hdul:
            data_shape = hdul[0].data.squeeze().shape
            wcs = WCS(hdul[0].header).celestial

        screen_x, screen_y = compute_warp_screen(
            anchors, wcs, data_shape, logger,
            method='rbf', rbf_kernel=cfg.WARP_RBF_KERNEL,
            rbf_smoothing=cfg.WARP_RBF_SMOOTHING,
            rbf_epsilon=cfg.WARP_RBF_EPSILON)

        if screen_x is None:
            logger.warning(f"  {subband}: warp screen computation failed")
            continue

        # Save warp screen for later standalone use
        _save_warp_screen(screen_x, screen_y, wcs, subband, diag_dir,
                          'rbf_from_flow', cfg.WARP_RBF_KERNEL,
                          len(anchors), logger, work_dir=work_dir)

        # Apply to all un-dewarped snapshots
        n_warped = 0
        for fpath in snap_files:
            out_path = fpath.replace('.fits', '_dewarped.fits')
            if os.path.exists(out_path):
                continue
            try:
                with fits.open(fpath) as hdul:
                    data = hdul[0].data.squeeze()
                    snap_hdr = hdul[0].header

                if data.shape != screen_x.shape:
                    logger.warning(f"  {subband}: shape mismatch "
                                   f"{data.shape} vs {screen_x.shape} "
                                   f" skipping {os.path.basename(fpath)}")
                    continue

                warped = apply_warp(data, screen_x, screen_y)
                if warped is not None:
                    fits.writeto(out_path, warped, snap_hdr, overwrite=True)
                    n_warped += 1
            except Exception as e:
                logger.warning(f"  {subband}: warp failed for "
                               f"{os.path.basename(fpath)}: {e}")

        if n_warped > 0:
            logger.info(f"  {subband}: dewarped {n_warped}/{len(snap_files)} snapshots")
            n_subbands += 1
            n_total += n_warped

    logger.info(f"\nSnapshot dewarping complete: {n_total} files across "
                f"{n_subbands} subbands")


def dewarp_10min_intervals(work_dir, subband_list, diag_dir, logger,
                           method='rbf', rbf_kernel='thin_plate_spline',
                           rbf_smoothing=None, rbf_epsilon=None,
                           snr_weighting=False):
    """Dewarp each 10-minute interval independently, in parallel.

    Each 10-minute I snapshot gets its own warp screen because the
    ionosphere changes on ~10 minute timescales. Uses the globally-
    corrected deep reference catalog as the astrometric truth.

    Parallelized across all intervals and subbands simultaneously.

    Parameters
    ----------
    work_dir : str
        Phase 2 working directory.
    subband_list : list of str
        Subbands to process (only those with successful deep dewarping).
    diag_dir : str
        Diagnostics output directory.
    logger : logging.Logger
    method : str
        Interpolation method: 'rbf' or 'zernike'.
    rbf_kernel : str
        Kernel for RBFInterpolator.
    rbf_smoothing : float or None
        Smoothing for RBFInterpolator. Defaults to cfg.WARP_RBF_SMOOTHING_10MIN.
    """
    if rbf_smoothing is None:
        rbf_smoothing = cfg.WARP_RBF_SMOOTHING_10MIN
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing

    logger.info(f"\n{'='*60}")
    logger.info("10-MINUTE INTERVAL DEWARPING (parallel)")
    logger.info(f"{'='*60}")

    # -- 1. Pre-extract reference catalogs (sequential  shared resource) --
    ref_data = {}  # subband -> {catalog_dict, beam_deg, shape, wcs_header}

    for subband in sorted(subband_list,
                          key=lambda x: int(x.replace('MHz', ''))):
        deep_dewarped = find_deep_pb_image_dewarped(work_dir, subband, logger)
        if deep_dewarped is None:
            continue

        with fits.open(deep_dewarped) as hdul:
            header = hdul[0].header
            shape = hdul[0].data.squeeze().shape
            beam_deg = header.get('BMAJ', None)
            # Serialize WCS as header dict (picklable)
            wcs_header = WCS(header).celestial.to_header()

        ref_catalog = extract_sources(deep_dewarped, logger)
        if ref_catalog.empty:
            continue

        try:
            freq_mhz = float(subband.replace('MHz', ''))
        except ValueError:
            continue

        # Convert catalog to dict of arrays (picklable)
        ref_data[subband] = {
            'catalog': {col: ref_catalog[col].values for col in ref_catalog.columns},
            'beam_deg': beam_deg,
            'shape': shape,
            'wcs_header': dict(wcs_header),
            'freq_mhz': freq_mhz,
            'n_sources': len(ref_catalog),
        }
        logger.info(f"  {subband}: {len(ref_catalog)} reference sources")

    # -- 2. Build task list --
    tasks = []

    for subband, rd in ref_data.items():
        i_10min_dir = os.path.join(work_dir, subband, "I", "10min")
        v_10min_dir = os.path.join(work_dir, subband, "V", "10min")
        if not os.path.isdir(i_10min_dir):
            continue

        i_images = sorted([
            f for f in glob.glob(os.path.join(i_10min_dir, "*Taper*pbcorr*.fits"))
            if "_dewarped" not in os.path.basename(f)
            and ".pybdsf" not in os.path.basename(f)
        ])

        for i_img in i_images:
            timestamp = _extract_timestamp(i_img)

            # Collect all image files to warp for this interval (I + V, image + pbcorr)
            files_for_interval = []
            for pol_dir in [i_10min_dir, v_10min_dir]:
                if not os.path.isdir(pol_dir):
                    continue
                for f in glob.glob(os.path.join(pol_dir, f"*{timestamp}*.fits")):
                    bn = os.path.basename(f)
                    if "_dewarped" in bn or ".pybdsf" in bn:
                        continue
                    if '-image' not in bn:
                        continue
                    files_for_interval.append(f)

            tasks.append({
                'subband': subband,
                'i_img': i_img,
                'timestamp': timestamp,
                'files_to_warp': files_for_interval,
                'ref_data': rd,
                'diag_dir': diag_dir,
                'method': method,
                'rbf_kernel': rbf_kernel,
                'rbf_smoothing': rbf_smoothing,
                'rbf_epsilon': rbf_epsilon,
                'snr_weighting': snr_weighting,
            })

    logger.info(f"\n  Total tasks: {len(tasks)} intervals across "
                f"{len(ref_data)} subbands")

    if not tasks:
        logger.warning("  No 10min intervals to dewarp")
        return

    # -- 3. Run in parallel --
    n_workers = min(8, max(1, multiprocessing.cpu_count() - 2))
    logger.info(f"  Using {n_workers} parallel workers")

    results_10min = {}
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_dewarp_one_interval, task): task
            for task in tasks
        }

        for future in as_completed(futures):
            task = futures[future]
            key = f"{task['subband']}_{task['timestamp']}"
            try:
                result = future.result()
                results_10min[key] = result
                if result['success']:
                    logger.info(f"  {task['subband']} {task['timestamp']}: "
                                f"{result['n_warped']} files, "
                                f"{result['n_anchors']} anchors, "
                                f"shift={result['median_shift_arcmin']:.2f}'")
                else:
                    logger.warning(f"  {task['subband']} {task['timestamp']}: "
                                   f"failed  {result.get('error', 'unknown')}")
            except Exception as e:
                logger.error(f"  {task['subband']} {task['timestamp']}: "
                             f"worker exception  {e}")

    n_ok = sum(1 for r in results_10min.values() if r.get('success'))
    n_fail = len(results_10min) - n_ok
    logger.info(f"\n  10-minute dewarping complete: {n_ok} OK, {n_fail} failed")

    # -- 4. Generate ionosphere evolution movie at 73 MHz --
    _make_ionosphere_movie(work_dir, results_10min, diag_dir, logger)


def _dewarp_one_interval(task):
    """Worker function: dewarp a single 10-minute interval.

    Runs in a subprocess via ProcessPoolExecutor. All inputs are
    picklable (no WCS objects, no loggers  uses dicts and arrays).

    Returns dict with success, n_warped, n_anchors, median_shift_arcmin,
    and optionally the warp screen stats for the movie.
    """
    import warnings
    warnings.filterwarnings('ignore')

    subband = task['subband']
    i_img = task['i_img']
    rd = task['ref_data']
    diag_dir = task['diag_dir']

    result = {'success': False, 'n_warped': 0, 'n_anchors': 0,
              'median_shift_arcmin': 0, 'subband': subband,
              'timestamp': task['timestamp']}

    try:
        # Reconstruct WCS from header dict
        from astropy.wcs import WCS as _WCS
        wcs_ref = _WCS(rd['wcs_header'])
        shape = rd['shape']
        beam_deg = rd['beam_deg']
        freq_mhz = rd['freq_mhz']

        # Reconstruct reference catalog as DataFrame
        ref_catalog = pd.DataFrame(rd['catalog'])

        # Extract sources from 10min I image
        # Use a minimal logger for subprocess
        import logging as _logging
        _logger = _logging.getLogger(f'worker_{subband}_{task["timestamp"]}')
        _logger.setLevel(_logging.WARNING)  # quiet in workers

        try:
            # Same elevation floor as extract_sources: the interval
            # images at galactic-plane LSTs wedge BDSF for hours on
            # low-elevation plane emission (the 2026-06-11 stall was in
            # THIS call path, not the deep one). ANCHOR_EL_CUT is set in
            # the parent and inherited by forked workers.
            _bdsf_in, _tmp_cut = i_img, None
            if ANCHOR_EL_CUT is not None:
                try:
                    _tmp_cut = _elevation_masked_copy(i_img, ANCHOR_EL_CUT,
                                                      _logger,
                                                      ANCHOR_EAST_EL_CUT)
                    _bdsf_in = _tmp_cut
                except Exception:
                    pass
            _mp = _anchor_maxpix(_bdsf_in)
            _mp_kw = {'maxpix_isl': _mp} if _mp else {}
            img = bdsf.process_image(
                _bdsf_in, thresh_pix=cfg.BDSF_THRESH_PIX,
                thresh_isl=cfg.BDSF_THRESH_ISL,
                adaptive_rms_box=True, quiet=True, **_mp_kw
            )
            if _tmp_cut and os.path.exists(_tmp_cut):
                try:
                    os.remove(_tmp_cut)
                except OSError:
                    pass
            sources = []
            for s in img.sources:
                if not np.isnan(s.posn_sky_max[0]):
                    sources.append({
                        'ra': s.posn_sky_max[0], 'dec': s.posn_sky_max[1],
                        'flux_peak': s.peak_flux_max,
                        'maj': getattr(s, 'maj_axis', 0.0),
                        'min': getattr(s, 'min_axis', 0.0),
                    })
            if not sources:
                result['error'] = 'no sources extracted'
                return result
            df_10min = pd.DataFrame(sources)
        except Exception as e:
            result['error'] = f'BDSF failed: {e}'
            return result

        # Cross-match
        freq_scale = (cfg.ANCHOR_FREQ_MHZ / freq_mhz) ** cfg.CASCADE_MATCH_RADIUS_EXPONENT
        match_radius = cfg.CASCADE_MATCH_RADIUS_BASE * freq_scale * u.arcmin

        # Inline cross-match (avoid passing logger to subprocess)
        tgt_coords = SkyCoord(df_10min['ra'].values, df_10min['dec'].values, unit='deg')
        ref_coords = SkyCoord(ref_catalog['ra'].values, ref_catalog['dec'].values, unit='deg')
        idx, d2d, _ = tgt_coords.match_to_catalog_sky(ref_coords)

        mask_pos = d2d < match_radius
        tgt_flux = df_10min['flux_peak'].values
        ref_flux = ref_catalog['flux_peak'].values[idx]
        ratio = tgt_flux / ref_flux
        mask_flux = (ratio > cfg.CASCADE_FLUX_RATIO_MIN) & (ratio < cfg.CASCADE_FLUX_RATIO_MAX)
        mask = mask_pos & mask_flux

        if np.sum(mask) < cfg.MIN_WARP_ANCHORS:
            result['error'] = f'only {np.sum(mask)} anchors'
            return result

        ra_lwa = df_10min['ra'].values[mask]
        dec_lwa = df_10min['dec'].values[mask]
        ra_cat = ref_catalog['ra'].values[idx[mask]]
        dec_cat = ref_catalog['dec'].values[idx[mask]]
        anchor_flux = df_10min['flux_peak'].values[mask]

        # Pixel coordinates and offsets
        mx, my = wcs_ref.all_world2pix(ra_lwa, dec_lwa, 0)
        tx, ty = wcs_ref.all_world2pix(ra_cat, dec_cat, 0)
        dx = mx - tx
        dy = my - ty

        # Simple sigma clip on offsets
        mag = np.sqrt(dx**2 + dy**2)
        clipped = sigma_clip(mag, sigma=3.0, maxiters=2, stdfunc=mad_std)
        keep = ~clipped.mask
        mx, my = mx[keep], my[keep]
        tx, ty = tx[keep], ty[keep]
        dx, dy = dx[keep], dy[keep]
        anchor_flux = anchor_flux[keep]

        if len(dx) < cfg.MIN_WARP_ANCHORS:
            result['error'] = f'only {len(dx)} anchors after clip'
            return result

        result['n_anchors'] = len(dx)

        # Warp screen fit
        points = np.column_stack([ty, tx])
        method = task.get('method', 'rbf')
        rbf_kernel = task.get('rbf_kernel', 'thin_plate_spline')
        rbf_smoothing = task.get('rbf_smoothing', cfg.WARP_RBF_SMOOTHING_10MIN)
        rbf_epsilon = task.get('rbf_epsilon', None)
        snr_weighting = task.get('snr_weighting', False)

        if method == 'zernike':
            raise RuntimeError('zernike engine retired to attic (§12.1)')
            model = fit_zernike_warp(
                tx, ty, dx, dy,
                image_shape=shape,
                max_order=cfg.WARP_ZERNIKE_ORDER_10MIN,
            )
            grid_y, grid_x = np.mgrid[0:shape[0], 0:shape[1]]
            screen_x, screen_y = evaluate_zernike_warp(
                model, grid_x, grid_y)
        else:
            rbf_kwargs = dict(kernel=rbf_kernel)

            # Smoothing: scalar or per-point array
            if snr_weighting and len(anchor_flux) > 0:
                flux = np.clip(anchor_flux, 1e-6, None)
                flux_ref = np.median(flux)
                smoothing_array = rbf_smoothing * (flux_ref / flux)**2
                smoothing_array = np.clip(smoothing_array,
                                          rbf_smoothing * 0.01,
                                          rbf_smoothing * 100)
                rbf_kwargs['smoothing'] = smoothing_array
            else:
                rbf_kwargs['smoothing'] = rbf_smoothing

            needs_epsilon = rbf_kernel in ('multiquadric',
                                           'inverse_multiquadric', 'gaussian')
            if needs_epsilon:
                if rbf_epsilon is None:
                    from scipy.spatial import cKDTree
                    tree = cKDTree(points)
                    dists, _ = tree.query(points, k=2)
                    rbf_epsilon = float(np.mean(dists[:, 1]))
                rbf_kwargs['epsilon'] = rbf_epsilon

            rbf_x = RBFInterpolator(points, dx, **rbf_kwargs)
            rbf_y = RBFInterpolator(points, dy, **rbf_kwargs)

            # Evaluate on coarse grid
            ny, nx_img = shape
            coarse_factor = max(1, min(ny, nx_img) // 512)
            cy_c = np.arange(0, ny, coarse_factor)
            cx_c = np.arange(0, nx_img, coarse_factor)
            cg_y, cg_x = np.meshgrid(cy_c, cx_c, indexing='ij')
            eval_pts = np.column_stack([cg_y.ravel(), cg_x.ravel()])

            sx_coarse = rbf_x(eval_pts).reshape(cg_y.shape)
            sy_coarse = rbf_y(eval_pts).reshape(cg_y.shape)

            from scipy.ndimage import zoom as _zoom
            zoom_y = ny / sx_coarse.shape[0]
            zoom_x = nx_img / sx_coarse.shape[1]
            screen_x = _zoom(sx_coarse, (zoom_y, zoom_x), order=1)[:ny, :nx_img]
            screen_y = _zoom(sy_coarse, (zoom_y, zoom_x), order=1)[:ny, :nx_img]

        # Compute shift stats
        try:
            scales = wcs_ref.proj_plane_pixel_scales()
            scale_arcmin = np.mean([s.value if hasattr(s, 'value') else s
                                    for s in scales]) * 60.0
        except Exception:
            scale_arcmin = 0.03125 * 60.0

        shift_mag = np.sqrt(screen_x**2 + screen_y**2) * scale_arcmin
        # Mask outside visible sky for stats
        cy_img, cx_img_c = ny / 2, nx_img / 2
        yy, xx = np.mgrid[0:ny, 0:nx_img]
        dist = np.sqrt((xx - cx_img_c)**2 + (yy - cy_img)**2)
        visible = dist < min(ny, nx_img) * 0.45
        result['median_shift_arcmin'] = float(np.median(shift_mag[visible]))

        # Store screen + anchor data for movie (only for 73 MHz to avoid
        # memory bloat in IPC). Screens for variability metric, anchors
        # for quiver-plot frames.
        if subband == '73MHz':
            result['screen_x'] = screen_x
            result['screen_y'] = screen_y
            result['scale_arcmin'] = scale_arcmin
            result['anchor_tx'] = tx.copy()
            result['anchor_ty'] = ty.copy()
            result['anchor_dx'] = dx.copy()
            result['anchor_dy'] = dy.copy()

        # Apply warp to all files for this interval
        n_warped = 0
        for fpath in task['files_to_warp']:
            out_path = fpath.replace('.fits', '_dewarped.fits')
            try:
                with fits.open(fpath) as hf:
                    data = hf[0].data.squeeze()
                    if data.shape == screen_x.shape:
                        warped = apply_warp(data, screen_x, screen_y)
                        if warped is not None:
                            fits.writeto(out_path, warped, hf[0].header,
                                         overwrite=True)
                            n_warped += 1
            except Exception:
                pass

        result['n_warped'] = n_warped
        result['success'] = True

    except Exception as e:
        result['error'] = str(e)

    return result


def find_deep_pb_image_dewarped(work_dir, subband, logger):
    """Find the dewarped deep tapered PB-corrected Stokes I image."""
    pat = os.path.join(work_dir, subband, "I", "deep",
                       "*Taper*pbcorr*_dewarped.fits")
    candidates = [f for f in glob.glob(pat)
                  if "NoTaper" not in os.path.basename(f)
                  and "_dewarped_dewarped" not in os.path.basename(f)
                  and ".pybdsf" not in os.path.basename(f)]
    if not candidates:
        return None
    robust075 = [f for f in candidates if "Robust-0.75" in os.path.basename(f)]
    if robust075:
        return sorted(robust075)[0]
    return sorted(candidates)[0]


def _extract_timestamp(filepath):
    """Extract YYYYMMDD_HHMMSS timestamp from a FITS filename."""
    import re
    m = re.search(r'(\d{8}_\d{6})', os.path.basename(filepath))
    return m.group(1) if m else os.path.basename(filepath)


def _make_ionosphere_movie(work_dir, results_10min, diag_dir, logger):
    """Generate an animated GIF of the ionosphere at 73 MHz across 10min intervals.

    Uses anchor vector (quiver) plots to show the per-source displacement
    field, with frame-to-frame variability metric computed from warp screens.

    Variability metric: for each consecutive pair of frames, compute
        delta = sqrt((sx_i - sx_{i-1})^2 + (sy_i - sy_{i-1})^2) * scale
        variability = median(delta[visible]) / median(shift_all[visible])
    where 0 = frozen ionosphere, ~1 = rapidly reorganizing.

    Returns dict with variability summary for email report, or None.
    """
    try:
        import imageio
    except ImportError:
        logger.warning("  imageio not installed — skipping ionosphere movie. "
                       "Install with: pip install imageio")
        return None

    logger.info("\n--- Ionosphere Evolution Movie ---")

    prefix = _diag_prefix_from_workdir(work_dir)

    # Collect 73 MHz results (or closest available)
    movie_subband = '73MHz'
    movie_results = sorted(
        [(k, v) for k, v in results_10min.items()
         if v.get('success') and v.get('subband') == movie_subband
         and 'screen_x' in v and 'anchor_tx' in v],
        key=lambda x: x[1]['timestamp']
    )

    if len(movie_results) < 3:
        # Fall back to the subband with most successful intervals
        from collections import Counter
        sb_counts = Counter(
            v['subband'] for v in results_10min.values()
            if v.get('success') and 'screen_x' in v and 'anchor_tx' in v
        )
        if sb_counts:
            movie_subband = sb_counts.most_common(1)[0][0]
            movie_results = sorted(
                [(k, v) for k, v in results_10min.items()
                 if v.get('success') and v.get('subband') == movie_subband
                 and 'screen_x' in v and 'anchor_tx' in v],
                key=lambda x: x[1]['timestamp']
            )

    if len(movie_results) < 3:
        logger.warning("  Not enough intervals for ionosphere movie")
        return None

    logger.info(f"  Making movie from {len(movie_results)} frames at {movie_subband}")

    # ---- Compute global color scale from anchor shift magnitudes ----
    scale_arcmin = movie_results[0][1]['scale_arcmin']
    all_anchor_shifts = []
    for _, r in movie_results:
        amag = np.sqrt(r['anchor_dx']**2 + r['anchor_dy']**2) * scale_arcmin
        all_anchor_shifts.append(amag)
    vmax = np.percentile(np.concatenate(all_anchor_shifts), 99)

    # Image shape for axis limits (from first screen)
    ny, nx = movie_results[0][1]['screen_x'].shape

    # ---- Compute variability metric from warp screens ----
    cy, cx = ny / 2, nx / 2
    yy, xx = np.mgrid[0:ny, 0:nx]
    dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    visible = dist < min(ny, nx) * 0.45

    all_shifts_screen = []
    for _, r in movie_results:
        smag = np.sqrt(r['screen_x']**2 + r['screen_y']**2) * scale_arcmin
        all_shifts_screen.append(smag)

    # Global median shift for normalization
    global_median = np.median(
        np.concatenate([s[visible].ravel() for s in all_shifts_screen]))

    variabilities = [0.0]  # first frame has no predecessor
    for i in range(1, len(movie_results)):
        delta = np.sqrt(
            (movie_results[i][1]['screen_x'] - movie_results[i-1][1]['screen_x'])**2
            + (movie_results[i][1]['screen_y'] - movie_results[i-1][1]['screen_y'])**2
        ) * scale_arcmin
        if global_median > 0:
            variabilities.append(
                float(np.median(delta[visible]) / global_median))
        else:
            variabilities.append(0.0)

    # Summary statistics
    mean_var = float(np.mean(variabilities[1:])) if len(variabilities) > 1 else 0.0
    max_var = float(np.max(variabilities[1:])) if len(variabilities) > 1 else 0.0
    median_shift_all = float(global_median)

    variability_summary = {
        'subband': movie_subband,
        'n_frames': len(movie_results),
        'mean_variability': mean_var,
        'max_variability': max_var,
        'median_shift_arcmin': median_shift_all,
        'per_frame': [],
    }

    # Classify ionospheric quality
    if mean_var < 0.10:
        iono_quality = "Excellent (stable)"
    elif mean_var < 0.25:
        iono_quality = "Good (slow evolution)"
    elif mean_var < 0.50:
        iono_quality = "Moderate (active)"
    else:
        iono_quality = "Poor (rapid evolution)"

    variability_summary['quality'] = iono_quality

    logger.info(f"  Ionosphere quality: {iono_quality} "
                f"(mean variability = {mean_var:.3f}, "
                f"median shift = {median_shift_all:.2f}')")

    # ---- Render quiver-plot frames ----
    frames = []
    for i, ((key, r), var_val) in enumerate(zip(movie_results, variabilities)):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_facecolor('black')

        atx = r['anchor_tx']
        aty = r['anchor_ty']
        adx = r['anchor_dx']
        ady = r['anchor_dy']
        amag = np.sqrt(adx**2 + ady**2) * scale_arcmin

        q = ax.quiver(atx, aty, adx, ady, amag,
                      cmap='plasma', scale=None, alpha=0.8,
                      clim=(0, vmax))
        plt.colorbar(q, ax=ax, label='Shift (arcmin)', shrink=0.8)

        ax.set_xlim(0, nx)
        ax.set_ylim(0, ny)
        ax.set_aspect('equal')

        # Format timestamp
        ts = r['timestamp']
        time_label = (f"{ts[:4]}-{ts[4:6]}-{ts[6:8]} "
                      f"{ts[9:11]}:{ts[11:13]}:{ts[13:15]} UTC")

        title_lines = (
            f'Ionosphere: {movie_subband}  {time_label}\n'
            f'{r["n_anchors"]} anchors, '
            f'median shift = {r["median_shift_arcmin"]:.2f}\', '
            f'variability = {var_val:.3f}'
        )
        ax.set_title(title_lines, fontsize=11)
        ax.set_xlabel('Pixel X')
        ax.set_ylabel('Pixel Y')

        # Render to image array
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)
        plt.close(fig)

        # Store per-frame data for email/report
        variability_summary['per_frame'].append({
            'timestamp': ts,
            'n_anchors': r['n_anchors'],
            'median_shift_arcmin': r['median_shift_arcmin'],
            'variability': var_val,
        })

    # ---- Save GIF (1 second per frame) ----
    gif_path = os.path.join(
        diag_dir,
        f"{prefix}ionosphere_evolution_{movie_subband}.gif")
    # imageio v3: duration in ms per frame; v2: duration in seconds.
    # Use loop=0 for infinite loop. Try ms first (v3 convention).
    try:
        imageio.mimsave(gif_path, frames, duration=1000, loop=0)
    except TypeError:
        # Older imageio: duration in seconds
        imageio.mimsave(gif_path, frames, duration=1.0, loop=0)
    logger.info(f"  Saved ionosphere movie: {gif_path} ({len(frames)} frames)")

    # ---- Save individual PNGs ----
    for i, (key, r) in enumerate(movie_results):
        ts = r['timestamp']
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_facecolor('black')
        amag = np.sqrt(r['anchor_dx']**2 + r['anchor_dy']**2) * scale_arcmin
        q = ax.quiver(r['anchor_tx'], r['anchor_ty'],
                      r['anchor_dx'], r['anchor_dy'], amag,
                      cmap='plasma', scale=None, alpha=0.8,
                      clim=(0, vmax))
        plt.colorbar(q, ax=ax, label='Shift (arcmin)', shrink=0.8)
        ax.set_xlim(0, nx)
        ax.set_ylim(0, ny)
        ax.set_aspect('equal')
        time_label = (f"{ts[:4]}-{ts[4:6]}-{ts[6:8]} "
                      f"{ts[9:11]}:{ts[11:13]}:{ts[13:15]}")
        ax.set_title(f'Ionosphere: {movie_subband}  {time_label}\n'
                     f'{r["n_anchors"]} anchors, '
                     f'variability = {variabilities[i]:.3f}')
        png_path = os.path.join(
            diag_dir,
            f"{prefix}iono_10min_{movie_subband}_{i:02d}_{ts}.png")
        plt.savefig(png_path, dpi=100, bbox_inches='tight')
        plt.close(fig)

    # ---- Save variability summary JSON ----
    json_path = os.path.join(
        diag_dir,
        f"{prefix}ionosphere_variability_{movie_subband}.json")
    with open(json_path, 'w') as f:
        json.dump(variability_summary, f, indent=2)
    logger.info(f"  Saved variability summary: {os.path.basename(json_path)}")

    return variability_summary
