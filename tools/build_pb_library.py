#!/usr/bin/env python
"""
build_pb_library.py -- OVRO-LWA primary-beam library for Phase-3 all-sky mosaics.

Builds the LST-INDEPENDENT PB library: 15 subbands x 7 time windows = 105 base maps
(6 x 10-minute bins `PB_{sb}_bin{k}.fits` + 1 x 60-minute deep `PB_{sb}_deep.fits`),
each the beam response TIME-INTEGRATED over its window (below-horizon -> 0). Because
every Phase-2/3 image is a north-up RA---SIN/DEC--SIN projection phase-centered on the
zenith at the LST-hour midpoint, a pixel's (hour-angle, Dec) -- hence its beam response
and its intra-hour time integral -- is identical across every LST hour. So the library
is built once and reused for all cells.

  build()      -> the 90 x 10-min bin maps
  build_deep() -> the 15 x 60-min deep maps (integrated directly over the full hour;
                  equals the equal-weight mean of the 6 bins by construction). The deep
                  map is what a full-hour (deep) synthesis image sees, so it is the one
                  the deep co-adds / mosaic are corrected with.

Derived products (see pb_apply.py, NOT stored here):
  wideband bin PB   = inverse-variance-weighted combo of the band's 5 subband bins
  wideband deep PB  = inverse-variance-weighted combo of the 5 subband deep PBs
  Full PB           = inverse-variance-weighted combo across all 15 subbands

See MOSAIC_DESIGN.md for the decisions behind this.
"""
import os
import sys
import argparse
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.time import Time
from astropy.coordinates import SkyCoord, AltAz
import astropy.units as u

import phase2_config as cfg
from extractor_pb_75 import BeamModel, BEAM_PATH
from pipeline_config import OVRO_LOC

# Field-center declination of the co-add masters (zenith pointing; from header).
CRVAL2_DEG = 37.215849166667

# EXACT co-add master geometry per band (verified uniform across all LST cells,
# 2026-07-04). The PB library must match the master pixel grid pixel-for-pixel, so
# these override the nominal phase2_config.NEW_IMAGE_PARAMS (which are odd-sized).
# Full shares Blue's grid (3122^2/0.037/crpix1562).
MASTER_GEOM = {
    'Red':   {'npix': 1508, 'cdelt_deg': 0.0767, 'crpix': 755.0},
    'Green': {'npix': 2358, 'cdelt_deg': 0.049,  'crpix': 1180.0},
    'Blue':  {'npix': 3122, 'cdelt_deg': 0.037,  'crpix': 1562.0},
}
# Reference epoch: any fixed UTC. Only precession (months-level) depends on it and
# that is negligible for the beam. CRVAL1 is set to this epoch's LST so mid-hour = transit.
T_REF = "2025-01-01T00:00:00"

N_BINS = 6                 # 6 sub-LST bins = the six 10-minute intervals of the hour
DEFAULT_NSUB_PER_BIN = 3   # 3 x 6 = 18 sub-times total (multiple of 6, per Gregg)

OUT_DIR = "/fast/gh/main/phase3/pb_library"


def _sub_time_offsets(nsub_per_bin):
    """Hours-from-midpoint for every sub-time, grouped by bin.

    Returns list of length N_BINS; each element is an array of nsub_per_bin offsets.
    Hour spans tau in [-0.5, +0.5]; bin k covers [-0.5+k/6, -0.5+(k+1)/6].
    """
    bins = []
    for k in range(N_BINS):
        lo = -0.5 + k / N_BINS
        offs = np.array([lo + (j + 0.5) / nsub_per_bin / N_BINS
                         for j in range(nsub_per_bin)])
        bins.append(offs)
    return bins


def _band_grid(band):
    """Build the canonical WCS + RA/Dec grid for a band geometry.

    CRVAL1 is set to the reference-epoch LST so the field-center pixel transits the
    zenith at tau=0. The returned map is LST-independent in pixel space.
    """
    p = MASTER_GEOM[band]
    npix, cdelt, crpix = p['npix'], p['cdelt_deg'], p['crpix']
    t_ref = Time(T_REF, location=OVRO_LOC)
    crval1 = t_ref.sidereal_time('apparent').deg   # LST of reference epoch, in deg

    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---SIN', 'DEC--SIN']
    w.wcs.crval = [crval1, CRVAL2_DEG]
    w.wcs.cdelt = [-cdelt, cdelt]
    w.wcs.crpix = [crpix, crpix]

    yy, xx = np.indices((npix, npix))
    ra, dec = w.all_pix2world(xx, yy, 0)
    return w, ra, dec, crval1, npix


def _altaz(ra, dec, offset_hours):
    """Alt/Az (theta=90-el, phi=az, el) of the grid at T_REF + offset_hours.

    Solar vs sidereal drift over +-0.5 h is ~0.27% => <0.02 deg HA error; ignored.
    Computed once per (band, sub-time) and reused across the band's 5 subbands.
    """
    t = Time(T_REF, location=OVRO_LOC) + offset_hours * u.hour
    sc = SkyCoord(ra.ravel(), dec.ravel(), unit='deg')
    aa = sc.transform_to(AltAz(obstime=t, location=OVRO_LOC))
    el = aa.alt.deg
    theta = 90.0 - el
    phi = aa.az.deg % 360.0
    return theta, phi, el


def build(nsub_per_bin=DEFAULT_NSUB_PER_BIN, bands=('Red', 'Green', 'Blue'),
          out_dir=OUT_DIR, dry_run=False):
    os.makedirs(out_dir, exist_ok=True)
    beam = BeamModel(BEAM_PATH)
    beam.load_data()
    interp = beam.interpolator

    bin_offsets = _sub_time_offsets(nsub_per_bin)
    written = []

    for band in bands:
        subbands = cfg.SUBBAND_BANDS[band]
        w, ra, dec, crval1, npix = _band_grid(band)
        print(f"[{band}] npix={npix} crval1={crval1:.4f} subbands={subbands}",
              flush=True)

        # Pre-compute Alt/Az once per (bin, sub-time); reuse across the 5 subbands.
        # altaz_cache[k] = list over sub-times of (theta, phi, el_mask_nan)
        altaz_cache = []
        for k in range(N_BINS):
            per_time = []
            for off in bin_offsets[k]:
                theta, phi, el = _altaz(ra, dec, off)
                # ~(el>=10) flags below-horizon AND off-projection NaN pixels.
                per_time.append((theta, phi, ~(el >= 10.0)))
            altaz_cache.append(per_time)

        for sb in subbands:
            freq = cfg.SUBBAND_FREQ_HZ[sb]['center']
            for k in range(N_BINS):
                acc = np.zeros(npix * npix, dtype=np.float64)
                for (theta, phi, el_bad) in altaz_cache[k]:
                    pts = np.column_stack((np.full(theta.size, freq), theta, phi))
                    resp = np.asarray(interp(pts), dtype=np.float64)
                    resp[el_bad] = 0.0          # below horizon contributes 0 to the mean
                    acc += resp
                pbmap = (acc / len(altaz_cache[k])).reshape((npix, npix)).astype(np.float32)

                out = os.path.join(out_dir, f"PB_{sb}_bin{k}.fits")
                if dry_run:
                    finite = np.isfinite(pbmap)
                    print(f"    {sb} bin{k}: max={pbmap[finite].max():.4f} "
                          f"peak_at_center={pbmap[npix//2, npix//2]:.4f}", flush=True)
                    continue
                hdr = w.to_header()
                hdr['PBLIB'] = (True, 'OVRO-LWA PB library base map')
                hdr['PBSCOPE'] = (sb, 'Subband scope')
                hdr['PBFREQ'] = (freq, '[Hz] subband center frequency')
                hdr['PBLSTBIN'] = (k, 'Sub-LST bin index (0-5, 10-min each)')
                hdr['PBNSUB'] = (nsub_per_bin, 'Sub-times averaged within this bin')
                hdr['PBBEAM'] = (os.path.basename(BEAM_PATH), 'Beam model')
                hdr['PBTREF'] = (T_REF, 'Reference epoch (LST-independent lib)')
                hdr['PBELCUT'] = (10.0, '[deg] elevation floor (below -> 0)')
                fits.writeto(out, pbmap, hdr, overwrite=True)
                written.append(out)
                print(f"    wrote {os.path.basename(out)} "
                      f"center={pbmap[npix//2, npix//2]:.3f}", flush=True)
    return written


def build_deep(nsub_per_bin=DEFAULT_NSUB_PER_BIN, bands=('Red', 'Green', 'Blue'),
               out_dir=OUT_DIR, dry_run=False):
    """Build the 60-minute (deep) integrated PB for each subband -> PB_{sb}_deep.fits.

    This is the beam response integrated DIRECTLY over the full hour (all 6*nsub
    sub-times, below-horizon -> 0, consistent with the 10-min bins), i.e. the effective
    beam of a full-hour synthesis (deep) image. It equals the equal-weight mean of the
    six 10-min bins by construction; storing it explicitly makes the library a proper
    15 x 7 = 105 set and makes "the mosaic uses the 60-min beam" explicit in the files.
    """
    os.makedirs(out_dir, exist_ok=True)
    beam = BeamModel(BEAM_PATH)
    beam.load_data()
    interp = beam.interpolator

    all_offsets = np.concatenate(_sub_time_offsets(nsub_per_bin))  # 6*nsub over [-0.5,+0.5] h
    n_sub = len(all_offsets)
    written = []

    for band in bands:
        subbands = cfg.SUBBAND_BANDS[band]
        w, ra, dec, crval1, npix = _band_grid(band)
        print(f"[{band} DEEP] npix={npix} n_subtimes={n_sub} subbands={subbands}",
              flush=True)
        # Alt/Az once per sub-time; reused across the band's 5 subbands.
        altaz = []
        for off in all_offsets:
            theta, phi, el = _altaz(ra, dec, off)
            altaz.append((theta, phi, ~(el >= 10.0)))

        for sb in subbands:
            freq = cfg.SUBBAND_FREQ_HZ[sb]['center']
            acc = np.zeros(npix * npix, dtype=np.float64)
            for (theta, phi, el_bad) in altaz:
                pts = np.column_stack((np.full(theta.size, freq), theta, phi))
                resp = np.asarray(interp(pts), dtype=np.float64)
                resp[el_bad] = 0.0          # below horizon contributes 0 to the integral
                acc += resp
            pbmap = (acc / n_sub).reshape((npix, npix)).astype(np.float32)

            out = os.path.join(out_dir, f"PB_{sb}_deep.fits")
            if dry_run:
                print(f"    {sb} deep: center={pbmap[npix//2, npix//2]:.4f}", flush=True)
                continue
            hdr = w.to_header()
            hdr['PBLIB'] = (True, 'OVRO-LWA PB library base map')
            hdr['PBSCOPE'] = (sb, 'Subband scope')
            hdr['PBFREQ'] = (freq, '[Hz] subband center frequency')
            hdr['PBLSTBIN'] = (-1, 'Deep/60-min integration (not a 10-min bin)')
            hdr['PBWINDOW'] = ('deep-60min', 'Time-integration window')
            hdr['PBNSUB'] = (n_sub, 'Sub-times integrated over the full hour')
            hdr['PBBEAM'] = (os.path.basename(BEAM_PATH), 'Beam model')
            hdr['PBTREF'] = (T_REF, 'Reference epoch (LST-independent lib)')
            hdr['PBELCUT'] = (10.0, '[deg] elevation floor (below -> 0)')
            fits.writeto(out, pbmap, hdr, overwrite=True)
            written.append(out)
            print(f"    wrote {os.path.basename(out)} "
                  f"center={pbmap[npix//2, npix//2]:.3f}", flush=True)
    return written


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--nsub', type=int, default=DEFAULT_NSUB_PER_BIN,
                    help='sub-times per 10-min bin (total = 6*nsub)')
    ap.add_argument('--bands', nargs='+', default=['Red', 'Green', 'Blue'])
    ap.add_argument('--out', default=OUT_DIR)
    ap.add_argument('--deep', action='store_true',
                    help='build ONLY the 60-min deep maps (PB_{sb}_deep.fits); '
                         'the 90 10-min bins are unchanged')
    ap.add_argument('--dry-run', action='store_true',
                    help='print stats, do not write FITS')
    args = ap.parse_args()
    if args.deep:
        build_deep(nsub_per_bin=args.nsub, bands=tuple(args.bands),
                   out_dir=args.out, dry_run=args.dry_run)
    else:
        build(nsub_per_bin=args.nsub, bands=tuple(args.bands),
              out_dir=args.out, dry_run=args.dry_run)
