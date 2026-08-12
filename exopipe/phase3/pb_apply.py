#!/usr/bin/env python
"""
pb_apply.py -- derive deep/wideband/Full PBs from the 90-map base library and apply
them to Phase-3 deep co-add masters (Stokes I).

Base library (build_pb_library.py): PB_{subband}_bin{k}.fits, 15 subbands x 6 bins,
all on a COMMON canonical sky frame (same T_REF CRVAL) so cross-band reprojection is
well defined.

Derivations:
  deep subband PB   = nanmean over the 6 bins for that subband
  band PB (R/G/B)   = inverse-variance-weighted sum of the band's 5 subband PBs
                      (same grid -> direct)
  Full PB           = inverse-variance-weighted sum over all 15 subband PBs,
                      Red/Green reprojected onto the Full (=Blue) grid
Weights = Sigma over the master's constituent nights of the per-subband `weight`
(=1/rms^2) column in that night's QA thermal_noise CSV (category=deep). Exact.

Floor: correct only where PB > 0.1 (below -> NaN).  See MOSAIC_DESIGN.md.
"""
import os
import re
import glob
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp

from exopipe.common import phase2_config as cfg

PB_DIR = "/fast/gh/main/phase3/pb_library"
QA_ROOT = "/lustre/pipeline/exopipe/phase2"
PB_FLOOR = 0.1

_FREQ_TO_LABEL = {float(int(l[:-3])): l for l in cfg.ALL_SUBBANDS}  # 18.0 -> '18MHz'


# ----------------------------------------------------------------------
# base-library access
# ----------------------------------------------------------------------
def _bin_path(subband, k, pb_dir):
    return os.path.join(pb_dir, f"PB_{subband}_bin{k}.fits")


_DEEP_CACHE = {}          # subband -> (deep_pb, header)
_FULL_REPROJ_CACHE = {}   # subband -> deep_pb reprojected onto the Full grid


def _deep_path(subband, pb_dir):
    return os.path.join(pb_dir, f"PB_{subband}_deep.fits")


def subband_deep_pb(subband, pb_dir=PB_DIR):
    """Deep (60-min) subband PB (data, header). Cached.

    Reads the stored 60-min map `PB_{subband}_deep.fits` (built directly over the full
    hour by build_pb_library.build_deep). Falls back to the equal-weight mean of the six
    10-min bins if the deep map is absent (numerically identical, for backward compat)."""
    if subband in _DEEP_CACHE:
        return _DEEP_CACHE[subband]
    deep_path = _deep_path(subband, pb_dir)
    if os.path.exists(deep_path):
        with fits.open(deep_path) as h:
            deep = h[0].data.astype(np.float32)
            hdr = h[0].header.copy()
    else:                                    # fallback: mean of the 6 bins
        stack = []
        hdr = None
        for k in range(6):
            with fits.open(_bin_path(subband, k, pb_dir)) as h:
                stack.append(h[0].data.astype(np.float32))
                if hdr is None:
                    hdr = h[0].header.copy()
        with np.errstate(invalid='ignore'):
            deep = np.nanmean(np.stack(stack, axis=0), axis=0).astype(np.float32)
    _DEEP_CACHE[subband] = (deep, hdr)
    return deep, hdr


def subband_bin_pb(subband, k, pb_dir=PB_DIR):
    with fits.open(_bin_path(subband, k, pb_dir)) as h:
        return h[0].data.astype(np.float32), h[0].header.copy()


# ----------------------------------------------------------------------
# weights from constituent-night QA CSVs
# ----------------------------------------------------------------------
def constituent_dates(master_path):
    """Parse 'COADD constituent YYYY-MM-DD' HISTORY lines from a master."""
    hdr = fits.getheader(master_path)
    hist = "\n".join(str(c) for c in hdr.get('HISTORY', []))
    return sorted(set(re.findall(r'constituent\s+(20\d\d-\d\d-\d\d)', hist)))


def _find_qa_csv(lst, date):
    pat = os.path.join(QA_ROOT, lst, date, 'Science_*', 'QA',
                       f'{date}_{lst}_thermal_noise.csv')
    hits = sorted(glob.glob(pat))
    return hits[-1] if hits else None


def subband_weights(lst, dates, category='deep'):
    """Sum per-subband inverse-variance weight over constituent nights.

    Returns (weights: dict label->float, used_dates, missing_dates).
    """
    import csv
    weights = {l: 0.0 for l in cfg.ALL_SUBBANDS}
    used, missing = [], []
    for date in dates:
        p = _find_qa_csv(lst, date)
        if not p:
            missing.append(date)
            continue
        used.append(date)
        with open(p) as f:
            for row in csv.DictReader(f):
                if row.get('category') != category:
                    continue
                label = _FREQ_TO_LABEL.get(float(row['freq']))
                w = float(row['weight'])
                if label and np.isfinite(w):
                    weights[label] += w
    return weights, used, missing


# ----------------------------------------------------------------------
# wideband / Full PB assembly
# ----------------------------------------------------------------------
def _canonical_full_wcs(pb_dir=PB_DIR):
    """Target WCS for the Full/Blue grid (3122^2, canonical CRVAL)."""
    _, hdr = subband_bin_pb('73MHz', 0, pb_dir)   # a Blue subband = Full geometry
    return WCS(hdr), hdr


def wideband_pb(scope, lst, master_path, pb_dir=PB_DIR, category='deep', binidx=None):
    """Effective PB for scope in {'Red','Green','Blue','Full', or a subband label}.

    binidx=None -> deep (mean of bins); binidx=k -> that 10-min sub-LST bin.
    Returns (pb_map, meta) where meta records weights/dates used.
    """
    dates = constituent_dates(master_path)
    weights, used, missing = subband_weights(lst, dates, category=category)

    if scope in cfg.ALL_SUBBANDS:
        members = [scope]
    elif scope == 'Full':
        members = list(cfg.ALL_SUBBANDS)
    else:
        members = list(cfg.SUBBAND_BANDS[scope])

    def _pb_of(sb):
        return (subband_deep_pb(sb, pb_dir) if binidx is None
                else subband_bin_pb(sb, binidx, pb_dir))

    if scope == 'Full':
        # combine on the Full (=Blue) grid; reproject Red/Green up.
        tgt_wcs, tgt_hdr = _canonical_full_wcs(pb_dir)
        shape = (tgt_hdr['NAXIS2'], tgt_hdr['NAXIS1'])
        acc = np.zeros(shape, np.float64)
        wsum = 0.0
        for sb in members:
            w = weights.get(sb, 0.0)
            if w <= 0:
                continue
            if binidx is None and sb in _FULL_REPROJ_CACHE:
                grid = _FULL_REPROJ_CACHE[sb]
            else:
                data, hdr = _pb_of(sb)
                if (hdr['NAXIS1'], hdr['NAXIS2']) == (tgt_hdr['NAXIS1'], tgt_hdr['NAXIS2']):
                    grid = data
                else:
                    grid, _ = reproject_interp((data, WCS(hdr)), tgt_wcs,
                                               shape_out=shape, order='bilinear')
                if binidx is None:
                    _FULL_REPROJ_CACHE[sb] = grid
            acc += w * np.nan_to_num(grid, nan=0.0)
            wsum += w
        pb = (acc / wsum).astype(np.float32) if wsum > 0 else acc.astype(np.float32)
    else:
        # single grid (subband or one band) -> direct weighted sum
        acc, wsum, ref = None, 0.0, None
        for sb in members:
            w = weights.get(sb, 0.0)
            if w <= 0:
                continue
            data, hdr = _pb_of(sb)
            if acc is None:
                acc = np.zeros_like(data, np.float64)
                ref = hdr
            acc += w * np.nan_to_num(data, nan=0.0)
            wsum += w
        pb = (acc / wsum).astype(np.float32) if (acc is not None and wsum > 0) else None

    meta = dict(scope=scope, lst=lst, cadence=('deep' if binidx is None else f'bin{binidx}'),
                n_dates=len(used), n_missing=len(missing), members=members,
                weights={k: weights[k] for k in members},
                used_dates=used, missing_dates=missing)
    return pb, meta


# ----------------------------------------------------------------------
# apply
# ----------------------------------------------------------------------
def apply_pbcorr(master_path, pb_map, out_path, floor=PB_FLOOR, meta=None):
    """Divide master by pb_map where pb>floor (else NaN); write _pbcorr.fits."""
    with fits.open(master_path) as h:
        data = h[0].data.astype(np.float32)
        hdr = h[0].header.copy()
    img = data.squeeze()
    if img.shape != pb_map.shape:
        raise ValueError(f"shape mismatch {img.shape} vs PB {pb_map.shape}")
    valid = np.isfinite(pb_map) & (pb_map > floor)
    out = np.full_like(img, np.nan)
    out[valid] = img[valid] / pb_map[valid]
    hdr['PBCORR'] = (True, 'OVRO-LWA PB library correction applied')
    hdr['PBFLOOR'] = (floor, 'PB floor below which pixels set NaN')
    hdr['PBLIBV'] = ('v1-2026-07-04', 'PB library version')
    if meta:
        hdr['PBNDATE'] = (meta.get('n_dates', -1), 'Constituent nights with QA weights')
        if meta.get('n_missing'):
            hdr['PBMISS'] = (meta['n_missing'], 'Constituent nights missing QA CSV')
    fits.writeto(out_path, out.astype(np.float32), hdr, overwrite=True)
    return out_path
