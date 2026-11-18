"""Helpers shared by the hybrid route's stages. Port of notaper_largescale.py (NL) at exopipe 4c7098c: log 85, rd 123,
diag_header 135, inputs_dir 144, hybrid_filter 1077, lowpass 1083, _sky_interior 1227; VARIANT / TAPER_OF 66-68 (NL md5 127: v2
uses lib/install/steps_common.md5).

Arrays are read with ordinary reads (memmap=False): never a memory map of a Lustre file, for reading or writing (Gregg 2026-11-17).
"""
import os
import time

import numpy as np

# weighting -> (co-add variant key, Phase 1 image variant, robust): R075 / R0 are the NoTaper stacks, T075 / T0 the Taper residual
# stacks of the same unit (NL 66-68)
VARIANT = {'R075': ('NoTaper_R075', 'NoTaper-Robust-0.75', -0.75), 'R0': ('NoTaper_R0', 'NoTaper-Robust-0', 0.0),
           'T075': ('Taper_R075', 'Taper-Robust-0.75', -0.75), 'T0': ('Taper_R0', 'Taper-Robust-0', 0.0)}
TAPER_OF = {'R075': 'T075', 'R0': 'T0'}

_T0 = time.time()


def log(*a):
    print(f'[{time.time() - _T0:7.0f} s]', *a, flush=True)


def fft_workers():
    from exopipe.phase3.config import CFG
    return int(CFG.hybrid.FFT_WORKERS)


def rd(path):
    """The primary image as float64, squeezed to 2-D (ordinary reads)."""
    from astropy.io import fits
    return np.squeeze(fits.getdata(path, memmap=False)).astype(np.float64)


def diag_header(mh, what):
    """Header for an intermediate or diagnostic file: the master's WCS and beam without its product identity."""
    h = mh.copy()
    for k in ('PRODNAME', 'MSTRGEN'):
        h.remove(k, ignore_missing=True, remove_all=True)
    h['LSDIAG'] = (what[:68], 'lsrecon intermediate/diagnostic - not a product')
    return h


def inputs_dir(workdir):
    return f'{workdir}/inputs'


def write_atomic(path, data, header):
    """fits.writeto through a temporary name in the same directory and an atomic rename (production wrote in place)."""
    from astropy.io import fits
    tmp = f'{path}.part'
    fits.writeto(tmp, data, header, overwrite=True, output_verify='exception')
    os.replace(tmp, path)


def hybrid_filter(N, cdelt_deg):
    """G on the rfft2 grid of an N x N image: 1 at baseline lengths below hybrid.HYB_U_IN wavelengths, a cosine to 0 at
    hybrid.HYB_U_OUT, 0 beyond."""
    from exopipe.phase3.config import CFG
    u_in, u_out = float(CFG.hybrid.HYB_U_IN), float(CFG.hybrid.HYB_U_OUT)
    u = np.hypot(np.fft.fftfreq(N)[:, None], np.fft.rfftfreq(N)[None, :]) / np.radians(cdelt_deg)
    return np.where(u <= u_in, 1.0, np.where(u >= u_out, 0.0, 0.5 * (1 + np.cos(np.pi * (u - u_in) / (u_out - u_in)))))


def lowpass(a, cdelt_deg):
    """G (x) a, NaN taken as 0; linear (zero-padded to twice the size), so opposite horizons never mix through the FFT's
    wrap-around."""
    from scipy import fft as sfft
    w = fft_workers()
    N = a.shape[0]
    P = np.zeros((2 * N, 2 * N))
    P[:N, :N] = np.nan_to_num(a)
    return sfft.irfft2(sfft.rfft2(P, workers=w) * hybrid_filter(2 * N, cdelt_deg), s=P.shape, workers=w)[:N, :N]


def sky_interior(ok, hdr):
    """ok inside 0.95 of the horizon radius (direction cosine) from the reference pixel."""
    N = ok.shape[0]
    yy, xx = np.indices((N, N))
    return ok & (np.hypot(np.radians((xx + 1 - hdr['CRPIX1']) * hdr['CDELT1']),
                          np.radians((yy + 1 - hdr['CRPIX2']) * hdr['CDELT2'])) < 0.95)
