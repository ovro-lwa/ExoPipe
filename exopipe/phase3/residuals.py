"""
exopipe.phase3.residuals — seed module (refactor step 2, extracted 2026-08-11).

Origins (VERBATIM extractions):
  /lustre/gh/main/clv15/phase3_subtract_lab.py — gaussian_stamp, match_to_beam,
      inner_rms (plus internal deps beam_area_px and FWHM2SIG that they call).
  /lustre/gh/main/clv15/phase3_i_lab2.py — fft_upsample2, fft_decimate2,
      write_up_fits.
The remaining residual/subtraction-capability content arrives at move time.
"""
import numpy as np
from astropy.io import fits
from astropy.stats import mad_std
from astropy.convolution import Gaussian2DKernel, convolve_fft

FWHM2SIG = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def beam_area_px(bmaj_deg, bmin_deg, cdelt_deg):
    """Gaussian beam solid angle in pixels."""
    return (np.pi / (4.0 * np.log(2.0))) * bmaj_deg * bmin_deg / cdelt_deg ** 2



def match_to_beam(data, beam, target_fwhm_deg, cdelt_deg):
    """Convolve a Jy/beam image to a circular target beam.

    beam = (bmaj_deg, bmin_deg, bpa_deg). The convolution kernel is the
    analytic Gaussian deconvolution of the current beam from the target
    (valid because the circular target exceeds every input BMAJ).
    Output is in Jy/(target beam). NaNs are preserved.
    """
    bmaj, bmin, bpa = beam
    if target_fwhm_deg < bmaj * (1.0 + 1e-6):
        raise ValueError(f"target beam {target_fwhm_deg*3600:.0f}\" smaller "
                         f"than image beam {bmaj*3600:.0f}\"")
    sig_t = target_fwhm_deg * FWHM2SIG / cdelt_deg
    sig_maj = bmaj * FWHM2SIG / cdelt_deg
    sig_min = bmin * FWHM2SIG / cdelt_deg
    # Deconvolution kernel: along the beam's own principal axes
    k_maj = np.sqrt(max(sig_t ** 2 - sig_maj ** 2, 0.25))
    k_min = np.sqrt(max(sig_t ** 2 - sig_min ** 2, 0.25))
    # Sky PA (deg E of N) -> pixel-frame angle for axes (x=-RA, y=Dec)
    theta = np.deg2rad(90.0 + bpa)
    kern = Gaussian2DKernel(x_stddev=k_maj, y_stddev=k_min, theta=theta)
    kern.normalize()

    nanmask = ~np.isfinite(data)
    area_from = beam_area_px(bmaj, bmin, cdelt_deg)
    area_to = beam_area_px(target_fwhm_deg, target_fwhm_deg, cdelt_deg)
    jy_px = np.where(nanmask, 0.0, data) / area_from
    sm = convolve_fft(jy_px, kern, normalize_kernel=True,
                      nan_treatment='fill', fill_value=0.0,
                      allow_huge=True)
    out = sm * area_to
    out[nanmask] = np.nan
    return out



def gaussian_stamp(shape, x0, y0, peak, beam, cdelt_deg):
    """Elliptical Gaussian point source with the image beam."""
    bmaj, bmin, bpa = beam
    sig_maj = bmaj * FWHM2SIG / cdelt_deg
    sig_min = bmin * FWHM2SIG / cdelt_deg
    theta = np.deg2rad(90.0 + bpa)
    r = int(np.ceil(4 * sig_maj))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    ct, st = np.cos(theta), np.sin(theta)
    xr = xx * ct + yy * st
    yr = -xx * st + yy * ct
    g = peak * np.exp(-0.5 * ((xr / sig_maj) ** 2 + (yr / sig_min) ** 2))
    stamp = np.zeros(shape)
    x0, y0 = int(round(x0)), int(round(y0))
    stamp[y0 - r:y0 + r + 1, x0 - r:x0 + r + 1] += g
    return stamp




def inner_rms(data, half=512):
    n = data.shape[0]
    c = n // 2
    return mad_std(data[c - half:c + half, c - half:c + half],
                   ignore_nan=True)



def fft_upsample2(a):
    """Band-limited ×2 upsample via Fourier zero-padding (peak-preserving)."""
    a = np.nan_to_num(a)
    n = a.shape[0]
    F = np.fft.fftshift(np.fft.fft2(a))
    P = np.zeros((2 * n, 2 * n), complex)
    P[n // 2:n // 2 + n, n // 2:n // 2 + n] = F
    return np.real(np.fft.ifft2(np.fft.ifftshift(P))) * 4.0



def fft_decimate2(a):
    """Inverse of fft_upsample2: crop the Fourier plane back to n."""
    n2 = a.shape[0]
    n = n2 // 2
    F = np.fft.fftshift(np.fft.fft2(a))
    C = F[n // 2:n // 2 + n, n // 2:n // 2 + n]
    return np.real(np.fft.ifft2(np.fft.ifftshift(C))) / 4.0



def write_up_fits(data, hdr0, path):
    """Upsampled image with consistent (halved-CDELT) WCS for flow."""
    h = fits.Header()
    for k in ['CTYPE1', 'CTYPE2', 'CRVAL1', 'CRVAL2']:
        if k in hdr0:
            h[k] = hdr0[k]
    h['CDELT1'] = hdr0['CDELT1'] / 2.0
    h['CDELT2'] = hdr0['CDELT2'] / 2.0
    h['CRPIX1'] = 2 * hdr0['CRPIX1'] - 1.0
    h['CRPIX2'] = 2 * hdr0['CRPIX2'] - 1.0
    for k in ['BMAJ', 'BMIN', 'BPA', 'BUNIT']:
        if k in hdr0:
            h[k] = hdr0[k]
    fits.writeto(path, data.astype(np.float32), h, overwrite=True)


