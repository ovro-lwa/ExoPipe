"""Stage `recon` of the hybrid route (spec plan/specs/hybrid.md 2.3): the NoTaper dirty stack re-deconvolved in the image domain
(multiscale CLEAN to the true horizon) plus a large-scale Tikhonov step toward the LWA1 survey prior; the split beam correction;
the content tests that refuse. A close port of production exopipe/phase3/notaper_largescale.py (NL) at exopipe main 4c7098c:
FFT helpers NL 90-122, prior 498-648, idclean 650-724, ls_step 727-778, test_compact_sources / compact_ratio 780-825, the level
test and caveat cards 834-868, recon_variant 870-915, write_pbcorr_split 917-941, installed_master 1089-1107, test_pbcorr_split
1881-1891, write_master 1893-1940. The function bodies are production's, line for line, so the products are bit-identical; the
differences are listed here:
  - the constants come from the config (hybrid_recon, hybrid, hybrid_unit) through bind(), called by every entry point before
    any computation (production: module constants of the same names and values);
  - a refusal raises ReconRefusal (a RuntimeError, as production raised) - the stage turns it into the unit's refusal;
  - every FITS file is read with memmap=False (production set astropy's default at its entry point, NL 2226);
  - the unit record is written through a temporary name and a rename (production: json.dump in place);
  - md5, rd, diag_header, inputs_dir, log come from lib/hybrid/common and lib/install/steps_common; CFP / PBA are the v2 copies
    lib/masters/coadd and lib/masters/pb_apply; build_name / parse_name / write_standard_header are core/products.
"""
import glob
import json
import os
import time

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy import fft as sfft
from scipy.ndimage import binary_dilation, zoom

from exopipe.phase3.core.products import build_name, parse_name, write_standard_header
from exopipe.phase3.lib.hybrid.common import VARIANT, diag_header, inputs_dir, log, rd
from exopipe.phase3.lib.install.steps_common import md5
from exopipe.phase3.lib.masters import coadd as CFP
from exopipe.phase3.lib.masters import pb_apply as PBA


class ReconRefusal(RuntimeError):
    """A gate or content test of the reconstruction refused (production raised RuntimeError '... - REFUSING')."""


K_B, C_LIGHT = 1.380649e-23, 2.99792458e8           # NL 79: Boltzmann constant (J/K), speed of light (m/s)
_BOUND = {}


def bind():
    """Bind the module's constants from the config (call before any computation; cheap, idempotent). Names and values as
    production's module constants (NL 62-80, 828-831, 949)."""
    from exopipe.phase3.config import CFG
    R, H = CFG.hybrid_recon, CFG.hybrid
    g = dict(SCALES=tuple(R.SCALES), NITER=int(R.NITER), GAIN=float(R.GAIN), MGAIN=float(R.MGAIN), BIAS=float(R.BIAS),
             AUTOMASK=float(R.AUTOMASK), AUTOTHRESH=float(R.AUTOTHRESH), PATCH=int(R.PATCH), MAXACT=int(R.MAXACT),
             MAXLOOP=int(R.MAXLOOP), MAXCYC=int(R.MAXCYC), TAU=float(R.TAU), STEP_FWHM_DEG=float(R.STEP_FWHM_DEG),
             CG_TOL=float(R.CG_TOL), CG_MAX=int(R.CG_MAX), LFSS_DIR=R.LFSS_DIR, CATALOGS=R.CATALOGS,
             PRIOR_SCALE_RANGE=tuple(R.PRIOR_SCALE_RANGE), PRIOR_RMS_FRAC=float(R.PRIOR_RMS_FRAC),
             COMPACT_TOL=float(R.COMPACT_TOL), COMPACT_TOL_LOW=float(R.COMPACT_TOL_LOW), COMPACT_LOW_SCOPES=tuple(R.COMPACT_LOW_SCOPES),
             COMPACT_N_SOURCES=int(R.COMPACT_N_SOURCES), COMPACT_N_MIN=int(R.COMPACT_N_MIN),
             LEVEL_LIMITS=tuple(R.LEVEL_LIMITS), DEC_BANDS=tuple(tuple(b) for b in R.DEC_BANDS),
             CAVEAT_SCOPES=tuple(H.CAVEAT_SCOPES), HYB_SMALL_LIMIT=float(CFG.hybrid_unit.SMALL_LIMIT), RECON_GEN=H.RECON_GEN,
             ARCHIVE_MASTERS=H.ARCHIVE_MASTERS, FFTW=int(H.FFT_WORKERS))
    globals().update(g)
    _BOUND.clear()
    _BOUND.update(g)
    return g


def _dump(rec, path):
    """The unit record (production: json.dump(rec, open(path, 'w'), indent=1, default=str)) through a temporary name."""
    tmp = f'{path}.part'
    with open(tmp, 'w') as f:
        json.dump(rec, f, indent=1, default=str)
    os.replace(tmp, path)


# ------------------------------------------------------------------------------------------------------------------------- FFT helpers
def gauss_kernel_fft(shape, fwhm_px):
    fy = np.fft.fftfreq(shape[0])[:, None]; fx = np.fft.rfftfreq(shape[1])[None, :]; s = fwhm_px / 2.3548
    return np.exp(-2 * (np.pi * s) ** 2 * (fx ** 2 + fy ** 2))


def smooth(a, fwhm_deg, cdelt, mask=None):
    """Gaussian smoothing (FWHM in degrees), mask-normalized, zero-padded."""
    fwhm_px = fwhm_deg / cdelt; good = np.isfinite(a) if mask is None else (mask & np.isfinite(a))
    pad = int(min(3 * fwhm_px, a.shape[0])); sh = (a.shape[0] + 2 * pad, a.shape[1] + 2 * pad)
    A = np.zeros(sh); W = np.zeros(sh); A[pad:pad + a.shape[0], pad:pad + a.shape[1]] = np.where(good, a, 0.0)
    W[pad:pad + a.shape[0], pad:pad + a.shape[1]] = good; k = gauss_kernel_fft(sh, fwhm_px)
    As = sfft.irfft2(sfft.rfft2(A, workers=FFTW) * k, s=sh, workers=FFTW); Ws = sfft.irfft2(sfft.rfft2(W, workers=FFTW) * k, s=sh, workers=FFTW)
    out = (As / np.where(Ws > 1e-3, Ws, np.nan))[pad:pad + a.shape[0], pad:pad + a.shape[1]]; out[~good] = np.nan
    return out


def conv_kernel(a, k, ky=None, kx=None):
    """Linear convolution of an image with a kernel of any size, kernel center at (ky, kx) (default its argmax)."""
    if ky is None:
        ky, kx = np.unravel_index(np.nanargmax(k), k.shape)
    ny, nx = a.shape; sh = (ny + k.shape[0], nx + k.shape[1])
    A = np.zeros(sh); A[:ny, :nx] = np.nan_to_num(a); K = np.zeros(sh); K[:k.shape[0], :k.shape[1]] = np.nan_to_num(k)
    return sfft.irfft2(sfft.rfft2(A, workers=FFTW) * sfft.rfft2(K, workers=FFTW), s=sh, workers=FFTW)[ky:ky + ny, kx:kx + nx]


def beam_image(hdr, shape):
    """Restoring beam (peak 1) from BMAJ/BMIN/BPA, centered at (N//2, N//2)."""
    cd = abs(float(hdr['CDELT1'])); smaj = float(hdr['BMAJ']) / cd / 2.3548; smin = float(hdr['BMIN']) / cd / 2.3548
    pa = np.radians(float(hdr.get('BPA', 0.0))); y, x = np.mgrid[0:shape[0], 0:shape[1]]; y = y - shape[0] // 2; x = x - shape[1] // 2
    u = -x * np.sin(pa) + y * np.cos(pa); w = x * np.cos(pa) + y * np.sin(pa)
    return np.exp(-0.5 * ((u / smaj) ** 2 + (w / smin) ** 2))


READS = set()                                      # v2: the sky-model and catalog files a reconstruction read (recorded as unit inputs)


def _lfss(f):
    import healpy as hp
    READS.add(f'{LFSS_DIR}/healpix/healpix-all-sky-rav-rsclean-map-{f}.fits')
    a = np.array(fits.open(f'{LFSS_DIR}/healpix/healpix-all-sky-rav-rsclean-map-{f}.fits', memmap=False)[1].data.field(0)).ravel().astype(np.float64)
    return np.where(np.isfinite(a) & (a > 0) & (a != hp.UNSEEN), a, np.nan)


_LFSM_MAPS = []


def _lfsm(f):
    from scipy.interpolate import interp1d
    READS.update((f'{LFSS_DIR}/lfsm/lfsm_component_maps_3.0deg.dat', f'{LFSS_DIR}/lfsm/lfsm_components.dat'))
    if not _LFSM_MAPS:                             # the 3.0-deg component maps, parsed once per process
        _LFSM_MAPS.append(np.loadtxt(f'{LFSS_DIR}/lfsm/lfsm_component_maps_3.0deg.dat'))
    maps = _LFSM_MAPS[0]
    rows = [[float(x) for x in ln.split()] for ln in open(f'{LFSS_DIR}/lfsm/lfsm_components.dat') if ln.strip() and not ln.startswith('#')]
    fr, sg, cp = np.array([r[0] for r in rows]), np.array([r[1] for r in rows]), np.array([r[2:] for r in rows])
    scale = np.exp(interp1d(np.log(fr), np.log(sg), kind='slinear')(np.log(f)))
    return scale * sum(interp1d(np.log(fr), cp[:, i], kind='cubic')(np.log(f)) * maps[:, i] for i in range(cp.shape[1]))


def sky_temperature(f_mhz):
    """Sky temperature [K] at f_mhz, equatorial HEALPix 256, peeled sources removed; returns (T, record)."""
    import healpy as hp
    grid = [35, 38, 40, 45, 50, 60, 70, 74, 80]
    if 35 <= f_mhz <= 80:
        lo = max(x for x in grid if x <= f_mhz); hi = min(x for x in grid if x >= f_mhz); tl = _lfss(lo)
        if hi == lo:
            T = tl; how = f'LFSS {lo} MHz'
        else:
            th = _lfss(hi); beta = np.clip(np.log(th / tl) / np.log(hi / lo), -3.2, -2.0); T = tl * (f_mhz / lo) ** beta
            how = f'LFSS {lo}-{hi} MHz per-pixel power law'
    else:
        edge = 35 if f_mhz < 35 else 80
        T = _lfss(edge) * _lfsm(f_mhz) / _lfsm(edge); how = f'LFSS {edge} MHz x LFSM({f_mhz:.2f})/LFSM({edge})'
    theta, _ = hp.pix2ang(256, np.arange(hp.nside2npix(256))); dec = 90 - np.degrees(theta)
    miss = ~np.isfinite(T); dcap = float(dec[np.isfinite(T)].min()); M = _lfsm(f_mhz); strip = np.isfinite(T) & (dec < dcap + 10)
    ratio = float(np.median(T[strip] / M[strip])); T = np.where(miss, M * ratio, T)
    fw = {35: 4.7, 38: 4.3, 40: 4.1, 45: 3.6, 50: 3.3, 60: 2.7, 70: 2.3, 74: 2.2, 80: 2.0}
    fwhm = fw[min(fw, key=lambda k: abs(k - f_mhz))] if 35 <= f_mhz <= 80 else (4.7 if f_mhz < 35 else 3.0)
    rec = dict(sky_T=how, lfss_southern_edge_dec=dcap, cap_fill_ratio=ratio, peeled={})
    for name, (sra, sdec) in {'Cyg A': (299.8682, 40.7339), 'Cas A': (350.8500, 58.8150), 'Vir A': (187.7059, 12.3911)}.items():
        R = max(1.5 * fwhm, 5.0); v0 = hp.ang2vec(np.radians(90 - sdec), np.radians(sra))
        inner = hp.query_disc(256, v0, np.radians(R)); outer = np.setdiff1d(hp.query_disc(256, v0, np.radians(2.5 * R)), hp.query_disc(256, v0, np.radians(1.5 * R)))

        def tp(ix):
            th, ph = hp.pix2ang(256, ix); return ((np.degrees(ph) - sra + 540) % 360 - 180) * np.cos(np.radians(sdec)), 90 - np.degrees(th) - sdec
        xo, yo = tp(outer); ok = np.isfinite(T[outer]); A = np.vstack([np.ones(ok.sum()), xo[ok], yo[ok]]).T
        cf_ = np.linalg.lstsq(A, T[outer][ok], rcond=None)[0]; xi, yi = tp(inner); before = float(np.nansum(T[inner]))
        T[inner] = cf_[0] + cf_[1] * xi + cf_[2] * yi; rec['peeled'][name] = before - float(np.nansum(T[inner]))
    return T, rec


def diffuse_prior(band, ind, freq_mhz=None):
    """Prior model image [Jy per pixel, apparent] on the cell's master grid (the Robust -0.75 master and PB map copied into the
    inputs directory ind), defined to the true horizon; returns (prior, sky = diffuse PB >= 0.1 (the prior fit and the level test),
    horizon = every pixel with l^2 + m^2 < 1 (CLEAN and the step), record)."""
    from reproject import reproject_from_healpix
    mfile = f'{ind}/I_deep_NoTaper_R075_{band}.fits'; mh = fits.getheader(mfile)
    f_hz = float(freq_mhz) * 1e6 if freq_mhz else float(mh['CRVAL3']); N = int(mh['NAXIS1']); cd = abs(float(mh['CDELT1'])); w = WCS(mh).celestial
    T, rec = sky_temperature(f_hz / 1e6)
    Tg, _ = reproject_from_healpix((T, 'icrs'), w, shape_out=(N, N), nested=False)
    ph = fits.getheader(f'{ind}/PB_deep_R075_{band}.fits'); pb = rd(f'{ind}/PB_deep_R075_{band}.fits')
    ch, _ = PBA.chromatic_gain(ph, pb.shape, float(PBA.cfg.SUBBAND_FREQ_HZ[band]['center']), float(ph['PBBMAJ']))
    pbd = pb / ch                                   # model PB x residual term (Gregg 2026-10-26: keep it); no bandwidth smearing
    yy, xx = np.indices((N, N)); l_ = np.radians((xx + 1 - mh['CRPIX1']) * mh['CDELT1']); m_ = np.radians((yy + 1 - mh['CRPIX2']) * mh['CDELT2'])
    n = np.sqrt(np.clip(1 - l_ ** 2 - m_ ** 2, 0, 1)); k2j = 2 * K_B * np.radians(cd) ** 2 / (C_LIGHT / f_hz) ** 2 / 1e-26
    sky = np.isfinite(rd(mfile)) & np.isfinite(pbd) & (pbd >= 0.1) & (n > 0) & np.isfinite(Tg)      # the prior fit and the level test
    horizon = np.isfinite(pbd) & (pbd > 0) & (n > 0) & np.isfinite(Tg)   # n > 0: l^2 + m^2 < 1, the true horizon (Gregg 2026-10-28)
    prior = np.where(horizon, Tg * k2j * pbd / np.where(n > 0, n, 1), 0.0)
    rec.update(freq_mhz=f_hz / 1e6, k_to_jy_per_pixel=k2j, prior_total_jy=float(prior[sky].sum()), prior_total_horizon_jy=float(prior[horizon].sum()))
    rec['test_prior_backconversion'] = test_prior_backconversion(prior, horizon, Tg, pbd, n, k2j, T, w)
    return prior, sky, horizon, rec


def test_prior_backconversion(prior, sky, Tg, pbd, n, k2j, T, w):
    """Jy/pixel converted back to K must reproduce the HEALPix sky temperature at 2,000 random sky pixels (median ratio 1 +- 1e-6)."""
    import healpy as hp
    rng = np.random.default_rng(1); iy, ix = np.where(sky); s = rng.choice(len(iy), min(2000, len(iy)), replace=False); iy, ix = iy[s], ix[s]
    ra, de = w.all_pix2world(ix, iy, 0); hv = hp.get_interp_val(T, np.radians(90 - de), np.radians(ra))
    r = np.median(prior[iy, ix] / k2j / pbd[iy, ix] * n[iy, ix] / hv)
    if not abs(r - 1) < 1e-6:
        raise ReconRefusal(f'test_prior_backconversion FAILED: median ratio {r} - REFUSING')
    return dict(median_ratio=float(r), verdict='pass')


# ------------------------------------------------------------------------------------------------------------------- PSF and calibration
def test_psf2x(qh, Q, Ps, mh, v):
    """The double-size PSF must be the cell's band and weighting: WSClean weighting Briggs(-0.75) / Briggs(0), the master's pixel scale,
    a frequency inside the band, and a main lobe (r < 5 px) matching the stack PSF to a relative rms below 0.2 (pilot, matching
    weighting: 0.022-0.096; the other weighting: 0.40-0.61)."""
    want = {'R075': "Briggs'(-0.75)", 'R0': "Briggs'(0)"}[v]; Pn = np.nan_to_num(Ps) / np.nanmax(Ps); N = Pn.shape[0]
    c1 = np.unravel_index(np.argmax(Pn), Pn.shape); Qn = np.nan_to_num(Q) / np.nanmax(Q); c2 = np.unravel_index(np.argmax(Qn), Qn.shape)
    d = Qn[c2[0] - c1[0]:c2[0] - c1[0] + N, c2[1] - c1[1]:c2[1] - c1[1] + N] - Pn; yy, xx = np.indices(Pn.shape); core = np.hypot(yy - c1[0], xx - c1[1]) < 5
    res = dict(weighting=str(qh.get('WSCWEIGH')), cdelt=float(qh['CDELT1']), freq_hz=float(qh['CRVAL3']),
               core_rel_rms=float(np.sqrt(np.mean(d[core] ** 2)) / np.sqrt(np.mean(Pn[core] ** 2))))
    ok = (res['weighting'] == want and abs(abs(res['cdelt']) - abs(float(mh['CDELT1']))) < 1e-9
          and float(mh['FREQ_MIN']) <= res['freq_hz'] <= float(mh['FREQ_MAX']) and res['core_rel_rms'] < 0.2)
    if not ok:
        raise ReconRefusal(f'test_psf2x FAILED ({v}, want {want}, CDELT {mh["CDELT1"]}, band {mh["FREQ_MIN"]}-{mh["FREQ_MAX"]} Hz): {res} - REFUSING')
    res['verdict'] = 'pass'
    return res


def splice_psf(psf_stack, psf2x):
    """The stack's own PSF inside +-N/2 of the peak, the double-size PSF beyond (peak 1)."""
    Ps = np.nan_to_num(psf_stack) / np.nanmax(psf_stack); N = Ps.shape[0]; cy, cx = np.unravel_index(np.argmax(Ps), Ps.shape)
    Q = np.nan_to_num(psf2x) / np.nanmax(psf2x); qy, qx = np.unravel_index(np.argmax(Q), Q.shape)
    if Q.shape[0] < 2 * N - 2:
        raise ReconRefusal(f'double-size PSF {Q.shape} is not about twice the image size {N} - REFUSING')
    S = Q.copy(); S[qy - cy:qy - cy + N, qx - cx:qx - cx + N] = Ps
    return S


def prior_scale(prior, dirty, psf, sky, hdr):
    """Scale of the prior to the image's own flux scale: dirty = a (prior (x) PSF) + c at 4 deg, 10-deg discs around the peeled sources out."""
    N = dirty.shape[0]; cd = abs(float(hdr['CDELT1'])); yy, xx = np.indices((N, N)); ra, dec = WCS(hdr).celestial.all_pix2world(xx, yy, 0)
    far = np.ones((N, N), bool)
    for sra, sdec in ((299.8682, 40.7339), (350.8500, 58.8150), (187.7059, 12.3911)):
        cs = np.sin(np.radians(dec)) * np.sin(np.radians(sdec)) + np.cos(np.radians(dec)) * np.cos(np.radians(sdec)) * np.cos(np.radians(ra - sra))
        far &= ~(np.degrees(np.arccos(np.clip(cs, -1, 1))) < 10)
    m = sky & far & np.isfinite(dirty); Ds = smooth(dirty, 4.0, cd, m); Ps = smooth(conv_kernel(prior, psf), 4.0, cd, m)
    ok = m & np.isfinite(Ds) & np.isfinite(Ps); A = np.vstack([Ps[ok], np.ones(ok.sum())]).T
    (a, c), *_ = np.linalg.lstsq(A, Ds[ok], rcond=None); rms = float(np.std(Ds[ok] - a * Ps[ok] - c))
    bands = {}
    for lo, hi in ((40, 90), (10, 40), (-10, 10), (-20, -10), (-30, -20)):
        b = ok & (dec >= lo) & (dec < hi)
        if b.sum() > 200:
            Ab = np.vstack([Ps[b], np.ones(b.sum())]).T; (ab, _), *_ = np.linalg.lstsq(Ab, Ds[b], rcond=None)
            bands[f'{lo:+d}..{hi:+d}'] = [round(float(ab), 4), round(float(np.corrcoef(Ps[b], Ds[b])[0, 1]), 4)]
    rec = dict(scale=float(a), offset=float(c), rms=rms, data_rms=float(np.std(Ds[ok])), by_dec_band_scale_corr=bands)
    rec['test_prior_fit'] = test_prior_fit(rec)
    return rec


def test_prior_fit(rec):
    """The survey prior must describe the dirty stack at 4 deg: scale 0.5-2.0 (a unit, frequency or beam error moves it by ~2x or more;
    production 18-36 MHz runs 0.63-1.69 with LST, 27 MHz 09h/10h at 1.63-1.69 continuing 06h-08h's 1.46-1.59; Gregg 2026-10-27 'go with
    your recommendation') and
    residual rms at most 0.6 of the data rms (fit correlation >= 0.8; a wrong grid or orientation gives ~0). The fraction rises with
    Galactic latitude, where 4-deg structure is weak: 18 / 23 MHz 0.14-0.25 near the plane, 0.32-0.43 / 0.39-0.52 at LST 09h-12h (the
    pilot's 0.5 refused 10h and 11h 23 MHz at 0.52, on that trend)."""
    if not (PRIOR_SCALE_RANGE[0] < rec['scale'] < PRIOR_SCALE_RANGE[1] and rec['rms'] <= PRIOR_RMS_FRAC * rec['data_rms']):
        raise ReconRefusal(f'test_prior_fit FAILED: scale {rec["scale"]:.3f}, rms {rec["rms"]:.3f} vs data {rec["data_rms"]:.3f} - REFUSING')
    return 'pass'


def half_stack_noise(halfA, halfB, sky, hdr):
    cd = abs(float(hdr['CDELT1'])); d = (halfA - halfB) / 2
    return float(np.nanstd(smooth(np.nan_to_num(d), STEP_FWHM_DEG, cd, sky)[sky]))


# ------------------------------------------------------------------------------------------------------------------------------ CLEAN
def idclean(D, P, mask, beam_hdr, scales=None, niter=None, test_mask=None, maxcyc=None):
    """Image-domain capped multiscale CLEAN with exact image-plane major cycles (model (x) PSF), over `mask`; the convergence test is
    taken over test_mask (default `mask`). Returns (model Jy/pixel, its point (scale-0) components, residual, restored image, record)."""
    scales = SCALES if scales is None else scales; niter = NITER if niter is None else niter     # v2: defaults bound at call time
    N = D.shape[0]; D = np.nan_to_num(D); P = np.nan_to_num(P / np.nanmax(P)); pky, pkx = np.unravel_index(np.argmax(P), P.shape)
    CM = mask & np.isfinite(D)

    def gk(size, widen=1.0):
        if size == 0:
            return np.ones((1, 1))
        sg = widen * 0.45 * size / 2.3548; h = int(np.ceil(4 * sg)); y, x = np.mgrid[-h:h + 1, -h:h + 1]
        g = np.exp(-(x * x + y * y) / (2 * sg * sg)); return g / g.sum()
    KS = [gk(s) for s in scales]; BP, BETA = [], []
    for s in scales:
        kk = gk(s, np.sqrt(2.0)); h = PATCH + kk.shape[0] // 2 + 1; sub = P[pky - h:pky + h + 1, pkx - h:pkx + h + 1]
        full = conv_kernel(sub, kk, kk.shape[0] // 2, kk.shape[1] // 2) if kk.size > 1 else sub
        c = full.shape[0] // 2; pt = full[c - PATCH:c + PATCH + 1, c - PATCH:c + PATCH + 1].copy(); BP.append(pt); BETA.append(float(pt[PATCH, PATCH]))
    SHP = (N + P.shape[0], N + P.shape[1]); KK = np.zeros(SHP); KK[:P.shape[0], :P.shape[1]] = P; PF = sfft.rfft2(KK, workers=FFTW); del KK

    def response(Mod):
        A = np.zeros(SHP); A[:N, :N] = Mod
        return sfft.irfft2(sfft.rfft2(A, workers=FFTW) * PF, s=SHP, workers=FFTW)[pky:pky + N, pkx:pkx + N]
    smallest = min(s for s in scales if s > 0); BOOST = [1.0 if s == 0 else BIAS ** (-np.log2(s / smallest)) for s in scales]

    def local_rms(R):
        b = 64; nb = int(np.ceil(N / b)); g = np.full((nb, nb), np.nan)
        for i in range(nb):
            for j in range(nb):
                blk = R[i * b:(i + 1) * b, j * b:(j + 1) * b][CM[i * b:(i + 1) * b, j * b:(j + 1) * b]]
                if blk.size > 200:
                    g[i, j] = 1.4826 * np.median(np.abs(blk - np.median(blk)))
        med = np.nanmedian(g); g = np.where(np.isfinite(g), g, med)
        return np.maximum(zoom(g, b, order=1)[:N, :N], 0.2 * med)
    M = np.zeros((N, N)); M0 = np.zeros((N, N)); R = D.copy(); n_total = 0; stage = 1; region = CM.copy(); cyc = 0; rec = dict(scales=list(scales), events=[])
    while n_total < niter and cyc < (maxcyc or MAXCYC):
        cyc += 1; sig = local_rms(R)
        Rs = [R if KS[i].size == 1 else conv_kernel(R, KS[i], KS[i].shape[0] // 2, KS[i].shape[1] // 2) for i in range(len(scales))]
        scores = [np.max(np.abs(Rs[i][region]) / sig[region]) * BOOST[i] for i in range(len(scales))]
        z0 = float(np.max(np.abs(R[region]) / sig[region])); stop = AUTOMASK if stage == 1 else AUTOTHRESH
        if z0 < stop:
            if stage == 1:
                stage = 2; region = CM & binary_dilation(np.abs(M) > 0, iterations=5); rec['events'].append(f'cycle {cyc}: stage 2 at {z0:.2f} sigma'); continue
            rec['events'].append(f'cycle {cyc}: auto-threshold at {z0:.2f} sigma'); break
        si = int(np.argmax(scores)); Rsi = Rs[si]; w = 1.0 / sig; zmap = np.abs(Rsi) * w; zmap[~region] = 0; zpk = float(zmap.max())
        lim = max((1 - MGAIN) * zpk, stop / BOOST[si] if si else stop); idx = np.flatnonzero(zmap > lim * 0.999)
        if len(idx) > MAXACT:
            idx = idx[np.argpartition(-zmap.ravel()[idx], MAXACT)[:MAXACT]]
        iy, ix = np.divmod(idx, N); vals = Rsi.ravel()[idx].copy(); wv = w.ravel()[idx]; comp = np.zeros(len(idx)); B = BP[si]; beta = BETA[si]; nl = 0
        while n_total < niter and nl < MAXLOOP:
            k = int(np.argmax(np.abs(vals) * wv))
            if abs(vals[k]) * wv[k] < lim:
                break
            A_ = GAIN * vals[k] / beta; comp[k] += A_; dy = iy - iy[k]; dx = ix - ix[k]; ins = (np.abs(dy) <= PATCH) & (np.abs(dx) <= PATCH)
            vals[ins] -= A_ * B[PATCH + dy[ins], PATCH + dx[ins]]; n_total += 1; nl += 1
        C = np.zeros(N * N); np.add.at(C, idx, comp); C = C.reshape(N, N)
        M += C if KS[si].size == 1 else conv_kernel(C, KS[si], KS[si].shape[0] // 2, KS[si].shape[1] // 2)
        if scales[si] == 0:
            M0 += C                                  # point components = the compact sources (split beam correction)
        R = D - response(M)
    I = conv_kernel(M, beam_image(beam_hdr, (N, N)), N // 2, N // 2) + R
    TM = CM if test_mask is None else (test_mask & CM)
    rec.update(components=n_total, cycles=cyc, model_total_jy=float(M.sum()), residual_rms_in_mask=float(np.std(R[CM])), dirty_rms_in_mask=float(np.std(D[CM])),
               residual_rms_test_region=float(np.std(R[TM])), dirty_rms_test_region=float(np.std(D[TM])))
    rec['test_clean_converged'] = test_clean_converged(rec)
    return M, M0, R, I, rec


def test_clean_converged(rec):
    """The CLEAN must not diverge: over the test region (the diffuse PB >= 0.1 area where the limit was calibrated), residual rms below
    the dirty rms and below 2 Jy/beam (the diverged Robust 0 run at 02h 46 MHz: model 4.6-5.1 MJy, residual 1.1-1.3 Jy/beam; converged runs
    0.22-0.42 at 46 MHz, 0.31-1.15 at 18 MHz). CLEAN itself runs to the true horizon (2026-10-28), where the 0-10 deg band holds the brightest
    residuals: over the whole CLEAN region 18 MHz LST 17h-20h came out 2.2-3.5 Jy/beam against dirty 20-69."""
    if not (rec['residual_rms_test_region'] < rec['dirty_rms_test_region'] and rec['residual_rms_test_region'] < 2.0):
        raise ReconRefusal(f'test_clean_converged FAILED: {rec} - REFUSING')
    return 'pass'


# ------------------------------------------------------------------------------------------------------------------------------- step
def ls_step(I, R, K, P, prior, sky, hdr, sig0):
    """Tikhonov large-scale step toward the prior (1-deg smoothing, 5-px coarse grid); the constant and the two linear gradient modes
    over the sky are projected out of the data term, so the survey prior alone sets them (Gregg 2026-10-28 "Both approved": the data barely
    constrain them - swapping the far-sidelobe PSF swung 12h 46 MHz's north from 0.99 to 0.20 of the survey; with the gradients from the
    survey 12h / 13h 46 MHz level 0.81-1.02 / 0.87-1.00, 18h within 5% of before). Returns (new master, x, record)."""
    cd = abs(float(hdr['CDELT1'])); N = I.shape[0]; P = P / np.nanmax(P); SKY = sky & np.isfinite(I)
    f = 5; o = f // 2; cy = np.arange(o, N, f); NC = len(cy); samp = lambda a: a[np.ix_(cy, cy)]; skyc = samp(SKY.astype(float)) > 0.5
    sig = np.full((NC, NC), sig0); rc = np.nan_to_num(samp(smooth(R, STEP_FWHM_DEG, cd, SKY)))
    Ps = smooth(P, STEP_FWHM_DEG, cd, np.isfinite(P)); pk = np.unravel_index(np.nanargmax(P), P.shape); nk = NC
    iy = pk[0] + f * np.arange(-nk, nk + 1); vy = (iy >= 0) & (iy < P.shape[0])
    Kc = np.zeros((2 * nk + 1, 2 * nk + 1)); Kc[np.ix_(vy, vy)] = np.nan_to_num(Ps[np.ix_(iy[vy], iy[vy])]); SH = (NC + 2 * nk + 1,) * 2
    KF = sfft.rfft2(np.pad(Kc, ((0, SH[0] - Kc.shape[0]), (0, SH[1] - Kc.shape[1]))), workers=FFTW)
    KFL = sfft.rfft2(np.pad(Kc[::-1, ::-1], ((0, SH[0] - Kc.shape[0]), (0, SH[1] - Kc.shape[1]))), workers=FFTW)

    def cv(x, kk):
        A = np.zeros(SH); A[:NC, :NC] = x
        return sfft.irfft2(sfft.rfft2(A, workers=FFTW) * kk, s=SH, workers=FFTW)[nk:nk + NC, nk:nk + NC]
    blk = lambda a: np.nan_to_num(samp(smooth(a, STEP_FWHM_DEG, cd, SKY))) * f * f
    x0 = blk(prior - K) * skyc
    tau = np.sqrt((TAU * np.abs(blk(prior)) + 0.05 * np.nanmedian(np.abs(blk(prior)[skyc])) + 1e-6) ** 2 + blk(K) ** 2)
    W = skyc / sig ** 2; T1 = skyc / tau ** 2
    yy, xx = np.meshgrid(cy, cy, indexing='ij'); lc = (xx - N / 2) / N; mc = (yy - N / 2) / N
    B = np.stack([skyc.astype(float), lc * skyc, mc * skyc]); Gi = np.linalg.inv(np.einsum('kij,ij,lij->kl', B, W, B))   # constant, l, m

    def Mop(e):                                    # (W - W B G^-1 B^T W) e: the prior alone sets the constant and the two gradient modes
        We = W * e; return We - W * np.einsum('k,kij->ij', Gi @ np.einsum('kij,ij->k', B, We), B)
    Aop = lambda x: cv(skyc * Mop(skyc * cv(x * skyc, KF)), KFL) * skyc + T1 * x
    b = cv(skyc * Mop(skyc * rc), KFL) * skyc + T1 * x0
    x = x0.copy(); r_ = b - Aop(x); p = r_.copy(); rs = float(np.sum(r_ * r_)); bn = float(np.sqrt(np.sum(b * b))); it = 0
    for it in range(CG_MAX):
        Ap = Aop(p); al = rs / float(np.sum(p * Ap)); x += al * p; r_ -= al * Ap; rn = float(np.sum(r_ * r_))
        if np.sqrt(rn) < CG_TOL * bn:
            break
        p = r_ + (rn / rs) * p; rs = rn
    x *= skyc; xf = np.zeros((N, N)); z = zoom(x / (f * f), f, order=3); xf[:min(N, z.shape[0]), :min(N, z.shape[1])] = z[:N, :N]
    xf = np.nan_to_num(smooth(xf, STEP_FWHM_DEG / 2, cd, SKY)) * SKY
    new = I - conv_kernel(xf, P) + conv_kernel(xf, beam_image(hdr, (N, N)), N // 2, N // 2)
    chi = lambda e: float(np.sum((e * Mop(e))[skyc]) / skyc.sum())
    rec = dict(cg_iterations=it + 1, cg_relative_residual=float(np.sqrt(rn) / bn), sigma_1deg=sig0, x_total_jy=float(xf.sum()),
               projected_modes='constant + l + m', chi2_dc_projected_before=chi(rc), chi2_dc_projected_after=chi(rc - cv(x * skyc, KF)))
    rec['test_step_converged'] = test_step_converged(rec)
    return new, xf, rec


def test_step_converged(rec):
    """The step's conjugate-gradient solve must reach a relative residual of 1e-6 within CG_MAX iterations (pilot: 511-2710 iterations;
    the diverged 82 MHz CLEAN left 2.3e-5 after 3000)."""
    if not rec['cg_relative_residual'] < CG_TOL:
        raise ReconRefusal(f'test_step_converged FAILED: {rec} - REFUSING')
    return 'pass'


# ------------------------------------------------------------------------------------------------------------------- product tests
def test_compact_sources(new, master, hdr, lst, band):
    """Compact sources of the new master against the installed master at the 200 brightest catalogued sources (background-subtracted
    apertures): median ratio within 10% of 1, 20% at 18 MHz where the flux scale is very uncertain (Gregg 2026-10-26: do not refuse for
    that; pilot: 0.992-1.017 at 46-82 MHz, 1.032-1.045 at 18 MHz)."""
    res = compact_ratio(new, master, hdr, lst, band)
    if not (res['n'] >= COMPACT_N_MIN and abs(res['median'] - 1) < res['tolerance']):
        raise ReconRefusal(f'test_compact_sources FAILED: {res} - REFUSING')
    res['verdict'] = 'pass'
    return res


def compact_ratio(new, master, hdr, lst, band):
    """test_compact_sources' measurement: the ratios of background-subtracted aperture sums of new / master at the 200 brightest
    catalogued sources (n, median, 16th / 84th percentiles) and the tolerance; no verdict."""
    src = []
    READS.add(f'{CATALOGS}/master_R075_{lst}_{band}.gaul.csv')
    for ln in open(f'{CATALOGS}/master_R075_{lst}_{band}.gaul.csv'):
        if ln.startswith('#') or not ln.strip():
            continue
        q = ln.split(',')
        try:
            src.append((float(q[4]) % 360, float(q[6]), float(q[10])))
        except (ValueError, IndexError):
            pass
    Sx = np.array(src); Sx = Sx[np.argsort(-Sx[:, 2])][:COMPACT_N_SOURCES]; sx, sy = WCS(hdr).celestial.all_world2pix(Sx[:, 0], Sx[:, 1], 0)
    cd = abs(float(hdr['CDELT1'])); bpx = float(hdr['BMAJ']) / cd; N = new.shape[0]; rat = []
    R1, R2, R3 = max(2, int(1.5 * bpx)), max(4, int(3 * bpx)), max(6, int(4.5 * bpx))
    for x0, y0 in zip(sx, sy):
        if not (np.isfinite(x0) and np.isfinite(y0)):
            continue
        xi, yi = int(round(x0)), int(round(y0))
        if not (R3 < xi < N - R3 and R3 < yi < N - R3):
            continue
        s1, s2 = new[yi - R3:yi + R3 + 1, xi - R3:xi + R3 + 1], master[yi - R3:yi + R3 + 1, xi - R3:xi + R3 + 1]
        if not (np.isfinite(s1).all() and np.isfinite(s2).all()):
            continue
        yy, xx = np.indices(s1.shape) - R3; rr = np.hypot(yy, xx)
        p_ = np.sum(s1[rr <= R1] - np.median(s1[(rr >= R2) & (rr <= R3)])); q_ = np.sum(s2[rr <= R1] - np.median(s2[(rr >= R2) & (rr <= R3)]))
        if q_ > 0:
            rat.append(p_ / q_)
    rat = np.array(rat); res = dict(n=int(len(rat)), median=float(np.median(rat)), p16=float(np.percentile(rat, 16)), p84=float(np.percentile(rat, 84)))
    res['tolerance'] = COMPACT_TOL_LOW if band in COMPACT_LOW_SCOPES else COMPACT_TOL       # Red holds 18 MHz
    return res


def _level_verdict(res, scope, **extra):
    bad = {k: v for k, v in res.items() if not LEVEL_LIMITS[0] <= v <= LEVEL_LIMITS[1]}
    if bad and scope not in CAVEAT_SCOPES:
        raise ReconRefusal(f'test_large_scale_level FAILED: {res} - REFUSING')
    return dict(ratio_by_dec_band=res, verdict='caveat' if bad else 'pass', outside_limits=bad, limits=list(LEVEL_LIMITS), **extra)


def test_large_scale_level(model_total, prior, sky, hdr, scope=None):
    """Large-scale level (CLEAN model + step model) against the prior, 4-deg smoothing, by Dec band north of -30: every band 0.5-1.6
    (pilot 18-73 MHz: 0.65-1.12; the failed 82 MHz runs: 0.16-0.49). In CAVEAT_SCOPES a band outside is recorded (verdict 'caveat').
    The 4-deg means of the model and the prior per band are kept (a wideband's level = the ratio of their weighted sums)."""
    N = model_total.shape[0]; cd = abs(float(hdr['CDELT1'])); yy, xx = np.indices((N, N)); _, dec = WCS(hdr).celestial.all_pix2world(xx, yy, 0)
    m = sky & np.isfinite(dec); a = smooth(np.nan_to_num(model_total), 4.0, cd, m); b = smooth(np.nan_to_num(prior), 4.0, cd, m)
    res, am, bm = {}, {}, {}
    for lo, hi in DEC_BANDS:
        k = m & (dec >= lo) & (dec < hi) & np.isfinite(a) & np.isfinite(b)
        if k.sum() > 200:
            key = f'{lo:+d}..{hi:+d}'; am[key] = float(np.mean(a[k])); bm[key] = float(np.mean(b[k])); res[key] = round(am[key] / bm[key], 4)
    return _level_verdict(res, scope, model_mean_by_dec_band=am, prior_mean_by_dec_band=bm)


def lscaveat_cards(level, small=None):
    """FITS cards of the recorded high-band caveats: LSCAVEAT (a test outside its 18-59 MHz limit), LSLVL1-5 = the large-scale level of
    the reconstruction / the survey prior per Dec band (DEC_BANDS order), LSSMALL = the hybrid's small-scale residual (test_hybrid)."""
    out = bool(level.get('outside_limits')) or (small is not None and not small < HYB_SMALL_LIMIT)
    c = [('LSCAVEAT', out, 'large-scale test outside 18-59 MHz limit')]
    for i, (lo, hi) in enumerate(DEC_BANDS, 1):
        key = f'{lo:+d}..{hi:+d}'
        if key in level['ratio_by_dec_band']:
            c.append((f'LSLVL{i}', level['ratio_by_dec_band'][key], f'LS level/prior Dec {key}, ok {LEVEL_LIMITS[0]:g}-{LEVEL_LIMITS[1]:g}'))
    if small is not None:
        c.append(('LSSMALL', float(f'{small:.4g}'), f'hybrid small-scale rel rms, ok < {HYB_SMALL_LIMIT:g}'))
    return c


# ------------------------------------------------------------------------------------------------------------------------------ recon
def recon_variant(lst, band, v, workdir, psf2x_path, freq_mhz=None, niter=None, maxcyc=None, clean_region='horizon', escalation=()):
    """CLEAN + step + products for one cell and weighting, from the stack files and copied inputs in workdir. Where the installed files
    are visible (the node holding coadd_sb), they must still match the copies (a master rebuilt since the stack refuses)."""
    bind(); niter = NITER if niter is None else niter; maxcyc = MAXCYC if maxcyc is None else maxcyc      # v2: config constants
    ind = inputs_dir(workdir)
    with open(f'{workdir}/{v}_stack.json') as f:
        srec = json.load(f)
    for b, it in srec['inputs'].items():
        if os.path.exists(it['installed']) and md5(it['installed']) != it['md5']:
            raise ReconRefusal(f'{it["installed"]} changed since the stack (md5) - restack - REFUSING')
    mfile = f'{ind}/I_deep_NoTaper_{v}_{band}.fits'; mh = fits.getheader(mfile); master = rd(mfile)
    D = rd(f'{workdir}/{v}_dirty_stack.fits'); hA, hB = rd(f'{workdir}/{v}_dirty_halfA.fits'), rd(f'{workdir}/{v}_dirty_halfB.fits')
    Ps, Q = rd(f'{workdir}/{v}_psf_stack.fits'), rd(psf2x_path); tpsf = test_psf2x(fits.getheader(psf2x_path), Q, Ps, mh, v)
    P = splice_psf(Ps, Q); del Q
    prior, sky, horizon, prec = diffuse_prior(band, ind, freq_mhz); sky &= np.isfinite(D); horizon &= np.isfinite(D)
    fit = prior_scale(prior, D, P, sky, mh); prior = prior * fit['scale']; s0 = half_stack_noise(hA, hB, sky, mh); log(lst, band, v, 'prior scale', round(fit['scale'], 4))
    adm = srec['test_stack_rebuild'].get('hybrid_scale_admission')
    if adm:                                       # Gregg 2026-10-31 "Admit them as hybrids": its mismatch through G below the half-stack noise
        adm = dict(adm, half_stack_noise_1deg=s0, verdict='pass' if adm['lowpassed_rms'] < s0 else 'refused')
        if adm['verdict'] != 'pass':
            raise ReconRefusal(f'{lst} {band} {v}: stack admitted at hybrid scales, but its mismatch through G is not below the half-stack noise '
                               f'{adm} - REFUSING')
        log(lst, band, v, 'stack admitted at hybrid scales: mismatch through G rms', round(adm['lowpassed_rms'], 5), '< half-stack noise', round(s0, 5))
    cmask = horizon
    if clean_region == 'elev10':                  # fallback after convergence refusals (Gregg 2026-10-28): CLEAN to 10 deg elevation, as WSClean's
        N_ = D.shape[0]; yy_, xx_ = np.indices((N_, N_))                      # -horizon-mask; the step and the prior stay to the true horizon
        r_ = np.hypot(np.radians((xx_ + 1 - mh['CRPIX1']) * mh['CDELT1']), np.radians((yy_ + 1 - mh['CRPIX2']) * mh['CDELT2']))
        cmask = horizon & (r_ < np.cos(np.radians(10.0)))
    K, K0, R, I, crec = idclean(D, P, cmask, mh, niter=niter, test_mask=sky, maxcyc=maxcyc); crec['budget'] = dict(niter=niter, maxcyc=maxcyc); log(lst, band, v, 'CLEAN', crec['components'], 'components, residual', round(crec['residual_rms_in_mask'], 4))
    new, x, strec = ls_step(I, R, K, P, prior, horizon, mh, s0); log(lst, band, v, 'step', strec['cg_iterations'], 'iterations')
    rec = dict(lst=lst, band=band, weighting=v, generation=RECON_GEN, psf2x=psf2x_path, psf2x_md5=md5(psf2x_path), test_psf2x=tpsf, prior=prec,
               regions=dict(clean=('l^2 + m^2 < 1 (true horizon)' if clean_region == 'horizon' else 'elevation >= 10 deg (convergence fallback)'),
                            clean_region=clean_region, step_and_prior='l^2 + m^2 < 1 (true horizon)', prior_fit_noise_level_test='diffuse PB >= 0.1'),
               escalation=list(escalation), prior_fit=fit,
               half_stack_noise_1deg=s0, clean=crec, step=strec)
    if adm:
        rec['stack_admission'] = adm
    rec['test_compact_sources'] = test_compact_sources(new, master, mh, lst, band)
    rec['test_large_scale_level'] = test_large_scale_level(K + x, prior, sky, mh, band)
    C = conv_kernel(K0, beam_image(mh, K0.shape), K0.shape[0] // 2, K0.shape[1] // 2)       # the compact sources, restored
    for name, arr, what in (('model', K + x, 'CLEAN + step model, Jy/pixel'), ('clean_residual', R, 'image-domain CLEAN residual, Jy/beam'),
                            ('compact_restored', C, 'restored point (scale-0) CLEAN components, Jy/beam'),
                            ('final_residual', R - conv_kernel(x, P), 'dirty stack - (CLEAN + step model) (x) PSF, Jy/beam')):
        fits.writeto(f'{workdir}/{v}_lsrecon_{name}.fits', arr.astype(np.float32), diag_header(mh, what), overwrite=True)
    rec['product'] = write_master(new, mh, lst, band, v, workdir, rec, srec['span'], srec.get('dirty_nights_dropped') or [])
    rec['product_pbcorr'] = write_pbcorr_split(rec['product'], C, band, v, ind, workdir, srec['span'])
    _dump(rec, f'{workdir}/{v}_lsrecon.json')
    return rec


def write_pbcorr_split(prod, C, band, v, ind, workdir, span):
    """Beam-corrected sibling with the split correction (Gregg 2026-10-26: the full beam for compact flux, the diffuse-correct beam for
    the NoTaper maps, corrected at the deconvolution stage): the restored point components (compact sources) divided by the full
    primary beam of the weighting (model x bandwidth smearing x residual, the archive convention for compact peaks), everything else
    (extended components, the large-scale step, the residual) by the diffuse beam (model x residual); NaN where the full beam is at or
    below the floor (as pb_apply.apply_pbcorr)."""
    new = rd(prod['path']); h = fits.getheader(prod['path']).copy()
    pbf = f'{ind}/{PBA.deep_pb_name(band, v)}'
    ph = fits.getheader(pbf); pb = rd(pbf)
    ch, _ = PBA.chromatic_gain(ph, pb.shape, float(PBA.cfg.SUBBAND_FREQ_HZ[band]['center']), float(ph['PBBMAJ']))
    valid = np.isfinite(pb) & (pb > PBA.PB_FLOOR) & np.isfinite(new) & np.isfinite(ch) & (ch > 0)
    out = np.full(new.shape, np.nan); out[valid] = C[valid] / pb[valid] + (new[valid] - C[valid]) * ch[valid] / pb[valid]
    pn = parse_name(prod['prodname']); pn['proc'] = list(pn['proc']) + ['pbcorr']; name = build_name(**pn)
    path = f'{workdir}/I_deep_NoTaper_{v}_lsrecon_pbcorr_{band}.fits'
    h['PBFLOOR'] = (PBA.PB_FLOOR, 'PB floor below which pixels set NaN')
    for k in ('PBLIBV', 'PBMODEL', 'PBCHROM', 'PBRESID', 'PBROBUST', 'PBBMAJ'):
        if k in ph:
            h[k] = (ph[k], ph.comments[k])
    h['PBMETHOD'] = ('lsrecon-split', 'compact / full PB; extended + step + residual / diffuse PB')
    h['PBMAP'] = (os.path.basename(pbf), 'full PB map; diffuse PB = it / bandwidth-smearing term')
    fits.writeto(path, out.astype(np.float32).reshape(new.shape if h['NAXIS'] == 2 else (1,) * (h['NAXIS'] - 2) + new.shape), h, overwrite=True)
    write_standard_header(path, scope=band, stokes='I', start=span['start'], end=span['end'], inttime_s=span['inttime_s'], time_basis='coadd-span',
                          beam=(float(h['BMAJ']), float(h['BMIN']), float(h.get('BPA', 0.0))), pbcorr=True, dewarped=True, aligned=True,
                          msub=False, ssub=False, extra={'PRODNAME': (name, 'Phase 3 product name (grammar; CONTINUE cards)')}, validate=True)
    return dict(path=path, prodname=name, md5=md5(path), test_pbcorr_split=test_pbcorr_split(path, new, C, pb, ch, valid))

def installed_master(lst, scope, v, taper):
    """The archived (installed) master of the cell and weighting, Taper or NoTaper, and its beam-corrected sibling: exactly one each,
    PRODNAME equal to the file name, no reconstruction tokens."""
    tok = 'Robust-0.75' if v == 'R075' else 'Robust+0.0'; tp = 'Taper' if taper else 'NoTaper'; a = f'{ARCHIVE_MASTERS}/{lst}/{scope}'
    g = glob.glob(f'{a}/{scope}_I_deep_{tp}_{tok}_*coadd_LST{lst}_*.fits'); m = [f for f in g if '_pbcorr_' not in f]; p = [f for f in g if '_pbcorr_' in f]
    if len(m) != 1 or len(p) != 1:
        raise ReconRefusal(f'{lst} {scope} {tp} {v}: {len(m)} archived masters, {len(p)} beam-corrected - REFUSING')
    pn = parse_name(os.path.basename(m[0]))
    if not (pn.get('grammar') == 'phase3' and pn['scope'] == scope and pn['lst'] == lst and pn['taper'] == tp and pn['robust'] == VARIANT[v][2]
            and 'coadd' in pn['proc'] and not {'lsrecon', 'lsfill'} & set(pn['proc']) and fits.getheader(m[0]).get('PRODNAME') == os.path.basename(m[0])):
        raise ReconRefusal(f'{m[0]}: not the {lst} {scope} {tp} {v} master - REFUSING')
    return m[0], p[0]


# STRIATION-RESTORED TAPER BASE (Gregg 2026-10-31, "All your recommendations are approved"; ledger
# R-GREGG-20261031-HIGH-BAND-HYBRID-PLAN-AND-STRIATION-RESTORED-BASE-APPROVED; 59 MHz R-GREGG-20261031-REBUILD-REPUSH-59MHZ-R075-ON-
# STRIATION-RESTORED-BASE): the hybrids at 59-82 MHz and Blue take as their
# Taper part the striation-restored Taper master (striation_restore.py, 2026-10-22: restored = installed master - fitted striation model)
# wherever the served Taper mosaics (Run_FEATHER3) used it - FEATHER3's selection table - and the installed master everywhere else.
# The restored files are the 2026-10-22 products staged from /data07 to Lustre with md5 (MANIFEST.csv there).
def test_pbcorr_split(path, new, C, pb, ch, valid):
    """The split beam-corrected file read back, times the diffuse beam (full beam / bandwidth-smearing term), must equal
    master + compact x (1/term - 1) to 1e-5 of the largest value where the full beam exceeds the floor, and be NaN elsewhere."""
    P = rd(path); lhs = P[valid] * pb[valid] / ch[valid]; rhs = new[valid] + C[valid] * (1 / ch[valid] - 1)
    res = dict(max_rel_err=float(np.max(np.abs(lhs - rhs)) / float(np.max(np.abs(rhs)))), nan_outside=bool(np.all(np.isnan(P[~valid]))),
               valid_px=int(valid.sum()), term_min=float(np.min(ch[valid])))
    if not (res['max_rel_err'] < 1e-5 and res['nan_outside']):
        raise ReconRefusal(f'test_pbcorr_split FAILED: {res} - REFUSING')
    res['verdict'] = 'pass'
    return res


def write_master(new, mh, lst, band, v, workdir, rec, span, dropped=()):
    """The new master: the installed master's WCS, beam and flux cards; PRODNAME = the installed master's name + 'lsrecon'; the
    constituent-span time block; the full header standard (validate=True); the file read back must equal the array."""
    rob = VARIANT[v][2]; h = mh.copy(); out = f'{workdir}/I_deep_NoTaper_{v}_lsrecon_{band}.fits'
    pn = parse_name(str(mh.get('PRODNAME', '')))
    if not (pn.get('grammar') == 'phase3' and pn['scope'] == band and pn['lst'] == lst and pn['taper'] == 'NoTaper' and pn['robust'] == rob
            and 'coadd' in pn['proc'] and 'lsrecon' not in pn['proc']):
        raise ReconRefusal(f'installed PRODNAME {mh.get("PRODNAME")!r} is not the {lst} {band} NoTaper {v} master - REFUSING')
    pn['proc'] = list(pn['proc']) + ['lsrecon']
    if dropped:                                   # name, NCOADD and constituent lines follow the nights actually stacked
        kept = [d for d in CFP._constituent_dates(mh) if d not in dropped]
        pn.update(date_start=kept[0].replace('-', ''), date_end=kept[-1].replace('-', ''), n_epochs=len(kept))
        hist = [str(x) for x in h.get('HISTORY', []) if not any(str(x).startswith(f'COADD constituent {d}') for d in dropped)]
        h.remove('HISTORY', ignore_missing=True, remove_all=True)
        for x in hist:
            h['HISTORY'] = x
        h['NCOADD'] = len(kept); h['HISTORY'] = f'lsrecon: night(s) not stacked (no Phase 1 dirty image, or no derivable warp screen): {", ".join(dropped)}'
    name = build_name(**pn)
    h['LSRECON'] = (True, 'NoTaper large-scale reconstruction (notaper_largescale)')
    h['LSPRIOR'] = (rec['prior']['sky_T'][:68], 'large-scale prior sky temperature')
    h['LSPSCALE'] = (round(rec['prior_fit']['scale'], 5), 'prior scaled to the image (4-deg fit through the PSF)')
    h['LSFREQ'] = (round(rec['prior']['freq_mhz'], 4), 'MHz, frequency of the prior')
    h['LSSCALES'] = (','.join(str(s) for s in SCALES), 'image-domain CLEAN scales (px)')
    h['LSNCOMP'] = (int(rec['clean']['components']), 'image-domain CLEAN components')
    h['LSSIGMA'] = (round(rec['half_stack_noise_1deg'], 5), 'Jy/beam, step data noise (half-stack, 1 deg)')
    h['LSRGEN'] = (RECON_GEN, 'CLEAN, step, prior to horizon; survey gradients')
    h['LSCLNRGN'] = (rec['regions']['clean_region'], 'CLEAN region: horizon, or elev10 (fallback)')
    if band in CAVEAT_SCOPES:                     # the recorded caveats (Gregg 2026-10-31)
        for k, val, com in lscaveat_cards(rec['test_large_scale_level']):
            h[k] = (val, com)
    if rec.get('stack_admission'):
        h['LSSTKADM'] = (band, 'stack admitted at hybrid scales (Gregg 10-31)')
    h['HISTORY'] = f"lsrecon {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}: dirty stack re-deconvolved in the image domain + large-scale step toward the LWA1 survey prior"
    h['HISTORY'] = (f"lsrecon time block = constituent span {span['start']} - {span['end']}, {span['inttime_s']:.1f} s (installed master: "
                    f"{mh.get('DATE-OBS')} - {mh.get('DATE-END')}, {mh.get('INTTIME')} s)")
    data = new.astype(np.float32)
    fits.writeto(out, data.reshape((1,) * (mh['NAXIS'] - 2) + data.shape) if mh['NAXIS'] > 2 else data, h, overwrite=True)
    write_standard_header(out, scope=band, stokes='I', start=span['start'], end=span['end'], inttime_s=span['inttime_s'],
                          time_basis='coadd-span', beam=(float(mh['BMAJ']), float(mh['BMIN']), float(mh.get('BPA', 0.0))),
                          imgrms=CFP.get_uniform_rms(new, ~np.isfinite(new) | (new == 0)), pbcorr=False, dewarped=True, aligned=True,
                          msub=False, ssub=False, extra={'PRODNAME': (name, 'Phase 3 product name (grammar; CONTINUE cards)'),
                                                         'MSTRGEN': (str(mh.get('COADDOP', ''))[:20], 'master generation of this product')},
                          validate=True)
    back = rd(out)
    if not (np.array_equal(np.isfinite(back), np.isfinite(data)) and np.array_equal(back[np.isfinite(back)], data[np.isfinite(data)].astype(np.float64))
            and fits.getheader(out)['PRODNAME'] == name):
        raise ReconRefusal(f'{out}: read-back differs from the reconstruction - REFUSING')
    return dict(path=out, prodname=name, md5=md5(out))

