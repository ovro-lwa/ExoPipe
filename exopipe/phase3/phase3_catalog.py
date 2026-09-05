#!/usr/bin/env python3
"""
phase3_catalog.py — production candidate catalog + cutouts (PHASE3_STRUCTURE.md).

Shared output layer for BOTH the deep and 10-min detection tiers. Given the
candidates that run_identification_i/v returns on a REAL residual, this:

  - astrometry      : pixel -> (ra, dec) via the residual WCS
  - photometry      : 2-D Gaussian fit on the cutout -> peak + integrated flux
  - cutouts         : FITS (WCS-carrying) + PNG, per candidate
  - body label      : solar-system body name if the position matches an
                      ephemeris (Jupiter/Moon/...), else blank — FLAGGED IN
                      PLACE, the candidate STAYS in the master catalog (Gregg's
                      ruling: no siloing to a separate channel)
  - repeats         : cross-match against the running master_catalog (any LST,
                      any date — de-siloed) -> repeat_group id
  - master_catalog  : one append-only CSV under transients/Run_{label}/

Everything lands in the single top-level transients/ master_catalog so a
transient that recurs across LSTs/dates is one searchable record, not N siloed
ones. Solar-system targets still get forced-photometry light curves via the
separate samples/solar_system/ tree (solar_system_cutout.py), independent of
this blind catalog.
"""
import re
import os, csv, math, fcntl
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, get_body, EarthLocation
from astropy.time import Time
import astropy.units as u
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from exopipe.common import phase3_paths as P
from exopipe.phase3.phase3_c50 import c50_sigma

OVRO = EarthLocation(lat=37.2339 * u.deg, lon=-118.2817 * u.deg, height=1222 * u.m)
EPH_BODIES = ['sun', 'moon', 'mercury', 'venus', 'mars', 'jupiter', 'saturn',
              'uranus', 'neptune']
EPH_MATCH_DEG = 1.0          # ephemeris association radius
REPEAT_MATCH_ARCMIN = 10.0   # repeat cross-match radius (de-siloed, any LST/date)
CUTOUT_DEG = 5.0             # candidate cutout is 5 x 5 DEGREES (Gregg standard 2026-07-18,
                             # re-ruled 2026-09-05: "for the 1000th time 5 degrees"); the
                             # pixel half-size follows the frame's own pixel scale
CUTOUT_HALF_PIX = 40         # legacy fixed half-size (81 px), used only if no pixel scale is readable


def cutout_half_pix(hdr):
    """Half-size in pixels for a CUTOUT_DEG-wide box on this frame."""
    cd = None
    for k in ('CDELT2', 'CDELT1', 'CD2_2', 'CD1_1'):
        if hdr.get(k):
            cd = abs(float(hdr[k])); break
    if not cd:
        return CUTOUT_HALF_PIX
    return int(round(CUTOUT_DEG / cd / 2.0))

# 50%-completeness is MEASURED (blind injection-recovery, phase3_v_c50.py) and read
# via phase3_c50.c50_sigma(band) -> LOCAL-rms sigma of the final gated chain. NO
# hard-wired numbers (Gregg 2026-06-30): an unmeasured band yields a blank c50_mjy.


def _inner_rms(a):
    """Robust rms over the inner quarter (matches the detection rms region)."""
    n = a.shape[0]
    c = a[n // 4:3 * n // 4, n // 4:3 * n // 4]
    c = c[np.isfinite(c) & (c != 0)]
    if c.size < 100:
        return 0.0
    return float(np.std(c[np.abs(c) < np.percentile(np.abs(c), 99)]))


def _gauss2d(c, a, x0, y0, sx, sy, th, o):
    X, Y = c
    xr = (X - x0) * np.cos(th) + (Y - y0) * np.sin(th)
    yr = -(X - x0) * np.sin(th) + (Y - y0) * np.cos(th)
    return (a * np.exp(-0.5 * ((xr / sx) ** 2 + (yr / sy) ** 2)) + o).ravel()


def gaussian_photometry(resid, x, y, bpix, cd):
    """2-D Gaussian fit on a stamp. Returns peak (Jy), integrated flux
    (Jy, = peak * 2pi sx sy / beam_area), fitted position, axis ratio."""
    r = max(8, int(round(2 * bpix)))
    st = resid[y - r:y + r + 1, x - r:x + r + 1]
    if st.shape != (2 * r + 1, 2 * r + 1) or not np.isfinite(st).all():
        return None
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    sgn = np.sign(np.nanmean(st[r - 1:r + 2, r - 1:r + 2])) or 1.0
    try:
        p, _pcov = curve_fit(_gauss2d, (xx, yy), st.ravel(),
                             p0=[sgn * np.nanmax(np.abs(st)), 0, 0, bpix / 2.355,
                                 bpix / 2.355, 0, 0], maxfev=10000)
    except Exception:
        return None
    peak = float(p[0]); sx, sy = abs(p[3]), abs(p[4])
    beam_area = (np.pi * bpix * bpix) / (4 * np.log(2))      # px^2
    integ = peak * 2 * np.pi * sx * sy / max(beam_area, 1e-9)
    mj, mn = max(sx, sy) * 2.355, min(sx, sy) * 2.355
    # 2026-09-04 (Card 9: keep the fit covariance): 1-sigma uncertainty of the
    # major-axis FWHM, for the size test's measurement allowance
    try:
        _imaj = 3 if sx >= sy else 4
        _emaj = float(np.sqrt(abs(_pcov[_imaj, _imaj]))) * 2.355 if np.all(np.isfinite(_pcov)) else float('nan')
    except Exception:
        _emaj = float('nan')
    return dict(peak_jy=round(peak, 4), int_jy=round(float(integ), 4),
                dx=float(p[1]), dy=float(p[2]),
                fwhm_maj_arcmin=round(mj * cd * 60, 2),
                fwhm_maj_err_arcmin=(round(_emaj * cd * 60, 3) if np.isfinite(_emaj) else None),
                axis_ratio=round(mj / max(mn, 0.1), 2))


def ephemeris_label(ra, dec, obs_time):
    """Solar-system body name if (ra, dec) is within EPH_MATCH_DEG of a body's
    apparent position at obs_time, else ''. Flag-in-place only — the candidate
    is NEVER removed; this just populates the catalog's `body` column."""
    co = SkyCoord(ra * u.deg, dec * u.deg)
    t = Time(obs_time)
    for b in EPH_BODIES:
        try:
            bp = get_body(b, t, OVRO)
            if SkyCoord(bp.ra, bp.dec).separation(co).deg < EPH_MATCH_DEG:
                return b
        except Exception:
            pass
    return ''


_WCS_CEL_KEYS = re.compile(r'^(NAXIS\d*|WCSAXES|CTYPE[12]|CRVAL[12]|CRPIX[12]|CDELT[12]|'
                           r'CUNIT[12]|CD\d_\d|PC\d_\d|LONPOLE|LATPOLE|RADESYS|EQUINOX|'
                           r'MJDREF|DATEREF|CRDER[12]|CSYER[12]|SIMPLE|BITPIX|EXTEND|COMMENT|HISTORY)$')


def _cutout(resid, master, x, y, hdr, frame_hdr=None):
    h = cutout_half_pix(hdr)
    n = resid.shape[0]
    x0, x1 = max(0, x - h), min(n, x + h + 1)
    y0, y1 = max(0, y - h), min(n, y + h + 1)
    w = WCS(hdr).celestial
    sub = w[y0:y1, x0:x1]
    rh = sub.to_header()
    if frame_hdr is not None:
        # 2026-09-04 conformant-from-birth cutouts: every non-celestial card
        # of the parent frame's standard header (time block, beam, frequency
        # block incl. degenerate axes 3/4 + WCSAXES=4, Stokes, noise,
        # provenance) is inherited; the cutout WCS stays the sub-image's.
        for card in frame_hdr.cards:
            k = card.keyword
            if not k or _WCS_CEL_KEYS.match(k):
                continue
            rh[k] = (card.value, card.comment)
        rh['WCSAXES'] = frame_hdr.get('WCSAXES', 2)
        for hc in frame_hdr.get('HISTORY', []) if 'HISTORY' in frame_hdr else []:
            rh.add_history(hc)
        rh['CUTOUT'] = (True, 'Candidate cutout of the parent frame')
        rh['CUTPARNT'] = (str(frame_hdr.get('PRODNAME', ''))[:68], 'Parent frame product name')
    else:
        for k in ['BMAJ', 'BMIN', 'BPA', 'BUNIT', 'DATE-OBS', 'RESTFRQ']:
            if k in hdr:
                rh[k] = hdr[k]
    return (resid[y0:y1, x0:x1].astype(np.float32),
            (master[y0:y1, x0:x1].astype(np.float32) if master is not None else None),
            rh)


def _png(resid_cut, master_cut, out_png, title):
    ncol = 2 if master_cut is not None else 1
    fig, axes = plt.subplots(1, ncol, figsize=(5 * ncol, 5), squeeze=False)
    finite = resid_cut[np.isfinite(resid_cut)]
    v = np.nanpercentile(np.abs(finite), 99) if finite.size else 1.0
    axes[0][0].imshow(resid_cut, origin='lower', cmap='RdBu_r', vmin=-v, vmax=v)
    axes[0][0].set_title('residual'); axes[0][0].plot(resid_cut.shape[1] / 2,
                                                      resid_cut.shape[0] / 2, 'g+', ms=14)
    if master_cut is not None:
        mf = master_cut[np.isfinite(master_cut)]
        mv = np.nanpercentile(np.abs(mf), 99) if mf.size else 1.0
        axes[0][1].imshow(master_cut, origin='lower', cmap='gray_r', vmin=-mv, vmax=mv)
        axes[0][1].set_title('master (deep)')
    fig.suptitle(title); fig.tight_layout()
    fig.savefig(out_png, dpi=90); plt.close(fig)


def _read_existing(catalog_csv):
    if not os.path.exists(catalog_csv):
        return [], 0
    rows = []
    with open(catalog_csv) as fh:
        rows = list(csv.DictReader(fh))
    return rows, len(rows)


def write_candidates(root, label, cands, resid, master, hdr, lst, date,
                     scope, stokes, cadence, bpix, cd, obs_time,
                     interval=None, frame_hdr=None):
    """Write each candidate's cutout + photometry + astrometry, label any
    solar-system body in place, cross-match repeats against the running
    master_catalog, and append rows. Returns the list of catalog rows written."""
    run_root = P.transients_run(root, label, make=True)
    catalog_csv = os.path.join(run_root, 'master_catalog.csv')
    _lk = open(os.path.join(run_root, '.catalog.lock'), 'w')
    fcntl.flock(_lk, fcntl.LOCK_EX)   # serialize catalog read+append across parallel cells
    existing, n0 = _read_existing(catalog_csv)
    ex_coords = (SkyCoord([float(r['ra']) for r in existing] * u.deg,
                          [float(r['dec']) for r in existing] * u.deg)
                 if existing else None)
    wcs = WCS(hdr).celestial
    _cs = c50_sigma(scope)          # MEASURED local-rms sigma, or None if unmeasured
    c50_mjy = round(_cs * _inner_rms(resid) * 1000.0, 1) if _cs else ''
    # 'band' = the frequency band (Red/Green/Blue/Full or 18MHz..82MHz). Renamed
    # from 'scope' (Gregg 2026-06-27). Readers alias band<->scope for back-compat.
    fields = ['cand_id', 'lst', 'date', 'band', 'stokes', 'cadence', 'interval',
              'ra', 'dec', 'peak_jy', 'int_jy', 'snr', 'axis_ratio',
              'fwhm_maj_arcmin', 'c50_mjy', 'mstruct', 'mstruct_raw', 'sstack',
              'jupiter_deg', 'n_v_epoch', 'body', 'repeat_group',
              'epoch_qa',   # 'outlier QA' tag: candidate from a coadd-omitted
                            # epoch, carried to transient review (Gregg 2026-08-26)
              'fp_prediction', 'global_snr', 'size_beams', 'size_flag',
              'nearest_ateam', 'moon_deg']   # 2026-09-04 additions
    written = []
    for i, c in enumerate(cands):
        x, y = int(c['pix_x']), int(c['pix_y'])
        sky = wcs.pixel_to_world(x, y)
        ra, dec = float(sky.ra.deg), float(sky.dec.deg)
        phot = gaussian_photometry(resid, x, y, bpix, cd)
        snr = round(float(c.get('local_snr', c.get('snr', 0))), 2)
        body = ephemeris_label(ra, dec, obs_time)
        # repeat cross-match (de-siloed: any LST/date in this Run)
        rg = ''
        co = SkyCoord(ra * u.deg, dec * u.deg)
        if ex_coords is not None and len(ex_coords):
            d = co.separation(ex_coords).arcminute
            if np.min(d) < REPEAT_MATCH_ARCMIN:
                rg = existing[int(np.argmin(d))].get('repeat_group') or \
                     existing[int(np.argmin(d))]['cand_id']
        cid = f'{stokes}{cadence[0].upper()}_{lst}_{date}_{scope}_{n0 + i:04d}'
        cdir = P.candidate_dir(root, label, cid, make=True)
        rc, mc, rh = _cutout(resid, master, x, y, hdr, frame_hdr)
        rh['CANDRA'] = (round(ra, 6), '[deg] Candidate RA'); rh['CANDDEC'] = (round(dec, 6), '[deg] Candidate Dec')
        rh['CANDID'] = (cid, 'Candidate id')
        _rp = os.path.join(cdir, f'{cid}_resid.fits')
        fits.writeto(_rp, rc, rh, overwrite=True)
        if frame_hdr is not None:
            from exopipe.common.validate_fits_headers import validate_header as _vh
            _rep = _vh(_rp, level='standard')
            if not _rep.ok:
                os.remove(_rp)
                raise RuntimeError(f'{cid}: cutout header validation FAILED '
                                   f'({"; ".join(_rep.errors)[:200]}) - REFUSING')
        if mc is not None:
            fits.writeto(os.path.join(cdir, f'{cid}_master.fits'), mc, rh, overwrite=True)
        _png(rc, mc, os.path.join(cdir, f'{cid}.png'),
             f'{cid}  {snr:.1f}sigma{("  " + body) if body else ""}')
        row = dict(cand_id=cid, lst=lst, date=date, band=scope, stokes=stokes,
                   cadence=cadence, interval='' if interval is None else interval,
                   ra=round(ra, 5), dec=round(dec, 5),
                   peak_jy=phot['peak_jy'] if phot else round(float(c['flux']), 4),
                   int_jy=phot['int_jy'] if phot else '',
                   snr=snr, axis_ratio=phot['axis_ratio'] if phot else '',
                   fwhm_maj_arcmin=phot['fwhm_maj_arcmin'] if phot else '',
                   c50_mjy=c50_mjy,
                   mstruct=c.get('mstruct', ''), mstruct_raw=c.get('mstruct_raw', ''),
                   sstack=c.get('sstack', ''), jupiter_deg=c.get('jupiter_deg', ''),
                   n_v_epoch=c.get('n_v_epoch', ''),
                   body=body, repeat_group=rg,
                   epoch_qa=c.get('epoch_qa', ''),
                   fp_prediction=c.get('fp_prediction', ''),
                   global_snr=(round(float(c['global_snr']), 2)
                               if c.get('global_snr') not in (None, '') else ''),
                   size_beams=c.get('size_beams', ''), size_flag=c.get('size_flag', ''),
                   nearest_ateam=c.get('nearest_ateam', ''), moon_deg=c.get('moon_deg', ''))
        written.append(row)
    # append to the single de-siloed master_catalog
    new_file = not os.path.exists(catalog_csv)
    with open(catalog_csv, 'a', newline='') as fh:
        wcsv = csv.DictWriter(fh, fieldnames=fields)
        if new_file:
            wcsv.writeheader()
        for row in written:
            wcsv.writerow(row)
    fcntl.flock(_lk, fcntl.LOCK_UN); _lk.close()
    return written
