"""Injection-calibrated peak-recovery correction maps (PKCORR) of a HEALPix Jy/sr mosaic: a port of production mosaic_pkcorr.py (MP,
exopipe 26ff18f, "the 2026-07-27 harness ... method unchanged") for `exopipe3 mosaic --generation v5`, stage `pkcorr` (spec
plan/specs/hybrid.md 2.13; PR-7b). The point-source set Gregg requires with the mosaics (R-GREGG-20261023-MOSAIC-ALL-PRODUCTS-ON-
LUSTRE-NOTAPER-ABOVE-TAPER-JYSR-AND-POINT-SOURCE-SET: "Versions in Jy/sr are required bot also a set optimized for point source
photometry").

The Jy/sr mosaics' point-source estimator peak x Omega_eff under-reads the true flux (the composite profile of a source is broader than
the weighted-mean beam). Per (variant, scope), mosaic_pkcorr's numbers:
  - Dec nodes DEC_START..DEC_STOP every DEC_STEP, the i-th at RA i x RA_STEP (mod 360), so no two sources are sky-adjacent;
  - ROUNDS rounds, round r offsetting every RA by r x ROUND_OFF;
  - each round adds S_INJ restoring-beam Gaussians (constant in pixel space) to COPIES of the staged tiles where PB > INJ_PB_MIN, runs
    the real co-add (coadd_jysr, the bound mosaic_coadd globals) and differences it against the clean co-add (the unit's unfilled
    Jy/sr map);
  - recovery = (max within max(PEAK_RADIUS_FWHM x FWHM_eff, PEAK_RADIUS_MIN_DEG) of the position) x Omega_eff / S_INJ, Omega_eff =
    the PSF map's median within OMEGA_RADIUS_DEG; per-node median over rounds -> curve -> per-pixel map (linear in Dec, constant
    beyond the ends): S_true [Jy] = peak [Jy/sr] x Omega_eff / PKCORR.
Content test (test_pkcorr): every node holds samples from >= min(MIN_ROUNDS, rounds) rounds, every node's median in REC_MIN..REC_MAX,
the map read back equals the curve within MAP_TOL.

Changes from MP (spec section 7 B1 and the v2 rules): every FITS read uses memmap=False (MP read the Lustre tiles memory-mapped); the
injected co-add must add every tile (the coadd unit's HG14 check); the test returns its verdict instead of raising (the unit records it
and removes the outputs of a failed test); NSIDE is the bound engine's.
"""
import json
import os
import shutil
import sys

import numpy as np


def _cfg():
    from exopipe.phase3.config import CFG
    return CFG.mosaic_pkcorr


def nodes():
    """(DECS, RA_BASE) of MP 33-34."""
    C = _cfg()
    decs = np.arange(float(C.DEC_START), float(C.DEC_STOP), float(C.DEC_STEP))
    ra = np.array([(i * float(C.RA_STEP)) % 360.0 for i in range(len(decs))])
    return decs, ra


def names(variant, band, tag, nside):
    """{key: file name} of MP 116-142 (tagv = <variant>_<band><tag>)."""
    tagv = f'{variant}_{band}{tag}'
    return {'map': f'healpix_PKCORR_I_deep_{variant}_{band}{tag}_nside{nside}.fits', 'recovery': f'peak_recovery_{tagv}.csv',
            'curve': f'peak_corr_curve_{tagv}.csv', 'json': f'pkcorr_{tagv}.json'}


def _gauss_add(data, hdr, x0, y0, peak):
    """MP 38-53: add an elliptical restoring-beam Gaussian (constant in pixel space)."""
    from astropy.wcs import WCS
    scale = abs(hdr['CDELT1']) if 'CDELT1' in hdr else abs(WCS(hdr).celestial.wcs.cdelt[0])
    fx = hdr['BMAJ'] / scale
    fn = hdr['BMIN'] / scale
    pa = np.radians(hdr.get('BPA', 0.0) + 90.0)   # BPA from north; grid y=north
    n = int(3 * fx) + 1
    yl, yh = max(0, int(y0) - n), min(data.shape[0], int(y0) + n + 1)
    xl, xh = max(0, int(x0) - n), min(data.shape[1], int(x0) + n + 1)
    if yl >= yh or xl >= xh:
        return False
    yy, xx = np.mgrid[yl:yh, xl:xh]
    dx, dy = xx - x0, yy - y0
    u = dx * np.cos(pa) + dy * np.sin(pa)
    v = -dx * np.sin(pa) + dy * np.cos(pa)
    g = peak * np.exp(-4 * np.log(2) * ((u / fx) ** 2 + (v / fn) ** 2))
    data[yl:yh, xl:xh] += g.astype(data.dtype)
    return True


def inject_round(root, band, variant, lsts, dst_root, ra_off):
    """MP 56-83: copies of the staged tiles with S_INJ Gaussians at (RA_BASE + ra_off, DECS); PB maps linked (weights only).
    memmap=False on every read."""
    from astropy.io import fits
    from astropy.wcs import WCS
    from exopipe.phase3.lib.mosaic import coadd_jysr as M
    C = _cfg()
    decs, ra_base = nodes()
    s_inj, pb_min = float(C.S_INJ), float(C.INJ_PB_MIN)
    n_added = np.zeros(len(decs), int)
    for lst in lsts:
        mp, pbp = M._paths(root, lst, variant, band)
        if not (os.path.exists(mp) and os.path.exists(pbp)):
            continue
        dd = os.path.join(dst_root, lst)
        os.makedirs(dd, exist_ok=True)
        lnk = os.path.join(dd, os.path.basename(pbp))
        if not os.path.exists(lnk):
            os.symlink(os.path.realpath(pbp), lnk)
        with fits.open(mp, memmap=False) as h:
            hdr = h[0].header
            data = h[0].data
        sq = np.squeeze(data)
        w = WCS(hdr).celestial
        pb = np.squeeze(fits.getdata(pbp, memmap=False))
        for i, dec in enumerate(decs):
            ra = (ra_base[i] + ra_off) % 360.0
            try:
                x, y = w.wcs_world2pix(ra, dec, 0)
                x, y = float(x), float(y)
            except Exception:
                continue
            if not (0 <= x < sq.shape[1] and 0 <= y < sq.shape[0]):
                continue
            if not np.isfinite(sq[int(y), int(x)]) or pb[int(y), int(x)] <= pb_min:
                continue
            if _gauss_add(sq, hdr, x, y, s_inj):
                n_added[i] += 1
        fits.writeto(os.path.join(dd, os.path.basename(mp)), data if data.ndim == 2 else sq.reshape(data.shape), hdr, overwrite=True)
    return n_added


def peak_recovery(diff, psf, ra, dec, nside):
    """MP 86-94: max within max(0.5 FWHM_eff, 0.05 deg) of the position; rec = peak x Omega_eff / S_INJ."""
    import healpy as hp
    C = _cfg()
    v = hp.ang2vec(ra, dec, lonlat=True)
    om = np.nanmedian(psf[hp.query_disc(nside, v, np.radians(float(C.OMEGA_RADIUS_DEG)))])
    if not np.isfinite(om):
        return np.nan
    fwhm = np.degrees(np.sqrt(om / 1.1331))
    disc = hp.query_disc(nside, v, np.radians(max(float(C.PEAK_RADIUS_FWHM) * fwhm, float(C.PEAK_RADIUS_MIN_DEG))))
    return np.nanmax(diff[disc]) * om / float(C.S_INJ)


class InjectionRefusal(RuntimeError):
    pass


def read_map(path):
    """MP 100: the clean map and the PSF map as production read them (hp.read_map, the file's float32; memmap=False)."""
    import healpy as hp
    from exopipe.phase3.lib.mosaic import coadd_jysr  # noqa: F401  (its import sets astropy's use_memmap off)
    return hp.read_map(path, verbose=False, memmap=False)


def _coadd_injected(M, lsts, variant, band, inj_root):
    """The injected co-add (MP 108) with the engine's per-tile verdicts captured: every tile must be added."""
    from exopipe.phase3.lib.mosaic.steps_worker import _VERDICT, _Tee
    tee = _Tee(sys.stdout)
    sys.stdout = tee
    try:
        mos_i, _ = M.coadd_jysr(lsts, variant, band, root=inj_root, domain='jysr')
    except SystemExit as e:
        raise InjectionRefusal(f'injected co-add: {e.code}')
    finally:
        sys.stdout = tee.real
    vd = {m.group(1): m.group(2) for m in (_VERDICT.match(x) for x in tee.lines) if m}
    bad = sorted((lst, vd.get(lst, 'absent')) for lst in lsts if vd.get(lst) != 'added')
    if bad:
        raise InjectionRefusal(f'injected co-add: tiles not added {bad}')
    return mos_i


def measure(root, band, variant, lsts, rounds, mos_c, psf, scratch, nside):
    """MP 103-128: the per-round recovery rows [(round, dec, ra, ntile, rec)] and the curve [(dec, median, mad, n, n_rounds)]."""
    from exopipe.phase3.lib.mosaic import coadd_jysr as M
    C = _cfg()
    decs, ra_base = nodes()
    roff = float(C.ROUND_OFF)
    tagv = f'{variant}_{band}'
    rows = []
    for r in range(rounds):
        inj_root = os.path.join(scratch, f'inj_{tagv}_r{r}')
        print(f'----- {tagv} round {r}: inject (RA offset {r * roff:.0f} deg) -----', flush=True)
        try:
            n_added = inject_round(root, band, variant, lsts, inj_root, r * roff)
            mos_i = _coadd_injected(M, lsts, variant, band, inj_root)
        finally:
            shutil.rmtree(inj_root, ignore_errors=True)
        diff = mos_i - mos_c
        for i, dec in enumerate(decs):
            if n_added[i]:
                ra = (ra_base[i] + r * roff) % 360.0
                rows.append((r, float(dec), ra, int(n_added[i]), float(peak_recovery(diff, psf, ra, dec, nside))))
        del mos_i, diff
    curve = []
    for dec in decs:
        rr = np.array([x[4] for x in rows if x[1] == dec and np.isfinite(x[4])])
        nr = len({x[0] for x in rows if x[1] == dec and np.isfinite(x[4])})
        if rr.size:
            curve.append((float(dec), float(np.median(rr)), float(1.4826 * np.median(np.abs(rr - np.median(rr)))), int(rr.size), nr))
    return rows, curve


def write(out, variant, band, tag, rounds, rows, curve, nside):
    """MP 116-142: the recovery and curve CSVs, the PKCORR map and the JSON (without the test); returns (paths, res)."""
    import healpy as hp
    from astropy.io import fits
    C = _cfg()
    nm = names(variant, band, tag, nside)
    P = {k: os.path.join(out, f) for k, f in nm.items()}
    with open(P['recovery'], 'w') as fh:
        fh.write('round,dec,ra,ntile,rec\n')
        for r_ in rows:
            fh.write(f'{r_[0]},{r_[1]:.1f},{r_[2]:.2f},{r_[3]},{r_[4]:.4f}\n')
    with open(P['curve'], 'w') as fh:
        fh.write('dec,rec_median,rec_mad,n,n_rounds\n')
        for c in curve:
            fh.write(f'{c[0]:.1f},{c[1]:.4f},{c[2]:.4f},{c[3]},{c[4]}\n')
    cd = np.array([c[0] for c in curve])
    cm = np.array([c[1] for c in curve])
    th, _ = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    corr = np.interp(90.0 - np.degrees(th), cd, cm).astype(np.float32)
    hdu = fits.BinTableHDU.from_columns([fits.Column(name='SIGNAL', format='E', array=corr)])
    for k, v in (('PIXTYPE', 'HEALPIX'), ('ORDERING', 'RING'), ('NSIDE', nside), ('COORDSYS', 'C'), ('INDXSCHM', 'IMPLICIT'),
                 ('BUNIT', ''), ('SINJ', float(C.S_INJ)), ('NROUNDS', rounds)):
        hdu.header[k] = v
    hdu.header.add_comment('PKCORR peak-dilution surface (injection-calibrated, dec-curve, RA-median).')
    hdu.header.add_comment('True flux [Jy] = peak[Jy/sr] * Omega_eff / PKCORR.')
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(P['map'], overwrite=True)
    res = dict(variant=variant, band=band, tag=tag, rounds=rounds, map=P['map'], rec_median=float(np.median(cm)),
               rec_min=float(cm.min()), rec_max=float(cm.max()), ra_scatter_median_mad=float(np.median([c[2] for c in curve])))
    return P, res


def test_pkcorr(path, curve, rounds, nside):
    """MP 148-161: every node from DEC_START holds samples from at least min(MIN_ROUNDS, rounds) rounds, every node's median is in
    REC_MIN..REC_MAX, and the map read back equals the curve at the pixel Decs (MAP_TOL). Returns the result, verdict pass / fail."""
    import healpy as hp
    from astropy.io import fits
    C = _cfg()
    decs, _ra = nodes()
    nd = {round(c[0], 1): c for c in curve}
    need = min(int(C.MIN_ROUNDS), rounds)
    missing = [float(d) for d in decs if round(float(d), 1) not in nd or nd[round(float(d), 1)][4] < need]
    out_of_range = [(c[0], c[1]) for c in curve if not float(C.REC_MIN) <= c[1] <= float(C.REC_MAX)]
    with fits.open(path, memmap=False) as h:
        m = np.asarray(h[1].data['SIGNAL'], dtype=np.float64).ravel()
    err = float('inf')
    if curve:
        th, _ = hp.pix2ang(nside, np.arange(len(m)))
        ref = np.interp(90.0 - np.degrees(th), [c[0] for c in curve], [c[1] for c in curve])
        err = float(np.max(np.abs(m - ref)))
    res = dict(nodes=len(curve), missing_or_few_rounds=missing, out_of_range=out_of_range, max_map_err=err)
    res['verdict'] = 'fail' if (missing or out_of_range or not err < float(C.MAP_TOL)) else 'pass'
    return res


def run_band(root, band, variant, out, lsts, rounds, tag, mos_c, psf, scratch, nside, record_dir=None):
    """MP run_band 97-145 with the clean map and PSF map given (the unit reads them): writes the four files into out, returns
    (paths, res) with res['test_pkcorr']; res['map'] names the map in record_dir when the caller moves the files there."""
    rows, curve = measure(root, band, variant, lsts, rounds, mos_c, psf, scratch, nside)
    if not curve:
        raise InjectionRefusal(f'{variant} {band}: no Dec node recovered a source in {rounds} rounds ({len(rows)} rows)')
    P, res = write(out, variant, band, tag, rounds, rows, curve, nside)
    res['test_pkcorr'] = test_pkcorr(P['map'], curve, rounds, nside)
    if record_dir:
        res['map'] = os.path.join(record_dir, os.path.basename(P['map']))
    with open(P['json'], 'w') as f:
        json.dump(res, f, indent=1)
    tv = f'{variant}_{band}{tag}'
    print(f'== {tv} VERDICT: rec median={res["rec_median"]:.3f} range=[{res["rec_min"]:.3f},{res["rec_max"]:.3f}] '
          f'RA-scatter(medMAD)={res["ra_scatter_median_mad"]:.3f} ==', flush=True)
    return P, res
