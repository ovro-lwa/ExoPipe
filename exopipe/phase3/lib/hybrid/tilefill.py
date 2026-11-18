"""Stage `tilefill` of the hybrid route (spec plan/specs/hybrid.md 2.14; design choices D7, D8, D9; gate HG13): the band fill field
of a filled hybrid mosaic and the per-tile zero-spacing fill of the hybrid masters. A close port of production
exopipe/phase3/mosaic_tpfill.py (MF) at exopipe main 4c7098c: _read 37, donor_mhz 42, lfss_jysr 46, apod 56, fill_correction 61,
_md5 202, band_increment 211, tile_increment 242, _identity 255, _compact 262, _roundtrip 289, fill_master_unit 306, rename_cleanres
374; and of the ruled caveat of the fill driver exopipe_ops/hybrid_master_lsfill_20261111/run_hybrid_master_lsfill_20261111.py
(HMF 41-44, job_retest 124-143). The tile conversion (omega_zen, n_map, _common_wcs_tile, GRID_N) and get_uniform_rms are v2's
copies (lib/mosaic/coadd_jysr.py, lib/masters/coadd.py), equal to production's line for line.

THE FILL (Gregg 2026-11-11 "Approved", per tile): the band's fill field (the full-sphere correction field of the filled mosaic, l < 12)
is sampled at each master pixel's sky position (bilinear) and converted to the beam-corrected domain (x Omega_zen / n, the inverse of
the mosaic's tile conversion): beam-corrected master += dP; plain master += dP x the diffuse PB. Blanking is unchanged; a finite pixel
off the sky is unchanged (TPOFFSKY counts them).

Differences from production (none changes a pixel):
- constants from the config through bind() (section hybrid_tilefill; the LFSS directory is hybrid_recon.LFSS_DIR + '/healpix');
- every FITS file is read with memmap=False (HG23), and healpy reads with memmap=False;
- a refusal raises TilefillRefusal (production RuntimeError '... - REFUSING');
- output paths are given by the caller (production wrote <out_dir>/<the source's basename>; the v2 work files carry work names);
- the ruled caveat (card TPCAVEAT, hybrid_tilefill.TPCAVEAT_UNITS) is decided BEFORE the write, from the compact check on the
  float32 pixels that are written, and is written with the other cards; production re-tested the staged file and set the card with
  fits.setval (HMF job_retest). The tests then run on the written file as in production, and the record says whether the caveat
  was used;
- D8: TPMOSAIC holds the filled mosaic's full name (CONTINUE cards), not its first 68 characters; so does the band field's TPFILLED;
- D9: GITTAG = this code's version (the writer's); a HISTORY line keeps the GITTAG of the master's build;
- D7: the Release 1 header-repair cards (hybrid_tilefill.REPAIR_KEYS) are carried from the installed Taper master when the unfilled
  master lacks them (the quarantined unfilled masters of 2026-11-11 predate the repair); a card present in both must agree. The
  content test card_superset (lib/hybrid/tests_tilefill.py) checks that the written file holds every card key of the installed file it will
  replace.
"""
import hashlib
import os
import time

import numpy as np
from astropy.io import fits

from exopipe.phase3.core.products import build_name, parse_name
from exopipe.phase3.lib.install.steps_common import md5


class _Healpy:
    """healpy, imported at its first use (the data-free CI has no healpy; recon.py imports it inside each function)."""
    def __getattr__(self, k):
        import healpy
        return getattr(healpy, k)


hp = _Healpy()

READS = set()                                      # every file the stage read; the step records those that are not its own outputs


class TilefillRefusal(RuntimeError):
    """A gate or content test of the tile fill refused (HG13; production raised RuntimeError '... - REFUSING')."""


def bind():
    """Bind the module's constants from the config (production MF 25-29, 192-199; HMF 41-44). Call before any computation."""
    from exopipe.phase3.config import CFG
    T = CFG.hybrid_tilefill
    g = dict(LFSS_DIR=os.path.join(CFG.hybrid_recon.LFSS_DIR, 'healpix'), LFSS_MHZ=tuple(int(f) for f in T.LFSS_MHZ), LMAX=int(T.LMAX),
             APOD=float(T.APOD_DEG), L_LO=int(T.L_LO), L_HI=int(T.L_HI), TILE_RT_NSIDE=int(T.TILE_RT_NSIDE),
             TILE_RT_MAX=float(T.TILE_RT_MAX), COMPACT_K=int(T.COMPACT_K), COMPACT_MED_MAX=float(T.COMPACT_MED_MAX),
             COMPACT_MAX=float(T.COMPACT_MAX), COMPACT_SIG=float(T.COMPACT_SIG),
             TPCAVEAT_UNITS=frozenset(tuple(x) for x in T.TPCAVEAT_UNITS), TPCAVEAT_TEXT=str(T.TPCAVEAT_TEXT),
             REPAIR_KEYS=tuple(T.REPAIR_KEYS))
    globals().update(g)
    return g


def _getheader(path, ext=0):
    READS.add(path)
    with fits.open(path, memmap=False) as h:
        return h[ext].header.copy()


def _read(p):
    """A HEALPix map with UNSEEN and non-finite pixels as NaN (MF 37)."""
    READS.add(p)
    kw = {'verbose': False} if 'verbose' in hp.read_map.__code__.co_varnames else {}
    m = hp.read_map(p, memmap=False, **kw)
    return np.where(np.isfinite(m) & (m != hp.UNSEEN), m, np.nan)


def donor_mhz(freq_hz):
    return min(LFSS_MHZ, key=lambda f: abs(f - freq_hz / 1e6))


def lfss_jysr(fmhz, nside):
    """The LFSS map at fmhz in Jy/sr at nside; refuses unless Cygnus A is bright at its equatorial position (MF 46)."""
    m = _read(f'{LFSS_DIR}/healpix-all-sky-rav-rsclean-map-{fmhz}.fits')
    eq = m[hp.ang2pix(hp.get_nside(m), 299.868, 40.734, lonlat=True)]
    if not eq > 2 * np.nanmedian(m[m > 0]):
        raise TilefillRefusal(f'LFSS {fmhz}: Cyg A not bright at its equatorial position - frame assumption wrong; REFUSING')
    K_B, C = 1.380649e-23, 2.99792458e8
    lam = C / (fmhz * 1e6)
    j = np.where(np.isfinite(m) & (m > 0), m * 2 * K_B / lam ** 2 * 1e26, 0.0)
    return hp.ud_grade(j, nside)


def apod(mask):
    a = hp.smoothing(mask.astype(float), fwhm=np.radians(APOD), iter=1); a[~mask] = 0.0
    return a


def fill_correction(t, fmhz):
    """The fill's correction field on the FULL sphere (band-limited, l < L_HI): inside the footprint it is what the mosaic fill adds;
    outside it is the same field's smooth continuation (MF 61)."""
    nside = hp.get_nside(t); mask = np.isfinite(t); am = apod(mask); fsky = np.mean(am ** 2)
    alm_t = hp.map2alm(np.where(mask, t, 0.0) * am, lmax=LMAX, iter=1)
    L = lfss_jysr(fmhz, nside); alm_L = hp.map2alm(np.where(mask, L, 0.0) * am, lmax=LMAX, iter=1)
    cl_t, cl_L = hp.alm2cl(alm_t) / fsky, hp.alm2cl(alm_L) / fsky
    s = float(np.sqrt(np.mean(cl_t[10:30] / cl_L[10:30])))
    ell = np.arange(LMAX + 1)
    w = np.where(ell <= L_LO, 1.0, np.where(ell >= L_HI, 0.0, 0.5 * (1 + np.cos(np.pi * (ell - L_LO) / (L_HI - L_LO)))))
    corr = hp.alm2map(hp.almxfl(alm_L, w * s) - hp.almxfl(alm_t, w), nside)
    return corr, s, L, mask, am


def band_increment(js_unfilled, js_filled, out):
    """The band's fill field on the full sphere -> out (float32 HEALPix, SIGNAL, Jy/sr) (MF 211). Refuses unless the donor is the
    unfilled map's nearest LFSS map, s equals the filled map's TPSCALE (5 digits), the footprints agree, and, inside the footprint,
    unfilled + field reproduces the filled map to float32 storage precision (4 rounding units + 1e-6 of the field's peak)."""
    if os.path.exists(out):
        raise TilefillRefusal(f'{out} exists - REFUSING')
    hf = _getheader(js_filled, 1); fmhz = int(hf['TPDONMHZ'])
    if fmhz != donor_mhz(float(_getheader(js_unfilled, 1)['FREQ'])):
        raise TilefillRefusal(f'{js_filled}: donor {fmhz} MHz is not the unfilled map\'s nearest LFSS map - REFUSING')
    t, f = _read(js_unfilled), _read(js_filled)
    corr, s, L, mask, am = fill_correction(t, fmhz)
    if round(s, 5) != float(hf['TPSCALE']):
        raise TilefillRefusal(f'{js_filled}: s {s:.6f} is not its TPSCALE {hf["TPSCALE"]} - REFUSING')
    if not np.array_equal(mask, np.isfinite(f)) or not np.all(np.isfinite(corr)):
        raise TilefillRefusal(f'{js_filled}: footprint differs from the unfilled map\'s, or the field is not finite everywhere - REFUSING')
    g = (t + corr).astype(np.float32)
    err = np.abs(g[mask].astype(np.float64) - f[mask]); allow = 4 * 2.0 ** -24 * np.abs(f[mask]) + 1e-6 * np.max(np.abs(corr[mask]))
    ident = float(np.max(err / allow))
    if not ident <= 1.0:
        raise TilefillRefusal(f'{js_filled}: unfilled + field does not reproduce the filled map (identity {ident:.3g}) - REFUSING')
    hdu = fits.BinTableHDU.from_columns([fits.Column(name='SIGNAL', format='E', array=corr.astype(np.float32))])
    for k, v in (('PIXTYPE', 'HEALPIX'), ('ORDERING', 'RING'), ('NSIDE', hp.get_nside(t)), ('COORDSYS', 'C'), ('INDXSCHM', 'IMPLICIT'),
                 ('BUNIT', 'Jy/sr'), ('TPDONMHZ', fmhz), ('TPSCALE', round(s, 5))):
        hdu.header[k] = v
    fmd5, umd5 = md5(js_filled), md5(js_unfilled)
    hdu.header['TPFILLED'] = (os.path.basename(js_filled), 'the archived filled map (D8: full name)')
    hdu.header['TPFMD5'] = (fmd5, 'md5 of the archived filled map')
    hdu.header['TPUMD5'] = (umd5, 'md5 of the unfilled map')
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(out + '.part', overwrite=True)
    os.replace(out + '.part', out)
    return dict(out=out, out_md5=md5(out), filled=js_filled, filled_md5=fmd5, unfilled=js_unfilled, unfilled_md5=umd5, donor_mhz=fmhz,
                s=s, identity_rel=ident, footprint_frac=float(mask.mean()), result='PASS')


def tile_increment(h, corr):
    """(dP, wcs): the field at each pixel's sky position (bilinear) x Omega_zen / n on the grid of header h, in Jy/beam of the
    beam-corrected domain (the inverse of the mosaic's tile conversion, native mode); NaN off the sky (MF 242)."""
    from astropy.wcs import WCS
    from exopipe.phase3.lib.mosaic import coadd_jysr as mcj
    w = WCS(h).celestial; ny, nx = int(h['NAXIS2']), int(h['NAXIS1'])
    yy, xx = np.mgrid[0:ny, 0:nx]
    ra, dec = w.all_pix2world(xx.ravel(), yy.ravel(), 0)
    v = np.full(ra.shape, np.nan); ok = np.isfinite(ra) & np.isfinite(dec)
    v[ok] = hp.get_interp_val(corr, ra[ok], dec[ok], lonlat=True)
    return v.reshape(ny, nx) * mcj.omega_zen(h) / mcj.n_map(w, (ny, nx)), w


def _identity(new, old, inc):
    """max over the pixels where all three are finite of |(new - old) - inc| / (4 float32 rounding units of |new| + |old| + 1e-6 of
    the increment's peak) (MF 255); <= 1 passes."""
    m = np.isfinite(new) & np.isfinite(old) & np.isfinite(inc)
    err = np.abs((new - old)[m] - inc[m])
    allow = 4 * 2.0 ** -24 * (np.abs(new[m]) + np.abs(old[m])) + 1e-6 * max(float(np.max(np.abs(inc[m]))), 1e-30)
    return float(np.max(err / allow))


def _compact(new, old, k=None, r_in=3, r_out=6):
    """The k brightest SIGNIFICANT local maxima of the unfilled master (contrast >= COMPACT_SIG x its robust rms):
    |(peak - annulus median) new - old| / (peak - annulus median) old (MF 262; Gregg 2026-11-11 "Significance + caveat")."""
    from scipy.ndimage import maximum_filter
    k = COMPACT_K if k is None else k
    o = np.where(np.isfinite(old), old, -np.inf); pk = (maximum_filter(o, size=2 * r_out + 1) == o) & np.isfinite(old)
    yy, xx = np.nonzero(pk); order = np.argsort(old[yy, xx])[::-1]
    yx = np.mgrid[-r_out:r_out + 1, -r_out:r_out + 1]; rr = np.hypot(*yx); ann = (rr >= r_in) & (rr <= r_out)
    f = old[np.isfinite(old)]; rob = 1.4826 * float(np.median(np.abs(f - np.median(f))))
    rel = []
    for i in order:
        y, x = yy[i], xx[i]
        if y < r_out or x < r_out or y + r_out >= old.shape[0] or x + r_out >= old.shape[1]:
            continue
        so, sn = old[y - r_out:y + r_out + 1, x - r_out:x + r_out + 1], new[y - r_out:y + r_out + 1, x - r_out:x + r_out + 1]
        if not (np.all(np.isfinite(so[ann])) and np.all(np.isfinite(sn[ann]))):
            continue
        co, cn = old[y, x] - np.median(so[ann]), new[y, x] - np.median(sn[ann])
        if co > 0 and co >= COMPACT_SIG * rob:
            rel.append(abs(cn - co) / co)
        if len(rel) >= k:
            break
    rel = np.array(rel)
    return dict(n=len(rel), sig_min=COMPACT_SIG, robust_rms=rob, median=float(np.median(rel)) if len(rel) else None,
                max=float(np.max(rel)) if len(rel) else None,
                ok=bool(len(rel) == 0 or (np.median(rel) <= COMPACT_MED_MAX and np.max(rel) <= COMPACT_MAX)))


def _roundtrip(added, w, h, lst, corr):
    """The written beam-corrected increment read back through the mosaic's own tile conversion (x n / Omega_zen, bilinear onto the
    LST's common grid, bilinear to HEALPix at NSIDE TILE_RT_NSIDE) against the field there, over HEALPix pixels wholly inside the
    tile (MF 289)."""
    from reproject import reproject_interp, reproject_to_healpix
    from exopipe.phase3.lib.mosaic import coadd_jysr as mcj
    cw = mcj._common_wcs_tile(lst); fin = np.isfinite(added)
    out = []
    for a in (np.where(fin, added, 0.0) * mcj.n_map(w, added.shape) / mcj.omega_zen(h), fin.astype(float)):
        arr, _ = reproject_interp((a, w), cw, shape_out=(mcj.GRID_N, mcj.GRID_N), order='bilinear')
        hpx, _ = reproject_to_healpix((np.nan_to_num(arr), cw), 'icrs', nside=TILE_RT_NSIDE, order='bilinear')
        out.append(np.nan_to_num(hpx))
    back, inside = out[0], out[1] > 0.999
    c = hp.ud_grade(corr, TILE_RT_NSIDE)
    rel = float(np.median(np.abs(back[inside] - c[inside])) / np.median(np.abs(c[inside])))
    return dict(n_pix=int(inside.sum()), rel=rel, ok=bool(inside.sum() > 100 and rel <= TILE_RT_MAX))


def _header_ok(path):
    from exopipe.common.validate_fits_headers import validate_header
    rep = validate_header(path, level='standard'); ok = rep.ok() if callable(rep.ok) else rep.ok
    return bool(ok), [str(e) for e in getattr(rep, 'errors', [])][:5]


def repair_cards(h, taper_master):
    """D7: the Release 1 header-repair cards of the installed Taper master that header h lacks, as [(key, value, comment)]; refuses
    when a card present in both disagrees (the unfilled master would then not be this master's hybrid)."""
    if not taper_master:
        return []
    th = _getheader(taper_master)
    add, bad = [], []
    for k in REPAIR_KEYS:
        if k not in th:
            continue
        if k in h:
            if h[k] != th[k]:
                bad.append(f'{k} {h[k]!r} vs {th[k]!r}')
        else:
            add.append((k, th[k], th.comments[k]))
    if bad:
        raise TilefillRefusal(f'repair card(s) differ from the installed Taper master {os.path.basename(taper_master)}: {bad} - REFUSING')
    return add


def _carry(h, add, taper_master):
    for k, v, c in add:
        h[k] = (v, c)
    if add:
        h['HISTORY'] = (f'Release 1 header-repair cards {" ".join(k for k, _v, _c in add)} carried from the installed Taper master '
                        f'{os.path.basename(taper_master)} (spec D7)')


def fill_master_unit(plain, pbcorr, pb, pb_how, field, outs, lst, scope, w, taper_master=None):
    """Write the filled plain and beam-corrected masters of one hybrid unit to outs = {'plain': path, 'pbcorr': path} (their 'lsfill'
    token now holds) and test the WRITTEN files: identity (written - unfilled = the increment at every pixel, float32 precision),
    blanking unchanged, the round trip through the mosaic's tile conversion (beam-corrected), bright compact sources unchanged (or the
    ruled caveat), header valid (MF 306; HMF 124-143). pb = the diffuse PB array on the master grid (float64 of the float32 map),
    pb_how = its source for the record. Returns the record; result PASS or FAIL."""
    from exopipe.common import phase2_config as pcfg
    from exopipe.phase3.lib.masters.coadd import get_uniform_rms
    from exopipe.phase3.lib.mosaic import coadd_jysr as mcj
    hi = _getheader(field, 1); corr = _read(field)
    if not np.all(np.isfinite(corr)):
        raise TilefillRefusal(f'{field}: the field is not finite everywhere - REFUSING')
    hP, hH = _getheader(pbcorr), _getheader(plain)
    for p, h, kind in ((plain, hH, 'plain'), (pbcorr, hP, 'pbcorr')):
        if h.get('TPFILL'):
            raise TilefillRefusal(f'{p} already carries TPFILL - REFUSING')
        pn = parse_name(str(h.get('PRODNAME', '')))
        proc = set(pn.get('proc') or ())
        if 'lsfill' not in proc or 'cleanres' in proc or ('pbcorr' in proc) != (kind == 'pbcorr'):
            raise TilefillRefusal(f'{p}: PRODNAME {h.get("PRODNAME")!r} is not a {kind} hybrid master (lsfill, no cleanres) - REFUSING')
        if os.path.basename(outs[kind]) != h['PRODNAME']:
            raise TilefillRefusal(f'{outs[kind]}: the output name is not the master\'s PRODNAME {h["PRODNAME"]} - REFUSING')
        if pn.get('lst') != lst or pn.get('scope') != scope:
            raise TilefillRefusal(f'{p}: PRODNAME {h["PRODNAME"]} is not of {lst} {scope} - REFUSING')
    for k in ('NAXIS1', 'NAXIS2', 'CTYPE1', 'CTYPE2', 'CRVAL1', 'CRVAL2', 'CRPIX1', 'CRPIX2', 'CDELT1', 'CDELT2', 'BMAJ', 'BMIN'):
        if hP.get(k) != hH.get(k):
            raise TilefillRefusal(f'{plain} / {pbcorr}: {k} differs - REFUSING')
    dP, wcs = tile_increment(hP, corr)
    if pb.shape != dP.shape:
        raise TilefillRefusal(f'{pb_how}: shape {pb.shape} is not the master grid {dP.shape} - REFUSING')
    mosaic = str(hi['TPFILLED'])
    rec = dict(plain=plain, pbcorr=pbcorr, pbdiffuse=pb_how, increment=field, increment_md5=md5(field), lst=lst, scope=scope, weighting=w,
               omega_zen_sr=mcj.omega_zen(hP), taper_master=taper_master, gittag=pcfg.PIPELINE_GITTAG)
    ok_all = True
    for kind, src, inc in (('pbcorr', pbcorr, dP), ('plain', plain, dP * pb)):
        out = outs[kind]
        if os.path.exists(out):
            raise TilefillRefusal(f'{out} exists - REFUSING')
        smd5 = md5(src); READS.add(src)
        with fits.open(src, memmap=False) as hl:
            shp = hl[0].data.shape; old = np.squeeze(hl[0].data).astype(np.float64)
            fin = np.isfinite(old); offsky = int((fin & ~np.isfinite(inc)).sum())
            new = np.where(fin & np.isfinite(inc), old + inc, old)
            n32 = new.astype(np.float32)
            caveat_unit = (lst, scope, w, kind) in TPCAVEAT_UNITS
            pre = _compact(n32.astype(np.float64), old) if caveat_unit else None
            caveat = bool(caveat_unit and not pre['ok'])
            hl[0].data = n32.reshape(shp)
            h = hl[0].header
            add = repair_cards(h, taper_master)
            build = h.get('GITTAG')
            for k, v in (('TPFILL', (True, 'zero-spacing fill: LWA1 LFSS replaces l<8, blend to 12')),
                         ('TPDONOR', ('LWA1 LFSS (Dowell+2017), LEDA zero level', 'donor')), ('TPDONMHZ', (int(hi['TPDONMHZ']), 'donor map frequency [MHz]')),
                         ('TPSCALE', (float(hi['TPSCALE']), 's = sqrt(Cl map / Cl LFSS), l 10-30')), ('TPLLO', (L_LO, 'l below: survey')),
                         ('TPLHI', (L_HI, 'l above: map')), ('TPMOSAIC', (mosaic, 'filled mosaic: the field (D8: full name)')),
                         ('TPMOSMD5', (hi['TPFMD5'], 'md5 of the filled mosaic')), ('TPSRCMD5', (smd5, 'md5 of the unfilled master')),
                         ('TPTILE', ('dS x Omega_zen / n' + (' x PBdiffuse' if kind == 'plain' else ''), 'per-tile fill (Gregg 2026-11-11)')),
                         ('TPOFFSKY', (offsky, 'finite pixels off the sky: unchanged'))):
                h[k] = v
            if caveat:
                h['TPCAVEAT'] = (TPCAVEAT_TEXT[:68], 'ruled caveat')
            _carry(h, add, taper_master)
            h['GITTAG'] = (pcfg.PIPELINE_GITTAG, h.comments['GITTAG'] if 'GITTAG' in h else 'Pipeline code version')
            if kind == 'plain' and 'IMGRMS' in h:
                old_rms = h['IMGRMS']; n2 = n32.astype(np.float64)
                h['IMGRMS'] = (get_uniform_rms(n2, ~np.isfinite(n2) | (n2 == 0)), h.comments['IMGRMS'])
                h['HISTORY'] = f'IMGRMS before the zero-spacing fill: {old_rms}'
            h['HISTORY'] = (f'lsfill (zero-spacing) {time.strftime("%Y-%m-%d", time.gmtime())}: the band fill field (LFSS {hi["TPDONMHZ"]} MHz x '
                            f'{hi["TPSCALE"]}, l<8 blend to 12) x Omega_zen / n' + (' x PBdiffuse' if kind == 'plain' else '')
                            + ' added (Gregg 2026-11-11 "Approved")')
            h['HISTORY'] = f'zero-spacing fill written by exopipe {pcfg.PIPELINE_GITTAG}; the hybrid master was built by {build} (spec D9)'
            hl.writeto(out + '.part', overwrite=True, output_verify='exception')
        os.replace(out + '.part', out)
        hok, herr = _header_ok(out)
        with fits.open(out, memmap=False) as hl:
            wr = np.squeeze(hl[0].data).astype(np.float64)
        t = dict(out=out, out_md5=md5(out), src_md5=smd5, offsky_finite=offsky, header_valid=hok, header_errors=herr,
                 blanking_unchanged=bool(np.array_equal(np.isfinite(wr), fin)), identity_rel=_identity(wr, old, inc), compact=_compact(wr, old),
                 repair_cards_carried=[k for k, _v, _c in add])
        if kind == 'pbcorr':
            t['roundtrip'] = _roundtrip(wr - old, wcs, hP, lst, corr)
        t['caveat'] = TPCAVEAT_TEXT if caveat else None
        others = t['header_valid'] and t['blanking_unchanged'] and t['identity_rel'] <= 1.0 and t.get('roundtrip', {'ok': True})['ok']
        consistent = caveat == bool(caveat_unit and not t['compact']['ok'])       # the decision before the write holds on the file
        t['result'] = 'PASS' if (others and consistent and (t['compact']['ok'] or caveat)) else 'FAIL'
        ok_all &= t['result'] == 'PASS'; rec[kind] = t
    rec['result'] = 'PASS' if ok_all else 'FAIL'
    return rec


def _data_md5(hdu):
    return hashlib.md5(np.ascontiguousarray(hdu.data).tobytes()).hexdigest()


def rename_cleanres(src, out, taper_master=None):
    """The hybrid CLEAN residual without the 'lsfill' token (it gets no fill), written to out (its new name) with PRODNAME updated and
    a HISTORY line (MF 374); the repair cards carried (D7). Test: data section identical (md5 of the array), PRODNAME = file name and
    parses, header validates."""
    h0 = _getheader(src)
    pn = parse_name(str(h0.get('PRODNAME', '')))
    if not {'lsfill', 'cleanres'} <= set(pn.get('proc') or ()):
        raise TilefillRefusal(f'{src}: PRODNAME {h0.get("PRODNAME")!r} is not an lsfill CLEAN residual - REFUSING')
    pn['proc'] = [p for p in pn['proc'] if p != 'lsfill']; pn.pop('grammar', None); name = build_name(**pn)
    if os.path.basename(out) != name:
        raise TilefillRefusal(f'{out}: not the renamed CLEAN residual {name} - REFUSING')
    if os.path.exists(out):
        raise TilefillRefusal(f'{out} exists - REFUSING')
    READS.add(src)
    with fits.open(src, memmap=False) as hl:
        d0 = _data_md5(hl[0])
        add = repair_cards(hl[0].header, taper_master)
        hl[0].header['PRODNAME'] = (name, hl[0].header.comments['PRODNAME'] if 'PRODNAME' in hl[0].header else 'Phase 3 product name')
        _carry(hl[0].header, add, taper_master)
        hl[0].header['HISTORY'] = (f"renamed {time.strftime('%Y-%m-%d', time.gmtime())}: 'lsfill' token dropped - the CLEAN residual gets no "
                                   'zero-spacing fill (Gregg 2026-11-11 "Approved"); data unchanged')
        hl.writeto(out + '.part', overwrite=True, output_verify='exception')
    os.replace(out + '.part', out)
    with fits.open(out, memmap=False) as hl:
        d1 = _data_md5(hl[0]); pr = hl[0].header.get('PRODNAME')
    hok, herr = _header_ok(out)
    rt = parse_name(pr) if pr else {}
    rt.pop('grammar', None)
    ok = d0 == d1 and pr == name and bool(rt) and build_name(**rt) == name and hok
    return dict(src=src, out=out, name=name, out_md5=md5(out), data_md5=d1, data_identical=d0 == d1, prodname_ok=pr == name,
                header_valid=hok, header_errors=herr, repair_cards_carried=[k for k, _v, _c in add], result='PASS' if ok else 'FAIL')
