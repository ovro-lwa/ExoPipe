"""Stage `caveat` of mosaic v5 (spec plan/specs/hybrid.md 2.13): the caveat map of one high-band hybrid mosaic (Gregg 2026-10-31, "ok
on caveats. Go on everything", R-GREGG-20261031-HIGH-BAND-CAVEATS-APPROVED-GO-ON-EVERYTHING). A close port of production
exopipe/phase3/notaper_largescale.py (NL) at exopipe main 4c7098c: caveat_map 1759, test_caveat_map 1835.

Per pixel (NSIDE mosaic_caveat.NSIDE, RING, equatorial; NaN south of the mosaic's Dec floor and where no tile has weight), with the
mosaic's own tile weights (lib/mosaic/coadd_jysr: PB^2 / sigma^2, sigma = 1.4826 x MAD of the apparent tile where PB > PB floor,
cosine-feathered over PB 0.15 -> 0.10, bilinear to HEALPix; the same tile guards):
  LEVEL     the weighted mean over the hybrid tiles of the tile's large-scale level / survey prior in the pixel's Dec band (cards
            LSLVL1-5 of the hybrid master; hybrid_recon.DEC_BANDS); NaN south of -30 deg and where no hybrid tile has weight;
  TAPERONLY the weight fraction of Taper-only tiles, 0-1. The v5 stage admits none (HG14), so it is 0 wherever the map is defined;
            the column stays for the archived format.

Differences from production (none changes a value):
- the tiles come from the stage record (path, weighting map, kind) instead of a staged link tree, and a tile that is not a hybrid
  refuses (production took a ruled Taper-only unit as its plain Taper master; v2 has no ruled Taper-only unit);
- the test record is returned (the unit records it) instead of being written as caveat_<v>_<scope>_lsfill.json;
- the content test's failure raises CaveatRefusal; every FITS file is read with memmap=False (HG23);
- the map and the table are installed together (core/placement.atomic_many: the replaced files' group, mode and ACL, or the
  directory's rule for new ones; an attribute refusal places neither, a failed rename of the table leaves the map placed - the caveat
  unit then removes both; a placement refusal is the unit's refusal), where production renamed each.
"""
import csv
import os

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from exopipe.phase3.core import placement


class CaveatRefusal(RuntimeError):
    """A gate or the content test of the caveat map refused."""


def _cfg():
    from exopipe.phase3.config import CFG
    return CFG


def names(variant, scope, nside):
    """(map name, per-tile table name) as production wrote them (NL 1813, 1823)."""
    stem = f'I_deep_{variant}_{scope}_lsfill'
    return f'healpix_CAVEAT_{stem}_nside{nside}.fits', f'caveats_{variant}_{scope}_lsfill.csv'


def _getdata(p):
    with fits.open(p, memmap=False) as h:
        return h[0].data.copy(), h[0].header.copy()


def caveat_map(tiles, variant, scope, out_dir, pbeff_mosaic, nside=None):
    """tiles: [{'lst', 'path' (the staged beam-corrected hybrid master), 'pb' (its weighting map), 'kind' ('hybrid')}] in LST order.
    Writes the caveat map and the per-tile table into out_dir; returns dict(map, table, n_hybrid, n_taper_only, taper_only, test)."""
    import healpy as hp
    from reproject import reproject_to_healpix
    from exopipe.phase3.lib.mosaic import coadd_jysr as MCJ
    C = _cfg()
    nside = int(nside or C.mosaic_caveat.NSIDE)
    dec_bands = tuple(tuple(b) for b in C.hybrid_recon.DEC_BANDS)
    npix = hp.nside2npix(nside)
    th, _ = hp.pix2ang(nside, np.arange(npix))
    dec = 90.0 - np.degrees(th)
    bidx = np.full(npix, -1)
    for i, (lo, hi) in enumerate(dec_bands):
        bidx[(dec >= lo) & (dec < hi)] = i
    den, wpb, num_l, den_l, num_t = (np.zeros(npix) for _ in range(5))
    lmin, lmax = np.full(npix, np.inf), np.full(npix, -np.inf)
    rows = []
    for t in tiles:
        lst, mp, pbp = t['lst'], t['path'], t['pb']
        if not (os.path.exists(mp) and os.path.exists(pbp)):
            raise CaveatRefusal(f'{lst} {scope} {variant}: staged tile missing - REFUSING')
        pbd, pbh = _getdata(pbp)
        pb = np.nan_to_num(pbd.squeeze().astype(np.float32))
        wcs_ = WCS(pbh).celestial
        md, h = _getdata(mp)
        app = (np.nan_to_num(md.squeeze().astype(np.float32)) * pb)[pb > MCJ.PBCORR_FLOOR]
        app = app[np.isfinite(app)]
        if app.size < 100:                         # the mosaic's guards: such a tile does not enter it
            rows.append(dict(lst=lst, kind='not in mosaic (no valid pixels)'))
            continue
        sigma = 1.4826 * np.median(np.abs(app - np.median(app))) or 1e-6
        if not np.isfinite(sigma) or sigma < 1e-4:
            rows.append(dict(lst=lst, kind=f'not in mosaic (flat, sigma {sigma:.2e})'))
            continue
        wmap = np.where(pb > MCJ.PBCORR_FLOOR, pb ** 2 / sigma ** 2, 0.0).astype(np.float32)
        ramp = np.clip((pb - MCJ.PBCORR_FLOOR) / (MCJ.PB_FEATHER - MCJ.PBCORR_FLOOR), 0.0, 1.0)
        wmap *= (0.5 - 0.5 * np.cos(np.pi * ramp)).astype(np.float32)
        w = np.nan_to_num(reproject_to_healpix((wmap, wcs_), 'icrs', nside=nside, order='bilinear')[0])
        den += w
        wpb += np.nan_to_num(reproject_to_healpix((wmap * pb, wcs_), 'icrs', nside=nside, order='bilinear')[0])
        hyb = t.get('kind') == 'hybrid'
        lv = [h.get(f'LSLVL{i}') for i in range(1, len(dec_bands) + 1)]
        rows.append(dict(lst=lst, kind='hybrid' if hyb else 'Taper-only', staged=os.path.realpath(mp), sigma_apparent=float(sigma),
                         LSCAVEAT=h.get('LSCAVEAT'), LSSMALL=h.get('LSSMALL'), LSSTKADM=h.get('LSSTKADM'), LSRAWNT=h.get('LSRAWNT'),
                         **{f'LSLVL{i + 1} ({lo:+d}..{hi:+d})': lv[i] for i, (lo, hi) in enumerate(dec_bands)},
                         taper_only_reason='' if hyb else t.get('reason', '')))
        if not hyb:
            num_t += w
            continue
        if h.get('LSCAVEAT') is None or all(x is None for x in lv):
            raise CaveatRefusal(f'{mp}: a high-band hybrid without its caveat cards (LSCAVEAT, LSLVL1-5) - REFUSING')
        for i, x in enumerate(lv):
            if x is None:
                continue
            k = (bidx == i) & (w > 0)
            num_l[k] += w[k] * float(x)
            den_l[k] += w[k]
            lmin[k] = np.minimum(lmin[k], float(x))
            lmax[k] = np.maximum(lmax[k], float(x))
    good = (den > 0) & (dec >= MCJ.DEC_FLOOR)
    level = np.full(npix, np.nan)
    taper = np.full(npix, np.nan)
    pbeff = np.full(npix, np.nan)
    gl = good & (den_l > 0)
    level[gl] = num_l[gl] / den_l[gl]
    taper[good] = num_t[good] / den[good]
    pbeff[good] = wpb[good] / den[good]
    os.makedirs(out_dir, exist_ok=True)
    mname, tname = names(variant, scope, nside)
    path = os.path.join(out_dir, mname)
    hdu = fits.BinTableHDU.from_columns([fits.Column(name='LEVEL', format='E', array=level.astype(np.float32)),
                                         fits.Column(name='TAPERONLY', format='E', array=taper.astype(np.float32))])
    for k, val in (('PIXTYPE', 'HEALPIX'), ('ORDERING', 'RING'), ('NSIDE', nside), ('COORDSYS', 'C'), ('INDXSCHM', 'IMPLICIT'),
                   ('BUNIT', ''), ('WEIGHTS', 'mosaic tile weights PB^2/sigma_beam^2, feathered'),
                   ('CAVEAT', 'Gregg 2026-10-31 ok on caveats')):
        hdu.header[k] = val
    hdu.header.add_comment('LEVEL = weighted mean over hybrid tiles of large-scale level / LWA1 survey prior in the pixel Dec band')
    hdu.header.add_comment('(4-deg means, Dec bands +40..+90, +10..+40, -10..+10, -20..-10, -30..-20; 18-59 MHz limits 0.5-1.6)')
    hdu.header.add_comment('TAPERONLY = weight fraction of Taper-only tiles (NoTaper stack refused: no large-scale fill)')
    keys = list(dict.fromkeys(k for r in rows for k in r))
    tpath = os.path.join(out_dir, tname)
    with placement.atomic_many([path, tpath]) as tmp:
        fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(tmp[path], overwrite=True)
        with open(tmp[tpath], 'w', newline='') as f:
            w_ = csv.DictWriter(f, fieldnames=keys)
            w_.writeheader()
            w_.writerows(rows)
    rec = dict(variant=variant, scope=scope, map=path, table=tpath, n_hybrid=sum(r['kind'] == 'hybrid' for r in rows),
               n_taper_only=sum(r['kind'] == 'Taper-only' for r in rows), taper_only=[r['lst'] for r in rows if r['kind'] == 'Taper-only'])
    rec['test'] = test_caveat_map(path, level, taper, pbeff, lmin, lmax, good, gl, num_t, den, pbeff_mosaic)
    return rec


def test_caveat_map(path, level, taper, pbeff, lmin, lmax, good, gl, num_t, den, pbeff_mosaic):
    """The caveat map read back (NL 1835): (1) LEVEL and TAPERONLY equal the computed arrays (float32), NaN exactly where they are;
    (2) TAPERONLY in [0, 1], exactly 0 where no Taper-only tile has weight and 1 where only Taper-only tiles have; (3) LEVEL within the
    range of the contributing tiles' levels (a weighted mean); (4) the weights are the mosaic's: the effective PB these weights give
    equals the mosaic's PB_eff map (NSIDE 2048, averaged to this NSIDE) - median |difference| below mosaic_caveat.PBEFF_MED_MAX, 99th
    percentile below PBEFF_P99_MAX, over more than PBEFF_MIN_PIXELS pixels where both exceed PBEFF_MIN (the rim, where the resolutions
    differ, left out). Returns the record with verdict 'pass'; raises CaveatRefusal otherwise."""
    import healpy as hp
    C = _cfg().mosaic_caveat
    with fits.open(path, memmap=False) as hd:
        L = np.asarray(hd[1].data['LEVEL'], dtype=np.float64)
        T = np.asarray(hd[1].data['TAPERONLY'], dtype=np.float64)
    res = {}
    res['readback_ok'] = bool(np.array_equal(np.isfinite(L), np.isfinite(level)) and np.array_equal(np.isfinite(T), np.isfinite(taper))
                              and np.array_equal(L[np.isfinite(L)], level[np.isfinite(level)].astype(np.float32).astype(np.float64))
                              and np.array_equal(T[np.isfinite(T)], taper[np.isfinite(taper)].astype(np.float32).astype(np.float64)))
    t = T[good]
    res['taperonly_range'] = [float(np.min(t)), float(np.max(t))] if t.size else None
    res['taperonly_ok'] = bool(t.size and np.all((t >= -1e-6) & (t <= 1 + 1e-6)) and np.all(T[good & (num_t == 0)] == 0)
                               and np.all(np.abs(T[good & (num_t > 0) & (num_t >= den * (1 - 1e-9))] - 1) < 1e-5))
    res['level_within_tiles'] = bool(np.all((L[gl] >= lmin[gl] - 1e-4) & (L[gl] <= lmax[gl] + 1e-4)))
    res['level_range'] = [float(np.nanmin(L)), float(np.nanmax(L))] if np.isfinite(L).any() else None
    if pbeff_mosaic and os.path.exists(pbeff_mosaic):
        kw = {'verbose': False} if 'verbose' in hp.read_map.__code__.co_varnames else {}
        m = hp.read_map(pbeff_mosaic, memmap=False, **kw)
        m = np.where(np.isfinite(m), m, hp.UNSEEN)
        nside = hp.npix2nside(len(level))
        md = hp.ud_grade(m, nside_out=nside)
        md = np.where(md == hp.UNSEEN, np.nan, md)
        lo = float(C.PBEFF_MIN)
        k = np.isfinite(md) & np.isfinite(pbeff) & (md > lo) & (pbeff > lo)
        d = np.abs(md[k] - pbeff[k])
        res['pbeff_vs_mosaic'] = dict(n=int(k.sum()), median_abs=float(np.median(d)) if d.size else None,
                                      p99_abs=float(np.percentile(d, 99)) if d.size else None)
        res['weights_ok'] = bool(k.sum() > int(C.PBEFF_MIN_PIXELS) and res['pbeff_vs_mosaic']['median_abs'] < C.PBEFF_MED_MAX
                                 and res['pbeff_vs_mosaic']['p99_abs'] < C.PBEFF_P99_MAX)
    else:
        res['weights_ok'] = False
        res['pbeff_vs_mosaic'] = 'mosaic PB_eff map missing'
    if not (res['readback_ok'] and res['taperonly_ok'] and res['level_within_tiles'] and res['weights_ok']):
        raise CaveatRefusal(f'test_caveat_map FAILED ({path}): {res} - REFUSING')
    res['verdict'] = 'pass'
    return res
