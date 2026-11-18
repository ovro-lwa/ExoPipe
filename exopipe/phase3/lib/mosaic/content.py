"""Content TEST 1 and TEST 2 of a mosaic against its own staged tiles: a port of production mosaic_content_test.py (MT, exopipe 4c7098c)
for `exopipe3 mosaic --generation v5`, stage `test` (spec plan/specs/hybrid.md 2.12; PR-7b).

TEST 1 - the map is the weighted co-add of its inputs (MT 5-13, run 108): at the brightest compact cataloged sources of the scope
  (the union parent catalogs master_R075_<LST>_<scope>.gaul.csv, isolated by mosaic_content.ISOLATION_BEAMS beams, Dec >= DEC_MIN +
  3 beams), the map's mean within APERTURE_BEAMS beams is compared with the same mean recomputed from the staged tiles (Jy/sr: the
  beam-corrected tile x n(l,m) / Omega_zen; Jy/beam: the tile x PB), combined with the co-add's weight at the source (PB^2 /
  sigma^2 x the PB feather; sigma = the tile's apparent-domain MAD). PASS: at least MIN_SOURCES sources, |median ratio - 1| <=
  MED_TOL, a fraction >= FRAC5_MIN of the sources within 5 %, every input a link.
TEST 2 - provenance (MT provenance 95): run() reports, per LST, the tile and PB files TEST 1 read (their link targets); the unit
  (v5.content_unit) passes TEST 2 only when they are exactly the files of its tile list, each at the md5 it verified this run. MT
  recorded the md5 of each tile's first 64 MB and compared it with nothing; v2 checks the whole file against the stage's record.

The arithmetic is MT's line for line; the engine module is the v2 coadd_jysr with the mosaic_coadd globals bound (v5._bind_engine).
Changes: every read uses memmap=False (MT read the Lustre tiles memory-mapped: spec section 7 B1); the tiles and the sources are read
once for both domains (MT ran once per domain); the isolation distance is astrometry.sep_arcmin (MT: the same spherical law of
cosines on unit vectors); no parseable catalog row gives a FAIL with 0 sources, not an error. The map tested is the UNFILLED
co-add: production's V5 tables/summary.csv rows are the unfilled maps' values (reproduced to the last digit on 2026-11-18 from the
PR-7a Blue Taper R-0.75 maps; the filled maps fail TEST 1 by design, since the fill adds the large scales the tiles lack).
"""
import glob
import os

import numpy as np


def _cfg():
    from exopipe.phase3.config import CFG
    return CFG.mosaic_content


def _sources(parents, band, nmax, beam_deg):
    """MT 27-48: the nmax brightest cataloged sources of the band, Dec >= DEC_MIN + 3 beams, isolated by ISOLATION_BEAMS beams
    (array of (RA, Dec, peak); empty when no catalog row parses)."""
    from exopipe.phase3.core.astrometry import sep_arcmin
    C = _cfg()
    rows = []
    for p in sorted(glob.glob(os.path.join(glob.escape(parents), f'master_R075_*_{band}.gaul.csv'))):
        with open(p) as f:
            for ln in f:
                if ln.startswith('#') or not ln.strip():
                    continue
                q = ln.split(',')
                try:
                    rows.append((float(q[4]) % 360, float(q[6]), float(q[10])))
                except (ValueError, IndexError):
                    pass
    S = np.array(rows, dtype=np.float64).reshape(-1, 3)
    S = S[(S[:, 1] >= float(C.DEC_MIN) + 3 * beam_deg) & np.isfinite(S).all(1)]
    S = S[np.argsort(-S[:, 2], kind='stable')]   # sorted catalogs + stable sort: ties resolve the same way on every filesystem
    pick = []
    iso_arcmin = float(C.ISOLATION_BEAMS) * beam_deg * 60.0
    for i in range(len(S)):
        if all(sep_arcmin(S[i, 0], S[i, 1], S[j, 0], S[j, 1]) > iso_arcmin for j in pick):
            pick.append(i)
        if len(pick) == nmax:
            break
    return S[pick]


class Tile:
    """MT 55-104: one staged tile (link tree under the engine's names) and its weighting map, for both domains."""

    def __init__(self, root, lst, variant, band, product='I'):
        from astropy.io import fits
        from astropy.wcs import WCS
        from exopipe.phase3.lib.mosaic import coadd_jysr as M
        self.M = M
        self.mp, self.pbp = M._paths(root, lst, variant, band, product)
        self.ok = os.path.exists(self.mp) and os.path.exists(self.pbp)
        if not self.ok:
            return
        self.read = dict(lst=lst, file=os.path.realpath(self.mp), pb=os.path.realpath(self.pbp),
                         symlink=os.path.islink(self.mp) and os.path.islink(self.pbp))
        with fits.open(self.mp, memmap=False) as h:
            self.h = h[0].header.copy()
            d = h[0].data
        self.w = WCS(self.h).celestial
        self.d = np.nan_to_num(d.astype(np.float32).squeeze())
        with fits.open(self.pbp, memmap=False) as h:
            self.pb = np.nan_to_num(h[0].data.astype(np.float32).squeeze())
        app = (self.d * self.pb)[self.pb > M.PBCORR_FLOOR]
        app = app[np.isfinite(app)]
        med = np.median(app)
        self.sigma = 1.4826 * np.median(np.abs(app - med)) or 1e-6
        self.om0 = M.omega_for(lst, band, variant, product, self.h)
        self.cd = abs(float(self.h['CDELT2']))

    def aperture(self, ra, dec, R, domain):
        """(mean of the tile in map units within R deg, co-add weight at the center) or None (MT 71-93)."""
        M = self.M
        x0, y0 = self.w.all_world2pix([[ra, dec]], 0)[0]
        if not (np.isfinite(x0) and np.isfinite(y0)):
            return None
        yi, xi = int(round(y0)), int(round(x0))
        H, W = self.d.shape
        if not (0 <= yi < H and 0 <= xi < W):
            return None
        pbc = float(self.pb[yi, xi])
        if pbc <= M.PBCORR_FLOOR:
            return None
        r = int(np.ceil(R / self.cd)) + 2
        y, x = np.mgrid[max(0, yi - r):min(H, yi + r + 1), max(0, xi - r):min(W, xi + r + 1)]
        a, b = self.w.all_pix2world(x, y, 0)
        # image pixels within R of the source (pixel selection, not an association): astrometry.sep_arcmin's formula, vectorized
        cosd = (np.sin(np.radians(b)) * np.sin(np.radians(dec))
                + np.cos(np.radians(b)) * np.cos(np.radians(dec)) * np.cos(np.radians(a - ra)))
        sel = np.degrees(np.arccos(np.clip(cosd, -1, 1))) <= R
        if sel.sum() < 3:
            return None
        v = self.d[y[sel], x[sel]].astype(np.float64)
        if domain == 'jybeam':
            v = v * self.pb[y[sel], x[sel]]
        else:
            l = np.radians((x[sel] + 1.0 - self.w.wcs.crpix[0]) * self.w.wcs.cdelt[0])  # noqa: E741
            m = np.radians((y[sel] + 1.0 - self.w.wcs.crpix[1]) * self.w.wcs.cdelt[1])
            n = np.sqrt(np.clip(1.0 - l * l - m * m, M.N_FLOOR ** 2, 1.0))
            v = v * n / self.om0
        ramp = np.clip((pbc - M.PBCORR_FLOOR) / (M.PB_FEATHER - M.PBCORR_FLOOR), 0.0, 1.0)
        wgt = pbc ** 2 / self.sigma ** 2 * (0.5 - 0.5 * np.cos(np.pi * ramp))
        return float(np.mean(v)), float(wgt)


def _read_map(path):
    from exopipe.phase3.lib.mosaic.pkcorr import read_map
    return read_map(path)


def run(maps, root, variant, band, parents, n=None):
    """TEST 1 (MT run 108-140) of each domain's map ({domain: path}) against one reading of the staged tiles and sources: {domain:
    report}. Each report holds result PASS / FAIL for TEST 1 and `read`, the files TEST 1 read (TEST 2 is judged by the unit)."""
    import healpy as hp
    C = _cfg()
    n = int(n or C.N_SOURCES)
    lsts = sorted(os.path.basename(d) for d in glob.glob(os.path.join(glob.escape(root), '*h')))
    tiles = {lst: Tile(root, lst, variant, band) for lst in lsts}
    tiles = {lst: t for lst, t in tiles.items() if t.ok}
    beam = float(np.median([t.h['BMAJ'] for t in tiles.values()])) if tiles else float('nan')
    R = float(C.APERTURE_BEAMS) * beam
    src = _sources(parents, band, n, beam) if tiles else np.zeros((0, 3))
    read = [t.read for t in tiles.values()]
    out = {}
    for domain, map_path in maps.items():
        mp = _read_map(map_path)
        nside = hp.get_nside(mp)
        ratios = []
        for ra, dec, _ in src:
            vec = hp.ang2vec(ra, dec, lonlat=True)
            pix = hp.query_disc(nside, vec, np.radians(R))
            vals = mp[pix]
            vals = vals[np.isfinite(vals) & (vals != hp.UNSEEN)]
            if vals.size < 3:
                continue
            num = den = 0.0
            for t in tiles.values():
                r = t.aperture(ra, dec, R, domain)
                if r is None:
                    continue
                num += r[1] * r[0]
                den += r[1]
            if den <= 0:
                continue
            exp = num / den
            if exp != 0:
                ratios.append(float(np.mean(vals) / exp))
        del mp
        ratios = np.array(ratios)
        med = float(np.median(ratios)) if len(ratios) else float('nan')
        frac5 = float(np.mean(np.abs(ratios - 1) <= 0.05)) if len(ratios) else 0.0
        ok = len(ratios) >= int(C.MIN_SOURCES) and abs(med - 1) <= float(C.MED_TOL) and frac5 >= float(C.FRAC5_MIN)
        out[domain] = dict(map=map_path, variant=variant, band=band, domain=domain, n_tiles=len(tiles), n_sources=int(len(ratios)),
                           median_ratio=med, p05=float(np.percentile(ratios, 5)) if len(ratios) else None,
                           p95=float(np.percentile(ratios, 95)) if len(ratios) else None, frac_within_5pct=frac5,
                           read=read, result='PASS' if ok else 'FAIL')
    return out
