"""lib/masters/striation.py - remove the earth-fixed striations from a deep Stokes I master, with the sky restored from other LSTs.

Port of production exopipe/phase3/striation_restore.py (exopipe 3bda09e = 4c7098c for that file) for the masters deep stage
`striation` (lib/masters/deep_stages.py). Ruling: Gregg 2026-10-22 (R-GREGG-20261022-STRIATION-RESTORE-AND-MOSAIC-SCOPE-MEM-OPTION-B):
"why not fit the striations in 08h and just use adjacent LSTs to restore subtracted sky flux?"; "you only need this for the Masters
that have bad striations ... Investigate what images need it and correct only those. Create a version with and without
striations."; window 8 beams (R-STRIATION-WINDOW-APPROVED-8BEAMS).

THE METHOD (unchanged from production; apparent masters of one band and variant, all 24 LSTs share one zenith-centred SIN grid shape):
  operator M_W(img, mask)  running median along RA over W beams inside 1-pixel Declination strips (true Dec of every pixel),
                           then a 31-pixel high-pass along image columns; zero where masked.
  sky estimate S_L(set)    PB_L x (PB^2-weighted mean over LSTs n in set of master_n / PB_n, bilinearly reprojected onto L's
                           grid; only PB_n >= 0.1)
  pass 1 (every LST n)     c_n = master_n - M_PASS1_WIN(master_n - S_n({n-1, n+1} raw))
  pass 2 (target L)        corrected_L = master_L - M_WIN(master_L - S_L({L+-1, L+-2, L+-3} pass-1)); model zero where no neighbor
                           covers the pixel; fit inputs exclude invalid pixels, |master - S| > 10 x rms and a one-beam zone around
                           pixels above 5 x rms of the master; the model is then made zero-mean on scales > 31 pixels.

DIFFERENCES FROM PRODUCTION (each keeps the products bit-identical, or refuses where production passed vacuously):
  - paths are arguments (the installed working tree, the archive run), not environment variables;
  - every FITS read is fits.open(memmap=False) (Gregg 2026-11-17: never memory-map a Lustre file);
  - a band's LSTs are loaded on demand and pass 1 is computed only for the LSTs a target needs (L-3..L+3): pass 1 at n reads only
    the raw masters n-1, n, n+1, so the result is the one production's all-24-LST pass gives;
  - pass-1 results may be cached on node-local disk, keyed by the size and mtime of the inputs they derive from;
  - a missing parents catalog refuses (production measured the catalogued-source flux ratio over 0 sources and passed it as 1.0).

STRIATION INDEX: rms of M_8(master) over rms of M_8(master rotated 90 deg) in the central field (sources > 5 rms, |b| < 5 deg and
  smoothed diffuse emission masked); ~1 without striations. Reference ceiling per variant = the highest index of the 23-55 MHz masters
  (census 2026-10-22).

PRODUCTS (the uncorrected master stays as it is): the corrected apparent master, its PB-corrected sibling and the subtracted model,
named with the grammar token TAG after the master's own tokens; header = the master's header + the standard header (validate=True)
+ STR* cards. CONTENT TESTS (refuse on failure): the working master equals the archived PB-corrected master after division by its PB
map; corrected = master - model exactly; the corrected sibling = corrected / PB; the striation index after correction <= the
reference ceiling; the off-plane mean and the catalogued-source flux within 1 per cent of the master's.
"""
import glob
import hashlib
import os

import numpy as np

VARIANTS = {'Taper_R075': ('Taper', -0.75, 'Taper_Robust-0.75'), 'Taper_R0': ('Taper', 0.0, 'Taper_Robust+0.0'),
            'NoTaper_R075': ('NoTaper', -0.75, 'NoTaper_Robust-0.75'), 'NoTaper_R0': ('NoTaper', 0.0, 'NoTaper_Robust+0.0')}


class StriationRefusal(RuntimeError):
    """An input or content test refuses (the stage maps it to a refused unit)."""


def rr(a):
    return 1.4826 * np.median(np.abs(a - np.median(a)))


def _params():
    from exopipe.phase3.config import CFG
    return CFG.masters_striation


def _read(path, header=False):
    from astropy.io import fits
    with fits.open(path, memmap=False) as h:
        d = np.array(h[0].data)
        return (d, h[0].header.copy()) if header else d


def master_path(root, lst, band, variant):
    return f'{root}/{lst}_{band}/coadd_final/I_deep_{variant}_{band}.fits'


def pb_path(root, lst, band, variant):
    """The PB map the variant's PB-corrected sibling was divided by (pb_apply.deep_pb_name convention)."""
    rob = VARIANTS[variant][1]
    return f'{root}/{lst}_{band}/coadd_final/PB_deep_R075_{band}.fits' if rob == -0.75 else f'{root}/{lst}_{band}/coadd_final/PB_deep_{band}.fits'


class Cell:
    """One LST master of a (band, variant): apparent pixels, PB map, sky coordinates of every pixel."""
    def __init__(self, root, lst, band, variant):
        from astropy.wcs import WCS
        self.lst, self.band, self.variant = lst, band, variant
        self.path = master_path(root, lst, band, variant); self.pb_path = pb_path(root, lst, band, variant)
        for p in (self.path, self.pb_path):
            if not os.path.isfile(p):
                raise StriationRefusal(f'{lst} {band} {variant}: input {p} is missing')
        raw, self.h = _read(self.path, header=True)
        self.raw_shape = raw.shape
        self.d = np.squeeze(raw).astype(np.float64)
        self.pb = np.squeeze(_read(self.pb_path)).astype(np.float64)
        self.w = WCS(self.h).celestial
        self.H, self.Wd = self.d.shape
        self.cd = abs(float(self.h['CDELT2'])); self.bpx = float(self.h['BMAJ']) / self.cd
        yy, xx = np.indices(self.d.shape)
        self.ra = np.full(self.d.shape, np.nan, np.float32); self.de = np.full(self.d.shape, np.nan, np.float32)
        for y0 in range(0, self.H, 512):
            r_, d_ = self.w.all_pix2world(xx[y0:y0 + 512], yy[y0:y0 + 512], 0); self.ra[y0:y0 + 512] = r_; self.de[y0:y0 + 512] = d_
        self.ok = np.isfinite(self.d) & (self.d != 0) & np.isfinite(self.ra) & np.isfinite(self.de)
        self.rms = rr(self.d[self.ok])

    def stamp(self):
        """size and mtime of the master and its PB map (the pass-1 cache key part this cell contributes)."""
        return tuple((os.stat(p).st_size, os.stat(p).st_mtime_ns) for p in (self.path, self.pb_path))


def operator(c, img, mask, win_beams):
    """M_W: running median along RA over win_beams beams in 1-pixel Dec strips, then a 31-pixel column high-pass."""
    import pandas as pd
    from scipy.ndimage import median_filter
    valid = ~mask & c.ok & np.isfinite(img)
    de = c.de.astype(np.float64)
    edges = np.arange(np.nanmin(de[valid]), np.nanmax(de[valid]) + c.cd, c.cd); nb = len(edges) - 1
    dbi = np.clip(np.digitize(de, edges) - 1, 0, nb - 1)
    wp = int(win_beams * c.bpx); wp += (wp % 2 == 0); m = np.zeros(img.shape)
    fl = np.flatnonzero(valid.ravel()); bins = dbi.ravel()[fl]; o = np.argsort(bins, kind='stable'); fl, bins = fl[o], bins[o]
    st = np.searchsorted(bins, np.arange(nb), 'left'); en = np.searchsorted(bins, np.arange(nb), 'right')
    for b in range(nb):
        n = en[b] - st[b]
        if n < 20:
            continue
        py, px = np.divmod(fl[st[b]:en[b]], c.Wd); v = img[py, px]; oo = np.argsort(c.ra[py, px], kind='stable')
        py, px, v = py[oo], px[oo], v[oo]
        m[py, px] = float(np.median(v)) if n < wp else pd.Series(v).rolling(wp, center=True, min_periods=1).median().to_numpy()
    m = m - median_filter(m, size=(31, 1), mode='nearest')
    return np.where(c.ok & ~mask, m, 0.0)


class Band:
    """The LSTs of one (band, variant), loaded on demand, with cached reprojection coordinates and pass-1 results."""
    def __init__(self, root, band, variant, cache_dir=None, pass1_win=None, win=None):
        if variant not in VARIANTS:
            raise StriationRefusal(f'variant {variant} is not one of {sorted(VARIANTS)}')
        self.root, self.band, self.variant, self.cache_dir = root, band, variant, cache_dir
        P = None if (pass1_win and win) else _params()
        self.pass1_win = pass1_win or P.PASS1_WIN; self.win = win or P.WIN
        self.cells, self._coord, self.p1 = {}, {}, {}
        self.pass1_from_cache = 0

    def cell(self, lst):
        if lst not in self.cells:
            self.cells[lst] = Cell(self.root, lst, self.band, self.variant)
        return self.cells[lst]

    @staticmethod
    def nb(lst, k):
        i = (int(lst[:2]) + k) % 24; return f'{i:02d}h'

    def coords(self, src, dst):
        key = (src, dst)
        if key not in self._coord:
            s, t = self.cell(src), self.cell(dst); x = np.full(t.d.shape, np.nan, np.float32); y = np.full_like(x, np.nan)
            for y0 in range(0, t.H, 512):
                r_, d_ = t.ra[y0:y0 + 512].astype(np.float64), t.de[y0:y0 + 512].astype(np.float64); f = np.isfinite(r_)
                xs, ys = s.w.all_world2pix(r_[f], d_[f], 0); xa = x[y0:y0 + 512]; ya = y[y0:y0 + 512]; xa[f] = xs; ya[f] = ys
            self._coord[key] = (x, y)
        return self._coord[key]

    def reproj(self, img, src, dst):
        from scipy.ndimage import map_coordinates
        x, y = self.coords(src, dst); f = np.isfinite(x); out = np.full(x.shape, np.nan)
        good = np.isfinite(img).astype(np.float64); v = np.nan_to_num(img)
        a = map_coordinates(v, [y[f], x[f]], order=1, cval=0.0); g = map_coordinates(good, [y[f], x[f]], order=1, cval=0.0)
        out[f] = np.where(g > 0.999, a, np.nan); return out

    def sky(self, t, ns, imgs):
        c = self.cell(t); num = np.zeros(c.d.shape); den = np.zeros(c.d.shape)
        for n in ns:
            cn = self.cell(n); s = np.where(cn.ok & (cn.pb >= 0.1), imgs[n] / np.where(cn.pb > 0, cn.pb, np.nan), np.nan)
            sr = self.reproj(s, n, t); pr = self.reproj(np.where(cn.ok, cn.pb, np.nan), n, t)
            f = np.isfinite(sr) & np.isfinite(pr) & (pr >= 0.1); wgt = np.where(f, pr ** 2, 0.0)
            num += np.where(f, wgt * sr, 0.0); den += wgt
        cov = (den > 0) & c.ok
        return np.where(cov, num / np.where(den > 0, den, 1.0) * c.pb, np.nan), cov

    def fit(self, t, ns, imgs, win_beams):
        from scipy.ndimage import binary_dilation, uniform_filter
        c = self.cell(t); S, cov = self.sky(t, ns, imgs); D = imgs[t] - S
        zone = binary_dilation(c.ok & (np.abs(np.nan_to_num(c.d)) > 5 * c.rms), iterations=max(1, int(round(c.bpx))))
        fm = ~c.ok | ~cov | ~np.isfinite(D) | (np.abs(np.nan_to_num(D)) > 10 * c.rms) | zone
        m = operator(c, np.nan_to_num(D), fm, win_beams)
        # zero mean on large scales (production 2026-10-22: with the bright-source zone the model's mean lifted the off-plane level
        # 0.5-1.6 %); the striations alternate within a few pixels in Dec, so their 31 x 31 mean is ~0 and they are kept
        w = (m != 0) & c.ok & ~fm
        num = uniform_filter(np.where(w, m, 0.0), size=31); den = uniform_filter(w.astype(np.float64), size=31)
        m = np.where(w, m - np.where(den > 0.05, num / np.where(den > 0, den, 1.0), 0.0), 0.0)
        return m, cov, fm

    def _p1_key(self, n):
        h = hashlib.sha256(repr((self.band, self.variant, n, self.pass1_win,
                                 [self.cell(k).stamp() for k in (self.nb(n, -1), n, self.nb(n, 1))])).encode()).hexdigest()[:24]
        return os.path.join(self.cache_dir, f'p1_{self.band}_{self.variant}_{n}_{h}.npy') if self.cache_dir else None

    def pass1(self, n):
        """c_n = master_n - M_PASS1_WIN(master_n - S_n({n-1, n+1} raw)); reads only the raw masters n-1, n, n+1."""
        if n not in self.p1:
            key = self._p1_key(n)
            if key and os.path.isfile(key):
                self.p1[n] = np.load(key, allow_pickle=False); self.pass1_from_cache += 1
            else:
                lo, hi = self.nb(n, -1), self.nb(n, 1)
                raw = {k: self.cell(k).d for k in (lo, n, hi)}
                m, _, _ = self.fit(n, [lo, hi], raw, self.pass1_win)
                self.p1[n] = raw[n] - m
                if key:
                    os.makedirs(self.cache_dir, exist_ok=True)
                    np.save(key + '.part.npy', self.p1[n], allow_pickle=False); os.replace(key + '.part.npy', key)
        return self.p1[n]

    def correct(self, t):
        """pass 2 for target t: (corrected apparent master, model, coverage)."""
        ns = [self.nb(t, k) for k in (-3, -2, -1, 1, 2, 3)]
        imgs = {n: self.pass1(n) for n in ns}; imgs[t] = self.cell(t).d
        m, cov, fm = self.fit(t, ns, imgs, self.win)
        return self.cell(t).d - m, m, cov


_GEO = {}


def _geo(c):
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    key = (c.d.shape, round(float(c.h['CRVAL1']), 4), round(float(c.h['CDELT2']), 6))
    if key not in _GEO:
        f = c.ok; gb = np.full(c.d.shape, np.nan)
        gb[f] = SkyCoord(c.ra[f].astype(np.float64) * u.deg, c.de[f].astype(np.float64) * u.deg).galactic.b.deg
        yy, xx = np.indices(c.d.shape); rad = np.hypot(yy - c.H / 2, xx - c.Wd / 2) < 0.5 * min(c.H, c.Wd) / 2
        _GEO[key] = (gb, rad)
    return _GEO[key]


def census_mask(c, img):
    from scipy.ndimage import binary_dilation, median_filter
    gb, rad = _geo(c); rms = rr(img[c.ok])
    sm = median_filter(np.where(c.ok, img, 0.0), size=25); srms = rr(sm[c.ok])
    return ~c.ok | (np.abs(np.nan_to_num(img)) > 5 * rms) | binary_dilation(c.ok & ((np.abs(gb) < 5) | (sm > 3 * srms)), iterations=5)


def striation_index(c, img):
    """rms of M_8(img) / rms of M_8(img rotated 90 deg), central field, census mask (module docstring)."""
    gb, rad = _geo(c); mask = census_mask(c, img); im = np.nan_to_num(img)
    m = operator(c, im, mask, 8)
    okr = np.rot90(c.ok); mr_ = np.rot90(mask) | ~c.ok
    mr = operator(c, np.rot90(im), mr_, 8)
    sel = rad & c.ok & ~mask; selr = rad & c.ok & okr & ~mr_
    return float(np.std(m[sel]) / np.std(mr[selr]))


def archive_pbcorr(archive_run, lst, band, variant):
    """The archived PB-corrected master of the cell (exactly one; single-directory glob)."""
    long = VARIANTS[variant][2]
    g = sorted(glob.glob(f'{archive_run}/{lst}/{band}/{band}_I_deep_{long}_shflux_pbcorr_dewarped_aligned_coadd_LST{lst}_*.fits')
               + glob.glob(f'{archive_run}/{lst}/{band}/{band}_I_deep_{long}_shflux_mdewarp_pbcorr_dewarped_aligned_coadd_LST{lst}_*.fits'))
    if len(g) != 1:
        raise StriationRefusal(f'{lst} {band} {variant}: {len(g)} archived PB-corrected masters in {archive_run} (need exactly 1)')
    return g[0]


def parents_csv(parents_dir, lst, band):
    return f'{parents_dir}/master_R075_{lst}_{band}.gaul.csv'


def measures(c, corrected, model, cov, parents):
    """Index before/after, coverage, off-plane and noise ratios, model rms, and the catalogued-source flux ratio over the 300
    brightest parents. Refuses when the parents catalog is missing or yields no measurable source (production passed 1.0)."""
    gb, rad = _geo(c); off = c.ok & (np.abs(gb) > 10)
    cm = census_mask(c, c.d)
    info = dict(index_before=striation_index(c, c.d), index_after=striation_index(c, corrected), coverage=float(cov[c.ok].mean()),
                offplane_ratio=float(np.mean(corrected[off]) / np.mean(c.d[off])) if off.sum() > 1000 else 1.0,
                noise_ratio=float(rr(corrected[c.ok & ~cm]) / rr(c.d[c.ok & ~cm])),
                model_rms_mjy=float(1e3 * np.std(model[c.ok])))
    if not os.path.isfile(parents):
        raise StriationRefusal(f'{c.lst} {c.band} {c.variant}: parents catalog {parents} is missing (the catalogued-source flux test '
                               f'needs it)')
    srcs = []
    for ln in open(parents):
        if ln.startswith('#') or not ln.strip():
            continue
        q = ln.split(',')
        try:
            srcs.append((float(q[4]) % 360, float(q[6]), float(q[10])))
        except (ValueError, IndexError):
            pass
    ratios = []
    if srcs:
        S = np.array(srcs); S = S[np.argsort(-S[:, 2])][:300]; sx, sy = c.w.all_world2pix(S[:, 0], S[:, 1], 0); R = 2 * c.bpx
        for x0, y0 in zip(sx, sy):
            if not (np.isfinite(x0) and 3 * c.bpx < x0 < c.Wd - 3 * c.bpx and 3 * c.bpx < y0 < c.H - 3 * c.bpx):
                continue
            Y, X = np.mgrid[int(y0 - R) - 1:int(y0 + R) + 2, int(x0 - R) - 1:int(x0 + R) + 2]; s = np.hypot(X - x0, Y - y0) <= R
            den = np.nansum(c.d[Y[s], X[s]])
            if den != 0:
                ratios.append(np.nansum(corrected[Y[s], X[s]]) / den)
    if not ratios:
        raise StriationRefusal(f'{c.lst} {c.band} {c.variant}: no catalogued source of {parents} lies on the master '
                               f'({len(srcs)} rows read) - the flux test cannot run')
    info['src_flux_ratio'] = float(np.median(ratios))
    info['n_src'] = len(ratios)
    return info


def product_names(c, archive_run, tag):
    """{kind: file name} of the three products, from the archived PB-corrected master's own name plus the tag."""
    from exopipe.phase3.core import products as P
    pn = P.parse_name(os.path.basename(archive_pbcorr(archive_run, c.lst, c.band, c.variant)))
    proc_pb = list(pn['proc']) + [tag]; proc_ap = [t for t in proc_pb if t != 'pbcorr']
    names = {'apparent': P.build_name(**{**pn, 'proc': proc_ap}), 'pbcorr': P.build_name(**{**pn, 'proc': proc_pb})}
    names['model'] = names['apparent'].replace('.fits', '_striationmodel.fits')
    return names


def write_products(c, corrected, model, out_dir, info, archive_run, tol, tag):
    """Write the corrected apparent master, its PB-corrected sibling and the model; standard header, validated. Refuses when the
    working master / PB differs from the archived PB-corrected master by tol x its rms or more, or a product exists."""
    from astropy.io import fits
    from exopipe.phase3.core import products as P
    arch = archive_pbcorr(archive_run, c.lst, c.band, c.variant)
    ap = np.squeeze(_read(arch)).astype(np.float64)
    f = c.ok & (c.pb > 0.1) & np.isfinite(ap)
    dev = np.nanmax(np.abs(c.d[f] / c.pb[f] - ap[f])) / max(rr(ap[f]), 1e-30)
    if not dev < tol:
        raise StriationRefusal(f'{c.lst} {c.band} {c.variant}: working master / PB differs from the archived master '
                               f'{os.path.basename(arch)} (max {dev:.2e} x rms)')
    names = product_names(c, archive_run, tag)
    ex = [n for n in names.values() if os.path.exists(f'{out_dir}/{n}')]
    if ex:                                        # all three checked before any is written (no partial set)
        raise StriationRefusal(f'{out_dir}: {ex} exist (no silent overwrite)')
    os.makedirs(out_dir, exist_ok=True); written = {}
    for kind, arr in (('apparent', corrected),
                      ('pbcorr', np.where(c.pb > 0.1, corrected / np.where(c.pb > 0, c.pb, np.nan), np.nan)),
                      ('model', model)):
        name = names[kind]; path = f'{out_dir}/{name}'
        fits.writeto(path, arr.reshape(c.raw_shape).astype(np.float32), c.h.copy())
        beam = (float(c.h['BMAJ']), float(c.h['BMIN']), float(c.h.get('BPA', 0.0)))
        extra = {'PRODNAME': (name, 'Phase 3 product name (grammar; CONTINUE cards)'),
                 'STRMETH': ('restore-v1', 'striations fit on master - sky from other LSTs'),
                 'STRWIN': (8, 'RA running-median window [beams], 1-px Dec strips'),
                 'STRZONE': ('5rms+1beam', 'fit excluded around bright pixels (model 0)'),
                 'STRNBR': ('+-1..3h p1', 'sky estimate LSTs (pass-1 destriped, W=4)'),
                 'STRIDXB': (round(info['index_before'], 4), 'striation index before'),
                 'STRIDXA': (round(info['index_after'], 4), 'striation index after'),
                 'STRCOV': (round(info['coverage'], 4), 'fraction of valid pixels with a model'),
                 'STRSRC': (os.path.basename(c.path), 'uncorrected working master'),
                 'STRKIND': (kind, 'apparent / pbcorr / model')}
        P.write_standard_header(path, scope=c.band, stokes='I', start=c.h.get('DATE-OBS'), end=c.h.get('DATE-END'),
                                inttime_s=c.h.get('INTTIME'), time_basis='coadd-span', beam=beam,
                                thermnoi=c.h.get('THERMNOI'), imgrms=float(rr(arr[c.ok])) if kind != 'pbcorr' else None,
                                pbcorr=(kind == 'pbcorr'), dewarped=True, aligned=True, msub=False, ssub=(kind != 'model'),
                                extra=extra, validate=True)
        written[kind] = path
    return written


def content_test(c, written, info, ref_ceiling):
    """[] = pass: exact arithmetic of the written files, index after <= ceiling, off-plane level and source flux kept (1 %)."""
    a = np.squeeze(_read(written['apparent'])).astype(np.float64)
    m = np.squeeze(_read(written['model'])).astype(np.float64)
    p = np.squeeze(_read(written['pbcorr'])).astype(np.float64)
    f = c.ok
    e1 = np.nanmax(np.abs(c.d[f] - m[f] - a[f])); tol = 1e-5 * max(np.nanmax(np.abs(c.d[f])), 1.0)
    g = f & (c.pb > 0.1); e2 = np.nanmax(np.abs(a[g] / c.pb[g] - p[g]) / np.maximum(np.abs(p[g]), rr(p[g])))   # float32 rounding scale
    fails = []
    if not e1 <= tol:
        fails.append(f'apparent != master - model (max {e1:.3g} Jy/beam)')
    if not e2 < 1e-5:
        fails.append(f'pbcorr != apparent / PB (max relative {e2:.3g})')
    ceil = ref_ceiling.get(c.variant)
    if ceil is not None and not info['index_after'] <= ceil:
        fails.append(f'index after {info["index_after"]:.3f} > ceiling {ceil}')
    if not abs(info['offplane_ratio'] - 1) <= 0.01:
        fails.append(f'off-plane mean ratio {info["offplane_ratio"]:.4f}')
    if not abs(info['src_flux_ratio'] - 1) <= 0.01:
        fails.append(f'catalogued-source flux ratio {info["src_flux_ratio"]:.4f}')
    return fails
