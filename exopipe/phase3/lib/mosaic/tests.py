"""Content tests of the `exopipe3 mosaic` product classes (registered with core.validate; discovered from lib/<step>/tests.py,
and imported by steps/mosaic.py, the OWNERS module of class 'mosaic').

  mosaic_map        <out>/healpix/healpix_I_deep_<V>_<S>_{jysr,jybeam}_nside2048.fits (and the conformed copies)
  mosaic_psfmap     <out>/healpix/healpix_PSF_I_deep_<V>_<S>_nside2048.fits
  mosaic_pbeff      <out>/healpix/healpix_PBeff_I_deep_<V>_<S>_nside2048.fits
  mosaic_render     <out>/renders/<S>_<V>_<D>_{Mollweide,RADec}.fits
  mosaic_qa_table   <out>/qa/flux_anchor_measured_<date>.csv, vlssr_astrometry_<date>.csv
  mosaic            the archive class: <out>/archive_staging/mosaic/... and the pushed copies (the push audit of
                    push_mosaic_v2_20260929.py:75-114, plus record-based checks when the step passes its context)
  mosaic_weight_pb  the staged weighting PB maps (node-local scratch; gates G5/G6 as a test)

Every test reads the file's CONTENT. A test returns (passed, value, detail). Context keys the step passes (all optional, so
`exopipe3 verify <class> <files>` works alone): tiles (the coadd record's tile list), partial (a waived < 24-tile build),
domain, psfmap / jysr_map / jybeam_map (sibling paths), source (the build product a render or staged copy came from),
kind, master, robust.
"""
import math
import os
import re

from exopipe.phase3.core.validate import content_test

_C_LIGHT, _K_B = 2.99792458e8, 1.380649e-23
_FWHM2OMEGA = math.pi / (4 * math.log(2))
_MAPS = {}            # (path, mtime_ns, size) -> (signal float32 array, header HDU1, header HDU0)
_MASKS = {}


def _cfg():
    from exopipe.phase3.config import CFG
    return CFG


def _read_hpx(path):
    import numpy as np
    from astropy.io import fits
    st = os.stat(path)
    key = (os.path.abspath(path), st.st_mtime_ns, st.st_size)
    if key in _MAPS:
        return _MAPS[key]
    with fits.open(path, memmap=True) as h:
        if len(h) < 2 or getattr(h[1], 'columns', None) is None or 'SIGNAL' not in h[1].columns.names:
            raise ValueError('no HDU 1 binary table with a SIGNAL column')
        sig = np.array(h[1].data['SIGNAL'], dtype=np.float32).ravel()
        out = (sig, h[1].header.copy(), h[0].header.copy())
    if len(_MAPS) >= 3:
        _MAPS.pop(next(iter(_MAPS)))
    _MAPS[key] = out
    return out


def _north_mask(nside, floor_deg):
    """Boolean per RING pixel: Dec >= floor, with the engine's own theta (hp.pix2ang of each ring; CJ:243-244)."""
    import numpy as np
    import healpy as hp
    key = (int(nside), float(floor_deg))
    if key not in _MASKS:
        rings = np.arange(1, 4 * nside)
        startpix, ringpix, _c, _s, _sh = hp.ringinfo(nside, rings)
        th, _ = hp.pix2ang(nside, startpix)
        ok = (90.0 - np.degrees(th)) >= floor_deg
        _MASKS.clear()
        _MASKS[key] = np.repeat(ok, ringpix)
    return _MASKS[key]


def _domain(path, ctx):
    b = os.path.basename(path)
    if ctx.get('domain') in ('jysr', 'jybeam') and 'PSF' not in b and 'PBeff' not in b:
        return ctx['domain']
    return 'jybeam' if 'jybeam' in b else 'jysr'


def _nside_expected(ctx):
    return int(ctx.get('nside') or _cfg().mosaic_coadd.NSIDE)


# ---------------------------------------------------------------------------------------------------------------------
# mosaic_map
# ---------------------------------------------------------------------------------------------------------------------

@content_test('mosaic_map')
def healpix_structure(path, **ctx):
    sig, h1, _ = _read_hpx(path)
    n = _nside_expected(ctx)
    probs = []
    for k, want in (('PIXTYPE', 'HEALPIX'), ('ORDERING', 'RING'), ('COORDSYS', 'C'), ('INDXSCHM', 'IMPLICIT')):
        if str(h1.get(k, '')).strip() != want:
            probs.append(f'{k} {h1.get(k)!r} != {want!r}')
    if int(h1.get('NSIDE', -1)) != n:
        probs.append(f'NSIDE {h1.get("NSIDE")} != {n}')
    if sig.size != 12 * n * n:
        probs.append(f'{sig.size} pixels != 12 x {n}^2')
    return not probs, sig.size, '; '.join(probs) or f'RING NSIDE {n}, {sig.size} pixels'


@content_test('mosaic_map')
def domain_cards(path, **ctx):
    C = _cfg()
    sig, h1, _ = _read_hpx(path)
    dom = _domain(path, ctx)
    cc = C.mosaic_coadd
    probs = []
    bunit = str(h1.get('BUNIT', '')).strip()
    if dom == 'jysr' and (bunit != 'Jy/sr' or h1.get('PBCORR') is not True):
        probs.append(f'jysr map: BUNIT {bunit!r} PBCORR {h1.get("PBCORR")!r} (want Jy/sr, True)')
    if dom == 'jybeam':
        if bunit != 'Jy/beam' or h1.get('PBCORR') is not False:
            probs.append(f'jybeam map: BUNIT {bunit!r} PBCORR {h1.get("PBCORR")!r} (want Jy/beam, False)')
        psf = str(h1.get('PSFMAP', ''))
        b = os.path.basename(path)
        m = re.match(r'^healpix_(I_deep_.+)_jybeam_nside(\d+)\.fits$', b)
        if not psf:
            probs.append('jybeam map without the PSFMAP card')
        elif m and psf != f'healpix_PSF_{m.group(1)}_nside{m.group(2)}.fits':
            probs.append(f'PSFMAP {psf!r} does not name the jysr pair psfmap')
    if h1.get('ALIGNED') is not False:
        probs.append(f'ALIGNED {h1.get("ALIGNED")!r} (no mosaic-stage flow, M-01)')
    feather = f'PB {cc.PB_FEATHER_START}->{cc.PB_FLOOR} cosine weight taper'
    if str(h1.get('FEATHER', '')) != feather:
        probs.append(f'FEATHER {h1.get("FEATHER")!r} != {feather!r}')
    if not str(h1.get('WEIGHTS', '')).startswith('PB^2/sigma_beam^2'):
        probs.append(f'WEIGHTS {h1.get("WEIGHTS")!r}')
    return not probs, dom, '; '.join(probs) or f'{dom}: BUNIT {bunit}, PBCORR {h1.get("PBCORR")}, {feather}'


@content_test('mosaic_map')
def dec_floor(path, **ctx):
    import numpy as np
    sig, h1, _ = _read_hpx(path)
    n = int(h1.get('NSIDE', 0)) or _nside_expected(ctx)
    north = _north_mask(n, _cfg().mosaic_coadd.DEC_FLOOR_DEG)
    south_finite = int(np.isfinite(sig[~north]).sum())
    return south_finite == 0, south_finite, (f'{south_finite} finite pixels south of Dec '
                                             f'{_cfg().mosaic_coadd.DEC_FLOOR_DEG} deg')


@content_test('mosaic_map')
def coverage(path, **ctx):
    import numpy as np
    C = _cfg()
    sig, h1, _ = _read_hpx(path)
    n = int(h1.get('NSIDE', 0)) or _nside_expected(ctx)
    north = _north_mask(n, C.mosaic_coadd.DEC_FLOOR_DEG)
    frac = float(np.isfinite(sig[north]).mean())
    scope = ctx.get('scope') or _scope_of(path)
    need = (C.mosaic_coadd.CONTENT_MIN_COVERAGE or {}).get(scope)
    if ctx.get('partial'):
        return frac > 0, round(frac, 5), f'partial build (waived): finite fraction {frac:.5f} north of the floor'
    if need is None:
        return frac > 0, round(frac, 5), (f'finite fraction {frac:.5f} north of the floor; no threshold for {scope!r} yet '
                                          f'(mosaic_coadd.CONTENT_MIN_COVERAGE, measured at parity P0)')
    return frac >= need, round(frac, 5), f'{scope}: finite fraction {frac:.5f} of the pixels north of the floor (>= {need})'


def _scope_of(path):
    b = os.path.basename(path)
    m = re.match(r'^healpix_(?:PSF_|PBeff_)?I_deep_(?:No)?Taper_R0(?:75)?_(\w+?)_(?:jysr_|jybeam_)?nside\d+\.fits$', b)
    if m:
        return m.group(1)
    return b.split('_', 1)[0]


@content_test('mosaic_map')
def jy2k(path, **ctx):
    sig, h1, _ = _read_hpx(path)
    C = _cfg()
    freq = h1.get('FREQ')
    try:
        freq = float(freq)
    except (TypeError, ValueError):
        return False, freq, 'FREQ missing or not a number'
    if not (freq > C.mosaic_coadd.MIN_FREQ_HZ):
        return False, freq, f'FREQ {freq} Hz not above {C.mosaic_coadd.MIN_FREQ_HZ:g} (JY2K would be 0)'
    if _domain(path, ctx) != 'jysr':
        return True, freq, f'FREQ {freq:.0f} Hz'
    want = 1e-26 * _C_LIGHT ** 2 / (2 * _K_B * freq ** 2)
    got = h1.get('JY2K')
    if got is None:
        return False, None, 'jysr map without JY2K'
    rel = abs(float(got) - want) / want
    return rel <= 1e-9, float(got), f'JY2K {float(got):.6g} vs c^2/(2 k nu^2) x 1e-26 = {want:.6g} (rel {rel:.1e})'


# ---------------------------------------------------------------------------------------------------------------------
# siblings (build names and conformed names)
# ---------------------------------------------------------------------------------------------------------------------

def _sibling(path, want):
    """The sibling map of a psfmap / pbeff / map, by build name (healpix_PSF_<stem>_nside..) or conformed name."""
    d, b = os.path.split(path)
    m = re.match(r'^healpix_(PSF|PBeff)_(I_deep_.+)_nside(\d+)\.fits$', b)
    if m:
        dom = {'jysr': 'jysr', 'jybeam': 'jybeam'}[want]
        p = os.path.join(d, f'healpix_{m.group(2)}_{dom}_nside{m.group(3)}.fits')
        return p if os.path.exists(p) else None
    try:
        from exopipe.phase3.core.products import parse_name, build_name
        pn = parse_name(b)
        proc = [t for t in (pn.get('proc') or []) if t not in ('psfmap', 'pbmap', 'jysr', 'jybeam')]
        if want == 'jysr':
            proc = proc + ['jysr']
        else:
            proc = [t for t in proc if t != 'pbcorr'] + ['jybeam']
        pn['proc'] = proc
        p = os.path.join(d, build_name(**pn))
        return p if os.path.exists(p) else None
    except Exception:
        return None


def _mask_compare(path, sib):
    import numpy as np
    if not sib:
        return True, None, 'no sibling map found (mask not compared)'
    a, _, _ = _read_hpx(path)
    b, _, _ = _read_hpx(sib)
    if a.size != b.size:
        return False, None, f'{a.size} pixels vs {b.size} in {os.path.basename(sib)}'
    nd = int((np.isnan(a) != np.isnan(b)).sum())
    return nd == 0, nd, f'{nd} pixels differ in NaN mask from {os.path.basename(sib)}'


@content_test('mosaic_psfmap')
def omega_bounds(path, **ctx):
    import numpy as np
    sig, h1, _ = _read_hpx(path)
    probs = []
    if str(h1.get('BUNIT', '')).strip() != 'sr':
        probs.append(f'BUNIT {h1.get("BUNIT")!r} != sr')
    v = sig[np.isfinite(sig)]
    if v.size == 0:
        return False, 0, 'no finite pixel'
    lo, hi = float(v.min()), float(v.max())
    tiles = [t for t in (ctx.get('tiles') or []) if t.get('verdict') == 'added' and t.get('BMAJ') and t.get('BMIN')]
    if tiles:
        om = [_FWHM2OMEGA * float(t['BMAJ']) * float(t['BMIN']) * (math.pi / 180) ** 2 for t in tiles]
        floor = _cfg().mosaic_coadd.COSZ_FLOOR
        a, b = min(om) * (1 - 1e-4), max(om) / floor * (1 + 1e-4)
        if lo < a or hi > b:
            probs.append(f'Omega_eff {lo:.4g}..{hi:.4g} sr outside [{a:.4g}, {b:.4g}] (min Omega_zen, max Omega_zen / '
                         f'{floor})')
    elif lo <= 0 or hi > 1e-2:
        probs.append(f'Omega_eff {lo:.4g}..{hi:.4g} sr not in (0, 0.01]')
    return not probs, round(float(np.median(v)), 12), '; '.join(probs) or (f'Omega_eff {lo:.4g}..{hi:.4g} sr'
                                                                           + (' (bounds from the tiles)' if tiles else ''))


@content_test('mosaic_psfmap')
def psf_mask_matches_map(path, **ctx):
    return _mask_compare(path, ctx.get('jysr_map') if ctx.get('jysr_map') and os.path.exists(ctx['jysr_map'])
                         else _sibling(path, 'jysr'))


@content_test('mosaic_pbeff')
def pb_bounds(path, **ctx):
    import numpy as np
    sig, h1, _ = _read_hpx(path)
    probs = []
    if str(h1.get('BUNIT', 'x')).strip() != '':
        probs.append(f'BUNIT {h1.get("BUNIT")!r} != dimensionless')
    v = sig[np.isfinite(sig)]
    if v.size == 0:
        return False, 0, 'no finite pixel'
    lo, hi = float(v.min()), float(v.max())
    floor = _cfg().mosaic_coadd.PB_FLOOR
    if lo < floor - 1e-6 or hi > 1.0 + 1e-6:
        probs.append(f'PB_eff {lo:.4f}..{hi:.4f} outside [{floor}, 1.0]')
    return not probs, round(hi, 4), '; '.join(probs) or f'PB_eff {lo:.4f}..{hi:.4f}'


@content_test('mosaic_pbeff')
def pbeff_mask_matches_map(path, **ctx):
    return _mask_compare(path, ctx.get('jybeam_map') if ctx.get('jybeam_map') and os.path.exists(ctx['jybeam_map'])
                         else _sibling(path, 'jybeam'))


# ---------------------------------------------------------------------------------------------------------------------
# mosaic_render
# ---------------------------------------------------------------------------------------------------------------------

def _render(path):
    import numpy as np
    from astropy.io import fits
    with fits.open(path, memmap=True) as h:
        img = np.asarray(h[0].data)
        hdr = h[0].header.copy()
    return img, hdr


def _render_source(path, ctx):
    if ctx.get('source') and os.path.exists(ctx['source']):
        return ctx['source']
    d, b = os.path.split(path)
    m = re.match(r'^(\w+?)_((?:No)?Taper_R0(?:75)?)_(jysr|jybeam)_(Mollweide|RADec)\.fits$', b)
    if m:
        p = os.path.join(os.path.dirname(d), 'healpix', f'healpix_I_deep_{m.group(2)}_{m.group(1)}_{m.group(3)}'
                                                        f'_nside{_cfg().mosaic_coadd.NSIDE}.fits')
        return p if os.path.exists(p) else None
    try:                                     # archive layout <band>/I/{mollweide,radec}/<name> -> ../healpix/<map name>
        from exopipe.phase3.core.products import parse_name, build_name
        pn = parse_name(b)
        pn['proc'] = [t for t in pn.get('proc') or [] if t not in ('molrender', 'carrender')]
        p = os.path.join(os.path.dirname(d), 'healpix', build_name(**pn))
        return p if os.path.exists(p) else None
    except Exception:
        return None


@content_test('mosaic_render')
def projection(path, **ctx):
    import numpy as np
    C = _cfg()
    img, h = _render(path)
    probs = []
    if img.ndim != 2 or img.dtype.kind != 'f' or img.dtype.itemsize != 4:
        return False, str(img.shape), f'not a 2-D float32 image ({img.dtype}, {img.shape})'
    ny, nx = img.shape
    c1, c2 = str(h.get('CTYPE1', '')), str(h.get('CTYPE2', ''))
    if (c1, c2) == ('GLON-MOL', 'GLAT-MOL'):
        if ny != nx // 2:
            probs.append(f'Mollweide shape {img.shape} is not (W/2, W)')
    elif (c1, c2) == ('RA---CAR', 'DEC--CAR'):
        cd = abs(float(h.get('CDELT1', 0)))
        r = C.mosaic_render
        want = (int(round((r.CAR_DEC_MAX_DEG - r.CAR_DEC_MIN_DEG) / cd)), int(round(360.0 / cd)))
        if (ny, nx) != want:
            probs.append(f'CAR shape {img.shape} != {want} for CDELT {cd}')
    else:
        probs.append(f'CTYPE {c1}/{c2} is neither GLON-MOL/GLAT-MOL nor RA---CAR/DEC--CAR')
    nz = int((img == 0).sum())
    if nz:
        probs.append(f'{nz} pixels exactly 0 (zeros are NaN in a render, BP:29)')
    fin = int(np.isfinite(img).sum())
    if fin == 0:
        probs.append('no finite pixel')
    return not probs, fin, '; '.join(probs) or f'{c1} {img.shape}, {fin} finite pixels'


@content_test('mosaic_render')
def resample_agrees(path, **ctx):
    import numpy as np
    import healpy as hp
    from astropy.wcs import WCS
    src = _render_source(path, ctx)
    if not src:
        return True, None, 'source HEALPix map not found (resampling not compared)'
    img, h = _render(path)
    sig, h1, _ = _read_hpx(src)
    if str(h.get('BUNIT', '')).strip() != str(h1.get('BUNIT', '')).strip():
        return False, None, f'BUNIT {h.get("BUNIT")!r} != source {h1.get("BUNIT")!r}'
    ys, xs = np.nonzero(np.isfinite(img))
    if ys.size == 0:
        return False, 0, 'no finite pixel'
    rng = np.random.RandomState(12345)
    pick = rng.choice(ys.size, size=min(1000, ys.size), replace=False)
    ys, xs = ys[pick], xs[pick]
    w = WCS(h).celestial
    a, b = w.pixel_to_world_values(xs, ys)
    if str(h.get('CTYPE1', '')).startswith('GLON'):
        import astropy.coordinates as ac
        import astropy.units as u
        sc = ac.SkyCoord(np.asarray(a) * u.deg, np.asarray(b) * u.deg, frame='galactic').icrs
        ra, dec = sc.ra.deg, sc.dec.deg
    else:
        ra, dec = np.asarray(a), np.asarray(b)
    val = hp.get_interp_val(np.asarray(sig), ra, dec, lonlat=True).astype(np.float32)
    got = img[ys, xs].astype(np.float32)
    ok = np.isfinite(val) & (val != 0)
    rel = np.abs(got[ok] - val[ok]) / np.maximum(np.abs(val[ok]), 1e-30)
    worst = float(rel.max()) if rel.size else 0.0
    return worst <= 1e-6, worst, f'{int(ok.sum())} sampled pixels; max relative difference {worst:.2e} vs ' \
                                 f'{os.path.basename(src)}'


@content_test('mosaic_render')
def beam_card(path, **ctx):
    import numpy as np
    C = _cfg()
    img, h = _render(path)
    bmaj = h.get('BMAJ')
    if bmaj is None:
        return False, None, 'no BMAJ card'
    psf = ctx.get('psfmap')
    if psf is None and ctx.get('kind') is None:           # verify alone: the psfmap beside the source map
        src = _render_source(path, ctx)
        if src:
            cand = re.sub(r'healpix_I_deep_(.+)_(jysr|jybeam)_nside', r'healpix_PSF_I_deep_\1_nside', src)
            psf = cand if cand != src and os.path.exists(cand) else None
    if psf and os.path.exists(psf):
        import warnings
        import healpy as hp
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')               # 'verbose' deprecation; the call is the renderer's own (BP:39)
            om = hp.read_map(psf, verbose=False)
        om = om[np.isfinite(om) & (om > 0) & (om != hp.UNSEEN)]
        if om.size:
            bm = float(np.degrees(np.sqrt(np.median(om) / C.mosaic_render.OMEGA_PER_FWHM2)) * 60.0)   # BP:40
            ok = abs(float(bmaj) - bm / 60) <= 1e-9
            return ok, round(bm, 4), f'BMAJ {float(bmaj) * 60:.4f} arcmin vs PSF-map beam {bm:.4f} arcmin'
    scope = ctx.get('scope') or str(h.get('BAND', ''))
    fb = C.mosaic_render.BEAM_FALLBACK_ARCMIN.get(scope)
    if fb is None:
        return False, float(bmaj) * 60, f'no PSF map and no fallback beam for {scope!r}'
    ok = abs(float(bmaj) - fb / 60) <= 1e-9
    return ok, fb, f'BMAJ {float(bmaj) * 60:.4f} arcmin = fallback table {fb} arcmin (no PSF map; B-45)'


# ---------------------------------------------------------------------------------------------------------------------
# mosaic_qa_table
# ---------------------------------------------------------------------------------------------------------------------

ANCHOR_COLUMNS = ['variant', 'band', 'freq_mhz', 'source', 'aperture_jy', 'model_jy', 'ratio', 'fwhm_eff_arcmin']


def _kind(path, ctx):
    return ctx.get('kind') or ('anchor' if 'flux_anchor' in os.path.basename(path) else
                               'vlssr' if 'vlssr' in os.path.basename(path) else None)


@content_test('mosaic_qa_table')
def anchor_rows(path, **ctx):
    import csv
    if _kind(path, ctx) != 'anchor':
        return True, None, 'not an anchor table'
    from exopipe.common import pipeline_config as pc
    from exopipe.phase3.lib.mosaic.verify_mosaic import model_flux
    with open(path, newline='') as f:
        rd = csv.DictReader(f)
        cols = rd.fieldnames or []
        rows = list(rd)
    if cols != ANCHOR_COLUMNS:
        return False, len(rows), f'columns {cols} != {ANCHOR_COLUMNS}'
    sh12 = [n for n, d in pc.CALIB_DATA.items() if d.get('scale', 'SH12') == 'SH12']
    groups = {}
    for r in rows:
        groups.setdefault((r['variant'], r['band']), []).append(r)
    probs = []
    exact = {}                                   # (variant, band) -> the map's FREQ in MHz, when the step names the maps
    for m in ctx.get('maps') or []:
        try:
            from astropy.io import fits
            exact[(m['variant'], m['band'])] = float(fits.getheader(m['jysr'], 1)['FREQ']) / 1e6
        except Exception:
            pass
    for key, rs in groups.items():
        names = [r['source'] for r in rs]
        if names[:-1] != sh12 or not names[-1].startswith('ANCHOR'):
            probs.append(f'{key}: sources {names}')
            continue
        f = {r['freq_mhz'] for r in rs}
        if len(f) != 1:
            probs.append(f'{key}: freq {f}')
            continue
        fr = float(f.pop())
        if key in exact and abs(exact[key] - fr) > 0.0005 + 1e-9:
            probs.append(f'{key}: freq_mhz {fr} vs the map FREQ {exact[key]:.6f} MHz')
        rat = {}
        for r in rs[:-1]:
            d = pc.CALIB_DATA[r['source']]
            mod = model_flux(d['coeffs'], exact.get(key, fr), 'SH12')
            # the script computes the model at the map's exact FREQ and writes freq_mhz rounded to 1 kHz: from the
            # rounded value the model can differ by |spectral index| x 0.0005 / freq_mhz (relative)
            tol = 0.0015 if key in exact else 0.0015 + 1e-4 * abs(mod)
            if abs(float(r['model_jy']) - mod) > tol:
                probs.append(f'{key} {r["source"]}: model {r["model_jy"]} vs {mod:.3f}')
            rat[r['source']] = float(r['ratio'])
        a = rs[-1]['ratio']
        exp = math.sqrt(rat.get('3C196', float('nan')) * rat.get('3C380', float('nan')))
        if a.lower() == 'nan' and math.isnan(exp):
            continue
        if not (abs(float(a) - exp) <= 2e-4):
            probs.append(f'{key}: ANCHOR {a} vs sqrt(3C196 x 3C380) {exp:.4f}')
    maps = ctx.get('maps') or []
    if maps:
        want = {(m['variant'], m['band']) for m in maps}
        if set(groups) != want:
            probs.append(f'maps {sorted(want)} vs table {sorted(groups)}')
    return not probs, len(groups), '; '.join(probs[:6]) or f'{len(groups)} maps x {len(sh12)} SH12 sources + ANCHOR'


@content_test('mosaic_qa_table')
def astrometry_rows(path, **ctx):
    import csv
    if _kind(path, ctx) != 'vlssr':
        return True, None, 'not a VLSSr table'
    from exopipe.phase3.lib.mosaic.steps_worker import VLSSR_COLUMNS
    with open(path, newline='') as f:
        rd = csv.DictReader(f)
        cols = rd.fieldnames or []
        rows = list(rd)
    if cols != VLSSR_COLUMNS:
        return False, len(rows), f'columns {cols} != {VLSSR_COLUMNS}'
    probs = []
    for r in rows:
        try:
            n = int(r['n_matched'])
            vals = [float(r[k]) for k in VLSSR_COLUMNS[3:10]]
        except (TypeError, ValueError):
            probs.append(f'{r["variant"]} {r["band"]}: no statistics ({r.get("note")})')
            continue
        if n <= 0 or not all(math.isfinite(v) for v in vals):
            probs.append(f'{r["variant"]} {r["band"]}: n_matched {n}')
    maps = ctx.get('maps') or []
    if maps and {(m['variant'], m['band']) for m in maps} != {(r['variant'], r['band']) for r in rows}:
        probs.append('rows do not match the maps measured')
    return not probs, len(rows), '; '.join(probs[:6]) or f'{len(rows)} maps matched to VLSSr'


# ---------------------------------------------------------------------------------------------------------------------
# mosaic (archive class): the push audit + record-based checks
# ---------------------------------------------------------------------------------------------------------------------

_UNIT = {'jysr': 'Jy/sr', 'jybeam': 'Jy/beam', 'psfmap': 'sr', 'pbmap': ''}


def _archive_read(path):
    import numpy as np
    from astropy.io import fits
    with fits.open(path, memmap=True) as h:
        dh = h[0] if h[0].data is not None else h[1]
        hdr_all = {**dict(h[0].header), **dict(dh.header)}
        arr = np.asarray(dh.data if dh is h[0] else dh.data['SIGNAL'])
        prov = None
        if 'PROVENANCE' in [x.name for x in h]:
            prov = h['PROVENANCE'].data
            prov = [{n: prov[n][i] for n in prov.columns.names} for i in range(len(prov))]
        h0 = h[0].header.copy()
    return hdr_all, arr, prov, h0


@content_test('mosaic')
def name_roundtrip(path, **ctx):
    from exopipe.phase3.core.products import build_name, parse_name
    n = os.path.basename(path)
    try:
        pn = parse_name(n)
    except ValueError as e:
        return False, None, f'name does not parse: {e}'
    ok = build_name(**pn) == n
    return ok, None, 'name round-trips' if ok else 'name does not round-trip'


@content_test('mosaic')
def cards_vs_name(path, **ctx):
    import numpy as np
    from exopipe.phase3.core.products import parse_name
    pn = parse_name(os.path.basename(path))
    proc = pn.get('proc') or []
    hdr_all, arr, _prov, _h0 = _archive_read(path)
    p = []
    if bool(hdr_all.get('PBCORR')) != ('pbcorr' in proc):
        p.append(f"PBCORR {hdr_all.get('PBCORR')} vs name")
    dom = next((t for t in ('psfmap', 'pbmap', 'jybeam', 'jysr') if t in proc), None)
    if dom and str(hdr_all.get('BUNIT', '')).strip() != _UNIT[dom]:
        p.append(f"BUNIT {hdr_all.get('BUNIT')!r} vs {dom}")
    if int(hdr_all.get('NSTACK', -1)) != int(pn.get('n_epochs') or -2):
        p.append(f"NSTACK {hdr_all.get('NSTACK')} vs N{pn.get('n_epochs')}")
    for k in ('DATE-OBS', 'DATE-END'):
        if not hdr_all.get(k):
            p.append(f'no {k}')
    if not np.isfinite(np.asarray(arr.ravel()[::97])).any():
        p.append('no finite pixel')
    return not p, dom, '; '.join(p) or f'{dom}: PBCORR, BUNIT, NSTACK, time block, finite pixels agree with the name'


@content_test('mosaic')
def pixels_unchanged(path, **ctx):
    """The staged copy's data equal the build product it was copied from (conform never changes pixels; G11)."""
    import hashlib
    import numpy as np
    src = ctx.get('source')
    if not src:
        return True, None, 'no source build product given (record-based check skipped)'
    _a, arr, _p, _h = _archive_read(path)
    from astropy.io import fits
    with fits.open(src, memmap=True) as h:
        dh = h[0] if h[0].data is not None else h[1]
        brr = np.asarray(dh.data if dh is h[0] else dh.data['SIGNAL'])
    ha = hashlib.md5(np.ascontiguousarray(arr).tobytes()).hexdigest()
    hb = hashlib.md5(np.ascontiguousarray(brr).tobytes()).hexdigest()
    return ha == hb, ha[:12], f'data md5 {ha[:12]} vs source {hb[:12]} ({os.path.basename(src)})'


def _contrib(ctx):
    return sorted((t for t in (ctx.get('tiles') or []) if t.get('verdict') == 'added'), key=lambda t: int(t['lst'][:2]))


@content_test('mosaic')
def provenance_vs_record(path, **ctx):
    tiles = _contrib(ctx)
    hdr_all, _arr, prov, _h0 = _archive_read(path)
    if prov is None:
        return False, None, 'no PROVENANCE table'
    if not tiles:
        return len(prov) == int(hdr_all.get('NSTACK', -1)), len(prov), \
            f'{len(prov)} provenance rows vs NSTACK {hdr_all.get("NSTACK")} (no record context)'
    p = []
    if len(prov) != len(tiles):
        p.append(f'{len(prov)} rows vs {len(tiles)} contributing tiles')

    def s(x):
        return x.decode() if isinstance(x, bytes) else str(x)
    files = [s(r['FILE']).strip() for r in prov]
    want = [t['realpath'] for t in tiles]
    if files != want:
        p.append('FILE column != the contributing masters (LST order)')
    for r, t in zip(prov, tiles):
        w = float(r['WEIGHT'])
        if not (w > 0) or (t.get('weight') is not None and w != float(t['weight'])):
            p.append(f'{t["lst"]}: WEIGHT {w} vs record {t.get("weight")}')
            break
    return not p, len(prov), '; '.join(p) or f'{len(prov)} rows = the contributing masters, weights = 1/sigma^2'


@content_test('mosaic')
def time_beam_vs_record(path, **ctx):
    import numpy as np
    from astropy.time import Time
    tiles = _contrib(ctx)
    if not tiles:
        return True, None, 'no record context (time and beam not compared)'
    hdr_all, _arr, _prov, _h0 = _archive_read(path)
    p = []
    t0 = min(t['MJD-OBS'] for t in tiles if t.get('MJD-OBS') is not None)
    t1 = max(Time(t['DATE-END']).mjd for t in tiles if t.get('DATE-END'))
    if hdr_all.get('DATE-OBS') != Time(t0, format='mjd').isot:
        p.append(f"DATE-OBS {hdr_all.get('DATE-OBS')} != {Time(t0, format='mjd').isot}")
    if hdr_all.get('DATE-END') != Time(t1, format='mjd').isot:
        p.append(f"DATE-END {hdr_all.get('DATE-END')} != {Time(t1, format='mjd').isot}")
    itt = sum(float(t.get('INTTIME') or 0) for t in tiles)
    if itt and abs(float(hdr_all.get('INTTIME', -1)) - itt) > 1e-6:
        p.append(f"INTTIME {hdr_all.get('INTTIME')} != {itt}")
    bt = [(float(t['BMAJ']), float(t['BMIN'])) for t in tiles if t.get('BMAJ') and t.get('BMIN')]
    if bt:
        bm = np.array(bt)
        for k, v in (('BMAJ', float(np.median(bm[:, 0]))), ('BMIN', float(np.median(bm[:, 1])))):
            if abs(float(hdr_all.get(k, -1)) - v) > 1e-12:
                p.append(f'{k} {hdr_all.get(k)} != median {v}')
    return not p, round(itt, 3), '; '.join(p) or 'DATE-OBS/DATE-END/INTTIME/BMAJ/BMIN = the constituents'


# ---------------------------------------------------------------------------------------------------------------------
# mosaic_weight_pb (scratch)
# ---------------------------------------------------------------------------------------------------------------------

_GRID = ('NAXIS1', 'NAXIS2', 'CRPIX1', 'CRPIX2', 'CRVAL1', 'CRVAL2', 'CDELT1', 'CDELT2')


def _pb_identity(path):
    m = re.match(r'^PB_deep_(R075_)?(\w+)\.fits$', os.path.basename(path))
    if not m:
        return None, None
    return ('R075' if m.group(1) else 'R0'), m.group(2)


@content_test('mosaic_weight_pb')
def grid_matches_master(path, **ctx):
    from astropy.io import fits
    rob, scope = _pb_identity(path)
    master = ctx.get('master')
    if not master and scope:
        cand = os.path.join(os.path.dirname(path), f'I_deep_Taper_{rob}_pbcorr_{scope}.fits')
        master = cand if os.path.exists(cand) else None
    if not master:
        return False, None, 'no master to compare the grid with'
    hp_ = fits.getheader(path)
    hm = fits.getheader(master)
    diff = [k for k in _GRID if hp_.get(k) != hm.get(k)]
    return not diff, len(diff), (f'grid cards differ from {os.path.basename(master)}: '
                                 + ', '.join(f'{k} {hp_.get(k)} vs {hm.get(k)}' for k in diff)) if diff else \
        f'grid = {os.path.basename(os.path.realpath(master))}'


@content_test('mosaic_weight_pb')
def robust_card(path, **ctx):
    from astropy.io import fits
    rob, _scope = _pb_identity(path)
    rob = ctx.get('robust') or rob
    h = fits.getheader(path)
    want = '-0.75' if rob == 'R075' else '0'
    got = str(h.get('PBROBUST', ''))
    pbfor = str(h.get('PBFOR', ''))
    tok = 'Taper_Robust-0.75' if rob == 'R075' else 'Taper_Robust+0.0'
    p = []
    if got != want:
        p.append(f'PBROBUST {got!r} != {want!r} (a map from before PB step 2 or of the other robust)')
    if f'_I_deep_{tok}' not in pbfor or '_I_deep_No' in pbfor:
        p.append(f'PBFOR {pbfor!r} is not the {tok} master')
    for k in ('PBCHROM', 'PBRESID'):
        if k not in h:
            p.append(f'no {k}')
    return not p, got, '; '.join(p) or f'PBROBUST {got}, PBFOR {pbfor[:40]}'


@content_test('mosaic_weight_pb')
def pb_range(path, **ctx):
    import numpy as np
    from astropy.io import fits
    d = np.asarray(fits.getdata(path), dtype=np.float32)
    v = d[np.isfinite(d)]
    if v.size == 0:
        return False, 0, 'no finite pixel'
    lo, hi = float(v.min()), float(v.max())
    floor = _cfg().mosaic_coadd.PB_FLOOR
    ok = lo >= 0 and hi <= 1.5 and (v > floor).any()
    return ok, round(hi, 4), f'PB {lo:.4f}..{hi:.4f}; {int((v > floor).sum())} pixels above {floor}'
