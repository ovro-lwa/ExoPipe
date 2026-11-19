"""Stages `conform` and `tables` of `exopipe3 mosaic --generation v3` (spec plan/specs/hybrid.md 2.15, PR-6d): the archive-staging
copies of one (scope, job), and the run's tables. A port of the v3 branch of production mosaic_conformance_pass.py (MC, exopipe main:
the Stokes I pass, MOSAIC_V3_PRODUCTS=V and =PKCORR / PKCORR_SSUB) and of the FEATHER3 tables (Run_FEATHER3_20241218-20251029/tables).

conform, per (scope, job), on the unit's own records (v3.conform_unit checks them):
  membership  MC membership (v3): per LST in order, the staged tile and its weighting map, read whole (memmap=False), with the build's
              guards (conform_v5.membership). PROVENANCE rows: FILE = the real path of the tile master (MC: os.path.realpath of the
              staged link), MJD_OBS, MJD_MID, INTTIME, WEIGHT = 1 / sigma^2, THERMNOI, P1RUN = the tile's GITTAG; the time block and
              the median beam as conform_v5.membership.
  THERMNOI    MC thermnoi_for: the scope's cleaned Stokes V Jy/sr mosaic (this run's Vclean coadd) over the zenith strip
              mosaic_conform.THERMNOI_DEC_STRIP_DEG: 1.4826 MAD [Jy/sr] x the median Omega_eff [sr] of its PSF map = Jy/beam. Every
              product of the scope carries it (Stokes I and V, the renders, the PKCORR map), as MC and the 2026-11-13 header repair.
  products    MC's order: the Jy/sr map and its two renders, the Jy/beam map, the PB_eff map, the Jy/beam renders, the PSF map, and
              for Stokes I the PKCORR map, under <dest>/<scope>/<I|V>/<healpix|mollweide|radec>/ (the archive run's layout). Names
              by core.products.build_name (lst 'allsky'; dates and N from the membership; Stokes V adds the token vclean, a
              striation-corrected job ssub-rst8).
  stamp       as conform_v5.stage: write_standard_header (time_basis mosaic_conform.TIME_BASIS; the renders with validate=True, the
              HEALPix maps with validate=mosaic_conform.HEADER_VALIDATE), PRODNAME (HDU 0, and HDU 1 of a HEALPix map), NSTACK, PROVBAS,
              BEAMNOTE, the cards MC copies from HDU 1, append_provenance_table, and the HISTORY line 'Mosaic <run>: N constituent
              masters ...' wrapped at 72. Every file is written under a temporary name and renamed into place only when the whole
              unit succeeded.
Changes from production (spec section 4 notes):
  - one form for every product: the archived files got PRODNAME and HISTORY (2026-11-11) and the Stokes I files THERMNOI (2026-11-13)
    from the Release 1 header repairs, whose own HISTORY lines a v2 product does not carry; a PKCORR map carries the wrapped HISTORY
    line and PRODNAME's standard comment (production's PKCORR pass wrote two short lines and 'Phase 3 product name'); a HEALPix map
    carries PRODNAME in HDU 1 as well as HDU 0;
  - the Jy/beam map's PSFMAP card names the staged PSF product (bug B14: production's names a work-tree file);
  - PROVBAS and BEAMNOTE state the v2 basis (production's name its staging tool, and a beam repair ledger the v3 beams never used: MC
    takes them from the master headers);
  - THERMNOI comes from this run's Stokes V coadd of the scope, which the conform therefore requires (production read the V mosaic of
    its working tree).

tables, one unit per run: each FEATHER3 table is the archived table (mosaic_v3.ARCHIVE_RUN/tables) with the rows of every job conformed
in this run replaced by this run's rows, in the archived row order: summary.csv (the Stokes I content reports, sorted by map name),
summary_vclean.csv (the Stokes V TEST 1R reports, sorted by map name), pkcorr_curves.csv (the curve files, a last column version plain |
ssub-rst8; in mosaic_v3.VARIANTS order, then band, then version); the striation selection tables (mosaic_v3.TABLES_CARRIED) carried
forward whole: they are this run's input (hybrid.STRIATION_SELECTION). A table refuses unless it holds exactly the rows of every job of
mosaic_v3.SCOPES (both domains), every result PASS.
"""
import json
import os

import numpy as np

from exopipe.phase3.lib.mosaic import conform_v5 as CF5
from exopipe.phase3.lib.mosaic import gen_common as GC
from exopipe.phase3.lib.mosaic.conform_v5 import ConformRefusal

SUMMARY_COLS = CF5.SUMMARY_COLS
VCLEAN_COLS = ('map', 'band', 'domain', 'n_tiles', 'n_positions', 'median_absdiff_over_scale', 'p90_absdiff_over_scale', 'result')
CURVE_COLS = CF5.CURVE_COLS + ('version',)
RENDERS = CF5.RENDERS


def _cfg():
    from exopipe.phase3.config import CFG
    return CFG


def archive_run():
    """The mosaic archive run of the FEATHER3 mosaics (its name in each product's HISTORY line; its tables are the base of this run's)."""
    return _cfg().mosaic_v3.ARCHIVE_RUN


def membership(tiles, scope, floor):
    """MC membership (v3) on the staged tiles: conform_v5.membership with FILE = the real path of the tile master."""
    return CF5.membership(tiles, scope, floor, file_of=lambda t: os.path.realpath(t['path']))


def thermnoi(vjysr, vpsf):
    """MC thermnoi_for on the scope's cleaned Stokes V Jy/sr map and its PSF map: 1.4826 MAD x median Omega_eff over the zenith strip
    (mosaic_conform.THERMNOI_DEC_STRIP_DEG, deg, exclusive bounds), in Jy/beam. Read whole by healpy (memmap=False), as MC.
    Raises ConformRefusal unless the value is finite and positive."""
    import healpy as hp
    lo, hi = _cfg().mosaic_conform.THERMNOI_DEC_STRIP_DEG
    v = hp.read_map(vjysr, memmap=False)
    p = hp.read_map(vpsf, memmap=False)
    if len(v) != len(p):
        raise ConformRefusal(f'THERMNOI: {os.path.basename(vjysr)} has {len(v)} pixels, its PSF map {len(p)}')
    th, _ = hp.pix2ang(hp.npix2nside(len(v)), np.arange(len(v)))
    dec = 90.0 - np.degrees(th)
    s = np.isfinite(v) & np.isfinite(p) & (dec > lo) & (dec < hi)
    if not s.any():
        raise ConformRefusal(f'THERMNOI: no finite pixel of {os.path.basename(vjysr)} in Dec {lo}-{hi} deg')
    mad = 1.4826 * np.median(np.abs(v[s] - np.median(v[s])))
    val = float(mad * np.median(p[s]))
    if not np.isfinite(val) or val <= 0:
        raise ConformRefusal(f'THERMNOI: {val!r} Jy/beam from {os.path.basename(vjysr)} (not finite and positive)')
    return val


def product_plan(P, job):
    """[(kind, source path, subdirectory, proc tokens, PBCORR)] in MC's order (v3.paths keys)."""
    C = _cfg()
    K = C.mosaic_conform
    v = ['vclean'] if job.product == 'Vclean' else []
    suf = [C.masters_striation.TAG] if job.ssub else []
    base = v + list(K.BASE_PROC) + suf
    app = v + [p for p in K.BASE_PROC if p not in K.APPARENT_DROP_PROC] + suf
    noj = v + [p for p in K.BASE_PROC if p != 'jysr'] + suf
    items = [('jysr', P['jysr'], 'healpix', base, True)]
    items += [(f'jysr_{k}', P[f'render_jysr_{k}'], k, base + [tok], True) for k, tok in RENDERS]
    items += [('jybeam', P['jybeam'], 'healpix', app + ['jybeam'], False), ('pbmap', P['pbeff'], 'healpix', app + ['pbmap'], False)]
    items += [(f'jybeam_{k}', P[f'render_jybeam_{k}'], k, app + ['jybeam', tok], False) for k, tok in RENDERS]
    items.append(('psfmap', P['psf'], 'healpix', noj + ['psfmap'], True))
    if job.product == 'I':
        items.append(('pkcorr', P['pkcorr_map'], 'healpix', noj + ['pkcorr'], True))
    return items


def stokes_of(job):
    return 'V' if job.product == 'Vclean' else 'I'


def names(scope, job, P, t0, t1, n):
    """{kind: (source, subdirectory, product name, PBCORR)}."""
    from astropy.time import Time
    from exopipe.phase3.core.products import build_name
    K = _cfg().mosaic_conform
    common = dict(scope=scope, stokes=stokes_of(job), cadence=K.CADENCE, taper=job.variant.split('_')[0],
                  robust=-0.75 if job.variant.endswith('R075') else 0.0, lst=K.LST_TOKEN,
                  date_start=Time(t0, format='mjd').strftime('%Y%m%d'), date_end=Time(t1, format='mjd').strftime('%Y%m%d'), n_epochs=n)
    return {k: (src, sub, build_name(proc=proc, **common), pbc) for k, src, sub, proc, pbc in product_plan(P, job)}


def subdir(dest, scope, job, sub):
    """<dest>/<scope>/<I|V>/<sub>/ (the archive run's layout)."""
    return os.path.join(dest, scope, stokes_of(job), sub)


def provbas():
    C = _cfg()
    return C.mosaic_v3.PROVBAS.format(input_run=os.path.relpath(C.mosaic.INPUT_COADD_RUN, C.paths.PHASE3_ARCHIVE_ROOT))


def stage(scope, job, P, tiles, dest, floor, therm, plan=None):
    """Stage the unit's products under dest/<scope>/<I|V>/<subdirectory>/. Returns ([(final path, source, kind)], info). plan(final
    paths) is called once the names are known, before anything is written. Raises ConformRefusal; nothing is renamed into place unless
    every product was written (gen_common.stamp_products)."""
    C = _cfg()
    K = C.mosaic_conform
    V3 = C.mosaic_v3
    rows, t0, t1, itt, beam = membership(tiles, scope, floor)
    n = len(rows)
    nm = names(scope, job, P, t0, t1, n)
    if plan:
        plan([os.path.join(subdir(dest, scope, job, sub), name) for _src, sub, name, _p in nm.values()])
    pv = provbas()
    items = []
    for kind, (src, sub, name, pbc) in nm.items():
        base = {'PRODNAME': (name, 'Phase 3 product name (grammar; CONTINUE cards)'), 'NSTACK': (n, 'Constituent LST tiles'),
                'PROVBAS': pv, 'BEAMNOTE': V3.BEAMNOTE}
        if kind == 'jysr':
            ex = CF5._hdr1(src, K.COPY_HEADER_KEYS, base)
        elif kind in ('jybeam', 'pbmap'):
            ex = CF5._hdr1(src, K.COPY_HEADER_KEYS_APPARENT, dict(base, **({'BUNIT': ''} if kind == 'pbmap' else {})))
        elif kind == 'psfmap':
            ex = dict(base, BUNIT='sr')
        elif kind == 'pkcorr':
            ex = dict(base, BUNIT='')
        else:                                                          # a render: MC stamps the base cards only
            ex = dict(base)
        items.append((kind, src, sub, os.path.join(subdir(dest, scope, job, sub), name), pbc, ex))
    out = GC.stamp_products(items, rows, GC.mosaic_history(os.path.basename(archive_run()), n), scope=scope, stokes=stokes_of(job),
                            t0=t0, t1=t1, itt=itt, beam=beam, thermnoi=therm, psf_name=nm['psfmap'][2])
    info = {'n': n, 't0': t0, 't1': t1, 'inttime_s': itt, 'beam': beam, 'names': {k: v[2] for k, v in nm.items()},
            'prov_files': [r['FILE'] for r in rows]}
    return out, info


# ---- tables -----------------------------------------------------------------------------------------------------------------------

def vclean_rows(content_json):
    """summary_vclean.csv: one row per domain of a Stokes V TEST 1R report (jybeam, then jysr)."""
    with open(content_json) as f:
        rep = json.load(f)
    return [[str(x) if not isinstance(x, str) else x for x in
             (os.path.basename(r['map']), r['band'], r['domain'], r['n_tiles'], r['n_positions'], r['median_absdiff_over_scale'],
              r['p90_absdiff_over_scale'], r['result'])] for r in (rep[d] for d in ('jybeam', 'jysr'))]


def curve_rows(variant, band, version, curve_csv):
    """pkcorr_curves.csv: the curve file's rows with the variant, band and version (plain | ssub-rst8)."""
    return [r + [version] for r in CF5.curve_rows(variant, band, curve_csv)]


def curve_sort(r):
    """The archived order: mosaic_v3.VARIANTS order, then band, then version (a stable sort keeps each curve's Dec order)."""
    return (list(_cfg().mosaic_v3.VARIANTS).index(r[0]), r[1], r[-1])


def map_sort(r):
    return r[0]
