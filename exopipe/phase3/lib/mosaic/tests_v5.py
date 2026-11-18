"""Content tests of the mosaic v5 classes (lib/mosaic/v5.py; spec plan/specs/hybrid.md 2.8, 2.10-2.14). Context keys as the v5 units
pass them (exopipe3 verify re-runs them from the record's ctx):
  mosaic_v5_tiles      scope, weighting
  mosaic_filled_map    the mosaic_map context (tiles, partial, scope, tag, psfmap, jysr_map, jybeam_map, domain) + unfilled (the map
                       it fills) and, for the Jy/beam map, filled_jysr (the Jy/sr map whose fill it carries)
  mosaic_caveat_map    scope
  mosaic_caveat_table  n_tiles
The numeric tests of the fill (MF test_tpfill, the Jy/beam identity) and of the caveat map (test_caveat_map) need the computed arrays:
the units record them as inline checks (HG19.fill_jysr, HG19.fill_jybeam, test_caveat_map)."""
import json
import os

from exopipe.phase3.core.validate import content_test
from exopipe.phase3.lib.mosaic import tests as MT


def _cfg():
    from exopipe.phase3.config import CFG
    return CFG


# ---- mosaic_v5_tiles (stage/tiles.json) -------------------------------------------------------------------------------------------

def _tiles(path):
    with open(path) as f:
        return json.load(f)


@content_test('mosaic_v5_tiles')
def tiles_complete(path, **ctx):
    """One tile per LST 00h-23h (HG14: mosaic_v5.REQUIRE_N_LSTS, no stand-in), each a hybrid of source work or pinned, with its md5,
    its weighting map and its nights; the scope and weighting are the unit's."""
    t = _tiles(path)
    rows = t.get('tiles') or []
    n = int(_cfg().mosaic_v5.REQUIRE_N_LSTS)
    probs = []
    lsts = [r.get('lst') for r in rows]
    want = [f'{h:02d}h' for h in range(24)][:n] if n <= 24 else None
    if len(rows) < n or len(set(lsts)) != len(lsts) or (want and sorted(lsts) != want):
        probs.append(f'LSTs {lsts}')
    for r in rows:
        miss = [k for k in ('path', 'md5', 'pb', 'source', 'nights', 'kind') if not r.get(k)]
        if miss or r.get('source') not in ('work', 'pinned') or r.get('kind') != 'hybrid':
            probs.append(f'{r.get("lst")}: missing {miss} or source {r.get("source")!r} / kind {r.get("kind")!r}')
    for k in ('scope', 'weighting'):
        if ctx.get(k) is not None and t.get(k) != ctx[k]:
            probs.append(f'{k} {t.get(k)!r} != {ctx[k]!r}')
    return not probs, len(rows), '; '.join(probs[:4]) or f'{len(rows)} tiles, one per LST'


@content_test('mosaic_v5_tiles')
def tiles_on_disk(path, **ctx):
    """Every tile and weighting map exists with the staged md5 (the files the co-add reads; v5._md5 keeps a file's md5 for the
    process, so the stage's own hashes are not re-read)."""
    from exopipe.phase3.lib.mosaic.v5 import _md5
    rows = _tiles(path).get('tiles') or []
    bad = [r['lst'] for r in rows if not (os.path.exists(r['path']) and os.path.exists(r['pb']))
           or _md5(r['path']) != r['md5'] or _md5(r['pb']) != r.get('pb_md5')]
    return not bad, len(bad), f'tiles missing or changed: {bad}' if bad else f'{len(rows)} tiles at their staged md5'


# ---- mosaic_filled_map (healpix/ filled Jy/sr and Jy/beam maps) -------------------------------------------------------------------

for _fn in (MT.healpix_structure, MT.domain_cards, MT.dec_floor, MT.coverage, MT.jy2k):
    content_test('mosaic_filled_map', _fn.__name__)(_fn)


def _h(path, ext):
    from astropy.io import fits
    with fits.open(path, memmap=False) as h:
        return h[ext].header.copy()


@content_test('mosaic_filled_map')
def fill_cards(path, **ctx):
    """TPFILL, TPDONOR, TPDONMHZ (the LFSS map nearest the map's FREQ), TPSCALE (> 0), TPLLO / TPLHI (hybrid_tilefill.L_LO / L_HI),
    TPSRC / TPSRCMD5 (the unfilled map's name and md5) in HDU 0 and 1; no PRODNAME (the conform stage names the archive file); the
    Jy/beam map also carries TPJBCONV and the Jy/sr map's TPSCALE."""
    from exopipe.phase3.lib.hybrid import tilefill as TF
    from exopipe.phase3.lib.install.steps_common import md5
    TF.bind()
    T = _cfg().hybrid_tilefill
    probs = []
    h1 = _h(path, 1)
    for ext in (0, 1):
        h = _h(path, ext)
        miss = [k for k in ('TPFILL', 'TPDONOR', 'TPDONMHZ', 'TPSCALE', 'TPLLO', 'TPLHI', 'TPSRC', 'TPSRCMD5') if k not in h]
        if miss:
            probs.append(f'HDU {ext} lacks {miss}')
        if 'PRODNAME' in h:
            probs.append(f'HDU {ext} carries PRODNAME {h["PRODNAME"]!r}')
    if probs:
        return False, None, '; '.join(probs)
    if h1['TPFILL'] is not True:
        probs.append(f'TPFILL {h1["TPFILL"]!r}')
    try:
        want = TF.donor_mhz(float(h1['FREQ']))
    except (KeyError, TypeError, ValueError):
        want = None
    if want is None or int(h1['TPDONMHZ']) != want:
        probs.append(f'TPDONMHZ {h1["TPDONMHZ"]} is not the LFSS map nearest FREQ {h1.get("FREQ")} ({want})')
    if not float(h1['TPSCALE']) > 0:
        probs.append(f'TPSCALE {h1["TPSCALE"]}')
    if (int(h1['TPLLO']), int(h1['TPLHI'])) != (int(T.L_LO), int(T.L_HI)):
        probs.append(f'TPLLO/TPLHI {h1["TPLLO"]}/{h1["TPLHI"]} != {T.L_LO}/{T.L_HI}')
    un = ctx.get('unfilled')
    if un:
        if h1['TPSRC'] != os.path.basename(un):
            probs.append(f'TPSRC {h1["TPSRC"]!r} != {os.path.basename(un)!r}')
        if not os.path.exists(un) or h1['TPSRCMD5'] != md5(un):
            probs.append(f'TPSRCMD5 {h1["TPSRCMD5"]} is not the md5 of {un}')
    if MT._domain(path, ctx) == 'jybeam':
        if 'TPJBCONV' not in h1:
            probs.append('Jy/beam map without TPJBCONV')
        fj = ctx.get('filled_jysr')
        if fj and os.path.exists(fj) and _h(fj, 1).get('TPSCALE') != h1['TPSCALE']:
            probs.append(f'TPSCALE {h1["TPSCALE"]} != the filled Jy/sr map\'s {_h(fj, 1).get("TPSCALE")}')
    return not probs, h1.get('TPSCALE'), '; '.join(probs) or f'donor {h1["TPDONMHZ"]} MHz, s {h1["TPSCALE"]}, source {h1["TPSRC"]}'


@content_test('mosaic_filled_map')
def footprint_kept(path, **ctx):
    """The filled map is finite exactly where the unfilled map is (the fill changes values inside the footprint, never it)."""
    import numpy as np
    un = ctx.get('unfilled')
    if not un or not os.path.exists(un):
        return False, None, 'the unfilled map (ctx unfilled) is missing'
    a, _, _ = MT._read_hpx(path)
    b, _, _ = MT._read_hpx(un)
    if a.size != b.size:
        return False, None, f'{a.size} pixels vs {b.size}'
    nd = int((np.isfinite(a) != np.isfinite(b)).sum())
    ch = int(np.sum(np.isfinite(a) & np.isfinite(b) & (a != b)))
    return nd == 0 and ch > 0, nd, f'{nd} pixels differ in the finite mask; {ch} values changed by the fill'


# ---- mosaic_caveat_map / mosaic_caveat_table --------------------------------------------------------------------------------------

@content_test('mosaic_caveat_map')
def caveat_structure(path, **ctx):
    """RING HEALPix at mosaic_caveat.NSIDE with float columns LEVEL and TAPERONLY; TAPERONLY in [0, 1] where finite; nothing finite
    south of the mosaic's Dec floor; LEVEL finite somewhere."""
    import numpy as np
    from astropy.io import fits
    C = _cfg()
    n = int(C.mosaic_caveat.NSIDE)
    with fits.open(path, memmap=False) as h:
        h1 = h[1].header
        cols = h[1].columns.names
        if 'LEVEL' not in cols or 'TAPERONLY' not in cols:
            return False, cols, f'columns {cols} (want LEVEL, TAPERONLY)'
        L = np.asarray(h[1].data['LEVEL'], dtype=float)
        T = np.asarray(h[1].data['TAPERONLY'], dtype=float)
    probs = []
    if int(h1.get('NSIDE', -1)) != n or str(h1.get('ORDERING')) != 'RING' or L.size != 12 * n * n:
        probs.append(f'NSIDE {h1.get("NSIDE")} ORDERING {h1.get("ORDERING")} pixels {L.size}')
    t = T[np.isfinite(T)]
    if t.size == 0 or np.any(t < -1e-6) or np.any(t > 1 + 1e-6):
        probs.append(f'TAPERONLY range {(float(t.min()), float(t.max())) if t.size else None}')
    if not np.isfinite(L).any():
        probs.append('LEVEL nowhere finite')
    if not probs:
        north = MT._north_mask(n, C.mosaic_coadd.DEC_FLOOR_DEG)
        s = int(np.isfinite(L[~north]).sum() + np.isfinite(T[~north]).sum())
        if s:
            probs.append(f'{s} finite values south of Dec {C.mosaic_coadd.DEC_FLOOR_DEG}')
    lv = (float(np.nanmin(L)), float(np.nanmax(L))) if np.isfinite(L).any() else None
    return not probs, lv, '; '.join(probs) or f'NSIDE {n}, LEVEL {lv[0]:.3f}..{lv[1]:.3f}, TAPERONLY in [0, 1]'


@content_test('mosaic_caveat_table')
def caveat_rows(path, n_tiles=None, **ctx):
    """One row per staged tile, each a hybrid with LSCAVEAT and the five LSLVL columns."""
    import csv
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    probs = []
    if n_tiles is not None and len(rows) != int(n_tiles):
        probs.append(f'{len(rows)} rows != {n_tiles} tiles')
    cols = list(rows[0]) if rows else []
    lvl = [c for c in cols if c.startswith('LSLVL')]
    if len(lvl) != len(_cfg().hybrid_recon.DEC_BANDS) or 'LSCAVEAT' not in cols:
        probs.append(f'columns {cols}')
    nh = [r['lst'] for r in rows if r.get('kind') != 'hybrid']
    if nh:
        probs.append(f'not hybrid: {nh}')
    return not probs, len(rows), '; '.join(probs) or f'{len(rows)} hybrid tiles'
