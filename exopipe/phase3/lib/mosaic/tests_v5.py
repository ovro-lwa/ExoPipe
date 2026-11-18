"""Content tests of the mosaic v5 classes (lib/mosaic/v5.py; spec plan/specs/hybrid.md 2.8, 2.10-2.14). Context keys as the v5 units
pass them (exopipe3 verify re-runs them from the record's ctx):
  mosaic_v5_tiles      scope, weighting
  mosaic_filled_map    the mosaic_map context (tiles, partial, scope, tag, psfmap, jysr_map, jybeam_map, domain) + unfilled (the map
                       it fills) and, for the Jy/beam map, filled_jysr (the Jy/sr map whose fill it carries)
  mosaic_caveat_map    scope
  mosaic_caveat_table  n_tiles
  mosaic_content_report  n_tiles, scope, variant
  mosaic_pkcorr_map    curve (the curve CSV), rounds, scope, variant; mosaic_pkcorr_curve, _recovery: rounds, scope, variant;
                       mosaic_pkcorr_report: map_path, rounds, scope, variant
The numeric tests of the fill (MF test_tpfill, the Jy/beam identity) and of the caveat map (test_caveat_map) need the computed arrays:
the units record them as inline checks (HG19.fill_jysr, HG19.fill_jybeam, test_caveat_map, test_pkcorr, TEST1, TEST2)."""
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


# ---- mosaic_content_report (content/content_I_deep_<V>_<S>_lsfill.json) -----------------------------------------------------------

@content_test('mosaic_content_report')
def content_passes(path, n_tiles=None, **ctx):
    """TEST 1 and TEST 2 of both domains (MT run 108, provenance 95), re-judged from the numbers with mosaic_content's limits: >=
    MIN_SOURCES sources, |median ratio - 1| <= MED_TOL, a fraction >= FRAC5_MIN within 5 %, one link per tile, every input a link to
    a staged tile."""
    C = _cfg().mosaic_content
    with open(path) as f:
        rep = json.load(f)
    probs = []
    for dom in ('jysr', 'jybeam'):
        r = rep.get(dom)
        if not r:
            probs.append(f'{dom}: missing')
            continue
        med, fr, n = r.get('median_ratio'), r.get('frac_within_5pct'), r.get('n_sources') or 0
        ok1 = (n >= int(C.MIN_SOURCES) and med is not None and abs(med - 1) <= float(C.MED_TOL) and fr is not None
               and fr >= float(C.FRAC5_MIN))
        if not ok1 or r.get('result') != 'PASS':
            probs.append(f'{dom} TEST 1: {n} sources, median {med}, within 5 % {fr}, result {r.get("result")}')
        pv = r.get('provenance') or []
        if (n_tiles is not None and len(pv) != int(n_tiles)) or not all(x.get('symlink') for x in pv) \
                or not r.get('provenance_matches_tiles'):
            probs.append(f'{dom} TEST 2: {len(pv)} links, all links {all(x.get("symlink") for x in pv)}, '
                         f'the staged tiles {r.get("provenance_matches_tiles")}')
    return not probs, {d: (rep.get(d) or {}).get('median_ratio') for d in ('jysr', 'jybeam')}, \
        '; '.join(probs) or 'TEST 1 and TEST 2 pass in Jy/sr and Jy/beam'


# ---- PKCORR (pkcorr/: the map, the curve, the recovery rows, the JSON) -----------------------------------------------------------

def _curve_rows(path):
    import csv
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


@content_test('mosaic_pkcorr_curve')
def pkcorr_curve_nodes(path, rounds=None, **ctx):
    """MP test_pkcorr on the written curve: one row per Dec node (mosaic_pkcorr DEC_START..DEC_STOP every DEC_STEP), each with samples
    from >= min(MIN_ROUNDS, rounds) rounds and a median in REC_MIN..REC_MAX."""
    from exopipe.phase3.lib.mosaic.pkcorr import nodes
    C = _cfg().mosaic_pkcorr
    rows = _curve_rows(path)
    cols = list(rows[0]) if rows else []
    if cols != ['dec', 'rec_median', 'rec_mad', 'n', 'n_rounds']:
        return False, cols, f'columns {cols}'
    need = min(int(C.MIN_ROUNDS), int(rounds if rounds is not None else C.ROUNDS))
    have = {round(float(r['dec']), 1): r for r in rows}
    decs, _ra = nodes()
    few = [float(d) for d in decs if round(float(d), 1) not in have or int(have[round(float(d), 1)]['n_rounds']) < need]
    bad = [(r['dec'], r['rec_median']) for r in rows if not float(C.REC_MIN) <= float(r['rec_median']) <= float(C.REC_MAX)]
    probs = ([f'nodes missing or under {need} rounds: {few[:6]}'] if few else []) + ([f'medians out of range: {bad[:6]}'] if bad else [])
    return not probs, len(rows), '; '.join(probs) or f'{len(rows)} nodes, each >= {need} rounds, medians in range'


@content_test('mosaic_pkcorr_map')
def pkcorr_map_is_curve(path, curve=None, rounds=None, **ctx):
    """RING HEALPix at mosaic_coadd.NSIDE, column SIGNAL, BUNIT '', SINJ = mosaic_pkcorr.S_INJ, NROUNDS = the unit's rounds; the map
    equals the curve file interpolated in Dec within the CSV's 4-decimal rounding (5e-5) + MAP_TOL (the unit's inline test_pkcorr
    compares it with the unrounded curve at MAP_TOL)."""
    import healpy as hp
    import numpy as np
    from astropy.io import fits
    C = _cfg()
    n = int(C.mosaic_coadd.NSIDE)
    with fits.open(path, memmap=False) as h:
        h1 = h[1].header
        if 'SIGNAL' not in h[1].columns.names:
            return False, h[1].columns.names, 'no SIGNAL column'
        m = np.asarray(h[1].data['SIGNAL'], dtype=np.float64).ravel()
    probs = []
    if int(h1.get('NSIDE', -1)) != n or str(h1.get('ORDERING')) != 'RING' or m.size != 12 * n * n:
        probs.append(f'NSIDE {h1.get("NSIDE")} ORDERING {h1.get("ORDERING")} pixels {m.size}')
    if str(h1.get('BUNIT', None)) != '' or float(h1.get('SINJ', -1)) != float(C.mosaic_pkcorr.S_INJ) \
            or (rounds is not None and int(h1.get('NROUNDS', -1)) != int(rounds)):
        probs.append(f'BUNIT {h1.get("BUNIT")!r} SINJ {h1.get("SINJ")} NROUNDS {h1.get("NROUNDS")}')
    err = None
    if curve and not probs:
        rows = _curve_rows(curve)
        th, _ = hp.pix2ang(n, np.arange(m.size))
        ref = np.interp(90.0 - np.degrees(th), [float(r['dec']) for r in rows], [float(r['rec_median']) for r in rows])
        err = float(np.max(np.abs(m - ref)))
        if not err <= 5e-5 + float(C.mosaic_pkcorr.MAP_TOL):
            probs.append(f'map vs curve file {err:.2e}')
    if not np.all(np.isfinite(m)):
        probs.append(f'{int((~np.isfinite(m)).sum())} non-finite pixels')
    return not probs, err, '; '.join(probs) or f'NSIDE {n}, map = curve (max {err:.1e})'


@content_test('mosaic_pkcorr_recovery')
def pkcorr_recovery_rounds(path, rounds=None, **ctx):
    """MP 116-119: columns round,dec,ra,ntile,rec; every round 0..rounds-1 present; ntile >= 1."""
    import csv
    with open(path, newline='') as f:
        rd = csv.DictReader(f)
        cols, rows = rd.fieldnames or [], list(rd)
    if cols != ['round', 'dec', 'ra', 'ntile', 'rec']:
        return False, cols, f'columns {cols}'
    want = set(range(int(rounds))) if rounds is not None else None
    got = {int(r['round']) for r in rows}
    probs = ([f'rounds {sorted(got)}'] if want is not None and got != want else []) + \
        ([f'{sum(int(r["ntile"]) < 1 for r in rows)} rows without a tile'] if any(int(r['ntile']) < 1 for r in rows) else [])
    return not probs, len(rows), '; '.join(probs) or f'{len(rows)} rows, rounds {sorted(got)}'


@content_test('mosaic_pkcorr_report')
def pkcorr_report_pass(path, map_path=None, rounds=None, **ctx):
    """MP 139-142: the JSON's test_pkcorr verdict is pass, its rounds are the unit's, and it names the unit's map."""
    with open(path) as f:
        r = json.load(f)
    probs = []
    if (r.get('test_pkcorr') or {}).get('verdict') != 'pass':
        probs.append(f'test_pkcorr {r.get("test_pkcorr")}')
    if rounds is not None and r.get('rounds') != int(rounds):
        probs.append(f'rounds {r.get("rounds")} != {rounds}')
    if map_path is not None and r.get('map') != map_path:
        probs.append(f'map {r.get("map")} != {map_path}')
    return not probs, r.get('rec_median'), '; '.join(probs) or f'test_pkcorr pass, median recovery {r.get("rec_median")}'
