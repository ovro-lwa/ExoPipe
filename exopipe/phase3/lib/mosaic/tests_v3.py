"""Content tests of the mosaic v3 tile list (lib/mosaic/v3.py; spec plan/specs/hybrid.md 2.9-2.13). Context keys as the stage unit
passes them (exopipe3 verify re-runs them from the record's ctx): mosaic_v3_tiles  scope, job; mosaic_content_random_report
n_tiles, scope, variant, job; mosaic_v3_product kind, scope, nstack, thermnoi; mosaic_v3_table columns, n_rows (the Stokes I content
report and the PKCORR files use tests_v5's classes)."""
import json
import os

from exopipe.phase3.core.validate import content_test
from exopipe.phase3.lib.mosaic import tests_v5 as _t5

_KINDS = {'plain': ('plain',), 'ssub': ('plain', 'ssub'), 'vclean': ('vclean',)}


def _cfg():
    from exopipe.phase3.config import CFG
    return CFG


def _tiles(path):
    with open(path) as f:
        return json.load(f)


def _job_kind(t):
    if t.get('product') == 'Vclean':
        return 'vclean'
    return 'ssub' if t.get('tag') else 'plain'


@content_test('mosaic_v3_tiles')
def tiles_complete(path, **ctx):
    """At least mosaic_v3.REQUIRE_N_LSTS tiles, one per LST of 00h-23h, each with its master, md5, weighting map, the weighting map's md5 and
    reference master, and its nights; the tile kinds fit the job (a plain job: plain masters only; a striation-corrected job: at least
    one corrected master; Stokes V: cleaned-V masters only); the scope and job are the unit's."""
    t = _tiles(path)
    rows = t.get('tiles') or []
    n = int(_cfg().mosaic_v3.REQUIRE_N_LSTS)
    every = {f'{h:02d}h' for h in range(24)}
    lsts = [r.get('lst') for r in rows]
    probs = []
    if len(rows) < n or len(set(lsts)) != len(lsts) or not set(lsts) <= every:
        probs.append(f'LSTs {lsts} (at least {n} distinct of 00h-23h)')
    kind = _job_kind(t)
    for r in rows:
        miss = [k for k in ('path', 'md5', 'pb', 'pb_md5', 'pbfor', 'nights', 'kind') if not r.get(k)]
        if miss or r.get('kind') not in _KINDS[kind]:
            probs.append(f'{r.get("lst")}: missing {miss} or kind {r.get("kind")!r} in a {kind} job')
    if kind == 'ssub' and not any(r.get('kind') == 'ssub' for r in rows):
        probs.append('no striation-corrected master in a striation-corrected job')
    for k in ('scope', 'job'):
        if ctx.get(k) is not None and t.get(k) != ctx[k]:
            probs.append(f'{k} {t.get(k)!r} != {ctx[k]!r}')
    return not probs, len(rows), '; '.join(probs[:4]) or f'{len(rows)} tiles, one per LST ({kind})'


@content_test('mosaic_v3_tiles')
def tiles_on_disk(path, **ctx):
    """Every master and weighting map exists with the staged md5 (the files the co-add reads)."""
    from exopipe.phase3.lib.mosaic.gen_common import _md5
    rows = _tiles(path).get('tiles') or []
    bad = [r['lst'] for r in rows if not (os.path.exists(r['path']) and os.path.exists(r['pb']))
           or _md5(r['path']) != r['md5'] or _md5(r['pb']) != r.get('pb_md5')]
    return not bad, len(bad), f'tiles missing or changed: {bad}' if bad else f'{len(rows)} tiles at their staged md5'


# ---- mosaic_content_random_report (content/content_V_deep_Taper_R0_clean_<S>.json) --------------------------------------------------

@content_test('mosaic_content_random_report')
def content_random_passes(path, n_tiles=None, **ctx):
    """TEST 1R and TEST 2 of both domains (MT run_random 157, provenance 95), re-judged from the numbers with mosaic_content's limits:
    >= RAND_MIN_POSITIONS positions, median |map - tiles| / scale <= RAND_MED_MAX, 90th percentile <= RAND_P90_MAX; one link per
    tile, every input a link to a staged tile."""
    C = _cfg().mosaic_content
    with open(path) as f:
        rep = json.load(f)
    probs = []
    for dom in ('jysr', 'jybeam'):
        r = rep.get(dom)
        if not r:
            probs.append(f'{dom}: missing')
            continue
        n, med, p90 = r.get('n_positions') or 0, r.get('median_absdiff_over_scale'), r.get('p90_absdiff_over_scale')
        ok = (n >= int(C.RAND_MIN_POSITIONS) and med is not None and med <= float(C.RAND_MED_MAX) and p90 is not None
              and p90 <= float(C.RAND_P90_MAX))
        if not ok or r.get('result') != 'PASS':
            probs.append(f'{dom} TEST 1R: {n} positions, median {med}, 90th percentile {p90}, result {r.get("result")}')
        pv = r.get('provenance') or []
        if (n_tiles is not None and len(pv) != int(n_tiles)) or not all(x.get('symlink') for x in pv) \
                or not r.get('provenance_matches_tiles'):
            probs.append(f'{dom} TEST 2: {len(pv)} links, all links {all(x.get("symlink") for x in pv)}, '
                         f'the staged tiles {r.get("provenance_matches_tiles")}')
    return not probs, {d: (rep.get(d) or {}).get('median_absdiff_over_scale') for d in ('jysr', 'jybeam')}, \
        '; '.join(probs) or 'TEST 1R and TEST 2 pass in Jy/sr and Jy/beam'


@content_test('mosaic_v3_product')
def product_conformant(path, kind=None, scope=None, nstack=None, thermnoi=None, **ctx):
    """A product conform staged (conform_v3): tests_v5.product_problems (the push audit, PRODNAME in HDU 0 and HDU 1 of a HEALPix map,
    NSTACK PROVENANCE rows, the 'Mosaic <mosaic_v3.ARCHIVE_RUN>' HISTORY line, TIMEFIX, a Jy/beam map's PSFMAP naming a psfmap product),
    and THERMNOI the unit's value (finite, positive, in Jy/beam)."""
    import math
    from exopipe.phase3.lib.mosaic.tests_v5 import product_problems

    def therm(h0):
        v = h0.get('THERMNOI')
        if not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
            return [f'THERMNOI {v!r}']
        # the card holds the value as astropy formats it (16 significant digits)
        return [] if thermnoi is None or abs(v - float(thermnoi)) <= 1e-12 * abs(float(thermnoi)) else [
            f'THERMNOI {v!r} != the unit\'s {thermnoi!r}']
    probs = product_problems(path, kind, nstack, _cfg().mosaic_v3.ARCHIVE_RUN, extra=therm)
    return not probs, len(probs), '; '.join(probs) or 'audit, PRODNAME, PROVENANCE, HISTORY, TIMEFIX, THERMNOI agree'


content_test('mosaic_v3_table')(_t5.table_complete)
#   a FEATHER3 table: its columns are the unit's, it holds n_rows rows, every result PASS (the mosaic_v5_table test)
