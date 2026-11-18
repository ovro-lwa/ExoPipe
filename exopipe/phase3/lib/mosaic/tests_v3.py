"""Content tests of the mosaic v3 tile list (lib/mosaic/v3.py; spec plan/specs/hybrid.md 2.9-2.13). Context keys as the stage unit
passes them (exopipe3 verify re-runs them from the record's ctx): mosaic_v3_tiles  scope, job; mosaic_content_random_report
n_tiles, scope, variant, job (the Stokes I content report and the PKCORR files use tests_v5's classes)."""
import json
import os

from exopipe.phase3.core.validate import content_test

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
    from exopipe.phase3.lib.mosaic.v5 import _md5
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
