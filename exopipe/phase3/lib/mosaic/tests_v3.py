"""Content tests of the mosaic v3 tile list (lib/mosaic/v3.py; spec plan/specs/hybrid.md 2.9-2.13). Context keys as the stage unit
passes them (exopipe3 verify re-runs them from the record's ctx): mosaic_v3_tiles  scope, job."""
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
    """One tile per LST 00h-23h (mosaic_v3.REQUIRE_N_LSTS), each with its master, md5, weighting map, the weighting map's md5 and
    reference master, and its nights; the tile kinds fit the job (a plain job: plain masters only; a striation-corrected job: at least
    one corrected master; Stokes V: cleaned-V masters only); the scope and job are the unit's."""
    t = _tiles(path)
    rows = t.get('tiles') or []
    n = int(_cfg().mosaic_v3.REQUIRE_N_LSTS)
    want = [f'{h:02d}h' for h in range(24)][:n]
    lsts = [r.get('lst') for r in rows]
    probs = []
    if len(rows) < n or len(set(lsts)) != len(lsts) or sorted(lsts) != want:
        probs.append(f'LSTs {lsts}')
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
