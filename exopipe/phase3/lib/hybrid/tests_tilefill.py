"""Content tests of the stage `tilefill` (spec plan/specs/hybrid.md 3.5: tilefill_{identity, blanking, compact, roundtrip},
cleanres_rename_md5, card_superset; registered with core.validate). Each reads the file's content with ordinary reads (memmap=False).

Classes (pull request PR-5):
  hybrid_filled_master     a filled hybrid master or beam-corrected sibling: PRODNAME = the file name, a phase3 name with lsrecon +
                           lsfill (no cleanres); the fill cards (TPFILL true, TPMOSAIC the full name, TPSRCMD5 = md5 of its unfilled
                           source); the header validates; card_superset (D7)
  hybrid_cleanres_renamed  the CLEAN residual renamed without lsfill: PRODNAME = the file name, cleanres without lsfill; data identical
                           to its source's (cleanres_rename_md5); card_superset (D7)
  hybrid_tilefill_record   <w>_tilefill.json: the unit; per file, the production tests re-evaluated against the config limits
                           (identity <= 1, blanking unchanged, compact sources or the ruled caveat of hybrid_tilefill.TPCAVEAT_UNITS,
                           round trip of the beam-corrected file); every product's md5 equal to the file's
"""
import hashlib
import json
import os

import numpy as np

from exopipe.phase3.core.validate import content_test

FILL_CARDS = ('TPFILL', 'TPDONOR', 'TPDONMHZ', 'TPSCALE', 'TPLLO', 'TPLHI', 'TPMOSAIC', 'TPMOSMD5', 'TPSRCMD5', 'TPTILE', 'TPOFFSKY')


def _header(path):
    from astropy.io import fits
    with fits.open(path, memmap=False) as h:
        return h[0].header.copy()


def _data_md5(path):
    from astropy.io import fits
    with fits.open(path, memmap=False) as h:
        return hashlib.md5(np.ascontiguousarray(h[0].data).tobytes()).hexdigest()


def _name(path, want_proc, bad_proc):
    from exopipe.phase3.core.products import parse_name
    h = _header(path)
    pn = parse_name(str(h.get('PRODNAME', ''))) if h.get('PRODNAME') else {}
    proc = set(pn.get('proc') or ())
    bad = []
    if h.get('PRODNAME') != os.path.basename(path):
        bad.append(f'PRODNAME {h.get("PRODNAME")!r} is not the file name')
    if pn.get('grammar') != 'phase3':
        bad.append('not a phase3 name')
    if not want_proc <= proc or proc & bad_proc:
        bad.append(f'proc {sorted(proc)}')
    return not bad, h.get('PRODNAME'), '; '.join(bad) if bad else f'phase3 name, proc {sorted(proc)}'


def card_superset(path, replaces=None, **ctx):
    """D7: every card key of the installed file this one would replace is present (commentary cards aside)."""
    if not replaces:
        return True, None, 'replaces no installed file'
    have = set(_header(path).keys())
    old = {k for k in _header(replaces).keys() if k not in ('HISTORY', 'COMMENT', '')}
    miss = sorted(old - have)
    return not miss, len(miss), (f'missing {miss[:8]} of {os.path.basename(replaces)}' if miss else
                                 f'holds every card key of {os.path.basename(replaces)}')


@content_test('hybrid_filled_master')
def filled_name(path, kind='plain', **ctx):
    return _name(path, {'lsrecon', 'lsfill'} | ({'pbcorr'} if kind == 'pbcorr' else set()),
                 {'cleanres'} | (set() if kind == 'pbcorr' else {'pbcorr'}))


@content_test('hybrid_filled_master')
def filled_cards(path, src=None, **ctx):
    from exopipe.phase3.lib.install.steps_common import md5
    h = _header(path)
    bad = [k for k in FILL_CARDS if k not in h]
    if h.get('TPFILL') is not True:
        bad.append(f'TPFILL {h.get("TPFILL")!r}')
    mos = str(h.get('TPMOSAIC', ''))
    if not (mos.endswith('.fits') and 'mosaic_jysr' in mos):
        bad.append(f'TPMOSAIC {mos!r} is not a full mosaic name')
    if src and h.get('TPSRCMD5') != md5(src):
        bad.append(f'TPSRCMD5 {h.get("TPSRCMD5")} is not the md5 of {os.path.basename(src)}')
    return not bad, h.get('TPSCALE'), '; '.join(bad) if bad else f'fill cards present, TPSCALE {h.get("TPSCALE")}, TPSRCMD5 = its source'


@content_test('hybrid_filled_master')
def filled_header_valid(path, **ctx):
    from exopipe.phase3.lib.hybrid.tilefill import _header_ok
    ok, errs = _header_ok(path)
    return ok, None, 'standard header valid' if ok else f'errors {errs}'


@content_test('hybrid_filled_master')
def filled_card_superset(path, replaces=None, **ctx):
    return card_superset(path, replaces)


@content_test('hybrid_cleanres_renamed')
def renamed_name(path, **ctx):
    return _name(path, {'lsrecon', 'cleanres'}, {'lsfill'})


@content_test('hybrid_cleanres_renamed')
def cleanres_rename_md5(path, src=None, **ctx):
    if not src:
        return False, None, 'no source in the context'
    a, b = _data_md5(path), _data_md5(src)
    return a == b, a, 'data md5 equal to the source\'s' if a == b else f'data md5 {a} differs from the source\'s {b}'


@content_test('hybrid_cleanres_renamed')
def renamed_card_superset(path, replaces=None, **ctx):
    return card_superset(path, replaces)


def _rules():
    from exopipe.phase3.lib.hybrid import tilefill as TF
    return TF.bind()


def _load(path):
    with open(path) as f:
        return json.load(f)


@content_test('hybrid_tilefill_record')
def tilefill_record_unit(path, lst=None, scope=None, weighting=None, **ctx):
    rec = _load(path)
    bad = [f'{k} {rec.get(k)!r} != {w!r}' for k, w in (('lst', lst), ('scope', scope), ('weighting', weighting)) if rec.get(k) != w]
    if rec.get('result') != 'PASS':
        bad.append(f'result {rec.get("result")!r}')
    return not bad, rec.get('source'), '; '.join(bad) if bad else f'{lst} {scope} {weighting}, source {rec.get("source")}'


def _per_kind(path, fn):
    rec = _load(path)
    bad, vals = [], {}
    for kind in ('pbcorr', 'plain'):
        t = rec.get(kind) or {}
        ok, v, why = fn(rec, kind, t)
        vals[kind] = v
        if not ok:
            bad.append(f'{kind}: {why}')
    return not bad, vals, '; '.join(bad) if bad else json.dumps(vals, default=str)


@content_test('hybrid_tilefill_record')
def tilefill_identity(path, **ctx):
    return _per_kind(path, lambda r, k, t: (isinstance(t.get('identity_rel'), float) and t['identity_rel'] <= 1.0, t.get('identity_rel'),
                                            f'identity {t.get("identity_rel")} (rule <= 1: written - unfilled = the increment)'))


@content_test('hybrid_tilefill_record')
def tilefill_blanking(path, **ctx):
    return _per_kind(path, lambda r, k, t: (t.get('blanking_unchanged') is True, t.get('blanking_unchanged'), 'blanking changed'))


@content_test('hybrid_tilefill_record')
def tilefill_compact(path, **ctx):
    g = _rules()

    def one(r, k, t):
        c = t.get('compact') or {}
        n = c.get('n') or 0
        ok_c = n == 0 or (c.get('median') is not None and c['median'] <= g['COMPACT_MED_MAX'] and c['max'] <= g['COMPACT_MAX'])
        ruled = (r.get('lst'), r.get('scope'), r.get('weighting'), k) in g['TPCAVEAT_UNITS']
        if ok_c:
            return t.get('caveat') is None, c.get('median'), f'caveat {t.get("caveat")!r} recorded although the check passed'
        return ruled and t.get('caveat') == g['TPCAVEAT_TEXT'], c.get('max'), (
            f'compact median {c.get("median")} max {c.get("max")} (rules <= {g["COMPACT_MED_MAX"]} / {g["COMPACT_MAX"]}); '
            f'{"ruled caveat not recorded" if ruled else "not a ruled caveat unit"}')
    return _per_kind(path, one)


@content_test('hybrid_tilefill_record')
def tilefill_roundtrip(path, **ctx):
    g = _rules()

    def one(r, k, t):
        if k != 'pbcorr':
            return True, None, ''
        rt = t.get('roundtrip') or {}
        ok = (rt.get('n_pix') or 0) > 100 and isinstance(rt.get('rel'), float) and rt['rel'] <= g['TILE_RT_MAX']
        return ok, rt.get('rel'), f'round trip {rt} (rule > 100 pixels, rel <= {g["TILE_RT_MAX"]})'
    return _per_kind(path, one)


@content_test('hybrid_tilefill_record')
def tilefill_product_md5(path, **ctx):
    from exopipe.phase3.lib.install.steps_common import md5
    rec = _load(path)
    bad = []
    for k in ('pbcorr', 'plain', 'cleanres'):
        t = rec.get(k)
        if t is None:
            continue
        if not os.path.exists(t['out']) or md5(t['out']) != t['out_md5']:
            bad.append(f'{k} {os.path.basename(t["out"])}')
    return not bad, None, f'md5 differs or missing: {bad}' if bad else 'every product\'s md5 equals the record'
