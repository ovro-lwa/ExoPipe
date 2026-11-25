"""Content tests of the hybrid route's products (registered with core.validate; spec plan/specs/hybrid.md 3.5). Each reads the
file's content (pixels, header values, record fields against the rules), with ordinary reads (memmap=False).

Classes (pull request PR-2, stage stack):
  hybrid_stack_image   a NoTaper dirty / half stack or a Taper residual stack: the installed master's grid, finite non-zero
                       pixels, marked as an intermediate (LSDIAG) and not a product (no PRODNAME / MSTRGEN)
  hybrid_psf_stack     the weighted mean Phase 1 PSF: the master's grid, peak exactly 1, not a product
  hybrid_stack_record  <v>_stack.json: the unit it names, a rebuild verdict of the rule, the night-drop rules (at most
                       hybrid_stack.MAX_UNDERIVABLE night without a screen; at least max(2, N - MIN_NIGHTS_DROP) nights stacked)
  hybrid_stack_input   an inputs/ copy of an installed file: its md5 equals the installed file's md5 recorded at copy time
"""
import json
import os

import numpy as np

from exopipe.phase3.core.validate import content_test

GRID_CARDS = ('NAXIS1', 'NAXIS2', 'CTYPE1', 'CTYPE2', 'CRVAL1', 'CRVAL2', 'CRPIX1', 'CRPIX2', 'CDELT1', 'CDELT2')


def _read(path):
    from astropy.io import fits
    with fits.open(path, memmap=False) as h:
        return np.squeeze(h[0].data), h[0].header.copy()


def _grid(path, master):
    from astropy.io import fits
    _d, h = _read(path)
    mh = fits.getheader(master)
    bad = [k for k in GRID_CARDS if h.get(k) != mh.get(k)]
    return not bad, len(bad), ('grid cards equal the master\'s ' + os.path.basename(master) if not bad else
                               'differ from the master: ' + ', '.join(f'{k} {h.get(k)!r} vs {mh.get(k)!r}' for k in bad[:4]))


def _not_product(path):
    _d, h = _read(path)
    bad = [k for k in ('PRODNAME', 'MSTRGEN') if k in h]
    ok = bool(h.get('LSDIAG')) and not bad
    return ok, h.get('LSDIAG'), ('LSDIAG intermediate, no product identity' if ok else f'LSDIAG {h.get("LSDIAG")!r}, product cards {bad}')


@content_test('hybrid_stack_image')
def master_grid(path, master=None, **ctx):
    if not master:
        return False, None, 'no master in the context'
    return _grid(path, master)


@content_test('hybrid_stack_image')
def finite_nonzero(path, **ctx):
    d, _h = _read(path)
    f = np.isfinite(d)
    nz = int(np.count_nonzero(d[f]))
    return d.ndim == 2 and nz > 0, round(float(f.mean()), 4), f'{d.shape}, finite fraction {f.mean():.4f}, {nz} non-zero finite pixels'


@content_test('hybrid_stack_image')
def intermediate_not_product(path, **ctx):
    return _not_product(path)


@content_test('hybrid_psf_stack')
def master_grid_psf(path, master=None, **ctx):
    if not master:
        return False, None, 'no master in the context'
    return _grid(path, master)


@content_test('hybrid_psf_stack')
def peak_one(path, **ctx):
    d, _h = _read(path)
    if not np.all(np.isfinite(d)):
        return False, None, 'non-finite pixels in the PSF stack'
    pk = float(d.max())
    j, i = np.unravel_index(int(np.argmax(d)), d.shape)
    return abs(pk - 1.0) < 1e-6, pk, f'peak {pk:.8f} at pixel ({j}, {i}) of {d.shape}'


@content_test('hybrid_psf_stack')
def intermediate_not_product_psf(path, **ctx):
    return _not_product(path)


@content_test('hybrid_stack_record')
def stack_rules(path, lst=None, band=None, weighting=None, **ctx):
    from exopipe.phase3.config import CFG
    from exopipe.phase3.lib.hybrid.common import TAPER_OF
    from exopipe.phase3.lib.hybrid.stack import VERDICTS
    S = CFG.hybrid_stack
    with open(path) as f:
        rec = json.load(f)
    bad = []
    for k, want in (('lst', lst), ('band', band), ('weighting', weighting)):
        if want is not None and rec.get(k) != want:
            bad.append(f'{k} {rec.get(k)!r} != {want!r}')
    verdict = (rec.get('test_stack_rebuild') or {}).get('verdict')
    if verdict not in VERDICTS:
        bad.append(f'rebuild verdict {verdict!r}')
    nights = [n['date'] for n in rec.get('nights', [])]
    stacked = rec.get('nights_stacked', nights)
    dropped = rec.get('dirty_nights_dropped', [])
    if sorted(set(nights) - set(dropped)) != sorted(stacked):
        bad.append(f'nights {len(nights)} - dropped {dropped} != stacked {len(stacked)}')
    if len(stacked) < max(2, len(nights) - S.MIN_NIGHTS_DROP):
        bad.append(f'{len(stacked)} nights stacked < max(2, {len(nights)} - {S.MIN_NIGHTS_DROP})')
    raw = rec.get('nights_raw_admitted') or []
    if raw and (rec.get('lst'), rec.get('band')) not in S.RAW_NIGHT_ADMIT:
        bad.append(f'nights entered unaligned {raw} outside hybrid_stack.RAW_NIGHT_ADMIT')
    if not set(raw) <= set(stacked):
        bad.append(f'nights entered unaligned {sorted(set(raw) - set(stacked))} are not among the nights stacked')
    if len(rec.get('nights_dropped_no_screen', {})) > S.MAX_UNDERIVABLE:
        bad.append(f'{len(rec["nights_dropped_no_screen"])} nights without a screen > {S.MAX_UNDERIVABLE}')
    if rec.get('weighting') in TAPER_OF and not rec.get('inputs'):
        bad.append('a NoTaper stack record without its inputs/ md5s')
    return not bad, verdict, '; '.join(bad) if bad else f'{len(stacked)} of {len(nights)} nights stacked, {verdict}'


@content_test('hybrid_stack_input')
def md5_equals_installed(path, md5=None, installed=None, **ctx):
    from exopipe.phase3.lib.install.steps_common import md5 as _md5
    if not md5:
        return False, None, 'no recorded md5 in the context'
    got = _md5(path)
    return got == md5, got, f'md5 {got} ' + ('==' if got == md5 else '!=') + f' {md5} of {installed} at copy time'


# ------------------------------------------------------------------------------------------- pull request PR-3: recon, wideband
# Classes:
#   hybrid_recon_master   the reconstruction (apparent, Jy/beam), subband or wideband: a phase3 product name with lsrecon and
#                         without pbcorr naming the unit's scope and robust, the reconstruction cards, the installed master's
#                         grid, finite non-zero pixels
#   hybrid_recon_pbcorr   its split beam-corrected sibling: the split identity recomputed from the files (NL 1881), PBMETHOD,
#                         PBFLOOR, the name with pbcorr
#   hybrid_recon_work     a work file (model, residuals, restored point components): the master's grid, not a product
#   hybrid_recon_record   <w>_lsrecon.json: the unit, and each of production's record tests re-evaluated against the config
#                         rules (prior back-conversion, PSF2x core and md5, prior fit, CLEAN and step convergence, compact
#                         sources, large-scale level)
#   hybrid_wideband_record <w>_lsrecon_wb.json: the members, the weights, the identity, compact sources, level

def _recon_rules():
    from exopipe.phase3.lib.hybrid import recon as REC
    return REC, REC.bind()


def _refuses(fn, *a, **k):
    """None when production's test function passes on the recorded values, else its refusal text."""
    from exopipe.phase3.lib.hybrid.recon import ReconRefusal
    try:
        fn(*a, **k)
        return None
    except ReconRefusal as e:
        return str(e)[:300]


@content_test('hybrid_recon_master')
def recon_name(path, scope=None, weighting=None, **ctx):
    from exopipe.phase3.core.products import parse_name
    from exopipe.phase3.lib.hybrid.common import VARIANT
    _d, h = _read(path)
    pn = parse_name(str(h.get('PRODNAME', '')))
    bad = []
    if pn.get('grammar') != 'phase3':
        bad.append(f'PRODNAME {h.get("PRODNAME")!r} is not a phase3 name')
    else:
        proc = set(pn['proc'])
        for k, want in (('scope', scope), ('taper', 'NoTaper'), ('robust', VARIANT[weighting][2] if weighting else None)):
            if want is not None and pn.get(k) != want:
                bad.append(f'{k} {pn.get(k)!r} != {want!r}')
        if 'lsrecon' not in proc or 'pbcorr' in proc or 'lsfill' in proc:
            bad.append(f'proc {sorted(proc)}')
    return not bad, h.get('PRODNAME'), '; '.join(bad) if bad else 'phase3 name, lsrecon, apparent'


@content_test('hybrid_recon_master')
def recon_cards(path, scope=None, wideband=False, members=None, **ctx):
    REC, g = _recon_rules()
    _d, h = _read(path)
    bad = []
    if h.get('LSRECON') is not True:
        bad.append(f'LSRECON {h.get("LSRECON")!r}')
    if wideband:
        if h.get('MSTRGEN') != 'lsrecon-sbcomb-v1' or h.get('COADDOP') != 'lsrecon-sbcomb-v1':
            bad.append(f'MSTRGEN {h.get("MSTRGEN")!r} COADDOP {h.get("COADDOP")!r}')
        if members is not None and str(h.get('LSWBSB')) != ','.join(members):
            bad.append(f'LSWBSB {h.get("LSWBSB")!r} != {",".join(members)}')
    else:
        if h.get('LSRGEN') != g['RECON_GEN']:
            bad.append(f'LSRGEN {h.get("LSRGEN")!r} != {g["RECON_GEN"]!r}')
        if h.get('LSSCALES') != ','.join(str(s) for s in g['SCALES']):
            bad.append(f'LSSCALES {h.get("LSSCALES")!r}')
        if h.get('LSCLNRGN') not in ('horizon', 'elev10'):
            bad.append(f'LSCLNRGN {h.get("LSCLNRGN")!r}')
        lo, hi = g['PRIOR_SCALE_RANGE']
        if not (isinstance(h.get('LSPSCALE'), float) and lo < h['LSPSCALE'] < hi):
            bad.append(f'LSPSCALE {h.get("LSPSCALE")!r} outside ({lo}, {hi})')
    if scope in g['CAVEAT_SCOPES'] and 'LSCAVEAT' not in h:
        bad.append('no LSCAVEAT card in a caveat scope')
    return not bad, h.get('LSPSCALE', h.get('LSWBW')), '; '.join(bad) if bad else 'reconstruction cards of the rule'


@content_test('hybrid_recon_master')
def recon_grid(path, master=None, **ctx):
    if not master:
        return False, None, 'no master in the context'
    return _grid(path, master)


@content_test('hybrid_recon_master')
def recon_finite(path, **ctx):
    return finite_nonzero(path)


@content_test('hybrid_recon_pbcorr')
def pbcorr_split(path, apparent=None, compact=None, pbs=None, **ctx):
    """NL 1881 recomputed from the files: P x PBD = A + C x (PBD / PBF - 1) to 1e-5 of the largest value where the full beam
    exceeds the floor (PBF = the weighted full primary beams, PBD = the weighted diffuse beams), NaN elsewhere."""
    from exopipe.phase3.core.products import parse_name
    from exopipe.phase3.lib.hybrid.common import rd
    from exopipe.phase3.lib.masters import pb_apply as PBA
    from astropy.io import fits
    if not (apparent and compact and pbs):
        return False, None, 'apparent, compact and pbs are needed in the context'
    P, h = _read(path)
    P = P.astype(np.float64)
    A, C = rd(apparent), rd(compact)
    PBF = 0.0
    PBD = 0.0
    for pbf, band, w in pbs:
        ph = fits.getheader(pbf)
        pb = rd(pbf)
        ch, _ = PBA.chromatic_gain(ph, pb.shape, float(PBA.cfg.SUBBAND_FREQ_HZ[band]['center']), float(ph['PBBMAJ']))
        PBF = PBF + w * pb
        PBD = PBD + w * (pb / ch)
    valid = np.isfinite(PBF) & (PBF > PBA.PB_FLOOR) & np.isfinite(A) & np.isfinite(PBD) & (PBD > 0)
    lhs = P[valid] * PBD[valid]
    rhs = A[valid] + C[valid] * (PBD[valid] / PBF[valid] - 1)
    err = float(np.max(np.abs(lhs - rhs)) / np.max(np.abs(rhs))) if valid.any() else None
    nan_out = bool(np.all(np.isnan(P[~valid])))
    pn = parse_name(str(h.get('PRODNAME', '')))
    bad = []
    if err is None or not err < 1e-5:
        bad.append(f'split identity max rel err {err}')
    if not nan_out:
        bad.append('finite pixels where the full beam is at or below the floor')
    if h.get('PBMETHOD') != 'lsrecon-split' or h.get('PBFLOOR') != PBA.PB_FLOOR:
        bad.append(f'PBMETHOD {h.get("PBMETHOD")!r} PBFLOOR {h.get("PBFLOOR")!r}')
    if pn.get('grammar') != 'phase3' or not {'lsrecon', 'pbcorr'} <= set(pn.get('proc', ())):
        bad.append(f'PRODNAME {h.get("PRODNAME")!r}')
    return not bad, err, '; '.join(bad) if bad else f'split identity to {err:.2g}, NaN below the floor ({int(valid.sum())} valid px)'


@content_test('hybrid_recon_work')
def recon_work_grid(path, master=None, **ctx):
    if not master:
        return False, None, 'no master in the context'
    return _grid(path, master)


@content_test('hybrid_recon_work')
def recon_work_not_product(path, **ctx):
    return _not_product(path)


def _load(path):
    with open(path) as f:
        return json.load(f)


def _product_md5s(rec, keys):
    """Problems with the products a record names: each must exist with the md5 the record holds."""
    from exopipe.phase3.lib.install.steps_common import md5
    bad = []
    for k in keys:
        p = rec.get(k) or {}
        if not p.get('md5') or not p.get('path'):
            bad.append(f'no {k} md5')
        elif not os.path.exists(p['path']):
            bad.append(f'{k} {p["path"]} missing')
        elif md5(p['path']) != p['md5']:
            bad.append(f'{k} {os.path.basename(p["path"])} md5 differs from the record')
    return bad


@content_test('hybrid_recon_record')
def recon_unit(path, lst=None, band=None, weighting=None, **ctx):
    _REC, g = _recon_rules()
    rec = _load(path)
    bad = [f'{k} {rec.get(k)!r} != {w!r}' for k, w in (('lst', lst), ('band', band), ('weighting', weighting)) if rec.get(k) != w]
    if rec.get('generation') != g['RECON_GEN']:
        bad.append(f'generation {rec.get("generation")!r}')
    bad += _product_md5s(rec, ('product', 'product_pbcorr'))
    return not bad, rec.get('generation'), '; '.join(bad) if bad else f'{lst} {band} {weighting}, {g["RECON_GEN"]}'


@content_test('hybrid_recon_record')
def prior_backconversion(path, **ctx):
    t = (_load(path).get('prior') or {}).get('test_prior_backconversion') or {}
    r = t.get('median_ratio')
    ok = t.get('verdict') == 'pass' and r is not None and abs(r - 1) < 1e-6
    return ok, r, f'median K / K ratio {r} (rule |r - 1| < 1e-6)'


@content_test('hybrid_recon_record')
def psf2x_core(path, band=None, weighting=None, **ctx):
    from exopipe.phase3.config import CFG
    rec = _load(path)
    t = rec.get('test_psf2x') or {}
    ent = CFG.hybrid_psf2x.REGISTRY.get((band, weighting))
    bad = []
    if not (t.get('verdict') == 'pass' and isinstance(t.get('core_rel_rms'), float) and t['core_rel_rms'] < 0.2):
        bad.append(f'test_psf2x {t}')
    if ent is None or rec.get('psf2x') != ent[0] or rec.get('psf2x_md5') != ent[1]:
        bad.append(f'PSF2x {rec.get("psf2x")} md5 {rec.get("psf2x_md5")} is not the registry entry {ent and ent[:2]}')
    return not bad, t.get('core_rel_rms'), '; '.join(bad) if bad else f'core rel rms {t["core_rel_rms"]:.3f} < 0.2, registry md5'


@content_test('hybrid_recon_record')
def prior_fit(path, **ctx):
    REC, _g = _recon_rules()
    f = _load(path).get('prior_fit') or {}
    why = _refuses(REC.test_prior_fit, f) if {'scale', 'rms', 'data_rms'} <= set(f) else 'no prior_fit values'
    return why is None, f.get('scale'), why or f'scale {f["scale"]:.4f}, rms {f["rms"]:.4g} vs data {f["data_rms"]:.4g}'


@content_test('hybrid_recon_record')
def clean_converged(path, **ctx):
    REC, _g = _recon_rules()
    c = _load(path).get('clean') or {}
    need = {'residual_rms_test_region', 'dirty_rms_test_region'}
    why = _refuses(REC.test_clean_converged, c) if need <= set(c) else 'no CLEAN values'
    return why is None, c.get('residual_rms_test_region'), why or (f"residual {c['residual_rms_test_region']:.4g} < dirty "
                                                                   f"{c['dirty_rms_test_region']:.4g} and < 2 Jy/beam")


@content_test('hybrid_recon_record')
def step_converged(path, **ctx):
    REC, _g = _recon_rules()
    s = _load(path).get('step') or {}
    why = _refuses(REC.test_step_converged, s) if 'cg_relative_residual' in s else 'no step values'
    return why is None, s.get('cg_relative_residual'), why or f"CG relative residual {s['cg_relative_residual']:.3g}"


def _compact(rec, scope):
    _REC, g = _recon_rules()
    t = rec.get('test_compact_sources') or {}
    tol = g['COMPACT_TOL_LOW'] if scope in g['COMPACT_LOW_SCOPES'] else g['COMPACT_TOL']
    ok = (isinstance(t.get('n'), int) and t['n'] >= g['COMPACT_N_MIN'] and isinstance(t.get('median'), float)
          and abs(t['median'] - 1) < tol)
    return ok, t.get('median'), f"median {t.get('median')} over {t.get('n')} sources (rule n >= {g['COMPACT_N_MIN']}, |m - 1| < {tol})"


def _level(rec, scope):
    REC, _g = _recon_rules()
    t = rec.get('test_large_scale_level')
    if t is None:
        return False, None, 'no large-scale level in the record'
    why = _refuses(REC._level_verdict, t.get('ratio_by_dec_band') or {}, scope)
    if why is None:
        again = REC._level_verdict(t['ratio_by_dec_band'], scope)
        if again['verdict'] != t.get('verdict') or not t['ratio_by_dec_band']:
            why = f"recorded verdict {t.get('verdict')!r}, rule gives {again['verdict']!r} on {t['ratio_by_dec_band']}"
    return why is None, t.get('verdict'), why or f"{t['verdict']}: {t['ratio_by_dec_band']}"


@content_test('hybrid_recon_record')
def compact_sources(path, band=None, **ctx):
    return _compact(_load(path), band)


@content_test('hybrid_recon_record')
def large_scale_level(path, band=None, **ctx):
    return _level(_load(path), band)


@content_test('hybrid_wideband_record')
def wideband_unit(path, lst=None, scope=None, weighting=None, members=None, **ctx):
    rec = _load(path)
    bad = [f'{k} {rec.get(k)!r} != {w!r}' for k, w in (('lst', lst), ('scope', scope), ('weighting', weighting)) if rec.get(k) != w]
    subs = rec.get('subbands') or []
    if members is not None and [s.get('band') for s in subs] != list(members):
        bad.append(f'subbands {[s.get("band") for s in subs]} != {list(members)}')
    w = [s.get('weight') for s in subs]
    if not (w and all(isinstance(x, float) and x > 0 for x in w) and abs(sum(w) - 1) < 1e-9):
        bad.append(f'weights {w}')
    bad += _product_md5s(rec, ('product', 'product_pbcorr'))
    return not bad, w, '; '.join(bad) if bad else f'{len(subs)} members, weights sum to 1, products as recorded'


@content_test('hybrid_wideband_record')
def wideband_identity(path, **ctx):
    t = _load(path).get('test_wideband_identity') or {}
    ok = (t.get('verdict') == 'pass' and isinstance(t.get('max_rel_err'), float) and t['max_rel_err'] < 1e-6
          and t.get('nan_masks_equal') is True and t.get('prodname_ok') is True)
    return ok, t.get('max_rel_err'), f'{t}'


@content_test('hybrid_wideband_record')
def wideband_compact_sources(path, scope=None, **ctx):
    return _compact(_load(path), scope)


@content_test('hybrid_wideband_record')
def wideband_large_scale_level(path, scope=None, **ctx):
    _REC, g = _recon_rules()
    rec = _load(path)
    if scope not in g['CAVEAT_SCOPES']:
        ok = 'test_large_scale_level' not in rec
        return ok, None, 'no wideband level outside the caveat scopes (production records it only there)' if ok else 'unexpected level'
    return _level(rec, scope)


# ------------------------------------------------------------------------------------------- pull request PR-4: hybrid, full
# Classes:
#   hybrid_master_product  a hybrid product (the hybrid, its beam-corrected sibling, its CLEAN residual): a phase3 name with
#                          lsrecon + lsfill naming the unit's scope, Taper and robust, the kind's token (none / pbcorr / cleanres),
#                          the Taper master's grid, finite non-zero pixels, the caveat cards in the caveat scopes (the hybrid)
#   hybrid_pbdiffuse       a cell's diffuse PB (Full: the full stage; others: tilefill, work source): the grid, not a product,
#                          finite positive values, a clean PB-map header (D6)
#   hybrid_record          <w>_hybrid.json: the unit; test_hybrid re-evaluated against the config limits (identity, NaN mask,
#                          large scales, small scales or a caveat only in the caveat scopes, compact sources); the beam-corrected
#                          identity; the CLEAN residual read back or a recorded reason; every product's md5 equal to the file's

def _hybrid_rules():
    from exopipe.phase3.lib.hybrid import master as M
    return M.bind()


@content_test('hybrid_master_product')
def hybrid_name(path, scope=None, weighting=None, kind='hybrid', **ctx):
    from exopipe.phase3.core.products import parse_name
    from exopipe.phase3.lib.hybrid.common import VARIANT
    _d, h = _read(path)
    pn = parse_name(str(h.get('PRODNAME', '')))
    if pn.get('grammar') != 'phase3':
        return False, h.get('PRODNAME'), f'PRODNAME {h.get("PRODNAME")!r} is not a phase3 name'
    proc = set(pn['proc'])
    want = {'pbcorr': {'pbcorr'}, 'cleanres': {'cleanres'}}.get(kind, set())
    bad = [f'{k} {pn.get(k)!r} != {w!r}' for k, w in (('scope', scope), ('taper', 'Taper'),
                                                       ('robust', VARIANT[weighting][2] if weighting else None)) if w is not None and pn.get(k) != w]
    if not {'lsrecon', 'lsfill'} <= proc or (proc & {'pbcorr', 'cleanres'}) != want:
        bad.append(f'proc {sorted(proc)} for a {kind}')
    return not bad, h.get('PRODNAME'), '; '.join(bad) if bad else f'phase3 name, lsrecon + lsfill, {kind}'


@content_test('hybrid_master_product')
def hybrid_grid(path, taper=None, **ctx):
    if not taper:
        return False, None, 'no Taper master in the context'
    return _grid(path, taper)


@content_test('hybrid_master_product')
def hybrid_finite(path, **ctx):
    return finite_nonzero(path)


@content_test('hybrid_master_product')
def hybrid_caveat_cards(path, scope=None, kind='hybrid', **ctx):
    g = _hybrid_rules()
    _d, h = _read(path)
    if kind != 'hybrid' or scope not in g['CAVEAT_SCOPES']:
        return True, None, 'no caveat cards owed'
    ok = 'LSCAVEAT' in h and isinstance(h.get('LSSMALL'), float)
    return ok, h.get('LSSMALL'), f'LSCAVEAT {h.get("LSCAVEAT")!r}, LSSMALL {h.get("LSSMALL")!r}'


@content_test('hybrid_pbdiffuse')
def pbdiffuse_grid(path, taper=None, **ctx):
    if not taper:
        return False, None, 'no Taper master in the context'
    return _grid(path, taper)


@content_test('hybrid_pbdiffuse')
def pbdiffuse_values(path, **ctx):
    """Finite somewhere, a positive peak of at most 1 (a primary beam). Small negative values are kept: the installed PB maps
    carry them near the horizon (46 MHz Robust -0.75 down to -0.005; 18h Full's diffuse map -0.0019, production and v2 alike)."""
    d, _h = _read(path)
    f = np.isfinite(d)
    ok = bool(f.any()) and 0 < float(np.nanmax(d)) <= 1 + 1e-6
    return ok, float(np.nanmax(d)) if f.any() else None, f'finite fraction {f.mean():.4f}, range {np.nanmin(d) if f.any() else None} .. {np.nanmax(d) if f.any() else None}'


@content_test('hybrid_pbdiffuse')
def pbdiffuse_not_product(path, **ctx):
    return _not_product(path)


@content_test('hybrid_pbdiffuse')
def pbdiffuse_header(path, **ctx):
    """Spec D6: no SH* card, no master flux or noise card, BUNIT blank (lib/hybrid/common.pbdiffuse_header_problems; rulings
    R-GREGG-20261116-PB-MAPS-SH-CARDS-REMOVE-BUNIT-BLANK-GO, R-GREGG-20261118-HYBRID-Q1-Q3-GO-WITH-RECOMMENDATIONS)."""
    from exopipe.phase3.lib.hybrid.common import pbdiffuse_header_problems
    _d, h = _read(path)
    bad = pbdiffuse_header_problems(h)
    return not bad, len(bad), '; '.join(bad) or 'no SH* or master card, BUNIT blank'


@content_test('hybrid_record')
def hybrid_record_unit(path, lst=None, scope=None, weighting=None, **ctx):
    rec = _load(path)
    bad = [f'{k} {rec.get(k)!r} != {w!r}' for k, w in (('lst', lst), ('scope', scope), ('weighting', weighting)) if rec.get(k) != w]
    return not bad, None, '; '.join(bad) if bad else f'{lst} {scope} {weighting}'


@content_test('hybrid_record')
def hybrid_test(path, scope=None, **ctx):
    g = _hybrid_rules()
    t = _load(path).get('test_hybrid') or {}
    small_ok = isinstance(t.get('small_scale_rel'), float) and t['small_scale_rel'] < g['HYB_SMALL_LIMIT']
    bad = []
    if not (isinstance(t.get('max_rel_err'), float) and t['max_rel_err'] < g['IDENTITY']):
        bad.append(f"identity {t.get('max_rel_err')} (rule < {g['IDENTITY']})")
    if t.get('nan_mask_ok') is not True:
        bad.append('NaN mask')
    if not (isinstance(t.get('large_scale_rel'), float) and t['large_scale_rel'] < g['LARGE_REL']):
        bad.append(f"large scales {t.get('large_scale_rel')} (rule < {g['LARGE_REL']})")
    if not small_ok and scope not in g['CAVEAT_SCOPES']:
        bad.append(f"small scales {t.get('small_scale_rel')} (rule < {g['HYB_SMALL_LIMIT']} outside the caveat scopes)")
    if t.get('verdict') != ('pass' if small_ok else 'caveat'):
        bad.append(f"verdict {t.get('verdict')!r}")
    c = t.get('test_compact_sources') or {}
    ok_c, _m, det = _compact(t, scope)
    if not ok_c:
        bad.append(f'compact sources: {det}')
    return not bad, t.get('verdict'), '; '.join(bad) if bad else (f"identity {t['max_rel_err']:.2g}, large {t['large_scale_rel']:.2g}, "
                                                                 f"small {t['small_scale_rel']:.3g} ({t['verdict']}), compact {c['median']:.4f}")


@content_test('hybrid_record')
def hybrid_product_tests(path, **ctx):
    """The tests the record holds for its products: the beam-corrected identity passed; the CLEAN residual was read back, or the
    record says why there is none."""
    g = _hybrid_rules()
    rec = _load(path)
    bad = []
    t = (rec.get('product_pbcorr') or {}).get('test_hybrid_pbcorr') or {}
    if not (t.get('verdict') == 'pass' and isinstance(t.get('max_rel_err'), float) and t['max_rel_err'] < g['IDENTITY']):
        bad.append(f'test_hybrid_pbcorr {t}')
    if rec.get('product_cleanres'):
        r = rec['product_cleanres'].get('test_readback')
        if not (r and r.get('verdict') == 'pass'):
            bad.append(f'cleanres read-back {r}')
    elif not rec.get('cleanres_skipped'):
        bad.append('no product_cleanres and no recorded reason')
    return not bad, None, '; '.join(bad) if bad else 'pbcorr identity passed; CLEAN residual read back or its absence explained'


@content_test('hybrid_record')
def hybrid_products(path, **ctx):
    """Every product the record names exists with the recorded md5."""
    rec = _load(path)
    bad = _product_md5s(rec, ('product', 'product_pbcorr') + (('product_cleanres',) if rec.get('product_cleanres') else ()))
    return not bad, None, '; '.join(bad) if bad else 'products as recorded'
