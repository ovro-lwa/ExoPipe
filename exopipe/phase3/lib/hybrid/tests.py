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
