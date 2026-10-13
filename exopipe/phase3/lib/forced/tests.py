"""Content tests of the deep forced product classes (spec forced.md 6; refactor chat decision 6: the family-A measurement
classes belong to `photometry`). Each test reads the file's CONTENT and recomputes what it can with code of its own; context
the step passes (``**ctx``) sharpens a test, and every test also runs from the path alone (``exopipe3 verify <class>
<path>``, ``--resume``).

Classes: catalog_forced_columns (family B, stage catflux), forced_pages (family A pages, stage panels).
Snapshot classes: lib/forced/tests_snapshot.py.
"""
import csv
import json
import math
import os
import random
import re

from exopipe.phase3.core.validate import content_test


def _cfg():
    from exopipe.phase3.config import CFG
    return CFG


def _fail_detail(bad, n=5):
    return '; '.join(str(b) for b in bad[:n]) + (' ... (%d)' % len(bad) if len(bad) > n else '')


def _num(s):
    try:
        v = float(s)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _read_csv(path):
    with open(path, newline='') as fh:
        rd = csv.reader(fh)
        header = next(rd, [])
        rows = [dict(zip(header, r)) for r in rd]
    return header, rows


# ---------------------------------------------------------------------------------------------------------------------
# catalog_forced_columns (family B: the survivors catalogues with forced_*)
# ---------------------------------------------------------------------------------------------------------------------

_METHOD_RE = re.compile(r'^fixed-gauss at catalog position, beam \d+\.\dx\d+\.\d arcmin from frame header$')


def fixed_gauss(frame, ra, dec):
    """Least-squares amplitude of a Gaussian with the frame header's beam centred on (ra, dec) over a stamp of
    max(6, 1.5 BMAJ) px (A = sum(m d) / sum(m m)); None off the frame, at an edge, on a non-finite stamp or without a
    beam - recomputed here for the content test with the definition of p3s3_forced_phot.py:45-67, INCLUDING its beam
    orientation (major axis at angle BPA from the pixel x axis; see the impl note's finding on the 90 deg rotation)."""
    import numpy as np
    from astropy.io import fits
    from astropy.wcs import WCS
    cf = _cfg().forced_catflux
    with fits.open(frame, memmap=True) as hd:
        h = hd[0].header
        d = hd[0].data
        while d.ndim > 2:
            d = d[0]
        w = WCS(h).celestial
        x, y = (float(np.asarray(v)) for v in w.world_to_pixel_values(ra, dec))
        if not (np.isfinite(x) and np.isfinite(y)):
            return None
        cd = abs(float(h.get('CDELT2') or h.get('CDELT1') or 0)) or abs(float(w.wcs.cdelt[1]))
        bmaj, bmin = float(h.get('BMAJ') or 0) / cd, float(h.get('BMIN') or 0) / cd
        if bmaj <= 0 or bmin <= 0:
            return None
        bpa = math.radians(float(h.get('BPA') or 0.0))
        r = int(max(cf.STAMP_RADIUS_MIN_PX, round(cf.STAMP_RADIUS_BEAMS * bmaj)))
        xi, yi = int(round(x)), int(round(y))
        if not (r <= yi < d.shape[0] - r and r <= xi < d.shape[1] - r):
            return None
        st = np.asarray(d[yi - r:yi + r + 1, xi - r:xi + r + 1], dtype=np.float32)
        if not np.isfinite(st).all():
            return None
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        X, Y = xx - (x - xi), yy - (y - yi)
        sx, sy = bmaj / cf.FWHM_TO_SIGMA, bmin / cf.FWHM_TO_SIGMA
        xr = X * math.cos(bpa) + Y * math.sin(bpa)
        yr = -X * math.sin(bpa) + Y * math.cos(bpa)
        m = np.exp(-0.5 * ((xr / sx) ** 2 + (yr / sy) ** 2))
        return float(np.sum(m * st) / max(np.sum(m * m), 1e-12))


def _cat_frame(r):
    """(lst, band, date) of the row's best detection, as p3s3_forced_phot.py:76-80 derives them."""
    lst = (r.get('lst_cells') or '').split('+')[0]
    bc = r.get('best_cand_id', '')
    if bc and len(bc.split('_')) > 1:
        lst = bc.split('_')[1]
    return lst, r.get('band', ''), r.get('best_date', '')


def _roots():
    out = {}
    with open(_cfg().forced_catflux.ROOTS_CSV) as f:
        for line in f:
            p = line.strip().split(',')
            if len(p) >= 3 and p[0]:
                out[p[0]] = p[2]
    return out


@content_test('catalog_forced_columns')
def catflux_columns(path, run_dir=None, recheck=True, **ctx):
    """Every survivors row carries the 5 forced columns; a measured row names the run's OWN search frame of its best
    detection (<run>/vresid or iresid; Stokes I: or the cell's subtracted/ frame, counted as the B-06 fallback), the
    fixed-gauss method line and finite values; a row 'no frame / off-frame' has blank values; RECHECK_ROWS measured rows
    (seed RECHECK_SEED) re-measured on their frame with code of its own equal forced_jy within RECHECK_TOL_JY."""
    cf = _cfg().forced_catflux
    run_dir = run_dir or os.path.dirname(os.path.dirname(os.path.abspath(path)))
    dname = os.path.basename(os.path.dirname(os.path.abspath(path)))
    parts = dname.split('_')
    S = parts[2] if len(parts) > 2 else ''
    if S not in ('I', 'V'):
        return False, None, 'not a per-Stokes catalogue directory: %s' % dname
    header, rows = _read_csv(path)
    miss = [c for c in cf.COLUMNS if c not in header]
    if miss:
        return False, None, 'columns missing: %s' % miss
    roots = None
    bad, measured, fallback, noframe = [], [], 0, 0
    for i, r in enumerate(rows):
        lst, band, date = _cat_frame(r)
        own = '%s_%s_redo_resid_%s.fits' % (date, S, band)
        if r['forced_method'] == cf.NO_FRAME:
            noframe += 1
            if any(r[c] for c in ('forced_jy', 'forced_pixel_jy', 'forced_rms_jy', 'forced_frame')):
                bad.append('row %d: no frame but values' % i)
            continue
        if not _METHOD_RE.match(r['forced_method']):
            bad.append('row %d: method %r' % (i, r['forced_method'][:60]))
        if any(_num(r[c]) is None for c in ('forced_jy', 'forced_pixel_jy', 'forced_rms_jy')):
            bad.append('row %d: values not finite' % i)
        fpath = os.path.join(run_dir, '%sresid' % S.lower(), '%s_%s' % (lst, band), r['forced_frame'])
        if r['forced_frame'] != own:
            sub = '%s_I_sub_%s.fits' % (date, band)
            if S == 'I' and cf.I_FALLBACK_SUBTRACTED and r['forced_frame'] == sub:
                fallback += 1
                roots = roots if roots is not None else _roots()
                fpath = os.path.join(roots.get('%s_%s' % (lst, band), ''), '%s_%s' % (lst, band), 'subtracted', sub)
            else:
                bad.append('row %d: frame %s is not the run\'s own %s' % (i, r['forced_frame'], own))
                continue
        if not os.path.exists(fpath):
            bad.append('row %d: frame %s not found' % (i, fpath))
            continue
        measured.append((i, r, fpath))
    n_re = 0
    if recheck and measured:
        for i, r, fpath in random.Random(cf.RECHECK_SEED).sample(measured, min(cf.RECHECK_ROWS, len(measured))):
            a = fixed_gauss(fpath, float(r['ra']), float(r['dec']))
            n_re += 1
            if a is None or abs(round(a, cf.DECIMALS) - float(r['forced_jy'])) > cf.RECHECK_TOL_JY:
                bad.append('row %d: re-measured %s vs forced_jy %s on %s' % (i, a, r['forced_jy'], os.path.basename(fpath)))
    if bad:
        return False, len(bad), _fail_detail(bad)
    return True, len(rows), '%d rows: %d measured (%d on the Stokes I fallback), %d without a frame; %d re-measured' % (
        len(rows), len(measured), fallback, noframe, n_re)


# ---------------------------------------------------------------------------------------------------------------------
# forced_pages (the isolated deep forced page run; judgements/queue_forced.txt)
# ---------------------------------------------------------------------------------------------------------------------

def _panels_meta(P, key):
    p = os.path.join(P, '_records', 'forced', 'panels.json')
    try:
        with open(p) as f:
            return (json.load(f).get('meta') or {}).get(key)
    except (OSError, ValueError):
        return None


@content_test('forced_pages')
def deep_forced_panels_verify(path, targets_csv=None, src_run=None, **ctx):
    """judgements/queue_forced.txt = the per-Stokes queues (queue_forced_I + _V, comment lines excluded); and
    deep_forced_panels.py --verify <rows> <page run>, run in place (core.legacy.run), prints DEEP-FORCED-VERIFY-OK: every
    page served once, every served position its event's radio position or blank, info boxes as recorded at render, shared
    names on their own list entry - recomputed from the input rows (deep_forced_panels.py:245-333). The rows: ctx, else the
    panels record of the page run; the searched run: ctx, else the page run's iresid link."""
    from exopipe.phase3.core import legacy
    from exopipe.phase3.lib.forced import common as K
    c = _cfg()
    P = os.path.dirname(os.path.dirname(os.path.abspath(path)))
    targets_csv = targets_csv or _panels_meta(P, 'targets_csv')
    if not targets_csv or not os.path.exists(targets_csv):
        return False, None, 'the rows the pages were built from are unknown (pass targets_csv; no panels record in %s)' % P
    if not src_run:
        lk = os.path.join(P, 'iresid')
        src_run = os.path.dirname(os.readlink(lk).rstrip('/')) if os.path.islink(lk) else None
    if not src_run:
        return False, None, 'searched run unknown (no iresid link in %s)' % P
    body = [l for l in open(path) if not l.startswith('#')]
    per = []
    for S in ('I', 'V'):
        q = os.path.join(P, 'judgements', 'queue_forced_%s.txt' % S)
        if os.path.exists(q):
            per += [l for l in open(q) if not l.startswith('#')]
    if body != per:
        return False, None, 'queue_forced.txt (%d entries) != queue_forced_I + _V (%d)' % (len(body), len(per))
    res = legacy.run(K.script('panels'), ['--verify', targets_csv, P], python=c.forced.PYTHON_KEY['panels'],
                     env={'DEEP_FORCED_SRC_RUN': src_run}, cwd=P)
    ok = res.rc == 0 and c.forced_panels.VERIFY_OK in res.tail
    return ok, len(body), (res.tail.strip().splitlines() or [''])[-1][:400]
