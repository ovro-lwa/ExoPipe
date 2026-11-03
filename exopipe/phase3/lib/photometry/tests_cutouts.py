"""Content tests of the photometry CUTOUT stages (`photometry --stages cutouts|cutmeasure`), registered with @content_test
(core/validate discovers this module). Nothing here imports an in-place engine: every check is written out from the production
code or content test it ports (named per function), the rule of the production content tests (test_snapshot_forced_reserve.py
docstring lines 6-8).

Classes (one product each):
  photometry_cutout_manifest  <root>/_manifest/<10s|deep|10min>/<unit key>[.<tags>].csv - one row per cutout a unit placed:
                              CM1 every listed file exists with its size, its structure passes (C1-C4 / TC1-TC4) and a sample's
                              sha256 is the recorded one; CM2 one file per (sample, band, object, LST, date, Stokes) - no other
                              time stamp of the unit's cell beside it, no file of the unit's cell and samples the unit did not
                              write (CUTOVER_READINESS 'Photometry cutout writer, snapshot tier': the stale T131122 beside
                              T131121 of 12h 2024-12-31)
  photometry_tier_manifest    <tier forced root>/<10min|deep>/<LST>/forced_manifest.csv - TM1 header + statuses, no REFUSED row,
                              an ok row; TM2 every ok cutout's object is a name of its (sample, band, Stokes) npz
  photometry_tier_npz         <tier forced root>/<10min|deep>/<LST>/forced_<sample>_<band>_<S>.npz - TN1 shapes and keys (moving
                              targets: ra10 / dec10, orbit controls), TN2 the target's value and stamp rms re-measured from its
                              cutout at 3 planes of 2 targets (each plane's own pixel for the planets)

Identity of a written cutout with its source (called by the stage on the STAGED files, before they reach the generation root,
and recorded as unit checks): identity_subband / identity_wideband (the regeneration's own content test,
snapshot_cutout_regen_20261025_test.py, with its exact wideband rebuild bound) and tier_recut
(photometry_tier_cutouts_20261025.content_test)."""
import csv
import glob
import hashlib
import math
import os
import random
import re

from exopipe.phase3.core.validate import content_test

SNAP_RE = re.compile(r'^(?P<obj>.+)_(?P<lst>\d\dh)_(?P<ymd>\d{8})T(?P<hms>\d{6})_(?P<st>[IV])\.fits$')
TIER_RE = re.compile(r'^(?P<obj>.+)_(?P<lst>\d\dh)_(?P<tier>10min|deep)_(?P<st>[IV])\.fits$')
WIDEBANDS = ('Red', 'Green', 'Blue')
TIER_NPZ_KEYS = ('names', 'ra', 'dec', 'ring_beams', 'ts', 'night', 'key', 'admitted', 'val10', 'rms10')
TIER_MANIFEST_HEADER = ('file', 'status', 'ring_beams', 'n_ctl', 'tier')     # photometry_tier_forced_20261026.py:194
_NPZ_RE = re.compile(r'^forced_(?P<smp>.+)_(?P<band>\d+MHz|Red|Green|Blue|Full)_(?P<st>[IV])\.npz$')


def _cfg():
    from exopipe.phase3.config import CFG
    return CFG


def _fmt(items, n=3):
    items = list(items)
    return '; '.join(str(x) for x in items[:n]) + (' ... (%d)' % len(items) if len(items) > n else '')


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 22), b''):
            h.update(blk)
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------------------------------
# names and layout
# ---------------------------------------------------------------------------------------------------------------------

def parse_cutout(path):
    """(sample, band, object dir, match, kind) of a cutout path <root>/<sample>/<band>/<object>/<file>; kind '10s' or the tier
    token; None for any other name."""
    parts = os.path.abspath(path).split('/')
    fn = parts[-1]
    m = SNAP_RE.match(fn)
    if m:
        return parts[-4], parts[-3], parts[-2], m, '10s'
    m = TIER_RE.match(fn)
    if m:
        return parts[-4], parts[-3], parts[-2], m, m['tier']
    return None


def manifest_dir(root, token):
    return os.path.join(root, _cfg().photometry_cutouts.MANIFEST_SUBDIR, token)


def read_manifest(path):
    with open(path, newline='', encoding='utf-8') as f:
        rd = csv.DictReader(f)
        return list(rd.fieldnames or []), list(rd)


def write_manifest(path, rows):
    cols = list(_cfg().photometry_cutouts.MANIFEST_COLUMNS)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = '%s.tmp.%d' % (path, os.getpid())
    with open(tmp, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in sorted(rows, key=lambda r: r['file']):
            w.writerow({k: r[k] for k in cols})
    os.replace(tmp, path)
    return path


def _header_cell(path):
    from astropy.io import fits
    try:
        return str(fits.getheader(path).get('CELL', ''))
    except (OSError, ValueError):
        return ''


def superseded(root, token, key, band, samples, keep):
    """Files of the unit in ``root`` that ``keep`` (paths relative to root) does not hold - the one-file-per-key rule.
    10 s (token '10s', key = the night cell <LST>_<date>): every <sample>/<band>/*/*_<LST>_<YYYYMMDD>T*_[IV].fits of the
    samples whose header CELL is the cell (YYYYMMDD = the cell's date and the kept files' own dates); tiers (key = <LST>):
    every <sample>/<band>/*/*_<LST>_<token>_[IV].fits. Returns (stale, orphan): stale = another time stamp of a kept
    (sample, band, object, Stokes); orphan = a file of an object / Stokes the unit did not write."""
    keep = {os.path.normpath(k) for k in keep}
    kept_keys = set()
    for k in keep:
        p = parse_cutout(os.path.join(root, k))
        if p:
            kept_keys.add((p[0], p[1], p[2], p[3]['st']))
    found = set()
    if token == '10s':
        lst, date = key.split('_', 1)
        ymds = {date.replace('-', '')} | {parse_cutout(os.path.join(root, k))[3]['ymd'] for k in keep
                                          if parse_cutout(os.path.join(root, k))}
        if not keep:            # nothing written (steps_cutouts.nothing_written): the cell's files may start after 0 UT, so the
            import datetime     # next day's names are read too - the header CELL below decides
            ymds.add((datetime.datetime.strptime(date, '%Y-%m-%d') + datetime.timedelta(days=1)).strftime('%Y%m%d'))
        for smp in samples:
            for ymd in sorted(ymds):
                found.update(glob.glob(os.path.join(root, smp, band, '*', '*_%s_%sT*_[IV].fits' % (lst, ymd))))
    else:
        lst = key
        for smp in samples:
            found.update(glob.glob(os.path.join(root, smp, band, '*', '*_%s_%s_[IV].fits' % (lst, token))))
    stale, orphan = [], []
    for p in sorted(found):
        rel = os.path.normpath(os.path.relpath(p, root))
        if rel in keep:
            continue
        q = parse_cutout(p)
        if not q or q[2] != q[3]['obj']:
            continue
        if token == '10s' and _header_cell(p) != key:
            continue
        (stale if (q[0], q[1], q[2], q[3]['st']) in kept_keys else orphan).append(rel)
    return stale, orphan


# ---------------------------------------------------------------------------------------------------------------------
# structure of one cutout (photometry_cutout C1-C4 for 10 s; TC1-TC4 for the tier files)
# ---------------------------------------------------------------------------------------------------------------------

def tier_structure(path):
    """TC1 cube side = round(CUT_DEG / CDELT2), TC2 the tier's TIMES columns with one row per plane (NPLANES), TC3 header
    cards = path (SAMPLE, BAND, STOKES, LST, TIER, OBJECT), TC4 the first plane's target inside the box
    (photometry_tier_cutouts_20261025.py :178-209)."""
    import numpy as np
    from astropy.io import fits
    cc = _cfg().photometry_cutouts
    p = parse_cutout(path)
    if not p or p[4] == '10s':
        return False, None, 'file name is not <object>_<LST>_<10min|deep>_<S>.fits'
    smp, band, objd, mt, token = p
    tier = {v: k for k, v in cc.TIER_TOKEN.items()}[token]
    bad = []
    with fits.open(path, memmap=True) as h:
        H = h[0].header
        shp = h[0].data.shape if h[0].data is not None else ()
        if len(shp) != 3 or shp[1] != shp[2]:
            bad.append('TC1 cube shape %s' % (shp,))
        elif shp[1] != int(round(cc.CUT_DEG / abs(float(H['CDELT2'])))):
            bad.append('TC1 side %d px != round(%g deg / CDELT2)' % (shp[1], cc.CUT_DEG))
        if 'TIMES' not in h:
            bad.append('TC2 no TIMES table')
        else:
            T = h['TIMES'].data
            cols = tuple(h['TIMES'].columns.names)
            if cols != tuple(cc.TIER_TIMES_COLUMNS[tier]):
                bad.append('TC2 TIMES columns %s' % (cols,))
            elif shp and (len(T) != shp[0] or int(H.get('NPLANES', -1)) != shp[0]):
                bad.append('TC2 TIMES rows %d / NPLANES %s != planes %d' % (len(T), H.get('NPLANES'), shp[0]))
            elif len(shp) == 3:
                tx, ty = float(T['target_x'][0]), float(T['target_y'][0])
                if np.isfinite(tx) and not (0 <= tx <= shp[2] - 1 and 0 <= ty <= shp[1] - 1):
                    bad.append('TC4 first-plane target (%.1f, %.1f) outside the box' % (tx, ty))
        miss = [k for k in cc.TIER_HEADER_CARDS if k not in H]
        if miss:
            bad.append('TC3 header cards missing %s' % miss)
        else:
            want = {'SAMPLE': smp, 'BAND': band, 'STOKES': mt['st'], 'LST': mt['lst'], 'TIER': token}
            bad += ['TC3 %s %s != path %s' % (k, H[k], v) for k, v in want.items() if str(H[k]) != v]
            from exopipe.phase3.lib.photometry.tests import safe_name
            if safe_name(str(H['OBJECT'])) != objd:
                bad.append('TC3 OBJECT %s vs path %s' % (H['OBJECT'], objd))
    if bad:
        return False, len(bad), _fmt(bad)
    return True, shp[0] if shp else 0, 'cube %s, TIMES %d rows, header = path, target inside' % (shp, shp[0])


def structure(path):
    """C1-C4 (10 s, tests.c1) or TC1-TC4 (tier files)."""
    p = parse_cutout(path)
    if p and p[4] != '10s':
        return tier_structure(path)
    from exopipe.phase3.lib.photometry import tests as T
    return T.c1(path)


# ---------------------------------------------------------------------------------------------------------------------
# identity of 10-s cutouts with their cube (port of snapshot_cutout_regen_20261025_test.py, exopipe_ops bd8bf26)
# ---------------------------------------------------------------------------------------------------------------------

def rrms(a):
    """1.4826 x MAD of the finite values, NaN with 50 or fewer (the test's rrms, :34-36)."""
    import numpy as np
    a = a[np.isfinite(a)]
    return 1.4826 * float(np.median(np.abs(a - np.median(a)))) if a.size > 50 else float('nan')


def cube_path(cell, sb, stokes, cube_root=None):
    root = cube_root or _cfg().paths.SNAPSHOT_CUBES_ROOT
    return os.path.join(root, cell, '%s_%s_%s_flowcube.fits' % (cell, sb, stokes))


def frame_index(lst, date, sb, stokes, tss, hdrs, search_out=None):
    """The cube frames the search used: every frame when no DATE-OBS repeats; else the frames of the search's own frame
    list (noise_<sb>_<S>_median.csv under the snapshot search outputs), cube order kept - ONE FRAME PER INTEGRATION
    (phase3_snapshot_search.flowcube_frame_index :1333-1357, clv15 ae781ac), written out. Raises ValueError where the
    engine refuses."""
    dob = [str(h.get('DATE-OBS', '')) for h in hdrs]
    if len(set(dob)) == len(dob):
        return list(range(len(tss)))
    p = os.path.join(search_out or _cfg().paths.SNAPSHOT_FLEET_OUT, '%s_%s' % (lst, date),
                     'noise_%s_%s_median.csv' % (sb, stokes))
    if not os.path.exists(p):
        raise ValueError('the %s %s %s %s cube repeats integrations and the search\'s frame list %s is missing'
                         % (lst, date, sb, stokes, p))
    with open(p, newline='') as fh:
        keep = {r['ts'] for r in csv.DictReader(fh)}
    idx = [i for i, t in enumerate(tss) if t in keep]
    kd = [dob[i] for i in idx]
    if len(set(kd)) != len(kd) or len(idx) != len(keep):
        raise ValueError('the search\'s own frame list of %s %s %s %s repeats integrations or names frames the cube lacks'
                         % (lst, date, sb, stokes))
    return idx


def _cube_table(path):
    from astropy.io import fits
    with fits.open(path, memmap=True) as h:
        t = h['FRAME_TABLE'].data
        return [str(x) for x in t['ts']], [fits.Header.fromstring(str(x), sep='\n') for x in t['header']]


def _hdr_ok(ch, band, stokes, cell, lst, sample):
    return (str(ch.get('BAND')) == band and str(ch.get('STOKES')) == stokes and str(ch.get('CELL')) == cell
            and str(ch.get('LST')) == lst and str(ch.get('SAMPLE')) == sample)


def identity_subband(files, cell, band, stokes, cube_root=None, search_out=None):
    """[(file, passed, value, detail)] of each subband cutout against its consolidated flow cube (the regeneration test's
    non-wideband branch, :142-216): times = the cube's frame list; header = path and cube; identity: R_k = cube_k[box] -
    cutout_k is one image in every plane (minus the hour-median model) - max |R_k - median_k R_k| - 2 eps32 (|cube| + |model|)
    <= IDENTITY_TOL x the cutout's robust rms; a planet: R of consecutive admitted planes agree where their boxes overlap."""
    import numpy as np
    from astropy.io import fits
    from astropy.wcs import WCS
    cc = _cfg().photometry_cutouts
    eps32 = float(np.finfo(np.float32).eps)
    lst, date = cell.split('_', 1)
    cp = cube_path(cell, band, stokes, cube_root)
    with fits.open(cp, memmap=True) as h:
        cube = np.array(h['FRAMES'].data, dtype=np.float32)
        t = h['FRAME_TABLE'].data
        cts = [str(x) for x in t['ts']]
        hdrs = [fits.Header.fromstring(str(x), sep='\n') for x in t['header']]
    ix = frame_index(lst, date, band, stokes, cts, hdrs, search_out)
    if len(ix) != len(cts):
        cube = np.ascontiguousarray(cube[ix])
        cts = [cts[i] for i in ix]
        hdrs = [hdrs[i] for i in ix]
    k0 = next((i for i, hd in enumerate(hdrs) if np.isfinite(float(hd.get('BMAJ', np.nan))) and float(hd.get('BMAJ', 0)) > 0
               and np.isfinite(cube[i]).mean() >= 0.5), 0)
    h0 = hdrs[k0]
    w0 = WCS(h0).celestial
    ny, nx = cube.shape[1:]
    out = []
    for f in files:
        sample = os.path.abspath(f).split('/')[-4]
        with fits.open(f) as hf:
            cut = np.asarray(hf[0].data, dtype=np.float32)
            ch = hf[0].header
            T = hf['TIMES'].data
        times_ok = [str(x) for x in T['ts']] == cts
        hdr_ok = _hdr_ok(ch, band, stokes, cell, lst, sample)
        n = cut.shape[1]
        half = n // 2
        moving = sample == _cfg().photometry.PLANET_SAMPLE
        if moving:
            px, py = w0.all_world2pix(np.asarray(T['ra_deg'], float), np.asarray(T['dec_deg'], float), 0)
            origins = [(int(round(float(x))) - half, int(round(float(y))) - half) for x, y in zip(px, py)]
        else:
            x00 = int(round(float(h0['CRPIX1']) - float(ch['CRPIX1'])))
            y00 = int(round(float(h0['CRPIX2']) - float(ch['CRPIX2'])))
            origins = [(x00, y00)] * cut.shape[0]
        R = np.full(cut.shape, np.nan, np.float32)
        for k, (x0, y0) in enumerate(origins[:min(len(origins), cube.shape[0])]):
            xa, xb, ya, yb = max(0, x0), min(nx, x0 + n), max(0, y0), min(ny, y0 + n)
            if xb <= xa or yb <= ya:
                continue
            R[k, ya - y0:yb - y0, xa - x0:xb - x0] = cube[k, ya:yb, xa:xb] - cut[k, ya - y0:yb - y0, xa - x0:xb - x0]
        rms = rrms(cut)
        if moving:
            adm = np.asarray(T['admitted'], bool)
            vals = [(x0, y0, k) for k, (x0, y0) in enumerate(origins[:cube.shape[0]]) if adm[k]]
            diffs = []
            for (xa0, ya0, ka), (xb0, yb0, kb) in zip(vals[:-1], vals[1:]):
                dx, dy = xb0 - xa0, yb0 - ya0
                if abs(dx) >= n or abs(dy) >= n:
                    continue
                a_ = R[ka][max(0, dy):n + min(0, dy), max(0, dx):n + min(0, dx)]
                b_ = R[kb][max(0, -dy):n + min(0, -dy), max(0, -dx):n + min(0, -dx)]
                d = np.abs(a_ - b_)
                d = d[np.isfinite(d)]
                if d.size:
                    diffs.append(float(d.max()))
            ident = (max(diffs) / rms) if diffs and rms > 0 else float('nan')
        else:
            with np.errstate(all='ignore'):
                Rm = np.nanmedian(R, axis=0)
                dev = np.abs(R - Rm[None]) - cc.EPS32_FACTOR * eps32 * (np.abs(R + cut) + np.abs(Rm)[None])
                ident = float(np.nanmax(dev)) / rms if rms > 0 and np.isfinite(dev).any() else float('nan')
        ok = times_ok and hdr_ok and np.isfinite(ident) and ident <= cc.IDENTITY_TOL
        out.append((f, bool(ok), ident, 'times %s, header %s, identity %.2e x rms' % (
            'ok' if times_ok else 'MISMATCH', 'ok' if hdr_ok else 'MISMATCH', ident)))
    return out


def wideband_stack(cell, stokes, members_csv, group, files, cube_root=None, search_out=None):
    """The group frame the group search formed, rebuilt inside each file's box from the member cubes and the job's member
    table (wideband_members_<S>.csv: per wideband frame ts, each member's subband, frame ts and weight): stack_k = sum w x
    nan_to_num(member frame) / sum w, and A_k = sum w |member| / sum w (the rounding bound) - the regeneration test's
    wideband_stack (:39-91), written out. Returns (ts list, {file: stack}, {file: (n, origins)}, {file: A})."""
    import numpy as np
    from astropy.io import fits
    from astropy.wcs import WCS
    with open(members_csv, newline='') as fh:
        rows = [r for r in csv.DictReader(fh) if r['group'] == group]
    subs = sorted({r['subband'] for r in rows}, key=lambda s: float(s.replace('MHz', '')))
    lst_, date_ = cell.split('_', 1)
    tabs, orig = {}, {}
    for sb in subs:
        ts0, hd0 = _cube_table(cube_path(cell, sb, stokes, cube_root))
        ix = frame_index(lst_, date_, sb, stokes, ts0, hd0, search_out)
        tabs[sb] = ([ts0[i] for i in ix], [hd0[i] for i in ix])
        orig[sb] = {ts0[i]: i for i in ix}
    ref = max(subs, key=lambda s: len(tabs[s][0]))
    tss = tabs[ref][0]
    tix = {ts: i for i, ts in enumerate(tss)}
    h0 = tabs[subs[0]][1][0]
    w0 = WCS(h0).celestial
    boxes = {}
    for f in files:
        with fits.open(f) as hf:
            ch = hf[0].header
            T = hf['TIMES'].data
            n = hf[0].data.shape[1]
        half = n // 2
        if os.path.abspath(f).split('/')[-4] == _cfg().photometry.PLANET_SAMPLE:
            px, py = w0.all_world2pix(np.asarray(T['ra_deg'], float), np.asarray(T['dec_deg'], float), 0)
            boxes[f] = (n, [(int(round(float(x))) - half, int(round(float(y))) - half) for x, y in zip(px, py)])
        else:
            o = (int(round(float(h0['CRPIX1']) - float(ch['CRPIX1']))), int(round(float(h0['CRPIX2']) - float(ch['CRPIX2']))))
            boxes[f] = (n, [o] * len(tss))
    acc = {f: np.zeros((len(tss), b[0], b[0]), np.float64) for f, b in boxes.items()}
    absacc = {f: np.zeros((len(tss), b[0], b[0]), np.float32) for f, b in boxes.items()}
    wsum = np.zeros(len(tss))
    for r in rows:
        wsum[tix[r['ts']]] += float(r['weight'])
    for sb in subs:
        with fits.open(cube_path(cell, sb, stokes, cube_root), memmap=True) as h:
            cube = np.array(h['FRAMES'].data, dtype=np.float32)
        mix = orig[sb]
        ny, nx = cube.shape[1:]
        for r in rows:
            if r['subband'] != sb:
                continue
            k, j, wt = tix[r['ts']], mix[r['member_ts']], np.float32(float(r['weight']))
            frame = np.nan_to_num(cube[j], nan=0.0) * wt
            aframe = np.abs(frame)
            for f, (n, origins) in boxes.items():
                x0, y0 = origins[k]
                xa, xb, ya, yb = max(0, x0), min(nx, x0 + n), max(0, y0), min(ny, y0 + n)
                if xb > xa and yb > ya:
                    acc[f][k, ya - y0:yb - y0, xa - x0:xb - x0] += frame[ya:yb, xa:xb]
                    absacc[f][k, ya - y0:yb - y0, xa - x0:xb - x0] += aframe[ya:yb, xa:xb]
        del cube
    nz = np.where(wsum > 0, wsum, 1.0)
    return (tss, {f: (acc[f] / nz[:, None, None]).astype(np.float32) for f in acc}, boxes,
            {f: (absacc[f] / nz[:, None, None]).astype(np.float32) for f in absacc})


def wideband_identity(cut, S_, A_, moving, origins=None, rel=None):
    """identity_max_over_rms of one wideband cutout against its rebuilt stack (:121-137): the member table's '%.6g'
    weights (|dw/w| <= WEIGHT_REL_PRECISION) bound the rebuilt S = sum w x / sum w by rel x (A + |S|) per pixel, on top of
    IDENTITY_TOL x rms. Fixed target: max over planes and pixels of |R - median_k R| - rel x (A + |S| + median_k (A + |S|));
    planet: consecutive planes' R where their boxes overlap, less the bound of both."""
    import numpy as np
    rel = _cfg().photometry_cutouts.WEIGHT_REL_PRECISION if rel is None else rel
    R = S_ - cut
    rms = rrms(cut)
    with np.errstate(all='ignore'):
        if moving:
            n = cut.shape[1]
            diffs = []
            for k in range(len(origins) - 1):
                dx, dy = origins[k + 1][0] - origins[k][0], origins[k + 1][1] - origins[k][1]
                if abs(dx) >= n or abs(dy) >= n:
                    continue
                sl_a = (slice(max(0, dy), n + min(0, dy)), slice(max(0, dx), n + min(0, dx)))
                sl_b = (slice(max(0, -dy), n + min(0, -dy)), slice(max(0, -dx), n + min(0, -dx)))
                d = np.abs(R[k][sl_a] - R[k + 1][sl_b]) - rel * (A_[k][sl_a] + np.abs(S_[k][sl_a]) + A_[k + 1][sl_b]
                                                                 + np.abs(S_[k + 1][sl_b]))
                d = d[np.isfinite(d)]
                if d.size:
                    diffs.append(float(d.max()))
            return max(diffs) / rms if diffs and rms > 0 else float('nan')
        Rm = np.nanmedian(R, axis=0)
        dev = np.abs(R - Rm[None]) - rel * (A_ + np.abs(S_) + np.nanmedian(A_ + np.abs(S_), axis=0)[None])
        return float(np.nanmax(dev)) / rms if rms > 0 and np.isfinite(dev).any() else float('nan')


def identity_wideband(files, cell, group, stokes, members_csv, cube_root=None, search_out=None):
    """[(file, passed, value, detail)] of each wideband cutout against the stack rebuilt from its member cubes (the
    regeneration test's wideband branch, :105-141, with the exact first-order bound of bd8bf26)."""
    import numpy as np
    from astropy.io import fits
    cc = _cfg().photometry_cutouts
    lst = cell.split('_', 1)[0]
    tss, stacks, boxes, mabs = wideband_stack(cell, stokes, members_csv, group, files, cube_root, search_out)
    out = []
    for f in files:
        sample = os.path.abspath(f).split('/')[-4]
        with fits.open(f) as hf:
            cut = np.asarray(hf[0].data, dtype=np.float32)
            ch = hf[0].header
            T = hf['TIMES'].data
        times_ok = [str(x) for x in T['ts']] == tss
        hdr_ok = _hdr_ok(ch, group, stokes, cell, lst, sample)
        ident = wideband_identity(cut, stacks[f], mabs[f], sample == _cfg().photometry.PLANET_SAMPLE, boxes[f][1])
        ok = times_ok and hdr_ok and np.isfinite(ident) and ident <= cc.IDENTITY_TOL
        out.append((f, bool(ok), ident, 'times %s, header %s, identity %.2e x rms (wideband, members %s)' % (
            'ok' if times_ok else 'MISMATCH', 'ok' if hdr_ok else 'MISMATCH', ident, os.path.basename(members_csv))))
    return out


# ---------------------------------------------------------------------------------------------------------------------
# tier cutouts: one plane re-cut from its source frame (port of photometry_tier_cutouts_20261025.content_test :212-230)
# ---------------------------------------------------------------------------------------------------------------------

def tier_box(sec, ny, nx, px, py, n, lead):
    """The n x n box whose corner is round(p) - n // 2, NaN outside the image (the tool's box, :98-105)."""
    import numpy as np
    half = n // 2
    x0, y0 = int(round(px)) - half, int(round(py)) - half
    out = np.full((n, n), np.nan, dtype=np.float32)
    xa, xb, ya, yb = max(0, x0), min(nx, x0 + n), max(0, y0), min(ny, y0 + n)
    if xb > xa and yb > ya:
        out[ya - y0:yb - y0, xa - x0:xb - x0] = np.asarray(sec[lead + (slice(ya, yb), slice(xa, xb))], dtype=np.float32)
    return out


def tier_recut(files, root=None):
    """[(file, passed, plane, detail)]: for each tier cutout ONE plane (seeded per file by photometry_cutouts.TIER_RECUT_SEED
    and its path) re-cut from the source frame its TIMES row names, at that row's RA / Dec: NaN mask and finite values equal
    to the file's plane (the tool's own --test). A plane without a position (NaN RA) is skipped, as the tool skips it."""
    import numpy as np
    from astropy.io import fits
    from astropy.wcs import WCS
    seed = _cfg().photometry_cutouts.TIER_RECUT_SEED
    out = []
    for fn in files:
        rel = os.path.relpath(fn, root) if root else os.path.basename(fn)
        rnd = random.Random('%d:%s' % (seed, rel))
        with fits.open(fn) as hf:
            cube = hf[0].data
            T = hf['TIMES'].data
            i = rnd.randrange(len(T))
            src = str(T['source'][i]).strip()
            ra, dec = float(T['ra_deg'][i]), float(T['dec_deg'][i])
            plane = np.array(cube[i], dtype=np.float32)
        if not np.isfinite(ra):
            out.append((fn, True, i, 'plane %d has no position (skipped, as the tool)' % i))
            continue
        with fits.open(src, memmap=True) as hs:
            hh = hs[0].header
            w = WCS(hh).celestial
            px, py = [float(v) for v in w.all_world2pix(ra, dec, 0)]
            b = tier_box(hs[0].section, hh['NAXIS2'], hh['NAXIS1'], px, py, plane.shape[0], (0,) * (hh['NAXIS'] - 2))
        same = np.array_equal(np.isnan(b), np.isnan(plane)) and np.array_equal(b[np.isfinite(b)], plane[np.isfinite(plane)])
        out.append((fn, bool(same), i, 'plane %d re-cut from %s %s' % (i, os.path.basename(src), 'equal' if same else 'DIFFERS')))
    return out


# ---------------------------------------------------------------------------------------------------------------------
# photometry_cutout_manifest
# ---------------------------------------------------------------------------------------------------------------------

def _manifest_unit(path, ctx):
    """(root, token, key, band, samples, rows) of a unit manifest <root>/_manifest/<token>/<key>_<band>[.<tags>].csv."""
    cols, rows = read_manifest(path)
    d = os.path.dirname(os.path.abspath(path))
    root = ctx.get('root') or os.path.dirname(os.path.dirname(d))
    token = os.path.basename(d)
    stem = os.path.basename(path)[:-4].split('.', 1)[0]
    key, band = stem.rsplit('_', 1)
    samples = list(ctx.get('samples') or sorted({r['sample'] for r in rows}))
    return cols, root, token, key, band, samples, rows


@content_test('photometry_cutout_manifest', 'CM1_files')
def cm1(path, **ctx):
    cc = _cfg().photometry_cutouts
    cols, root, token, key, band, samples, rows = _manifest_unit(path, ctx)
    if tuple(cols) != tuple(cc.MANIFEST_COLUMNS):
        return False, None, 'columns %s != %s' % (cols, cc.MANIFEST_COLUMNS)
    if not rows:
        return False, 0, 'no row'
    bad = []
    for r in rows:
        p = os.path.join(root, r['file'])
        q = parse_cutout(p)
        if not q or q[0] != r['sample'] or q[1] != band or q[2] != r['object'] or q[3]['st'] != r['stokes'] or q[4] != token:
            bad.append('%s: path does not name its row (sample, band %s, object, Stokes, tier %s)' % (r['file'], band, token))
            continue
        if not os.path.isfile(p):
            bad.append('%s missing' % r['file'])
            continue
        if os.path.getsize(p) != int(r['bytes']):
            bad.append('%s: %d bytes, manifest %s' % (r['file'], os.path.getsize(p), r['bytes']))
            continue
        try:
            ok, _, det = structure(p)
        except Exception as e:                       # noqa: BLE001 - an unreadable cutout is a failed row, named
            ok, det = False, 'structure test raised %s: %s' % (type(e).__name__, e)
        if not ok:
            bad.append('%s: %s' % (r['file'], det))
    pick = sorted({int(round(x)) for x in [i * (len(rows) - 1) / max(1, cc.MANIFEST_HASH_FILES - 1)
                                            for i in range(min(cc.MANIFEST_HASH_FILES, len(rows)))]})
    for i in pick:
        p = os.path.join(root, rows[i]['file'])
        if os.path.isfile(p) and sha256_file(p) != rows[i]['sha256']:
            bad.append('%s: sha256 differs from the manifest' % rows[i]['file'])
    if bad:
        return False, len(bad), _fmt(bad)
    return True, len(rows), '%d files: present, sizes = manifest, structure passes; sha256 of %d equal' % (len(rows), len(pick))


@content_test('photometry_cutout_manifest', 'CM2_one_per_key')
def cm2(path, **ctx):
    _, root, token, key, band, samples, rows = _manifest_unit(path, ctx)
    stale, orphan = superseded(root, token, key, band, samples, [r['file'] for r in rows])
    if stale or orphan:
        return False, len(stale) + len(orphan), ('%d other time stamps of a listed key (%s); %d files of the unit\'s cell this '
                                                 'unit did not write (%s)' % (len(stale), _fmt(stale), len(orphan),
                                                                              _fmt(orphan)))
    return True, len(rows), 'one file per (sample, band, object, LST, date, Stokes): no other time stamp and no file of ' \
                            'samples %s for %s %s beside the unit\'s' % (','.join(samples), key, band)


# ---------------------------------------------------------------------------------------------------------------------
# photometry_tier_manifest, photometry_tier_npz (photometry_tier_forced_20261026.py products)
# ---------------------------------------------------------------------------------------------------------------------

@content_test('photometry_tier_manifest', 'TM1_statuses')
def tm1(path, **ctx):
    with open(path, newline='', encoding='utf-8') as f:
        rd = csv.DictReader(f)
        cols, rows = list(rd.fieldnames or []), list(rd)
    if tuple(cols) != TIER_MANIFEST_HEADER:
        return False, None, 'header %s != %s' % (cols, TIER_MANIFEST_HEADER)
    tier = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(path))))
    bad = [r for r in rows if r['status'] != 'ok' or r['tier'] != tier]
    if bad:
        return False, len(bad), '%d rows not ok / of another tier (e.g. %s: %s)' % (len(bad), os.path.basename(bad[0]['file']),
                                                                                   bad[0]['status'])
    if not rows:
        return False, 0, 'no row'
    return True, len(rows), '%d cutouts measured (tier %s), none refused' % (len(rows), tier)


@content_test('photometry_tier_manifest', 'TM2_coverage')
def tm2(path, npz_dirs=None, **ctx):
    """Every ok row's object is a name of its (sample, band, Stokes) npz, read beside the manifest - or, for a sample of
    ``npz_dirs`` ({sample: directory}), in that directory: a STAGED manifest of a sample subset (the tool merges the other
    samples' rows) is tested before placement with the other samples' npz where they are served (steps_cutouts)."""
    import numpy as np
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    groups = {}
    for r in rows:
        q = parse_cutout(r['file'])
        if q and r['status'] == 'ok':
            groups.setdefault((q[0], q[1], q[3]['st']), set()).add(q[2])
    d = os.path.dirname(os.path.abspath(path))
    bad = []
    for (smp, band, st), objs in sorted(groups.items()):
        p = os.path.join((npz_dirs or {}).get(smp, d), 'forced_%s_%s_%s.npz' % (smp, band, st))
        if not os.path.exists(p):
            bad.append('no %s for %d measured cutouts' % (os.path.basename(p), len(objs)))
            continue
        with np.load(p, allow_pickle=False) as z:
            names = {str(n) for n in z['names']}
        if names != objs:
            bad.append('%s: %d names vs %d ok cutouts' % (os.path.basename(p), len(names), len(objs)))
    if bad:
        return False, len(bad), _fmt(bad)
    return True, len(groups), '%d (sample, band, Stokes) groups: npz names = the ok cutouts' % len(groups)


def _tier_npz(path):
    import numpy as np
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


@content_test('photometry_tier_npz', 'TN1_shapes')
def tn1(path, **ctx):
    m = _NPZ_RE.match(os.path.basename(path))
    if not m:
        return False, None, 'file name is not forced_<sample>_<band>_<S>.npz'
    d = _tier_npz(path)
    miss = [k for k in TIER_NPZ_KEYS if k not in d]
    if miss:
        return False, None, 'arrays missing: %s' % miss
    T, N = len(d['names']), len(d['ts'])
    nc = _cfg().photometry_measure.N_CONTROLS
    want = {'ra': (T,), 'dec': (T,), 'ring_beams': (T,), 'night': (N,), 'key': (N,), 'admitted': (N,),
            'val10': (T, nc + 1, N), 'rms10': (T, nc + 1, N)}
    if m['smp'] == _cfg().photometry.PLANET_SAMPLE:
        want.update(ra10=(T, N), dec10=(T, N))
        if 'ctl_kind' in d:
            want.update(ctl_ra=(T, nc, N), ctl_dec=(T, nc, N))
    bad = ['%s %s != %s' % (k, None if k not in d else tuple(d[k].shape), v) for k, v in want.items()
           if k not in d or tuple(d[k].shape) != v]
    if not T or not N:
        bad.append('empty')
    if bad:
        return False, len(bad), _fmt(bad)
    return True, T, '%d targets x %d positions x %d planes' % (T, nc + 1, N)


@content_test('photometry_tier_npz', 'TN2_target_remeasure')
def tn2(path, cutout_root=None, **ctx):
    """The target's value (bilinear) and stamp rms (1.4826 MAD of the 41 x 41 stamp, central 5 x 5 out) re-measured from its
    cutout at 3 planes of 2 targets: fixed targets at the median TIMES pixel, planets at each plane's own pixel
    (photometry_tier_forced_20261026.group_job; test_photometry_tier_forced_moving.py bilin / stamp_rms)."""
    import numpy as np
    from astropy.io import fits
    from exopipe.phase3.lib.photometry import tests as T
    m = _NPZ_RE.match(os.path.basename(path))
    if not m:
        return False, None, 'file name is not forced_<sample>_<band>_<S>.npz'
    c = _cfg()
    mm = c.photometry_measure
    tier, lst = os.path.abspath(path).split('/')[-3], os.path.abspath(path).split('/')[-2]
    root = cutout_root or c.photometry.CUTOUT_ROOT
    d = _tier_npz(path)
    names = [str(x) for x in d['names']]
    axis = [str(k) for k in (d['key'] if tier == '10min' else d['night'])]
    col = {k: i for i, k in enumerate(axis)}
    moving = m['smp'] == c.photometry.PLANET_SAMPLE
    pick = sorted(set(np.linspace(0, len(names) - 1, min(2, len(names))).astype(int)))
    bad, done = [], 0
    for t in pick:
        fp = os.path.join(root, m['smp'], m['band'], names[t], '%s_%s_%s_%s.fits' % (names[t], lst, tier, m['st']))
        if not os.path.exists(fp):
            bad.append('%s: cutout %s not found' % (names[t], fp))
            continue
        with fits.open(fp, memmap=True) as h:
            Tt = h['TIMES'].data
            keys = [str(k).strip() for k in (Tt['key'] if tier == '10min' else Tt['night'])]
            txs, tys = np.asarray(Tt['target_x'], float), np.asarray(Tt['target_y'], float)
            planes = sorted(set(np.linspace(0, len(keys) - 1, min(3, len(keys))).astype(int)))
            for p in planes:
                x, y = (txs[p], tys[p]) if moving else (float(np.nanmedian(txs)), float(np.nanmedian(tys)))
                if not (np.isfinite(x) and np.isfinite(y)) or keys[p] not in col:
                    continue
                got = T._snap_measure(np.asarray(h[0].data[p:p + 1], np.float32), x, y, mm.STAMP_PX, mm.STAMP_CORE_PX,
                                      mm.MIN_STAMP_FRAC['snapshot'], mm.RMS_MAD_FACTOR)
                v_npz, r_npz = float(d['val10'][t, 0, col[keys[p]]]), float(d['rms10'][t, 0, col[keys[p]]])
                if got is None:
                    if np.isfinite(v_npz):
                        bad.append('%s plane %d: no measurement possible, npz has a value' % (names[t], p))
                    continue
                for a, b_, what in ((float(got[0][0]), v_npz, 'value'), (float(got[1][0]), r_npz, 'rms')):
                    if not ((math.isnan(a) and math.isnan(b_)) or abs(a - b_) <= 1e-5 * max(1.0, abs(a))):
                        bad.append('%s plane %d %s %r != npz %r' % (names[t], p, what, a, b_))
                done += 1
    if bad:
        return False, len(bad), _fmt(bad)
    if not done:
        return False, 0, 'nothing re-measurable (cutouts under %s)' % root
    return True, done, '%d target planes re-measured from their cutouts: value and stamp rms equal (1e-5)%s' % (
        done, ' at each plane\'s own pixel' if moving else '')
