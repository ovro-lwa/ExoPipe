#!/usr/bin/env python
"""AD HOC 10-S CUTOUTS BY THE STAMP METHOD: the engine of `exopipe3 photometry --tier snapshot --stages cutouts --targets` on a
subband (Gregg 2026-11-21, "All approved", R-GREGG-20261121-THREE-DECISIONS-APPROVED item 2; the method is the snapshot chat's test,
RESULT-20261119-10S-STAMP-READ-TEST-*: pixel-identical to generation 6 in 31 (band, Stokes), about 55 s per cell-band against
about 9 min for the search's whole path).

Run by lib/photometry/steps_cutouts.snapshot_unit as a child process (core/legacy.run) under the cutouts_subband interpreter. It
imports the production modules of --code-root (phase3_snapshot_search, phase3_snapshot_cutouts: the same frame index, frame
metadata, night rule and write_cutouts as the search) and never the v2 package.

Per Stokes (V, then I) of one (cell, subband):
  1. the consolidated flow cube, read once (astropy, memmap=False), frames ordered as the search orders them
     (flowcube_frame_index); per-frame metadata (_frame_meta); the frames of the hour median = finite fraction >= 0.5, not all
     zero, not refused by the night rule (night_refused_mask) - the search's own rules;
  2. frame admission and per-frame global rms are READ, not recomputed: from the REFERENCE, a generation-6 cutout of the same
     (cell, subband, Stokes) (its TIMES admitted and global_rms_jyb), whose frame list must equal the cube's;
  3. per target (the ad hoc set of P3_SNAP_CUTOUT_SAMPLES, merged by name as write_cutouts merges it; the planets only with
     P3_SNAP_CUTOUT_PLANETS=1): a crop holding every frame's box (cutout box + write_cutouts' pad 2 + margin 4), the hour median
     of the crop over the median frames, and the production write_cutouts on the crop (WCS CRPIX shifted by the crop origin);
  4. PARITY with generation 6: the reference target is cut the same way into a scratch directory and must equal the reference
     file pixel for pixel (shape, NaN pattern, every finite value) with equal TIMES columns (ts, date_obs, admitted,
     global_rms_jyb, ra_deg, dec_deg, target_x, target_y). Run before the targets are cut.
Exclusion: a (cell, subband) with verdict 'exclude' in data_decisions is refused by the search's own gate
(phase3_snapshot_search.excluded_subbands; line 'EXCLUDED: ...', exit 3, nothing written), before anything is read.
Refusals (exit 2, a line 'REFUSED: ...'; nothing is reported written): no generation-6 flux-scale record of the (cell, subband)
(not regenerated, or refused / no cutout there); a cube that changed since generation 6 (size or mtime differ from that record);
no reference cutout of a Stokes; frame lists that differ; a failed parity check.
Done line: 'STAMP-CUTOUTS <lst> <date> <subband>: V <n> I <n>' (n = cutout files written per Stokes).
usage: stamp_engine.py --code-root D --ref-root R --cell CELL --band SUBBAND --out ROOT   (cube root: P3_SNAP_CUBE_ROOT)
"""
import argparse
import glob
import json
import os
import shutil
import sys

import numpy as np

REF_SAMPLES = ('LPT', '10pc', 'HWJ')        # searched in this order for a reference cutout (fixed targets only)
MARGIN_PX = 4                               # crop margin beyond write_cutouts' pad (the snapshot chat's test: 2 + 4)
TIMES_COLS = ('ts', 'date_obs', 'admitted', 'global_rms_jyb', 'ra_deg', 'dec_deg', 'target_x', 'target_y')


class Refused(Exception):
    pass


def fluxscale_record(ref_root, cell, band):
    p = os.path.join(ref_root, '_fluxscale', band, '%s_%s_fluxscale.json' % (cell, band))
    if not os.path.isfile(p):
        raise Refused('no generation-6 flux-scale record %s: the (cell, subband) was not regenerated, or the search refused it or '
                      'cut nothing there' % p)
    with open(p) as f:
        return p, json.load(f)


def check_cube_unchanged(rec, stokes, cube):
    r = (rec.get('cubes') or {}).get(stokes) or {}
    st = os.stat(cube)
    if os.path.realpath(r.get('path', '')) != os.path.realpath(cube):
        raise Refused('generation 6 cut %s from %s, this run reads %s' % (stokes, r.get('path'), cube))
    if (r.get('size'), r.get('mtime_ns')) != (st.st_size, st.st_mtime_ns):
        raise Refused('cube %s changed since generation 6 (size %s -> %d, mtime_ns %s -> %d)' % (
            cube, r.get('size'), st.st_size, r.get('mtime_ns'), st.st_mtime_ns))


def find_reference(ref_root, cell, band, stokes):
    """(path, other-Stokes path) of the first generation-6 cutout of (cell, band) in REF_SAMPLES order, objects sorted."""
    lst, date = cell.split('_', 1)
    pat = '_%s_%sT' % (lst, date.replace('-', ''))
    for sample in REF_SAMPLES:
        d = os.path.join(ref_root, sample, band)
        if not os.path.isdir(d):
            continue
        for obj in sorted(os.listdir(d)):
            hits = sorted(glob.glob(os.path.join(d, obj, '%s%s*_%s.fits' % (glob.escape(obj), pat, stokes))))
            if hits:
                return hits[0]
    raise Refused('no generation-6 cutout of %s %s %s under %s/{%s}' % (cell, band, stokes, ref_root, ','.join(REF_SAMPLES)))


def read_cutout(path):
    from astropy.io import fits
    with fits.open(path, memmap=False) as h:
        hd = h[0].header
        tb = h['TIMES'].data
        times = {}
        for c in TIMES_COLS:
            v = np.array(tb[c])
            times[c] = np.array([str(x).strip() for x in v]) if v.dtype.kind in 'SU' else v
        # the position of the TIMES table (doubles as written), not TARGRA/TARGDEC (a header float card keeps 16 digits)
        out = dict(path=path, data=np.array(h[0].data, np.float32), sample=str(hd['SAMPLE']), name=str(hd['OBJECT']),
                   ra=float(times['ra_deg'][0]), dec=float(times['dec_deg'][0]), times=times)
    return out


def same_array(a, b):
    if a.shape != b.shape:
        return False, 'shape %s vs %s' % (a.shape, b.shape)
    if a.dtype.kind in 'fc':
        na, nb = np.isnan(a), np.isnan(b)
        if not np.array_equal(na, nb):
            return False, 'NaN pattern differs (%d vs %d NaN)' % (na.sum(), nb.sum())
        fin = ~na
        if fin.any():
            d = float(np.max(np.abs(a[fin].astype(np.float64) - b[fin].astype(np.float64))))
            if d != 0.0:
                return False, 'max |difference| %.3g' % d
        return True, '%d finite values equal' % int(fin.sum())
    return (bool(np.array_equal(a, b)), 'equal' if np.array_equal(a, b) else 'values differ')


def parity(mine, ref):
    """[] when the cut of the reference target equals the generation-6 file, else the differences."""
    bad = []
    ok, why = same_array(mine['data'], ref['data'])
    if not ok:
        bad.append('pixels: ' + why)
    for c in TIMES_COLS:
        ok, why = same_array(np.asarray(mine['times'][c]), np.asarray(ref['times'][c]))
        if not ok:
            bad.append('TIMES %s: %s' % (c, why))
    return bad


def crop_box(w, positions, half, n, nx, ny):
    """(ya, yb, xa, xb) holding every frame's cutout box plus pad and margin, or None when no position is on the image."""
    x0s, y0s = [], []
    for ra, dec in positions:
        px, py = (float(v) for v in w.all_world2pix(ra, dec, 0))
        if np.isfinite(px) and np.isfinite(py):
            x0s.append(int(round(px)) - half)
            y0s.append(int(round(py)) - half)
    if not x0s:
        return None
    m = 2 + MARGIN_PX
    xa, xb = max(0, min(x0s) - m), min(nx, max(x0s) + n + m)
    ya, yb = max(0, min(y0s) - m), min(ny, max(y0s) + n + m)
    if xb <= xa or yb <= ya:
        return None
    return ya, yb, xa, xb


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--code-root', required=True)
    ap.add_argument('--ref-root', required=True)
    ap.add_argument('--cell', required=True)
    ap.add_argument('--band', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    # the production modules of --code-root, run from there as production runs them (`cd $OPS`); never this package's directory
    here = os.path.dirname(os.path.realpath(__file__))
    sys.path[:] = [a.code_root] + [p for p in sys.path if os.path.realpath(p or '.') != here]
    os.chdir(a.code_root)
    from astropy.io import fits
    from astropy.wcs import WCS
    import phase3_snapshot_search as SS
    import phase3_snapshot_cutouts as CUT
    lst, date = a.cell.split('_', 1)
    if a.band in SS.excluded_subbands(lst, date):
        print("EXCLUDED: %s %s %s carries verdict 'exclude' in %s - REFUSING to cut (no output written)" % (
            lst, date, a.band, SS.DATA_DECISIONS), flush=True)
        return 3
    if not os.environ.get('P3_SNAP_CUTOUT_SAMPLES'):
        raise Refused('the stamp engine cuts an ad hoc target set only (P3_SNAP_CUTOUT_SAMPLES unset)')
    rec_path, rec = fluxscale_record(a.ref_root, a.cell, a.band)
    print('STAMP-RECORD %s' % rec_path, flush=True)
    planets_on = CUT._planets()
    targets = list(CUT._targets())
    scratch = os.path.join(a.out, '_stamp_parity')
    counts = {}
    for S in ('V', 'I'):
        cube_path = os.path.join(SS.CUBE_ROOT, a.cell, '%s_%s_%s_flowcube.fits' % (a.cell, a.band, S))
        check_cube_unchanged(rec, S, cube_path)
        ref = read_cutout(find_reference(a.ref_root, a.cell, a.band, S))
        if ref['sample'] == 'SolarSystem':
            raise Refused('reference %s is a moving target' % ref['path'])
        with fits.open(cube_path, memmap=False) as h:
            tab = h['FRAME_TABLE'].data
            tss = [str(x) for x in tab['ts']]
            hdrs = [fits.Header.fromstring(str(x), sep='\n') for x in tab['header']]
            idx = SS.flowcube_frame_index(lst, date, a.band, S, tss, hdrs)
            cube = np.array(h['FRAMES'].data, dtype=np.float32)
        if len(idx) != len(tss):
            cube = np.ascontiguousarray(cube[idx])
            tss = [tss[i] for i in idx]
            hdrs = [hdrs[i] for i in idx]
        if list(ref['times']['ts']) != tss:
            raise Refused('%s: the reference frames differ from the cube frames (%d vs %d): %s' % (
                S, len(ref['times']['ts']), len(tss), ref['path']))
        meta = [SS._frame_meta(tss[i], hdrs[i], cube[i]) for i in range(len(tss))]
        k0 = next((i for i, m in enumerate(meta) if np.isfinite(m['bmaj']) and m['bmaj'] > 0
                   and m.get('finite_frac', 1.0) >= 0.5), 0)
        hdr0 = hdrs[k0]
        finite_ok = np.array([m.get('finite_frac', 1.0) >= 0.5 for m in meta])
        zero = np.array([bool(m.get('all_zero', False)) for m in meta])
        night = SS.night_refused_mask(meta, lst, date, a.band)
        stats = np.where(finite_ok & ~zero & ~night)[0]
        admitted = np.asarray(ref['times']['admitted'], bool)
        noise_rows = [{'ts': t, 'global_rms_jyb': float(r)} for t, r in zip(tss, ref['times']['global_rms_jyb'])
                      if np.isfinite(r)]
        w = WCS(hdr0).celestial
        n = int(round(CUT.CUT_DEG / abs(float(hdr0['CDELT2']))))
        half = n // 2
        nt, ny, nx = cube.shape
        saved = (CUT._TARGETS, CUT._planets)

        def cut(job, root):
            """write_cutouts of one target on its crop: job = ('fixed', (tag, name, ra, dec)) or ('planet', body)."""
            if job[0] == 'fixed':
                positions = [job[1][2:]]
                CUT._TARGETS, CUT._planets = [job[1]], (lambda: ())
            else:
                positions = [CUT.planet_position(job[1], m['ts']) for m in meta]
                CUT._TARGETS, CUT._planets = [], (lambda b=job[1]: (b,))
            box = crop_box(w, positions, half, n, nx, ny)
            if box is None:
                return 0
            ya, yb, xa, xb = box
            sub = np.ascontiguousarray(cube[:, ya:yb, xa:xb])
            model = np.median(sub[stats], axis=0)
            h2 = hdr0.copy()
            h2['CRPIX1'] = float(h2['CRPIX1']) - xa
            h2['CRPIX2'] = float(h2['CRPIX2']) - ya
            z = np.zeros(sub.shape[1:], np.float32)
            return CUT.write_cutouts(sub, model, meta, admitted, h2, z, z, lst, date, a.band, S, noise_rows=noise_rows, root=root)

        try:
            # 4. parity first: the reference target, cut the same way, equals generation 6's file
            if os.path.isdir(scratch):
                shutil.rmtree(scratch)
            if not cut(('fixed', (ref['sample'], ref['name'], ref['ra'], ref['dec'])), scratch):
                raise Refused('%s: the reference target %s was not cut' % (S, ref['name']))
            mine = glob.glob(os.path.join(scratch, ref['sample'], a.band, CUT.safe_name(ref['name']), '*_%s.fits' % S))
            if len(mine) != 1:
                raise Refused('%s: %d parity cutouts of %s' % (S, len(mine), ref['name']))
            bad = parity(read_cutout(mine[0]), ref)
            if bad:
                raise Refused('%s: PARITY with generation 6 FAILED for %s: %s' % (S, ref['path'], '; '.join(bad)))
            print('STAMP-PARITY %s %s: equal to generation 6 (%s)' % (a.band, S, ref['path']), flush=True)
            shutil.rmtree(scratch)
            jobs = [('fixed', t) for t in targets] + [('planet', b) for b in planets_on]
            counts[S] = sum(cut(j, a.out) for j in jobs)
        finally:
            CUT._TARGETS, CUT._planets = saved
        del cube
    print('STAMP-CUTOUTS %s %s %s: V %s I %s' % (lst, date, a.band, counts['V'], counts['I']), flush=True)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Refused as e:
        print('REFUSED: %s' % e, flush=True)
        sys.exit(2)
