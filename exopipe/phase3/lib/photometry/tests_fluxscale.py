"""Content checks of a 10-s cutout unit regenerated from the rescaled flow cubes (config photometry_fluxscale), and the content
test of the flux-scale record (class photometry_fluxscale_record).

Ported from the generation-6 tests (exopipe_ops/gen6_20261117/gen6_cutout_regen_test_DRAFT.py job checks (a)(b)(c),
gen6_wb_test_DRAFT.py check (w)). Each returns (passed, value, detail); the step records a failed check as a refusal of the unit
(spec section 6 says report-only; gates refuse, never warn):

  (a) check_a   the cube carries HDU-0 FSAPPLD; the flux-scale record exists, its cubes[S].fsappld equals the cube's (within
                FSAPPLD_TOL) and its path / size / mtime_ns are the cube's now (the cutouts were cut from the cube tested)
  (b) check_b   on the generation-5 sample (fluxscale.gen5_sample): the same file set as generation 5 (samples new since then
                left out), each file's TIMES ts and shape and NaN pattern equal to generation 5's, and over all planes of all
                files the median of the per-plane least-squares slope new/gen5 (sum new gen5 / sum gen5^2 over the plane's
                finite pixels) x FSAPPLD within 1 +- GEN5_SLOPE_TOL
  (c) check_c   the distinct (sample, object directory) set equals the set the writer selects for the cube - list entries
                (merged by name within MERGE_ARCMIN, J-tagged beyond) above MIN_EL_DEG at the cube's mid-time and inside the
                frame of its first valid frame's WCS, plus the planets by the same rules - and no object has two files
  (w) WeightCheck  on WEIGHT_SAMPLE_FRAMES wideband frames spread over the hour, each member row's weight = 1 / (F x rms^2),
                F = the member cube's HDU-0 FSAPPLD, rms = 1.4826 MAD over the central RMS_CENTRAL_FRAC of the area of the
                member frame after the identity dewarp - both written out here, not taken from the group search (generation 6
                imported the engine's central_rms and dewarp_frame, so its check (w) was not independent of the engine)
"""
import json
import math
import os

from exopipe.phase3.core.validate import content_test


def _cfg():
    from exopipe.phase3.config import CFG
    return CFG


def _parts(path):
    """(sample, band, object directory) of <root>/<sample>/<band>/<object>/<file>."""
    p = os.path.abspath(path).split('/')
    return p[-4], p[-3], p[-2]


# ---------------------------------------------------------------------------------------------------------------------
# (a) the flux-scale record against the cube
# ---------------------------------------------------------------------------------------------------------------------

def _record_vs_cube(doc, stokes, cube):
    """[problem] of the record's cubes[stokes] against ``cube`` now."""
    from exopipe.phase3.lib.photometry import fluxscale
    tol = _cfg().photometry_fluxscale.FSAPPLD_TOL
    if not os.path.exists(cube):
        return ['no cube %s' % cube]
    try:
        h = fluxscale.read_cube_header(cube)
    except Exception as e:                                   # noqa: BLE001
        return ['cube header unreadable (%s)' % str(e)[:100]]
    if h['fsappld'] is None:
        return ['cube %s has no FSAPPLD (frame-unit cube)' % os.path.basename(cube)]
    d = (doc.get('cubes') or {}).get(stokes) or {}
    bad = []
    if d.get('fsappld') is None or abs(float(d['fsappld']) - h['fsappld']) > tol:
        bad.append('record FSAPPLD %s != cube %.6f' % (d.get('fsappld'), h['fsappld']))
    if os.path.normpath(str(d.get('path'))) != os.path.normpath(cube) or d.get('size') != h['size'] \
            or d.get('mtime_ns') != h['mtime_ns']:
        bad.append('record cube identity (path %s, size %s, mtime_ns %s) differs from the cube now (%s, %s, %s)'
                   % (d.get('path'), d.get('size'), d.get('mtime_ns'), cube, h['size'], h['mtime_ns']))
    return bad


def check_a(root, cell, band, stokes, cube):
    """(a) for one Stokes: (passed, FSAPPLD or None, detail)."""
    from exopipe.phase3.lib.photometry import fluxscale
    sp = fluxscale.sidecar_path(root, cell, band)
    if not os.path.exists(sp):
        try:
            fs = fluxscale.read_cube_header(cube)['fsappld'] if os.path.exists(cube) else None
        except Exception:                                    # noqa: BLE001
            fs = None
        if fs is None:
            return False, None, 'cube %s has no FSAPPLD (frame-unit cube) and no flux-scale record %s' % (cube, sp)
        return False, fs, 'no flux-scale record %s' % sp
    with open(sp) as f:
        doc = json.load(f)
    bad = _record_vs_cube(doc, stokes, cube)
    fs = ((doc.get('cubes') or {}).get(stokes) or {}).get('fsappld')
    if bad:
        return False, fs, '; '.join(bad)
    return True, fs, 'flux-scale record FSAPPLD %.6f = cube, cube unchanged since the cut' % float(fs)


@content_test('photometry_fluxscale_record')
def fs1_record(path, **ctx):
    """FS1: the record parses, names the (cell, subband) of its file name, carries the regeneration's result line, and every
    cube it lists is the cube now with the FSAPPLD it records (check (a) for each Stokes)."""
    from exopipe.phase3.lib.photometry import stamp_engine
    with open(path) as f:
        doc = json.load(f)
    name = os.path.basename(path)
    cell, sb = doc.get('cell'), doc.get('subband')
    if name != '%s_%s_fluxscale.json' % (cell, sb):
        return False, None, 'file name %s does not name the record\'s cell %s and subband %s' % (name, cell, sb)
    cubes = doc.get('cubes') or {}
    if not cubes:
        return False, 0, 'no cube in the record'
    counts, _ = stamp_engine.generation_counts(doc)
    if not counts:
        return False, len(cubes), 'no regeneration result line (": V <n> I <n>") in regen %r' % doc.get('regen')
    bad = []
    for S, d in sorted(cubes.items()):
        bad += ['%s: %s' % (S, x) for x in _record_vs_cube(doc, S, d.get('path') or '')]
    if bad:
        return False, len(cubes), '; '.join(bad)
    return True, len(cubes), '%s cubes unchanged, FSAPPLD %s' % ('+'.join(sorted(cubes)), ', '.join(
        '%s %.6f' % (S, float(cubes[S]['fsappld'])) for S in sorted(cubes)))


# ---------------------------------------------------------------------------------------------------------------------
# (b) generation-5 comparison
# ---------------------------------------------------------------------------------------------------------------------

def gen5_files(gen5_root, cell, band, stokes):
    """{relative path} of generation 5's files of (cell, band, Stokes): <sample>/<band>/<object>/*_<LST>_<YYYYMMDD>T*_<S>.fits
    (each sample's band directory listed, then each object directory)."""
    import fnmatch
    lst, date = cell.split('_', 1)
    pat = '*_%s_%sT*_%s.fits' % (lst, date.replace('-', ''), stokes)
    out = set()
    for smp in sorted(os.listdir(gen5_root)) if os.path.isdir(gen5_root) else ():
        bd = os.path.join(gen5_root, smp, band)
        if smp.startswith('_') or not os.path.isdir(bd):
            continue
        for obj in sorted(os.listdir(bd)):
            od = os.path.join(bd, obj)
            if os.path.isdir(od):
                out.update(os.path.join(smp, band, obj, n) for n in os.listdir(od) if fnmatch.fnmatch(n, pat))
    return out


def _gen5_one(f, root, gen5_root, fs):
    """(slopes [planes] or None, nan pattern equal, why not comparable)."""
    import numpy as np
    from astropy.io import fits
    g5p = os.path.join(gen5_root, os.path.relpath(f, root))
    if not os.path.exists(g5p):
        return None, False, 'no generation-5 file'
    with fits.open(f, memmap=False) as h:
        cut = np.asarray(h[0].data, dtype=np.float32)
        fts = [str(x) for x in h['TIMES'].data['ts']]
    with fits.open(g5p, memmap=False) as h:
        g5 = np.asarray(h[0].data, dtype=np.float32)
        t5 = [str(x) for x in h['TIMES'].data['ts']]
    if t5 != fts or g5.shape != cut.shape:
        return None, False, 'generation-5 file has other frames or shape'
    nan_eq = bool(np.array_equal(np.isnan(g5), np.isnan(cut)))
    m = np.isfinite(g5) & np.isfinite(cut)
    num = np.where(m, g5.astype(np.float64) * cut, 0).sum(axis=(1, 2))
    den = np.where(m, g5.astype(np.float64) ** 2, 0).sum(axis=(1, 2))
    sl = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan) * (fs if fs is not None else np.nan)
    return sl, nan_eq, ''


def check_b(files, root, gen5_root, cell, band, stokes, fs):
    """(b) for one Stokes of a sampled (cell, subband): ``files`` the unit's files under ``root`` (staged or placed), ``fs`` the
    cube's FSAPPLD. -> (passed, median slope x FSAPPLD, detail)."""
    import numpy as np
    fx = _cfg().photometry_fluxscale
    new = set(fx.SAMPLES_NEW_SINCE_GEN5)
    if os.path.realpath(root) == os.path.realpath(gen5_root):
        return False, None, 'the root tested is the generation-5 tree itself'
    mine = {os.path.relpath(f, root) for f in files if _parts(f)[0] not in new}
    g5set = gen5_files(gen5_root, cell, band, stokes)
    g5set = {r for r in g5set if r.split('/')[0] not in new}
    slopes, nan_bad, missing = [], 0, []
    for f in sorted(files):
        if _parts(f)[0] in new:
            continue
        sl, nan_eq, why = _gen5_one(f, root, gen5_root, fs)
        if sl is None:
            missing.append('%s (%s)' % (os.path.basename(f), why))
            continue
        slopes.append(sl)
        nan_bad += not nan_eq
    allsl = np.concatenate(slopes) if slopes else np.array([np.nan])
    fin = np.isfinite(allsl)
    med = float(np.nanmedian(allsl)) if fin.any() else float('nan')
    ok = (mine == g5set and not missing and nan_bad == 0 and math.isfinite(med) and abs(med - 1.0) <= fx.GEN5_SLOPE_TOL)
    detail = ('files %d, generation 5 %d, same set %s%s; NaN pattern differs in %d files; per-plane slope new/gen5 x FSAPPLD over '
              '%d planes: median %.6f (p1 %s, p99 %s; tolerance 1 +- %g)%s'
              % (len(mine), len(g5set), mine == g5set,
                 '' if mine == g5set else ' (only new: %s; only generation 5: %s)' % (sorted(mine - g5set)[:3],
                                                                                    sorted(g5set - mine)[:3]),
                 nan_bad, int(fin.sum()), med,
                 '%.6f' % np.nanpercentile(allsl, 1) if fin.any() else 'nan',
                 '%.6f' % np.nanpercentile(allsl, 99) if fin.any() else 'nan', fx.GEN5_SLOPE_TOL,
                 '; not comparable %d: %s' % (len(missing), missing[:3]) if missing else ''))
    return ok, med, detail


# ---------------------------------------------------------------------------------------------------------------------
# (c) the expected target set
# ---------------------------------------------------------------------------------------------------------------------

def _elevation(ra, dec, lst_deg, lat):
    """Elevation (deg), the writer's formula (phase3_snapshot_cutouts._elevation :142-145), written out."""
    ha = math.radians(ra - lst_deg)
    se = (math.sin(math.radians(dec)) * math.sin(math.radians(lat))
          + math.cos(math.radians(dec)) * math.cos(math.radians(lat)) * math.cos(ha))
    return math.degrees(math.asin(max(-1, min(1, se))))


def planet_radec(body, ts):
    """Topocentric RA, Dec (deg) of ``body`` at the minute of frame time ``ts`` ('YYYYMMDD_HHMMSS'), the writer's computation
    (phase3_snapshot_cutouts.planet_position :124-135: astropy get_body at the site, the GCRS RA/Dec taken as they are, no
    frame transformation), written out without its process-wide cache."""
    from astropy.coordinates import EarthLocation, get_body
    from astropy.time import Time
    import astropy.units as u
    lat, lon, height = _cfg().photometry_cutouts.PLANET_SITE
    loc = EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=height * u.m)
    t = Time('%s-%s-%sT%s:%s:00' % (ts[0:4], ts[4:6], ts[6:8], ts[9:11], ts[11:13]), format='isot', scale='utc')
    b = get_body(body, t, loc)
    return float(b.ra.deg), float(b.dec.deg)


def expected_targets(cts, h0, shape, lists, planets):
    """(set of (sample, object directory), number of list entries selected) the writer selects for a cube with frame times
    ``cts`` (the frames it cut), first valid frame header ``h0`` and frame shape ``shape`` (ny, nx) (write_cutouts :160-186)."""
    import numpy as np
    from astropy.wcs import WCS
    from exopipe.phase3.lib.photometry import tests as T
    from exopipe.phase3.lib.snapshot_engine.phase3_snapshot_search import lst_deg_at, ts_to_epoch_seconds
    cc, ph = _cfg().photometry_cutouts, _cfg().photometry
    lat = cc.PLANET_SITE[0]
    w = WCS(h0).celestial
    ny, nx = shape
    times = [ts_to_epoch_seconds(t) for t in cts]
    lst_mid = lst_deg_at(0.5 * (min(times) + max(times)))

    def inside(ra, dec):
        px, py = w.all_world2pix(ra, dec, 0)
        return bool(np.isfinite(px) and np.isfinite(py) and 0 <= px < nx and 0 <= py < ny)

    exp, entries = set(), 0
    for tag, nm, ra, dec in T.sample_targets(None, lists=lists):
        if _elevation(ra, dec, lst_mid, lat) < cc.MIN_EL_DEG or not inside(ra, dec):
            continue
        exp.add((tag, T.safe_name(nm)))
        entries += 1
    mid = cts[len(cts) // 2]
    first = next(t for t in cts if t[:12] == mid[:12])   # the writer's ephemeris entry: the first frame of the mid frame's
    #                                                      10-minute key ts[:12] (planet_position :126, PLANET_EPHEM_QUANTUM_MIN)
    for body in planets:
        ra0, dec0 = planet_radec(body, first)
        if _elevation(ra0, dec0, lst_mid, lat) < cc.MIN_EL_DEG or not inside(ra0, dec0):
            continue
        exp.add((ph.PLANET_SAMPLE, body.capitalize()))
        entries += 1
    return exp, entries


def check_c(files, cts, h0, shape, lists=None, planets=None):
    """(c) for one Stokes: ``files`` the unit's files of the Stokes; ``cts``, ``h0``, ``shape`` the cube's frames as the
    identity test read them (tests_cutouts.identity_subband frame_info); ``lists`` {sample: list path} (default the lists the
    cutters read, steps_common.sample_paths(cut=True)); ``planets`` (default photometry_cutouts.PLANETS). -> (passed, number of
    distinct objects, detail)."""
    from exopipe.phase3.lib.photometry import steps_common
    if lists is None:
        lists = steps_common.sample_paths('snapshot', cut=True)
    if planets is None:
        planets = _cfg().photometry_cutouts.PLANETS
    exp, entries = expected_targets(cts, h0, shape, lists, planets)
    got = {(_parts(f)[0], _parts(f)[2]) for f in files}
    ok = got == exp and len(files) == len(got)
    return ok, len(got), ('%d files, %d distinct objects; the writer selects %d entries = %d distinct objects%s%s'
                          % (len(files), len(got), entries, len(exp),
                             '; missing %s' % sorted(exp - got)[:5] if exp - got else '',
                             '; unexpected %s' % sorted(got - exp)[:5] if got - exp else ''))


# ---------------------------------------------------------------------------------------------------------------------
# (w) the wideband member weights, recomputed independently
# ---------------------------------------------------------------------------------------------------------------------

def identity_dewarp(frame):
    """The member frame as the group search's dewarp leaves it under an identity screen (bilinear sampling at the pixel's own
    integer coordinates, constant NaN outside), written out: the value is the pixel's own (float32), and NaN where its
    neighbor at +1 row, +1 column or +1 row and column is not finite (the zero-weight corners of the bilinear sum carry a
    NaN into it: 0 x NaN = 0 x inf = NaN); a neighbor beyond the last row or column does not (scipy >= 1.6)."""
    import numpy as np
    a = np.array(frame, dtype=np.float32)
    bad = ~np.isfinite(a)
    nb = np.zeros_like(bad)
    nb[:-1, :] |= bad[1:, :]
    nb[:, :-1] |= bad[:, 1:]
    nb[:-1, :-1] |= bad[1:, 1:]
    a[nb] = np.nan
    return a


def central_rms(img):
    """1.4826 x MAD over the central RMS_CENTRAL_FRAC of the image area (finite pixels), NaN below RMS_MIN_PIXELS - the search's
    noise measure (phase3_snapshot_search.central_rms :606-621), written out."""
    import numpy as np
    fx = _cfg().photometry_fluxscale
    ny, nx = img.shape
    h = int(ny * np.sqrt(fx.RMS_CENTRAL_FRAC) / 2)
    w = int(nx * np.sqrt(fx.RMS_CENTRAL_FRAC) / 2)
    c = img[ny // 2 - h:ny // 2 + h, nx // 2 - w:nx // 2 + w]
    c = c[np.isfinite(c)]
    if c.size < fx.RMS_MIN_PIXELS:
        return float('nan')
    med = np.median(c)
    return float(1.4826 * np.median(np.abs(c - med)))


class WeightCheck:
    """(w): pass as ``member_hook`` to tests_cutouts.wideband_stack (called once per member cube, with the cube in memory);
    ``result()`` gives (passed, worst relative deviation, detail). A member cube without FSAPPLD, a row whose recomputed
    weight is not finite, or no row checked fails."""

    def __init__(self, rows):
        import numpy as np
        fx = _cfg().photometry_fluxscale
        self.tol = fx.WEIGHT_REL_TOL
        wts = sorted({r['ts'] for r in rows})
        n = min(fx.WEIGHT_SAMPLE_FRAMES, len(wts))
        self.pick = {wts[i] for i in sorted(set(np.linspace(0, len(wts) - 1, n).astype(int)))} if wts else set()
        self.worst, self.n, self.frames, self.missing, self.nonfinite = 0.0, 0, set(), [], []

    def __call__(self, sb, F, cube, mix, rows):
        if F is None:
            self.missing.append(sb)
            return
        F = float(F)
        for r in rows:
            if r['subband'] != sb or r['ts'] not in self.pick:
                continue
            rms = central_rms(identity_dewarp(cube[mix[r['member_ts']]]))
            w = 1.0 / (F * rms ** 2) if (math.isfinite(rms) and rms > 0) else float('nan')
            dev = abs(float(r['weight']) / w - 1.0) if math.isfinite(w) and w > 0 else float('nan')
            self.n += 1
            self.frames.add(r['ts'])
            if not math.isfinite(dev):
                self.nonfinite.append('%s %s' % (sb, r['member_ts']))
            else:
                self.worst = max(self.worst, dev)

    def result(self):
        ok = self.n > 0 and not self.missing and not self.nonfinite and self.worst <= self.tol
        return ok, self.worst, ('%d member rows on %d wideband frames: max |table weight / (1/(F rms^2)) - 1| = %.2e (tol %.0e)%s%s'
                                % (self.n, len(self.frames), self.worst, self.tol,
                                   '; NO FSAPPLD on %s (not a rescaled cube)' % self.missing if self.missing else '',
                                   '; no finite recomputed weight for %s' % self.nonfinite[:3] if self.nonfinite else ''))
