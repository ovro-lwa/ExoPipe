"""D-ADMIT (a), slice 1 (Gregg 2026-10-13: "gates against the build's recorded statistics"): the population numbers of the
cell-median gates of the build and of the search, recorded when the build ends in <cell>/coadd_final/build_stats.json. A
night admitted later is judged against the population it did not join and is never added to it. Each number is derived
from the build's own records exactly as its gate computes it:
  stage2       per Stokes, over the finite provisional residual rms values of the build record (membership.stage2_record's
               source: coadd_final/stage2_verdict.csv, else residuals_provisional.csv): median, MAD, the high threshold
               median + OUTLIER_SIGMA x 1.4826 MAD and the low one median - LOW_OUTLIER_SIGMA x 1.4826 MAD
               (coadd.detect_outliers_mad; fewer than 5 values: no gate). The record's own i_thresh_mjy / v_thresh_mjy are
               the high thresholds the build used: they must agree.
  beam_gate    the fitted-beam area (BMAJ x BMIN of the image the co-add admits) of each night the build gated - the
               record's nights, coadd.coadd_beam_gate - and their median as that gate takes it (the upper middle element of
               the sorted areas; fewer than BEAM_GATE_MIN_AREAS: no gate), with the ratio.
  sanity       the cell median qa3 rms of weights.csv as the search's sanity gates take it (deep_search: the upper middle
               element of the finite positive values; fewer than SANITY_MIN_NIGHTS: no gate), the 10x discard and SANITY_I_X.
  search_tier  per Stokes, over the stage-1 rms of residuals_provisional.csv: numpy median, 1.4826 x MAD and the threshold
               median + SEARCH_TIER_SIGMA x that (deep_search; fewer than SEARCH_TIER_MIN: no gate).
The masters build writes it when masters_build.WRITE_BUILD_STATS is True (ruled; production False writes nothing);
`masters --stages build-stats` recomputes it for a cell already built (read only besides the file). Content test (class
build_stats): every number re-derived from the files it names.
"""
import csv
import json
import math
import os

FILE = 'build_stats.json'
VERSION = 1


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float('nan')
    return v


def _rows(p):
    with open(p, newline='') as f:
        return list(csv.DictReader(f))


def _median(vals):
    import numpy as np
    return float(np.median(np.asarray(vals, dtype=float)))


def stage2_source(cell_dir):
    """The record membership.stage2_record reads: (path, rows)."""
    from exopipe.phase3.lib.masters import membership as MB
    fd = os.path.join(cell_dir, 'coadd_final')
    v = os.path.join(fd, MB.VERDICT_FILE)
    if os.path.exists(v):
        rows = _rows(v)
        if rows and {r.get('metric') for r in rows} == {MB.STAGE2_METRIC}:
            return v, rows
    p = os.path.join(cell_dir, MB.BUILD_RECORD)
    return p, (_rows(p) if os.path.exists(p) else [])


def stage2_stats(rows, outlier_sigma, low_sigma):
    """{'I': {...}, 'V': {...}} as coadd.detect_outliers_mad computes them (median and MAD of the finite values)."""
    out = {}
    for st, col in (('I', 'i_rms_mjy'), ('V', 'v_rms_mjy')):
        vals = [x for x in (_f(r.get(col)) for r in rows) if math.isfinite(x)]
        if len(vals) < 5:
            out[st] = dict(n=len(vals), gate=False)
            continue
        med = _median(vals)
        mad = _median([abs(x - med) for x in vals])
        sig = mad * 1.4826
        out[st] = dict(n=len(vals), gate=True, median=med, mad=mad, thr_hi=med + outlier_sigma * sig,
                       thr_lo=med - low_sigma * sig)
    return out


def beam_stats(nights_areas, ratio, min_areas):
    """The co-add beam gate's median as it takes it: the upper middle element of the sorted areas."""
    areas = sorted(a for a in nights_areas.values() if a is not None and math.isfinite(a))
    if len(areas) < min_areas:
        return dict(n=len(areas), gate=False, ratio=ratio)
    return dict(n=len(areas), gate=True, median_area=areas[len(areas) // 2], ratio=ratio)


def sanity_stats(weights_rows, x_i, min_nights):
    """The search's sanity gates' cell median of the weights.csv qa3 rms (the upper middle element)."""
    vals = sorted(x for x in (_f(r.get('qa3_rms_mjy')) for r in weights_rows if r.get('qa3_rms_mjy') not in (None, '', 'nan'))
                  if math.isfinite(x) and x > 0)
    if len(vals) < min_nights:
        return dict(n=len(vals), gate=False, x_discard=10.0, x_i=x_i)
    return dict(n=len(vals), gate=True, qa3_median=vals[len(vals) // 2], x_discard=10.0, x_i=x_i)


def search_tier_stats(prov_rows, sigma, min_n):
    out = {}
    for st, col in (('I', 'i_rms_mjy'), ('V', 'v_rms_mjy')):
        vals = sorted(x for x in (_f(r.get(col)) for r in prov_rows if r.get(col) not in (None, '', 'nan'))
                      if math.isfinite(x) and x > 0)
        if len(vals) < min_n:
            out[st] = dict(n=len(vals), gate=False, sigma=sigma)
            continue
        med = _median(vals)
        mad = _median([abs(x - med) for x in vals]) * 1.4826
        out[st] = dict(n=len(vals), gate=True, median=med, mad=mad, thr=med + sigma * mad, sigma=sigma)
    return out


def compute(cell_dir, lst, scope, co, cfg):
    """The record of one built cell (co: the in-place co-add module, for its beam gate and its thresholds)."""
    import contextlib
    import io
    b, d = cfg.masters_build, cfg.search_detect
    s2_path, s2_rows = stage2_source(cell_dir)
    prov = os.path.join(cell_dir, 'residuals_provisional.csv')
    prov_rows = _rows(prov) if os.path.exists(prov) else []
    wpath = os.path.join(cell_dir, 'coadd_final', 'weights.csv')
    w_rows = _rows(wpath) if os.path.exists(wpath) else []
    nights = [r['date'] for r in prov_rows]
    with contextlib.redirect_stdout(io.StringIO()):
        _out, areas = co.coadd_beam_gate(lst, scope, nights)
    images = {}
    fl = scope
    for n in nights:
        sci = co.find_latest_science_dir(lst, n)
        img = None
        if sci:
            with contextlib.redirect_stdout(io.StringIO()):
                img = (co.find_subband_alignment_image(sci, fl[:-3]) if fl.endswith('MHz')
                       else co.find_wideband_alignment_image(sci, fl, lst, n))
        images[n] = img
    stats = dict(version=VERSION, cell=f'{lst}_{scope}',
                 sources=dict(stage2=s2_path, residuals_provisional=prov, weights=wpath),
                 stage2=stage2_stats(s2_rows, float(co.OUTLIER_SIGMA), float(co.LOW_OUTLIER_SIGMA)),
                 stage2_recorded_thr={st: sorted({round(_f(r.get(col)), 6) for r in s2_rows if math.isfinite(_f(r.get(col)))})
                                      for st, col in (('I', 'i_thresh_mjy'), ('V', 'v_thresh_mjy'))},
                 beam_gate=dict(beam_stats(areas, float(b.BEAM_GATE_RATIO), int(b.BEAM_GATE_MIN_AREAS)),
                                nights={n: dict(image=images.get(n), area=areas.get(n)) for n in nights}),
                 sanity=sanity_stats(w_rows, float(d.SANITY_X), int(d.SANITY_MIN_NIGHTS)),
                 search_tier=search_tier_stats(prov_rows, float(d.SEARCH_TIER_SIGMA), int(d.SEARCH_TIER_MIN)))
    return stats


def write(cell_dir, stats):
    p = os.path.join(cell_dir, 'coadd_final', FILE)
    with open(p + '.tmp', 'w') as f:
        json.dump(stats, f, indent=1, sort_keys=True)
    os.replace(p + '.tmp', p)
    return p


def read(cell_dir):
    p = os.path.join(cell_dir, 'coadd_final', FILE)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def recheck(path, cfg, co_thresholds):
    """[problems] of a build_stats.json: every number re-derived from the files it names (the beam areas from the images
    it names, read again). co_thresholds = (OUTLIER_SIGMA, LOW_OUTLIER_SIGMA) in force."""
    from astropy.io import fits
    with open(path) as f:
        s = json.load(f)
    probs = []
    b, d = cfg.masters_build, cfg.search_detect
    src = s.get('sources', {})

    def close(a, b_, what):
        if isinstance(a, dict) and isinstance(b_, dict):
            for k in set(a) | set(b_):
                close(a.get(k), b_.get(k), f'{what}.{k}')
        elif isinstance(a, float) or isinstance(b_, float):
            if not (isinstance(a, (int, float)) and isinstance(b_, (int, float))
                    and abs(a - b_) <= 1e-9 * max(1.0, abs(a), abs(b_))):
                probs.append(f'{what}: recorded {b_!r}, re-derived {a!r}')
        elif a != b_:
            probs.append(f'{what}: recorded {b_!r}, re-derived {a!r}')
    s2_rows = _rows(src['stage2']) if os.path.exists(src.get('stage2', '')) else []
    close(stage2_stats(s2_rows, *co_thresholds), s.get('stage2'), 'stage2')
    for st in ('I', 'V'):
        rec = s.get('stage2', {}).get(st, {})
        thr = s.get('stage2_recorded_thr', {}).get(st) or []
        if rec.get('gate') and thr and not all(abs(t - rec['thr_hi']) <= 1e-3 * max(1.0, abs(t)) for t in thr):
            probs.append(f'stage2.{st}: the build record\'s threshold {thr} is not the re-derived high threshold {rec["thr_hi"]}')
    prov_rows = _rows(src['residuals_provisional']) if os.path.exists(src.get('residuals_provisional', '')) else []
    close(search_tier_stats(prov_rows, float(d.SEARCH_TIER_SIGMA), int(d.SEARCH_TIER_MIN)), s.get('search_tier'), 'search_tier')
    w_rows = _rows(src['weights']) if os.path.exists(src.get('weights', '')) else []
    close(sanity_stats(w_rows, float(d.SANITY_X), int(d.SANITY_MIN_NIGHTS)), s.get('sanity'), 'sanity')
    nights = s.get('beam_gate', {}).get('nights', {})
    areas = {}
    for n, x in nights.items():
        img = x.get('image')
        if img and os.path.exists(img):
            try:
                h = fits.getheader(img)
                areas[n] = float(h['BMAJ']) * float(h['BMIN'])
            except Exception:
                areas[n] = None
        else:
            areas[n] = None
        if x.get('area') is not None and areas[n] is not None and abs(areas[n] - x['area']) > 1e-12 * max(1.0, abs(x['area'])):
            probs.append(f'beam_gate.nights.{n}: recorded area {x["area"]}, the image says {areas[n]}')
    bs = beam_stats({n: x.get('area') for n, x in nights.items()}, float(b.BEAM_GATE_RATIO), int(b.BEAM_GATE_MIN_AREAS))
    close(bs, {k: v for k, v in s.get('beam_gate', {}).items() if k != 'nights'}, 'beam_gate')
    return probs
