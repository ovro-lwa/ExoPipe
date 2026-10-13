"""Content tests of the `exopipe3 masters --tier sublst` product classes (discovered by core/validate: lib/<step>/tests*.py).

Every test checks CONTENT - header values against the data, the sibling products, the admission record, the frozen deep
record or the master they were built from - never only presence. A test returns (passed, value, detail). Every test works
on a production file with no context (``exopipe3 verify <class> <file>``); the step passes context where it has it.

  sublst_input_copy       a prep copy (coadd_<LST>/..., catalog_inputs/bdsf_catalogs_aug_<LST>/..., parent_catalogs_<LST>/...):
                          md5 = its source's md5 (at copy time when the step gives it, else the source now - the cpv rule)
  sublst_input_manifest   catalog_inputs/roots_<LST>.csv: one row per band (15 subbands + wideband groups), each band's 5
                          co-add files + residuals_provisional.csv present, >= PREP_MIN_PARENTS parent files, footprints
  sublst_job_plan         _plans/sublst_jobs_<LST>[_<night>][_deepflow].txt: well-formed jobs, I and V per (cell, subband), none excluded now
  sublst_admission        <cell>/<sb>/<S>/admission_*.csv: production columns; used frames per window = each window's NFRAMES;
                          a window file exists exactly where frames were used
  sublst_fluxdc_record    <cell>/<sb>/I/fluxdc_*.json: flux_dc finite > 0; B = the master's pixel factor now; master and Stokes I
                          cube unchanged since (size, mtime) - a record of other inputs is stale
  sublst_window           sublst_<cell>_<sb>_<S>_w<k>.fits: header census (FLUXBAKE, FTBASIS, PRODUCT, TEFFSEC, pixel factor =
                          the master's B, FLUXDC = the I record) - the CENSUS of production test (a); the time block, NFRAMES and
                          NOISEMED against the admission rows; NaN mask = (frame count == 0) on the _nframes sibling
  sublst_window_nframes   _nframes.fits: integer counts 0 .. NFRAMES, no flux cards
  sublst_window_pb        _pb.fits: finite, within masters_snapshot.PB_LIBRARY_VALUE_RANGE, PBWIN0/1 = LSTW0/1 -+ T/120, no flux cards
  sublst_window_master    <LST>_masters/<sb>/<S>/sublst_master_*_w<k>.fits: COADDOP, NCOADD = weighted nights (>= 2), CONSTDIG,
                          HISTORY constituents, weights = the frozen deep record (in_master 0 or vetoed -> 0), no excluded night
                          with weight (loo:93-134; R-RAT2-H1-hygiene)
  sublst_loo_residual     *_resid_loo.fits (arm A): LOONCOAD = the independent count (>= 2, Q11), LOOMSTR, LSTBIN, no BINMSTR,
                          DURATION, and the pixel identity residual = stack - (M W - w a) / (W - w) on the pixels finite in stack
                          and master (Jupiter-masked pixels differ by construction: masters_sublst.LOO_IDENTITY)
  sublst_window_sample    the <LST> directory: production content test (a), sublst_reserve_test_windows.py, in place
The sub-LST parent catalogues are class master_catalog (lib/masters/tests.py, shared with the deep tier). Wideband windows and
production content test (b) (sublst_reserve_test_residuals.py) are frames' classes (refactor decision 6, crosscheck 1.1 #3, #13).
"""
import csv
import glob
import hashlib
import json
import os
import re
import threading

import numpy as np

from exopipe.phase3.core.validate import content_test

_WIN_RE = re.compile(r'^sublst_(?P<lst>\d\dh)_(?P<night>\d{4}-\d\d-\d\d)_(?P<band>\d+MHz|Red|Green|Blue|Full)_(?P<S>[IV])'
                     r'_w(?P<k>\d)(?P<sfx>_nframes|_pb|_resid_loo)?\.fits$')
_MASTER_RE = re.compile(r'^sublst_master_(?P<lst>\d\dh)_(?P<band>\d+MHz|Red|Green|Blue|Full)_(?P<S>[IV])_w(?P<k>\d)\.fits$')
_FLUX_CARDS = ('FLUXBAKE', 'FLUXDC', 'FLUXDCN', 'FLUXDCF', 'SHSCALE', 'FLUXCORR', 'FLUXABS', 'FLUXBND', 'FLUXVER')
ADMISSION_COLUMNS = ['ts', 'ts_int', 'date_obs', 't_mid_unix', 'lst_min', 'window', 'finite_frac', 'beam_area_deg2',
                     'noise_jyb', 'sun_elev_deg', 'sun_branch', 'used', 'refused']        # phase3_sublst_stack.py:398-405
_CACHE = {}
_LOCK = threading.Lock()


def _cfg():
    from exopipe.phase3.config import CFG
    return CFG


def _sec():
    return _cfg().masters_sublst


def _key(p):
    st = os.stat(p)
    return (os.path.abspath(p), st.st_mtime_ns, st.st_size)


def _cached(kind, p, fn, keep=48):
    k = (kind,) + _key(p)
    with _LOCK:
        if k in _CACHE:
            return _CACHE[k]
    v = fn(p)
    with _LOCK:
        if len(_CACHE) >= keep:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[k] = v
    return v


def _hdr(p):
    from astropy.io import fits
    return _cached('h', p, lambda q: fits.getheader(q, 0))


def _data(p):
    from astropy.io import fits
    return _cached('d', p, lambda q: np.squeeze(fits.getdata(q, 0)), keep=12)


def _md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def _parse_window(path):
    m = _WIN_RE.match(os.path.basename(path))
    if not m:
        raise ValueError(f'not a sub-LST window product name: {os.path.basename(path)}')
    d = m.groupdict()
    d['cell'] = f"{d['lst']}_{d['night']}"
    d['k'] = int(d['k'])
    d['sfx'] = d['sfx'] or ''
    return d


def _fc():
    from exopipe.phase3.lib.masters import flux_correction as FC
    return FC


def _T_mod():
    from exopipe.phase3.lib.masters import tier_sublst as T
    return T


def _master_of(d, ctx):
    return ctx.get('master') or _T_mod().master_path(d['lst'], d['band'])


def _fluxdc_path(path, d):
    sb_dir = os.path.dirname(os.path.dirname(os.path.abspath(path)))
    return os.path.join(sb_dir, 'I', f"fluxdc_{d['cell']}_{d['band']}.json")


def _admission_rows(path, d):
    p = os.path.join(os.path.dirname(os.path.abspath(path)), f"admission_{d['cell']}_{d['band']}_{d['S']}.csv")
    return _cached('csv', p, lambda q: list(csv.DictReader(open(q))))


def _T():
    return float(_sec().INTEGRATION_S)


def _verdict(problems, value, ok_detail):
    if problems:
        return False, value, '; '.join(problems[:6]) + (f' (+{len(problems) - 6} more)' if len(problems) > 6 else '')
    return True, value, ok_detail


# =====================================================================================================================
# prep
# =====================================================================================================================

def prep_source_of(path):
    """The source a prep copy was made from, by the prep layout rule (sublst_prep_lst.sh:24-50); None if not a prep copy."""
    T = _T_mod()
    coadd_sb, parents = _cfg().paths.COADD_SB_ROOT, T.prep_parents_dir()
    p = os.path.abspath(path)
    parts = p.split(os.sep)
    b = os.path.basename(p)
    m = re.match(r'^coadd_(\d\dh)$', parts[-4]) if len(parts) >= 4 else None
    if m and parts[-2] == 'coadd_final':
        cellb = parts[-3]
        if b == 'weights.csv':
            return os.path.join(coadd_sb, cellb, 'coadd_final', 'weights.csv')
        return os.path.join(T.frozen_masters(), cellb, 'coadd_final', b)
    if b == 'residuals_provisional.csv' and len(parts) >= 3 and re.match(r'^coadd_\d\dh$', parts[-3]):
        return os.path.join(coadd_sb, parts[-2], 'residuals_provisional.csv')
    if re.match(r'^bdsf_catalogs_aug_\d\dh$', parts[-2]):
        f1 = os.path.join(_sec().PREP_FOOTPRINT_DIR, b)
        return f1 if os.path.isfile(f1) else os.path.join(parents, b)
    if re.match(r'^parent_catalogs_\d\dh$', parts[-2]):
        return os.path.join(parents, b)
    return None


@content_test('sublst_input_copy')
def copy_md5_equals_source(path, source=None, source_md5=None, **ctx):
    """md5 of the copy = md5 of its source (at copy time when the step passes it; else the source read now)."""
    got = _md5(path)
    if source_md5:
        return got == source_md5, got[:8], f'copy {got[:8]} vs source at copy time {source_md5[:8]} ({source or "?"})'
    src = source or prep_source_of(path)
    if not src or not os.path.isfile(src):
        return False, got[:8], f'source of this prep copy not found ({src})'
    want = _md5(src)
    return got == want, got[:8], f'copy {got[:8]} vs source now {want[:8]} ({src})' + (
        '' if got == want else ' - a live source (weights.csv, residuals_provisional.csv) changed since prep (B-20)?')


@content_test('sublst_input_manifest')
def roots_rows_and_files(path, lst=None, **ctx):
    """roots_<LST>.csv: one row '<LST>_<b>,,<coadd root>' per band in prep order, and every band's prep products present."""
    s = _sec()
    lst = lst or re.match(r'^roots_(\d\dh)\.csv$', os.path.basename(path)).group(1)
    bands = list(_cfg().instrument.ALL_SUBBANDS) + list(s.PREP_WIDEBAND)
    rows = [l.strip().split(',') for l in open(path) if l.strip()]
    problems = []
    got = [r[0] for r in rows]
    if got != [f'{lst}_{b}' for b in bands]:
        problems.append(f'rows {got[:3]}... != the {len(bands)} bands in prep order')
    roots = {r[2] for r in rows if len(r) == 3}
    if len(roots) != 1 or any(len(r) != 3 or r[1] != '' for r in rows):
        problems.append(f'malformed rows or several coadd roots {sorted(roots)[:3]}')
    root = next(iter(roots)) if roots else ''
    for b in bands:
        need = [os.path.join(root, f'{lst}_{b}', 'coadd_final', f.format(b=b)) for f in s.PREP_FILES] + \
               [os.path.join(root, f'{lst}_{b}', 'coadd_final', 'weights.csv'),
                os.path.join(root, f'{lst}_{b}', 'residuals_provisional.csv')]
        miss = [p for p in need if not os.path.isfile(p)]
        if miss:
            problems.append(f'{b}: missing {miss[0]}')
    out = os.path.dirname(os.path.dirname(os.path.abspath(path)))
    npar = len(glob.glob(os.path.join(out, f'parent_catalogs_{lst}', f'master_R075_{lst}_*')))
    if npar < s.PREP_MIN_PARENTS:
        problems.append(f'{npar} parent catalogue files < {s.PREP_MIN_PARENTS}')
    fp = [b for b in _cfg().instrument.ALL_SUBBANDS
          if not os.path.isfile(os.path.join(out, 'catalog_inputs', f'bdsf_catalogs_aug_{lst}', f'master_R075_{lst}_{b}.gaul.csv'))]
    if fp:
        problems.append(f'no footprint catalogue for {fp[:4]}')
    return _verdict(problems, len(rows), f'{len(rows)} bands, products present, {npar} parent files')


# =====================================================================================================================
# admit
# =====================================================================================================================

@content_test('sublst_job_plan')
def plan_well_formed(path, lst=None, **ctx):
    """Every line 'cell subband Stokes' of the LST; I and V for every (cell, subband); no duplicate; none excluded now."""
    from exopipe.phase3.core import exclusions
    m = re.match(r'^sublst_jobs_(\d\dh)(?:_\d{4}-\d\d-\d\d)?(_deepflow)?\.txt$', os.path.basename(path))
    lst = lst or (m.group(1) if m else None)
    lines = [l.split() for l in open(path) if l.strip() and not l.startswith('#')]
    problems, seen, pairs = [], set(), {}
    ex = exclusions.load()
    sbs = set(_cfg().instrument.ALL_SUBBANDS)
    for f in lines:
        if len(f) != 3 or not re.match(rf'^{lst}_\d{{4}}-\d\d-\d\d$', f[0]) or f[1] not in sbs or f[2] not in ('I', 'V'):
            problems.append(f'malformed job {" ".join(f)}')
            continue
        if tuple(f) in seen:
            problems.append(f'duplicate job {" ".join(f)}')
        seen.add(tuple(f))
        pairs.setdefault((f[0], f[1]), set()).add(f[2])
        if ex.is_excluded(lst, f[1], f[0].split('_', 1)[1]):
            problems.append(f'{f[0]} {f[1]} is excluded in data_decisions.csv now (plan stale)')
    half = [k for k, v in pairs.items() if v != {'I', 'V'}]
    if half:
        problems.append(f'{len(half)} (cell, subband) without both Stokes, e.g. {half[0]}')
    return _verdict(problems, len(lines), f'{len(lines)} jobs, {len(pairs)} (cell, subband)')


# =====================================================================================================================
# stack
# =====================================================================================================================

@content_test('sublst_admission')
def admission_matches_windows(path, **ctx):
    """Production columns; used rows have no refusal and a window 0..N-1; per window the used frames = its NFRAMES, and
    a window image exists exactly where frames were used."""
    m = re.match(r'^admission_(\d\dh_\d{4}-\d\d-\d\d)_(\S+)_([IV])\.csv$', os.path.basename(path))
    if not m:
        return False, None, 'not an admission CSV name'
    cell, band, S = m.groups()
    with open(path) as f:
        rd = csv.DictReader(f)
        cols = rd.fieldnames or []
        rows = list(rd)
    problems = []
    if cols != ADMISSION_COLUMNS:
        problems.append(f'columns {cols} != production {ADMISSION_COLUMNS}')
    nwin = int(_sec().N_WINDOWS)
    used = {}
    for r in rows:
        if r.get('used') not in ('0', '1'):
            problems.append(f'used={r.get("used")!r} at {r.get("ts")}')
            continue
        if r['used'] == '1':
            if r.get('refused'):
                problems.append(f'{r["ts"]} used but refused {r["refused"]!r}')
            try:
                k = int(r['window'])
            except ValueError:
                problems.append(f'{r["ts"]} used without a window')
                continue
            if not 0 <= k < nwin:
                problems.append(f'{r["ts"]} window {k} outside 0..{nwin - 1}')
            used[k] = used.get(k, 0) + 1
    d = os.path.dirname(os.path.abspath(path))
    for k in range(nwin):
        w = os.path.join(d, f'sublst_{cell}_{band}_{S}_w{k}.fits')
        n = used.get(k, 0)
        if os.path.isfile(w) != (n > 0):
            problems.append(f'w{k}: {n} used frames but window file {"present" if os.path.isfile(w) else "absent"}')
        elif n and int(_hdr(w).get('NFRAMES', -1)) != n:
            problems.append(f'w{k}: NFRAMES {_hdr(w).get("NFRAMES")} != {n} used frames')
    nu = sum(used.values())
    return _verdict(problems, nu, f'{len(rows)} frames, {nu} used in {len(used)} windows')


@content_test('sublst_fluxdc_record')
def fluxdc_current(path, master=None, **ctx):
    """flux_dc finite > 0 with its source count; B = the master's pixel factor now; master and Stokes I cube unchanged
    (size, integer mtime) since the record (phase3_sublst_stack.fluxdc_identity: a record of other inputs is never used)."""
    rec = json.load(open(path))
    problems = []
    for k in ('cube_I', 'cube_I_size', 'cube_I_mtime', 'master', 'master_size', 'master_mtime', 'cell', 'subband', 'flux_dc',
              'n_src', 'n_frames', 'B'):
        if k not in rec:
            problems.append(f'no {k}')
    if problems:
        return False, None, '; '.join(problems)
    m = re.match(r'^fluxdc_(\d\dh_\d{4}-\d\d-\d\d)_(\S+)\.json$', os.path.basename(path))
    if not m or (rec['cell'], rec['subband']) != m.groups():
        problems.append(f'record of {rec["cell"]} {rec["subband"]} under the name {os.path.basename(path)}')
    fdc = rec['flux_dc']
    if fdc is None or not np.isfinite(float(fdc)) or float(fdc) <= 0:
        problems.append(f'flux_dc {fdc} not measurable ({rec["n_src"]} sources)')
    for kind in ('master', 'cube_I'):
        p = rec[kind]
        if not os.path.isfile(p):
            problems.append(f'{kind} {p} missing')
            continue
        st = os.stat(p)
        if st.st_size != rec[f'{kind}_size'] or int(st.st_mtime) != rec[f'{kind}_mtime']:
            problems.append(f'{kind} {os.path.basename(p)} changed since the record (size/mtime)')
    mp = master or rec['master']
    if os.path.isfile(mp):
        B = _fc().pixel_factor(_hdr(mp))
        tol = _sec().WINDOW_TEST_TOL['factor']
        if B is None or abs(float(rec['B']) - B) > tol:
            problems.append(f'B {rec["B"]} != master pixel factor {B}')
    return _verdict(problems, fdc, f'flux_dc {fdc} from {rec["n_src"]} sources, B {rec["B"]}')


@content_test('sublst_window')
def window_header_census(path, **ctx):
    """FLUXBAKE True, FTBASIS, PRODUCT card, LSTHOUR/LSTWIN = the name, TEFFSEC = NFRAMES x T, pixel factor = the master's
    B within 1e-9, FLUXDC = the Stokes I record within 1e-12 (production test (a) census, sublst_reserve_test_windows.py:60-74)."""
    d = _parse_window(path)
    s = _sec()
    tol = s.WINDOW_TEST_TOL
    h = _hdr(path)
    problems = []
    if h.get('FLUXBAKE') is not True:
        problems.append(f'FLUXBAKE {h.get("FLUXBAKE")}')
    if not h.get('FTBASIS'):
        problems.append('no FTBASIS')
    if h.get('PRODUCT') != s.PRODUCT_CARD:
        problems.append(f'PRODUCT {h.get("PRODUCT")!r} != {s.PRODUCT_CARD!r}')
    if str(h.get('LSTHOUR', '')).strip() != d['lst'] or h.get('LSTWIN') != d['k']:
        problems.append(f'LSTHOUR/LSTWIN {h.get("LSTHOUR")}/{h.get("LSTWIN")} != name {d["lst"]}/w{d["k"]}')
    nf = int(h.get('NFRAMES', 0) or 0)
    if nf <= 0 or abs(float(h.get('TEFFSEC', -1)) - round(_T() * nf, 3)) > 1e-3:
        problems.append(f'NFRAMES {nf} / TEFFSEC {h.get("TEFFSEC")} inconsistent')
    got = _fc().pixel_factor(h) if h.get('SHSCALE') is not None else None
    mp = _master_of(d, ctx)
    B = _fc().pixel_factor(_hdr(mp)) if os.path.isfile(mp) else None
    if got is None or B is None or abs(got - B) >= tol['factor']:
        problems.append(f'pixel factor {got} != master {os.path.basename(mp)} B {B}')
    rp = _fluxdc_path(path, d)
    rec = json.load(open(rp)) if os.path.isfile(rp) else None
    if rec is None or rec.get('flux_dc') is None or abs(float(h.get('FLUXDC', np.nan)) - float(rec['flux_dc'])) >= tol['fluxdc']:
        problems.append(f'FLUXDC {h.get("FLUXDC")} != Stokes I record {None if rec is None else rec.get("flux_dc")} ({rp})')
    return _verdict(problems, got, f'factor {got} = master B, FLUXDC = record, {nf} frames')


@content_test('sublst_window')
def window_matches_admission(path, **ctx):
    """NFRAMES = the used admission rows of the window; DATE-OBS = the first one's date_obs; DATE-END = the last one's + T;
    MJD-MID = their 1/noise^2-weighted mean midpoint; NOISEMED = median(noise) / FLUXDC x pixel factor (test (a) :111-115)."""
    from astropy.time import Time
    d = _parse_window(path)
    tol = _sec().WINDOW_TEST_TOL
    h = _hdr(path)
    sel = [r for r in _admission_rows(path, d) if r['used'] == '1' and r['window'] == str(d['k'])]
    if not sel:
        return False, 0, 'no used admission row for this window'
    problems = []
    if int(h.get('NFRAMES', -1)) != len(sel):
        problems.append(f'NFRAMES {h.get("NFRAMES")} != {len(sel)} used rows')
    if str(h.get('DATE-OBS')) != sel[0]['date_obs']:
        problems.append(f'DATE-OBS {h.get("DATE-OBS")} != first frame {sel[0]["date_obs"]}')
    t_end = Time(sel[-1]['date_obs'], format='isot', scale='utc').unix + _T()
    if abs(Time(str(h.get('DATE-END')), format='isot', scale='utc').unix - t_end) >= tol['time_s']:
        problems.append(f'DATE-END {h.get("DATE-END")} != last frame + T')
    ts = np.array([float(r['t_mid_unix']) for r in sel])
    noise = np.array([float(r['noise_jyb']) for r in sel])
    mjd = Time(float(np.average(ts, weights=1.0 / noise ** 2)), format='unix', scale='utc').mjd
    if abs(float(h.get('MJD-MID', np.nan)) - mjd) >= tol['mjd_d']:
        problems.append(f'MJD-MID {h.get("MJD-MID")} != weighted frame midpoint {mjd:.9f}')
    f = _fc().pixel_factor(h) if h.get('SHSCALE') is not None else None
    if f is not None and h.get('FLUXDC'):
        want = float(np.median(noise)) / float(h['FLUXDC']) * f
        if abs(float(h.get('NOISEMED', np.nan)) / want - 1.0) > tol['noise_rel']:
            problems.append(f'NOISEMED {h.get("NOISEMED")} != median noise / FLUXDC x factor {want:.6g}')
    return _verdict(problems, len(sel), f'{len(sel)} frames; time block and noise agree with the admission record')


@content_test('sublst_window')
def window_siblings(path, **ctx):
    """_nframes and _pb on the same grid, without flux cards; the image is NaN exactly where no frame contributed."""
    d = _parse_window(path)
    base = path[:-5]
    nfp, pbp = base + '_nframes.fits', base + '_pb.fits'
    problems = []
    for p in (nfp, pbp):
        if not os.path.isfile(p):
            return False, None, f'sibling {os.path.basename(p)} missing'
        bad = [c for c in _FLUX_CARDS if c in _hdr(p)]
        if bad:
            problems.append(f'{os.path.basename(p)} carries flux cards {bad}')
    img, nf, pb = _data(path), _data(nfp), _data(pbp)
    if not (img.shape == nf.shape == pb.shape):
        problems.append(f'shapes {img.shape} {nf.shape} {pb.shape} differ')
    else:
        mism = int(np.sum(np.isnan(img) != (nf == 0)))
        if mism:
            problems.append(f'{mism} pixels where NaN != (frame count == 0)')
        if nf.max() > int(_hdr(path).get('NFRAMES', 0)):
            problems.append(f'frame count {nf.max()} > NFRAMES {_hdr(path).get("NFRAMES")}')
    return _verdict(problems, int(np.isnan(img).sum()), f'{d["band"]} w{d["k"]}: siblings consistent')


@content_test('sublst_window_nframes')
def nframes_counts(path, **ctx):
    """Integer per-pixel counts, 0 <= count <= NFRAMES, some pixel > 0, no flux cards."""
    h = _hdr(path)
    a = _data(path)
    problems = []
    if not np.issubdtype(a.dtype, np.integer):
        problems.append(f'dtype {a.dtype} not integer')
    nf = int(h.get('NFRAMES', 0) or 0)
    if a.min() < 0 or a.max() > nf or a.max() <= 0:
        problems.append(f'counts {a.min()}..{a.max()} outside 1..NFRAMES {nf}')
    bad = [c for c in _FLUX_CARDS if c in h]
    if bad:
        problems.append(f'flux cards {bad} on a count map')
    return _verdict(problems, int(a.max()), f'counts {a.min()}..{a.max()} of {nf} frames')


@content_test('sublst_window_pb')
def window_pb_values(path, **ctx):
    """The window model PB: finite, within masters_snapshot.PB_LIBRARY_VALUE_RANGE, some pixel above 0.1; PBWIN0/PBWIN1 =
    LSTW0 - T/120 / LSTW1 + T/120 (phase3_sublst_stack.py:466-469); no flux cards."""
    h = _hdr(path)
    a = _data(path)
    lo, hi = _cfg().masters_snapshot.PB_LIBRARY_VALUE_RANGE
    problems = []
    if not np.isfinite(a).all():
        problems.append(f'{int((~np.isfinite(a)).sum())} non-finite pixels')
    elif a.min() < lo or a.max() > hi or a.max() <= 0.1:
        problems.append(f'values {a.min():.4g}..{a.max():.4g} outside {lo}..{hi} or never above 0.1')
    dt = _T() / 120.0
    for card, src, sign in (('PBWIN0', 'LSTW0', -1), ('PBWIN1', 'LSTW1', 1)):
        if card not in h or src not in h or abs(float(h[card]) - (float(h[src]) + sign * dt)) > 1e-9:
            problems.append(f'{card} {h.get(card)} != {src} {h.get(src)} {"-+"[sign > 0]} T/120')
    bad = [c for c in _FLUX_CARDS if c in h]
    if bad:
        problems.append(f'flux cards {bad} on a PB map')
    return _verdict(problems, float(a.max()) if a.size else None, f'PB {a.min():.4g}..{a.max():.4g}')


# =====================================================================================================================
# loo
# =====================================================================================================================

def _weights_csv(p):
    def rd(q):
        with open(q) as f:
            return [(r['night'], float(r['weight'])) for r in csv.DictReader(f)]
    return _cached('w', p, rd)


def _vetoed(lst, band, night):
    from exopipe.phase3.core import env
    env.legacy()
    from phase3_tier_rules import membership_vetoed      # clv15, in place (the pinned veto list, md5-checked by the module)
    return membership_vetoed(lst, band, night)


@content_test('sublst_window_master')
def window_master_record(path, lst=None, subband=None, **ctx):
    """COADDOP; NCOADD = weighted nights >= LOO_MIN_WEIGHTED_NIGHTS; CONSTDIG = md5 of the sorted night list; HISTORY lists
    the weighted nights as constituents and the others as NOT in the Master; weights = the frozen deep record (in_master 0
    or a membership veto -> 0; phase3_sublst_loo.py:67-78); no night excluded in data_decisions.csv carries weight."""
    from exopipe.phase3.core import exclusions
    s = _sec()
    m = _MASTER_RE.match(os.path.basename(path))
    if not m:
        return False, None, 'not a window master name'
    lst, band, k = m.group('lst'), m.group('band'), int(m.group('k'))
    h = _hdr(path)
    wp = path[:-5] + '_weights.csv'
    if not os.path.isfile(wp):
        return False, None, f'no weights record {os.path.basename(wp)}'
    rows = _weights_csv(wp)
    nights = [n for n, _ in rows]
    weighted = sorted(n for n, w in rows if w > 0)
    problems = []
    if h.get('COADDOP') != s.WINDOW_COADDOP:
        problems.append(f'COADDOP {h.get("COADDOP")!r}')
    if h.get('LSTWIN') != k:
        problems.append(f'LSTWIN {h.get("LSTWIN")} != w{k}')
    if int(h.get('NCOADD', -1)) != len(weighted) or len(weighted) < s.LOO_MIN_WEIGHTED_NIGHTS or len(rows) < 2:
        problems.append(f'NCOADD {h.get("NCOADD")} vs {len(weighted)} weighted of {len(rows)} nights')
    if h.get('CONSTDIG') != hashlib.md5(','.join(sorted(nights)).encode()).hexdigest():
        problems.append('CONSTDIG != md5 of the sorted night list')
    hist = [str(x) for x in h.get('HISTORY', [])]
    cons = sorted(x.rsplit(' ', 1)[1] for x in hist if x.startswith('window-Master constituent '))
    notin = sorted(x.rsplit(' ', 1)[1] for x in hist if x.startswith('NOT in the Master'))
    if cons != weighted or notin != sorted(n for n, w in rows if w <= 0):
        problems.append(f'HISTORY constituents {len(cons)} / not-in {len(notin)} disagree with the weights record')
    if 'JUPMASK' not in h:
        problems.append('no JUPMASK card')
    fr = _T_mod().frozen_weights(lst, band)
    if not os.path.isfile(fr):
        problems.append(f'frozen record {fr} missing')
    else:
        want = {}
        for r in csv.DictReader(open(fr)):
            try:
                want[r['date']] = 0.0 if (str(r.get('in_master', '1')).strip() == '0' or _vetoed(lst, band, r['date'])) \
                    else float(r['weight'])
            except (KeyError, ValueError, TypeError):
                pass
        off = [n for n, w in rows if n not in want or abs(w - want[n]) > 1e-9 * max(1.0, abs(w))]
        if off:
            problems.append(f'{len(off)} night weight(s) != the frozen deep record, e.g. {off[0]}')
    ex = exclusions.load()
    exw = [n for n in weighted if ex.is_excluded(lst, band, n)]
    if exw:
        problems.append(f'excluded night(s) with weight: {exw}')
    return _verdict(problems, len(weighted), f'{len(weighted)} weighted of {len(rows)} nights; record consistent')


@content_test('sublst_loo_residual')
def loo_residual_identity(path, master_root=None, **ctx):
    """LOONCOAD = weighted nights of the window master - (1 if this night is weighted) and >= PER_NIGHT_MIN (Q11);
    LOOMSTR, LSTBIN, no BINMSTR, FLUXBAKE carried, DURATION; NaN wherever the stack is NaN; residual = stack - LOO with
    LOO = (M W - w a) / (W - w) on the pixels finite in stack and master, within LOO_IDENTITY (pure check)."""
    s = _sec()
    d = _parse_window(path)
    if d['sfx'] != '_resid_loo':
        return False, None, 'not a leave-one-out residual name'
    S_dir = os.path.dirname(os.path.abspath(path))
    lst_dir = os.path.dirname(os.path.dirname(os.path.dirname(S_dir)))
    root = os.path.dirname(lst_dir)
    mroot = master_root or os.path.join(root, f"{d['lst']}_masters")
    mp = os.path.join(mroot, d['band'], d['S'], f"sublst_master_{d['lst']}_{d['band']}_{d['S']}_w{d['k']}.fits")
    stack = path[:-len('_resid_loo.fits')] + '.fits'
    for p in (mp, mp[:-5] + '_weights.csv', stack):
        if not os.path.isfile(p):
            return False, None, f'{p} missing'
    h, hm = _hdr(path), _hdr(mp)
    rows = dict(_weights_csv(mp[:-5] + '_weights.csv'))
    problems = []
    if d['night'] not in rows:
        return False, None, f'night {d["night"]} not in the window master weights'
    wn = rows[d['night']]
    nw = sum(1 for w in rows.values() if w > 0)
    want = nw - (1 if wn > 0 else 0)
    if int(h.get('LOONCOAD', -1)) != want or want < s.PER_NIGHT_MIN:
        problems.append(f'LOONCOAD {h.get("LOONCOAD")} != {want} (>= {s.PER_NIGHT_MIN} required)')
    if h.get('LOOMSTR') != os.path.basename(mp)[:68] or h.get('LSTBIN') != d['k'] or 'BINMSTR' in h:
        problems.append(f'LOOMSTR {h.get("LOOMSTR")} / LSTBIN {h.get("LSTBIN")} / BINMSTR {"BINMSTR" in h}')
    if str(h.get('SUBTRACT', '')) != 'leave-one-out window Master' or h.get('FLUXBAKE') is not True:
        problems.append(f'SUBTRACT {h.get("SUBTRACT")!r} FLUXBAKE {h.get("FLUXBAKE")}')
    if h.get('JUPMASK') != hm.get('JUPMASK'):
        problems.append(f'JUPMASK {h.get("JUPMASK")} != master {hm.get("JUPMASK")}')
    try:
        from astropy.time import Time
        dur = round((Time(str(h['DATE-END'])) - Time(str(h['DATE-OBS']))).sec, 1)
        if abs(float(h.get('DURATION', -1)) - dur) > 0.051:
            problems.append(f'DURATION {h.get("DURATION")} != {dur}')
    except Exception as e:
        problems.append(f'DURATION not checkable ({e})')
    a = np.asarray(_data(stack), dtype=np.float64)
    M = np.asarray(_data(mp), dtype=np.float64)
    r = np.asarray(_data(path), dtype=np.float64)
    frac = None
    if not (a.shape == M.shape == r.shape):
        problems.append(f'shapes {a.shape} {M.shape} {r.shape}')
    else:
        if np.any(np.isnan(a) & np.isfinite(r)):
            problems.append('finite residual where the stack is NaN')
        W = float(sum(rows.values()))
        v = np.isfinite(a) & np.isfinite(M) & np.isfinite(r)
        if W - wn <= 0 or not v.any():
            problems.append('no pixel to test the identity on')
        else:
            pred = a[v] - (M[v] * W - wn * a[v]) / (W - wn)
            rv = r[v]
            rms = 1.4826 * float(np.median(np.abs(rv - np.median(rv)))) or 1.0
            frac = float(np.mean(np.abs(rv - pred) <= s.LOO_IDENTITY['tol_rms'] * rms))
            if frac < s.LOO_IDENTITY['frac_min']:
                problems.append(f'identity holds on {frac:.4f} of {int(v.sum())} pixels < {s.LOO_IDENTITY["frac_min"]}')
    return _verdict(problems, frac, f'LOONCOAD {want}; identity on {frac if frac is None else round(frac, 5)} of pixels')


# =====================================================================================================================
# LST-level production content tests (in place)
# =====================================================================================================================

@content_test('sublst_window_sample')
def production_test_windows(path, deepflow_jobs=None, **ctx):
    """Production content test (a) on <root>/<LST>: every window's header census and, on a seeded sample of
    masters_sublst.WINDOW_SAMPLE_TEST n (night, subband) cells, windows = frames x stated factors re-derived from the cubes
    (exopipe_ops/sublst_reserve_test_windows.py in place through core.legacy.run, interpreter masters_sublst.TEST_WINDOWS_PYTHON
    = py36 as sublst_catalog_publish.sh:87 runs it). Passes when the script ends 'WINDOWS-TEST <LST>: PASS'."""
    from exopipe.phase3.core import legacy
    T = _T_mod()
    s = _sec()
    lst = os.path.basename(os.path.abspath(path))
    root = os.path.dirname(os.path.abspath(path))
    tail = [lst, '--n', str(s.WINDOW_SAMPLE_TEST['n'])]
    if s.WINDOW_SAMPLE_TEST.get('seed'):
        tail += ['--seed', str(s.WINDOW_SAMPLE_TEST['seed'])]
    if os.path.realpath(root) != os.path.realpath(T.sublst_root()):
        tail += ['--root', root]
    tail += ['--cubes', T.cube_root(), '--deepflow-cubes', T.deepflow_root()]
    if deepflow_jobs:
        tail += ['--deepflow-jobs', deepflow_jobs]
    res = legacy.run(T.script('test_windows'), tail, python=T.python_key('test_windows'), refusal_codes=(),
                     refusal_text=(), tail_lines=400)      # census + bad-window lines + 2n SAMPLE lines + warnings
    lines = res.tail.splitlines()
    fin = [l for l in lines if l.startswith('WINDOWS-TEST ')]
    last = fin[-1] if fin else (lines[-1] if lines else '')
    ok = res.rc == 0 and bool(fin) and fin[-1].rstrip().endswith('PASS')
    detail = ' | '.join(l for l in lines if l.startswith(('CENSUS', 'SAMPLE')))[:2000]   # census + every SAMPLE line
    return ok, last[:200], (detail + ' | ' if detail else '') + f'exit {res.rc}: {last[:200]}'
