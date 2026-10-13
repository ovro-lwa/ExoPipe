"""Content tests of the `exopipe3 flow` product classes (spec flow.md section 6). Discovered by core/validate (lib/<step>/
tests*.py); also imported by steps/flow.py. Every test checks the file's CONTENT (pixels, rows, header values against the
data and against the inputs), never only its presence.

Classes (all owned by exopipe.phase3.steps.flow):
  flow_reference         prep/ref_master_prev*.fits: the co-add's snapshot of the reference masters stage-in wrote
  flow_stagein_manifest  <out>/_records/flow/manifests/<cell>_stagein.json (the stored-flow copy)
  flow_prep_scaled       prep/<date>_scaled.fits (flux-scaled flow input) with prep/<date>_scale.json
  coadd_flow             flow/<date>_scaled_flow.npy (alias 'flow': the OWNERS entry in core/validate, to be renamed)
  flow_dewarp_image      flow/<date>_scaled_dewarp.fits
  snapshot_flowed_frame  <stage>/<cell>/<sb>/<frame>_flowed.fits; a staged SUBBAND DIRECTORY runs the set test over it
  snapshot_flow_field    flow_<sb>_<ts>.npy; a DIRECTORY (work flow dir or snapshot_flow/<cell>) runs the set test
  snapshot_flow_stats    flow_<sb>.csv
  sublst_x2_residual     <LST>_x2/<LST>_<night>/<sb>/<S>/..._resid_loo.fits; the <LST>_x2 DIRECTORY runs R1 + R2

Set tests (one subband's complete set) run on a directory path; on a single file they report 'n/a' (the step runs them
once per unit through the same functions, set_complete_* below). Tests that need an input the product does not name (the
source master of a reference, the input image of a flow) take it from the caller's context, else derive it from the
production layout; when it cannot be found the test says so in its detail (value 'no-source' / 'form-only').
"""
import csv
import glob
import json
import math
import os
import re

from exopipe.phase3.core.validate import content_test
from exopipe.phase3.lib.flow import steps_common as C

_RNG_SEED = 0
FLOWTREE_DIR = 'snapshot_flow'      # <out>/snapshot_flow/<cell>/ (production: paths.SNAPSHOT_FLOW_ROOT)
_NSAMPLE = 1000


def _cfg():
    return C.cfg()


def _sample_idx(mask, n=_NSAMPLE):
    import numpy as np
    idx = np.flatnonzero(mask.ravel())
    if idx.size > n:
        idx = np.random.default_rng(_RNG_SEED).choice(idx, n, replace=False)
    return idx


# =====================================================================================================================
# deep
# =====================================================================================================================

def _deep_cell_dir(path):
    """<W> for <W>/<sub>/<file> (coadd_final, coadd_final_ref, prep, flow)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(path)))


def _deep_cell(path):
    from exopipe.phase3.core import cells
    try:
        return cells.parse_cell(os.path.basename(_deep_cell_dir(path)))
    except ValueError:
        return None, None


REF_SNAPSHOTS = {'ref_master_prev.fits': 'I_deep_Taper_R075_{scope}.fits',
                 'ref_master_prev_V.fits': 'V_deep_Taper_R0_{scope}.fits'}


@content_test('flow_reference')
def snapshot_of_reference(path, copy_of=None, **ctx):
    """prep/ref_master_prev.fits (and _V) is the byte copy the in-place co-add takes of the cell's previous-generation master
    before the build overwrites coadd_final/ (coadd_full_pipeline.py:1641-1648): byte-equal to ``copy_of`` when given, else
    to masters stage-in's kept copy coadd_final_ref/<name> (rebuild_cell.sh:118-120; masters owns that reference,
    _CROSSCHECK decision 6), else to coadd_final/<name> while it is still the reference."""
    W = _deep_cell_dir(path)
    lst, sc = _deep_cell(path)
    name = REF_SNAPSHOTS.get(os.path.basename(path))
    if copy_of is None and name and sc:
        for sub in ('coadd_final_ref', 'coadd_final'):
            q = os.path.join(W, sub, name.format(scope=sc))
            if os.path.exists(q):
                copy_of = q
                break
    if not copy_of:
        return False, None, 'no reference master in coadd_final_ref/ or coadd_final/ to compare with'
    ok = C.same_bytes(path, copy_of)
    if not ok and os.path.basename(os.path.dirname(copy_of)) == 'coadd_final':
        a, b = C.header_cards(path), C.header_cards(copy_of)
        if C.card(a, 'CONSTDIG') != C.card(b, 'CONSTDIG') or C.card(b, 'COADDOP') != C.card(a, 'COADDOP'):
            return True, 'superseded', (f'{os.path.relpath(copy_of, W)} has been rebuilt since (CONSTDIG '
                                        f'{C.card(b, "CONSTDIG")}); snapshot of CONSTDIG {C.card(a, "CONSTDIG")}')
    return ok, 'byte-equal' if ok else 'differs', f'copy of {os.path.relpath(copy_of, W)}'


@content_test('flow_reference')
def native_unwarped(path, **ctx):
    """The reference the flows align to is the native, un-warped, unscaled master: no MDWARP and no SH*/FLUX* card (masters
    stage-in strips them, rebuild_cell.sh:74/91/111), CONSTDIG and COADDOP present (they make .reference_resolved), some
    finite pixel."""
    import numpy as np
    from astropy.io import fits
    with fits.open(path, memmap=False) as h:
        hdr = h[0].header
        data = np.squeeze(h[0].data)
    bad = sorted({k for k in hdr.keys() if k and (k.startswith(('SH', 'FLUX')) or k == 'MDWARP')})
    ff = float(np.isfinite(data).mean()) if data is not None and data.size else 0.0
    problems = []
    if bad:
        problems.append(f'cards of a scaled or warped master present: {bad[:6]}')
    for k in ('CONSTDIG', 'COADDOP'):
        if not hdr.get(k):
            problems.append(f'no {k}')
    if ff <= 0:
        problems.append('no finite pixel')
    return (not problems, round(ff, 4), '; '.join(problems) or
            f"COADDOP {hdr.get('COADDOP')} CONSTDIG {hdr.get('CONSTDIG')} NCOADD {hdr.get('NCOADD')} finite {ff:.4f}")


@content_test('flow_stagein_manifest')
def copy_identical(path, **ctx):
    """The stage-in manifest (<out>/_records/flow/manifests/<cell>_stagein.json) lists every file of the copied flow/
    directory: each still present in the destination with the recorded size and mtime (to the second - the freshness gate
    compares mtimes, so a copy that changed them would change admission) and, for small files (.reference markers), the
    recorded sha1; every file of the source it names is present in the destination (rsync -a, rebuild_cell.sh:44)."""
    m = json.load(open(path))
    dst, src = m.get('dest'), m.get('source')
    bad, n, rewritten = [], 0, 0
    for r in m.get('files', []):
        p = os.path.join(dst, r['rel'])
        n += 1
        if not os.path.exists(p):
            bad.append(f'{r["rel"]}: missing')
            continue
        st = os.stat(p)
        if st.st_size != r['size'] or int(st.st_mtime) != r['mtime_s']:
            rewritten += 1    # prep / solve rewrote it after the copy (recomputed flow, new reference marker): not a copy error
            continue
        if r.get('sha1') and C.sha1_file(p) != r['sha1']:
            bad.append(f'{r["rel"]}: sha1 differs')
    miss_src = []
    if src and os.path.isdir(src):
        listed = {r['rel'] for r in m.get('files', [])}
        for root, _d, fs in os.walk(src):
            for f in fs:
                rel = os.path.relpath(os.path.join(root, f), src)
                if rel not in listed:
                    miss_src.append(rel)
    if miss_src:
        bad.append(f'{len(miss_src)} source file(s) not in the copy, e.g. {miss_src[:3]}')
    return not bad and n > 0, n, '; '.join(bad[:4]) or (f'{n} files of {src} present in {dst}'
                                                         + (f' ({rewritten} rewritten since by prep/solve)' if rewritten else ''))


def _prep_paths(path):
    W = _deep_cell_dir(path)
    date = os.path.basename(path).split('_scaled')[0]
    return W, date


def _alignment_image(lst, scope, date):
    """The night's alignment image as coadd.discover_dates finds it (latest Science_*, dewarped preferred)."""
    from exopipe.phase3.lib.masters import coadd as CO
    sci = CO.find_latest_science_dir(lst, date)
    if not sci:
        return None
    if CO.is_subband(scope.replace('MHz', '')):
        return CO.find_subband_alignment_image(sci, int(scope.replace('MHz', '')))
    return CO.find_wideband_alignment_image(sci, scope, lst, date)


@content_test('flow_prep_scaled')
def scaled_equals_input_over_scale(path, input_image=None, **ctx):
    """<date>_scaled.fits = input (or its pre-cutover regrid; NaN as 0) / safe(scale map rebuilt from _scale.json coeffs on
    the normalised grid; |s| <= SAFE_SCALE_MIN -> 1) at 1,000 sampled pixels, rtol 1e-6; _scale.json has its six keys and
    weight = mean_scale^2 / tgt_rms^2 (coadd_full_pipeline.py:1772-1798)."""
    import numpy as np
    from astropy.io import fits
    W, date = _prep_paths(path)
    js = os.path.join(W, 'prep', f'{date}_scale.json')
    if not os.path.exists(js):
        return False, None, f'{js} missing'
    j = json.load(open(js))
    keys = ('coeffs', 'n_sources', 'mean_scale', 'tgt_rms', 'tgt_rms_scaled', 'weight')
    miss = [k for k in keys if k not in j]
    if miss:
        return False, None, f'_scale.json lacks {miss}'
    w_exp = float(j['mean_scale'] ** 2 / j['tgt_rms'] ** 2) if j['tgt_rms'] > 0 else 0.0
    if not math.isclose(w_exp, j['weight'], rel_tol=1e-9, abs_tol=1e-12):
        return False, j['weight'], f'weight {j["weight"]} != mean_scale^2/tgt_rms^2 = {w_exp}'
    if input_image is None:
        rg = os.path.join(W, 'prep', f'{date}_regrid.fits')
        if os.path.exists(rg):
            input_image = rg
        else:
            lst, sc = _deep_cell(path)
            input_image = _alignment_image(lst, sc, date) if lst else None
    if not input_image or not os.path.exists(input_image):
        return False, None, 'input image not found'
    with fits.open(input_image, memmap=False) as h:
        src = h[0].data.astype(np.float64)
    while src.ndim > 2:
        src = src[0]
    src = np.nan_to_num(src, nan=0.0)
    with fits.open(path, memmap=False) as h:
        got = np.asarray(np.squeeze(h[0].data))
    if got.shape != src.shape:
        return False, 'shape', f'shape {got.shape} != input {src.shape}'
    hgt, wid = src.shape
    c0, c1, c2 = (float(x) for x in j['coeffs'])
    idx = _sample_idx(np.ones(src.shape, bool))
    yy, xx = np.divmod(idx, wid)
    smap = c0 + c1 * (2.0 * xx / wid - 1.0) + c2 * (2.0 * yy / hgt - 1.0)
    safe = np.where(np.abs(smap) > _cfg().masters_build.SAFE_SCALE_FLOOR, smap, 1.0)
    want = (src.ravel()[idx] / safe).astype(np.float32).astype(np.float64)
    have = got.ravel()[idx].astype(np.float64)
    rel = float(np.max(np.abs(have - want) / np.maximum(np.abs(want), 1e-12)))
    return rel <= 1e-6, round(rel, 9), f'{idx.size} sampled pixels vs {os.path.basename(input_image)}, max rel {rel:.3g}'


def _flow_grid(path):
    """(ny, nx) of the night's grid: the scaled flow input if still present, else the dewarped by-product beside the flow."""
    W, date = _prep_paths(path)
    for p in (os.path.join(W, 'prep', f'{date}_scaled.fits'), os.path.join(W, 'flow', f'{date}_scaled_dewarp.fits')):
        if os.path.exists(p):
            c = C.header_cards(p)
            return (int(C.card(c, 'NAXIS2')), int(C.card(c, 'NAXIS1'))), p
    return None, None


def flow_stats(arr, frac=None):
    """(finite fraction, median |flow|, p99 |flow|) inside the statistics disc (report only)."""
    import numpy as np
    frac = _cfg().flow.STATS_FOV_FRACTION if frac is None else frac
    a = np.asarray(arr, dtype=np.float32)
    ff = float(np.isfinite(a).mean()) if a.size else 0.0
    h, w = a.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    disc = (xx - w // 2) ** 2 + (yy - h // 2) ** 2 < (min(h, w) * frac) ** 2
    mag = np.hypot(a[..., 0], a[..., 1])[disc]
    mag = mag[np.isfinite(mag)]
    if not mag.size:
        return ff, float('nan'), float('nan')
    return ff, float(np.median(mag)), float(np.percentile(mag, 99))


@content_test('coadd_flow')
def shape_and_finite(path, shape=None, **ctx):
    """float32 array (NAXIS2, NAXIS1, 2) of the night's grid; finite fraction >= masters_build.FLOW_MAX_NAN_FRACTION
    (_require_flow, coadd_full_pipeline.py:1825-1843; ledger 1733). Median and 99th-percentile |flow| in the 0.4 disc are
    reported (no threshold without a ruling, spec G3)."""
    import numpy as np
    a = np.load(path, mmap_mode='r')
    exp, src = (tuple(shape), 'context') if shape else _flow_grid(path)
    problems = []
    if a.dtype != np.float32:
        problems.append(f'dtype {a.dtype} (float32 expected)')
    if a.ndim != 3 or a.shape[2] != 2:
        problems.append(f'shape {a.shape} is not (ny, nx, 2)')
    elif exp is None:
        problems.append('night grid unknown (no scaled image and no dewarp image beside the flow)')
    elif tuple(a.shape[:2]) != tuple(exp):
        problems.append(f'shape {a.shape[:2]} != grid {exp} of {src}')
    ff, med, p99 = flow_stats(a) if a.ndim == 3 and a.shape[2] == 2 else (0.0, float('nan'), float('nan'))
    lim = _cfg().masters_build.FLOW_MAX_NAN_FRACTION
    if ff < lim:
        problems.append(f'finite fraction {ff:.3f} < {lim} (would warp the night to zeros)')
    return (not problems, round(ff, 4),
            '; '.join(problems) or f'finite {ff:.4f}; |flow| in 0.4 disc median {med:.3f} px, p99 {p99:.3f} px')


@content_test('coadd_flow')
def fresh(path, input_image=None, input_fp=None, **ctx):
    """Not older than its input image (coadd._flow_freshness, :4155-4184) and, when the run recorded it, the input's
    fingerprint unchanged (record identity, DESIGN 11.5)."""
    if input_image is None:
        lst, sc = _deep_cell(path)
        date = os.path.basename(path).split('_scaled')[0]
        input_image = _alignment_image(lst, sc, date) if lst else None
    if not input_image or not os.path.exists(input_image):
        return False, None, 'input image of the night not found'
    ok = os.path.getmtime(input_image) <= os.path.getmtime(path)
    det = f'input {os.path.basename(input_image)}'
    if not ok:
        return False, 'older', f'flow older than its input image ({det})'
    if input_fp:
        from exopipe.phase3.core import record
        if not record.same_fingerprint(input_fp, record.fingerprint(input_image)):
            return False, 'input-changed', f'input image changed since the flow was computed ({det})'
        det += ', fingerprint unchanged'
    return True, 'fresh', det


@content_test('coadd_flow')
def reference_signature(path, reference_signature=None, **ctx):
    """flow/.reference_resolved names the reference the flow was computed against: equal to the run's recorded signature,
    else consistent with the reference copy kept in coadd_final_ref/ or prep/ref_master_prev.fits (MASTER:<COADDOP>:
    <CONSTDIG>, coadd_full_pipeline.py:1628-1651)."""
    W = _deep_cell_dir(path)
    mk = os.path.join(W, 'flow', '.reference_resolved')
    if not os.path.exists(mk):
        return False, None, f'{mk} missing'
    sig = open(mk).read().strip()
    if reference_signature is not None:
        return sig == reference_signature, sig, f'recorded {reference_signature}'
    if not (re.match(r'^MASTER:[^:]+:[0-9a-f]{32}$', sig) or re.match(r'^MASTER:[^:]+:nodigest$', sig)
            or re.match(r'^\d{4}-\d\d-\d\d$', sig)):
        return False, sig, 'malformed signature'
    lst, sc = _deep_cell(path)
    for p in (os.path.join(W, 'coadd_final_ref', f'I_deep_Taper_R075_{sc}.fits'),
              os.path.join(W, 'prep', 'ref_master_prev.fits')):
        if sc and os.path.exists(p):
            c = C.header_cards(p)
            want = f"MASTER:{C.card(c, 'COADDOP')}:{C.card(c, 'CONSTDIG', 'nodigest')}"
            if sig.startswith('MASTER:'):
                return sig == want, sig, f'reference copy {os.path.relpath(p, W)} -> {want}'
    return True, 'form-only', f'{sig} (no reference copy in the cell to compare with)'


@content_test('flow_dewarp_image')
def structure(path, **ctx):
    """2-D; shape = the flow's first two axes; WCS (CRVAL, CDELT, CRPIX) = the scaled input's within 1e-9 when the input is
    still present; finite fraction > 0."""
    import numpy as np
    W, date = _prep_paths(path)
    fl = os.path.join(W, 'flow', f'{date}_scaled_flow.npy')
    data = C.image_data(path)
    problems, det = [], []
    if data.ndim != 2:
        problems.append(f'{data.ndim}-D')
    if os.path.exists(fl):
        fs = np.load(fl, mmap_mode='r').shape
        if tuple(fs[:2]) != tuple(data.shape):
            problems.append(f'shape {data.shape} != flow {fs[:2]}')
    else:
        problems.append(f'flow {os.path.basename(fl)} missing')
    sc = os.path.join(W, 'prep', f'{date}_scaled.fits')
    if os.path.exists(sc):
        a, b = C.header_cards(path), C.header_cards(sc)
        for k in ('CRVAL1', 'CRVAL2', 'CDELT1', 'CDELT2', 'CRPIX1', 'CRPIX2'):
            va, vb = C.card(a, k), C.card(b, k)
            if va is None or vb is None or abs(float(va) - float(vb)) > 1e-9:
                problems.append(f'{k} {va} != scaled input {vb}')
        det.append('WCS = scaled input')
    else:
        det.append('scaled input gone (masters provisional removes prep/): WCS not compared')
    ff = float(np.isfinite(data).mean()) if data.size else 0.0
    if ff <= 0:
        problems.append('no finite pixel')
    return not problems, round(ff, 4), '; '.join(problems) or '; '.join(det) + f'; finite {ff:.4f}'


# 'flow' is the class name core/validate OWNERS lists for exopipe.phase3.steps.flow; spec 4.4 renames it coadd_flow.
for _fn in (shape_and_finite, fresh, reference_signature):
    content_test('flow', _fn.__name__)(_fn)


# =====================================================================================================================
# snapshot
# =====================================================================================================================

_HDR_CACHE = {}


def _hdr(path):
    """Header cards of a Master (cached per path and mtime: every frame and field of a subband reads the same one)."""
    try:
        key = (os.path.abspath(path), os.stat(path).st_mtime_ns)
    except OSError:
        return None
    if key not in _HDR_CACHE:
        _HDR_CACHE[key] = C.header_cards(path)
    return _HDR_CACHE[key]


def _pmap(fn, items, n=16):
    """fn over items with threads (small Lustre files are read in parallel, as test F does)."""
    from concurrent.futures import ThreadPoolExecutor
    items = list(items)
    if len(items) < 8:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(max_workers=n) as ex:
        return list(ex.map(fn, items))


def _master_for(path, master=None):
    if master:
        return master
    cell = C.snapshot_cell_of_path(path)
    info = C.frame_info(path) or (None,)
    sb = info[0]
    if sb is None:
        m = C.FIELD_PAT.match(os.path.basename(path)) or re.match(r'^flow_(\d+MHz)', os.path.basename(path))
        sb = m.group(1) if m else None
    if not cell or not sb:
        return None
    return C.snapshot_master(cell.split('_')[0], sb)


def check_flowed_header(path, master=None):
    """problems of one flowed frame's header/grid (list of text)."""
    master = _master_for(path, master)
    c = C.header_cards(path)
    st = (C.frame_info(path) or (None, None, None))[2]
    out = []
    if C.card(c, 'NAXIS') != 2:
        out.append(f'NAXIS {C.card(c, "NAXIS")} (2-D expected)')
    m = _hdr(master) if master else None
    if m:
        for k in ('NAXIS1', 'NAXIS2'):
            if C.card(c, k) != C.card(m, k):
                out.append(f'{k} {C.card(c, k)} != Master {C.card(m, k)}')
        for k in ('CRVAL1', 'CRVAL2', 'CDELT1', 'CDELT2'):
            va, vb = C.card(c, k), C.card(m, k)
            if va is None or vb is None or abs(float(va) - float(vb)) > 1e-9:
                out.append(f'{k} {va} != Master {vb}')
    else:
        out.append(f'subband Master {master} not found')
    if 'FLOWED' not in c:
        out.append('no FLOWED card')
    elif C.card(c, 'FLOWED') is True:
        ref = os.path.basename(master)[:60] if master else None
        if C.card(c, 'FLOWREF') != ref:
            out.append(f'FLOWREF {C.card(c, "FLOWREF")!r} != {ref!r}')
        blk = C.card(c, 'FLOWBLK')
        if st == 'I' and blk != _cfg().flow_snapshot.BLOCK_S:
            out.append(f'FLOWBLK {blk!r} != {_cfg().flow_snapshot.BLOCK_S}')
        if st == 'V' and blk not in (None, _cfg().flow_snapshot.BLOCK_S):
            out.append(f'FLOWBLK {blk!r} on a V frame')
    return out


def check_blank_legit(path):
    """problems: FLOWED=False only for a Stokes I frame blank by the module's own rule (recomputed from the raw staged
    frame) or a V frame whose FLOWED comment says no valid I flow existed in the subband-hour."""
    c = C.header_cards(path)
    if C.card(c, 'FLOWED') is not False:
        return []
    st = (C.frame_info(path) or (None, None, None))[2]
    if st == 'V':
        com = c['FLOWED'][1]
        return [] if 'no valid I flow' in com else [f'V frame not flowed without the no-valid-I-flow reason ({com!r})']
    raw = C.raw_of_flowed(path)
    if not os.path.exists(raw):
        return [f'raw staged frame {os.path.basename(raw)} gone: blankness not recomputable']
    blank, ff = C.is_blank_frame(raw)
    return [] if blank else [f'Stokes I frame not flowed but not blank (finite {ff:.3f})']


def set_complete_flowed(stage_sb_dir, frames=None, zerovis=None):
    """G-S6 / test set_complete for one staged subband directory, BY NAME (snapshot defect D12, cf72d51): expected = staged
    frames (I and V; ``frames`` = the stage record's list when given, else the directory listing) minus the frames whose own
    header records WSCNVIS = 0; flowed = <frame>_flowed.fits present. Returns (ok, n_expected, missing names, info)."""
    if frames is None:
        frames = [os.path.join(stage_sb_dir, f) for f in os.listdir(stage_sb_dir) if C.is_staged_frame(f)]
    if zerovis is None:
        zerovis = {p for p in frames if C.zero_visibility(p)[0]}
    expected = sorted(set(frames) - set(zerovis))
    missing = [os.path.basename(p) for p in expected if not os.path.exists(C.flowed_path(p))]
    info = {'staged': len(frames), 'zero_vis': len(zerovis), 'expected': len(expected),
            'flowed': len(expected) - len(missing),
            'zero_vis_written_through': sum(1 for p in zerovis if os.path.exists(C.flowed_path(p)))}
    return not missing, len(expected), missing, info


def _dir_files(path, pattern):
    return sorted(glob.glob(os.path.join(path, pattern)))


@content_test('snapshot_flowed_frame')
def set_complete(path, frames=None, **ctx):
    """On a staged subband directory: every expected frame (staged minus WSCNVIS = 0) has its _flowed.fits - a SET
    comparison per subband, never a count (D12; cf72d51). On a single file: n/a (the step runs it once per subband)."""
    if not os.path.isdir(path):
        return True, 'n/a', 'set test runs on the staged subband directory'
    ok, n, missing, info = set_complete_flowed(path, frames=frames)
    return ok, info['flowed'], (f'{len(missing)} of {n} expected frames not flowed (first: {missing[:3]}); {info}'
                                if missing else f'{n} of {n} expected frames flowed; {info}')


@content_test('snapshot_flowed_frame')
def header_and_grid(path, master=None, **ctx):
    """2-D, NAXIS1/2 and CRVAL/CDELT of the subband Master (1e-9); FLOWED present; FLOWED=True -> FLOWREF =
    basename(Master)[:60] and (Stokes I) FLOWBLK = 0 (phase3_snapshot_flow.py:165-220)."""
    files = _dir_files(path, '*_flowed.fits') if os.path.isdir(path) else [path]
    bad = {os.path.basename(p): pr for p, pr in zip(files, _pmap(lambda q: check_flowed_header(q, master), files)) if pr}
    return not bad and bool(files), len(files), (f'{len(bad)} of {len(files)} bad, e.g. {next(iter(bad.items()))}'
                                                  if bad else f'{len(files)} flowed frame header(s) on the Master grid'
                                                  if files else 'no flowed frame')


@content_test('snapshot_flowed_frame')
def blank_legit(path, **ctx):
    """FLOWED=False only where the module's own blank rule (or the no-valid-I-flow V rule) allows it."""
    files = _dir_files(path, '*_flowed.fits') if os.path.isdir(path) else [path]
    unflowed = [p for p, c in zip(files, _pmap(C.header_cards, files)) if C.card(c, 'FLOWED') is False]
    nfalse = len(unflowed)
    bad = {os.path.basename(p): pr for p, pr in zip(unflowed, _pmap(check_blank_legit, unflowed)) if pr}
    return not bad, nfalse, (f'{len(bad)} of {nfalse} unflowed frames not legitimate, e.g. {next(iter(bad.items()))}'
                             if bad else f'{nfalse} unflowed frame(s), all blank Stokes I or V without a valid I flow')


def _fields_of(dirpath, sb):
    out = {}
    for p in glob.glob(os.path.join(dirpath, f'flow_{sb}_*.npy')):
        m = C.FIELD_PAT.match(os.path.basename(p))
        if m:
            out[m['ts']] = p
    return out


def field_shape_problems(path, master=None):
    import numpy as np
    master = _master_for(path, master)
    a = np.load(path, mmap_mode='r')
    out = []
    if a.dtype != np.float16:
        out.append(f'dtype {a.dtype} (float16 expected)')
    lowres = _cfg().flow_snapshot.LOWRES
    m = _hdr(master) if master else None
    if m:
        ny, nx = int(C.card(m, 'NAXIS2')), int(C.card(m, 'NAXIS1'))
        want = (-(-ny // lowres), -(-nx // lowres), 2)
        if tuple(a.shape) != want:
            out.append(f'shape {a.shape} != {want} (Master {ny}x{nx} every {lowres}th pixel)')
    else:
        out.append(f'subband Master {master} not found')
    if not np.isfinite(np.asarray(a, dtype=np.float32)).any():
        out.append('no finite value')
    return out


def flowed_i_ts(stage_sb_dir):
    """{ts} of the FLOWED=True, FLOWBLK=0 Stokes I frames of a staged subband directory (the frames that have a field)."""
    out = set()
    for p in glob.glob(os.path.join(stage_sb_dir, '*_flowed.fits')):
        fi = C.frame_info(p)
        if not fi or fi[2] != 'I':
            continue
        c = C.header_cards(p)
        if C.card(c, 'FLOWED') is True and C.card(c, 'FLOWBLK') == _cfg().flow_snapshot.BLOCK_S:
            out.add(fi[1])
    return out


def _stats_ok_ts(csv_path):
    rows = list(csv.DictReader(open(csv_path)))
    return {r['ts'] for r in rows if r.get('state') == 'ok'}


def _ship_candidates(path, cell):
    """Where the QA files (flow_<sb>.csv, series, mp4) of a fields directory live: beside it, <D>/snapshot_fleet_out/<cell>/
    flow (parity layout) or the production paths.SNAPSHOT_FLEET_OUT."""
    d = os.path.abspath(path if os.path.isdir(path) else os.path.dirname(path))
    out = [d]
    parts = d.split(os.sep)
    if cell and len(parts) >= 2 and parts[-1] == cell and parts[-2] == FLOWTREE_DIR:
        out.append(os.path.join(os.sep.join(parts[:-2]), 'snapshot_fleet_out', cell, 'flow'))
    if cell:
        out.append(os.path.join(_cfg().paths.SNAPSHOT_FLEET_OUT, cell, 'flow'))
    return out


def _field_candidates(path, cell):
    """Where the fields of a QA file live: beside it, <D>/snapshot_flow/<cell> (parity layout), the production flow tree."""
    d = os.path.dirname(os.path.abspath(path))
    out = [d]
    parts = d.split(os.sep)
    if cell and len(parts) >= 3 and parts[-1] == 'flow' and parts[-2] == cell and parts[-3] == 'snapshot_fleet_out':
        out.append(os.path.join(os.sep.join(parts[:-3]), FLOWTREE_DIR, cell))
    if cell:
        out.append(os.path.join(_cfg().paths.SNAPSHOT_FLOW_ROOT, cell))
    return out


def set_complete_fields(flow_dir, sb, stage_sb_dir=None, stats_csv=None):
    """One field per FLOWED=True, FLOWBLK=0 Stokes I frame (by ts): the reference set comes from the staged subband
    directory when given, else from the 'ok' rows of the subband's flow_<sb>.csv. Returns (ok, n_fields, detail)."""
    have = set(_fields_of(flow_dir, sb))
    if stage_sb_dir:
        want, src = flowed_i_ts(stage_sb_dir), 'flowed Stokes I frames'
    elif stats_csv and os.path.exists(stats_csv):
        want, src = _stats_ok_ts(stats_csv), os.path.basename(stats_csv) + ' ok rows'
    else:
        return False, len(have), 'no reference set (no staged subband directory and no flow_<sb>.csv)'
    extra, miss = sorted(have - want), sorted(want - have)
    return (not extra and not miss and bool(want)), len(have), (
        f'{len(have)} fields vs {len(want)} {src}; missing {miss[:3]} ({len(miss)}), extra {extra[:3]} ({len(extra)})')


@content_test('snapshot_flow_field')
def field_shape(path, master=None, **ctx):
    """float16 (ceil(ny/8), ceil(nx/8), 2) of the subband Master grid, some finite value (phase3_snapshot_flow.py:221-225).
    On a directory: every field in it."""
    files = _dir_files(path, 'flow_*MHz_*.npy') if os.path.isdir(path) else [path]
    bad = {os.path.basename(p): pr for p, pr in zip(files, _pmap(lambda q: field_shape_problems(q, master), files)) if pr}
    return (not bad and bool(files), len(files),
            f'{len(bad)} of {len(files)} bad, e.g. {next(iter(bad.items()))}' if bad else f'{len(files)} field(s) ok')


@content_test('snapshot_flow_field')
def field_set(path, stage_sb_dir=None, subband=None, **ctx):
    """On a directory: per subband, the set of fields (by ts) equals the set of flowed Stokes I frames (staged directory
    given) or the 'ok' rows of the subband's flow_<sb>.csv. On a single file: n/a."""
    if not os.path.isdir(path):
        return True, 'n/a', 'set test runs on the fields directory'
    cell = C.snapshot_cell_of_path(path)
    sbs = [subband] if subband else sorted({C.FIELD_PAT.match(os.path.basename(p))['sb']
                                           for p in _dir_files(path, 'flow_*MHz_*.npy')})
    if not sbs:
        return False, 0, 'no field in the directory'
    bad, n = [], 0
    for sb in sbs:
        stats = next((os.path.join(d, f'flow_{sb}.csv') for d in _ship_candidates(path, cell)
                      if os.path.exists(os.path.join(d, f'flow_{sb}.csv'))), None)
        ok, k, det = set_complete_fields(path, sb, stage_sb_dir if subband else None, stats)
        n += k
        if not ok:
            bad.append(f'{sb}: {det}')
    return not bad, n, '; '.join(bad[:4]) or f'{len(sbs)} subband(s): field sets complete'


@content_test('snapshot_flow_field')
def against_shipped(path, shipped_dir=None, **ctx):
    """With ``shipped_dir`` (flow --against-shipped; snapshot_frametime_content_test.py test F :733-786,
    snapshot_masters_freeze_flowcheck.sh:21-27): each field equals the field already shipped for that ts within 0.001 px at
    every finite pixel, with identical finite masks. Without it: n/a."""
    import numpy as np
    if not shipped_dir:
        return True, 'n/a', 'no shipped field set given (--against-shipped)'
    files = _dir_files(path, 'flow_*MHz_*.npy') if os.path.isdir(path) else [path]
    worst, bad, n = 0.0, [], 0
    for p in files:
        q = os.path.join(shipped_dir, os.path.basename(p))
        if not os.path.exists(q):
            bad.append(f'{os.path.basename(p)}: no shipped counterpart')
            continue
        a, b = np.load(p).astype(np.float32), np.load(q).astype(np.float32)
        if a.shape != b.shape:
            bad.append(f'{os.path.basename(p)}: shapes {a.shape} vs {b.shape}')
            continue
        m = np.isfinite(a) & np.isfinite(b)
        d = float(np.max(np.abs(a[m] - b[m]))) if m.any() else 0.0
        worst, n = max(worst, d), n + 1
        if d > 0.001 or (np.isfinite(a) != np.isfinite(b)).any():
            bad.append(f'{os.path.basename(p)}: max |difference| {d:.4f} px')
    return not bad and n > 0, round(worst, 5), '; '.join(bad[:4]) or f'{n} field(s) within 0.001 px of the shipped ones'


STATS_COLUMNS = ['ts', 'state', 'median_flow_px', 'p90_flow_px', 'max_flow_px', 'mean_dx_px', 'mean_dy_px', 'seconds']


@content_test('snapshot_flow_stats')
def rows(path, i_frames=None, fields_dir=None, **ctx):
    """Exactly the 8 columns; one row per Stokes I frame of the unit (``i_frames`` = their ts, when the run gives them; no
    duplicate ts otherwise); no FAILED row; median <= p90 <= max, non-negative; 'ok' rows = the fields (by ts)."""
    sb = (re.match(r'^flow_(\d+MHz)\.csv$', os.path.basename(path)) or [None, None])[1]
    with open(path, newline='') as f:
        rd = csv.reader(f)
        head = next(rd, [])
        body = [r for r in rd if r]
    problems = []
    if head != STATS_COLUMNS:
        problems.append(f'columns {head}')
    ts = [r[0] for r in body]
    dup = sorted({t for t in ts if ts.count(t) > 1})
    if dup:
        problems.append(f'{len(dup)} duplicate ts (appended runs?), e.g. {dup[:2]}')
    failed = [r[0] for r in body if len(r) > 1 and r[1].startswith('FAILED')]
    if failed:
        problems.append(f'{len(failed)} FAILED rows, e.g. {failed[:2]}')
    ok_ts = set()
    for r in body:
        if len(r) != 8:
            problems.append(f'row of {len(r)} fields: {r[:2]}')
            continue
        if r[1] != 'ok':
            continue
        ok_ts.add(r[0])
        try:
            md, p9, mx = float(r[2]), float(r[3]), float(r[4])
        except ValueError:
            problems.append(f'{r[0]}: non-numeric statistics')
            continue
        if not (0 <= md <= p9 <= mx):
            problems.append(f'{r[0]}: median {md} / p90 {p9} / max {mx} not ordered')
    if i_frames is not None and set(ts) != set(i_frames):
        problems.append(f'rows {len(set(ts))} != the unit\'s {len(set(i_frames))} Stokes I frames')
    cell = C.snapshot_cell_of_path(path)
    cands = [fields_dir] if fields_dir else _field_candidates(path, cell)
    fd = next((d for d in cands if d and sb and _fields_of(d, sb)), None)
    if fd is None:
        problems.append('no fields found to compare the ok rows with')
    else:
        have = set(_fields_of(fd, sb))
        if have != ok_ts:
            problems.append(f'ok rows {len(ok_ts)} != fields {len(have)} in {fd}')
    return not problems, len(body), '; '.join(problems[:5]) or f'{len(body)} rows, {len(ok_ts)} ok = fields'


# =====================================================================================================================
# sub-LST
# =====================================================================================================================

_RES_PAT = re.compile(r'^sublst_(?P<lst>\d\dh)_(?P<night>\d{4}-\d\d-\d\d)_(?P<band>\w+?)_(?P<S>[IV])_w(?P<k>\d)'
                      r'(?P<res>_resid_loo)?\.fits$')
_MODEL_CACHE = {}


def _weights(lst, sb):
    """Frozen deep co-add night weights of (LST, subband) with in_master 0 and the membership vetoes at weight 0
    (phase3_sublst_refine.py:83-94; the veto read in place from clv15 phase3_tier_rules, md5-pinned there)."""
    wp = _cfg().flow_refine.WEIGHTS_CSV.format(lst=lst, sb=sb)
    if not os.path.exists(wp):
        return None, wp
    import phase3_tier_rules as TR      # clv15, in place (core/env.legacy puts the root on sys.path)
    wts = {}
    for r in csv.DictReader(open(wp)):
        try:
            wts[r['date']] = 0.0 if (str(r.get('in_master', '1')).strip() == '0' or TR.membership_vetoed(lst, sb, r['date'])) \
                else float(r['weight'])
        except (KeyError, ValueError, TypeError):
            pass
    return wts, wp


def approved_rows(lst, sb):
    """{(night, window): approval text} of APPROVED_REFUSALS_CSV for (LST, subband) (sublst_refine_approved.py:15)."""
    out = {}
    p = _cfg().flow_refine.APPROVED_REFUSALS_CSV
    for r in csv.DictReader(open(p)):
        if r.get('lst') == lst and r.get('subband') == sb:
            out[(r['night'], int(r['window']))] = r.get('approval', '')
    return out


def sublst_model(lst, sb, stack_root=None, refresh=False):
    """Independent model of what the refine owes for (LST, subband), from the frozen weights, the vetoes and the stack
    listing (sublst_reserve_test_residuals.py R1/R2 logic, x2 arm): stacks {(S, k): {night: path}}, nloo {(S,k,night)},
    allowed {(S,k,night)}, wok {(S,k)}, expected {(S,k,night)} (allowed AND the night has a Stokes I stack in an allowed I
    window - the I flow registers V), weights path."""
    stack_root = stack_root or os.path.join(_cfg().paths.SUBLST_ROOT, lst)
    key = (os.path.abspath(stack_root), lst, sb)
    if key in _MODEL_CACHE and not refresh:
        return _MODEL_CACHE[key]
    fr = _cfg().flow_refine
    wts, wp = _weights(lst, sb)
    stacks = {}
    for p in glob.glob(os.path.join(stack_root, f'{lst}_*', sb, '[IV]', f'sublst_{lst}_*_{sb}_*_w*.fits')):
        m = _RES_PAT.match(os.path.basename(p))
        if not m or m['res'] or m['band'] != sb:
            continue
        stacks.setdefault((m['S'], int(m['k'])), {})[m['night']] = p
    nloo, allowed, wok, no_weight = {}, set(), {}, set()
    for (S, k), nd in stacks.items():
        ns = [n for n in nd if wts is not None and n in wts]
        no_weight |= {(S, k, n) for n in nd if wts is None or n not in wts}
        nw = sum(1 for n in ns if wts[n] > 0)
        wok[(S, k)] = len(ns) >= fr.MIN_NIGHTS_PER_WINDOW and nw >= fr.MIN_WEIGHTED_NIGHTS_PER_WINDOW
        for n in ns:
            c = nw - (1 if wts[n] > 0 else 0)
            nloo[(S, k, n)] = c
            if wok[(S, k)] and c >= fr.MIN_LOO_WEIGHTED_NIGHTS:
                allowed.add((S, k, n))
    expected = {(S, k, n) for (S, k, n) in allowed
                if n in stacks.get(('I', k), {}) and wok.get(('I', k), False)}
    out = dict(stacks=stacks, nloo=nloo, allowed=allowed, wok=wok, expected=expected, weights=wts, weights_path=wp,
               no_weight=no_weight, stack_root=stack_root)
    _MODEL_CACHE[key] = out
    return out


def residual_path(x2_root, lst, night, sb, S, k):
    return os.path.join(x2_root, f'{lst}_{night}', sb, S, f'sublst_{lst}_{night}_{sb}_{S}_w{k}_resid_loo.fits')


def residuals_on_disk(x2_root, lst, sb, nights=None):
    out = {}
    for p in glob.glob(os.path.join(x2_root, f'{lst}_*', sb, '[IV]', f'sublst_{lst}_*_{sb}_*_resid_loo.fits')):
        m = _RES_PAT.match(os.path.basename(p))
        if m and m['res'] and m['band'] == sb and (nights is None or m['night'] in nights):
            out[(m['S'], int(m['k']), m['night'])] = p
    return out


def r2_missing(x2_root, lst, sb, nights=None, model=None):
    """(missing {(S,k,night): expected path}, approved {(night,k): text} restricted to the nights): R2 of the x2 arm."""
    model = model or sublst_model(lst, sb)
    have = residuals_on_disk(x2_root, lst, sb, nights)
    exp = {k3 for k3 in model['expected'] if nights is None or k3[2] in nights}
    missing = {k3: residual_path(x2_root, lst, k3[2], sb, k3[0], k3[1]) for k3 in sorted(exp) if k3 not in have}
    appr = {k2: t for k2, t in approved_rows(lst, sb).items() if nights is None or k2[0] in nights}
    return missing, appr


def judge_missing(missing, approved):
    """G-C7 structurally (spec 5): (ok, waived {(night,k)}, reason). No missing residual = complete. Missing ones are
    accepted only when the set {(night, window)} they cover equals EXACTLY the approved rows (sublst_refine_approved.py:36)."""
    if not missing:
        return True, set(), 'every expected residual written'
    nw = {(n, k) for (_S, k, n) in missing}
    if nw == set(approved):
        return True, nw, f'{len(missing)} residual(s) of {len(nw)} approved (night, window) pair(s) not aligned'
    return False, set(), (f'{len(missing)} expected residual(s) missing over (night, window) {sorted(nw)[:6]}; approved '
                          f'{sorted(approved)[:6]} - not exactly the approved set')


def _x2_roots(path, lst, stack_root=None, masters_root=None):
    """(stack root <..>/<LST>, arm-A Masters root <..>/<LST>_masters) for a residual or an <LST>_x2 directory: the caller's
    context, else the siblings of <LST>_x2 when they exist (parity mirrors), else paths.SUBLST_ROOT."""
    p = os.path.abspath(path)
    x2 = p if os.path.basename(p) == f'{lst}_x2' else next(
        (os.sep.join(p.split(os.sep)[:i + 1]) for i, part in enumerate(p.split(os.sep)) if part == f'{lst}_x2'), None)
    sib = os.path.dirname(x2) if x2 else None
    base = _cfg().paths.SUBLST_ROOT
    st = stack_root or (os.path.join(sib, lst) if sib and os.path.isdir(os.path.join(sib, lst)) else os.path.join(base, lst))
    md = _cfg().flow_refine.MASTERS_DIR.format(lst=lst)
    mr = masters_root or (os.path.join(sib, md) if sib and os.path.isdir(os.path.join(sib, md)) else os.path.join(base, md))
    return st, mr


def _residual_key(path):
    m = _RES_PAT.match(os.path.basename(path))
    if not m or not m['res']:
        return None
    return m['lst'], m['night'], m['band'], m['S'], int(m['k'])


@content_test('sublst_x2_residual')
def allowed_and_complete(path, nights=None, subbands=None, stack_root=None, **ctx):
    """R1 on a residual: allowed by the window rule (>= 2 nights, >= 2 weighted) and the per-night rule (own LOO >= 2
    weighted), LOONCOAD = that count. On an <LST>_x2 directory: R1 for every subband residual and R2 - every expected
    (Stokes, window, night) has its residual except the approved (night, window) rows (sublst_reserve_test_residuals.py)."""
    if os.path.isdir(path):
        lst = os.path.basename(os.path.abspath(path)).split('_')[0]
        sbs = subbands or [s for s in _cfg().instrument.ALL_SUBBANDS
                           if os.path.exists(_cfg().flow_refine.WEIGHTS_CSV.format(lst=lst, sb=s))]
        bad, n, n_exp = [], 0, 0
        st_root, _ = _x2_roots(path, lst, stack_root)
        for sb in sbs:
            model = sublst_model(lst, sb, stack_root=st_root)
            n_exp += sum(1 for k3 in model['expected'] if nights is None or k3[2] in nights)
            for k3, p in residuals_on_disk(path, lst, sb, nights).items():
                n += 1
                pr = _r1_problems(p, model)
                if pr:
                    bad.append(f'{os.path.basename(p)}: {pr}')
            missing, appr = r2_missing(path, lst, sb, nights, model)
            ok, _, why = judge_missing(missing, appr)
            if not ok:
                bad.append(f'{sb} R2: {why}')
        return not bad and (n > 0 or n_exp == 0), n, '; '.join(bad[:4]) or (
            f'{n} residuals of {len(sbs)} subbands: R1 and R2 pass' if n else
            f'no residual owed in {len(sbs)} subband(s): every window refused by the >= 2 weighted nights rules')
    k = _residual_key(path)
    if not k:
        return False, None, 'not a sublst_..._resid_loo.fits name'
    lst, night, sb, S, kk = k
    pr = _r1_problems(path, sublst_model(lst, sb, stack_root=_x2_roots(path, lst, stack_root)[0]))
    return not pr, None, pr or 'allowed; LOONCOAD = independent count'


def _r1_problems(path, model):
    lst, night, sb, S, k = _residual_key(path)
    k3 = (S, k, night)
    if k3 not in model['allowed']:
        return f'not allowed (window ok {model["wok"].get((S, k))}, own LOO weighted nights {model["nloo"].get(k3)})'
    c = C.header_cards(path)
    if C.card(c, 'LOONCOAD') != model['nloo'][k3]:
        return f'LOONCOAD {C.card(c, "LOONCOAD")} != independent count {model["nloo"][k3]}'
    return ''


REQUIRED_CARDS = ('SUBTRACT', 'LOONCOAD', 'LOOMSTR', 'CODE', 'LSTBIN', 'DURATION', 'REFINE', 'REFREF', 'FLOWMEDP', 'JUPMASK')


@content_test('sublst_x2_residual')
def cards(path, masters_root=None, stack_root=None, **ctx):
    """REFINE, REFREF, FLOWMEDP, JUPMASK, LOOMSTR, LSTBIN, DURATION (and SUBTRACT, LOONCOAD, CODE) present, BINMSTR absent
    (phase3_sublst_refine.py:222-243); LOOMSTR names an existing arm-A window Master (sublst_resume_from_x2.sh x2ok, G-C9);
    LSTBIN = the window of the name; every pixel where the window stack has no finite data is NaN."""
    import numpy as np
    k = _residual_key(path)
    if not k:
        return True, 'n/a', 'not a residual file' if not os.path.isdir(path) else 'directory: cards run per file'
    lst, night, sb, S, kk = k
    c = C.header_cards(path)
    problems = [f'no {x}' for x in REQUIRED_CARDS if x not in c]
    if 'BINMSTR' in c:
        problems.append('BINMSTR present')
    if C.card(c, 'LSTBIN') not in (None, kk):
        problems.append(f'LSTBIN {C.card(c, "LSTBIN")} != window {kk}')
    st_root, mroot = _x2_roots(path, lst, stack_root, masters_root)
    lm = C.card(c, 'LOOMSTR')
    if lm and not os.path.isfile(os.path.join(mroot, sb, S, str(lm))):
        problems.append(f'LOOMSTR {lm} names no arm-A window Master under {mroot}/{sb}/{S}')
    stack = os.path.join(st_root, f'{lst}_{night}', sb, S,
                         os.path.basename(path).replace('_resid_loo.fits', '.fits'))
    nbad = None
    if os.path.exists(stack):
        r, s = C.image_data(path), C.image_data(stack)
        if r.shape != s.shape:
            problems.append(f'shape {r.shape} != stack {s.shape}')
        else:
            nbad = int((~np.isfinite(s) & np.isfinite(r)).sum())
            if nbad:
                problems.append(f'{nbad} finite pixels where the stack has no data')
    else:
        problems.append(f'window stack {os.path.basename(stack)} not found')
    return not problems, nbad, '; '.join(problems) or f'cards present; LOOMSTR {lm} exists; NaN outside the stack data'
