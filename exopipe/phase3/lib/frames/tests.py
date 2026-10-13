"""Content tests of the product classes `exopipe3 frames` owns (deep tier; spec plan/specs/frames.md section 6).

Classes (core/validate discovers this module; validate.OWNERS maps search_frame, noise_map and epoch_image to
steps.frames, which imports it):

- ``search_frame``   a deep (or sub-LST) search frame AND its primary-beam-corrected sibling (``*_redo_resid_pbcorr_*``),
                     in the attempt directory or as the archive copy. `search` (the deep frame producer during the refactor
                     window) registers both under this class, so frames does too (one class for the search-frame family;
                     the masters' pbcorr_sibling tests look for the PB map beside the file and a cell's master variants, which
                     a search-frame sibling does not have - crosscheck 4.2, open question in plan/impl/frames-deep.md).
                     Tests: full header, own time, Stokes, PRODNAME round trip, rms cards recomputed from the pixels (plain
                     frames), STRIWIN, master actually subtracted, pixels, archive copy byte-identical (ctx copy_of), and for
                     a sibling: sibling x PB == plain where PB > floor, NaN at or below it, PBFILE robust-matched.
- ``noise_map``      the per-cell sigma stack (sigstack): floored at 1, half the clean sky at the floor, cards.
- ``frame_refusals`` the pbcorr-missing record (pbcorr_missing_<cell>.csv) the frame writer appends when a PB map is
                     missing. The per-gate refusal CSVs (refusals/<cell>_<kind>_excluded.csv) are search's class
                     ``refusal_record`` (one owner per class); frames registers them under it and checks "no frame for a
                     whole-epoch removal" per unit.
- ``epoch_image``    the epoch image the co-add writes (subtracted/<date>_<S>_epoch_<scope>.fits; masters' product): runs
                     masters' ``coadd_epoch`` tests (the one definition, coadd._epoch_content_arrays; crosscheck 4.2 makes
                     coadd_epoch the class - epoch_image stays only because validate.OWNERS still names it).

Every test returns (passed, value, detail) and reads the file's CONTENT; heavy imports stay inside the functions.
Thresholds come from the effective config keys (lib/frames/deep_engine.engine_values: search's sections while search
produces the deep frames). Context keys a step may pass (all optional): stokes, in_master (bool), ncoadd (int, the
own-Stokes central master's NCOADD), copy_of (source of an archive copy), pb_path (the PB map of a sibling),
plain_path (the plain frame of a sibling), floor, cell, tier.
"""
import csv
import os
import re

from exopipe.phase3.core.validate import content_test

_FRAME_RE = re.compile(r'^(?P<d>\d{4}-\d\d-\d\d(?:T\d\d)?)_(?P<s>[IV])_redo_resid(?P<pb>_pbcorr)?_(?P<scope>\w+?)\.fits$')
_SIGSTACK_RE = re.compile(r'^(?P<lst>\d\dh)_(?P<scope>\w+?)_(?P<s>[IV])_sigstack\.fits$')
_PBMISS_RE = re.compile(r'^pbcorr_missing_(?P<lst>\d\dh)_(?P<scope>\w+?)\.csv$')
_DATE = re.compile(r'^\d{4}-\d\d-\d\d(T\d\d)?$')
_DATE_DEEP = re.compile(r'^\d{4}-\d\d-\d\d$')


def _ev(tier='deep'):
    from exopipe.phase3.lib.frames.deep_engine import engine_values
    return engine_values(tier)


def _header_data(path):
    import numpy as np
    from astropy.io import fits
    with fits.open(path, memmap=False) as h:
        return h[0].header.copy(), np.asarray(h[0].data)


def _frame_ident(path, hdr=None):
    """(date, Stokes, scope, pbcorr) of a frame from its attempt name, else from its archive path and PRODNAME."""
    m = _FRAME_RE.match(os.path.basename(path))
    if m:
        return m['d'], m['s'], m['scope'], bool(m['pb'])
    name = str((hdr or {}).get('PRODNAME', '') or os.path.basename(path))
    # archive layout <archive>/<LST>/<night>/<RUN>/<scope>/<S>/<cadence>/<PRODNAME> (phase3_deep.py:868): the night
    # (the frame key) is the directory, not the UTC date of the epoch token
    parts = os.path.abspath(path).split(os.sep)
    night = parts[-6] if len(parts) >= 7 and _DATE_DEEP.match(parts[-6]) else None
    try:
        from exopipe.phase3.core.products import parse_name
        p = parse_name(name)
        ep = str(p.get('epoch') or '')
        d = night or (f'{ep[:4]}-{ep[4:6]}-{ep[6:8]}' if len(ep) >= 8 else None)
        if d and p.get('lst_bin') is not None and 'T' not in d:
            d = None          # a 10-minute key needs its window; not derivable from the name alone
        return d, p.get('stokes'), p.get('scope'), 'pbcorr' in (p.get('proc') or ())
    except Exception:
        return None, (hdr or {}).get('STOKES'), None, bool((hdr or {}).get('PBCORR'))


def is_sibling(path, hdr=None):
    """True for a search-frame pbcorr sibling (attempt name or an archive copy with PBCORR and a msub PRODNAME)."""
    if '_redo_resid_pbcorr_' in os.path.basename(path):
        return True
    if hdr is None:
        from astropy.io import fits
        hdr = fits.getheader(path)
    pn = str(hdr.get('PRODNAME', '') or '')
    return bool(hdr.get('PBCORR')) and '_msub' in pn


# ---------------------------------------------------------------------------------------------------------------------
# search_frame (plain frames and their pbcorr siblings)
# ---------------------------------------------------------------------------------------------------------------------

@content_test('search_frame')
def header_full(path, **ctx):
    """validate_header(level='full') passes (verify_p3s3_stage.py:129-137; R-CONFORMANT-FROM-BIRTH-INCL-HEADERS)."""
    from exopipe.common.validate_fits_headers import validate_header
    rep = validate_header(path, level='full')
    return rep.ok, len(rep.errors), ('; '.join(rep.errors)[:200] if rep.errors else 'full header valid')


@content_test('search_frame')
def own_time(path, **ctx):
    """The frame carries its OWN time block: DATE-OBS, DATE-END, MJD-MID present and MJD-MID != the parent master's
    MSTRMJD (verify_p3s3_stage.py:133)."""
    hdr, _ = _header_data(path)
    mid, mst = hdr.get('MJD-MID'), hdr.get('MSTRMJD')
    if mid is None:
        return False, None, 'no MJD-MID'
    if hdr.get('DATE-OBS') is None or hdr.get('DATE-END') is None:
        return False, mid, 'no DATE-OBS / DATE-END'
    return mid != mst, mid, f'MJD-MID {mid} vs master MSTRMJD {mst}'


@content_test('search_frame')
def stokes_card(path, stokes=None, **ctx):
    """STOKES equals the unit's Stokes (ctx) or the Stokes of the frame's name."""
    hdr, _ = _header_data(path)
    want = stokes or _frame_ident(path, hdr)[1]
    got = str(hdr.get('STOKES', '')).strip()
    return bool(want) and got == want, got, f'STOKES {got!r}, expected {want!r}'


@content_test('search_frame')
def prodname(path, **ctx):
    """PRODNAME present, not 'unknown', parses with core.products.parse_name and round-trips build_name; its pbcorr token
    agrees with PBCORR; an archive copy is named exactly PRODNAME (phase3_deep.py:698-707, :816-818, :864-866)."""
    from exopipe.phase3.core.products import build_name, parse_name
    hdr, _ = _header_data(path)
    pn = str(hdr.get('PRODNAME', '') or '')
    if not pn or pn == 'unknown':
        return False, pn, 'PRODNAME missing or unknown (build_name refused at write)'
    try:
        p = parse_name(pn)
    except ValueError as e:
        return False, pn, f'PRODNAME does not parse: {e}'
    if p.get('grammar') != 'phase3' or build_name(**p) != pn:
        return False, pn, 'PRODNAME does not round-trip build_name'
    is_attempt = bool(_FRAME_RE.match(os.path.basename(path)))
    if not is_attempt and os.path.basename(path) != (pn if pn.endswith('.fits') else pn + '.fits'):
        return False, pn, f'archive copy named {os.path.basename(path)}, PRODNAME {pn}'
    pb = 'pbcorr' in (p.get('proc') or ())
    if pb != bool(hdr.get('PBCORR')):
        return False, pn, f'PRODNAME pbcorr token {pb} != PBCORR {hdr.get("PBCORR")}'
    return True, pn, 'round-trips'


@content_test('search_frame')
def rms_cards(path, **ctx):
    """Plain frame: IMGRMS (and THERMNOI on Stokes V) equal mad_std of the central box of half-size min(512, N/4),
    recomputed from the stored pixels (relative 1e-6 after the 9-decimal rounding; phase3_deep.py:1617-1622,
    1731-1735). A sibling carries the plain frame's cards (copied header): n/a."""
    import numpy as np
    from astropy.stats import mad_std
    hdr, d = _header_data(path)
    if hdr.get('PBCORR'):
        return True, None, 'n/a (pbcorr sibling: the rms cards are the plain frame\'s)'
    d = np.squeeze(d)
    n0, n1 = d.shape
    half = min(int(_ev(ctx.get('tier', 'deep'))['rms_box_half_px']), n0 // 4, n1 // 4)
    box = d[n0 // 2 - half:n0 // 2 + half, n1 // 2 - half:n1 // 2 + half].astype(np.float64)
    r = float(mad_std(box, ignore_nan=True))
    keys = ['IMGRMS'] + (['THERMNOI'] if str(hdr.get('STOKES', '')).strip() == 'V' else [])
    for k in keys:
        v = hdr.get(k)
        if v is None:
            return False, None, f'no {k}'
        if not abs(float(v) - r) <= 1e-6 * max(abs(r), 1e-30) + 1e-9:
            return False, float(v), f'{k} {v} != recomputed {r:.9g}'
    return True, r, f'{"/".join(keys)} = {r:.6g} recomputed'


@content_test('search_frame')
def striwin(path, **ctx):
    """STRIWIN = STRIATION_WINDOW_BEAMS (8) x EBMAJ (deg, within 1e-6) when the striation model was subtracted (SSUB),
    else 0 (R-STRIATION-WINDOW-APPROVED-8BEAMS; phase3_deep.py:669)."""
    hdr, _ = _header_data(path)
    sw, eb, ssub = hdr.get('STRIWIN'), hdr.get('EBMAJ'), hdr.get('SSUB')
    if sw is None:
        return False, None, 'no STRIWIN'
    if ssub:
        if eb is None:
            return False, sw, 'SSUB but no EBMAJ'
        want = _ev(ctx.get('tier', 'deep'))['striation_window_beams'] * float(eb)
        return abs(float(sw) - want) <= 1e-6, float(sw), f'STRIWIN {sw} vs {want:.9g} (8 x EBMAJ)'
    return float(sw) == 0.0, float(sw), 'no striation model: STRIWIN must be 0'


@content_test('search_frame')
def master_used(path, in_master=None, ncoadd=None, **ctx):
    """The master actually subtracted: LOOMSTR True <=> MSTRFILE is the epoch's own leave-one-out master
    (<date>_<S>_loo_*) with LOOEXCL = the epoch; MSUBFILE = MSTRFILE; with ctx in_master and ncoadd, an in-master epoch of
    a 2 <= NCOADD <= 20 stack used its LOO master (R-LOO-MASTERS; audit R-19) and a night outside the master (no LOO
    master exists for it) the central master (phase3_deep.py:661-664, 684-685, 1394-1433)."""
    hdr, _ = _header_data(path)
    d, s, scope, _ = _frame_ident(path, hdr)
    loo, mf, ms = hdr.get('LOOMSTR'), str(hdr.get('MSTRFILE', '') or ''), str(hdr.get('MSUBFILE', '') or '')
    if loo is None or not mf:
        return False, None, 'no LOOMSTR / MSTRFILE'
    if ms and ms != mf:
        return False, mf, f'MSUBFILE {ms} != MSTRFILE {mf}'
    if loo:
        if '_loo_' not in mf or (d and not mf.startswith(f'{d}_{s}_loo_')):
            return False, mf, f'LOOMSTR True but MSTRFILE {mf} is not this epoch\'s LOO master'
        if d and str(hdr.get('LOOEXCL', '')) != d:
            return False, mf, f'LOOEXCL {hdr.get("LOOEXCL")!r} != epoch {d}'
    elif '_loo_' in mf:
        return False, mf, 'LOOMSTR False but MSTRFILE names a LOO master'
    if in_master is not None and ncoadd is not None and str(hdr.get('CADENCE', 'deep')).strip() == 'deep':
        lo, hi = _ev('deep')['loo_ncoadd_range']
        if bool(in_master) and lo <= int(ncoadd) <= hi and not loo:
            return False, mf, f'in-master epoch of an N={ncoadd} stack subtracted the central master'
        if not bool(in_master) and loo:
            return False, mf, 'night outside the master subtracted a LOO master'
    return True, mf, ('LOO master' if loo else 'central master')


@content_test('search_frame')
def pixels(path, **ctx):
    """2-D float32 with finite, non-zero pixels."""
    import numpy as np
    hdr, d = _header_data(path)
    d = np.squeeze(d)
    if d.ndim != 2 or d.dtype.kind != 'f' or d.dtype.itemsize != 4:
        return False, str(d.dtype), f'data {d.dtype} {d.shape}: expected 2-D float32'
    fin = np.isfinite(d)
    frac = float(fin.mean())
    if frac <= 0 or not np.any(d[fin] != 0):
        return False, frac, 'no finite non-zero pixel'
    return True, round(frac, 4), f'finite fraction {frac:.4f}'


@content_test('search_frame')
def copy_identical(path, copy_of=None, **ctx):
    """An archive copy is byte-identical to the attempt frame it was copied from (phase3_deep.py:850-873 copy2)."""
    if not copy_of:
        return True, None, 'n/a (not an archive copy in this context)'
    import hashlib

    def md5(p):
        h = hashlib.md5()
        with open(p, 'rb') as f:
            for blk in iter(lambda: f.read(1 << 22), b''):
                h.update(blk)
        return h.hexdigest()
    a, b = md5(path), md5(copy_of)
    return a == b, a[:12], ('identical to ' if a == b else 'DIFFERS from ') + os.path.basename(copy_of)


def _resolve_pb(path, hdr, cell=None):
    """The PB map of a deep search-frame sibling: <co-add root>/<cell>/coadd_final/<PBFILE>, the root from the co-add root
    table (search_inputs.ROOTS_CSV). None when it cannot be located on this node."""
    pbfile = str(hdr.get('PBFILE', '') or '')
    if not pbfile:
        return None
    if cell is None:
        parent = os.path.basename(os.path.dirname(path))
        cell = parent if re.match(r'^\d\dh_\w+$', parent) else None
    if cell is None:
        return None
    try:
        from exopipe.phase3.lib.frames.deep_cells import roots_table
        root = roots_table().get(cell)
    except Exception:
        root = None
    if not root:
        return None
    p = os.path.join(root, cell, 'coadd_final', pbfile)
    return p if os.path.exists(p) else None


def _plain_of(path):
    b = os.path.basename(path)
    if '_redo_resid_pbcorr_' in b:
        return os.path.join(os.path.dirname(path), b.replace('_redo_resid_pbcorr_', '_redo_resid_', 1))
    return os.path.join(os.path.dirname(path), b.replace('_pbcorr_dewarped_', '_dewarped_', 1))


@content_test('search_frame')
def pbcorr_matches_plain(path, pb_path=None, plain_path=None, floor=None, cell=None, **ctx):
    """Sibling only (plain frame: n/a). Search-frame sibling (phase3_deep.py:787-822; the pb_apply.sibling_check rule with
    the search-frame floor search_detect.PB_FLOOR = 0.05, carried in no card): PBCORR True; PBFILE = the robust-matched map
    (I -> PB_deep_R075_<scope>, V -> PB_deep_<scope>; 10 min -> PB_bin<k>); PRODNAME has _pbcorr_dewarped_; sibling x PB
    == plain within 1e-5 relative where PB > floor, NaN at or below it. When the PB map cannot be located on this node:
    plain / sibling must lie in (floor, 1] (the PB values) wherever both are non-zero."""
    import numpy as np
    from astropy.io import fits
    hdr = fits.getheader(path)
    if not is_sibling(path, hdr):
        return True, None, 'n/a (plain search frame)'
    floor = float(_ev(ctx.get('tier', 'deep'))['pb_floor'] if floor is None else floor)
    d, s, scope, _ = _frame_ident(path, hdr)
    if hdr.get('PBCORR') is not True:
        return False, None, 'PBCORR not True'
    pbfile = str(hdr.get('PBFILE', '') or '')
    cad = str(hdr.get('CADENCE', 'deep')).strip()
    if cad == '10min':
        ok_pb = bool(re.match(rf'^PB_bin\d+_{re.escape(str(scope))}\.fits$', pbfile))
    else:
        ok_pb = pbfile == (f'PB_deep_R075_{scope}.fits' if s == 'I' else f'PB_deep_{scope}.fits')
    if not ok_pb:
        return False, pbfile, f'PBFILE {pbfile!r} is not the robust-matched map for Stokes {s} ({cad})'
    if '_pbcorr_dewarped_' not in str(hdr.get('PRODNAME', '')):
        return False, hdr.get('PRODNAME'), 'PRODNAME lacks _pbcorr_dewarped_'
    plain_path = plain_path or _plain_of(path)
    if not os.path.exists(plain_path):
        return False, None, f'plain frame {os.path.basename(plain_path)} not found next to the sibling'
    sib = np.squeeze(fits.getdata(path)).astype(np.float64)
    plain = np.squeeze(fits.getdata(plain_path)).astype(np.float64)
    if sib.shape != plain.shape:
        return False, None, f'shape {sib.shape} != plain {plain.shape}'
    pbp = pb_path or _resolve_pb(path, hdr, cell)
    if pbp:
        pb = np.squeeze(fits.getdata(pbp)).astype(np.float32).astype(np.float64)    # the writer's float32 PB
        if pb.shape != sib.shape:
            return False, None, f'PB map shape {pb.shape} != sibling {sib.shape}'
        inside = pb > floor
        k = inside & np.isfinite(plain)
        err = np.abs(sib[k] * pb[k] - plain[k])
        bad = int(np.count_nonzero(~(err <= 1e-5 * np.abs(plain[k]) + 1e-12)))
        outside_ok = bool(np.all(np.isnan(sib[~inside])))
        if bad or not outside_ok or not k.any():
            return False, bad, '; '.join(x for x in ((f'{bad} pixels with sibling x PB != plain' if bad else ''),
                                                     ('' if outside_ok else f'pixels at PB <= {floor} not NaN'),
                                                     ('' if k.any() else 'no pixel above the floor')) if x)
        return True, int(k.sum()), f'sibling x PB == plain at {int(k.sum())} pixels (PB {os.path.basename(pbp)})'
    k = np.isfinite(sib) & (sib != 0) & (plain != 0)
    if not k.any():
        return False, 0, 'no pixel where sibling and plain are both non-zero'
    r = plain[k] / sib[k]
    ok = bool(np.all((r > floor * (1 - 1e-5)) & (r <= 1.0 + 1e-5)))
    return ok, round(float(np.nanmin(r)), 6), (f'plain/sibling in [{np.nanmin(r):.6f}, {np.nanmax(r):.6f}] '
                                                f'(PB map not located on this node: range check only)')


# ---------------------------------------------------------------------------------------------------------------------
# noise_map
# ---------------------------------------------------------------------------------------------------------------------

@content_test('noise_map')
def sigstack_floor(path, **ctx):
    """2-D float32, every pixel finite, minimum >= frames_residuals.SIGSTACK_FLOOR (1.0; phase3_pipeline.py:1383-1390)."""
    import numpy as np
    hdr, d = _header_data(path)
    d = np.squeeze(d)
    if d.ndim != 2 or d.dtype.kind != 'f' or d.dtype.itemsize != 4:
        return False, str(d.dtype), f'data {d.dtype} {d.shape}: expected 2-D float32'
    if not np.isfinite(d).all():
        return False, int((~np.isfinite(d)).sum()), 'non-finite pixels'
    fl = _ev()['sigstack_floor']
    mn = float(d.min())
    return mn >= fl, mn, f'min {mn} (floor {fl})'


@content_test('noise_map')
def sigstack_norm(path, **ctx):
    """Clean-sky normalisation: the map is divided by the median of its central half (rows and columns n/4..3n/4) and
    then floored at 1 (phase3_pipeline.py:1385-1390), so at least half of the central-half pixels sit exactly at the floor
    - the lower middle order statistic equals the floor (the plain median of an even pixel count averages it with the next
    value, e.g. 1.0000014 on 07h 18MHz I: not a defect)."""
    import numpy as np
    _, d = _header_data(path)
    d = np.squeeze(d).astype(np.float64)
    q = d.shape[0] // 4
    c = np.sort(d[q:3 * q, q:3 * q].ravel())
    fl = _ev()['sigstack_floor']
    lower_mid = float(c[len(c) // 2 - 1]) if len(c) > 1 else float('nan')
    frac = float((c == fl).mean()) if len(c) else 0.0
    return lower_mid == fl, round(frac, 6), (f'{frac:.4f} of the central half at the floor {fl} (lower middle value '
                                             f'{lower_mid:.9g}); normalised to the clean-sky median before the floor')


@content_test('noise_map')
def sigstack_cards(path, **ctx):
    """BUNIT 'ratio', PRODUCT 'sigstack', STOKES = the file name's Stokes, time and beam cards copied from the master
    (phase3_deep.py:825-843)."""
    hdr, _ = _header_data(path)
    m = _SIGSTACK_RE.match(os.path.basename(path))
    want_s = m['s'] if m else None
    probs = []
    if str(hdr.get('BUNIT', '')).strip() != 'ratio':
        probs.append(f'BUNIT {hdr.get("BUNIT")!r}')
    if str(hdr.get('PRODUCT', '')).strip() != 'sigstack':
        probs.append(f'PRODUCT {hdr.get("PRODUCT")!r}')
    st = str(hdr.get('STOKES', '')).strip()
    if st not in ('I', 'V') or (want_s and st != want_s):
        probs.append(f'STOKES {st!r} (name says {want_s!r})')
    for k in ('DATE-OBS', 'BMAJ', 'BMIN'):
        if hdr.get(k) is None:
            probs.append(f'no {k}')
    return not probs, st, '; '.join(probs) if probs else 'cards ok'


# ---------------------------------------------------------------------------------------------------------------------
# frame_refusals: the pbcorr-missing record
# ---------------------------------------------------------------------------------------------------------------------

def refusal_rows(path):
    """Rows of a headerless refusal CSV (csv.reader: the provenance gate quotes its reason)."""
    with open(path, newline='') as f:
        return [r for r in csv.reader(f) if r]


@content_test('frame_refusals')
def pbcorr_missing_rows(path, **ctx):
    """pbcorr_missing_<lst>_<scope>.csv (phase3_deep.py:803-805; headerless, appended): every row has 4 columns
    '<lst> <scope>', date, frame file name, PB map path; the cell is the file's; the frame named is that date's frame
    (<date>_<S>_redo_resid_<scope>.fits); the PB map named does not exist (a missing map is why the row was written)."""
    m = _PBMISS_RE.match(os.path.basename(path))
    if not m:
        return False, None, f'not a pbcorr_missing_<lst>_<scope>.csv name: {os.path.basename(path)}'
    rows = refusal_rows(path)
    if not rows:
        return False, 0, 'empty pbcorr-missing record (the writer appends a row per frame)'
    bad = []
    for i, r in enumerate(rows, 1):
        if len(r) != 4:
            bad.append(f'row {i}: {len(r)} columns != 4')
            continue
        if r[0] != f"{m['lst']} {m['scope']}":
            bad.append(f'row {i}: cell {r[0]!r}')
        if not _DATE.match(r[1]):
            bad.append(f'row {i}: date {r[1]!r}')
        fm = _FRAME_RE.match(r[2])
        if not fm or fm['d'] != r[1] or fm['scope'] != m['scope']:
            bad.append(f'row {i}: frame {r[2]!r} is not the {r[1]} frame of {m["scope"]}')
        if os.path.exists(r[3]):
            bad.append(f'row {i}: PB map {r[3]} exists (the sibling is owed: make it)')
    return not bad, len(rows), '; '.join(bad[:4]) if bad else f'{len(rows)} frames without a sibling (PB map missing)'


def removed_without_frame(refusal_path, frame_dirs):
    """(ok, detail) of the unit-level check: no frame in ``frame_dirs`` for an epoch a whole-epoch gate removed (rows of
    a refusals/<cell>_<kind>_excluded.csv; R-BEAM-AND-10X-REFUSED-EPOCHS-STAY-OUT-NO-FRAMES-OWED)."""
    dates = {r[2] for r in refusal_rows(refusal_path) if len(r) > 2}
    hits = []
    for dd in frame_dirs:
        if os.path.isdir(dd):
            hits += [f for f in os.listdir(dd) if _FRAME_RE.match(f) and _FRAME_RE.match(f)['d'] in dates]
    return (not hits), (('frames exist for removed epochs: ' + ', '.join(sorted(hits)[:4])) if hits else
                        f'{len(dates)} removed epoch(s), no frame')


# ---------------------------------------------------------------------------------------------------------------------
# epoch_image = masters' coadd_epoch (one definition)
# ---------------------------------------------------------------------------------------------------------------------

@content_test('epoch_image')
def as_coadd_epoch(path, **ctx):
    """Every test of masters' class coadd_epoch on this file (identity epoch - residual - master; Stokes I flux ratio at
    the master's compact sources; coadd._epoch_content_arrays): epoch_image is that product under its OWNERS name."""
    from exopipe.phase3.core import validate
    res = validate.run('coadd_epoch', path, **ctx)
    bad = [r for r in res if not r['passed']]
    return not bad, len(res), ('; '.join(f"{r['name']}: {r['detail']}" for r in bad)[:300] if bad else
                               f"{len(res)} coadd_epoch tests pass: " + '; '.join(str(r['detail']) for r in res)[:200])
