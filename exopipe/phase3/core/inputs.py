"""Inputs - the ONE resolver of Phase 1 snapshot frames (and, as steps move in, Phase 2 images) for every tier.

Phase 1 frames: ``phase1_frames(lst, date, subband)`` -> {'I': [(key, path)], 'V': [(key, path)]}, one frame per 10-s
INTEGRATION. A cell can hold several Phase 1 runs (re-images, patch runs). The pre-refactor discovery
(clv15/phase3_snapshot_search.discover_frames) keyed frames by their FILE-NAME timestamp with 'latest run wins'; two
runs of the same hour carry file-name timestamps 1-9 s apart, so it kept BOTH: 83 cell-bands in 41 cells mixed 2-3 runs
(12h 2025-01-10 55 MHz: 681 Stokes I frames for one hour; ledger IMPLEMENTED-DATA-DECISIONS-VIS50-20261012, FINDING;
handed to the refactor by Gregg 2026-10-13 ~00:35 UTC, "Hand it to refactor").

The rule here: frames are keyed by their INTEGRATION (the integration midpoint from the header DATE-OBS, truncated to the
second - the label Phase 1 names frames by since exopipe 192c81c), and the LATEST run wins per integration; an older run
contributes only integrations the later runs lack (a patch run fills its gap and nothing else). Measured on 12h 2025-01-10
55 MHz: the three runs share 357-358 integration labels exactly (DATE-OBS agrees across runs); the union is 358, not 681.
Headers are read only where more than one run holds frames of the subband (elsewhere a file-name timestamp is already one
integration). ``phase1_frames`` refuses a result with more frames than integrations in the hour (content test).

Frame file names (Phase 1): old era '<sb>-Pilot-Snapshot-<ts>-<I|V>-image.fits[.fs|.fz]' in snapshots/; new era
'<sb>-Pilot-Snapshot-<ts>-image.fits' = Stokes V in snapshots/, '<sb>-Clean-Snapshot-<ts>-image.fits' = Stokes I in
snapshots_clean/. CLEAN files named by interval index (t0000...) come from CLEAN runs that stopped partway and are not
frames (refactor finding D12): they are counted and reported, never used.
"""
import datetime
import glob
import os
import re

from exopipe.phase3.core.record import Refusal

_PAT_OLD = re.compile(r'^(?P<sb>\d+MHz)-Pilot-Snapshot-(?P<ts>\d{8}_\d{6})-(?P<st>[IV])-image\.fits(?P<fz>\.f[sz])?$')
_PAT_NEW_V = re.compile(r'^(?P<sb>\d+MHz)-Pilot-Snapshot-(?P<ts>\d{8}_\d{6})-image\.fits(?P<fz>\.f[sz])?$')
_PAT_NEW_I = re.compile(r'^(?P<sb>\d+MHz)-Clean-Snapshot-(?P<ts>\d{8}_\d{6})-image\.fits(?P<fz>\.f[sz])?$')
_PAT_INDEX = re.compile(r'^(?P<sb>\d+MHz)-.*-t\d{4}-image\.fits(\.f[sz])?$')
_EPOCH = datetime.datetime(1970, 1, 1)
_DATEOBS_KEY = b'DATE-OBS='


def _cfg():
    from exopipe.phase3.config import CFG
    return CFG


def read_dateobs(path, nblk=8):
    """DATE-OBS of a FITS file (plain or tile-compressed), read from the first nblk 2880-byte blocks; '' if absent."""
    fd = os.open(path, os.O_RDONLY)
    try:
        buf = os.pread(fd, 2880 * nblk, 0)
    finally:
        os.close(fd)
    for i in range(0, len(buf) - 79, 80):
        card = buf[i:i + 80]
        if card.startswith(_DATEOBS_KEY):
            return card[10:].split(b'/')[0].strip().strip(b"'").strip().decode()
    if len(buf) == 2880 * nblk:
        from astropy.io import fits
        with fits.open(path, memmap=False) as hl:
            for hdu in hl:
                if 'DATE-OBS' in hdu.header:
                    return str(hdu.header['DATE-OBS']).strip()
    return ''


def integration_label(date_obs):
    """'YYYYMMDD_HHMMSS' of the integration MIDPOINT (DATE-OBS = integration start) truncated to the second."""
    s = date_obs.strip().replace('T', ' ')
    fmt = '%Y-%m-%d %H:%M:%S.%f' if '.' in s else '%Y-%m-%d %H:%M:%S'
    t0 = datetime.datetime.strptime(s, fmt)
    mid = (t0 - _EPOCH).total_seconds() + _cfg().instrument.INTEGRATION_S / 2.0
    return (_EPOCH + datetime.timedelta(seconds=int(mid // 1))).strftime('%Y%m%d_%H%M%S')


def _dateobs_many(paths, nthreads=16):
    from concurrent.futures import ThreadPoolExecutor
    if not paths:
        return {}
    with ThreadPoolExecutor(max_workers=max(1, min(nthreads, len(paths)))) as ex:
        return dict(zip(paths, ex.map(read_dateobs, paths)))


def _scan_run(rd, subband):
    """{'I': {ts: path}, 'V': {ts: path}} of one run directory, plus the count of index-named CLEAN files."""
    out = {'I': {}, 'V': {}}
    n_index = 0
    for sub, hint in (('snapshots', None), ('snapshots_clean', 'I')):
        d = os.path.join(rd, subband, sub)
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            m = _PAT_OLD.match(fn)
            if m:
                out[m['st']][m['ts']] = os.path.join(d, fn)
                continue
            m = _PAT_NEW_I.match(fn)
            if m:
                out['I'][m['ts']] = os.path.join(d, fn)
                continue
            m = _PAT_NEW_V.match(fn)
            if m and hint != 'I':
                out['V'][m['ts']] = os.path.join(d, fn)
                continue
            if _PAT_INDEX.match(fn):
                n_index += 1
    return out, n_index


def phase1_frames(lst, date, subband, phase1_root=None, report=None):
    """One frame per integration per Stokes for (LST, date, subband): {'I': [(key, path)], 'V': [(key, path)]}, sorted.

    key = the file-name timestamp when only one run holds frames of this subband, else the integration label from the
    header. ``report`` (a dict, optional) receives: runs (run -> frame counts), mixed (bool), dropped (frames of older
    runs superseded per integration), index_named (CLEAN files named by interval index, not used)."""
    root = phase1_root or _cfg().paths.PHASE1_ROOT
    rundirs = sorted(glob.glob(os.path.join(root, lst, date, 'Run_*')))
    if not rundirs:
        raise Refusal(f'no Phase 1 runs for {lst} {date} under {root}')
    per_run = []
    n_index = 0
    for rd in rundirs:
        fr, ni = _scan_run(rd, subband)
        n_index += ni
        if fr['I'] or fr['V']:
            per_run.append((os.path.basename(rd), fr))
    rep = {'runs': {r: {'I': len(f['I']), 'V': len(f['V'])} for r, f in per_run}, 'index_named': n_index,
           'mixed': False, 'dropped': {'I': 0, 'V': 0}}
    out = {}
    for st in ('I', 'V'):
        contributing = [(r, f[st]) for r, f in per_run if f[st]]
        if len(contributing) <= 1:
            items = sorted(contributing[0][1].items()) if contributing else []
        else:
            rep['mixed'] = True
            chosen = {}
            # later runs first: the latest run wins per integration; older runs only fill integrations still missing
            for r, frames in reversed(contributing):
                dob = _dateobs_many(list(frames.values()))
                for ts, p in frames.items():
                    d = dob.get(p, '')
                    if not d:
                        raise Refusal(f'{p}: no DATE-OBS - cannot assign the frame to an integration')
                    key = integration_label(d)
                    if key in chosen:
                        rep['dropped'][st] += 1
                    else:
                        chosen[key] = p
            items = sorted(chosen.items())
        out[st] = items
    # content test: never more frames than integrations in the hour (+1 for a frame straddling the boundary)
    max_frames = int(3600.0 / _cfg().instrument.INTEGRATION_S) + 2
    for st in ('I', 'V'):
        if len(out[st]) > max_frames:
            raise Refusal(f'{lst} {date} {subband} Stokes {st}: {len(out[st])} frames for one hour > {max_frames} '
                          f'integrations - frames from several runs are still mixed')
    if report is not None:
        report.update(rep)
    return out


# =====================================================================================================================
# Phase 2 deep images of one night (B-03 (a), Gregg 2026-10-13; the same-state reading, 2026-10-14)
# =====================================================================================================================
# The ONE copy of the co-add's five image finders (lib/masters/coadd.py = production coadd_full_pipeline.py):
#   find_subband_alignment_image, find_wideband_alignment_image  -> survey_image   (I Taper Robust-0.75: the flow is measured on it)
#   find_subband_variant_image, find_wideband_deep_images        -> variant_image  (the I variants)
#   find_stokes_v_deep_image                                     -> stokes_v_image (V Taper Robust-0)
# Each chooses as its finder does: the first match in the sorted directory listing, the dewarped file preferred and the raw
# file the fallback (the co-add's "failsafe"). FINDER_PINS are the sha1 of each finder's source text (from its 'def' to the
# next top-level statement) this copy was made from; lib/masters/input_resolver refuses to stand in for a finder whose text
# changed (a fold-in of a production edit needs a re-port here first).
FINDER_PINS = {'find_subband_alignment_image': '538ca2958ddf70966cac8253dea9f34572577d6e',
               'find_wideband_alignment_image': '2f1081e46c0010d2de1d08cc895b2093b1a2ada3',
               'find_subband_variant_image': 'fd19d7ffbe0a6556ac77908cc7004fa8392478f8',
               'find_wideband_deep_images': '2b6f4f885cfcfd1ea2a89ce89c6121f29ab10136',
               'find_stokes_v_deep_image': '2ea538378dddfc8ce4bfb2e0e314925c1a21d31f',
               'is_subband': 'd55ee27938d347484d6c7508dad692d619b86b8e',
               'find_latest_science_dir': 'b5cea611a115359a2a15c1fcc83e78d155c5f0c8'}
VARIANTS = ('Taper_R0', 'Taper_R075', 'NoTaper_R0', 'NoTaper_R075')


def source_block(text, name):
    """The source text of the top-level function ``name`` in a module's text: its 'def' line to the next top-level statement
    (the unit FINDER_PINS hash); None when absent."""
    m = re.search(r'^def ' + re.escape(name) + r'\(.*?(?=^\S)', text, re.S | re.M)
    return m.group(0) if m else None


def pin_mismatches(module_text):
    """[finder] whose source text in ``module_text`` differs from FINDER_PINS (absent = differs)."""
    import hashlib
    out = []
    for name, want in FINDER_PINS.items():
        blk = source_block(module_text, name)
        if blk is None or hashlib.sha1(blk.encode()).hexdigest() != want:
            out.append(name)
    return out


def is_subband(freq):
    """= coadd.is_subband: a subband given by its MHz number (int or str), one of the instrument's subbands."""
    try:
        return int(freq) in {int(str(s).replace('MHz', '')) for s in _cfg().instrument.ALL_SUBBANDS}
    except (ValueError, TypeError):
        return False


def latest_science_dir(lst, date, phase2_root=None):
    """= coadd.find_latest_science_dir: the night's latest Phase 2 Science_* directory, or None."""
    dirs = sorted(glob.glob(f'{phase2_root or _cfg().paths.PHASE2_ROOT}/{lst}/{date}/Science_*'))
    return dirs[-1] if dirs else None


def _listing(d):
    return sorted(os.listdir(d)) if os.path.isdir(d) else None


def survey_subband(sci_dir, freq_mhz):
    """= coadd.find_subband_alignment_image (without its print): (path or None, raw) of a subband night's I survey image,
    Taper Robust-0.75, not pbcorr; raw: no dewarped file, the raw one is used (the co-add's '[failsafe]')."""
    d = f'{sci_dir}/{freq_mhz}MHz/I/deep'
    files = _listing(d)
    if files is None:
        return None, False

    def base(f):
        return 'Taper' in f and 'NoTaper' not in f and 'Robust-0.75' in f and 'pbcorr' not in f and f.endswith('.fits')
    for f in files:
        if base(f) and f.endswith('_dewarped.fits'):
            return os.path.join(d, f), False
    for f in files:
        if base(f) and 'dewarped' not in f and f.endswith('-image.fits'):
            return os.path.join(d, f), True
    return None, False


def survey_wideband(sci_dir, band):
    """= coadd.find_wideband_alignment_image (without its print): (path or None, raw) of a band night's I survey image."""
    d = f'{sci_dir}/{band}/I/deep'
    files = _listing(d)
    if files is None:
        return None, False

    def base(f):
        return 'Taper' in f and 'NoTaper' not in f and 'Robust-0.75' in f and 'pbcorr' not in f and f.endswith('.fits')
    for f in files:
        if base(f) and 'dewarped' in f:
            return os.path.join(d, f), False
    for f in files:
        if base(f) and 'dewarped' not in f:
            return os.path.join(d, f), True
    return None, False


def variant_subband(sci_dir, freq_mhz, variant):
    """= coadd.find_subband_variant_image: a subband night's I variant image ('Taper_R0', 'NoTaper_R0', 'NoTaper_R075'),
    not pbcorr, dewarped preferred."""
    taper, rob = variant.split('_')
    rtok = 'Robust-0.75' if rob == 'R075' else 'Robust-0'
    d = f'{sci_dir}/{freq_mhz}MHz/I/deep'
    files = _listing(d)
    if files is None:
        return None

    def base(f):
        if 'pbcorr' in f or not f.endswith('.fits'):
            return False
        if taper == 'Taper' and 'NoTaper' in f:
            return False
        if taper == 'NoTaper' and 'NoTaper' not in f:
            return False
        if rtok == 'Robust-0':
            return 'Robust-0-' in f
        return 'Robust-0.75' in f
    for f in files:
        if base(f) and f.endswith('_dewarped.fits'):
            return os.path.join(d, f)
    for f in files:
        if base(f) and 'dewarped' not in f and f.endswith('-image.fits'):
            return os.path.join(d, f)
    return None


def variant_wideband(sci_dir, band, stokes, variant):
    """= coadd.find_wideband_deep_images: a band night's deep image of ``variant`` ('Taper_R0', 'Taper_R075', 'NoTaper_R0',
    'NoTaper_R075') and Stokes, not pbcorr, dewarped preferred."""
    d = f'{sci_dir}/{band}/{stokes}/deep'
    files = _listing(d)
    if files is None:
        return None
    taper_key = 'NoTaper' if 'NoTaper' in variant else 'Taper'
    robust_key = 'Robust-0.75' if 'R075' in variant else 'Robust-0'

    def matches(f):
        if not f.endswith('.fits') or 'pbcorr' in f:
            return False
        if taper_key == 'NoTaper' and 'NoTaper' not in f:
            return False
        if taper_key == 'Taper' and 'NoTaper' in f:
            return False
        if robust_key == 'Robust-0.75' and 'Robust-0.75' not in f:
            return False
        if robust_key == 'Robust-0' and 'Robust-0.75' in f:
            return False
        return robust_key in f and taper_key in f
    for f in files:
        if matches(f) and 'dewarped' in f:
            return os.path.join(d, f)
    for f in files:
        if matches(f) and 'dewarped' not in f:
            return os.path.join(d, f)
    return None


def stokes_v_image(sci_dir, freq_str):
    """= coadd.find_stokes_v_deep_image: the night's V deep image, Taper Robust-0, not pbcorr, dewarped preferred."""
    d = f'{sci_dir}/{freq_str}MHz/V/deep' if is_subband(freq_str) else f'{sci_dir}/{freq_str}/V/deep'
    files = _listing(d)
    if files is None:
        return None

    def base(f):
        return ('Taper' in f and 'NoTaper' not in f and 'Robust-0' in f and 'Robust-0.75' not in f and 'pbcorr' not in f
                and f.endswith('.fits'))
    for f in files:
        if base(f) and 'dewarped' in f:
            return os.path.join(d, f)
    for f in files:
        if base(f) and 'dewarped' not in f:
            return os.path.join(d, f)
    return None


def survey_image(sci_dir, scope):
    """(path or None, raw) of the night's I survey image - the image its co-add flow is measured on, and so the alignment of
    every product of the night. scope: a subband's MHz number (int or str) or a band name, as the co-add passes it."""
    return survey_subband(sci_dir, scope) if is_subband(scope) else survey_wideband(sci_dir, scope)


def product_image(sci_dir, scope, product):
    """The night's image of ``product``: 'V' (V Taper Robust-0) or an I variant ('Taper_R0', 'Taper_R075', 'NoTaper_R0',
    'NoTaper_R075'), as the co-add's finder for it chooses."""
    if product == 'V':
        return stokes_v_image(sci_dir, scope)
    if product not in VARIANTS:
        raise ValueError(f'unknown Phase 2 product {product!r} (V or one of {VARIANTS})')
    if is_subband(scope):
        return variant_subband(sci_dir, scope, product)
    return variant_wideband(sci_dir, scope, 'I', product)


def dewarped(path):
    """The dewarp state of a chosen Phase 2 image: 'dewarped' in its file name - the test every finder applies to choose it."""
    return path is not None and 'dewarped' in os.path.basename(path)


def state_mismatch(sci_dir, scope, path, product):
    """B-03 same-state (Gregg 2026-10-14): None when ``path`` - the night's V or I-variant image - is in the same dewarp state
    as the night's survey image, whose flow aligns every product of the night; else why it is refused. A raw product on a
    dewarped survey image keeps its ionospheric warp (the survey's flow no longer carries it); a dewarped product on a raw
    survey image gets the raw flow's warp correction a second time. Both raw is kept: the flow was measured on that night's
    raw survey image, which carries the same warp."""
    if path is None:
        return None
    surv, _raw = survey_image(sci_dir, scope)
    if surv is None:
        return f'{product}: no survey image of the night to take the flow from'
    if dewarped(path) == dewarped(surv):
        return None
    if dewarped(surv):
        return f'raw {product} on a dewarped survey image (its warp would stay: the flow is measured on the dewarped image)'
    return f'dewarped {product} on a raw survey image (the raw image\'s flow would apply its warp correction again)'
