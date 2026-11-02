"""Content tests of the product class ``light_curve`` (the snapshot light-curve table; spec curves.md section 6).

Each test reads the FILE ONLY (values from config when ``exopipe3 verify light_curve`` calls it without context) and checks
its content against the writer, clv15/phase3_snapshot_lightcurves.py (LC), and the rulings it carries:

  structure   (LC1) HDUs FLUX, RMS, PB, SOURCES, FRAMES; FLUX/RMS float32 (NROWS, NFRAMES); PB (NROWS, NBINS); SOURCES has
                    NROWS rows and the 11 columns; FRAMES NFRAMES rows and the 6 columns; CELL, LST, DATE, SUBBAND, STOKES
                    equal the file-name tokens; NADMIT = sum(FRAMES.admitted)                        (LC:195-231)
  frames      (LC2) FRAMES.ts unique and increasing; epoch_s = seconds of ts since 1970-01-01 exactly; bin recomputed
                    exactly; flow_ts non-empty for every frame                                        (LC:129-131, :145-147)
  values      (LC3) finite RMS >= 0; every MASTER row has a finite FLUX value (target/chain rows too when
                    curves.LC3_TARGET_ROWS_NEED_FLUX - the spec's wording; production tables carry target rows inside the
                    frame rectangle but outside the imaged disc, all-NaN in every frame: 73 in 09h_2024-12-24 36MHz I,
                    4 of 105 sampled tables, never a master row); PB finite everywhere; SOURCES px, py >= margin (and
                    < N - margin when the frame size is known: ctx naxis); reported: admitted frames with no finite FLUX,
                    target/chain rows with no finite FLUX                                             (LC:115-121, :152-163)
  flux_scale  (LC4) Stokes I: NREFSCAL = sum(scale_ref), refs only on master:S rows at peak/rms >= REF_SNR with finite
                    median_flux, FLUXSCAL = median(median_flux / master_peak) (rtol SCALE_RECOMPUTE_RTOL) when NREFSCAL >=
                    MIN_SCALE_REFS else -1.0. Stokes V: no scale_ref row; FLUXSCAL and NREFSCAL equal the sibling _I_
                    table's (abs V_SCALE_TOL) - ledger 2311 (M30). Every V table written before clv15 98595b3 fails here.
  rows        (LC5) kinds master:<code> / target:<sample tag> (/ chain:<rung> only with CHAIN_RUN_DIR); >= 1 master row;
                    master ids M_<LST>_<sb>_<Gaus_id:05d>, unique, each Gaussian in the Master catalogue with the same RA,
                    DEC, Peak_flux (float32), Isl_rms (float32) and S_Code; with ctx frame_header (the first frame's
                    header, which the step passes): the master rows are EXACTLY the catalogue's finite-position Gaussians
                    inside that frame with the extractor's margin - every parent in the field has a row (regeneration test
                    T2, exopipe_ops snapshot_lightcurves_regen_test.py:59-76); reported: duplicate target ids (spec Q5)
                    (LC:65-96, :120-132; ledger 1413)
  parent_catalogue  MCATSHA (clv15 e0785b7) = sha256 of the Master catalogue of the rows (regeneration test T1, :57-58); a
                    table without the card passes on its rows alone, unless ctx require_mcatsha (the step: the current
                    extractor always writes it)
  same_as_previous  only with ctx previous (the table a --supersede replaces): at the frames both tables hold, the admitted
                    flags are equal and FLUX and RMS are bit-identical (NaN-aware) at every row whose (ra, dec) is
                    bit-identical in it (regeneration tests T3 and T4, :78-99): a re-extraction from the same frames and
                    flow fields reproduces the published values; a legitimate change needs --waive with Gregg's ruling.
                    The frame list itself may change (e.g. a table of every discovered frame re-extracted with
                    --search-frames): dropped / added frames are reported, no frame in common fails
  integrations      at most one frame per integration - by each frame's DATE-OBS as the search recorded it (the admission
                    table; ctx admission), its stamp only where no DATE-OBS is recorded - and at most MAX_FRAMES_PER_CELL
                    frames (a table holding two Phase-1 runs' frames of one integration fails)
  provenance  (LC6) PRODUCT, BUNIT, PBCORR False, PBMODEL 'model beam only', STAMP, NBINS, PBLIB, FLOWTREE, RUNDIR as
                    configured; MASTCAT, CODEVER, EXTRACT present                                    (LC:197-210; ledger 1231)
  frames also checks FRAMES.admitted against the search's admission table (a ts absent from it is admitted, LC:105-109,
                    :163) when that table exists (ctx admission; default paths.SNAPSHOT_FLEET_OUT/<cell>/admission_<sb>.csv;
                    regeneration test T3 without an old table, :103-107)
  search_frames     only for a table extracted with --search-frames (ctx search_frames, the step's per-pair truth; default
                    curves.SEARCH_FRAMES, False = production's plain call, every discovered frame): FRAMES.ts is EXACTLY
                    the search's own frame list of (subband, Stokes) - curves.SEARCH_FRAME_LIST under the cell's search
                    outputs, its 'ts' column - no frame twice, none added, none missing (clv15 55469ba --search-frames; the
                    2026-11-01 regeneration test LC-REGEN-TEST (b), exopipe_ops snapshot_lc_regen_20261101.py). A frame the
                    search dropped (12h 2024-12-28: 14 / 26 frames of 27 / 69 MHz I that did not flow) fails; so does a
                    table whose search list cannot be read (nothing to show it is the search's) - except a night-refused
                    pair (night-refusal record, no list), extracted with the plain call (B-70) and not compared

Context keys (all optional): naxis=(NAXIS1, NAXIS2) of the frames; frame_header={card: value} of the first frame;
sibling_i=path of the Stokes I table (default: the _I_ file beside a _V_ file); master_catalog=path (default from
curves.MASTER_CATALOG_DIR); flowtree=path (default paths.SNAPSHOT_FLOW_ROOT); chain_rows=bool (default:
curves.CHAIN_RUN_DIR set); admission=path ('' = not compared); require_mcatsha=bool; previous=path; search_dir=the cell's
search output directory holding SEARCH_FRAME_LIST (default paths.SNAPSHOT_FLEET_OUT/<cell>); search_frames=bool (the table
was extracted with --search-frames; default curves.SEARCH_FRAMES).
The unit checks of the step (LC7 curves.complete, LC8 curves.no_excluded_products) live in steps/curves.py.
"""
import datetime
import os
import re

from exopipe.phase3.core.validate import content_test

CLS = 'light_curve'
NAME_RE = re.compile(r'^(?P<lst>\d\dh)_(?P<date>\d{4}-\d{2}-\d{2})_(?P<sb>\d+MHz)_(?P<st>[IV])_lightcurves\.fits$')
SOURCE_COLS = ('id', 'kind', 'name', 'ra', 'dec', 'px', 'py', 'master_peak', 'master_rms', 'median_flux', 'scale_ref')
FRAME_COLS = ('ts', 'epoch_s', 'admitted', 'bin', 'flow_ts', 'flow_exact')
_EPOCH = datetime.datetime(1970, 1, 1)
_CACHE = {}
_CAT_CACHE = {}


def _C():
    from exopipe.phase3.config import CFG
    return CFG.curves


def _flow_root():
    from exopipe.phase3.config import CFG
    return CFG.paths.SNAPSHOT_FLOW_ROOT


def _s(x):
    return x.decode() if isinstance(x, bytes) else str(x)


def _load(path):
    """Everything the tests need from one table, read once per (path, mtime, size): header, HDU names, array shapes and
    dtypes, SOURCES and FRAMES columns, and derived FLUX/RMS/PB statistics."""
    import numpy as np
    from astropy.io import fits
    st = os.stat(path)
    key = (os.path.abspath(path), st.st_mtime_ns, st.st_size)
    if key in _CACHE:
        return _CACHE[key]
    t = {'path': path, 'name': os.path.basename(path)}
    with fits.open(path, memmap=False) as h:
        t['header'] = dict(h[0].header)
        t['hdus'] = [x.name for x in h]
        for nm in ('FLUX', 'RMS', 'PB'):
            if nm in t['hdus']:
                d = h[nm].data
                t[nm] = {'shape': None if d is None else tuple(d.shape), 'dtype': None if d is None else d.dtype}
                if d is not None:
                    d = np.asarray(d, dtype=np.float32)
                    if nm == 'FLUX':
                        fin = np.isfinite(d)
                        t['row_has_flux'] = fin.any(axis=1)
                        t['frame_has_flux'] = fin.any(axis=0)
                    elif nm == 'RMS':
                        fin = np.isfinite(d)
                        t['rms_negative'] = int((d[fin] < 0).sum())
                        t['rms_finite'] = int(fin.sum())
                    else:
                        t['pb_nonfinite'] = int((~np.isfinite(d)).sum())
        for nm, cols in (('SOURCES', SOURCE_COLS), ('FRAMES', FRAME_COLS)):
            if nm in t['hdus']:
                data = h[nm].data
                t[nm] = {'names': tuple(data.columns.names), 'nrows': len(data),
                         'cols': {c: np.array(data[c]) for c in data.columns.names}}
    if len(_CACHE) >= 4:
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[key] = t
    return t


def _tokens(t):
    m = NAME_RE.match(t['name'])
    if not m:
        raise ValueError(f"file name {t['name']!r} is not <LST>_<YYYY-MM-DD>_<sb>_<I|V>_lightcurves.fits")
    return m.groupdict()


def _hdr(t, k, default=None):
    v = t['header'].get(k, default)
    return v.strip() if isinstance(v, str) else v


@content_test(CLS, 'structure')
def structure(path, **ctx):
    import numpy as np
    t = _load(path)
    tok = _tokens(t)
    C = _C()
    bad = []
    for nm in ('FLUX', 'RMS', 'PB', 'SOURCES', 'FRAMES'):
        if nm not in t['hdus']:
            bad.append(f'HDU {nm} missing')
    if bad:
        return False, None, '; '.join(bad)
    nrows, nfr = _hdr(t, 'NROWS'), _hdr(t, 'NFRAMES')
    for nm in ('FLUX', 'RMS'):
        if t[nm]['shape'] != (nrows, nfr):
            bad.append(f"{nm} shape {t[nm]['shape']} != (NROWS {nrows}, NFRAMES {nfr})")
        if t[nm]['dtype'] is None or np.dtype(t[nm]['dtype']).kind != 'f' or np.dtype(t[nm]['dtype']).itemsize != 4:
            bad.append(f"{nm} dtype {t[nm]['dtype']} is not float32")
    nb = _hdr(t, 'NBINS')
    if t['PB']['shape'] != (nrows, C.NBINS) or nb != C.NBINS:
        bad.append(f"PB shape {t['PB']['shape']} != (NROWS {nrows}, NBINS {C.NBINS}) (header NBINS {nb})")
    if t['SOURCES']['names'] != SOURCE_COLS or t['SOURCES']['nrows'] != nrows:
        bad.append(f"SOURCES {t['SOURCES']['nrows']} rows x {t['SOURCES']['names']} != {nrows} x {SOURCE_COLS}")
    if t['FRAMES']['names'] != FRAME_COLS or t['FRAMES']['nrows'] != nfr:
        bad.append(f"FRAMES {t['FRAMES']['nrows']} rows x {t['FRAMES']['names']} != {nfr} x {FRAME_COLS}")
    want = {'CELL': f"{tok['lst']}_{tok['date']}", 'LST': tok['lst'], 'DATE': tok['date'], 'SUBBAND': tok['sb'],
            'STOKES': tok['st']}
    for k, v in want.items():
        if _hdr(t, k) != v:
            bad.append(f'{k} {_hdr(t, k)!r} != file name {v!r}')
    if 'admitted' in t['FRAMES']['cols']:
        nad = int(np.sum(t['FRAMES']['cols']['admitted'].astype(bool)))
        if _hdr(t, 'NADMIT') != nad:
            bad.append(f"NADMIT {_hdr(t, 'NADMIT')} != sum(FRAMES.admitted) {nad}")
    if bad:
        return False, None, '; '.join(bad)
    return True, f'{nrows}x{nfr}', f'{nrows} rows x {nfr} frames, {nb} PB bins'


def _admission_flags(path):
    """{ts: admitted} of a search admission table as the extractor reads it (LC:105-109)."""
    import csv
    with open(path) as fh:
        return {r['ts']: r.get('admitted', '1') == '1' for r in csv.DictReader(fh)}


@content_test(CLS, 'frames')
def frames(path, admission=None, **ctx):
    import numpy as np
    t = _load(path)
    C = _C()
    f = t['FRAMES']['cols']
    ts = [_s(x).strip() for x in f['ts']]
    bad = []
    if len(set(ts)) != len(ts):
        bad.append(f'{len(ts) - len(set(ts))} duplicate ts')
    if any(a >= b for a, b in zip(ts, ts[1:])):
        bad.append('FRAMES.ts not increasing')
    try:
        secs = np.array([(datetime.datetime.strptime(x[:15], '%Y%m%d_%H%M%S') - _EPOCH).total_seconds() for x in ts])
    except ValueError as e:
        return False, None, f'unparseable ts: {e}'
    if len(secs) == 0:
        return False, 0, 'no frames'
    ep = np.asarray(f['epoch_s'], float)
    if not np.array_equal(ep, secs):
        bad.append(f'epoch_s != seconds of ts since 1970 in {int(np.sum(ep != secs))} frames')
    t0, span = secs.min(), max(secs.max() - secs.min(), 1.0)
    bins = np.minimum(np.floor((secs - t0) / span * C.NBINS).astype(int), C.NBINS - 1)
    if not np.array_equal(np.asarray(f['bin'], int), bins):
        bad.append(f'bin differs from the recomputed 10-minute bin in {int(np.sum(np.asarray(f["bin"], int) != bins))} '
                   f'frames')
    nflow = sum(1 for x in f['flow_ts'] if not _s(x).strip())
    if nflow:
        bad.append(f'{nflow} frames without flow_ts')
    nad = int(np.sum(f['admitted'].astype(bool)))
    fex = float(np.mean(f['flow_exact'].astype(bool)))
    rep = f'{len(ts)} frames, {nad} admitted, flow_exact fraction {fex:.3f}'
    if admission is None:
        from exopipe.phase3.config import CFG
        tok = _tokens(t)
        admission = os.path.join(CFG.paths.SNAPSHOT_FLEET_OUT, f"{tok['lst']}_{tok['date']}", f"admission_{tok['sb']}.csv")
    adm = _admission_flags(admission) if admission and os.path.exists(admission) else {}
    if adm:                                   # T3 (regen_test.py:103-107): as the extractor reads it, absent = admitted
        want = [adm.get(x, True) for x in ts]
        got = [bool(x) for x in f['admitted']]
        if got != want:
            bad.append(f'FRAMES.admitted differs from the admission table {admission} in '
                       f'{sum(1 for a, b in zip(got, want) if a != b)} frames')
        rep += f'; admitted flags = {os.path.basename(admission)}'
    else:
        rep += '; no admission table to compare'
    if bad:
        return False, len(ts), '; '.join(bad) + ' | ' + rep
    return True, len(ts), rep


def search_frame_list(search_dir, sb, st):
    """(path, [ts in order]) of the search's own 10-s frame list of (subband, Stokes) - curves.SEARCH_FRAME_LIST under the
    cell's search output directory, read as the extractor reads it with --search-frames (clv15 55469ba: its 'ts' column);
    the list is None when the file cannot be read."""
    import csv
    p = os.path.join(search_dir, _C().SEARCH_FRAME_LIST.format(sb=sb, st=st))
    try:
        with open(p, newline='') as fh:
            return p, [(r.get('ts') or '').strip() for r in csv.DictReader(fh)]
    except OSError:
        return p, None


@content_test(CLS, 'search_frames')
def search_frames(path, search_dir=None, search_frames=None, **ctx):
    """Only for a table extracted with --search-frames (ctx search_frames, the step's per-pair truth; without it
    curves.SEARCH_FRAMES): FRAMES.ts holds exactly the frames of the search's own list, each once (module docstring;
    LC-REGEN-TEST (b) compares the sets and counts repeats; the order is the frames test's: increasing ts, as the extractor
    sorts them). A table of the plain call (production's fleet and backfill, the default) is not compared: it holds every
    discovered frame, also those the search dropped. A night-refused pair (night-refusal record, no list) is extracted with
    the plain call under curves.SEARCH_FRAMES too (B-70) and is not compared either."""
    t = _load(path)
    tok = _tokens(t)
    applies = bool(_C().SEARCH_FRAMES if search_frames is None else search_frames)
    ts = [_s(x).strip() for x in t['FRAMES']['cols']['ts']]
    if not applies:
        return True, 'n/a', ('not compared: extracted with the plain call (every discovered frame; curves.SEARCH_FRAMES / '
                             'ctx search_frames False)')
    if not search_dir:
        from exopipe.phase3.config import CFG
        search_dir = os.path.join(CFG.paths.SNAPSHOT_FLEET_OUT, f"{tok['lst']}_{tok['date']}")
    lp, want = search_frame_list(search_dir, tok['sb'], tok['st'])
    if want is None:
        nr = os.path.join(search_dir, f"nightrefused_{tok['sb']}_{tok['st']}_median.json")
        if os.path.exists(nr):
            return True, 'n/a', (f'not compared: the search refused the night ({os.path.basename(nr)}, no frame list) - the '
                                 f'plain call, as today (B-70)')
        return False, len(ts), f'no search frame list {lp} to compare with: the frames cannot be shown to be the search\'s'
    extra, missing = sorted(set(ts) - set(want)), sorted(set(want) - set(ts))
    dup = len(ts) - len(set(ts))
    if not (extra or missing or dup):
        return True, len(ts), f'{len(ts)} frames = the search\'s own list {os.path.basename(lp)} ({len(want)} rows)'
    return False, len(ts), (f'{len(ts)} frames != the search\'s {len(set(want))} ({os.path.basename(lp)}): {len(extra)} not in '
                            f'the search\'s list{" e.g. " + extra[0] if extra else ""}, {len(missing)} of the list missing'
                            f'{" e.g. " + missing[0] if missing else ""}, {dup} repeated')


def _dateobs_unix(do):
    """Unix seconds of an ISO DATE-OBS ('2024-12-28T13:44:42.9'); None when it does not parse."""
    try:
        whole, _, frac = (do or '').strip().partition('.')
        t = datetime.datetime.strptime(whole, '%Y-%m-%dT%H:%M:%S')
        return (t - _EPOCH).total_seconds() + (float('0.' + frac) if frac.isdigit() else 0.0)
    except ValueError:
        return None


@content_test(CLS, 'integrations')
def integrations(path, admission=None, **ctx):
    """At most one frame per integration (the reviewer's gap check, 2026-11-02), on the frames' OBSERVATION times: each frame's
    DATE-OBS as the search recorded it (the admission table: ctx admission, default paths.SNAPSHOT_FLEET_OUT/<cell>/
    admission_<sb>.csv) gives its integration - the second of its midpoint, DATE-OBS + curves.INTEGRATION_S / 2 (clv15
    phase3_snapshot_search.integration_label) - and no two frames may share one. The file-name stamps alone cannot decide it:
    two Phase-1 runs name frames by the integration's start or its midpoint, so adjacent integrations of a cell holding both
    runs' frames can be stamped 5 s apart (12h 2024-12-28 27 MHz I: DATE-OBS 13:44:42.9 / 13:44:52.9, stamps 134447 / 134452)
    while one integration's two copies are 0-5 s apart (11h 2024-12-24 46 MHz I: 314 integrations twice). A frame with no
    recorded DATE-OBS (absent from the admission table, or blank) is placed by its stamp only where that cannot be wrong:
    two stamps less than INTEGRATION_S / 2 - 1 s apart are one integration. And at most curves.MAX_FRAMES_PER_CELL frames
    (one LST hour)."""
    import csv
    import numpy as np
    t = _load(path)
    C = _C()
    ts = [_s(x).strip() for x in t['FRAMES']['cols']['ts']]
    n = len(ts)
    if admission is None:
        from exopipe.phase3.config import CFG
        tok = _tokens(t)
        admission = os.path.join(CFG.paths.SNAPSHOT_FLEET_OUT, f"{tok['lst']}_{tok['date']}", f"admission_{tok['sb']}.csv")
    obs = {}
    if admission and os.path.exists(admission):
        with open(admission, newline='') as fh:
            for r in csv.DictReader(fh):
                u = _dateobs_unix(r.get('date_obs'))
                if u is not None:
                    obs[(r.get('ts') or '').strip()] = u
    half = float(C.INTEGRATION_S) / 2.0
    bad, by_int, unplaced = [], {}, []
    for x in ts:
        if x in obs:
            by_int.setdefault(int(np.floor(obs[x] + half)), []).append(x)
        else:
            unplaced.append(x)
    twice = {k: v for k, v in by_int.items() if len(v) > 1}
    if twice:
        k, v = next(iter(sorted(twice.items())))
        bad.append(f'{len(twice)} integrations held by more than one frame (by their recorded DATE-OBS; e.g. '
                   f'{" and ".join(v)} are one integration)')
    if unplaced:                       # stamps only: closer than any two integrations' stamps can come
        sec = {x: (datetime.datetime.strptime(x[:15], '%Y%m%d_%H%M%S') - _EPOCH).total_seconds() for x in ts}
        lim = half - 1.0
        allsec = np.sort(np.array([sec[x] for x in ts]))
        close = sum(1 for x in unplaced if np.sum(np.abs(allsec - sec[x]) < lim) > 1)
        if close:
            bad.append(f'{close} frames without a recorded DATE-OBS have another frame stamped less than {lim:.1f} s away '
                       f'(one integration)')
    if n > int(C.MAX_FRAMES_PER_CELL):
        bad.append(f'{n} frames > {C.MAX_FRAMES_PER_CELL}, more than one LST hour holds')
    rep = (f'{n} frames in {len(by_int) + len(unplaced)} integrations ({len(by_int)} by recorded DATE-OBS, {len(unplaced)} by '
           f'stamp; integration {C.INTEGRATION_S:.2f} s)')
    if bad:
        return False, n, '; '.join(bad) + ' | ' + rep
    return True, n, rep


@content_test(CLS, 'values')
def values(path, naxis=None, **ctx):
    import numpy as np
    t = _load(path)
    C = _C()
    bad = []
    if t.get('rms_negative'):
        bad.append(f"{t['rms_negative']} negative RMS values")
    s = t['SOURCES']['cols']
    kinds = np.array([_s(k).strip().partition(':')[0] for k in s['kind']])
    noflux = ~t['row_has_flux']
    n_master = int(np.sum(noflux & (kinds == 'master')))
    n_other = int(np.sum(noflux & (kinds != 'master')))
    if n_master:
        bad.append(f'{n_master} master rows without any finite FLUX value')
    if n_other and C.LC3_TARGET_ROWS_NEED_FLUX:
        bad.append(f'{n_other} target/chain rows without any finite FLUX value (curves.LC3_TARGET_ROWS_NEED_FLUX)')
    nflux = n_master + n_other
    if t.get('pb_nonfinite'):
        bad.append(f"{t['pb_nonfinite']} non-finite PB values")
    px, py = np.asarray(s['px'], float), np.asarray(s['py'], float)
    m = C.INFIELD_MARGIN_PX
    nout = int(np.sum(~(np.isfinite(px) & np.isfinite(py) & (px >= m) & (py >= m))))
    edge = 'lower edge only (frame size unknown)'
    if naxis and int(naxis[0]) > 0 and int(naxis[1]) > 0:
        nx, ny = int(naxis[0]), int(naxis[1])
        nout += int(np.sum((px >= nx - m) | (py >= ny - m)))
        edge = f'frame {nx}x{ny}'
    if nout:
        bad.append(f'{nout} SOURCES px/py outside the in-field margin {m} px ({edge})')
    adm = t['FRAMES']['cols']['admitted'].astype(bool)
    blank_adm = int(np.sum(adm & ~t['frame_has_flux']))
    rep = (f'admitted frames with no finite FLUX: {blank_adm}; target/chain rows with no finite FLUX (outside the imaged '
           f'disc): {n_other}; in-field check: {edge}')
    if bad:
        return False, nflux, '; '.join(bad) + ' | ' + rep
    return True, blank_adm, rep


def _sibling_i(path):
    d, n = os.path.split(path)
    return os.path.join(d, re.sub(r'_V_lightcurves\.fits$', '_I_lightcurves.fits', n))


@content_test(CLS, 'flux_scale')
def flux_scale(path, sibling_i=None, **ctx):
    import numpy as np
    from astropy.io import fits
    t = _load(path)
    C = _C()
    tok = _tokens(t)
    s = t['SOURCES']['cols']
    ref = s['scale_ref'].astype(bool)
    fs, nref = float(_hdr(t, 'FLUXSCAL')), int(_hdr(t, 'NREFSCAL'))
    if tok['st'] == 'V':
        if ref.any():
            return False, fs, f'{int(ref.sum())} scale_ref rows in a Stokes V table (V carries the I scale)'
        ip = sibling_i or _sibling_i(path)
        if not os.path.exists(ip):
            return False, fs, f'no sibling Stokes I table {ip} to compare the V scale with (V carries the I scale)'
        hI = fits.getheader(ip)
        fi, ni = float(hI['FLUXSCAL']), int(hI['NREFSCAL'])
        if abs(fs - fi) > C.V_SCALE_TOL or nref != ni:
            return False, fs, (f'FLUXSCAL {fs} / NREFSCAL {nref} != Stokes I {fi} / {ni} (ledger 2311 M30: V carries the '
                               f'I scale; V tables written before clv15 98595b3 carry -1.0 / 0)')
        return True, fs, f'FLUXSCAL {fs} and NREFSCAL {nref} = Stokes I table'
    kind = np.array([_s(k).strip() for k in s['kind']])
    mp = np.asarray(s['master_peak'], float)
    mr = np.asarray(s['master_rms'], float)
    med = np.asarray(s['median_flux'], float)
    with np.errstate(divide='ignore', invalid='ignore'):
        snr = np.where(mr > 0, mp / mr, np.nan)
    elig = (kind == C.SCALE_REF_KIND) & np.isfinite(mp) & np.isfinite(mr) & (mr > 0) & np.isfinite(med)
    tol = 1e-5 * C.REF_SNR                     # float32 storage of peak and rms (the writer compares float64 values)
    strict_in = elig & (snr >= C.REF_SNR + tol)
    strict_out = ~elig | ~(snr >= C.REF_SNR - tol)
    bad = []
    if nref != int(ref.sum()):
        bad.append(f'NREFSCAL {nref} != sum(scale_ref) {int(ref.sum())}')
    if np.any(ref & strict_out):
        bad.append(f'{int(np.sum(ref & strict_out))} scale_ref rows not {C.SCALE_REF_KIND} at peak/rms >= {C.REF_SNR} '
                   f'with finite median_flux')
    if np.any(strict_in & ~ref):
        bad.append(f'{int(np.sum(strict_in & ~ref))} eligible rows (kind {C.SCALE_REF_KIND}, peak/rms >= {C.REF_SNR}, '
                   f'finite median_flux) not marked scale_ref')
    if int(ref.sum()) >= C.MIN_SCALE_REFS:
        want = float(np.median(med[ref] / mp[ref]))
        if not (np.isfinite(fs) and abs(fs - want) <= C.SCALE_RECOMPUTE_RTOL * abs(want)):
            bad.append(f'FLUXSCAL {fs} != median(median_flux / master_peak) {want} over {int(ref.sum())} refs')
        detail = f'FLUXSCAL {fs:.6g} on {int(ref.sum())} refs (recomputed {want:.6g})'
    else:
        if fs != -1.0:
            bad.append(f'FLUXSCAL {fs} with {int(ref.sum())} < {C.MIN_SCALE_REFS} refs (expected -1.0)')
        detail = f'FLUXSCAL -1.0: {int(ref.sum())} refs < {C.MIN_SCALE_REFS}'
    if bad:
        return False, fs, '; '.join(bad)
    return True, fs, detail


def _catalog(path):
    """PyBDSF Gaussian list as LC reads it (RA.read_gaul_full: the '#' line naming RA is the header; rows whose numeric
    fields do not parse are skipped): {Gaus_id (or the row index when absent): (ra % 360, dec, peak, rms, code)}."""
    st = os.stat(path)
    key = (os.path.abspath(path), st.st_mtime_ns, st.st_size)
    if key in _CAT_CACHE:
        return _CAT_CACHE[key]
    hdr, out, i = None, {}, 0
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            if line.startswith('#'):
                fl = [x.strip() for x in line.lstrip('#').split(',')]
                if hdr is None and 'RA' in fl:
                    hdr = fl
                continue
            if not hdr:
                continue
            rec = dict(zip(hdr, [x.strip() for x in line.split(',')]))
            try:
                ra, dec, peak = float(rec['RA']) % 360.0, float(rec['DEC']), float(rec['Peak_flux'])
                rms = float(rec.get('Isl_rms', 'nan'))
                gid = int(float(rec['Gaus_id'])) if rec.get('Gaus_id', '') not in ('', None) else None
            except Exception:
                continue
            out[gid if gid is not None else i] = (ra, dec, peak, rms, rec.get('S_Code', ''))
            i += 1
    _CAT_CACHE.clear()
    _CAT_CACHE[key] = out
    return out


def _infield_master_ids(cat, frame_header, lst, sb, margin):
    """The master row ids the extractor writes for catalogue ``cat`` (``_catalog``) in a frame with ``frame_header``:
    finite-position Gaussians whose pixel (the header's celestial WCS, origin 0) lies in [margin, N - margin) on both axes
    (LC:65-81, :120-129; regen_test.py:59-71)."""
    import warnings
    import numpy as np
    from astropy.io import fits
    from astropy.wcs import WCS
    h = fits.Header()
    for k, v in frame_header.items():
        try:
            h[k] = v
        except (ValueError, KeyError, TypeError):
            pass
    keys = [k for k, g in cat.items() if np.isfinite(g[0]) and np.isfinite(g[1])]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        w = WCS(h).celestial
        px, py = w.all_world2pix(np.array([cat[k][0] for k in keys], float), np.array([cat[k][1] for k in keys], float), 0)
    px, py = np.asarray(px, float), np.asarray(py, float)
    nx, ny = int(h['NAXIS1']), int(h['NAXIS2'])
    inf = np.isfinite(px) & np.isfinite(py) & (px >= margin) & (px < nx - margin) & (py >= margin) & (py < ny - margin)
    return {'M_%s_%s_%05d' % (lst, sb, k) for k, ok in zip(keys, inf) if ok}


@content_test(CLS, 'rows')
def rows(path, master_catalog=None, chain_rows=None, frame_header=None, **ctx):
    import numpy as np
    t = _load(path)
    C = _C()
    tok = _tokens(t)
    s = t['SOURCES']['cols']
    kinds = [_s(k).strip() for k in s['kind']]
    ids = [_s(x).strip() for x in s['id']]
    tags = {tag for tag, _fn in C.SAMPLES}
    chain_ok = bool(C.CHAIN_RUN_DIR) if chain_rows is None else bool(chain_rows)
    allowed = set(C.ROW_KINDS) | ({'chain'} if chain_ok else set())
    bad = []
    nbadkind = 0
    for k in kinds:
        fam, _, sub = k.partition(':')
        if fam not in allowed or (fam == 'target' and sub not in tags) or not sub:
            nbadkind += 1
    if nbadkind:
        bad.append(f'{nbadkind} rows of a kind outside {sorted(allowed)} (targets: {sorted(tags)})')
    mrows = [j for j, k in enumerate(kinds) if k.startswith('master:')]
    if not mrows:
        bad.append('no master row')
    mids = [ids[j] for j in mrows]
    if len(set(mids)) != len(mids):
        bad.append(f'{len(mids) - len(set(mids))} duplicate master ids')
    catp = master_catalog or os.path.join(C.MASTER_CATALOG_DIR, f"master_R075_{tok['lst']}_{tok['sb']}.gaul.csv")
    nmis = 0
    if mrows:
        if not os.path.exists(catp):
            bad.append(f'Master catalogue {catp} absent')
        else:
            cat = _catalog(catp)
            pre = f"M_{tok['lst']}_{tok['sb']}_"
            first = ''
            for j in mrows:
                g = None
                if ids[j].startswith(pre):
                    try:
                        gid = int(ids[j][len(pre):])
                        g = cat.get(gid)
                        if g is not None and ids[j] != pre + '%05d' % gid:
                            g = None
                    except ValueError:
                        g = None
                ok = (g is not None and float(s['ra'][j]) == g[0] and float(s['dec'][j]) == g[1]
                      and np.float32(s['master_peak'][j]) == np.float32(g[2])
                      and (np.float32(s['master_rms'][j]) == np.float32(g[3])
                           or (np.isnan(s['master_rms'][j]) and np.isnan(g[3])))
                      and kinds[j] == 'master:' + (g[4] or '?'))
                if not ok:
                    nmis += 1
                    first = first or ids[j]
            if nmis:
                bad.append(f'{nmis} master rows not matching their Gaussian in {os.path.basename(catp)} (id, RA, DEC, '
                           f'Peak_flux, Isl_rms, S_Code; first {first})')
    complete = 'in-field completeness not checked (no first-frame header in the context)'
    if frame_header and os.path.exists(catp):        # T2: every parent in the field has a row, and only those
        want = _infield_master_ids(_catalog(catp), frame_header, tok['lst'], tok['sb'], C.INFIELD_MARGIN_PX)
        got = set(mids)
        if got != want:
            miss, extra = sorted(want - got), sorted(got - want)
            bad.append(f'master rows {len(got)} != the {len(want)} finite Gaussians of {os.path.basename(catp)} inside the '
                       f'first frame (missing {len(miss)}{" e.g. " + miss[0] if miss else ""}, extra {len(extra)}'
                       f'{" e.g. " + extra[0] if extra else ""})')
        complete = f'master rows = the {len(want)} catalogue Gaussians inside the first frame'
    tids = [ids[j] for j, k in enumerate(kinds) if k.startswith('target:')]
    dup_t = len(tids) - len(set(tids))
    rep = f'{len(mrows)} master rows, {len(tids)} target rows, duplicate target ids (reported): {dup_t}; {complete}'
    if bad:
        return False, len(mrows), '; '.join(bad) + ' | ' + rep
    return True, len(mrows), rep


@content_test(CLS, 'parent_catalogue')
def parent_catalogue(path, master_catalog=None, require_mcatsha=False, **ctx):
    """MCATSHA (clv15 e0785b7, 2026-10-13) = the sha256 of the Master catalogue the rows came from: a table built on other
    parents fails (regeneration test T1). A table written before the card existed passes on its rows alone (light_curve.rows
    checks them) - unless ``require_mcatsha`` (the step passes it: the extractor it runs always writes the card, and the
    regeneration test fails a table without it)."""
    import hashlib
    t = _load(path)
    got = _hdr(t, 'MCATSHA')
    if not got:
        if require_mcatsha:
            return False, None, 'no MCATSHA card: not written by the current extractor (clv15 e0785b7 stamps it)'
        return True, None, 'no MCATSHA (a table written before clv15 e0785b7); rows checked by light_curve.rows'
    tok = _tokens(t)
    catp = master_catalog or os.path.join(_C().MASTER_CATALOG_DIR, f"master_R075_{tok['lst']}_{tok['sb']}.gaul.csv")
    if not os.path.exists(catp):
        return False, got[:12], f'Master catalogue {catp} absent'
    h = hashlib.sha256()
    with open(catp, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    want = h.hexdigest()[:64]
    return got == want, got[:12], (f'MCATSHA {got[:12]} = sha256 of {os.path.basename(catp)}' if got == want else
                                   f'MCATSHA {got[:12]} != sha256 {want[:12]} of {os.path.basename(catp)} (other parents)')


@content_test(CLS, 'same_as_previous')
def same_as_previous(path, previous=None, **ctx):
    """Only with ``previous`` (the published table a --supersede replaces; the step passes it), at the frames both tables
    hold: T3 the admitted flags are equal; T4 at every row whose (ra, dec) is bit-identical in the previous table, FLUX and
    RMS are equal exactly, NaN where NaN (exopipe_ops snapshot_lightcurves_regen_test.py:78-99, the regeneration recipe's
    install gate: a re-extraction - from the archive or a stage - reproduces the published values). The frame list may
    change under --supersede (2026-11-02 review: a table of every discovered frame re-extracted with --search-frames drops
    the frames the search dropped): the frames dropped and added are reported (the step records the counts per pair);
    no frame in common fails."""
    import numpy as np
    from astropy.io import fits
    if not previous:
        return True, None, 'no previous table (nothing replaced)'
    if not os.path.exists(previous):
        return False, None, f'previous table {previous} absent'
    with fits.open(path, memmap=False) as h, fits.open(previous, memmap=False) as o:
        fr, fo = h['FRAMES'].data, o['FRAMES'].data
        ts_n, ts_o = [_s(x).strip() for x in fr['ts']], [_s(x).strip() for x in fo['ts']]
        i_o, i_n = {t: i for i, t in enumerate(ts_o)}, {t: j for j, t in enumerate(ts_n)}
        common = [t for t in ts_n if t in i_o]
        frames = (f'frames previous {len(ts_o)}, new {len(ts_n)}, both {len(common)} (dropped {len(set(ts_o) - set(ts_n))}, '
                  f'added {len(set(ts_n) - set(ts_o))})')
        if not common:
            return False, 0, f'no frame in common with the previous table {os.path.basename(previous)} ({frames})'
        co, cn = np.array([i_o[t] for t in common]), np.array([i_n[t] for t in common])
        bad = []
        n_adm = int(np.sum(fo['admitted'].astype(bool)[co] != fr['admitted'].astype(bool)[cn]))
        if n_adm:
            bad.append(f'admitted flag differs at {n_adm} of the {len(common)} frames both hold')
        so, src = o['SOURCES'].data, h['SOURCES'].data
        key_old = {(float(a), float(b)): i for i, (a, b) in enumerate(zip(so['ra'], so['dec']))}
        pairs = [(key_old[(float(a), float(b))], j) for j, (a, b) in enumerate(zip(src['ra'], src['dec']))
                 if (float(a), float(b)) in key_old]
        if pairs:
            I = np.array([p[0] for p in pairs])
            J = np.array([p[1] for p in pairs])
            for ext in ('FLUX', 'RMS'):
                a, b = o[ext].data[I][:, co], h[ext].data[J][:, cn]
                fa, fb = np.isfinite(a), np.isfinite(b)
                if not ((fa == fb).all() and (a[fa] == b[fb]).all()):
                    bad.append(f'{ext} differs from the previous table at {len(pairs)} identical positions')
    if bad:
        return False, len(pairs), '; '.join(bad) + f' ({os.path.basename(previous)}; {frames})'
    return True, len(pairs), (f'{len(pairs)} identical-position rows equal to the previous table at the frames both hold; '
                              f'admitted flags equal; {frames}')


@content_test(CLS, 'provenance')
def provenance(path, flowtree=None, **ctx):
    t = _load(path)
    C = _C()
    want = {'PRODUCT': 'snapshot_lightcurves', 'BUNIT': 'JY/BEAM', 'PBCORR': False, 'PBMODEL': 'model beam only',
            'STAMP': C.STAMP_PX, 'NBINS': C.NBINS, 'PBLIB': C.PB_LIBRARY[:68],
            'FLOWTREE': (flowtree or _flow_root())[:68],
            'RUNDIR': (os.path.basename(C.CHAIN_RUN_DIR.rstrip('/')) if C.CHAIN_RUN_DIR else '')[:68]}
    bad = []
    for k, v in want.items():
        got = _hdr(t, k)
        if got != v or (isinstance(v, bool) and not isinstance(got, bool)):
            bad.append(f'{k} {got!r} != {v!r}')
    for k in ('MASTCAT', 'CODEVER', 'EXTRACT'):
        if not _hdr(t, k):
            bad.append(f'{k} missing or empty')
    if bad:
        return False, None, '; '.join(bad)
    return True, _hdr(t, 'CODEVER'), f"CODEVER {_hdr(t, 'CODEVER')}, EXTRACT {_hdr(t, 'EXTRACT')}"
