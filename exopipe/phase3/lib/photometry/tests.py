"""Content tests of the `exopipe3 photometry` product classes (spec photometry.md section 6), registered with
@content_test (core/validate discovers this module). Each test reads the file's CONTENT and recomputes with code of its own;
nothing here imports the in-place engines (deep_forced_*, snapshot_forced_*, phase3_snapshot_*), the rule of the production
content tests these port (test_snapshot_forced_reserve.py docstring lines 6-8).

Classes (one product each):
  photometry_frame_list   frames_07.csv / frames_09.csv            FL1 header, FL2 frames, FL3 PB by weighting, FL4 parents,
                                                                   FL5 THERMNOI, FL6 refused cells absent
  photometry_stage_list   stage_list.txt                           SL1 sorted unique absolute paths under a stage prefix, named
                                                                   by the sibling frames_07.csv
  photometry_table        out/forced_NNN.csv.gz                    T1 header, T2 entry join, T3 radio record, T4 re-measure,
                                                                   T5 floors + exclusions, T6 arithmetic
  photometry_blank_null   out/blank_NNN.csv                        B1 counts, admitted frames only
  photometry_admission    out/frames_NNN.csv                       A1 reasons, exclusions recomputed, coverage of the shard's cells
  photometry_eval         out/eval/ (deep) or eval[_wb]_<tag>/     deep E1 thresholds recomputed, E2 lists, E3 exceedance, E4 png;
                          (snapshot)                               snapshot G1-G6 grid, V1 kind + exclusions, V2 rows, Z1 zones
                                                                   recomputed, S stage 1 present and newer than its search
  photometry_snapshot_npz stage1/<cell>/forced_<smp>_<band>_<S>.npz N1 shapes, N2 1-minute blocks, N3 ring, N4 re-measure from
                                                                   the cutout, N5 names
  photometry_manifest     stage1/<cell>/forced_manifest.csv         M1 statuses, M2 completion, M3 coverage of the npz files
  photometry_cutout       <cutout root>/<smp>/<band>/<obj>/*.fits   C1 cube, C2 TIMES, C3 header vs path, C4 target inside
  photometry_zone_flags   zones_recomputed_<tag>.csv                ZF1 header, every row's file screen = its recomputed screen

A test returns (passed, value, detail). A check that needs data this host cannot read (a node-local path of another host)
says so in its detail and does not fail; every other missing input fails. Optional context (ctx) from the step: frames_csv,
shard, nshards, refused_cells, cutout_root, stage1, max_cells (limits the Z1 / S cells - real-product spot checks only).
"""
import csv
import glob
import gzip
import json
import math
import os
import re

from exopipe.phase3.core.validate import content_test

# The deep forced table's columns in production order (deep_forced_phot.py:201-207, deep_forced_radio.RADIO_COLS :73-74):
# a FORMAT contract of the product, compared with the engine source by tests/test_photometry.py.
FORCED_COLUMNS = ('sample', 'target', 'lst', 'band', 'date', 'stokes', 'kind', 'ctrl', 'ra', 'dec', 'mjd', 'elev', 'value',
                  'rms', 'snr', 'pb', 'flux_pbcorr', 'zone', 'zone_parent_jy', 'zone_sep_beams', 'ateam_deg', 'jup_deg',
                  'moon_deg', 'jup_track_night', 'fit_peak', 'fit_dx_arcmin', 'fit_dy_arcmin', 'size_beams', 'size_err_beams',
                  'axis_norm', 'gate_size', 'inj5_snr', 'inj10_snr', 'frame_rms', 'frame_bmaj_deg', 'zone_withfaintV')
RADIO_COLUMNS = ('pos_kind', 'pos_ra', 'pos_dec', 'pos_err_maj_arcsec', 'pos_err_min_arcsec', 'pos_err_pa_deg',
                 'pos_stat_maj_arcsec', 'pos_stat_min_arcsec', 'pos_sys_arcsec', 'pos_snr', 'pos_offset_arcmin', 'pos_reason')
ENTRY_COLUMNS = ('target_ra', 'target_dec')
KEY = ('sample', 'target', 'lst', 'band', 'date', 'stokes')
BLANK_COLUMNS = ('lst', 'band', 'date', 'stokes', 'n', 'ge3', 'ge4', 'ge5', 'ge6', 'ge7')       # deep_forced_phot.py:209
ADMISSION_COLUMNS = ('lst', 'band', 'date', 'stokes', 'path', 'admitted', 'reason', 'n_targets', 'n_controls')   # :211
POS_KINDS = ('gaussian', 'template', 'none')
NPZ_KEYS = ('names', 'ra', 'dec', 'ring_beams', 'ts', 'admitted', 'val10', 'rms10', 'bts', 'bn', 'val1', 'rms1')
MANIFEST_STATUS_RE = re.compile(r'^(ok|skipped: not in the current \S+ list|REFUSED: .*)$')
ZONE_FLAG_COLUMNS = ('cell', 'target', 'band', 'stokes', 'tier', 'ts', 'screen_file', 'screen_recomputed')
SNAPSHOT_GROUP = dict([(b, 'Red') for b in ('18MHz', '23MHz', '27MHz', '32MHz', '36MHz')]
                      + [(b, 'Green') for b in ('41MHz', '46MHz', '50MHz', '55MHz', '59MHz')]
                      + [(b, 'Blue') for b in ('64MHz', '69MHz', '73MHz', '78MHz', '82MHz')]
                      + [('Red', 'Red'), ('Green', 'Green'), ('Blue', 'Blue')])


def _cfg():
    from exopipe.phase3.config import CFG
    return CFG


# ---------------------------------------------------------------------------------------------------------------------
# shared helpers (also used by steps_*.py)
# ---------------------------------------------------------------------------------------------------------------------

def read_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        rd = csv.DictReader(f)
        return list(rd.fieldnames or []), list(rd)


def safe_name(nm):
    """The cutout writer's object directory name (phase3_snapshot_cutouts.safe_name, :73-74), written out here."""
    return ''.join(ch if (ch.isalnum() or ch in '+-._') else '_' for ch in nm.strip()).strip('_')


def sample_targets(tags=None):
    """[(tag, name, ra, dec)] of the measured sample lists, read the way the cutout writer reads them
    (phase3_snapshot_cutouts._targets, :40-56: common_name|name|Name; ra_current|ra|RA; rows whose RA/Dec do not parse
    are left out)."""
    c = _cfg()
    out = []
    for tag in (tags or c.photometry.SAMPLES_MEASURED):
        p = os.path.join(c.paths.CLV15_ROOT, c.photometry.SAMPLE_FILES[tag])
        with open(p, encoding=c.photometry.SAMPLE_ENCODING) as f:
            for r in csv.DictReader(f):
                ra = r.get('ra_current') or r.get('ra') or r.get('RA')
                dec = r.get('dec_current') or r.get('dec') or r.get('DEC')
                nm = (r.get('common_name') or r.get('name') or r.get('Name') or '').strip()
                try:
                    out.append((tag, nm, float(ra), float(dec)))
                except (TypeError, ValueError):
                    continue
    return out


def current_safe_names():
    return {(t, safe_name(n)) for t, n, _, _ in sample_targets()}


def node_local_elsewhere(path):
    """The host whose node-local root holds ``path`` when that host is not this one (site NODE_LOCAL_ROOTS / SCRATCH_ROOT),
    else None. Such a path cannot be read here: a content check on it says 'not checkable here'."""
    from exopipe.phase3.core import env
    c = _cfg()
    me = env.host()
    roots = {}
    for h, rr in (c.site.NODE_LOCAL_ROOTS or {}).items():
        roots.setdefault(h, []).extend(rr)
    for h, r in (c.site.SCRATCH_ROOT or {}).items():
        roots.setdefault(h, []).append(r)
    mine = [r for r in roots.get(me, [])]
    p = os.path.abspath(path)
    if any(p == r or p.startswith(r.rstrip('/') + '/') for r in mine):
        return None
    for h, rr in roots.items():
        if h != me and any(p == r or p.startswith(r.rstrip('/') + '/') for r in rr):
            return h
    for pre in c.photometry_frames.STAGE_PREFIX_ROOTS:
        if (p == pre or p.startswith(pre.rstrip('/') + '/')) and not os.path.exists(pre):
            return 'a host with %s' % pre
    return None


def _exists_or_elsewhere(path):
    """'ok' | 'elsewhere' | 'missing'."""
    if path and os.path.exists(path):
        return 'ok'
    if path and node_local_elsewhere(path):
        return 'elsewhere'
    return 'missing'


_CACHE = {}


def _cached(kind, path, fn):
    st = os.stat(path)
    key = (kind, os.path.abspath(path), st.st_mtime_ns, st.st_size)
    if key not in _CACHE:
        if len(_CACHE) > 64:
            _CACHE.clear()
        _CACHE[key] = fn()
    return _CACHE[key]


def _excl():
    from exopipe.phase3.core import exclusions
    return exclusions.load()


def _sigfig_tol(printed):
    """Half a unit of the last printed digit of a '%.5g' value (relative 5e-5), plus a hair."""
    try:
        v = abs(float(printed))
    except ValueError:
        return float('inf')
    if v == 0:
        return 1e-12
    return 0.5 * 10 ** (math.floor(math.log10(v)) - 4) * 1.001 + 1e-15


def _fmt(n):
    return ', '.join(n[:5]) + (' ...' if len(n) > 5 else '')


# ---------------------------------------------------------------------------------------------------------------------
# photometry_frame_list (frames_07.csv, frames_09.csv)
# ---------------------------------------------------------------------------------------------------------------------

def _stage_map(path):
    """{staged path: original path} when stage_list.txt sits beside a staged frame list (frames_09.csv)."""
    sl = os.path.join(os.path.dirname(os.path.abspath(path)), _cfg().photometry_frames.STAGE_LIST_FILE)
    if not os.path.exists(sl):
        return {}
    orig = [l.strip() for l in open(sl) if l.strip()]
    return {'__originals__': orig}


def _unstage(p, smap):
    for o in smap.get('__originals__', ()):
        if p.endswith(o) and p != o:
            return o
    return p


@content_test('photometry_frame_list', 'FL1_header')
def fl1_header(path, **ctx):
    cols, rows = read_csv(path)
    want = list(_cfg().photometry_frames.FRAME_LIST_COLUMNS)
    if cols != want:
        return False, len(rows), 'columns %s != %s' % (cols, want)
    if not rows:
        return False, 0, 'no frame row'
    return True, len(rows), '%d frames over %d cells' % (len(rows), len({(r['lst'], r['band']) for r in rows}))


@content_test('photometry_frame_list', 'FL2_frames')
def fl2_frames(path, **ctx):
    f = _cfg().photometry_frames
    _, rows = read_csv(path)
    rx = re.compile(f.FRAME_NAME_RE)
    bad, seen, n_else = [], set(), 0
    for r in rows:
        m = rx.match(os.path.basename(r['path']))
        if not m or m.group(1) != r['date'] or m.group(2) != r['stokes'] or m.group(3) != r['band']:
            bad.append('%s: name does not match its row' % os.path.basename(r['path']))
            continue
        k = (r['lst'], r['band'], r['date'], r['stokes'])
        if k in seen:
            bad.append('duplicate %s' % '_'.join(k))
        seen.add(k)
    # existence of an evenly spaced sample (the measurement reads every frame and fails on a missing one; a stat of every
    # frame costs minutes on a loaded Lustre metadata server)
    step = max(1, len(rows) // max(1, f.FL2_EXISTENCE_ROWS))
    sample = rows[::step][:f.FL2_EXISTENCE_ROWS]
    smap = _stage_map(path)
    for r in sample:
        st = _exists_or_elsewhere(r['path'])
        if st == 'missing' and _unstage(r['path'], smap) != r['path']:
            st = _exists_or_elsewhere(_unstage(r['path'], smap))      # a staged copy on the measuring host: its original
        if st == 'missing':
            bad.append('missing %s' % r['path'])
        n_else += st == 'elsewhere'
    if bad:
        return False, len(bad), _fmt(bad)
    return True, len(rows), 'one row per (lst, band, date, Stokes); frame names match; %d sampled frames exist' % (
        len(sample) - n_else) + ('; %d node-local to another host (existence not checkable here)' % n_else if n_else else '')


@content_test('photometry_frame_list', 'FL3_pb_by_weighting')
def fl3_pb(path, **ctx):
    f = _cfg().photometry_frames
    _, rows = read_csv(path)
    bad, n_else = [], 0
    for r in rows:
        want = f.PB_NAME[r['stokes']].format(band=r['band'])
        if os.path.basename(r['pb_path']) != want:
            bad.append('%s_%s %s %s: PB %s != %s' % (r['lst'], r['band'], r['date'], r['stokes'],
                                                     os.path.basename(r['pb_path']), want))
            continue
        if r['weights_path'] and os.path.dirname(r['weights_path']) != os.path.dirname(r['pb_path']):
            bad.append('%s_%s: PB map and weights.csv in different directories' % (r['lst'], r['band']))
    smap = _stage_map(path)
    for p in sorted({r['pb_path'] for r in rows}):
        st = _exists_or_elsewhere(p)
        if st == 'missing' and _unstage(p, smap) != p:
            st = _exists_or_elsewhere(_unstage(p, smap))
        if st == 'missing':
            bad.append('missing %s' % p)
        n_else += st == 'elsewhere'
    if bad:
        return False, len(bad), _fmt(bad)
    return True, len({r['pb_path'] for r in rows}), 'I rows PB_deep_R075_<band>, V rows PB_deep_<band>' + (
        '; %d maps node-local to another host' % n_else if n_else else '')


def _master_digests(m):
    import hashlib
    import numpy as np
    from astropy.io import fits
    d = np.squeeze(fits.getdata(m)).astype(np.float32)
    return (hashlib.md5(np.ascontiguousarray(d).tobytes()).hexdigest(),
            hashlib.md5(np.ascontiguousarray(np.nan_to_num(d)).tobytes()).hexdigest())


@content_test('photometry_frame_list', 'FL4_parents')
def fl4_parents(path, **ctx):
    c = _cfg()
    f = c.photometry_frames
    _, rows = read_csv(path)
    smap = _stage_map(path)
    cells = {}
    bad = []
    for r in rows:
        if os.path.basename(r['parent_path']) != f.PARENT_NAME.format(lst=r['lst'], band=r['band']):
            bad.append('%s_%s: parent %s' % (r['lst'], r['band'], os.path.basename(r['parent_path'])))
        cells.setdefault((r['lst'], r['band']), set()).add(r['parent_path'])
    for k, ps in sorted(cells.items()):
        if len(ps) != 1:
            bad.append('%s_%s: %d parent catalogues' % (k[0], k[1], len(ps)))
    if bad:
        return False, len(bad), _fmt(bad)
    keys = sorted(cells)
    step = max(1, len(keys) // max(1, f.FL4_DIGEST_CELLS))
    sample = keys[::step][:f.FL4_DIGEST_CELLS]
    checked, n_else = 0, 0
    coadd = ctx.get('coadd_root') or c.paths.COADD_SB_ROOT
    for k in keys:
        par = _unstage(next(iter(cells[k])), smap)
        st = _exists_or_elsewhere(par)
        if st == 'elsewhere':
            n_else += 1
            continue
        gj = par + '.gen.json'
        if st == 'missing' or not os.path.exists(gj):
            bad.append('%s_%s: parent catalogue or its .gen.json missing (%s)' % (k[0], k[1], par))
            continue
        g = json.load(open(gj))
        want = f.MASTER_NAME.format(band=k[1])
        if g.get('master') != want:
            bad.append('%s_%s: built on %s, not %s' % (k[0], k[1], g.get('master'), want))
            continue
        if k in sample:
            m = os.path.join(coadd, '%s_%s' % k, 'coadd_final', want)
            if _exists_or_elsewhere(m) != 'ok':
                n_else += 1
                continue
            if g.get('master_md5_data') not in _master_digests(m):
                bad.append('%s_%s: parent catalogue md5 %s matches neither digest of the current master (stale generation)'
                           % (k[0], k[1], str(g.get('master_md5_data'))[:8]))
            checked += 1
    if bad:
        return False, len(bad), _fmt(bad)
    return True, len(keys), '%d cells name their own parent catalogue built on the current master name; master digest ' \
                            'recomputed for %d cells' % (len(keys), checked) + (
        '; %d not checkable here (node-local to another host)' % n_else if n_else else '')


@content_test('photometry_frame_list', 'FL5_thermnoi')
def fl5_thermnoi(path, **ctx):
    from exopipe.phase3.core.record import fits_primary_cards
    f = _cfg().photometry_frames
    _, rows = read_csv(path)
    bad = ['%s_%s %s I: thermnoi %r on a Stokes I row' % (r['lst'], r['band'], r['date'], r['thermnoi'])
           for r in rows if r['stokes'] == 'I' and r['thermnoi'] != '']
    v = [r for r in rows if r['stokes'] == 'V']
    step = max(1, len(v) // max(1, f.FL5_THERMNOI_ROWS))
    checked = n_else = 0
    smap = _stage_map(path)
    for r in v[::step][:f.FL5_THERMNOI_ROWS]:
        fp = r['path'] if os.path.exists(r['path']) else _unstage(r['path'], smap)
        if _exists_or_elsewhere(fp) != 'ok':
            n_else += 1
            continue
        t = fits_primary_cards(fp).get('THERMNOI')
        want = str(t or '') if not isinstance(t, bool) else str(t)
        if r['thermnoi'] != want:
            bad.append('%s: thermnoi %r, header %r' % (os.path.basename(r['path']), r['thermnoi'], want))
        checked += 1
    if bad:
        return False, len(bad), _fmt(bad)
    return True, checked, 'THERMNOI of %d sampled V frames equals the header; I rows blank' % checked + (
        '; %d not checkable here' % n_else if n_else else '')


@content_test('photometry_frame_list', 'FL6_refused_cells')
def fl6_refused(path, refused_cells=(), refused_stokes_i_cells=(), **ctx):
    """No row of a cell the frame-list engine refused whole (parents, PB_deep map, no search frames:
    deep_forced_frames.py:34, :36, :55), and no Stokes I row of a cell whose Robust -0.75 PB map is missing (each I frame
    refused, :50-51; its V rows stay listed)."""
    _, rows = read_csv(path)
    ref = {c for c in (refused_cells or ())}
    iref = {c for c in (refused_stokes_i_cells or ())}
    hit = sorted({'%s_%s' % (r['lst'], r['band']) for r in rows} & ref)
    hit_i = sorted({'%s_%s' % (r['lst'], r['band']) for r in rows if r['stokes'] == 'I'} & iref)
    if hit or hit_i:
        return False, len(hit) + len(hit_i), ('rows of refused cells: %s' % _fmt(hit) if hit else '') + (
            '%sStokes I rows of cells without their I PB map: %s' % ('; ' if hit else '', _fmt(hit_i)) if hit_i else '')
    return True, len(ref) + len(iref), 'no row of the %d refused cells%s' % (len(ref), (
        '; no Stokes I row of the %d cells without their I PB map' % len(iref)) if iref else '')


# ---------------------------------------------------------------------------------------------------------------------
# photometry_stage_list
# ---------------------------------------------------------------------------------------------------------------------

@content_test('photometry_stage_list', 'SL1_paths')
def sl1_paths(path, **ctx):
    f = _cfg().photometry_frames
    lines = [l.rstrip('\n') for l in open(path) if l.strip()]
    bad = []
    if lines != sorted(set(lines)):
        bad.append('not sorted and unique')
    for l in lines:
        if not os.path.isabs(l) or not any(l.startswith(p.rstrip('/') + '/') for p in f.STAGE_PREFIX_ROOTS):
            bad.append('%s not under a stage prefix %s' % (l, f.STAGE_PREFIX_ROOTS))
    fl = os.path.join(os.path.dirname(os.path.abspath(path)), f.FRAMES_FILE)
    if os.path.exists(fl):
        _, rows = read_csv(fl)
        named = {r[k] for r in rows for k in f.STAGE_COLUMNS if r[k]}
        extra = [l for l in lines if l not in named]
        need = sorted(p for p in named if any(p.startswith(x.rstrip('/') + '/') for x in f.STAGE_PREFIX_ROOTS))
        if extra:
            bad.append('%d paths not named by %s: %s' % (len(extra), f.FRAMES_FILE, _fmt(extra)))
        missing = [p for p in need if p not in set(lines)]
        if missing and ctx.get('complete', True):
            bad.append('%d inputs under a stage prefix not listed: %s' % (len(missing), _fmt(missing)))
    if bad:
        return False, len(bad), _fmt(bad)
    return True, len(lines), '%d staged inputs, all named by the frame list' % len(lines)


# ---------------------------------------------------------------------------------------------------------------------
# photometry_table (deep out/forced_NNN.csv.gz): one pass, every check
# ---------------------------------------------------------------------------------------------------------------------

def _measure_one(d, w, ra, dec, edge, stamp, core, minpix, madf):
    """Value (bilinear, scipy map_coordinates order 1) and stamp rms at one position - the deep measurement written out
    here (deep_forced_phot.measure :102-115)."""
    import numpy as np
    from scipy.ndimage import map_coordinates
    x, y = w.world_to_pixel_values(ra, dec)
    x, y = float(x), float(y)
    ny, nx = d.shape
    if not (np.isfinite(x) and np.isfinite(y) and x > edge and y > edge and x < nx - edge - 1 and y < ny - edge - 1):
        return float('nan'), float('nan')
    val = float(map_coordinates(d, [[y], [x]], order=1, mode='constant', cval=np.nan)[0])
    xi, yi = int(round(x)), int(round(y))
    h = stamp // 2
    s = d[yi - h:yi + h + 1, xi - h:xi + h + 1].copy()
    a = h - core // 2
    s[a:a + core, a:a + core] = np.nan
    s = s[np.isfinite(s) & (s != 0)]
    rms = madf * float(np.median(np.abs(s - np.median(s)))) if s.size > minpix else float('nan')
    return val, rms


def _table_pass(path):
    c = _cfg()
    m = c.photometry_measure
    floor = m.MIN_ELEVATION_DEG['deep']
    res = {}
    with gzip.open(path, 'rt', newline='') as fh:
        rd = csv.reader(fh)
        hdr = next(rd, [])
        ix = {h: i for i, h in enumerate(hdr)}
        want = list(FORCED_COLUMNS) + list(RADIO_COLUMNS) + list(ENTRY_COLUMNS)
        res['T1'] = (hdr == want, len(hdr), 'header = the %d production columns' % len(want) if hdr == want else
                     'header differs: missing %s; extra %s; order %s' % ([h for h in want if h not in ix],
                                                                       [h for h in hdr if h not in want],
                                                                       'same' if [h for h in hdr if h in want] ==
                                                                       [h for h in want if h in ix] else 'differs'))
        need = [k for k in KEY + ('kind', 'ra', 'dec', 'elev', 'value', 'rms', 'snr', 'pb', 'flux_pbcorr') if k not in ix]
        if need:
            for t in ('T2', 'T3', 'T4', 'T5', 'T6'):
                res[t] = (False, None, 'columns missing: %s' % need)
            return res
        has_entry = all(k in ix for k in ENTRY_COLUMNS)
        has_pos = all(k in ix for k in RADIO_COLUMNS)
        kix = [ix[k] for k in KEY]
        b2, b3, b5, b6 = [], [], [], []
        n_t = n_c = 0
        last = None
        excl = _excl()
        exc_cache = {}
        t4_rows = {}
        for n, row in enumerate(rd, 2):
            if len(row) != len(hdr):
                b2.append('line %d: %d fields' % (n, len(row)))
                continue
            kind = row[ix['kind']]
            key = tuple(row[i] for i in kix)
            try:
                v, r_, s_ = float(row[ix['value']]), float(row[ix['rms']]), float(row[ix['snr']])
            except ValueError:
                b6.append('line %d: value/rms/snr not numbers' % n)
                continue
            if not (r_ > 0 and abs(s_ - v / r_) <= 5.001e-4 + abs(v / r_) * 1.1e-4):
                b6.append('line %d: snr %s != value/rms %.4f' % (n, row[ix['snr']], v / r_ if r_ else float('nan')))
            if kind == 'target':
                n_t += 1
                last = (key, row[ix['ra']], row[ix['dec']])
                if has_entry and (abs(float(row[ix['target_ra']]) - float(row[ix['ra']])) > 1e-5 or
                                  abs(float(row[ix['target_dec']]) - float(row[ix['dec']])) > 1e-5):
                    b2.append('line %d: target row target_ra/dec != its ra/dec' % n)
                S = key[5]
                try:
                    el = float(row[ix['elev']])
                except ValueError:
                    el = float('nan')
                if not el >= floor.get(S, 90.0):
                    b5.append('line %d: elevation %s below the %s floor %s deg' % (n, row[ix['elev']], S, floor.get(S)))
                ek = (key[2], key[3], key[4])
                if ek not in exc_cache:
                    exc_cache[ek] = excl.is_excluded(*ek)
                if exc_cache[ek]:
                    b5.append('line %d: %s %s %s is excluded (data_decisions.csv, core.exclusions)' % ((n,) + ek))
                pbs, fp = row[ix['pb']], row[ix['flux_pbcorr']]
                try:
                    pb = float(pbs)
                except ValueError:
                    pb = float('nan')
                pf = m.PB_FLOOR['deep']
                if pb > pf + 6e-5 and fp == '':
                    b6.append('line %d: pb %s > %s but flux_pbcorr blank' % (n, pbs, pf))
                elif not (pb > pf - 6e-5) and fp != '':
                    b6.append('line %d: pb %s <= %s but flux_pbcorr %s' % (n, pbs, pf, fp))
                elif fp != '' and pb > 0 and abs(float(fp) - v / pb) > _sigfig_tol(fp) + abs(v / pb) * 3e-4:
                    b6.append('line %d: flux_pbcorr %s != value/pb %.5g' % (n, fp, v / pb))
                if has_pos and abs(s_) >= c.fitting.RADIO_SNR_FLOOR:
                    pk, pr = row[ix['pos_kind']], row[ix['pos_reason']]
                    if pk not in POS_KINDS or not pr:
                        b3.append('line %d: |snr| %s without a radio-position record (pos_kind %r)' % (n, row[ix['snr']], pk))
                fk = key[2:]
                t4_rows.setdefault(fk, []).append((row[ix['ra']], row[ix['dec']], row[ix['value']], row[ix['rms']], n,
                                                   key[0], key[1]))
            elif kind == 'control':
                n_c += 1
                if last is None:
                    b2.append('line %d: control before any target row' % n)
                    continue
                if key != last[0]:
                    b2.append('line %d: control key %s differs from the target row it follows %s' % (n, key, last[0]))
                elif has_entry and (abs(float(row[ix['target_ra']]) - float(last[1])) > 1e-5 or
                                    abs(float(row[ix['target_dec']]) - float(last[2])) > 1e-5):
                    b2.append('line %d: control target_ra/dec != the target row it follows' % n)
            else:
                b2.append('line %d: kind %r' % (n, kind))
        res['T2'] = (not b2, len(b2), _fmt(b2) if b2 else '%d target rows, %d controls, each after its own target entry%s'
                     % (n_t, n_c, '' if has_entry else ' (no target_ra/dec columns: order only)'))
        if not has_pos:
            res['T3'] = (False, None, 'no radio-position columns (pos_*): measured before deep_forced_radio')
        else:
            res['T3'] = (not b3, len(b3), _fmt(b3) if b3 else 'every target row at |snr| >= %g carries pos_kind and pos_reason'
                         % c.fitting.RADIO_SNR_FLOOR)
        res['T5'] = (not b5, len(b5), _fmt(b5) if b5 else 'every target row above its Stokes floor %s; none on an excluded '
                     'epoch (%d epochs checked)' % (floor, len(exc_cache)))
        res['T6'] = (not b6, len(b6), _fmt(b6) if b6 else 'snr = value/rms on %d rows; flux_pbcorr blank exactly where pb <= %g'
                     % (n_t + n_c, m.PB_FLOOR['deep']))
        res['_t4'] = t4_rows
    return res


def _list_positions():
    """{(sample, name): [(ra, dec)]} of the measured sample lists - the exact positions the engine measured at
    (deep_forced_phot.py:86-89 floats of ra_current / dec_current); a row prints them to 5 decimals (:288), and the 10 pc
    list carries up to 9, so a re-measurement at the printed position differs by the rounding (0.04 arcsec)."""
    out = {}
    for tag, nm, ra, dec in sample_targets():
        out.setdefault((tag, nm), []).append((ra, dec))
    return out


def _t4(path, res):
    """Re-measure T4_ROWS_PER_FRAME target rows on T4_FRAMES frames of the shard file (frame path from the sibling
    admission file frames_NNN.csv) at the target's EXACT list position (the entry of the current list whose position
    prints as the row's ra/dec); a row whose entry is no longer in the lists is re-measured at its printed position."""
    import numpy as np
    from astropy.io import fits
    from astropy.wcs import WCS
    m = _cfg().photometry_measure
    rows = res.get('_t4') or {}
    if not rows:
        return True, 0, 'no target row to re-measure'
    adm = os.path.join(os.path.dirname(path), re.sub(r'^forced_(\d+)\.csv\.gz$', r'frames_\1.csv', os.path.basename(path)))
    if not os.path.exists(adm):
        return False, None, 'sibling admission file %s missing (frame paths)' % os.path.basename(adm)
    _, ar = read_csv(adm)
    fpath = {(r['lst'], r['band'], r['date'], r['stokes']): r['path'] for r in ar}
    keys = sorted(rows)
    step = max(1, len(keys) // max(1, m.T4_FRAMES))
    bad, done, n_else, n_printed = [], 0, 0, 0
    lists = _list_positions()
    for k in keys[::step][:m.T4_FRAMES]:
        p = fpath.get(k)
        if not p:
            bad.append('%s: no admission row' % '_'.join(k))
            continue
        if _exists_or_elsewhere(p) != 'ok':
            n_else += 1
            continue
        with fits.open(p, memmap=False) as h:
            d = np.squeeze(h[0].data).astype(np.float64)
            w = WCS(h[0].header).celestial
        for ras, decs, vs, rs, n, smp, nm in rows[k][:m.T4_ROWS_PER_FRAME]:
            exact = [(a, b) for a, b in lists.get((smp, nm), ()) if '%.5f' % a == ras and '%.5f' % b == decs]
            if exact:
                ra, dec = exact[0]
            else:
                ra, dec = float(ras), float(decs)
                n_printed += 1
            v, r_ = _measure_one(d, w, ra, dec, m.EDGE_PX['deep'], m.STAMP_PX, m.STAMP_CORE_PX,
                                 m.STAMP_MIN_PIXELS['deep'], m.RMS_MAD_FACTOR)
            if not (abs(v - float(vs)) <= _sigfig_tol(vs) and abs(r_ - float(rs)) <= _sigfig_tol(rs)):
                bad.append('line %d (%s): value/rms %s/%s, recomputed %.5g/%.5g' % (n, '_'.join(k), vs, rs, v, r_))
            done += 1
    if bad:
        return False, len(bad), _fmt(bad)
    return True, done, '%d target rows re-measured on their frames: value and stamp rms equal to 5 significant figures' % done \
        + ('; %d at the printed position (entry no longer in the lists)' % n_printed if n_printed else '') \
        + ('; %d frames not checkable here (node-local to another host)' % n_else if n_else else '')


def _table(path):
    return _cached('table', path, lambda: _table_pass(path))


@content_test('photometry_table', 'T1_header')
def t1(path, **ctx):
    return _table(path)['T1']


@content_test('photometry_table', 'T2_entry_join')
def t2(path, **ctx):
    return _table(path)['T2']


@content_test('photometry_table', 'T3_radio_record')
def t3(path, **ctx):
    return _table(path)['T3']


@content_test('photometry_table', 'T4_remeasure')
def t4(path, **ctx):
    return _cached('t4', path, lambda: _t4(path, _table(path)))


@content_test('photometry_table', 'T5_floors_exclusions')
def t5(path, **ctx):
    return _table(path)['T5']


@content_test('photometry_table', 'T6_arithmetic')
def t6(path, **ctx):
    return _table(path)['T6']


# ---------------------------------------------------------------------------------------------------------------------
# photometry_blank_null, photometry_admission
# ---------------------------------------------------------------------------------------------------------------------

def _sibling(path, prefix):
    m = re.match(r'^(?:blank|frames|forced)_(\d+)\.csv(?:\.gz)?$', os.path.basename(path))
    if not m:
        return None
    ext = '.csv.gz' if prefix == 'forced' else '.csv'
    return os.path.join(os.path.dirname(path), '%s_%s%s' % (prefix, m.group(1), ext))


@content_test('photometry_blank_null', 'B1_counts')
def b1(path, **ctx):
    m = _cfg().photometry_measure
    cols, rows = read_csv(path)
    if tuple(cols) != BLANK_COLUMNS:
        return False, None, 'columns %s' % cols
    bad, seen = [], set()
    nmax = m.N_BLANK['deep']
    for r in rows:
        k = (r['lst'], r['band'], r['date'], r['stokes'])
        if k in seen:
            bad.append('duplicate %s' % '_'.join(k))
        seen.add(k)
        try:
            n = int(r['n'])
            g = [int(r['ge%d' % t]) for t in m.BLANK_SNR_COUNTS['deep']]
        except ValueError:
            bad.append('%s: counts not integers' % '_'.join(k))
            continue
        if not 0 < n <= nmax:
            bad.append('%s: n %d outside 1..%d' % ('_'.join(k), n, nmax))
        if any(b > a for a, b in zip(g, g[1:])) or g[0] > n or min(g) < 0:
            bad.append('%s: counts %s not non-increasing within n %d' % ('_'.join(k), g, n))
    adm = _sibling(path, 'frames')
    if adm and os.path.exists(adm):
        _, ar = read_csv(adm)
        ok = {(r['lst'], r['band'], r['date'], r['stokes']) for r in ar if r['admitted'] == '1'}
        stray = sorted('_'.join(k) for k in seen - ok)
        if stray:
            bad.append('blank rows of frames not admitted: %s' % _fmt(stray))
    elif ctx.get('need_admission', True):
        bad.append('sibling admission file missing')
    if bad:
        return False, len(bad), _fmt(bad)
    return True, len(rows), '%d admitted frames, 0 < n <= %d, counts non-increasing in |S/N|' % (len(rows), nmax)


def _frame_list_rows(ctx):
    p = ctx.get('frames_csv')
    if not p or not os.path.exists(p):
        return None
    return read_csv(p)[1]


def shard_cells(rows, shard, nshards):
    cells = sorted({(r['lst'], r['band']) for r in rows})
    return [cl for i, cl in enumerate(cells) if i % nshards == shard]


@content_test('photometry_admission', 'A1_admission')
def a1(path, **ctx):
    m = _cfg().photometry_measure
    cols, rows = read_csv(path)
    if tuple(cols) != ADMISSION_COLUMNS:
        return False, None, 'columns %s' % cols
    rx = [re.compile(x) for x in m.ADMISSION_REASONS['deep']]
    excl = _excl()
    bad, seen = [], set()
    nadm = 0
    for r in rows:
        k = (r['lst'], r['band'], r['date'], r['stokes'])
        if k in seen:
            bad.append('duplicate %s' % '_'.join(k))
        seen.add(k)
        ex = excl.is_excluded(r['lst'], r['band'], r['date'])
        if r['admitted'] == '1':
            nadm += 1
            if r['reason']:
                bad.append('%s admitted with reason %r' % ('_'.join(k), r['reason']))
            if ex:
                bad.append('%s admitted but excluded (core.exclusions)' % '_'.join(k))
        elif r['admitted'] == '0':
            if not any(x.match(r['reason']) for x in rx):
                bad.append('%s refused with an unknown reason %r' % ('_'.join(k), r['reason']))
            if ex != bool(rx[0].match(r['reason'])):
                bad.append('%s: reason %r but core.exclusions says %s' % ('_'.join(k), r['reason'],
                                                                          'excluded' if ex else 'admitted'))
            if r['n_targets'] != '0' or r['n_controls'] != '0':
                bad.append('%s refused but measured' % '_'.join(k))
        else:
            bad.append('%s admitted %r' % ('_'.join(k), r['admitted']))
    cover = ''
    fl = _frame_list_rows(ctx)
    if fl is not None and ctx.get('shard') is not None and ctx.get('nshards'):
        mine = set(shard_cells(fl, int(ctx['shard']), int(ctx['nshards'])))
        want = {(r['lst'], r['band'], r['date'], r['stokes']) for r in fl if (r['lst'], r['band']) in mine}
        miss, extra = sorted(want - seen), sorted(seen - want)
        if miss:
            bad.append('%d frames of the shard cells not listed: %s' % (len(miss), _fmt(['_'.join(x) for x in miss])))
        if extra:
            bad.append('%d listed frames outside the shard cells: %s' % (len(extra), _fmt(['_'.join(x) for x in extra])))
        cover = '; every frame of the shard\'s %d cells listed once' % len(mine)
    if bad:
        return False, len(bad), _fmt(bad)
    return True, len(rows), '%d frames (%d admitted); refusal reasons known; exclusions agree with core.exclusions%s' % (
        len(rows), nadm, cover)


# ---------------------------------------------------------------------------------------------------------------------
# photometry_eval - deep (out/eval/)
# ---------------------------------------------------------------------------------------------------------------------

def _deep_load(eval_dir):
    """Target and control rows of the measurement the evaluation read (the forced files beside eval/), each control with its
    own target entry's A-team / Jupiter / Moon / track values; gated and stat columns as the ratified evaluation defines
    them (deep_forced_eval.py docstring lines 20-30), computed here."""
    import numpy as np
    import pandas as pd
    c = _cfg()
    m = c.photometry_measure
    D = os.path.dirname(os.path.abspath(eval_dir.rstrip('/')))
    files = sorted(glob.glob(os.path.join(D, 'forced_*.csv.gz')))
    if not files:
        raise ValueError('no forced_*.csv.gz beside %s' % eval_dir)
    use = list(KEY) + ['kind', 'ra', 'dec', 'snr', 'zone', 'ateam_deg', 'jup_deg', 'moon_deg', 'jup_track_night', 'gate_size',
                       'inj5_snr', 'inj10_snr']
    parts, pos = [], []
    for p in files:
        hd = list(pd.read_csv(p, nrows=0).columns)
        x = pd.read_csv(p, usecols=[u for u in use if u in hd],
                        dtype={'jup_track_night': str, 'gate_size': str, 'date': str, 'ctrl': str}, low_memory=False)
        ti = pd.Series(np.where(x.kind.to_numpy() == 'target', np.arange(len(x)), np.nan)).ffill().to_numpy()
        if np.isnan(ti).any():
            raise ValueError('%s: control rows before any target row' % p)
        ti = ti.astype(np.int64)
        x['meas_ra'] = x.ra.to_numpy()[ti]
        x['meas_dec'] = x.dec.to_numpy()[ti]
        parts.append(x)
        pos.append([h for h in hd if h.startswith('pos_')])
    df = pd.concat(parts, ignore_index=True)
    ent = ['meas_ra', 'meas_dec']
    tg = df[df.kind == 'target'].copy()
    ct = df[df.kind == 'control'].drop(columns=['ateam_deg', 'jup_deg', 'moon_deg', 'jup_track_night']).merge(
        tg[list(KEY) + ent + ['ateam_deg', 'jup_deg', 'moon_deg', 'jup_track_night']], on=list(KEY) + ent, how='left')
    groups = {}
    for b, g in c.instrument.SUBBAND_BANDS.items():
        for s in g:
            groups[s.replace('MHz', '')] = b
    jr = m.JUPITER_RADIUS_DEG['deep']
    ar = m.ATEAM_RADIUS_DEG['deep']

    def gated(x):
        j = x.band.astype(str).str.replace('MHz', '').map(lambda b: jr.get(groups.get(b, b), 0.5))
        return ((x.gate_size.fillna('') != '') | (x.ateam_deg < x.stokes.map(ar)) | (x.jup_deg < j) |
                (x.moon_deg < m.MOON_RADIUS_DEG['deep']) | (x.jup_track_night.fillna('') != ''))
    for x in (tg, ct):
        x['gated'] = gated(x)
        x['stat'] = np.where(x.stokes == 'I', x.snr, x.snr.abs())
    blanks = sorted(glob.glob(os.path.join(D, 'blank_*.csv')))
    return tg, ct, pos, blanks


def _deep_eval(eval_dir):
    res = {}
    try:
        res = _deep_eval_checks(eval_dir)
    except Exception as e:      # noqa: BLE001 - a check that cannot run on this evaluation fails with the reason
        for t in ('E1', 'E2', 'E3'):
            res.setdefault(t, (False, None, 'evaluation check failed: %s: %s' % (type(e).__name__, e)))
    for t in ('E1', 'E2', 'E3'):
        res.setdefault(t, (False, None, 'not computed (an earlier check raised)'))
    return res


def _deep_eval_checks(eval_dir):
    import numpy as np
    import pandas as pd
    from scipy.stats import norm
    c = _cfg()
    ev = c.photometry_evaluate
    res = {}
    try:
        tg, ct, pos, blanks = _deep_load(eval_dir)
    except Exception as e:      # noqa: BLE001
        for t in ('E1', 'E2', 'E3'):
            res[t] = (False, None, 'measurement unreadable: %s' % e)
        return res
    th = pd.read_csv(os.path.join(eval_dir, 'thresholds.csv'), dtype=str)
    bad1, thr = [], {}
    if len(th) != ev.THRESHOLD_ROWS['deep']:
        bad1.append('%d rows != %d' % (len(th), ev.THRESHOLD_ROWS['deep']))
    for _, r in th.iterrows():
        smp, S = r['sample'], r['stokes']
        tt = tg[(tg['sample'] == smp) & (tg.stokes == S)]
        cc = ct[(ct['sample'] == smp) & (ct.stokes == S)]
        cl = np.sort(cc[(cc.zone == 0) & ~cc.gated].stat.dropna().to_numpy())
        ntr = int(((tt.zone == 0) & ~tt.gated).sum())
        rate = 1.0 / ntr if ntr else float('nan')
        emp = float(cl[int(math.ceil(len(cl) * (1 - rate)))]) if len(cl) and rate * len(cl) >= 1 else float('nan')
        gth = float(norm.isf(rate)) if S == 'I' else float(norm.isf(rate / 2))
        want = {'target_trials_clean_gated': str(ntr), 'n_controls_clean_gated': str(len(cl)),
                'rate_for_1_false_alarm': '%.2e' % rate, 'gaussian_threshold': '%.2f' % gth,
                'control_threshold': ('%.2f' % emp) if np.isfinite(emp) else 'controls too few (need >= %d)'
                                                                                % int(math.ceil(1 / rate)),
                'controls_ge5': str(int((cl >= 5).sum())), 'controls_ge6': str(int((cl >= 6).sum())),
                'controls_ge7': str(int((cl >= 7).sum()))}
        for k, v in want.items():
            if str(r.get(k)) != v:
                bad1.append('%s %s %s: file %s recomputed %s' % (smp, S, k, r.get(k), v))
        thr[(smp, S)] = emp
    res['E1'] = (not bad1, len(bad1), _fmt(bad1) if bad1 else '%d thresholds recomputed from the clean gated controls '
                 '(one false alarm over the clean gated target trials): equal' % len(th))
    # E2 lists
    bad2 = []
    g4 = pd.read_csv(os.path.join(eval_dir, 'targets_ge4.csv'), dtype={'date': str, 'gate_size': str,
                                                                       'jup_track_night': str}, keep_default_na=True)
    if len(g4) and (g4.snr.abs() < ev.LIST_SNR).any():
        bad2.append('%d targets_ge4 rows below |snr| %g' % (int((g4.snr.abs() < ev.LIST_SNR).sum()), ev.LIST_SNR))
    entry = 'meas_ra' in g4.columns          # evaluations before 2026-10-11 carry no measurement point (name join)
    kc = list(KEY) + (['_r', '_d'] if entry else [])
    tk = tg.assign(_r=tg.meas_ra.round(5), _d=tg.meas_dec.round(5)).groupby(kc).size()
    if len(g4):
        k4 = g4.assign(_r=g4.meas_ra.round(5), _d=g4.meas_dec.round(5)) if entry else g4
        cnt = k4.set_index(kc).index.map(lambda i: int(tk.get(i, 0)))
        nbad = int((cnt != 1).sum()) if entry else int((cnt < 1).sum())
        if nbad:
            bad2.append('%d listed rows not found %s among the measurement\'s target rows' % (
                nbad, 'exactly once' if entry else 'at all (name join)'))
        if any(pos) and ('pos_kind' not in g4 or g4.pos_kind.fillna('').eq('').any()):
            bad2.append('listed rows without a radio-position record')
    n_ge4 = int((tg.snr.abs() >= ev.LIST_SNR).sum())
    if n_ge4 != len(g4):
        bad2.append('targets_ge4 holds %d rows; the measurement has %d target rows at |snr| >= %g'
                    % (len(g4), n_ge4, ev.LIST_SNR))
    ab = pd.read_csv(os.path.join(eval_dir, 'targets_above_threshold.csv'), dtype={'date': str})
    if len(ab):
        st = np.where(ab.stokes == 'I', ab.snr, ab.snr.abs())
        wrong = (ab.zone != 0) | (ab.gated.astype(str) != 'False') | (st < ab.threshold - 1e-9)
        if wrong.any():
            bad2.append('%d above-threshold rows not clean, gated or below their threshold' % int(wrong.sum()))
        for (smp, S), g in ab.groupby(['sample', 'stokes']):
            t = thr.get((smp, S))
            if t is None or not np.isfinite(t) or (g.threshold - round(t, 2)).abs().max() > 1e-6:
                bad2.append('%s %s threshold %s != thresholds.csv %s' % (smp, S, g.threshold.iloc[0], t))
    if len(g4):
        st4 = np.where(g4.stokes == 'I', g4.snr, g4.snr.abs())
        thr4 = np.array([round(thr.get((a, b), float('nan')), 2) for a, b in zip(g4['sample'], g4.stokes)])
        n_above = int(((g4.zone == 0) & (g4.gated.astype(str) == 'False') & (st4 >= thr4)).sum())
        if n_above != len(ab):
            bad2.append('targets_above_threshold holds %d rows; targets_ge4 has %d clean ungated rows at or above their '
                        'threshold' % (len(ab), n_above))
    res['E2'] = (not bad2, len(bad2), _fmt(bad2) if bad2 else 'targets_ge4: %d rows, each once in the measurement; '
                 'targets_above_threshold: %d clean ungated rows at or above their threshold' % (len(g4), len(ab)))
    # E3 exceedance
    bad3 = []
    with open(os.path.join(eval_dir, 'exceedance.csv')) as f:
        ex = list(csv.reader(f))
    hdr, body = ex[0], ex[1:]
    nblank = {}
    for b in blanks:
        for r in csv.DictReader(open(b)):
            nblank[r['stokes']] = nblank.get(r['stokes'], 0) + int(r['n'])
    for r in body:
        smp, S, scr, sset, n = r[0], r[1], r[2], r[3], r[4]
        if smp in ('gaussian', 'blank_abs'):
            if smp == 'blank_abs' and int(n) != nblank.get(S, 0):
                bad3.append('blank_abs %s n %s != %d' % (S, n, nblank.get(S, 0)))
            continue
        z = 0 if scr == 'clean' else 1
        tt = tg[(tg['sample'] == smp) & (tg.stokes == S) & (tg.zone == z)]
        cc = ct[(ct['sample'] == smp) & (ct.stokes == S) & (ct.zone == z)]
        rr = {'targets': tt, 'controls': cc, 'controls_gated': cc[~cc.gated]}.get(sset)
        if rr is None:
            bad3.append('unknown set %s' % sset)
            continue
        if int(n) != int(rr.stat.notna().sum()):
            bad3.append('%s %s %s %s n %s != %d' % (smp, S, scr, sset, n, int(rr.stat.notna().sum())))
        fr = [float(v) for v in r[5:] if v != '']
        if any(b > a * (1 + 1e-3) + 1e-12 for a, b in zip(fr, fr[1:])):
            bad3.append('%s %s %s %s fractions increase with t' % (smp, S, scr, sset))
    res['E3'] = (not bad3, len(bad3), _fmt(bad3) if bad3 else '%d exceedance rows: n recomputed, fractions non-increasing'
                 % len(body))
    return res


# ---------------------------------------------------------------------------------------------------------------------
# photometry_eval - snapshot (eval_<tag>/, eval_wb_<tag>/)
# ---------------------------------------------------------------------------------------------------------------------

def _grid():
    lo, hi, st = _cfg().photometry_evaluate.THRESHOLD_GRID['snapshot']
    return [round(lo + st * k, 2) for k in range(int(round((hi - lo) / st)) + 1)]


def snapshot_grid_check(ev):
    """G1-G6 (port of test_snapshot_forced_eval_grid.check, :37-92): [problems]."""
    GRID = _grid()
    TOP = GRID[-1]
    bad = []
    th = read_csv(os.path.join(ev, 'thresholds.csv'))[1]
    gr = read_csv(os.path.join(ev, 'thresholds_grid.csv'))[1]
    ex = read_csv(os.path.join(ev, 'exceedance.csv'))[1]
    by = {}
    for r in gr:
        by.setdefault((r['sample'], r['stokes'], r['tier']), []).append(r)
    for t in th:
        k = (t['sample'], t['stokes'], t['tier'])
        g = by.get(k, [])
        ts = [round(float(r['t']), 2) for r in g]
        if ts != GRID:
            bad.append('G1 %s grid %s..%s (%d values) != %.2f..%.2f (%d)' % (k, ts[0] if ts else '-', ts[-1] if ts else '-',
                                                                           len(ts), GRID[0], TOP, len(GRID)))
            continue
        nge = [int(r['n_controls_ge']) for r in g]
        nc = int(g[0]['n_controls'])
        ntr = int(t['n_target_trials'])
        sf = float(g[0]['survey_factor'])
        if any(b > a for a, b in zip(nge, nge[1:])):
            bad.append('G2 %s controls at or above t increase with t' % (k,))
        if nc != int(t['n_controls']):
            bad.append('G2 %s n_controls grid %d thresholds %s' % (k, nc, t['n_controls']))
        for r, n in zip(g, nge):
            rate = n / nc if nc else float('nan')
            if abs(float(r['rate']) - rate) > 1e-6 * max(rate, 1e-300) + 1e-15:
                bad.append('G2 %s rate at %s' % (k, r['t']))
                break
            if abs(float(r['fa_here']) - rate * ntr) > 1e-5 * max(rate * ntr, 1e-12) + 1e-9:
                bad.append('G2 %s fa_here at %s' % (k, r['t']))
                break
            if abs(float(r['fa_survey']) - rate * ntr * sf) > 1e-5 * max(rate * ntr * sf, 1e-12) + 1e-9:
                bad.append('G2 %s fa_survey at %s' % (k, r['t']))
                break
        for col, fa in (('thr_here', [n / nc * ntr for n in nge]), ('thr_survey', [n / nc * ntr * sf for n in nge])):
            want = next((tt for tt, f in zip(GRID, fa) if f <= 1.0), float('nan'))
            got = float(t[col])
            if not ((math.isnan(want) and math.isnan(got)) or abs(want - got) < 1e-6):
                bad.append('G3 %s %s file %s recomputed %s' % (k, col, t[col], want))
        for col, n_ in (('thr_here_gauss', ntr), ('thr_survey_gauss', ntr * sf)):
            want = next((tt for tt in GRID if 0.5 * math.erfc(tt / math.sqrt(2)) * (1 if k[1] == 'I' else 2) * n_ <= 1.0),
                        float('nan'))
            if not ((math.isnan(want) and math.isnan(float(t[col]))) or abs(want - float(t[col])) <= 1e-6):
                bad.append('G3 %s %s file %s recomputed %s' % (k, col, t[col], want))
        note = t['survey_note']
        if 'exhausted' in note or 'MORE than one false alarm' not in note or ('%.2f' % TOP) not in note:
            bad.append('G4 %s note: %s' % (k, note))
        if abs(float(t['grid_top']) - TOP) > 1e-9 or abs(float(t['fa_here_at_grid_top']) - nge[-1] / nc * ntr) > \
                1e-2 * max(nge[-1] / nc * ntr, 1e-9) + 1e-9:
            bad.append('G5 %s grid top / false alarms there' % (k,))
        e = [r for r in ex if (r['sample'], r['stokes'], r['tier'], r['screen'], r['set']) == k + ('clean', 'controls')]
        if len(e) != 1 or int(e[0]['n']) != nc:
            bad.append('G6 %s exceedance n' % (k,))
        else:
            cols = [cc for cc in e[0] if cc.startswith('ge')]
            if 'ge9.00' not in cols or ('ge%.2f' % TOP) not in cols or not any(float(cc[2:]) > 9.0 for cc in cols):
                bad.append('G6 %s exceedance columns stop at %s' % (k, cols[-1] if cols else '-'))
            for cc in cols:
                tt = round(float(cc[2:]), 2)
                rate = nge[GRID.index(tt)] / nc
                if abs(float(e[0][cc]) - rate) > 2e-3 * max(rate, 1e-300) + 1e-15:
                    bad.append('G6 %s %s file %s grid %.3e' % (k, cc, e[0][cc], rate))
                    break
    return bad, len(th)


def _fwhm_deg(band):
    ev = _cfg().photometry_evaluate
    f = ev.SNAPSHOT_BAND_REP_MHZ.get(band) or int(band.replace('MHz', ''))
    import numpy as np
    k = sorted(ev.SNAPSHOT_FWHM_ARCMIN)
    return float(np.interp(f, k, [ev.SNAPSHOT_FWHM_ARCMIN[x] for x in k])) / 60.0


def _radius_bw(table, S):
    import numpy as np
    tab = sorted(table)
    mids = np.array([math.sqrt(a * min(b, 4 * a)) for a, b, _ in tab])
    R = np.array([r for _, _, r in tab])
    S = np.asarray(S, float)
    out = np.interp(np.log(np.maximum(S, 1e-9)), np.log(mids), R, left=0.0, right=R[-1])
    out[S < tab[0][0]] = 0.0
    return out


def _gaul(p):
    import numpy as np
    hdr, out = None, []
    for line in open(p):
        if not line.strip():
            continue
        if line.startswith('#'):
            f = [x.strip() for x in line.lstrip('#').split(',')]
            if hdr is None and 'RA' in f:
                hdr = f
            continue
        if hdr:
            r = dict(zip(hdr, [x.strip() for x in line.split(',')]))
            try:
                v = (float(r['RA']), float(r['DEC']), float(r['Peak_flux']))
            except (KeyError, ValueError):
                continue
            if all(math.isfinite(x) for x in v):
                out.append(v)
    return np.array(out).reshape(-1, 3)


def _median_rms(cell, band, st, tier):
    import numpy as np
    p = os.path.join(_cfg().paths.SNAPSHOT_FLEET_OUT, cell, 'noise_%s_%s_median%s.csv' % (band, st, '_1min' if tier == '1min'
                                                                                        else ''))
    if not os.path.exists(p):
        return float('nan')
    v = [float(r['global_rms_jyb']) for r in read_csv(p)[1] if r.get('admitted') == '1' and r.get('global_rms_jyb')]
    return float(np.median(v)) if v else float('nan')


_WCS = {}


def _pix_deg(lst, ra, dec):
    import numpy as np
    from astropy.wcs import WCS
    if lst not in _WCS:
        w = WCS(naxis=2)
        w.wcs.ctype = ['RA---SIN', 'DEC--SIN']
        w.wcs.crval = [15.0 * (int(lst[:2]) + 0.5), _cfg().photometry_evaluate.SNAPSHOT_SITE_LAT_DEG]
        w.wcs.crpix = [0.0, 0.0]
        w.wcs.cdelt = [-1.0, 1.0]
        _WCS[lst] = w
    x, y = _WCS[lst].wcs_world2pix(np.atleast_1d(ra), np.atleast_1d(dec), 0)
    return np.stack([x, y], -1)


_TABLES = None


def _curve_tables():
    global _TABLES
    if _TABLES is None:
        J = json.load(open(_cfg().catalog_snapshot.CURVE_CONSTANTS))
        if (J.get('meta') or {}).get('metric') != 'image-pixel':
            raise ValueError('the curve file does not declare the image-pixel metric')
        _TABLES = {k: v.get('table') for k, v in J['curves'].items()}
    return _TABLES


def zones_cell(task):
    """Z1 of ONE cell (port of test_snapshot_forced_reserve.zones_cell, :171-200): [(key, file screen, recomputed)]."""
    import numpy as np
    cell, rr, stage1 = task
    lst = cell[:3]
    grp, cat, out = {}, {}, []
    mcat = _cfg().catalog_snapshot.MASTER_CATALOGS
    tabs = _curve_tables()
    for r in rr:
        grp.setdefault((r['sample'], r['band'], r['stokes'], r['tier']), []).append(r)
    for (smp, band, st, tier), g in sorted(grp.items()):
        z = np.load(os.path.join(stage1, cell, 'forced_%s_%s_%s.npz' % (smp, band, st)), allow_pickle=False)
        names = [str(n) for n in z['names']]
        ra, dec = np.asarray(z['ra'], float), np.asarray(z['dec'], float)
        tab = tabs.get('%s_%s_%s' % (tier, st, SNAPSHOT_GROUP[band]))
        gp = os.path.join(mcat, 'master_R075_%s_%s.gaul.csv' % (lst, band))
        rms = _median_rms(cell, band, st, tier)
        if not tab or not os.path.exists(gp) or not (rms > 0):
            flag = np.zeros(len(names), bool)
        else:
            if band not in cat:
                cat[band] = _gaul(gp)
            P = cat[band]
            R = _radius_bw(tab, P[:, 2] / rms)
            m = R > 0
            if not m.any():
                flag = np.zeros(len(names), bool)
            else:
                a = _pix_deg(lst, ra, dec)
                b = _pix_deg(lst, P[m, 0], P[m, 1])
                d = np.hypot(a[:, None, 0] - b[None, :, 0], a[:, None, 1] - b[None, :, 1]) / _fwhm_deg(band)
                flag = (d < R[m][None, :]).any(1)
        idx = {n: i for i, n in enumerate(names)}
        for r in g:
            out.append(((cell, r['target'], band, st, tier, r['ts']), r['screen'],
                        'zone' if flag[idx[r['target']]] else 'clean'))
    return out


def _pool_map(fn, tasks, procs):
    if procs and procs > 1 and len(tasks) > 1:
        import multiprocessing as mp
        with mp.Pool(min(procs, len(tasks))) as pool:
            return pool.map(fn, tasks, chunksize=1)
    return [fn(t) for t in tasks]


def _stage1_root(ev, ctx):
    return ctx.get('stage1') or os.path.join(os.path.dirname(os.path.abspath(ev.rstrip('/'))), 'stage1')


def zone_rows(ev, stage1, max_cells=None, procs=1):
    """Every targets_ge4 row of an evaluation with its file screen and recomputed screen (cached per file)."""
    tp = os.path.join(ev, 'targets_ge4.csv')
    if not os.path.exists(tp):         # the engine writes no targets_ge4.csv when no row reaches the list (:294)
        return [], 0, 0

    def run():
        T = read_csv(tp)[1]
        bycell = {}
        for r in T:
            bycell.setdefault(r['cell'], []).append(r)
        cells = sorted(bycell)
        if max_cells:
            step = max(1, len(cells) // max_cells)
            cells = cells[::step][:max_cells]
        out = []
        for part in _pool_map(zones_cell, [(cl, bycell[cl], stage1) for cl in cells], procs):
            out.extend(part)
        return out, len(cells), len(bycell)
    return _cached('zones:%s:%s' % (stage1, max_cells), os.path.join(ev, 'targets_ge4.csv'), run)


def stage1_cell(task):
    """S of ONE evaluated cell (port of test_snapshot_forced_reserve.stage1_cell, :423-459)."""
    cell, kind, root, proto = task
    bad, nband, nfile, nnocut = [], 0, 0, 0
    man = []
    for mp_ in glob.glob(os.path.join(root, cell, 'forced_manifest*.csv')):
        man += [r['file'] for r in read_csv(mp_)[1] if r.get('status') == 'ok']
    for ri in sorted(glob.glob(os.path.join(proto, cell, 'runinfo_*_median.json'))):
        b, st = os.path.basename(ri)[len('runinfo_'):-len('_median.json')].rsplit('_', 1)
        if b == 'Full' or (kind == 'sb') != ('MHz' in b):
            continue
        nband += 1
        try:
            ncut = int(json.load(open(ri)).get('photometry_cutout_cubes', -1))
        except (ValueError, TypeError):
            ncut = -1
        npz = [p for p in (os.path.join(root, cell, 'forced_%s_%s_%s.npz' % (s_, b, st)) for s_ in ('10pc', 'HWJ'))
               if os.path.exists(p)]
        for smp in ('10pc', 'HWJ'):
            if any(('/%s/%s/' % (smp, b)) in f and f.endswith('_%s.fits' % st) for f in man) and \
                    not os.path.exists(os.path.join(root, cell, 'forced_%s_%s_%s.npz' % (smp, b, st))):
                bad.append('measured sample without its stage-1 file %s %s %s %s' % (cell, smp, b, st))
        if not npz:
            if ncut > 4:
                bad.append('searched band without stage 1 %s %s %s (cutout cubes %d)' % (cell, b, st, ncut))
            else:
                nnocut += 1
            continue
        t_ri = os.path.getmtime(ri)
        for p in npz:
            nfile += 1
            if os.path.getmtime(p) < t_ri:
                bad.append('stage 1 older than its search (superseded cutouts) %s' % os.path.relpath(p, root))
    return bad, nband, nfile, nnocut


def _snapshot_eval(ev, ctx):
    c = _cfg()
    e = c.photometry_evaluate
    res = {}
    try:
        bad, ncls = snapshot_grid_check(ev)
        res['G'] = (not bad, len(bad), _fmt(bad) if bad else '%d classes, grid %.2f-%.2f (%d values each), counts, rates, '
                    'thresholds, note, grid top and exceedance columns recomputed' % (ncls, _grid()[0], _grid()[-1],
                                                                                      len(_grid())))
    except Exception as x:      # noqa: BLE001
        res['G'] = (False, None, 'grid files unreadable: %s' % x)
    th = read_csv(os.path.join(ev, 'thresholds.csv'))[1] if os.path.exists(os.path.join(ev, 'thresholds.csv')) else []
    n = len(th)
    res['V2'] = (n == e.THRESHOLD_ROWS['snapshot'], n, 'thresholds.csv rows %d (want %d)' % (n, e.THRESHOLD_ROWS['snapshot']))
    cells = read_csv(os.path.join(ev, 'cells.csv'))[1] if os.path.exists(os.path.join(ev, 'cells.csv')) else []
    kinds = {r['kind'] for r in cells}
    T = read_csv(os.path.join(ev, 'targets_ge4.csv'))[1] if os.path.exists(os.path.join(ev, 'targets_ge4.csv')) else []
    bad = []
    if len(kinds) != 1:
        bad.append('cells.csv kinds %s' % sorted(kinds))
    kind = next(iter(kinds)) if kinds else '?'
    excl = _excl()
    seen = {}
    for r in T:
        b = r['band']
        if b == 'Full':
            bad.append('Full row %s %s' % (r['cell'], r['target']))
        elif (kind == 'sb') != ('MHz' in b):
            bad.append('%s band %s in a %s evaluation' % (r['cell'], b, kind))
        k = (r['cell'], b)
        if k not in seen:
            lst, date = r['cell'].split('_', 1)
            seen[k] = excl.is_excluded(lst, b, date)
        if seen[k]:
            bad.append('%s %s is excluded under core.exclusions' % k)
    bad = sorted(set(bad))
    res['V1'] = (not bad, len(bad), _fmt(bad) if bad else 'kind %s: %d rows, no Full, no band of the other kind, no excluded '
                 '(cell, band) of %d (core.exclusions)' % (kind, len(T), len(seen)))
    stage1 = _stage1_root(ev, ctx)
    procs = ctx.get('procs') or e.CONTENT_TEST_PROCS
    try:
        rows, ncell, nall = zone_rows(ev, stage1, ctx.get('max_cells'), procs)
        diff = ['%s: file %s recomputed %s' % (' '.join(k), a, b) for k, a, b in rows if a != b]
        nz = sum(1 for _, _, b in rows if b == 'zone')
        if not T:       # test_snapshot_forced_reserve.cmd_zones :215-216 refuses an evaluation without targets_ge4 rows
            res['Z1'] = (False, 0, 'no targets_ge4.csv rows to recompute (the production zone test refuses this)')
        else:
            res['Z1'] = (not diff, len(diff), _fmt(diff) if diff else '%d rows of %d cells%s recomputed (%d zone, %d clean): '
                         'all equal' % (len(rows), ncell, '' if ncell == nall else ' (of %d)' % nall, nz, len(rows) - nz))
    except Exception as x:      # noqa: BLE001
        res['Z1'] = (False, None, 'zone recomputation failed: %s' % x)
    tasks = [(r['cell'], r['kind'], stage1, c.paths.SNAPSHOT_FLEET_OUT) for r in cells]
    if ctx.get('max_cells'):
        step = max(1, len(tasks) // ctx['max_cells'])
        tasks = tasks[::step][:ctx['max_cells']]
    if not tasks:
        res['S'] = (False, 0, 'no cells.csv rows')
    else:
        out = _pool_map(stage1_cell, tasks, procs)
        bad = [b for o in out for b in o[0]]
        res['S'] = (not bad, len(bad), _fmt(bad) if bad else '%d evaluated cells: %d searched (band, Stokes), %d stage-1 files '
                    'all newer than their search, %d with no target cutout' % (len(tasks), sum(o[1] for o in out),
                                                                                sum(o[2] for o in out),
                                                                                sum(o[3] for o in out)))
    return res


def _eval_tier(ev):
    if os.path.exists(os.path.join(ev, 'thresholds_grid.csv')):
        return 'snapshot'
    if os.path.exists(os.path.join(ev, 'targets_above_threshold.csv')):
        return 'deep'
    return None


def _eval(path, ctx):
    tier = _eval_tier(path)
    if tier == 'deep':
        return 'deep', _cached('deep_eval', os.path.join(path, 'thresholds.csv'), lambda: _deep_eval(path))
    if tier == 'snapshot':
        key = 'snap_eval:%s:%s' % (ctx.get('max_cells'), ctx.get('stage1'))
        return 'snapshot', _cached(key, os.path.join(path, 'thresholds.csv'), lambda: _snapshot_eval(path, ctx))
    return None, {}


def _eval_result(path, ctx, deep_key, snap_key):
    tier, res = _eval(path, ctx)
    if tier is None:
        return False, None, 'not an evaluation directory (no thresholds_grid.csv / targets_above_threshold.csv)'
    k = deep_key if tier == 'deep' else snap_key
    if k is None:
        return True, None, 'not applicable to a %s evaluation' % tier
    return res[k]


@content_test('photometry_eval', 'E1_thresholds_or_G_grid')
def ev1(path, **ctx):
    return _eval_result(path, ctx, 'E1', 'G')


@content_test('photometry_eval', 'E2_lists_or_V1_kind_exclusions')
def ev2(path, **ctx):
    return _eval_result(path, ctx, 'E2', 'V1')


@content_test('photometry_eval', 'E3_exceedance_or_V2_rows')
def ev3(path, **ctx):
    return _eval_result(path, ctx, 'E3', 'V2')


@content_test('photometry_eval', 'Z1_zones')
def ev4(path, **ctx):
    return _eval_result(path, ctx, None, 'Z1')


@content_test('photometry_eval', 'S_stage1')
def ev5(path, **ctx):
    return _eval_result(path, ctx, None, 'S')


@content_test('photometry_eval', 'E4_plot')
def ev6(path, **ctx):
    """Display product (spec 6: existence and size only, recorded, never gating)."""
    if _eval_tier(path) != 'deep':
        return True, None, 'not applicable'
    p = os.path.join(path, 'exceedance.png')
    sz = os.path.getsize(p) if os.path.exists(p) else 0
    return True, sz, 'exceedance.png %s' % ('%d bytes' % sz if sz else 'ABSENT (display product: recorded, not gating)')


# ---------------------------------------------------------------------------------------------------------------------
# photometry_snapshot_npz, photometry_manifest
# ---------------------------------------------------------------------------------------------------------------------

_NPZ_RE = re.compile(r'^forced_(?P<smp>[^_]+)_(?P<band>[^_]+)_(?P<st>[IV])\.npz$')


def _npz(path):
    import numpy as np

    def load():
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    return _cached('npz', path, load)


@content_test('photometry_snapshot_npz', 'N1_shapes')
def n1(path, **ctx):
    d = _npz(path)
    miss = [k for k in NPZ_KEYS if k not in d]
    if miss:
        return False, None, 'arrays missing: %s' % miss
    T, F, B = len(d['names']), len(d['ts']), len(d['bts'])
    nc = _cfg().photometry_measure.N_CONTROLS + 1
    bad = []
    for k, want in (('ra', (T,)), ('dec', (T,)), ('ring_beams', (T,)), ('admitted', (F,)), ('val10', (T, nc, F)),
                    ('rms10', (T, nc, F)), ('bn', (B,)), ('val1', (T, nc, B)), ('rms1', (T, nc, B))):
        if tuple(d[k].shape) != want:
            bad.append('%s shape %s != %s' % (k, tuple(d[k].shape), want))
    if T == 0:
        bad.append('no target')
    if bad:
        return False, len(bad), _fmt(bad)
    return True, T, '%d targets x %d positions x %d frames, %d 1-minute blocks' % (T, nc, F, B)


@content_test('photometry_snapshot_npz', 'N2_blocks')
def n2(path, **ctx):
    d = _npz(path)
    m = _cfg().photometry_measure
    if m.MINUTE_KEY['snapshot'] != 'file-name':
        return False, None, 'MINUTE_KEY %r not implemented by this test' % m.MINUTE_KEY['snapshot']
    ts = [str(t) for t in d['ts']]
    adm = [bool(a) for a in d['admitted']]
    mins = {}
    for t, a in zip(ts, adm):
        if a:
            mins.setdefault(t[:13], 0)
            mins[t[:13]] += 1
    want = [(k + '00', n) for k, n in mins.items() if n >= m.BLOCK_MIN_FRAMES['snapshot']]
    got = list(zip([str(b) for b in d['bts']], [int(n) for n in d['bn']]))
    if got != want:
        return False, len(got), '1-minute blocks %d (first %s) != recomputed %d (first %s)' % (
            len(got), got[:2], len(want), want[:2])
    return True, len(got), '%d blocks = the file-name minutes of admitted frames with >= %d members' % (
        len(got), m.BLOCK_MIN_FRAMES['snapshot'])


@content_test('photometry_snapshot_npz', 'N3_ring')
def n3(path, **ctx):
    import numpy as np
    d = _npz(path)
    rb = np.asarray(d['ring_beams'], float)
    lim = _cfg().photometry_measure.RING_BEAMS
    bad = ~((rb > 0) & (rb <= lim + 1e-9))
    if bad.any():
        return False, int(bad.sum()), 'ring radius outside (0, %g] beams for %d targets' % (lim, int(bad.sum()))
    return True, float(rb.min()), 'ring %.2f-%.2f beams (<= %g)' % (rb.min(), rb.max(), lim)


def _cutout_for(root, smp, band, name, cell, st):
    lst, date = cell.split('_', 1)
    pat = os.path.join(root, smp, band, name, '%s_%s_%sT*_%s.fits' % (name, lst, date.replace('-', ''), st))
    return sorted(glob.glob(pat))


def _snap_measure(cube, x, y, stamp, core, minfrac, madf):
    """Target value and stamp rms in every plane (snapshot_forced_phot.measure :79-104 for one position, written out)."""
    import numpy as np
    N, ny, nx = cube.shape
    h = stamp // 2
    dy, dx = np.mgrid[-h:h + 1, -h:h + 1]
    c2 = core // 2
    sel = ~((np.abs(dy) <= c2) & (np.abs(dx) <= c2))
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    fx, fy = x - x0, y - y0
    xi, yi = int(round(x)), int(round(y))
    if x0 < 0 or y0 < 0 or x0 + 1 >= nx or y0 + 1 >= ny:
        return None
    ya, yb, xa, xb = max(yi - h, 0), min(yi + h + 1, ny), max(xi - h, 0), min(xi + h + 1, nx)
    sl = sel[ya - (yi - h):yb - (yi - h), xa - (xi - h):xb - (xi - h)]
    if sl.sum() < minfrac * sel.sum():
        return None
    val = ((1 - fx) * (1 - fy) * cube[:, y0, x0] + fx * (1 - fy) * cube[:, y0, x0 + 1]
           + (1 - fx) * fy * cube[:, y0 + 1, x0] + fx * fy * cube[:, y0 + 1, x0 + 1])
    st = cube[:, ya:yb, xa:xb].reshape(N, -1)[:, sl.ravel()]
    med = np.nanmedian(st, axis=1)
    rms = madf * np.nanmedian(np.abs(st - med[:, None]), axis=1)
    return val, rms


@content_test('photometry_snapshot_npz', 'N4_remeasure')
def n4(path, cutout_root=None, **ctx):
    import numpy as np
    from astropy.io import fits
    c = _cfg()
    m = c.photometry_measure
    mt = _NPZ_RE.match(os.path.basename(path))
    if not mt:
        return False, None, 'file name is not forced_<sample>_<band>_<S>.npz'
    cell = os.path.basename(os.path.dirname(os.path.abspath(path)))
    root = cutout_root or os.path.join(c.paths.PHASE3_ARCHIVE_ROOT, m.CUTOUT_SUBDIR)
    d = _npz(path)
    names = [str(n) for n in d['names']]
    T, F = len(names), len(d['ts'])
    if not T or not F:
        return False, None, 'empty npz'
    pick = sorted(set(np.linspace(0, T - 1, min(m.N4_TARGETS, T)).astype(int)))
    fr = sorted(set(np.linspace(0, F - 1, min(m.N4_FRAMES, F)).astype(int)))
    bad, done, notfound = [], 0, []
    for i in pick:
        cp = _cutout_for(root, mt['smp'], mt['band'], names[i], cell, mt['st'])
        if len(cp) != 1:
            notfound.append('%s (%d cutouts)' % (names[i], len(cp)))
            continue
        with fits.open(cp[0], memmap=True) as h:
            tsc = [t if isinstance(t, str) else t.decode() for t in h['TIMES'].data['ts']]
            if tsc != [str(t) for t in d['ts']]:
                bad.append('%s: the cutout\'s frame list differs from the npz (cutout rewritten since)' % names[i])
                continue
            tx = float(np.nanmedian(h['TIMES'].data['target_x']))
            ty = float(np.nanmedian(h['TIMES'].data['target_y']))
            cube = np.asarray(h[0].data[fr], np.float32)
        got = _snap_measure(cube, tx, ty, m.STAMP_PX, m.STAMP_CORE_PX, m.MIN_STAMP_FRAC['snapshot'], m.RMS_MAD_FACTOR)
        v_npz, r_npz = d['val10'][i, 0, fr], d['rms10'][i, 0, fr]
        if got is None:
            if np.isfinite(v_npz).any():
                bad.append('%s: no measurement possible at the target, npz has values' % names[i])
            continue
        v, r = got
        for a, b, what in ((v, v_npz, 'value'), (r, r_npz, 'rms')):
            a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
            same = (np.isnan(a) & np.isnan(b)) | (np.abs(a - b) <= 1e-5 * np.maximum(np.abs(a), np.abs(b)) + 1e-12)
            if not same.all():
                bad.append('%s %s differs at %d of %d frames' % (names[i], what, int((~same).sum()), len(fr)))
        done += 1
    if bad:
        return False, len(bad), _fmt(bad)
    if not done:
        return False, 0, 'no target re-measurable: cutouts not found %s under %s' % (_fmt(notfound), root)
    return True, done, '%d targets x %d frames re-measured from their cutouts: value and stamp rms equal (1e-5)' % (
        done, len(fr)) + ('; not found: %s' % _fmt(notfound) if notfound else '')


@content_test('photometry_snapshot_npz', 'N5_names')
def n5(path, **ctx):
    c = _cfg()
    mt = _NPZ_RE.match(os.path.basename(path))
    if not mt:
        return False, None, 'file name is not forced_<sample>_<band>_<S>.npz'
    bad = []
    if mt['band'] not in c.photometry.SNAPSHOT_BANDS:
        bad.append('band %s not measured (photometry.SNAPSHOT_BANDS)' % mt['band'])
    if mt['smp'] not in c.photometry.SAMPLES_MEASURED:
        bad.append('sample %s not measured' % mt['smp'])
    cur = current_safe_names()
    names = [str(n) for n in _npz(path)['names']]
    off = [n for n in names if (mt['smp'], n) not in cur]
    if off:
        bad.append('%d names not in the current %s list: %s' % (len(off), mt['smp'], _fmt(off)))
    if len(set(names)) != len(names):
        bad.append('duplicate names')
    if bad:
        return False, len(bad), _fmt(bad)
    return True, len(names), 'name, band and sample measured; every target in the current list'


def _manifest_rows(path):
    with open(path, newline='', encoding='utf-8') as f:
        rd = csv.DictReader(f)
        return list(rd.fieldnames or []), list(rd)


@content_test('photometry_manifest', 'M1_statuses')
def mm1(path, **ctx):
    cols, rows = _manifest_rows(path)
    hdr = _cfg().photometry_measure.MANIFEST_HEADER.split(',')
    if cols != hdr:
        return False, None, 'header %s != %s' % (cols, hdr)
    bad = [r['status'] for r in rows if not MANIFEST_STATUS_RE.match(r['status'] or '')]
    if bad:
        return False, len(bad), 'unknown statuses: %s' % _fmt(sorted(set(bad)))
    return True, len(rows), '%d rows, statuses ok / skipped (not in the current list) / REFUSED' % len(rows)


@content_test('photometry_manifest', 'M2_completion')
def mm2(path, **ctx):
    _, rows = _manifest_rows(path)
    ref = [r for r in rows if (r['status'] or '').startswith('REFUSED')]
    ok = [r for r in rows if r['status'] == 'ok']
    if ref:
        return False, len(ref), '%d REFUSED rows (e.g. %s: %s)' % (len(ref), os.path.basename(ref[0]['file']), ref[0]['status'])
    if not ok:
        return False, 0, 'no ok row'
    return True, len(ok), '%d cutouts measured, none refused' % len(ok)


@content_test('photometry_manifest', 'M3_coverage')
def mm3(path, **ctx):
    _, rows = _manifest_rows(path)
    groups, bands = {}, set()
    for r in rows:
        parts = r['file'].split('/')
        if len(parts) < 4:
            continue
        smp, band, obj, fn = parts[-4], parts[-3], parts[-2], parts[-1]
        bands.add(band)
        if r['status'] == 'ok':
            groups.setdefault((smp, band, fn[-6]), set()).add(obj)
    d = os.path.dirname(os.path.abspath(path))
    bad = []
    for (smp, band, st), objs in sorted(groups.items()):
        p = os.path.join(d, 'forced_%s_%s_%s.npz' % (smp, band, st))
        if not os.path.exists(p):
            bad.append('no %s for %d measured cutouts' % (os.path.basename(p), len(objs)))
            continue
        names = {str(n) for n in _npz(p)['names']}
        if names != objs:
            bad.append('%s: %d names vs %d ok cutouts' % (os.path.basename(p), len(names), len(objs)))
    for p in sorted(glob.glob(os.path.join(d, 'forced_*.npz'))):
        mt = _NPZ_RE.match(os.path.basename(p))
        if mt and mt['band'] in bands and (mt['smp'], mt['band'], mt['st']) not in groups:
            bad.append('%s without an ok manifest row (stale file?)' % os.path.basename(p))
    if bad:
        return False, len(bad), _fmt(bad)
    return True, len(groups), '%d (sample, band, Stokes) groups: npz names = the ok cutouts' % len(groups)


# ---------------------------------------------------------------------------------------------------------------------
# photometry_cutout (input of the snapshot measure; written by search --tier snapshot)
# ---------------------------------------------------------------------------------------------------------------------

_CUT_RE = re.compile(r'^(?P<obj>.+)_(?P<lst>\d\dh)_(?P<ymd>\d{8})T(?P<hms>\d{6})_(?P<st>[IV])\.fits$')


@content_test('photometry_cutout', 'C1_C4_cube')
def c1(path, **ctx):
    import numpy as np
    from astropy.io import fits
    cc = _cfg().photometry_cutouts
    mt = _CUT_RE.match(os.path.basename(path))
    if not mt:
        return False, None, 'file name is not <object>_<LST>_<YYYYMMDD>T<HHMMSS>_<S>.fits'
    parts = os.path.abspath(path).split('/')
    smp, band, objd = parts[-4], parts[-3], parts[-2]
    bad = []
    with fits.open(path, memmap=True) as h:
        H = h[0].header
        shp = h[0].data.shape if h[0].data is not None else ()
        if len(shp) != 3 or shp[1] != shp[2]:
            bad.append('C1 cube shape %s' % (shp,))
        else:
            n = int(round(cc.CUT_DEG / abs(float(H['CDELT2']))))
            if shp[1] != n:
                bad.append('C1 side %d px != round(%g deg / CDELT2) = %d' % (shp[1], cc.CUT_DEG, n))
        if 'TIMES' not in h:
            bad.append('C2 no TIMES table')
        else:
            T = h['TIMES'].data
            cols = tuple(h['TIMES'].columns.names)
            if cols != tuple(cc.TIMES_COLUMNS):
                bad.append('C2 TIMES columns %s' % (cols,))
            elif shp and len(T) != shp[0]:
                bad.append('C2 TIMES rows %d != frames %d' % (len(T), shp[0]))
            else:
                tx, ty = float(np.nanmedian(T['target_x'])), float(np.nanmedian(T['target_y']))
                if len(shp) == 3 and not (0 <= tx <= shp[2] - 1 and 0 <= ty <= shp[1] - 1):
                    bad.append('C4 median target (%.1f, %.1f) outside the box' % (tx, ty))
        miss = [k for k in cc.HEADER_CARDS if k not in H]
        if miss:
            bad.append('C3 header cards missing %s' % miss)
        else:
            want = {'SAMPLE': smp, 'BAND': band, 'STOKES': mt['st'], 'LST': mt['lst']}
            for k, v in want.items():
                if str(H[k]) != v:
                    bad.append('C3 %s %s != path %s' % (k, H[k], v))
            if safe_name(str(H['OBJECT'])) != objd or str(H['CELL']) != '%s_%s' % (mt['lst'], str(H['DATE'])):
                bad.append('C3 OBJECT/CELL %s/%s vs path %s' % (H['OBJECT'], H['CELL'], objd))
    if bad:
        return False, len(bad), _fmt(bad)
    return True, shp[0] if shp else 0, 'cube %s, TIMES %d rows x 8 columns, header = path, target inside' % (shp, shp[0])


# ---------------------------------------------------------------------------------------------------------------------
# photometry_zone_flags
# ---------------------------------------------------------------------------------------------------------------------

@content_test('photometry_zone_flags', 'ZF1_rows')
def zf1(path, **ctx):
    cols, rows = read_csv(path)
    if tuple(cols) != ZONE_FLAG_COLUMNS:
        return False, None, 'columns %s' % cols
    keys = [tuple(r[k] for k in ZONE_FLAG_COLUMNS[:6]) for r in rows]
    bad = []
    if len(set(keys)) != len(keys):
        bad.append('%d duplicate rows' % (len(keys) - len(set(keys))))
    diff = [r for r in rows if r['screen_file'] != r['screen_recomputed']
            or r['screen_file'] not in ('zone', 'clean')]
    if diff:
        bad.append('%d rows whose recomputed screen differs (e.g. %s)' % (len(diff), ' '.join(diff[0][k] for k in
                                                                                           ZONE_FLAG_COLUMNS[:6])))
    if bad:
        return False, len(bad), _fmt(bad)
    return True, len(rows), '%d rows, every recomputed screen equals the evaluation\'s' % len(rows) + (
        ' (no listed target row)' if not rows else '')


def write_zone_flags(path, rows):
    """Write the recomputed screens (Z1) as the production flags file (test_snapshot_forced_reserve.cmd_zones :225-239):
    one row per key, the LAST of the rows in evaluation order (sb then wb, cells sorted) - production's dict 'seen' - and
    the keys sorted."""
    seen = {}
    for k, a, b in rows:
        seen[tuple(k)] = (a, b)
    tmp = '%s.tmp.%d' % (path, os.getpid())
    with open(tmp, 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(list(ZONE_FLAG_COLUMNS))
        for k, v in sorted(seen.items()):
            wr.writerow(list(k) + list(v))
    os.replace(tmp, path)
    return path

