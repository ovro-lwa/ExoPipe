"""Content tests of the snapshot frames products (unit frames-tiers; spec plan/specs/frames.md section 6).

Classes (each reads the product's CONTENT; context keys are optional, so `exopipe3 verify <class> <files>` and --resume
re-tests run the same checks from the path alone):
  snapshot_stage   a staged subband directory <stage>/<cell>/<sb>/: the staged *-image.fits names equal the discovery list
                   (.filelist, decompressed names) per Stokes; no .fs/.fz/.tmp left; every staged header readable with
                   NAXIS = 2+ and its file name's subband
  snapshot_regrid  a regridded subband directory: every raw frame on its own Master's grid (NAXIS1/2, CRPIX, CDELT, CRVAL equal
                   the Master's); frames carrying REGRID have REGRIDN = their source NAXIS1 != the Master's
  snapshot_cube    <cell>_<sb>_<S>_flowcube.fits: PRODUCT snapshot-flowcube-v1, CELL/SUBBAND/STOKES = the name, FLOWED and
                   not NOMASTER, NFRAMES = FRAME_TABLE rows, timestamps unique and sorted, FRAMES/MASK shaped (n, ny, nx), the
                   FLOW extension decodes for every frame (finite field where has_flow, null elsewhere), listed in the cell's
                   MANIFEST.txt with copy-OK and '-> OK' (md5 compared when the writer passes it);
                   MANIFEST.txt: one line per cube in the directory, every one copy-OK and -> OK, last line COMPLETE with
                   cubes = 2 x subbands = the cube files present
  frame_movie      frames_<cell>_<band>_I.mp4: non-empty, ffprobe decodes >= 1 frame; movies.log: a TOTAL line, and the
                   bands it lists (or ctx bands) each have their mp4
  flow_movie       flow_<sb>.mp4: decodes >= 1 frame; flow_<sb>_series.csv beside it with the 4 production columns and one
                   row per flow field of the subband (ctx flow_dir, else paths.SNAPSHOT_FLOW_ROOT/<cell>)
Not tested stand-alone (needs the staging): the cube pixels against the flowed frames - that is the writer's own
`phase3_snapshot_cube.py verify` (run at write time, its VERIFY line is in MANIFEST.txt and checked here).
"""
import csv
import glob
import os
import re
import subprocess

import numpy as np

from exopipe.phase3.core.validate import content_test

_FRAME_RE = re.compile(r'^(?P<sb>\d+MHz)-(?P<kind>Clean|Pilot)-Snapshot-(?P<ts>\d{8}_\d{6})(?:-(?P<st>[IV]))?-image\.fits$')
_CUBE_RE = re.compile(r'^(?P<cell>\d\dh_\d{4}-\d\d-\d\d)_(?P<sb>\d+MHz)_(?P<S>[IV])_flowcube\.fits$')
_FMOVIE_RE = re.compile(r'^frames_(?P<cell>\d\dh_\d{4}-\d\d-\d\d)_(?P<band>\d+MHz|Red|Green|Blue|Full)_I\.mp4$')
_FLOWMOVIE_RE = re.compile(r'^flow_(?P<sb>\d+MHz)\.mp4$')


def _cfg():
    from exopipe.phase3.config import CFG
    return CFG


def _cards(path):
    from exopipe.phase3.core.record import fits_primary_cards
    try:
        return fits_primary_cards(path)
    except OSError:
        return {}


def _fail(msgs, value=None, ok_detail='ok'):
    return (not msgs), (len(msgs) if value is None else value), ('; '.join(msgs[:6]) if msgs else ok_detail)


def _stokes(name):
    m = _FRAME_RE.match(name)
    return None if not m else (m.group('st') or ('I' if m.group('kind') == 'Clean' else 'V'))


def _decompressed(name):
    for sfx in ('.fs', '.fz'):
        if name.endswith(sfx):
            return name[:-len(sfx)]
    return name


def _ffprobe():
    """ffprobe beside the ffmpeg the movie engines use (phase3_snapshot_frame_movie.py FFMPEG default: the site 'default'
    interpreter's environment)."""
    py = (_cfg().site.PYTHON or {}).get('default', '')
    p = os.path.join(os.path.dirname(py), 'ffprobe')
    return p if os.access(p, os.X_OK) else 'ffprobe'


def _decoded_frames(path):
    r = subprocess.run([_ffprobe(), '-v', 'error', '-count_frames', '-select_streams', 'v:0', '-show_entries',
                        'stream=nb_read_frames', '-of', 'csv=p=0', path], capture_output=True, text=True, timeout=600)
    try:
        return int(r.stdout.strip().splitlines()[0])
    except (ValueError, IndexError):
        return 0


# ---------------------------------------------------------------------------------------------------------------------
# snapshot_stage
# ---------------------------------------------------------------------------------------------------------------------

@content_test('snapshot_stage')
def stage_set(path, **ctx):
    """Staged set = discovery set per Stokes (.filelist, decompressed names); no leftovers; headers readable, 2-D+, and the
    subband of the file name = the directory."""
    sb = os.path.basename(path.rstrip('/'))
    fl = os.path.join(path, '.filelist')
    if not os.path.isfile(fl):
        return False, None, f'no .filelist in {path}'
    want = sorted(_decompressed(os.path.basename(l.strip())) for l in open(fl) if l.strip())
    have = sorted(f for f in os.listdir(path) if f.endswith('-image.fits'))
    left = sorted(f for f in os.listdir(path) if f.endswith(('.fs', '.fz', '.tmp')))
    msgs = []
    for st in ('I', 'V'):
        w = {x for x in want if _stokes(x) == st}
        h = {x for x in have if _stokes(x) == st}
        if w != h:
            msgs.append(f'{st}: staged {len(h)} != listed {len(w)} (missing {sorted(w - h)[:2]}, extra {sorted(h - w)[:2]})')
    if left:
        msgs.append(f'{len(left)} compressed/tmp leftovers, e.g. {left[0]}')
    if not have:
        msgs.append('no staged frame')
    bad = []
    for f in have:
        c = _cards(os.path.join(path, f))
        if not c or int(c.get('NAXIS', 0) or 0) < 2 or not f.startswith(sb + '-'):
            bad.append(f)
    if bad:
        msgs.append(f'{len(bad)} frame(s) unreadable, not 2-D or of another subband, e.g. {bad[0]}')
    idx = sum(1 for x in want if re.search(r'-t\d{4}-image\.fits$', x))
    return _fail(msgs, value=len(have), ok_detail=f'{len(have)} frames = .filelist' + (f'; {idx} index-named' if idx else ''))


# ---------------------------------------------------------------------------------------------------------------------
# snapshot_regrid
# ---------------------------------------------------------------------------------------------------------------------

def _master_of(path):
    """The own Master of a staged subband directory <stage>/<LST>_<date>/<sb> (masters_snapshot.LINK_ROOT)."""
    sb = os.path.basename(path.rstrip('/'))
    cell = os.path.basename(os.path.dirname(path.rstrip('/')))
    lst = cell.split('_')[0]
    C = _cfg()
    return C.frames_snapshot.MASTER_FILE.format(links=C.masters_snapshot.LINK_ROOT, lst=lst, sb=sb)


@content_test('snapshot_regrid')
def on_master_grid(path, master=None, **ctx):
    """Every raw staged frame has the own Master's NAXIS1/2, CRPIX1/2, CDELT1/2, CRVAL1/2; a frame with REGRID has
    REGRIDN (its source NAXIS1) != the Master's NAXIS1 and a REGRIDCD card."""
    m = master or _master_of(path)
    mc = _cards(m)
    if not mc:
        return False, None, f'own Master {m} unreadable'
    keys = ('NAXIS1', 'NAXIS2', 'CRPIX1', 'CRPIX2', 'CDELT1', 'CDELT2', 'CRVAL1', 'CRVAL2')
    frames = sorted(f for f in os.listdir(path) if f.endswith('-image.fits'))
    off, badcard, nreg = [], [], 0
    for f in frames:
        c = _cards(os.path.join(path, f))
        diff = [k for k in keys if not (k in c and k in mc and np.isclose(float(c[k]), float(mc[k]), rtol=0, atol=1e-9))]
        if diff:
            off.append(f'{f} ({diff[0]} {c.get(diff[0])} vs {mc.get(diff[0])})')
        if c.get('REGRID'):
            nreg += 1
            if int(c.get('REGRIDN', mc['NAXIS1'])) == int(mc['NAXIS1']) or 'REGRIDCD' not in c:
                badcard.append(f)
    msgs = []
    if not frames:
        msgs.append('no staged frame')
    if off:
        msgs.append(f'{len(off)} frame(s) off the Master grid, e.g. {off[0]}')
    if badcard:
        msgs.append(f'{len(badcard)} REGRID frame(s) with inconsistent REGRIDN/REGRIDCD, e.g. {badcard[0]}')
    return _fail(msgs, value=len(frames), ok_detail=f'{len(frames)} frames on the Master grid ({nreg} regridded)')


# ---------------------------------------------------------------------------------------------------------------------
# snapshot_cube
# ---------------------------------------------------------------------------------------------------------------------

def _manifest_lines(d):
    p = os.path.join(d, 'MANIFEST.txt')
    return [l.rstrip('\n') for l in open(p)] if os.path.exists(p) else None


def _cube_checks(path, md5=None):
    from astropy.io import fits
    b = os.path.basename(path)
    m = _CUBE_RE.match(b)
    if not m:
        return [f'not a flowcube name: {b}'], None
    msgs = []
    with fits.open(path, memmap=True) as h:
        names = [x.name for x in h]
        ph = h[0].header
        for k, want in (('PRODUCT', _cfg().frames_snapshot.CUBE_PRODUCT), ('CELL', m['cell']), ('SUBBAND', m['sb']),
                        ('STOKES', m['S'])):
            if str(ph.get(k, '')).strip() != want:
                msgs.append(f'{k} {ph.get(k)!r} != {want!r}')
        if not ph.get('FLOWED') or ph.get('NOMASTER'):
            msgs.append(f"FLOWED={ph.get('FLOWED')} NOMASTER={ph.get('NOMASTER')} (registered frames only)")
        for need in ('FRAMES', 'MASK', 'FRAME_TABLE'):
            if need not in names:
                msgs.append(f'no {need} extension')
        if msgs:
            return msgs, None
        t = h['FRAME_TABLE'].data
        n = len(t)
        ts = [s.strip() if isinstance(s, str) else s.decode().strip() for s in t['ts']]
        if int(ph.get('NFRAMES', -1)) != n:
            msgs.append(f"NFRAMES {ph.get('NFRAMES')} != {n} table rows")
        if len(set(ts)) != n or ts != sorted(ts):
            msgs.append('timestamps not unique and sorted')
        fh, mh = h['FRAMES'].header, h['MASK'].header
        for nm, hh in (('FRAMES', fh), ('MASK', mh)):
            shp = (hh.get('ZNAXIS3', hh.get('NAXIS3')), hh.get('ZNAXIS2', hh.get('NAXIS2')), hh.get('ZNAXIS1', hh.get('NAXIS1')))
            if shp[0] != n:
                msgs.append(f'{nm} has {shp[0]} planes, table {n}')
        if 'FLOW' in names:
            fl = h['FLOW']
            q = np.asarray(fl.data)
            null = int(fl.header.get('FLOWNULL', -32768))
            has = np.asarray(t['has_flow'], dtype=bool)
            if q.shape[0] != n:
                msgs.append(f'FLOW has {q.shape[0]} frames, table {n}')
            else:
                allnull = np.all(q.reshape(n, -1) == null, axis=1)
                if np.any(has & allnull):
                    msgs.append(f'{int(np.sum(has & allnull))} frame(s) with has_flow but no decoded field')
                if np.any(~has & ~allnull):
                    msgs.append(f'{int(np.sum(~has & ~allnull))} frame(s) without has_flow carry a field')
        elif np.any(np.asarray(t['has_flow'], dtype=bool)):
            msgs.append('has_flow rows but no FLOW extension')
    man = _manifest_lines(os.path.dirname(path))
    if man is None:
        msgs.append('no MANIFEST.txt beside the cube')
    else:
        line = next((l for l in man if l.startswith(b + ' ')), None)
        if not line:
            msgs.append('cube not listed in MANIFEST.txt')
        else:
            if 'copy-OK' not in line or not line.rstrip().endswith('-> OK'):
                msgs.append(f'MANIFEST line not copy-OK / -> OK: {line[:160]}')
            mm = re.search(r'md5=([0-9a-f]{32})', line)
            if md5 and (not mm or mm.group(1) != md5):
                msgs.append(f'md5 {md5} != MANIFEST {mm.group(1) if mm else None}')
    return msgs, n


@content_test('snapshot_cube')
def cube_content(path, md5=None, staged_subbands=None, **ctx):
    """A cube: header, table, MASK, FLOW extension decode, MANIFEST listing (md5 when given). MANIFEST.txt: one copy-OK /
    -> OK line per cube file present, COMPLETE line with cubes = 2 x subbands = the cubes present."""
    if os.path.basename(path) == 'MANIFEST.txt':
        d = os.path.dirname(path)
        lines = _manifest_lines(d)
        cubes = sorted(f for f in os.listdir(d) if _CUBE_RE.match(f))
        listed = [l.split(' ', 1)[0] for l in lines if _CUBE_RE.match(l.split(' ', 1)[0])]
        msgs = []
        if sorted(listed) != cubes:
            msgs.append(f'{len(listed)} listed != {len(cubes)} cube files present')
        bad = [l for l in lines if _CUBE_RE.match(l.split(' ', 1)[0]) and ('copy-OK' not in l or not l.rstrip().endswith('-> OK'))]
        if bad:
            msgs.append(f'{len(bad)} line(s) not copy-OK / -> OK, e.g. {bad[0][:120]}')
        comp = [l for l in lines if l.startswith('COMPLETE ')]
        if not comp or comp[-1] != lines[-1]:
            msgs.append('last line is not COMPLETE')
        else:
            q = re.search(r'cubes=(\d+) subbands=(\d+)', comp[-1])
            if not q or int(q.group(1)) != len(cubes) or int(q.group(1)) != 2 * int(q.group(2)) or \
                    (staged_subbands is not None and int(q.group(2)) != int(staged_subbands)):
                msgs.append(f'COMPLETE counts {q.groups() if q else None} vs {len(cubes)} cubes'
                            + (f', {staged_subbands} staged subbands' if staged_subbands is not None else ''))
        return _fail(msgs, value=len(cubes), ok_detail=f'{len(cubes)} cubes, COMPLETE')
    msgs, n = _cube_checks(path, md5)
    return _fail(msgs, value=n, ok_detail=f'{n} frames; table, MASK, FLOW, MANIFEST consistent')


# ---------------------------------------------------------------------------------------------------------------------
# frame_movie, flow_movie
# ---------------------------------------------------------------------------------------------------------------------

@content_test('frame_movie')
def frame_movie_content(path, bands=None, **ctx):
    """An mp4: non-empty and decodable (ffprobe frame count >= 1). movies.log: a TOTAL line; every band it reports (or ctx
    bands) has its frames_<cell>_<band>_I.mp4 beside it."""
    b = os.path.basename(path)
    if b == 'movies.log':
        lines = open(path, errors='replace').read().splitlines()
        d = os.path.dirname(path)
        cell = os.path.basename(d)
        rep = [re.match(r'^(\S+) (\S+): (\d+) frames, ', l) for l in lines]
        rep = [(m.group(2), int(m.group(3))) for m in rep if m]
        want = list(bands) if bands is not None else [bd for bd, _ in rep]
        msgs = []
        if not any(l.startswith('TOTAL') for l in lines):
            msgs.append('no TOTAL line')
        miss = [bd for bd in want if not os.path.exists(os.path.join(d, f'frames_{cell}_{bd}_I.mp4'))]
        if miss:
            msgs.append(f'bands without their movie: {miss}')
        return _fail(msgs, value=len(want), ok_detail=f'{len(want)} band movies reported, TOTAL present')
    if not _FMOVIE_RE.match(b):
        return False, None, f'not a frame movie name: {b}'
    if os.path.getsize(path) == 0:
        return False, 0, 'empty file'
    n = _decoded_frames(path)
    return n >= 1, n, f'{n} decoded frames'


@content_test('flow_movie')
def flow_movie_content(path, flow_dir=None, **ctx):
    """flow_<sb>.mp4 decodable (>= 1 frame); flow_<sb>_series.csv beside it with columns ts, median_arcmin, p90_arcmin,
    max_arcmin and one row per flow field flow_<sb>_<ts>.npy of the cell (ctx flow_dir, else paths.SNAPSHOT_FLOW_ROOT/<cell>,
    the cell being the parent of the 'flow' directory)."""
    b = os.path.basename(path)
    m = _FLOWMOVIE_RE.match(b)
    if not m:
        return False, None, f'not a flow movie name: {b}'
    sb = m.group('sb')
    msgs = []
    n = _decoded_frames(path) if os.path.getsize(path) else 0
    if n < 1:
        msgs.append('mp4 empty or not decodable')
    ser = os.path.join(os.path.dirname(path), f'flow_{sb}_series.csv')
    if not os.path.exists(ser):
        msgs.append(f'no {os.path.basename(ser)}')
        return _fail(msgs, value=n)
    with open(ser, newline='') as f:
        rd = csv.reader(f)
        head = next(rd, [])
        rows = list(rd)
    if head != ['ts', 'median_arcmin', 'p90_arcmin', 'max_arcmin']:
        msgs.append(f'series columns {head}')
    fd = flow_dir
    if fd is None:
        d = os.path.dirname(os.path.abspath(path))
        cell = os.path.basename(os.path.dirname(d)) if os.path.basename(d) == 'flow' else None
        fd = os.path.join(_cfg().paths.SNAPSHOT_FLOW_ROOT, cell) if cell else None
    if fd and os.path.isdir(fd):
        nf = len(glob.glob(os.path.join(fd, f'flow_{sb}_*.npy')))
        if nf != len(rows):
            msgs.append(f'{len(rows)} series rows != {nf} flow fields in {fd}')
    return _fail(msgs, value=n, ok_detail=f'{n} decoded frames; {len(rows)} series rows')
