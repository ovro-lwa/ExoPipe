#!/usr/bin/env python
"""Writer safety of the 10-s cutouts stage (lib/photometry/steps_cutouts.py, core/fitsio.py, core/record.py;
FINDING-20261122-CUTOUT-WRITER-DISCARDED-PAGES-PASS-AS-DONE: a Lustre eviction discarded the writer's dirty pages, cutouts were
left cut short at 24 MiB and the unit passed as done; a FAILED log line lost to EIO let a job be skipped silently).

The step's unit (steps_cutouts.snapshot_unit, through run_unit as the stage calls it) runs in this process on synthetic inputs,
its engines FAKES in a stub code root (an importable phase3_snapshot_search.py with the subband engine's exclusion reader and
the writer's time functions, phase3_snapshot_group_search.py, and phase3_snapshot_cutouts.py with the writer's target list
and elevation test) under this interpreter, the group search's deep-rule reader a stub production package
(exopipe/phase3/coadd_full_pipeline.py under photometry_cutouts.GROUP_EXCLUSION_ROOT); the exclusion list is a synthetic
data_decisions.csv, the flow cubes carry FRAMES and FRAME_TABLE (frame headers with a SIN WCS). Asserts:
  - fitsio.fits_implied_size / settle_fits on a whole file, a file cut short, bytes beyond its last HDU, a sparse file (hole);
  - C-WRITE: a staged cutout cut short by N bytes, or an fsync that raises OSError, refuses the unit, nothing placed;
  - C-COUNT: a done line 'V 3 I 3' with V files only refuses; 'V 2 I 2' with no file refuses (it was C-NOCUT, no input);
    'V 2 I 2' with 2 + 2 files is done; list entries sharing one file name (the writer writes that file once per entry in
    its view and counts each): a file credited with its entries that pass the writer's view test (elevation, pixel range) -
    a group straddling the elevation limit (1 of 2 in view) with 1 file for 1 counted passes, the same with another file lost
    refuses, a k=3 group in view with its file lost refuses, an entry off the grid is not credited, frames whose geometries
    disagree on the count refuse; the wideband's 'None' (group not built) refuses C-ENGINE;
  - exclusion BEFORE the engine, one fresh read per unit (verdict and resume identity): an excluded subband cell-band ends
    C-EXCLUDED and the engine never ran; a wideband with an excluded member other than 18 MHz the same; with only its 18 MHz
    member excluded the wideband engine runs; any of the readers (the subband engine's, the group search's) disagreeing
    (PHOT-EXCL-READERS) or reading another file (PHOT-EXCL-FILE) refuses, the engine never ran; one probe call serves every
    unit of a stage, a cell is probed again when data_decisions changed; the probe runs with the ad hoc environment;
  - durable records: atomic_write_json(durable=True) fsyncs (file and directory), the default does not; a record write that
    raises OSError, or a ValueError of a record that cannot be serialized, ends the unit 'error' (exit 1), leaves NO record
    (an earlier done record removed), and a resumed run re-runs the unit.
The flux-scale gate, its re-check and the generation checks (C-FLUXSCALE, C-GENCHECK) are stubbed to pass here (the synthetic
cubes carry no FSAPPLD): tests/test_photometry_cube_gate.py and tests/test_photometry.py test them.
Scratch: a fresh directory under $EXOPIPE3_TEST_SCRATCH, else the system temp directory, removed at exit.
Run: python tests/test_photometry_writer_safety.py
"""
import argparse
import csv
import errno
import json
import os
import shutil
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')

import numpy as np                                                   # noqa: E402
from astropy.io import fits                                          # noqa: E402

from exopipe.phase3.config import CFG                                # noqa: E402
from exopipe.phase3.core import fitsio, record                       # noqa: E402
from exopipe.phase3.lib.photometry import steps_common as K          # noqa: E402
from exopipe.phase3.lib.photometry import steps_cutouts as SC        # noqa: E402
from exopipe.phase3.lib.photometry import tests_cutouts as TC        # noqa: E402

FAILS = []
SAMPLES = ['HWJ']
SUB_CELL, SUB_BAND = '00h_2025-01-05', '55MHz'          # no row
EXCL_CELL = '00h_2025-01-02'                            # 55 MHz excluded
RED_EXCL_CELL = '01h_2025-01-03'                        # 23 MHz (a Red member) excluded
RED_18_CELL = '02h_2025-01-04'                          # 18 MHz only excluded: Red kept (exclusions.SUBBAND_ONLY)
ROWS = [('00h', '55MHz', '2025-01-02', 'exclude', 'synthetic subband row'),
        ('01h', '23MHz', '2025-01-03', 'exclude', 'synthetic Red member row'),
        ('02h', '18MHz', '2025-01-04', 'exclude', 'synthetic 18 MHz row')]

FAKE_SS = r'''"""FAKE phase3_snapshot_search.py of tests/test_photometry_writer_safety.py. Importable: DATA_DECISIONS and excluded_subbands
read the exclusion list as the real reader does (params 'data_decisions'; 'extra_excluded' adds subbands - a disagreeing
reader); ts_to_epoch_seconds / lst_deg_at for the writer's view test. Run with --cutouts-from-cubes: appends its argv to
calls.jsonl, writes params 'write'[S] conforming cutouts T0, T1, ... per Stokes (params 'skip': names not written) into
P3_SNAP_CUTOUT_ROOT, cuts params 'truncate' bytes off the first one, prints the done line with params 'count'."""
import csv, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
P = json.load(open(os.path.join(HERE, 'params.json')))
DATA_DECISIONS = P['data_decisions']


def excluded_subbands(lst, date, path=DATA_DECISIONS):
    with open(path) as f:
        out = {r['subband'].strip() for r in csv.DictReader(f) if r['lst'].strip() == lst and r['date'].strip() == date
               and r['verdict'].strip() == 'exclude'}
    return out | set(P.get('extra_excluded', []))


def ts_to_epoch_seconds(ts):
    return float(ts[-2:])


def lst_deg_at(t):
    return t


def cutout(root, lst, date, band, S, k):
    import numpy as np
    from astropy.io import fits
    obj = 'T%d' % k
    d = os.path.join(root, 'HWJ', band, obj)
    os.makedirs(d, exist_ok=True)
    stamp = date.replace('-', '') + 'T010203'
    nt, n = 3, 5
    h = fits.Header()
    h['CTYPE1'], h['CTYPE2'], h['CDELT1'], h['CDELT2'] = 'RA---SIN', 'DEC--SIN', -1.0, 1.0
    for kk, v in (('OBJECT', obj), ('SAMPLE', 'HWJ'), ('LST', lst), ('DATE', date), ('BAND', band), ('STOKES', S),
                  ('CELL', '%s_%s' % (lst, date)), ('UTCSTART', '01:02:03'), ('TARGRA', 10.0), ('TARGDEC', 20.0), ('CUTDEG', 5.0)):
        h[kk] = v
    cols = [fits.Column(name='ts', format='15A', array=np.array(['%s%02d' % (stamp, i) for i in range(nt)])),
            fits.Column(name='date_obs', format='26A', array=np.array(['%sT01:02:%02d' % (date, i) for i in range(nt)])),
            fits.Column(name='admitted', format='L', array=np.ones(nt, bool)),
            fits.Column(name='global_rms_jyb', format='E', array=np.ones(nt, np.float32)),
            fits.Column(name='ra_deg', format='D', array=np.full(nt, 10.0)),
            fits.Column(name='dec_deg', format='D', array=np.full(nt, 20.0)),
            fits.Column(name='target_x', format='E', array=np.full(nt, 2.0, np.float32)),
            fits.Column(name='target_y', format='E', array=np.full(nt, 2.0, np.float32))]
    p = os.path.join(d, '%s_%s_%s_%s.fits' % (obj, lst, stamp, S))
    data = np.arange(nt * n * n, dtype=np.float32).reshape(nt, n, n) + k
    fits.HDUList([fits.PrimaryHDU(data, header=h), fits.BinTableHDU.from_columns(cols, name='TIMES')]).writeto(p)
    return p


def main():
    import argparse
    ap = argparse.ArgumentParser()
    for k in ('--lst', '--date', '--subband', '--stokes', '--mode', '--out'):
        ap.add_argument(k)
    ap.add_argument('--cutouts-from-cubes', action='store_true')
    a = ap.parse_args()
    with open(os.path.join(HERE, 'calls.jsonl'), 'a') as f:
        f.write(json.dumps(['subband'] + sys.argv[1:]) + '\n')
    root = os.environ['P3_SNAP_CUTOUT_ROOT']
    written = []
    for S in ('V', 'I'):
        written += [cutout(root, a.lst, a.date, a.subband, S, k) for k in range(P['write'][S])
                    if 'T%d' % k not in P.get('skip', [])]
    if P.get('truncate') and written:
        os.truncate(written[0], os.path.getsize(written[0]) - P['truncate'])
    print('2026-11-23 00:00:00,000 INFO CUTOUTS-REGEN %s %s %s: V %s I %s' % (a.lst, a.date, a.subband, P['count']['V'],
                                                                             P['count']['I']), flush=True)


if __name__ == '__main__':
    main()
'''

FAKE_CUT = r'''"""FAKE phase3_snapshot_cutouts.py of tests/test_photometry_writer_safety.py: the writer's target list (params
'targets', [sample, name, ra, dec] rows; under P3_SNAP_CUTOUT_SAMPLES two entries ADHOC/X) and safe_name as the writer's; the
elevation of a target is its declination here, MIN_EL_DEG 30."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
MIN_EL_DEG = 30.0


def _elevation(ra, dec, lst):
    return dec


def safe_name(nm):
    return "".join(ch if (ch.isalnum() or ch in "+-._") else "_" for ch in nm.strip()).strip("_")


def _targets():
    if os.environ.get('P3_SNAP_CUTOUT_SAMPLES'):
        return [('ADHOC', 'X', 1.0, 2.0), ('ADHOC', 'X', 1.0, 2.001)]
    return [tuple(t) for t in json.load(open(os.path.join(HERE, 'params.json')))['targets']]
'''

FAKE_GS = r'''"""FAKE phase3_snapshot_group_search.py of tests/test_photometry_writer_safety.py: appends its argv to calls.jsonl, writes
nothing, prints the wideband done line with params 'wb_count' (default 0) cutout cubes. Importable: _excl_scope ('snapshot'),
excluded_subbands (the fake search's, as the real module imports it) and GROUPS (params 'gs_groups', else
instrument.SUBBAND_BANDS: 'gs_groups' is a group search reading other members - a disagreeing reader)."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
from phase3_snapshot_search import excluded_subbands      # noqa: F401 - regen_group_cutouts :494 calls it by this name
_P = json.load(open(os.path.join(HERE, 'params.json')))
GROUPS = _P.get('gs_groups') or {b: [m + 'MHz' for m in ms] for b, ms in _P['wb_subs'].items()}


def _excl_scope(lst, date):
    return 'snapshot'


def main():
    import argparse
    ap = argparse.ArgumentParser()
    for k in ('--lst', '--date', '--stokes', '--out', '--cutouts-from-cubes'):
        ap.add_argument(k)
    a = ap.parse_args()
    with open(os.path.join(HERE, 'calls.jsonl'), 'a') as f:
        f.write(json.dumps(['wideband'] + sys.argv[1:]) + '\n')
    n = json.load(open(os.path.join(HERE, 'params.json'))).get('wb_count', 0)
    print('2026-11-23 00:00:00,000 INFO CUTOUTS-REGEN-WIDEBAND %s %s %s %s: %s cutout cubes'
          % (a.lst, a.date, a.cutouts_from_cubes, a.stokes, n), flush=True)


if __name__ == '__main__':
    main()
'''

FAKE_CF = r'''"""FAKE exopipe/phase3/coadd_full_pipeline.py (the group search's deep-rule reader, photometry_cutouts.GROUP_EXCLUSION_ROOT) of
tests/test_photometry_writer_safety.py: load_exclusions / is_excluded as the real module's (params 'group_decisions' else
'data_decisions'; 'extra_group_excluded' names it calls excluded besides - a disagreeing reader)."""
import csv, json, os
P = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'params.json')))
DECISIONS_CSV = P.get('group_decisions') or P['data_decisions']
SUBBAND_ONLY_EXCL = {'18'}
_WB_SUBS = {k: set(v) for k, v in P['wb_subs'].items()}


def load_exclusions(scope=None):
    if scope not in (None, 'snapshot', 'snapshot-gpu'):
        raise RuntimeError('REFUSING: unknown exclusion scope %r' % scope)
    if not os.path.exists(DECISIONS_CSV):
        raise RuntimeError('REFUSING to run: exclusion list not found at %s' % DECISIONS_CSV)
    with open(DECISIONS_CSV) as f:
        return {(r['lst'].strip(), r['subband'].strip().replace('MHz', ''), r['date'].strip()) for r in csv.DictReader(f)
                if r['verdict'].strip() == 'exclude'}


def is_excluded(lst, freq_str, date, exclusions):
    fs = str(freq_str).replace('MHz', '')
    if fs in P.get('extra_group_excluded', []):
        return True
    if fs in _WB_SUBS:
        bad = {s for (l, s, d) in exclusions if l == lst and d == date}
        return bool(_WB_SUBS[fs] & (bad - SUBBAND_ONLY_EXCL)) or (lst, fs, date) in exclusions
    return (lst, fs, date) in exclusions
'''


def check(label, cond, detail=''):
    print(('PASS ' if cond else 'FAIL ') + label + ('' if cond else '  ' + str(detail)[-1500:]), flush=True)
    if not cond:
        FAILS.append(label)


class Env:
    """The synthetic world: scratch, exclusion list, sample lists, cubes, the fake code root, CFG patched (restored by close)."""

    def __init__(self):
        base = os.environ.get('EXOPIPE3_TEST_SCRATCH') or None
        self.d = tempfile.mkdtemp(prefix='test_photometry_writer_safety_', dir=base)
        self.code = os.path.join(self.d, 'code')
        os.makedirs(self.code)
        self.dd = os.path.join(self.d, 'data_decisions.csv')
        with open(self.dd, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['lst', 'subband', 'date', 'verdict', 'reason'])
            w.writerows(ROWS)
        self.clv = os.path.join(self.d, 'clv15')
        os.makedirs(self.clv)
        for name in CFG.photometry.SAMPLE_FILES.values():
            with open(os.path.join(self.clv, name), 'w') as f:
                f.write('name,ra,dec\nT0,10.0,20.0\n')
        self.solar = os.path.join(self.d, 'solar_audit.csv')
        open(self.solar, 'w').write('night,band,verdict\n')
        self.cubes_root = os.path.join(self.d, 'cubes')
        for name, src in (('phase3_snapshot_search.py', FAKE_SS), ('phase3_snapshot_group_search.py', FAKE_GS),
                          ('phase3_snapshot_cutouts.py', FAKE_CUT)):
            open(os.path.join(self.code, name), 'w').write(src)
        self.prod = os.path.join(self.d, 'exopipe_prod')            # the group search's reader package (GROUP_EXCLUSION_ROOT)
        self.cf = os.path.join(self.prod, 'exopipe', 'phase3')
        os.makedirs(self.cf)
        for name, src in (('exopipe/__init__.py', ''), ('exopipe/phase3/__init__.py', ''),
                          ('exopipe/phase3/coadd_full_pipeline.py', FAKE_CF)):
            open(os.path.join(self.prod, name), 'w').write(src)
        self.params(write={'V': 2, 'I': 2}, count={'V': 2, 'I': 2})
        self.saved = []
        for obj, k, v in ((CFG.paths, 'DATA_DECISIONS', self.dd),
                          (CFG.paths, 'SNAPSHOT_GPU_FRAME_CELLS', os.path.join(self.d, 'no_gpu_cells.txt')),
                          (CFG.paths, 'SNAPSHOT_CUBES_ROOT', self.cubes_root),
                          (CFG.paths, 'SNAPSHOT_FLEET_OUT', os.path.join(self.d, 'search_out')),
                          (CFG.paths, 'CLV15_ROOT', self.clv),
                          (CFG.search_snapshot, 'SOLAR_AUDIT', self.solar),
                          (CFG.site, 'PYTHON', dict(CFG.site.PYTHON, py36=sys.executable, default=sys.executable)),
                          (CFG.site, 'LEGACY_CODE_ROOTS', [self.code]),
                          (CFG.photometry_cutouts, 'GROUP_EXCLUSION_ROOT', self.prod),
                          (CFG.paths, 'OPS_ROOT', os.path.join(self.d, 'ops')),     # no rescale records: the gate is stubbed
                          (SC, 'fluxscale_gate', lambda *a: {}), (SC, 'fluxscale_recheck', lambda *a: ([], [])),
                          (SC, 'generation_checks', lambda *a: None), (SC, 'fluxscale_in_place', lambda *a: (lambda: None)),
                          (K, 'engine', self.engine), (TC, 'identity_subband', self.identity)):
            self.saved.append((obj, k, getattr(obj, k)))
            setattr(obj, k, v)
        self.real_engine = self.saved[-2][2]
        from exopipe.phase3.core import exclusions
        exclusions._CACHE.clear()
        self.n = 0

    def engine(self, key):
        if key == 'cutouts_subband':
            return os.path.join(self.code, 'phase3_snapshot_search.py')
        if key == 'cutouts_wideband':
            return os.path.join(self.code, 'phase3_snapshot_group_search.py')
        return self.real_engine(key)            # cutouts_probe: the real probe (photometry.ENGINE_V2)

    @staticmethod
    def identity(files, cell, band, stokes, **kw):
        """The identity with the cube is tests/test_photometry.py's (synthetic cubes + the real writer); here every file passes."""
        return [(f, True, 0.0, 'identity not tested here') for f in files]

    def params(self, **kw):
        p = dict(data_decisions=self.dd, write={'V': 0, 'I': 0}, count={'V': 0, 'I': 0},
                 targets=[['HWJ', 'T%d' % k, 10.0, 40.0] for k in range(3)],
                 wb_subs={b: [m.replace('MHz', '') for m in ms] for b, ms in CFG.instrument.SUBBAND_BANDS.items()})
        p.update(kw)
        json.dump(p, open(os.path.join(self.code, 'params.json'), 'w'))
        json.dump(p, open(os.path.join(self.cf, 'params.json'), 'w'))
        if os.path.exists(os.path.join(self.code, 'calls.jsonl')):
            os.remove(os.path.join(self.code, 'calls.jsonl'))

    def calls(self):
        p = os.path.join(self.code, 'calls.jsonl')
        return [json.loads(ln) for ln in open(p)] if os.path.exists(p) else []

    def cubes(self, cell, band, crvals=None):
        """The cell-band's V and I flow cubes: FRAMES (2 frames of 50 x 50) and FRAME_TABLE (ts, the frame headers: SIN, 1 deg
        pixels, reference pixel 25, reference value ``crvals[frame]``, default (10, 40) - written again when given)."""
        out = {}
        for S in 'VI':
            p = TC.cube_path(cell, band, S, self.cubes_root)
            if not os.path.exists(p) or crvals:
                os.makedirs(os.path.dirname(p), exist_ok=True)
                hs = []
                for ra0, dec0 in crvals or [(10.0, 40.0)] * 2:
                    h = fits.Header()
                    for k, v in (('NAXIS', 2), ('NAXIS1', 50), ('NAXIS2', 50), ('CTYPE1', 'RA---SIN'), ('CTYPE2', 'DEC--SIN'),
                                 ('CRVAL1', ra0), ('CRVAL2', dec0), ('CRPIX1', 25.0), ('CRPIX2', 25.0), ('CDELT1', -1.0),
                                 ('CDELT2', 1.0)):
                        h[k] = v
                    hs.append(h.tostring(sep='\n'))
                tab = fits.BinTableHDU.from_columns(
                    [fits.Column(name='ts', format='15A', array=np.array(['20250105T0102%02d' % i for i in range(len(hs))])),
                     fits.Column(name='header', format='%dA' % max(map(len, hs)), array=np.array(hs))], name='FRAME_TABLE')
                fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(np.zeros((len(hs), 50, 50), np.float32), name='FRAMES'),
                              tab]).writeto(p, overwrite=True)
            out[S] = [p]
        return out

    def run(self, cell, band, tag, resume=False, reads=None, crvals=None):
        """One unit through run_unit as stage_snapshot_cutouts calls it (``reads``: the stage's EngineReads, None: the unit's
        own): (run, unit, cutout root, out)."""
        self.n += 1
        out = os.path.join(self.d, 'out_%s' % tag)
        root = os.path.join(self.d, 'root_%s' % tag)
        os.makedirs(out, exist_ok=True)
        os.makedirs(root, exist_ok=True)
        run = record.Run('photometry', ['test'], out, CFG, resume=resume)
        run.run_id = '%s_%d' % (run.run_id, self.n)
        args = argparse.Namespace(threads=None)
        name = 'cutouts.%s.%s' % (cell, band)
        SC.run_unit(run, name, dict(tier='snapshot', stage='cutouts', cell=cell, band=band, cutout_root=root),
                    SC.snapshot_unit, run, args, cell, band, root, [], False, SAMPLES, self.cubes(cell, band, crvals), reads)
        return run, run.units[-1], root, out

    def close(self):
        for obj, k, v in reversed(self.saved):
            setattr(obj, k, v)
        shutil.rmtree(self.d, ignore_errors=True)


def placed(root):
    """The cutouts in a root (the sample directories; staging and manifest directories aside)."""
    return sorted(os.path.relpath(os.path.join(dp, f), root) for s in SAMPLES for dp, _, fn in os.walk(os.path.join(root, s))
                  for f in fn)


def test_fitsio(E):
    p = os.path.join(E.d, 'whole.fits')
    nt = 4
    cols = [fits.Column(name='ts', format='15A', array=np.array(['x'] * nt))]
    fits.HDUList([fits.PrimaryHDU(np.ones((nt, 7, 7), np.float32)), fits.BinTableHDU.from_columns(cols)]).writeto(p)
    size, nhdu = fitsio.fits_implied_size(p)
    check('fits_implied_size: a whole 2-HDU file implies its own size', (size, nhdu) == (os.path.getsize(p), 2),
          (size, nhdu, os.path.getsize(p)))
    check('settle_fits: a whole file passes (fsync, size, no hole, cache dropped) and returns its HDU count',
          fitsio.settle_fits(p) == (None, 2), fitsio.settle_fits(p))
    q = os.path.join(E.d, 'short.fits')
    shutil.copyfile(p, q)
    os.truncate(q, os.path.getsize(q) - 1)
    why = fitsio.settle_fits(q)[0]
    check('settle_fits: a file 1 byte short fails, the shortfall named', why is not None and '(-1)' in why, why)
    shutil.copyfile(p, q)
    os.truncate(q, 2880 * 3)                     # inside the first HDU's data
    why = fitsio.settle_fits(q)[0]
    check('settle_fits: a file cut inside its first HDU fails', why is not None and 'imply' in why, why)
    shutil.copyfile(p, q)
    with open(q, 'ab') as f:
        f.write(b'\0' * 2880)
    why = fitsio.settle_fits(q)[0]
    check('settle_fits: bytes beyond the last HDU fail', why is not None and 'do not start an HDU' in why, why)
    os.truncate(q, 100)
    why = fitsio.settle_fits(q)[0]
    check('settle_fits: a header cut short fails', why is not None and 'cut short' in why, why)
    # a sparse file: the header and the last 8 KiB written, the 4 MiB between never written (a hole; its size is right)
    dense = os.path.join(E.d, 'dense.fits')
    fits.PrimaryHDU(np.ones((1024, 1024), np.float32)).writeto(dense)
    raw = open(dense, 'rb').read()
    sparse = os.path.join(E.d, 'sparse.fits')
    with open(sparse, 'wb') as f:
        f.write(raw[:8192])
        f.seek(len(raw) - 8192)
        f.write(raw[-8192:])
    why, nhdu = fitsio.settle_fits(sparse)
    check('settle_fits: a sparse file of the right size (4 MiB never written after byte 8192) fails, the hole named; the '
          'dense file passes', os.path.getsize(sparse) == len(raw) and why is not None and 'hole at byte 8192 ' in why
          and nhdu == 1 and fitsio.settle_fits(dense) == (None, 1), (why, nhdu, fitsio.settle_fits(dense)))


def test_write_gates(E):
    # pass: V 2 I 2 with 2 + 2 files
    E.params(write={'V': 2, 'I': 2}, count={'V': 2, 'I': 2})
    run, u, root, out = E.run(SUB_CELL, SUB_BAND, 'pass')
    tests = {t['name'].split('.', 1)[1]: t for t in u.tests}
    check('done line V 2 I 2 with 2 + 2 staged files: unit done, W1 fsync/size and W2 count pass, 4 cutouts placed, exit 0',
          u.status == 'done' and tests.get('W1_fsync_size', {}).get('passed') and tests.get('W2_count', {}).get('passed')
          and len(placed(root)) == 4 and run.exit_code() == 0,
          (u.status, u.reason, {k: (t['passed'], t.get('detail')) for k, t in tests.items()}, placed(root)))
    check('the probe read the engines\' reader before the engine (one exclusion read recorded, engine called once)',
          u.meta.get('exclusions', {}).get('engine_excluded') == [] and len(E.calls()) == 1, (u.meta.get('exclusions'), E.calls()))
    # a staged file cut short by N bytes
    E.params(write={'V': 2, 'I': 2}, count={'V': 2, 'I': 2}, truncate=1000)
    run, u, root, _ = E.run(SUB_CELL, SUB_BAND, 'trunc')
    check('a staged cutout cut short by 1000 bytes: refused C-WRITE (on-disk size != header-implied size), nothing placed, exit 2',
          u.status == 'refused' and 'C-WRITE' in u.reason and '-1000' in u.reason and placed(root) == [] and run.exit_code() == 2,
          (u.status, u.reason, placed(root)))
    # fsync raises OSError
    E.params(write={'V': 2, 'I': 2}, count={'V': 2, 'I': 2})
    real = fitsio.fsync_fd

    def eio(fd):                # the staged FITS files' fsync fails; the records' (JSON) do not
        if os.readlink('/proc/self/fd/%d' % fd).endswith('.fits'):
            raise OSError(errno.EIO, 'Input/output error (test)')
        return real(fd)
    fitsio.fsync_fd = eio
    try:
        run, u, root, _ = E.run(SUB_CELL, SUB_BAND, 'eio')
    finally:
        fitsio.fsync_fd = real
    check('fsync raising OSError (EIO): refused C-WRITE naming the fsync failure, nothing placed, exit 2',
          u.status == 'refused' and 'C-WRITE' in u.reason and 'fsync failed' in u.reason and placed(root) == []
          and run.exit_code() == 2, (u.status, u.reason, placed(root)))
    # count V 3 I 3 with V files only
    E.params(write={'V': 3, 'I': 0}, count={'V': 3, 'I': 3})
    run, u, root, _ = E.run(SUB_CELL, SUB_BAND, 'count')
    check('done line V 3 I 3 with V files only: refused C-COUNT (I: 3 reported, 0 staged), nothing placed, exit 2',
          u.status == 'refused' and 'C-COUNT' in u.reason and "I: engine reported '3', 0 file(s) staged" in u.reason
          and placed(root) == [] and run.exit_code() == 2, (u.status, u.reason))
    # a positive count with no file at all (was: C-NOCUT, no input, exit 3)
    E.params(write={'V': 0, 'I': 0}, count={'V': 2, 'I': 2})
    run, u, root, _ = E.run(SUB_CELL, SUB_BAND, 'nofile')
    check('done line V 2 I 2 with no file staged: refused C-COUNT, not C-NOCUT (no input), exit 2',
          u.status == 'refused' and 'C-COUNT' in u.reason and run.exit_code() == 2, (u.status, u.reason))
    # an engine count the files cannot be held to
    E.params(write={'V': 0, 'I': 0}, count={'V': 'FAILED: test', 'I': 0})
    run, u, root, _ = E.run(SUB_CELL, SUB_BAND, 'failed')
    check('done line V FAILED: ...: refused C-COUNT (no count to hold the files to), exit 2',
          u.status == 'refused' and 'C-COUNT' in u.reason, (u.status, u.reason))
    # list entries sharing one file name: T0 named by two entries within MERGE_ARCMIN (one object, one file, written and
    # counted once per entry in the writer's view: elevation = dec here, MIN_EL_DEG 30; the cube grid 50 x 50, 1 deg pixels
    # about (10, 40))
    dup = [['HWJ', 'T0', 10.0, 40.0], ['HWJ', 'T0', 10.0, 40.001], ['HWJ', 'T1', 11.0, 40.0]]
    E.params(write={'V': 2, 'I': 2}, count={'V': 3, 'I': 3}, targets=dup)
    run, u, root, _ = E.run(SUB_CELL, SUB_BAND, 'dup')
    siv = u.meta.get('shared_in_view') or {}
    check('two list entries in view share T0\'s file: done line V 3 I 3 with 2 + 2 files (T0 credited with its 2 entries in '
          'view) is done, 4 placed; the shared name and the writer\'s view of it recorded', u.status == 'done'
          and len(placed(root)) == 4 and (u.meta.get('engine_reads') or {}).get('targets', {}).get('multiple') == {'HWJ/T0': 2}
          and all(siv.get(S, {}).get('in_view') == {'HWJ/T0': 2} for S in 'VI'), (u.status, u.reason, u.meta.get('engine_reads'),
                                                                                    siv))
    E.params(write={'V': 2, 'I': 2}, count={'V': 2, 'I': 2}, targets=dup)
    run, u, root, _ = E.run(SUB_CELL, SUB_BAND, 'dup_once')
    check('the same with V 2 I 2 (a shared file counted once): refused C-COUNT naming the list entries in view, exit 2',
          u.status == 'refused' and 'C-COUNT' in u.reason and '(3 list entries in view)' in u.reason, (u.status, u.reason))
    E.params(write={'V': 1, 'I': 1}, count={'V': 3, 'I': 3}, targets=[dup[2], ['HWJ', 'T1', 11.0, 40.001], dup[0]])
    run, u, root, _ = E.run(SUB_CELL, SUB_BAND, 'dup_lost')
    check('V 3 I 3 with T1 the shared name and its file missing (T0 staged, 1 entry): refused C-COUNT, nothing placed',
          u.status == 'refused' and 'C-COUNT' in u.reason and placed(root) == [], (u.status, u.reason))
    E.params(write={'V': 1, 'I': 1}, count={'V': 3, 'I': 3}, targets=dup)
    run, u, root, _ = E.run(SUB_CELL, SUB_BAND, 'dup_missing')
    check('V 3 I 3 with the shared T0 file staged but T1\'s missing (2 entries staged): refused C-COUNT, nothing placed',
          u.status == 'refused' and 'C-COUNT' in u.reason and placed(root) == [], (u.status, u.reason))
    # a group straddling the elevation limit: T0 at elevation 40 (written, counted) and 20 (below MIN_EL_DEG: skipped)
    straddle = [['HWJ', 'T0', 10.0, 40.0], ['HWJ', 'T0', 10.0, 20.0]]
    E.params(write={'V': 1, 'I': 1}, count={'V': 1, 'I': 1}, targets=straddle)
    run, u, root, _ = E.run(SUB_CELL, SUB_BAND, 'straddle')
    check('a group straddling the elevation limit (1 of 2 entries in view), 1 file, the engine reporting 1: done, 2 placed, T0 '
          'credited with 1', u.status == 'done' and len(placed(root)) == 2
          and all((u.meta.get('shared_in_view') or {}).get(S, {}).get('in_view') == {'HWJ/T0': 1} for S in 'VI'),
          (u.status, u.reason, u.meta.get('shared_in_view')))
    E.params(write={'V': 1, 'I': 1}, count={'V': 2, 'I': 2}, targets=straddle + [['HWJ', 'T1', 11.0, 40.0]], skip=['T1'])
    run, u, root, _ = E.run(SUB_CELL, SUB_BAND, 'straddle_lost')
    check('the straddling group plus T1, T1\'s file lost, the engine reporting 2 (T0 1 + T1 1): refused C-COUNT (the group\'s '
          'size, 2, would have hidden the loss), nothing placed', u.status == 'refused' and 'C-COUNT' in u.reason
          and placed(root) == [], (u.status, u.reason))
    k3 = [['HWJ', 'T0', 10.0, 40.0], ['HWJ', 'T0', 10.0, 40.001], ['HWJ', 'T0', 10.0, 40.002], ['HWJ', 'T1', 11.0, 40.0]]
    E.params(write={'V': 2, 'I': 2}, count={'V': 4, 'I': 4}, targets=k3)
    run, u, root, _ = E.run(SUB_CELL, SUB_BAND, 'k3')
    check('control: a k=3 group in view and T1, the engine reporting 4, both files: done (T0 credited with 3)',
          u.status == 'done' and len(placed(root)) == 4, (u.status, u.reason))
    E.params(write={'V': 2, 'I': 2}, count={'V': 4, 'I': 4}, targets=k3, skip=['T0'])
    run, u, root, _ = E.run(SUB_CELL, SUB_BAND, 'k3_lost')
    check('the k=3 group in view with its file lost (T1 staged, the engine reporting 4): refused C-COUNT, nothing placed',
          u.status == 'refused' and 'C-COUNT' in u.reason and placed(root) == [], (u.status, u.reason))
    # an entry off the cube's grid: (10, 80) projects 37 pixels from the reference pixel, beyond the 50-pixel grid
    E.params(write={'V': 1, 'I': 1}, count={'V': 1, 'I': 1}, targets=[['HWJ', 'T0', 10.0, 40.0], ['HWJ', 'T0', 10.0, 80.0]])
    run, u, root, _ = E.run(SUB_CELL, SUB_BAND, 'offgrid')
    check('a group with one entry off the cube\'s grid, 1 file, the engine reporting 1: done (the off-grid entry not credited)',
          u.status == 'done' and all((u.meta.get('shared_in_view') or {}).get(S, {}).get('in_view') == {'HWJ/T0': 1}
                                     for S in 'VI'), (u.status, u.reason, u.meta.get('shared_in_view')))
    # frames whose geometries disagree: frame 0 about dec 40 sees (10, 70) off the grid, frame 1 about dec 60 sees it on it
    E.params(write={'V': 1, 'I': 1}, count={'V': 1, 'I': 1}, targets=[['HWJ', 'T0', 10.0, 40.0], ['HWJ', 'T0', 10.0, 70.0]])
    run, u, root, _ = E.run('03h_2025-01-06', SUB_BAND, 'ambiguous', crvals=[(10.0, 40.0), (10.0, 60.0)])
    check('cube frames whose geometries disagree on the count of a shared name (1 vs 2 in view): refused C-COUNT, nothing placed',
          u.status == 'refused' and 'C-COUNT' in u.reason and 'disagree' in u.reason and placed(root) == [],
          (u.status, u.reason))
    # the night rule refused both Stokes: no file, no input
    E.params(write={'V': 0, 'I': 0}, count={'V': 'None', 'I': 'None'})
    run, u, root, _ = E.run(SUB_CELL, SUB_BAND, 'night')
    check('done line V None I None (night rule) with no file: no input (C-NOCUT), exit 3',
          u.status == 'no-input' and 'C-NOCUT' in u.reason and run.exit_code() == 3, (u.status, u.reason))


def test_exclusion_gate(E):
    E.params(write={'V': 2, 'I': 2}, count={'V': 2, 'I': 2})
    run, u, root, _ = E.run(EXCL_CELL, SUB_BAND, 'excl_sub')
    check('an excluded subband cell-band: C-EXCLUDED with its own record, the engine never started, exit 3',
          u.status == 'no-input' and 'C-EXCLUDED' in u.reason and 'no engine started' in u.reason and E.calls() == []
          and os.path.exists(record.record_path(run.out_root, 'photometry', u.name)) and run.exit_code() == 3,
          (u.status, u.reason, E.calls()))
    E.params()
    run, u, root, _ = E.run(RED_EXCL_CELL, 'Red', 'excl_red')
    check('a wideband with an excluded member other than 18 MHz (23 MHz in Red): C-EXCLUDED, the engine never started',
          u.status == 'no-input' and 'C-EXCLUDED' in u.reason and E.calls() == [], (u.status, u.reason, E.calls()))
    E.params()
    run, u, root, _ = E.run(RED_18_CELL, 'Red', 'red18')
    wb = [c for c in E.calls() if c[0] == 'wideband']
    check('a wideband whose only excluded member is 18 MHz: not excluded, the wideband engine runs (V and I), the unit ends on '
          'its output (C-NOCUT: 0 cutout cubes)', 'C-EXCLUDED' not in u.reason and len(wb) == 2 and u.status == 'no-input'
          and 'C-NOCUT' in u.reason, (u.status, u.reason, E.calls()))
    E.params(write={'V': 2, 'I': 2}, count={'V': 2, 'I': 2}, extra_excluded=['55MHz'])
    run, u, root, _ = E.run(SUB_CELL, SUB_BAND, 'disagree')
    check('readers that disagree (the engines\' reader excludes 55 MHz, core/exclusions does not): refused PHOT-EXCL-READERS, '
          'the engine never started, exit 2', u.status == 'refused' and 'PHOT-EXCL-READERS' in u.reason and E.calls() == []
          and run.exit_code() == 2, (u.status, u.reason, E.calls()))
    other = os.path.join(E.d, 'other_data_decisions.csv')
    shutil.copyfile(E.dd, other)
    E.params(write={'V': 2, 'I': 2}, count={'V': 2, 'I': 2}, data_decisions=other)
    run, u, root, _ = E.run(SUB_CELL, SUB_BAND, 'otherfile')
    check('the engines\' reader reads another exclusion list: refused PHOT-EXCL-FILE, the engine never started',
          u.status == 'refused' and 'PHOT-EXCL-FILE' in u.reason and E.calls() == [], (u.status, u.reason))
    E.params()
    run = record.Run('photometry', ['test'], os.path.join(E.d, 'out_adhoc'), CFG)
    reads = SC.EngineReads(run, [SUB_CELL], [('ADHOC', os.path.join(E.d, 'x.csv'))], False, [])
    check('the probe runs with the engines\' ad hoc environment (P3_SNAP_CUTOUT_SAMPLES): the ad hoc list\'s shared name read',
          reads.error is None and reads.multiple == {'ADHOC/X': 2} and reads.cells[SUB_CELL].get('excluded') == [],
          (reads.error, reads.multiple, reads.cells))
    # the group search's deep-rule reader (the third reader) must agree with core/exclusions too
    E.params(extra_group_excluded=['Red'])
    run, u, root, _ = E.run(RED_18_CELL, 'Red', 'gr_band')
    check('the group search\'s reader calls Red excluded where core/exclusions does not (18 MHz only): refused '
          'PHOT-EXCL-READERS naming that reader, the engine never started', u.status == 'refused'
          and 'PHOT-EXCL-READERS' in u.reason and 'group search' in u.reason and E.calls() == [], (u.status, u.reason, E.calls()))
    E.params(extra_group_excluded=['27'])
    run, u, root, _ = E.run(RED_18_CELL, 'Red', 'gr_member')
    check('the group search\'s reader calls a Red member (27 MHz) excluded, core/exclusions does not: refused '
          'PHOT-EXCL-READERS, the engine never started', u.status == 'refused' and 'PHOT-EXCL-READERS' in u.reason
          and '27MHz' in u.reason and E.calls() == [], (u.status, u.reason, E.calls()))
    E.params(write={'V': 2, 'I': 2}, count={'V': 2, 'I': 2}, extra_group_excluded=['Red', '55'])
    run, u, root, _ = E.run(SUB_CELL, SUB_BAND, 'gr_subband')
    check('control: a subband unit does not consult the group search\'s reader (its engine does not use it; that reader calls '
          '55 MHz excluded here): done',
          u.status == 'done', (u.status, u.reason))
    E.params(group_decisions=other)
    run, u, root, _ = E.run(RED_18_CELL, 'Red', 'gr_file')
    check('the group search\'s reader reads another exclusion list: refused PHOT-EXCL-FILE, the engine never started',
          u.status == 'refused' and 'PHOT-EXCL-FILE' in u.reason and E.calls() == [], (u.status, u.reason))
    E.params(group_decisions=os.path.join(E.d, 'missing.csv'))
    run, u, root, _ = E.run(RED_18_CELL, 'Red', 'gr_missing')
    check('the group search\'s reader cannot read its list (missing): refused PHOT-EXCL-FILE naming that reader',
          u.status == 'refused' and 'PHOT-EXCL-FILE' in u.reason and 'group search' in u.reason and E.calls() == [],
          (u.status, u.reason))
    # the wideband engine's 'None': the group was not built - no count to hold the files to
    check('expected_files: the subband engine\'s None (night rule) is 0 files, the group search\'s None is no count',
          SC.expected_files('None') == 0 and SC.expected_files('None', wideband=True) is None
          and SC.expected_files('excluded', wideband=True) == 0, (SC.expected_files('None'), SC.expected_files('None', True)))
    E.params(wb_count='None')
    run, u, root, _ = E.run(RED_18_CELL, 'Red', 'wb_none')
    check('the group search reports None cutout cubes for Red (not built): refused C-ENGINE naming the group, exit 2',
          u.status == 'refused' and 'C-ENGINE' in u.reason and 'did not build Red' in u.reason and run.exit_code() == 2,
          (u.status, u.reason))
    check('the group search\'s member list of Red (its GROUPS without its excluded_subbands, read by the probe) agrees with '
          'core/exclusions\' and is the list the flux-scale gate checks (u.meta exclusions.engine_members): 18 MHz dropped',
          (u.meta.get('exclusions') or {}).get('engine_members') == ['23MHz', '27MHz', '32MHz', '36MHz'],
          u.meta.get('exclusions'))
    E.params(gs_groups={'Red': ['18MHz', '23MHz', '27MHz', '32MHz'], 'Green': [], 'Blue': []})
    run, u, root, _ = E.run(RED_18_CELL, 'Red', 'gs_members')
    check('the group search reads another member list of Red (36 MHz missing: a disagreeing reader): refused PHOT-EXCL-READERS '
          'naming both lists, the engine never started (review of PR 153, finding 5)', u.status == 'refused'
          and 'PHOT-EXCL-READERS' in u.reason and 'the members of Red' in u.reason and "'36MHz'" in u.reason
          and E.calls() == [], (u.status, u.reason))
    E.params(gs_groups={'Green': [], 'Blue': []})
    run, u, root, _ = E.run(RED_18_CELL, 'Red', 'gs_nogroup')
    check('the group search has no group Red: refused PHOT-EXCL-READERS, the engine never started', u.status == 'refused'
          and 'PHOT-EXCL-READERS' in u.reason and 'the group search has no group Red' in u.reason and E.calls() == [],
          (u.status, u.reason))
    # one probe call per stage serves every unit; a cell is probed again when data_decisions changed after the probe
    E.params(write={'V': 2, 'I': 2}, count={'V': 2, 'I': 2})
    run0 = record.Run('photometry', ['test'], os.path.join(E.d, 'out_stage'), CFG)
    reads = SC.EngineReads(run0, [SUB_CELL, RED_18_CELL, EXCL_CELL], [], False, ['Red'])
    units = [E.run(SUB_CELL, SUB_BAND, 'stage_a', reads=reads)[1], E.run(RED_18_CELL, 'Red', 'stage_b', reads=reads)[1],
             E.run(EXCL_CELL, SUB_BAND, 'stage_c', reads=reads)[1]]
    log = open(os.path.join(K.logs_dir(run0), 'cutouts_probe_run.log')).read()
    check('one probe call (\'run\' mode, every cell, the group reader\'s scopes) serves the stage\'s units: done, C-NOCUT, '
          'C-EXCLUDED, no further probe call', reads.error is None and len(reads.calls) == 1
          and [x.status for x in units] == ['done', 'no-input', 'no-input'] and 'C-EXCLUDED' in units[2].reason
          and log.count('ENGINE-TARGETS ') == 1 and log.count('ENGINE-EXCLUSIONS ') == 3
          and 'group_reader' in reads.cells[RED_18_CELL],
          (reads.error, reads.calls, [(x.status, x.reason) for x in units]))
    rows = open(E.dd).read()
    try:
        with open(E.dd, 'a', newline='') as f:
            csv.writer(f).writerow(['05h', '55MHz', '2025-01-09', 'exclude', 'synthetic row added after the probe'])
        run, u, root, _ = E.run(SUB_CELL, SUB_BAND, 'stage_d', reads=reads)
    finally:
        open(E.dd, 'w').write(rows)
    check('data_decisions changed after the stage\'s probe: the unit\'s cell probed again (2 probe calls), the unit done on '
          'the new bytes', u.status == 'done' and len(reads.calls) == 2
          and u.meta.get('exclusions', {}).get('sha1') == reads.cells[SUB_CELL]['sha1'], (u.status, u.reason, reads.calls))
    # one fresh read of data_decisions per unit, for the verdict and the resume identity alike
    from exopipe.phase3.core import exclusions
    real_ft, fresh_reads = exclusions.for_tier, []

    def counting(tier, fresh=False, path=None):
        if fresh:
            fresh_reads.append(tier)
        return real_ft(tier, fresh=fresh, path=path)
    exclusions.for_tier = counting
    try:
        run, u, root, _ = E.run(SUB_CELL, SUB_BAND, 'one_read')
    finally:
        exclusions.for_tier = real_ft
    check('one fresh read of data_decisions in the unit (its verdict and its resume identity)', u.status == 'done'
          and fresh_reads == ['snapshot'], (u.status, u.reason, fresh_reads))
    E.params(write={'V': 2, 'I': 2}, count={'V': 2, 'I': 2}, data_decisions=os.path.join(E.d, 'missing.csv'))
    run, u, root, _ = E.run(SUB_CELL, SUB_BAND, 'missingfile')
    check('the engines\' reader cannot read its list (missing): refused PHOT-EXCL-FILE, the engine never started',
          u.status == 'refused' and 'PHOT-EXCL-FILE' in u.reason and E.calls() == [], (u.status, u.reason))


def test_records(E):
    d = os.path.join(E.d, 'rec')
    p = os.path.join(d, 'x.json')
    calls = []
    real_fsync = os.fsync

    def spy(fd):
        calls.append(os.path.realpath('/proc/self/fd/%d' % fd))
        return real_fsync(fd)
    os.fsync = spy
    try:
        record.atomic_write_json(p, {'a': 1}, durable=True)
        durable = list(calls)
        del calls[:]
        record.atomic_write_json(os.path.join(d, 'y.json'), {'b': 1})
    finally:
        os.fsync = real_fsync
    check('atomic_write_json(durable=True) fsyncs the temporary file and the directory, no temporary left; the default '
          '(every other caller) does not fsync', json.load(open(p)) == {'a': 1} and os.path.realpath(d) in durable
          and any('.tmp.' in c for c in durable) and calls == [] and sorted(os.listdir(d)) == ['x.json', 'y.json'],
          (durable, calls, os.listdir(d)))

    real_commit = record.git_commit
    record.git_commit = lambda path: 'abc1234'          # a clean tree: a done record may be resumed (control below)
    try:
        _records_fail_and_resume(E)
    finally:
        record.git_commit = real_commit


def _records_fail_and_resume(E):
    def failing(path, obj, durable=False):
        raise OSError(errno.EIO, 'Input/output error (test)', path)
    real_write = record.atomic_write_json

    # 1. a done unit, then the same unit refused (a file cut short) and its record write fails
    E.params(write={'V': 2, 'I': 2}, count={'V': 2, 'I': 2})
    run, u, root, out = E.run(SUB_CELL, SUB_BAND, 'rec')
    rp = record.record_path(out, 'photometry', u.name)
    first = (u.status, json.load(open(rp)).get('status') if os.path.exists(rp) else None)
    E.params(write={'V': 2, 'I': 2}, count={'V': 2, 'I': 2}, truncate=10)
    record.atomic_write_json = failing
    try:
        run, u, root, out = E.run(SUB_CELL, SUB_BAND, 'rec')
    finally:
        record.atomic_write_json = real_write
    check('a refused unit whose record write raises OSError: status error naming RECORD-WRITE and the refusal, exit 1, NO record '
          'left (the earlier done record removed)', first == ('done', 'done') and u.status == 'error'
          and 'RECORD-WRITE' in u.reason and 'C-WRITE' in u.reason and run.exit_code() == 1 and not os.path.exists(rp),
          (first, u.status, u.reason, os.path.exists(rp)))
    E.params(write={'V': 2, 'I': 2}, count={'V': 2, 'I': 2})
    run, u, root, out = E.run(SUB_CELL, SUB_BAND, 'rec', resume=True)
    check('--resume after it: the unit re-runs (no record), done, its record written, exit 0',
          u.status == 'done' and len(E.calls()) == 1 and os.path.exists(rp) and json.load(open(rp))['status'] == 'done'
          and run.exit_code() == 0, (u.status, u.reason, E.calls()))
    # 2. a done unit whose record write fails once
    n = []

    def once(path, obj, durable=False):
        if not n and path == rp:
            n.append(path)
            raise OSError(errno.EIO, 'Input/output error (test)', path)
        return real_write(path, obj, durable)
    record.atomic_write_json = once
    try:
        run, u, root, out = E.run(SUB_CELL, SUB_BAND, 'rec')
    finally:
        record.atomic_write_json = real_write
    check('a done unit whose record write raises OSError: status error (outcome was done), exit 1, no record (the earlier done '
          'record removed)',
          n and u.status == 'error' and 'RECORD-WRITE' in u.reason and 'was done' in u.reason and run.exit_code() == 1
          and not os.path.exists(rp), (u.status, u.reason))
    run, u, root, out = E.run(SUB_CELL, SUB_BAND, 'rec', resume=True)
    check('--resume after it: re-run, done, record written', u.status == 'done' and os.path.exists(rp), (u.status, u.reason))
    # 3. a done unit whose record cannot be serialized (a circular reference: json.dump raises ValueError, not OSError)
    real_td = record.UnitRecord.to_dict

    def circular(self):
        d = real_td(self)
        d['meta'] = dict(d.get('meta') or {}, loop=d)
        return d
    E.params(write={'V': 2, 'I': 2}, count={'V': 2, 'I': 2})
    record.UnitRecord.to_dict = circular
    try:
        run, u, root, out = E.run(SUB_CELL, SUB_BAND, 'rec')
    finally:
        record.UnitRecord.to_dict = real_td
    left = sorted(os.listdir(os.path.dirname(rp)))
    check('a done unit whose record write raises ValueError (circular meta): status error naming RECORD-WRITE and ValueError, '
          'exit 1, no record left (the earlier done record removed), no temporary file left', u.status == 'error'
          and 'RECORD-WRITE' in u.reason and 'ValueError' in u.reason and run.exit_code() == 1 and not os.path.exists(rp)
          and not any('.tmp.' in f for f in left), (u.status, u.reason, left))
    E.params(write={'V': 2, 'I': 2}, count={'V': 2, 'I': 2})
    run, u, root, out = E.run(SUB_CELL, SUB_BAND, 'rec', resume=True)
    check('--resume after it: the unit re-runs (engine called), done, record written', u.status == 'done' and len(E.calls()) == 1
          and os.path.exists(rp), (u.status, u.reason, E.calls()))
    E.params(write={'V': 2, 'I': 2}, count={'V': 2, 'I': 2})
    run, u, root, out = E.run(SUB_CELL, SUB_BAND, 'rec', resume=True)
    check('control: --resume with that done record present skips the unit, no engine call (the re-runs above are the missing '
          'record\'s)', u.status == 'skipped-valid' and E.calls() == [] and run.exit_code() == 0, (u.status, u.reason))


def main():
    E = Env()
    try:
        test_fitsio(E)
        test_write_gates(E)
        test_exclusion_gate(E)
        test_records(E)
    finally:
        E.close()
    print('%d check(s) failed: %s' % (len(FAILS), FAILS) if FAILS else 'all checks passed', flush=True)
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
