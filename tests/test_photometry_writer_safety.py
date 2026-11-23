#!/usr/bin/env python
"""Writer safety of the 10-s cutouts stage (lib/photometry/steps_cutouts.py, core/fitsio.py, core/record.py;
FINDING-20261122-CUTOUT-WRITER-DISCARDED-PAGES-PASS-AS-DONE: a Lustre eviction discarded the writer's dirty pages, cutouts were
left cut short at 24 MiB and the unit passed as done; a FAILED log line lost to EIO let a job be skipped silently).

The step's unit (steps_cutouts.snapshot_unit, through run_unit as the stage calls it) runs in this process on synthetic inputs,
its engines FAKES in a stub code root (an importable phase3_snapshot_search.py with the engines' exclusion reader,
phase3_snapshot_group_search.py, and phase3_snapshot_cutouts.py with the writer's target list) under this interpreter; the
exclusion list is a synthetic data_decisions.csv. Asserts:
  - fitsio.fits_implied_size / settle_fits on a whole file, a file cut short, a file with bytes beyond its last HDU;
  - C-WRITE: a staged cutout cut short by N bytes, or an fsync that raises OSError, refuses the unit, nothing placed;
  - C-COUNT: a done line 'V 3 I 3' with V files only refuses; 'V 2 I 2' with no file refuses (it was C-NOCUT, no input);
    'V 2 I 2' with 2 + 2 files is done; list entries sharing one file name (the writer writes that file once per entry and
    counts each): 'V 3' with 2 files, one of them shared by 2 entries, is done; the shared file or the other missing, or a
    shared file counted once, refused;
  - exclusion BEFORE the engine: an excluded subband cell-band ends C-EXCLUDED and the engine never ran; a wideband with an
    excluded member other than 18 MHz the same; with only its 18 MHz member excluded the wideband engine runs; readers that
    disagree (PHOT-EXCL-READERS) or read different files (PHOT-EXCL-FILE) refuse, the engine never ran; the probe runs with
    the engines' ad hoc environment;
  - durable records: atomic_write_json fsyncs (file and directory); a record write that raises OSError ends the unit 'error'
    (exit 1), leaves NO record (an earlier done record removed), and a resumed run re-runs the unit.
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
reader). Run with --cutouts-from-cubes: appends its argv to calls.jsonl, writes params 'write'[S] conforming cutouts per Stokes
into P3_SNAP_CUTOUT_ROOT, cuts params 'truncate' bytes off the first one, prints the done line with params 'count'."""
import csv, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
P = json.load(open(os.path.join(HERE, 'params.json')))
DATA_DECISIONS = P['data_decisions']


def excluded_subbands(lst, date, path=DATA_DECISIONS):
    with open(path) as f:
        out = {r['subband'].strip() for r in csv.DictReader(f) if r['lst'].strip() == lst and r['date'].strip() == date
               and r['verdict'].strip() == 'exclude'}
    return out | set(P.get('extra_excluded', []))


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
        written += [cutout(root, a.lst, a.date, a.subband, S, k) for k in range(P['write'][S])]
    if P.get('truncate') and written:
        os.truncate(written[0], os.path.getsize(written[0]) - P['truncate'])
    print('2026-11-23 00:00:00,000 INFO CUTOUTS-REGEN %s %s %s: V %s I %s' % (a.lst, a.date, a.subband, P['count']['V'],
                                                                             P['count']['I']), flush=True)


if __name__ == '__main__':
    main()
'''

FAKE_CUT = r'''"""FAKE phase3_snapshot_cutouts.py of tests/test_photometry_writer_safety.py: the writer's target list (params
'targets', [sample, name, ra, dec] rows; under P3_SNAP_CUTOUT_SAMPLES two entries ADHOC/X) and safe_name as the writer's."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))


def safe_name(nm):
    return "".join(ch if (ch.isalnum() or ch in "+-._") else "_" for ch in nm.strip()).strip("_")


def _targets():
    if os.environ.get('P3_SNAP_CUTOUT_SAMPLES'):
        return [('ADHOC', 'X', 1.0, 2.0), ('ADHOC', 'X', 1.0, 2.001)]
    return [tuple(t) for t in json.load(open(os.path.join(HERE, 'params.json')))['targets']]
'''

FAKE_GS = r'''"""FAKE phase3_snapshot_group_search.py of tests/test_photometry_writer_safety.py: appends its argv to calls.jsonl, writes
nothing, prints the wideband done line with 0 cutout cubes."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    for k in ('--lst', '--date', '--stokes', '--out', '--cutouts-from-cubes'):
        ap.add_argument(k)
    a = ap.parse_args()
    with open(os.path.join(HERE, 'calls.jsonl'), 'a') as f:
        f.write(json.dumps(['wideband'] + sys.argv[1:]) + '\n')
    print('2026-11-23 00:00:00,000 INFO CUTOUTS-REGEN-WIDEBAND %s %s %s %s: 0 cutout cubes'
          % (a.lst, a.date, a.cutouts_from_cubes, a.stokes), flush=True)


if __name__ == '__main__':
    main()
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
                 targets=[['HWJ', 'T%d' % k, 10.0, 20.0] for k in range(3)])
        p.update(kw)
        json.dump(p, open(os.path.join(self.code, 'params.json'), 'w'))
        if os.path.exists(os.path.join(self.code, 'calls.jsonl')):
            os.remove(os.path.join(self.code, 'calls.jsonl'))

    def calls(self):
        p = os.path.join(self.code, 'calls.jsonl')
        return [json.loads(ln) for ln in open(p)] if os.path.exists(p) else []

    def cubes(self, cell, band):
        out = {}
        for S in 'VI':
            p = TC.cube_path(cell, band, S, self.cubes_root)
            if not os.path.exists(p):
                os.makedirs(os.path.dirname(p), exist_ok=True)
                fits.PrimaryHDU(np.zeros((2, 2), np.float32)).writeto(p)
            out[S] = [p]
        return out

    def run(self, cell, band, tag, resume=False):
        """One unit through run_unit as stage_snapshot_cutouts calls it: (run, unit, cutout root, out)."""
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
                    SC.snapshot_unit, run, args, cell, band, root, [], False, SAMPLES, self.cubes(cell, band))
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
    check('settle_fits: a whole file passes (fsync, size, cache dropped)', fitsio.settle_fits(p) is None)
    q = os.path.join(E.d, 'short.fits')
    shutil.copyfile(p, q)
    os.truncate(q, os.path.getsize(q) - 1)
    why = fitsio.settle_fits(q)
    check('settle_fits: a file 1 byte short fails, the shortfall named', why is not None and '(-1)' in why, why)
    shutil.copyfile(p, q)
    os.truncate(q, 2880 * 3)                     # inside the first HDU's data
    why = fitsio.settle_fits(q)
    check('settle_fits: a file cut inside its first HDU fails', why is not None and 'imply' in why, why)
    shutil.copyfile(p, q)
    with open(q, 'ab') as f:
        f.write(b'\0' * 2880)
    why = fitsio.settle_fits(q)
    check('settle_fits: bytes beyond the last HDU fail', why is not None and 'do not start an HDU' in why, why)
    os.truncate(q, 100)
    why = fitsio.settle_fits(q)
    check('settle_fits: a header cut short fails', why is not None and 'cut short' in why, why)


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
    real = fitsio.fsync_path

    def eio(path):
        raise OSError(errno.EIO, 'Input/output error (test)', path)
    fitsio.fsync_path = eio
    try:
        run, u, root, _ = E.run(SUB_CELL, SUB_BAND, 'eio')
    finally:
        fitsio.fsync_path = real
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
    # list entries sharing one file name: T0 named by two entries within MERGE_ARCMIN (one object, one file, counted twice)
    dup = [['HWJ', 'T0', 10.0, 20.0], ['HWJ', 'T0', 10.0, 20.001], ['HWJ', 'T1', 11.0, 20.0]]
    E.params(write={'V': 2, 'I': 2}, count={'V': 3, 'I': 3}, targets=dup)
    run, u, root, _ = E.run(SUB_CELL, SUB_BAND, 'dup')
    check('two list entries share T0\'s file: done line V 3 I 3 with 2 + 2 files (T0 accounts for 2 entries) is done, 4 placed, '
          'the multiplicity recorded', u.status == 'done' and len(placed(root)) == 4
          and u.meta.get('engine_targets', {}).get('multiple') == {'HWJ/T0': 2}, (u.status, u.reason, u.meta.get('engine_targets')))
    E.params(write={'V': 2, 'I': 2}, count={'V': 2, 'I': 2}, targets=dup)
    run, u, root, _ = E.run(SUB_CELL, SUB_BAND, 'dup_once')
    check('the same with V 2 I 2 (a shared file counted once): refused C-COUNT naming the list entries, exit 2',
          u.status == 'refused' and 'C-COUNT' in u.reason and '(3 list entries)' in u.reason, (u.status, u.reason))
    E.params(write={'V': 1, 'I': 1}, count={'V': 3, 'I': 3}, targets=[dup[2], ['HWJ', 'T1', 11.0, 20.001], dup[0]])
    run, u, root, _ = E.run(SUB_CELL, SUB_BAND, 'dup_lost')
    check('V 3 I 3 with T1 the shared name and its file missing (T0 staged, 1 entry): refused C-COUNT, nothing placed',
          u.status == 'refused' and 'C-COUNT' in u.reason and placed(root) == [], (u.status, u.reason))
    E.params(write={'V': 1, 'I': 1}, count={'V': 3, 'I': 3}, targets=dup)
    run, u, root, _ = E.run(SUB_CELL, SUB_BAND, 'dup_missing')
    check('V 3 I 3 with the shared T0 file staged but T1\'s missing (2 entries staged): refused C-COUNT, nothing placed',
          u.status == 'refused' and 'C-COUNT' in u.reason and placed(root) == [], (u.status, u.reason))
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
    with run.unit('probe.adhoc', tier='snapshot', stage='cutouts') as u:
        got = SC.engine_probe(u, run, SUB_CELL, SUB_BAND, [('ADHOC', os.path.join(E.d, 'x.csv'))], False)
    check('the probe runs with the engines\' ad hoc environment (P3_SNAP_CUTOUT_SAMPLES): the ad hoc list\'s shared name read',
          got == (False, {'ADHOC/X': 2}), got)
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
        record.atomic_write_json(p, {'a': 1})
    finally:
        os.fsync = real_fsync
    check('atomic_write_json fsyncs the temporary file and the directory, no temporary left',
          json.load(open(p)) == {'a': 1} and os.path.realpath(d) in calls and any('.tmp.' in c for c in calls)
          and os.listdir(d) == ['x.json'], (calls, os.listdir(d)))

    real_commit = record.git_commit
    record.git_commit = lambda path: 'abc1234'          # a clean tree: a done record may be resumed (control below)
    try:
        _records_fail_and_resume(E)
    finally:
        record.git_commit = real_commit


def _records_fail_and_resume(E):
    def failing(path, obj):
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

    def once(path, obj):
        if not n and path == rp:
            n.append(path)
            raise OSError(errno.EIO, 'Input/output error (test)', path)
        return real_write(path, obj)
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
