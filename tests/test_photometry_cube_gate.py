#!/usr/bin/env python
"""lib/photometry/fluxscale.py and tests_fluxscale.py, the flux-scale gate of the 10-s cutout regeneration from the rescaled flow
cubes and its content checks (ported from exopipe_ops/gen6_20261117: gen6_cube_gate_DRAFT.py, gen6_wb_regen_DRAFT.sh :42-45,
gen6_cutout_regen_test_DRAFT.py (a)(b)(c), gen6_gen5_sample_DRAFT.py, gen6_wb_test_DRAFT.py (w)), on synthetic cubes and
records in a scratch directory (no cluster data):

  records   RESULTS rows win over plan rows; a newer plan overrides an older one and a newer plan's row that is not 'done' removes
            the older plan's record; of several RESULTS rows the newest by utc wins, whatever the file order; no RESULTS row at
            all refuses (RecordsError)
  gate      pass; no FSAPPLD; FSAPPLD off its record; the 11h 2024-12-31 73 MHz shape (V 1.0422, I without FSAPPLD); V != I;
            no record; no cube; a header longer than one positioned read; tile-compressed and plain FRAMES
  sidecar   written after an unchanged re-check (content = the check); a cube whose size, mtime or FSAPPLD changed between the
            check and the write refuses and writes nothing; the regeneration line loses its log time stamp
  wideband  every member cube the group search reads is gated: a member without FSAPPLD refuses, a member with one Stokes cube
            is checked, an excluded member is not read and not gated
  (a)(b)(c)(w)  each passes on a consistent unit and fails on the defect it exists for; the flux-scale record's content test
  sample    the generation-5 comparison sample: md5 order per LST hour, eligibility, generation membership
  step      the wideband side of lib/photometry/steps_cutouts.py: the member gate, its re-check after the cut and check (w)
            refuse the unit (C-FLUXSCALE, C-GENCHECK); the subband side runs end to end in tests/test_photometry.py
  review    (PR 153) a malformed record row refuses naming <file>:<line> (gate, dry run, in-place check); a tie at the newest
            utc with other factors is ambiguous (refused), a fractional utc orders by time; the cube identity carries the
            inode and ctime (a same-size copy with its mtime copied is a change); the wideband gate checks the engine's member
            list; the member cubes re-checked inside the placement transaction (a change undoes it); the generation-5 sample
            drawn once per hour per stage; records() cached by the record files' signature; one cube-path helper
Run: python tests/test_photometry_cube_gate.py
"""
import contextlib
import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from exopipe.phase3.config import CFG                                    # noqa: E402
from exopipe.phase3.lib.photometry import fluxscale as FS               # noqa: E402
from exopipe.phase3.lib.photometry import tests_fluxscale as TF         # noqa: E402
from exopipe.phase3.lib.photometry import tests_cutouts as TC           # noqa: E402
from exopipe.phase3.core.record import Refusal                           # noqa: E402

warnings.filterwarnings('ignore')
from astropy.utils import iers                                          # noqa: E402
iers.conf.auto_download = False         # the frame times below are inside the bundled IERS-B table: no network needed

FAILS = []
CELL = '00h_2022-09-26'


def check(label, cond, detail=''):
    print(('PASS ' if cond else 'FAIL ') + label + ('' if cond else '  ' + str(detail)[-1500:]))
    if not cond:
        FAILS.append(label)


@contextlib.contextmanager
def patched(obj, **kw):
    old = {k: getattr(obj, k) for k in kw}
    for k, v in kw.items():
        setattr(obj, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(obj, k, v)


class Ex:
    """An exclusion read (core.exclusions interface) excluding the (lst, subband, date) given."""
    def __init__(self, excl=()):
        self.excl = set(excl)

    def is_excluded(self, lst, sb, date):
        return (lst, sb, date) in self.excl


def frame_header(k, ts, n, ra=10.0, dec=37.0, cd=0.5):
    from astropy.io import fits
    h = fits.Header()
    for kk, v in (('NAXIS', 2), ('NAXIS1', n), ('NAXIS2', n), ('CTYPE1', 'RA---CAR'), ('CTYPE2', 'DEC--CAR'),
                  ('CRVAL1', ra), ('CRVAL2', dec), ('CRPIX1', n / 2.0 + 0.5), ('CRPIX2', n / 2.0 + 0.5), ('CDELT1', -cd),
                  ('CDELT2', cd), ('BMAJ', 0.2), ('BMIN', 0.15), ('BPA', 0.0)):
        h[kk] = v
    h['DATE-OBS'] = '%s-%s-%sT%s:%s:%s.000' % (ts[0:4], ts[4:6], ts[6:8], ts[9:11], ts[11:13], ts[13:15])
    return h


def frame_ts(cell, nt):
    ymd = cell.split('_', 1)[1].replace('-', '')
    return ['%s_07%02d%02d' % (ymd, (k * 10) // 60, (k * 10) % 60) for k in range(nt)]


def write_cube(path, fs, frames=None, nt=4, n=40, seed=0, compressed=True, comments=0, mtime_ns=None):
    """A consolidated flow cube of the search's layout: HDU 0 without data (FSAPPLD unless fs is None, NFRAMES), FRAMES
    (tile-compressed or plain) [T, Y, X] float32, FRAME_TABLE (ts, newline-separated frame header)."""
    from astropy.io import fits
    if frames is None:
        frames = np.random.default_rng(seed).normal(0, 1, (nt, n, n)).astype(np.float32)
    nt, n = frames.shape[0], frames.shape[1]
    ts = frame_ts(CELL, nt)
    h0 = fits.Header()
    if fs is not None:
        h0['FSAPPLD'] = fs
    h0['NFRAMES'] = nt
    for i in range(comments):
        h0.add_comment('padding card %d of a long primary header' % i)
    hdrs = [frame_header(k, ts[k], n).tostring(sep='\n') for k in range(nt)]
    tab = fits.BinTableHDU.from_columns([fits.Column(name='ts', format='15A', array=np.array(ts)),
                                         fits.Column(name='header', format='%dA' % max(len(x) for x in hdrs),
                                                     array=np.array(hdrs))], name='FRAME_TABLE')
    img = (fits.CompImageHDU(frames, name='FRAMES') if compressed else fits.ImageHDU(frames, name='FRAMES'))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fits.HDUList([fits.PrimaryHDU(header=h0), img, tab]).writeto(path, overwrite=True)
    if mtime_ns is not None:
        os.utime(path, ns=(mtime_ns, mtime_ns))
    return path


def write_csv(path, cols, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(rows)


def result_row(cube, fs, utc, quarantine=''):
    return [utc, cube, '%.6f' % fs, 'md5-%s' % os.path.basename(cube)[:20], '', quarantine, '0.1']


RES_COLS = ['utc', 'cube', 'fsappld', 'md5_new', 'md5_frame_units', 'quarantine', 'worst_pixel_over_noise']
PLAN_COLS = ['cube', 'cell', 'subband', 'stokes', 'status', 'bytes', 'factor']


def setup_records(ops, cubes, results, plans):
    """results {host: [row]}, plans {name: [row]} under the scratch ops root."""
    for host, rows in results.items():
        write_csv(os.path.join(ops, 'snapshot_cube_rescale_results', 'RESULTS_%s.csv' % host), RES_COLS, rows)
    for name, rows in plans.items():
        write_csv(os.path.join(ops, name), PLAN_COLS, rows)


def main():
    base = tempfile.mkdtemp(prefix='test_cube_gate_')
    ops, cubes = os.path.join(base, 'ops'), os.path.join(base, 'cubes')
    os.makedirs(ops)
    try:
        with patched(CFG.paths, OPS_ROOT=ops, SNAPSHOT_CUBES_ROOT=cubes):
            run(base, ops, cubes)
    finally:
        shutil.rmtree(base)
    print('ALL PASS' if not FAILS else 'FAILED: %d' % len(FAILS))
    return 1 if FAILS else 0


def cp(sb, S):
    return FS.cube_path(CELL, sb, S)


def run(base, ops, cubes):
    from astropy.io import fits
    # ---- no RESULTS row: the gate cannot run
    setup_records(ops, cubes, {}, {'snapshot_cube_rescale_plan_20261112.csv': []})
    try:
        FS.records()
        got = None
    except FS.RecordsError as e:
        got = str(e)
    check('records: no rescale RESULTS row -> RecordsError (the gate refuses, never runs without its records)',
          got is not None and 'no rescale RESULTS rows' in got, got)

    # ---- the cubes and their records
    for sb in ('55MHz', '50MHz', '46MHz', '41MHz', '36MHz', '32MHz', '27MHz', '23MHz', '18MHz'):
        for S in 'VI':
            if (sb, S) == ('36MHz', 'I'):
                continue                                       # no cube
            fs = {('46MHz', 'I'): 1.06, ('18MHz', 'V'): 1.07, ('18MHz', 'I'): 1.07}.get((sb, S), 1.05)
            write_cube(cp(sb, S), fs, seed=1)
    write_cube(cp('59MHz', 'V'), 1.05)
    write_cube(cp('59MHz', 'I'), None)                         # a frame-unit cube
    write_cube(cp('73MHz', 'V'), 1.0422, comments=1000)        # 11h 2024-12-31 73 MHz: V rescaled (header > 64 KiB) ...
    write_cube(cp('73MHz', 'I'), None, compressed=False)       # ... I not
    t0, t1 = '2026-11-14T10:00:00Z', '2026-11-15T10:00:00Z'
    rows_a = [result_row(cp(sb, S), 1.05, t0) for sb in ('55MHz', '59MHz', '41MHz', '36MHz', '32MHz') for S in 'VI'
              if (sb, S) not in (('41MHz', 'I'),)]
    rows_a += [result_row(cp('50MHz', 'V'), 1.06, t0), result_row(cp('50MHz', 'I'), 1.05, t0),
               result_row(cp('46MHz', 'V'), 1.05, t0), result_row(cp('46MHz', 'I'), 1.06, t0),
               result_row(cp('73MHz', 'V'), 1.0422, t0), result_row(cp('73MHz', 'I'), 1.0422, t0),
               result_row(cp('18MHz', 'V'), 1.07, t1), result_row(cp('18MHz', 'I'), 1.07, t0)]
    rows_b = [result_row(cp('18MHz', 'V'), 1.05, t0)]          # older row, in a file later in name order
    plan1 = [[cp('32MHz', 'V'), CELL, '32MHz', 'V', 'done', 1, '1.2'],          # RESULTS (1.05) win over this plan row
             [cp('27MHz', 'V'), CELL, '27MHz', 'V', 'done', 1, '1.05'],
             [cp('27MHz', 'I'), CELL, '27MHz', 'I', 'done', 1, '1.05'],          # removed by the newer plan below
             [cp('23MHz', 'V'), CELL, '23MHz', 'V', 'done', 1, '1.3'],           # overridden by the newer plan below
             [cp('23MHz', 'I'), CELL, '23MHz', 'I', 'done', 1, '1.05']]
    plan2 = [[cp('27MHz', 'I'), CELL, '27MHz', 'I', 'pending', 1, ''],
             [cp('23MHz', 'V'), CELL, '23MHz', 'V', 'done', 1, '1.05']]
    setup_records(ops, cubes, {'hostA': rows_a, 'hostB': rows_b},
                  {'snapshot_cube_rescale_plan_20261112.csv': plan1, 'snapshot_cube_rescale_plan_20261115_wave1.csv': plan2})
    rec = FS.records()

    def gate(sb):
        return FS.check(CELL, sb)

    ok, why, info = gate('55MHz')
    check('gate: both cubes rescaled, FSAPPLD = record, V = I -> PASS', ok and info['V']['fsappld'] == 1.05
          and info['V']['record']['source'] == 'rescale-results', (why, info))
    check('gate: the FRAMES shape is read from the tile-compressed header (ZNAXISn) and NFRAMES from HDU 0',
          info['V']['frames_shape'] == [4, 40, 40] and info['V']['nframes'] == 4, info['V'])
    ok, why, _ = gate('59MHz')
    check('gate: a Stokes cube without FSAPPLD (frame units) -> REFUSED', not ok and why == ['I: no FSAPPLD (frame-unit cube)'],
          why)
    ok, why, _ = gate('50MHz')
    check('gate: FSAPPLD 1.05 against the record 1.06 -> REFUSED', not ok and any('FSAPPLD 1.050000 != recorded 1.060000' in x
                                                                                   for x in why), why)
    ok, why, info = gate('73MHz')
    check('gate: the 11h 2024-12-31 73 MHz shape (V FSAPPLD 1.0422, I none) -> REFUSED; the V header (1000 extra cards, '
          'longer than one positioned read) and the plain FRAMES of I are read',
          not ok and why == ['I: no FSAPPLD (frame-unit cube)'] and info['V']['fsappld'] == 1.0422
          and info['I']['frames_shape'] == [4, 40, 40], (why, info))
    ok, why, _ = gate('46MHz')
    check('gate: V 1.05 and I 1.06 (each = its record) -> REFUSED by V = I', not ok and why == [
        'V FSAPPLD 1.050000 != I FSAPPLD 1.060000 (one factor per hour-band)'], why)
    ok, why, _ = gate('41MHz')
    check('gate: no record of the I cube -> REFUSED', not ok and why == [
        "I: no rescale record (neither RESULTS row nor plan 'done' row)"], why)
    ok, why, _ = gate('36MHz')
    check('gate: no I cube -> REFUSED', not ok and len(why) == 1 and why[0].startswith('I: no cube'), why)
    ok, why, info = gate('32MHz')
    check('records: a RESULTS row (1.05) wins over a plan "done" row (1.2) -> PASS', ok
          and info['V']['record']['source'] == 'rescale-results', (why, info))
    ok, why, _ = gate('27MHz')
    check('records: a newer plan whose row is not "done" removes the older plan\'s record -> REFUSED (no record of I)',
          not ok and why == ["I: no rescale record (neither RESULTS row nor plan 'done' row)"], why)
    ok, why, info = gate('23MHz')
    check('records: a newer plan\'s "done" factor (1.05) overrides the older plan\'s (1.3) -> PASS, plan-native',
          ok and info['V']['record'] == dict(source='plan-native', factor=1.05,
                                             plan='snapshot_cube_rescale_plan_20261115_wave1.csv',
                                             row='%s:3' % os.path.join(ops, 'snapshot_cube_rescale_plan_20261115_wave1.csv'),
                                             status='done'),
          (why, info))
    ok, why, info = gate('18MHz')
    check('records: of two RESULTS rows the newest by utc wins (1.07, file RESULTS_hostA) over the row of a file later in '
          'name order (1.05, older) -> PASS', ok and FS.record_of(cp('18MHz', 'V'), rec)['factor'] == 1.07
          and info['V']['record']['results'] == 'RESULTS_hostA.csv', (why, info))

    # ---- sidecar: written only after an unchanged re-check
    code = os.path.join(base, 'code')
    os.makedirs(code)
    for m in ('phase3_snapshot_search.py', 'phase3_snapshot_cutouts.py'):
        with open(os.path.join(code, m), 'w') as f:
            f.write('# stub %s\n' % m)
    root = os.path.join(base, 'root')
    regen = '2026-11-18 09:58:12,345 INFO CUTOUTS-REGEN 00h 2022-09-26 55MHz: V 3 I 3'
    ok, why, before = gate('55MHz')
    sp = FS.write_sidecar(root, CELL, '55MHz', before, regen, code)
    doc = json.load(open(sp))
    check('sidecar: written at <root>/_fluxscale/<sb>/<cell>_<sb>_fluxscale.json with the checked cubes, the regeneration '
          'line without its log time stamp, the module md5s',
          sp == os.path.join(root, '_fluxscale', '55MHz', '%s_55MHz_fluxscale.json' % CELL)
          and doc['regen'] == 'INFO CUTOUTS-REGEN 00h 2022-09-26 55MHz: V 3 I 3'
          and doc['cubes']['V']['path'] == cp('55MHz', 'V') and doc['cubes']['I']['fsappld'] == 1.05
          and doc['cubes']['V']['record_source'] == 'rescale-results' and doc['cubes']['V']['size'] == before['V']['size']
          and doc['code']['phase3_snapshot_search.py'].split()[-1] == hashlib.md5(b'# stub phase3_snapshot_search.py\n').hexdigest()
          and sorted(doc) == sorted(['product', 'cell', 'subband', 'unit', 'cubes', 'regen', 'code', 'host', 'utc']), doc)
    from exopipe.phase3.lib.photometry import stamp_engine
    check('sidecar: the stamp engine reads it (fluxscale_record, generation_counts, check_cube_unchanged)',
          stamp_engine.generation_counts(stamp_engine.fluxscale_record(root, CELL, '55MHz')[1])[0] == {'V': '3', 'I': '3'}
          and stamp_engine.check_cube_unchanged(doc, 'V', cp('55MHz', 'V')) is None)
    from exopipe.phase3.core import validate
    res = {r['name'].split('.')[-1]: r for r in validate.run('photometry_fluxscale_record', sp)}
    check('FS1 (the flux-scale record\'s content test): PASS on the record just written', res['fs1_record']['passed'], res)
    for S in 'VI':
        a = TF.check_a(root, CELL, '55MHz', S, cp('55MHz', S))
        check('(a) %s: the record = the cube (FSAPPLD, path, size, mtime) -> PASS' % S, a[0] and a[1] == 1.05, a)

    # a cube changed between the check and the sidecar: size, mtime, FSAPPLD - each refuses and writes nothing
    def changed_case(what, sb):
        ok, why, bef = gate(sb)
        p = cp(sb, 'V')
        st = os.stat(p)
        if what == 'size':
            write_cube(p, 1.05, nt=5, seed=7, mtime_ns=st.st_mtime_ns)
        elif what == 'mtime':
            os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 10 ** 9))
        else:
            data = open(p, 'rb').read()
            old = fits.Card('FSAPPLD', 1.05).image.encode()
            new = fits.Card('FSAPPLD', 1.0500015).image.encode()
            assert data.count(old) == 1 and len(old) == len(new)
            with open(p, 'wb') as f:
                f.write(data.replace(old, new))
            os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))
        try:
            FS.write_sidecar(root, CELL, sb, bef, regen, code)
            err = None
        except FS.SidecarRefused as e:
            err = str(e)
        after = FS.read_cube_header(p)
        check('sidecar: the V cube\'s %s changed between the check and the write -> REFUSED, no record written (size %s -> '
              '%s, mtime equal %s, FSAPPLD %s -> %s)' % (what, st.st_size, after['size'], after['mtime_ns'] == st.st_mtime_ns,
                                                       bef['V']['fsappld'], after['fsappld']),
              ok and err is not None and 'V cube %s changed' % {'size': 'size', 'mtime': 'mtime_ns', 'fsappld': 'fsappld'}[what]
              in err and not os.path.exists(FS.sidecar_path(root, CELL, sb)), err)
    # the records changed between the check and the write (a newer RESULTS row with another factor; the cube unchanged)
    ok, why, bef = gate('55MHz')
    setup_records(ops, cubes, {'hostA': rows_a + [result_row(cp('55MHz', 'V'), 1.06, '2026-11-16T00:00:00Z')],
                               'hostB': rows_b}, {})
    os.remove(sp)
    try:
        FS.write_sidecar(root, CELL, '55MHz', bef, regen, code)
        err = None
    except FS.SidecarRefused as e:
        err = str(e)
    check('sidecar: the record of an unchanged cube changed between the check and the write (a newer RESULTS row) -> '
          'REFUSED by the re-check, no record written', ok and err is not None and 'gate now fails' in err
          and not os.path.exists(sp), err)
    setup_records(ops, cubes, {'hostA': rows_a, 'hostB': rows_b}, {})
    sp = FS.write_sidecar(root, CELL, '55MHz', bef, regen, code)
    for what, sb in (('size', '32MHz'), ('mtime', '23MHz'), ('fsappld', '41MHz')):
        if sb == '41MHz':                                       # give the I cube a record so the gate passes before the change
            rows_a.append(result_row(cp('41MHz', 'I'), 1.05, t0))
            setup_records(ops, cubes, {'hostA': rows_a, 'hostB': rows_b}, {})
        changed_case(what, sb)

    # ---- (a) failures and FS1 failures: the 55 MHz I cube rewritten after its record
    write_cube(cp('55MHz', 'I'), 1.05, seed=3)
    a = TF.check_a(root, CELL, '55MHz', 'I', cp('55MHz', 'I'))
    check('(a): the I cube rewritten after the record (another size / mtime) -> FAIL', not a[0] and 'identity' in a[2], a)
    r = TF.fs1_record(sp)
    check('FS1: a listed cube changed since the record -> FAIL', not r[0] and 'I:' in r[2], r)
    d2 = dict(doc, regen='')
    sp2 = os.path.join(root, '_fluxscale', '55MHz', 'x', '%s_55MHz_fluxscale.json' % CELL)
    os.makedirs(os.path.dirname(sp2))
    json.dump(d2, open(sp2, 'w'))
    r = TF.fs1_record(sp2)
    check('FS1: no regeneration result line -> FAIL', not r[0] and 'no regeneration result line' in r[2], r)
    a = TF.check_a(root, CELL, '18MHz', 'V', cp('18MHz', 'V'))
    check('(a): no flux-scale record of the (cell, subband) -> FAIL', not a[0] and 'no flux-scale record' in a[2], a)
    a = TF.check_a(root, CELL, '59MHz', 'I', cp('59MHz', 'I'))
    check('(a): a frame-unit cube (no FSAPPLD) -> FAIL', not a[0] and 'no FSAPPLD' in a[2], a)
    ok, why, b18 = gate('18MHz')
    sp18 = FS.write_sidecar(root, CELL, '18MHz', b18, regen.replace('55MHz', '18MHz'), code)
    d18 = json.load(open(sp18))
    d18['cubes']['V']['fsappld'] = 1.08
    json.dump(d18, open(sp18, 'w'))
    a = TF.check_a(root, CELL, '18MHz', 'V', cp('18MHz', 'V'))
    check('(a): the record\'s FSAPPLD (1.08) differs from the cube\'s (1.07) -> FAIL', not a[0] and 'record FSAPPLD' in a[2], a)

    # ---- wideband member gate
    ex = Ex({('00h', '64MHz', '2022-09-26')})
    write_cube(cp('64MHz', 'V'), None)                         # excluded: never read, so not gated
    write_cube(cp('69MHz', 'V'), 1.05)                         # one Stokes cube only: checked
    for sb in ('78MHz', '82MHz'):
        for S in 'VI':
            write_cube(cp(sb, S), 1.05)
    extra = [result_row(cp(sb, S), 1.05, t0) for sb in ('78MHz', '82MHz') for S in 'VI'] + [result_row(cp('69MHz', 'V'), 1.05, t0)]
    rows_a += extra
    setup_records(ops, cubes, {'hostA': rows_a, 'hostB': rows_b}, {})
    check('wideband: the members read are the group\'s members not excluded with a cube (64 MHz excluded)',
          FS.members_read(CELL, 'Blue', ex=ex) == ['69MHz', '73MHz', '78MHz', '82MHz'], FS.members_read(CELL, 'Blue', ex=ex))
    ok, why, info = FS.check_wideband(CELL, 'Blue', ex=ex)
    check('wideband: a member cube without FSAPPLD (73 MHz I) -> REFUSED; the excluded member (64 MHz, frame units) is not '
          'gated; the one-Stokes member (69 MHz V) is', not ok and why == ['73MHz I: no FSAPPLD (frame-unit cube)']
          and '64MHz' not in info and set(info['69MHz']) == {'V'}, why)
    ok, why, info = FS.check_wideband(CELL, 'Blue', ex=Ex({('00h', '64MHz', '2022-09-26'), ('00h', '73MHz', '2022-09-26')}))
    check('wideband: every member read rescaled and on its record -> PASS', ok and sorted(info) == ['69MHz', '78MHz', '82MHz'],
          why)
    ok, why, info = FS.check_wideband(CELL, 'Blue', ex=Ex({('00h', s, '2022-09-26') for s in CFG.instrument.SUBBAND_BANDS['Blue']}))
    check('wideband: no member to read -> REFUSED', not ok and 'no member subband' in why[0], why)
    test_step_wideband(base, ex)
    test_review(base, ops, rows_a, rows_b, ex)

    check('regen_text: the log time stamp is cut, a line without one is kept',
          FS.regen_text(regen) == 'INFO CUTOUTS-REGEN 00h 2022-09-26 55MHz: V 3 I 3'
          and FS.regen_text('INFO CUTOUTS-REGEN x: V 1 I None') == 'INFO CUTOUTS-REGEN x: V 1 I None')

    info = {}
    TC.identity_subband([], CELL, '55MHz', 'V', frame_info=info)
    check('identity_subband frame_info: the frames cut, the first valid frame header, the frame shape and FSAPPLD of the cube '
          'read once (for check (c))', info.get('cts') == frame_ts(CELL, 4) and info.get('shape') == (40, 40)
          and info.get('fsappld') == 1.05 and info['h0']['CTYPE1'] == 'RA---CAR', info)

    test_check_b(base)
    test_check_c(base)
    test_check_w(base)
    test_gen5_sample(ops, cubes)


def write_cutout(path, data, ts):
    from astropy.io import fits
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fits.HDUList([fits.PrimaryHDU(data.astype(np.float32)), fits.BinTableHDU.from_columns(
        [fits.Column(name='ts', format='15A', array=np.array(ts))], name='TIMES')]).writeto(path, overwrite=True)
    return path


def test_check_b(base):
    """(b) on a sampled unit: new = generation 5 / FSAPPLD, file for file."""
    new, g5 = os.path.join(base, 'b_new'), os.path.join(base, 'b_gen5')
    ts = frame_ts(CELL, 4)
    rng = np.random.default_rng(5)
    fs = 1.05
    files = []
    for obj in ('O1', 'O2'):
        x = rng.normal(0, 1, (4, 12, 12)).astype(np.float32)
        x[:, 0, 0] = np.nan
        rel = os.path.join('HWJ', '23MHz', obj, '%s_00h_20220926T070000_V.fits' % obj)
        write_cutout(os.path.join(g5, rel), x, ts)
        files.append(write_cutout(os.path.join(new, rel), x / fs, ts))
    files.append(write_cutout(os.path.join(new, 'LPT', '23MHz', 'L1', 'L1_00h_20220926T070000_V.fits'),
                              np.ones((4, 12, 12)), ts))
    ok, med, det = TF.check_b(files, new, g5, CELL, '23MHz', 'V', fs)
    check('(b): new = generation 5 / FSAPPLD file for file, same set (LPT, new since generation 5, left out) -> PASS '
          '(median slope x FSAPPLD %.6f)' % med, ok and abs(med - 1) < 1e-6, det)
    ok, med, det = TF.check_b(files, new, g5, CELL, '23MHz', 'V', None)
    check('(b): no FSAPPLD -> FAIL', not ok, det)
    x = np.asarray(__import__('astropy.io.fits', fromlist=['getdata']).getdata(files[0]), np.float32)
    write_cutout(files[0], x * 1.01, ts)
    ok, med, det = TF.check_b(files, new, g5, CELL, '23MHz', 'V', fs)
    check('(b): one of two files 1%% off -> median slope %.4f, outside 1 +- 0.002 -> FAIL' % med, not ok and abs(med - 1) > 0.002,
          det)
    y = x.copy()
    y[1, 5, 5] = np.nan
    write_cutout(files[0], y, ts)
    ok, med, det = TF.check_b(files, new, g5, CELL, '23MHz', 'V', fs)
    check('(b): one NaN more in one file -> FAIL (NaN pattern)', not ok and 'NaN pattern differs in 1 files' in det, det)
    write_cutout(files[0], x, ts[::-1])
    ok, med, det = TF.check_b(files, new, g5, CELL, '23MHz', 'V', fs)
    check('(b): other TIMES -> FAIL (not comparable)', not ok and 'not comparable 1' in det, det)
    write_cutout(files[0], x, ts)
    write_cutout(os.path.join(g5, 'HWJ', '23MHz', 'O3', 'O3_00h_20220926T070000_V.fits'), x, ts)
    ok, med, det = TF.check_b(files, new, g5, CELL, '23MHz', 'V', fs)
    check('(b): generation 5 holds a file the unit did not write -> FAIL (file set)', not ok and 'same set False' in det, det)
    g5files = [os.path.join(g5, 'HWJ', '23MHz', o, '%s_00h_20220926T070000_V.fits' % o) for o in ('O1', 'O2', 'O3')]
    ok, med, det = TF.check_b(g5files, g5, g5, CELL, '23MHz', 'V', 1.0)
    check('(b): the root tested is the generation-5 tree (itself: same set, slope 1 with FSAPPLD 1) -> FAIL',
          not ok and 'generation-5 tree itself' in det, det)


def test_check_c(base):
    """(c): the distinct target set the writer selects (elevation at the cube's mid-time, inside the frame, names merged within
    1 arcmin and J-tagged beyond)."""
    from astropy.io import fits
    from exopipe.phase3.lib.snapshot_engine.phase3_snapshot_search import lst_deg_at, ts_to_epoch_seconds
    from exopipe.phase3.lib.photometry.tests import _jtag
    cts = frame_ts(CELL, 4)
    tt = [ts_to_epoch_seconds(t) for t in cts]
    lst = lst_deg_at(0.5 * (min(tt) + max(tt)))
    n = 160
    h0 = fits.Header()
    for kk, v in (('NAXIS', 2), ('NAXIS1', n), ('NAXIS2', n), ('CTYPE1', 'RA---CAR'), ('CTYPE2', 'DEC--CAR'),
                  ('CRVAL1', lst), ('CRVAL2', 0.0), ('CRPIX1', 80.5), ('CRPIX2', 80.5), ('CDELT1', -1.0), ('CDELT2', 1.0)):
        h0[kk] = v
    lst_path = os.path.join(base, 'c_list.csv')
    r = lambda d: (lst + d) % 360.0                                                     # noqa: E731
    ents = [('Alpha', r(5), 40.0), ('Beta', r(-75), 0.0), ('Gamma', r(170), 37.0), ('Delta', r(-10), 30.0),
            ('Delta', r(-10), 30.005), ('Eps', r(10), 20.0), ('Eps', r(20), 25.0), ('Zeta x', r(-3), 50.0),
            ('Eta', r(30), 85.0)]
    write_csv(lst_path, ['name', 'ra', 'dec'], [[a, '%.6f' % b, '%.6f' % c] for a, b, c in ents])
    want = {('T', 'Alpha'), ('T', 'Delta'), ('T', 'Eps ' + _jtag(r(10), 20.0)), ('T', 'Eps ' + _jtag(r(20), 25.0)),
            ('T', 'Zeta_x')}
    from exopipe.phase3.lib.photometry.tests import safe_name
    root = os.path.join(base, 'c_root')
    files = [os.path.join(root, t, '41MHz', safe_name(o), '%s_00h_20220926T070000_V.fits' % safe_name(o)) for t, o in want]
    want = {(t, safe_name(o)) for t, o in want}
    ok, nobj, det = TF.check_c(files, cts, h0, (n, n), lists={'T': lst_path}, planets=())
    check('(c): one file per selected object (1 merged pair within 1 arcmin, 2 J-tagged entries beyond; one below 20 deg '
          'elevation inside the frame, one above it outside the frame and one below the horizon left out) -> PASS (el of the '
          'one outside the frame %.0f deg)' % TF._elevation(r(30), 85.0, lst, CFG.photometry_cutouts.PLANET_SITE[0]),
          ok and nobj == 5 and TF._elevation(r(30), 85.0, lst, CFG.photometry_cutouts.PLANET_SITE[0]) > 20, det)
    ok, _, det = TF.check_c(files[1:], cts, h0, (n, n), lists={'T': lst_path}, planets=())
    check('(c): a selected object without its file -> FAIL', not ok and 'missing' in det, det)
    beta = os.path.join(root, 'T', '41MHz', 'Beta', 'Beta_00h_20220926T070000_V.fits')
    ok, _, det = TF.check_c(files + [beta], cts, h0, (n, n), lists={'T': lst_path}, planets=())
    check('(c): a file of an object below the elevation limit -> FAIL', not ok and 'unexpected' in det, det)
    dup = os.path.join(os.path.dirname(files[0]), 'other_00h_20220926T070010_V.fits')
    ok, _, det = TF.check_c(files + [dup], cts, h0, (n, n), lists={'T': lst_path}, planets=())
    check('(c): two files of one object -> FAIL', not ok, det)
    # a planet: the frame centered on Jupiter (2022-09-26, opposition), 8 deg wide; Saturn is 40 deg away
    ra, dec = TF.planet_radec('jupiter', cts[0])
    hj = h0.copy()
    hj['CRVAL1'], hj['CRVAL2'], hj['CDELT1'], hj['CDELT2'], hj['CRPIX1'], hj['CRPIX2'] = ra, dec, -0.1, 0.1, 40.5, 40.5
    el = TF._elevation(ra, dec, lst, CFG.photometry_cutouts.PLANET_SITE[0])
    jf = [os.path.join(root, 'SolarSystem', '41MHz', 'Jupiter', 'Jupiter_00h_20220926T070000_V.fits')]
    ok, nobj, det = TF.check_c(jf, cts, hj, (80, 80), lists={}, planets=('jupiter', 'saturn'))
    check('(c): a planet in the frame above the limit (Jupiter, elevation %.0f deg) is expected, one outside (Saturn) is not '
          '-> PASS' % el, ok and el > 20, det)
    ok, nobj, det = TF.check_c([], cts, hj, (80, 80), lists={}, planets=('jupiter', 'saturn'))
    check('(c): the planet without its file -> FAIL', not ok and 'Jupiter' in det, det)


class Unit:
    """The parts of core.record.Unit the cutouts step's gate functions and its placement transaction use: refuse (raises
    Refusal), check, failed_tests, meta, outputs, note, name."""
    def __init__(self):
        self.tests, self.meta, self.outputs, self.notes, self.name = [], {}, [], [], 'cutouts.test'

    def note(self, text):
        self.notes.append(text)

    def refuse(self, reason):
        raise Refusal(reason)

    def check(self, name, passed, value=None, detail='', cls=None, path=None):
        self.tests.append(dict(name=name, passed=bool(passed), value=value, detail=detail))

    def failed_tests(self):
        return [t for t in self.tests if not t['passed']]


def refused(fn, *a):
    """The refusal text of fn(*a), or None when it does not refuse."""
    try:
        fn(*a)
    except Refusal as e:
        return str(e)
    return None


def test_step_wideband(base, ex):
    """The wideband side of the step's wiring (lib/photometry/steps_cutouts.py): the member gate before the group search, its
    re-check before placement, check (w) before placement - each refuses the unit (C-FLUXSCALE, C-GENCHECK)."""
    from exopipe.phase3.lib.photometry import steps_cutouts as SC
    r = refused(SC.fluxscale_gate, Unit(), CELL, 'Blue', ex)
    check('step: the wideband member gate refuses before the group search (C-FLUXSCALE, the engine not started, the member '
          'named)', r and r.startswith('C-FLUXSCALE') and '73MHz I: no FSAPPLD' in r and 'the engine not started' in r, r)
    ex2 = Ex({('00h', '64MHz', '2022-09-26'), ('00h', '73MHz', '2022-09-26')})
    u = Unit()
    before = SC.fluxscale_gate(u, CELL, 'Blue', ex2)
    check('step: every member read passes -> the gate returns the member cubes (the re-check compares them), the record\'s '
          'meta names them', sorted(before) == ['69MHz', '78MHz', '82MHz'] and u.meta.get('fluxscale'), u.meta)
    stage = os.path.join(base, 'wb_stage')
    out = SC.fluxscale_recheck(Unit(), CELL, 'Blue', os.path.join(base, 'wb_root'), stage, before, None, ex2)
    check('step: the member re-check after the cut, nothing changed -> no refusal, no flux-scale record (widebands have none)',
          out == ([], []), out)
    p = FS.cube_path(CELL, '78MHz', 'V')
    st = os.stat(p)
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 10 ** 9))
    try:
        r = refused(SC.fluxscale_recheck, Unit(), CELL, 'Blue', os.path.join(base, 'wb_root'), stage, before, None, ex2)
    finally:
        os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))
    check('step: a member cube changed during the cut (mtime) -> C-FLUXSCALE before placement, the member named',
          r and r.startswith('C-FLUXSCALE') and '78MHz' in r and 'changed' in r, r)
    rels = ['HWJ/Blue/X/X_00h_2022-09-26_Blue_V.fits', 'HWJ/Blue/X/X_00h_2022-09-26_Blue_I.fits']

    class W:
        def __init__(self, res):
            self.res = res

        def result(self):
            return self.res
    good, bad = W((True, 1e-7, 'ok')), W((False, 0.3, 'weights off by 30%'))
    args = (CELL, 'Blue', stage, rels, None, False, ex2, before, {})
    u = Unit()
    r = refused(SC.generation_checks, u, *args, {'V': good, 'I': good})
    check('step: (w) passing for both Stokes -> no refusal, GW_weights_V / _I recorded',
          r is None and [t['name'].split('.')[-1] for t in u.tests] == ['GW_weights_V', 'GW_weights_I'], (r, u.tests))
    r = refused(SC.generation_checks, Unit(), *args, {'V': good, 'I': bad})
    check('step: (w) failing for one Stokes -> C-GENCHECK before placement, the check named', r and r.startswith('C-GENCHECK')
          and 'GW_weights_I' in r and 'weights off by 30%' in r, r)
    r = refused(SC.generation_checks, Unit(), *args, {'V': good})
    check('step: a Stokes cut without its weight check -> C-GENCHECK (never a silent pass)', r and r.startswith('C-GENCHECK')
          and 'no weight check ran' in r, r)


@contextlib.contextmanager
def record_file(path, cols, rows):
    """A rescale record file present for the block, removed after it."""
    write_csv(path, cols, rows)
    try:
        yield path
    finally:
        os.remove(path)


def records_error():
    try:
        FS.records()
    except FS.RecordsError as e:
        return str(e)
    except Exception as e:                       # noqa: BLE001 - any other exception is the defect under test
        return 'NOT A RecordsError: %s: %s' % (type(e).__name__, e)
    return None


def test_review(base, ops, rows_a, rows_b, ex):
    """PR 153 review findings 1, 3 (the resume identity), 4, 5, 6, 8, 9, 10 (2 and 3 also run end to end in
    tests/test_photometry.py, 5's reader in tests/test_photometry_writer_safety.py)."""
    from exopipe.phase3.lib.photometry import steps_cutouts as SC
    rdir = os.path.join(ops, 'snapshot_cube_rescale_results')
    z, y = os.path.join(rdir, 'RESULTS_hostZ.csv'), os.path.join(rdir, 'RESULTS_hostY.csv')
    t0 = '2026-11-14T10:00:00Z'
    good = result_row(cp('55MHz', 'V'), 1.05, t0)
    # ---- 4 / 9: a malformed row -> RecordsError naming <file>:<line>
    for what, cols, rows, want in (
            ('a non-numeric fsappld', RES_COLS, [good, good[:2] + ['abc'] + good[3:]],
             "%s:3 (fsappld 'abc' is not a number)" % z),
            ('an empty fsappld (no writer writes one: snapshot_cube_rescale.py appends a row only after it rewrote the cube, '
             'always with its factor)', RES_COLS, [good[:2] + [''] + good[3:]], '%s:2 (no fsappld' % z),
            ('a NaN fsappld', RES_COLS, [good[:2] + ['nan'] + good[3:]], "%s:2 (fsappld 'nan' is not a finite positive" % z),
            ('an empty utc', RES_COLS, [[''] + good[1:]], "%s:2 (utc '' is not an ISO UTC time" % z),
            ('a utc that is not ISO', RES_COLS, [['yesterday'] + good[1:]], "%s:2 (utc 'yesterday'" % z),
            ('a short row', RES_COLS, [good, good[:3]], '%s:3 (3 fields, the header has 7)' % z),
            ('no utc column', RES_COLS[1:], [good[1:]], '%s:1 (no column utc' % z),
            ('an empty cube', RES_COLS, [good[:1] + [''] + good[2:]], '%s:2 (no cube)' % z)):
        with record_file(z, cols, rows):
            e = records_error()
        check('records (review 4/9): %s -> RecordsError "no rescale record readable: <file>:<line>"' % what,
              e is not None and ('no rescale record readable: ' + want) in e, (e, want))
    plan = os.path.join(ops, 'snapshot_cube_rescale_plan_20261120_bad.csv')
    with record_file(plan, PLAN_COLS, [[cp('27MHz', 'V'), CELL, '27MHz', 'V', 'done', 1, 'x']]):
        e = records_error()
    check('records (review 4): a plan\'s "done" row with a non-numeric factor -> RecordsError naming the plan and its line',
          e is not None and "no rescale record readable: %s:2 (factor 'x' is not a number)" % plan in e, e)
    with record_file(z, RES_COLS, [good, good[:2] + ['abc'] + good[3:]]):
        r = refused(SC.fluxscale_gate, Unit(), CELL, '55MHz', ex)
        dry = SC.fluxscale_dry(CELL, '55MHz', ex)
        u = Unit()
        SC.fluxscale_in_place(u, CELL, '55MHz', {}, ex)()
    check('step (review 4): a malformed record row refuses the unit before the engine (C-FLUXSCALE naming <file>:<line>)',
          r and r.startswith('C-FLUXSCALE: no rescale record readable: %s:3' % z) and 'the engine not started' in r, r)
    check('dry run (review 4): the malformed row reported for the unit (would refuse C-FLUXSCALE, <file>:<line>), no crash',
          dry.startswith('would refuse C-FLUXSCALE: no rescale record readable: %s:3' % z), dry)
    check('in-place check (review 4): records unreadable during placement -> the check fails (the placement is undone), no '
          'crash', [t['passed'] for t in u.tests] == [False] and 'RecordsError' in u.tests[0]['detail'], u.tests)
    # ---- 9: a tie at the newest utc
    tn = '2026-11-16T00:00:00Z'
    with record_file(z, RES_COLS, [result_row(cp('55MHz', 'V'), 1.05, tn)]), \
            record_file(y, RES_COLS, [result_row(cp('55MHz', 'V'), 1.06, tn)]):
        ok, why, info = FS.check(CELL, '55MHz')
        dry = SC.fluxscale_dry(CELL, '55MHz', ex)
    check('records (review 9): two RESULTS rows of one cube share the newest utc with other factors (1.05, 1.06) -> the cube '
          'refused as ambiguous, both <file>:<line> named', not ok and len(why) == 1 and 'V: ambiguous rescale record' in why[0]
          and '%s:2 fsappld 1.060000' % y in why[0] and '%s:2 fsappld 1.050000' % z in why[0]
          and info['V']['record']['source'] == 'ambiguous' and 'ambiguous' in dry, (why, dry))
    with record_file(z, RES_COLS, [result_row(cp('55MHz', 'V'), 1.05, tn)]), \
            record_file(y, RES_COLS, [result_row(cp('55MHz', 'V'), 1.05, tn)]):
        ok, why, _ = FS.check(CELL, '55MHz')
    check('records (review 9): a tie with the same factor -> PASS', ok, why)
    with record_file(z, RES_COLS, [result_row(cp('55MHz', 'V'), 1.05, tn),
                                   result_row(cp('55MHz', 'V'), 1.05, '2026-11-17T00:00:00Z')]), \
            record_file(y, RES_COLS, [result_row(cp('55MHz', 'V'), 1.06, tn)]):
        ok, why, info = FS.check(CELL, '55MHz')
    check('records (review 9): a newer row after a tie decides -> PASS on the newest', ok
          and info['V']['record']['utc'] == '2026-11-17T00:00:00Z', why)
    with record_file(z, RES_COLS, [result_row(cp('55MHz', 'V'), 1.06, '2026-11-16T00:00:00Z')]), \
            record_file(y, RES_COLS, [result_row(cp('55MHz', 'V'), 1.05, '2026-11-16T00:00:00.500000Z')]):
        ok, why, info = FS.check(CELL, '55MHz')
    check('records (review 9): rows order by time, not by text (00:00:00.5Z is newer than 00:00:00Z; as text it sorts first) '
          '-> the 1.05 row wins, PASS', ok and info['V']['record']['factor'] == 1.05, why)
    # ---- 8: records() parsed once per signature of the record files
    r1, r2 = FS.records(), FS.records()
    with record_file(z, RES_COLS, [good]):
        r3 = FS.records()
    check('records (review 8): parsed once while the record files are unchanged (the same object), again when one changes',
          r1 is r2 and r3 is not r1, (id(r1), id(r2), id(r3)))
    # ---- 6: the identity carries the inode and the ctime (fstat of the descriptor the header is read from)
    ok, why, bef = FS.check(CELL, '55MHz')
    p = cp('55MHz', 'V')
    tmp = p + '.copy'
    shutil.copy2(p, tmp)
    os.replace(tmp, p)
    ok2, why2, aft = FS.check(CELL, '55MHz')
    ch = FS.changed(bef, aft)
    try:
        FS.write_sidecar(os.path.join(base, 'root6'), CELL, '55MHz', bef, 'x', os.path.join(base, 'code'))
        err = None
    except FS.SidecarRefused as e:
        err = str(e)
    check('identity (review 6): the V cube replaced by a copy of the same size with its mtime copied (cp -p) -> a change '
          '(inode, ctime), the flux-scale record refused', ok and ok2 and aft['V']['size'] == bef['V']['size']
          and aft['V']['mtime_ns'] == bef['V']['mtime_ns'] and aft['V']['ino'] == os.stat(p).st_ino
          and any('V cube ino changed' in x for x in ch) and err and 'V cube ino changed' in err, (ch, err))
    # ---- 5: the wideband gate checks the engine's member list
    mem = ['69MHz', '78MHz', '82MHz']
    ok, why, info = FS.check_wideband(CELL, 'Blue', ex=ex, members=mem)
    check('wideband (review 5): the engine\'s member list without 73 MHz -> its frame-unit I cube not gated, PASS on '
          'exactly those members', ok and sorted(info) == mem, (why, sorted(info)))
    ok, why, _ = FS.check_wideband(CELL, 'Blue', ex=ex, members=['64MHz'] + mem)
    check('wideband (review 5): the engine\'s member list with 64 MHz (core: excluded) -> its frame-unit cube gated, REFUSED',
          not ok and '64MHz V: no FSAPPLD (frame-unit cube)' in why and all(x.startswith('64MHz V: ') for x in why), why)
    u = Unit()
    before = SC.fluxscale_gate(u, CELL, 'Blue', ex, mem)
    check('step (review 5): fluxscale_gate gates the members it is given (the engine\'s), not core\'s list',
          sorted(before) == mem, sorted(before))
    # ---- 1: the member cubes re-checked INSIDE the placement transaction
    root, stage = os.path.join(base, 'tx_root'), os.path.join(base, 'tx_stage')
    os.makedirs(stage)
    rels = ['HWJ/Blue/X/a.fits', 'HWJ/Blue/X/b.fits']

    def staged():
        for r in rels:
            os.makedirs(os.path.dirname(os.path.join(stage, r)), exist_ok=True)
            with open(os.path.join(stage, r), 'w') as f:
                f.write(r)
        return [(os.path.join(stage, r), r) for r in rels]
    u = Unit()
    pairs = staged()
    SC.placement(u, None, root, pairs, SC.fluxscale_in_place(u, CELL, 'Blue', before, ex, mem), 'test')
    check('placement (review 1): the member cubes unchanged -> placed, GP_cubes_unchanged passed inside the transaction',
          all(os.path.isfile(os.path.join(root, r)) for r in rels)
          and [(t['name'].split('.')[-1], t['passed']) for t in u.tests] == [('GP_cubes_unchanged', True)], u.tests)
    shutil.rmtree(root)
    w = os.path.join(rdir, 'RESULTS_hostW.csv')
    for what, act, sign in (('replaced by a same-size copy with its mtime copied (78 MHz V)', 'copy', '78MHz V cube ino changed'),
                            ('appearing (69 MHz I, a member\'s other Stokes)', 'stokes', '69MHz I cube path changed'),
                            ('appearing for a member that had none (64 MHz V, rescaled and recorded: it passes the gate)',
                             'member', '64MHz V cube path changed')):
        q = {'copy': cp('78MHz', 'V'), 'stokes': cp('69MHz', 'I'), 'member': cp('64MHz', 'V')}[act]
        mem_ = mem + ['64MHz'] if act == 'member' else mem
        if act == 'member':
            os.rename(q, q + '.aside')                         # no 64 MHz cube at the check before the engine
        before = SC.fluxscale_gate(Unit(), CELL, 'Blue', ex, mem_)
        u = Unit()
        pairs = staged()
        chk = SC.fluxscale_in_place(u, CELL, 'Blue', before, ex, mem_)

        def outputs():
            if act == 'copy':
                shutil.copy2(q, q + '.copy')
                os.replace(q + '.copy', q)
            else:
                write_cube(q, 1.05)
                write_csv(w, RES_COLS, [result_row(q, 1.05, t0)])
            chk()
        r = refused(SC.placement, u, None, root, pairs, outputs, 'test')
        if act != 'copy':
            os.remove(q)
            os.remove(w)
        if act == 'member':
            os.rename(q + '.aside', q)
        check('placement (review 1): a member cube %s during the placement -> C-PLACE, every move undone (the files back in '
              'the staging directory, none in the root)' % what, r and r.startswith('C-PLACE') and 'GP_cubes_unchanged' in r
              and all(os.path.isfile(a) for a, _ in pairs) and not any(os.path.exists(os.path.join(root, x)) for x in rels)
              and u.meta.get('placement_undone', {}).get('placed') == 2 and sign in u.tests[0]['detail']
              and (act != 'member' or '64MHz' not in before), (r, u.meta, u.tests))
    # ---- 8: the generation-5 sample drawn once per LST hour per stage
    calls, real = [], FS.gen5_sample
    FS.gen5_sample = lambda **k: calls.append(k) or [('00h_2022-09-26', '55MHz')]
    try:
        g = SC.Gen5Samples()
        got = [g.of('00h', ex), g.of('00h', ex), g.of('01h', ex)]
        u = Unit()
        refused(SC.generation_checks, u, CELL, '55MHz', stage, [], None, False, ex, {}, {}, {}, g)
    finally:
        FS.gen5_sample = real
    check('gen5 sample (review 8): drawn once per LST hour in a stage (2 draws for 3 asks of 2 hours); a unit takes the '
          'stage\'s draw', len(calls) == 2 and g.calls == 2 and got[0] is got[1]
          and [c['lsts'] for c in calls] == [['00h'], ['01h']] and u.meta.get('generation5_sample') is True, (calls, u.meta))
    check('one cube-path helper (review 10): fluxscale.cube_path is tests_cutouts.cube_path', FS.cube_path is TC.cube_path)
    # ---- 3 (follow-up): a unit's resume identity is the RESOLVED record of each cube it reads, not the record files
    pid = os.path.join(ops, 'snapshot_cube_rescale_plan_20261121_identity.csv')
    nv, ghost = cp('55MHz', 'V'), os.path.join(base, 'identity_test_V_flowcube.fits')    # ghost: no record anywhere yet
    tz = '2026-11-30T00:00:00Z'                                                         # newer than every fixture row
    cubes = [nv, ghost]
    with record_file(z, RES_COLS, [good]):
        r0, d0 = SC.rescale_identity(cubes)
        write_csv(z, RES_COLS, [good, result_row(cp('78MHz', 'I'), 1.2, tz)])
        r1, d1 = SC.rescale_identity(cubes)
        write_csv(z, RES_COLS, [good, result_row(cp('78MHz', 'I'), 1.2, tz), result_row(nv, 1.05, tz)])
        r2, d2 = SC.rescale_identity(cubes)
        with record_file(pid, PLAN_COLS, [[ghost, CELL, '99MHz', 'V', 'done', 1, '1.05']]):
            r3, d3 = SC.rescale_identity(cubes)
            write_csv(pid, PLAN_COLS, [[ghost, CELL, '99MHz', 'V', 'failed', 1, '1.05']])
            r4, d4 = SC.rescale_identity(cubes)
        write_csv(z, RES_COLS, [good, good[:2] + ['abc'] + good[3:]])
        r5, d5 = SC.rescale_identity(cubes)
    check('resume identity (review 3, follow-up): a RESULTS row appended for ANOTHER cube (78 MHz I) leaves the unit\'s '
          'identity unchanged (no re-run on --resume)', r1 == r0 and d1 == d0 and r0[ghost] is None, (r0, r1))
    check('resume identity (review 3, follow-up): a newer RESULTS row of the unit\'s cube (55 MHz V, the same factor) changes '
          'it (a re-run): the record names that row (<file>:<line>, utc, fsappld); the other cube\'s unchanged',
          d2 != d0 and r2[nv] == {'source': 'rescale-results', 'row': '%s:4' % z, 'utc': tz, 'factor': 1.05}
          and r2[ghost] == r0[ghost], r2)
    check('resume identity (review 3, follow-up): a plan "done" row for the cube names the plan row, status and factor; the '
          'row changing status (done -> failed) changes the identity (a re-run: the gate then finds no record)',
          r3[ghost] == {'source': 'plan-native', 'row': '%s:2' % pid, 'status': 'done', 'factor': 1.05} and r4[ghost] is None
          and d3 != d2 and d4 != d3, (r3, r4))
    check('resume identity (review 3, follow-up): records that cannot be read -> no resolved records and a text no done unit '
          'carries (the gate refuses C-FLUXSCALE)', r5 == {} and d5.startswith('rescale records unreadable: RecordsError: no '
                                                                               'rescale record readable: %s:3' % z), d5)
    setup_records(ops, None, {'hostA': rows_a, 'hostB': rows_b}, {})


def test_check_w(base):
    """(w): the member weights against 1 / (F rms^2) recomputed here; the engine's dewarp and rms (the v2 port) only build the
    table of the passing case."""
    from scipy import ndimage
    from exopipe.phase3.lib.snapshot_engine.phase3_snapshot_group_search import dewarp_frame
    from exopipe.phase3.lib.snapshot_engine.phase3_snapshot_search import central_rms
    rng = np.random.default_rng(9)
    a = rng.normal(0, 1, (90, 70)).astype(np.float32)
    a[45, 35] = np.nan
    a[40:43, 30] = np.inf
    a[0, 0] = np.nan
    a[-1, -1] = np.nan
    z = np.zeros(a.shape, np.float32)
    ref = dewarp_frame(a, z, z)
    mine = TF.identity_dewarp(a)
    same = (mine == ref) | (np.isnan(mine) & np.isnan(ref))
    gy, gx = np.mgrid[0:a.shape[0], 0:a.shape[1]]
    check('(w): the identity dewarp written out = scipy map_coordinates (order 1, constant NaN) pixel for pixel, NaN and inf '
          'neighbors included (%d NaN after, %d before)' % (np.isnan(mine).sum(), np.isnan(a).sum()),
          bool(same.all()) and mine.dtype == np.float32 and np.isnan(mine).sum() > np.isnan(a).sum()
          and np.array_equal(np.isnan(ndimage.map_coordinates(a, [gy, gx], order=1, mode='constant', cval=np.nan)),
                             np.isnan(mine)))
    check('(w): the central rms written out = the search\'s', TF.central_rms(mine) == central_rms(ref))

    # member cubes of the Red group (two members, 6 frames, a NaN block across the center: the NaN spread changes the rms)
    cube_root = CFG.paths.SNAPSHOT_CUBES_ROOT
    nt, n = 6, 80
    ts = frame_ts(CELL, nt)
    F = {'27MHz': 1.05, '32MHz': 0.97}
    cubes = {}
    for i, sb in enumerate(F):
        x = rng.normal(0, 1 + 0.2 * i, (nt, n, n)).astype(np.float32)
        x[:, 38:42, 20:60:3] = np.nan
        for S in 'V':
            write_cube(FS.cube_path(CELL, sb, S, os.path.join(base, 'wcubes')), F[sb], frames=x, compressed=False)
        cubes[sb] = x
    rows = []
    for sb in F:
        for k in range(nt):
            w = 1.0 / (F[sb] * central_rms(dewarp_frame(cubes[sb][k], np.zeros((n, n), np.float32),
                                                       np.zeros((n, n), np.float32))) ** 2)
            rows.append(dict(group='Red', ts=ts[k], date_obs='', subband=sb, member_ts=ts[k], member_date_obs='',
                             weight='%.6g' % w))
    mem = os.path.join(base, 'wideband_members_V.csv')
    with open(mem, 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    wc = TF.WeightCheck(rows)
    TC.wideband_stack(CELL, 'V', mem, 'Red', [], cube_root=os.path.join(base, 'wcubes'), member_hook=wc)
    ok, worst, det = wc.result()
    check('(w): the table\'s weights = 1/(F rms^2) recomputed from the member cubes read by the identity test\'s stack '
          'rebuild (its member hook) -> PASS (%s)' % det, ok and wc.n == 2 * nt and worst <= 2e-5, det)
    mix = {t: k for k, t in enumerate(ts)}
    wc = TF.WeightCheck(rows)
    wc('27MHz', None, cubes['27MHz'], mix, rows)
    wc('32MHz', F['32MHz'], cubes['32MHz'], mix, rows)
    ok, _, det = wc.result()
    check('(w): a member cube without FSAPPLD -> FAIL', not ok and 'NO FSAPPLD' in det, det)
    bad = [dict(r, weight='%.6g' % (float(r['weight']) * F[r['subband']])) for r in rows]   # frame-unit weights 1/rms^2
    wc = TF.WeightCheck(bad)
    for sb in F:
        wc(sb, F[sb], cubes[sb], mix, bad)
    ok, worst, det = wc.result()
    check('(w): weights 1/rms^2 without the factor (the frame-unit form) -> FAIL (worst %.3f)' % worst, not ok and worst > 0.02, det)
    raw = [dict(r, weight='%.6g' % (1.0 / (F[r['subband']] * central_rms(cubes[r['subband']][mix[r['member_ts']]]) ** 2)))
           for r in rows]
    wc = TF.WeightCheck(raw)
    for sb in F:
        wc(sb, F[sb], cubes[sb], mix, raw)
    ok, worst, det = wc.result()
    check('(w): weights from the rms of the frame without the dewarp\'s NaN spread -> FAIL (worst %.2e)' % worst,
          not ok and worst > 2e-5, det)
    hole = {sb: c.copy() for sb, c in cubes.items()}
    hole['27MHz'][:, 20:60, 20:60] = np.nan
    wc = TF.WeightCheck(rows)
    for sb in F:
        wc(sb, F[sb], hole[sb], mix, rows)
    ok, _, det = wc.result()
    check('(w): a member frame whose recomputed rms is not finite (center blank) -> FAIL (never a silent pass)',
          not ok and 'no finite recomputed weight' in det, det)
    wc = TF.WeightCheck([])
    check('(w): no member row checked -> FAIL', not wc.result()[0])


def test_gen5_sample(ops, cubes):
    """The generation-5 comparison sample: md5 order per LST hour, eligibility, generation membership."""
    fx = CFG.photometry_fluxscale
    done = os.path.join(ops, fx.GEN5_DONE_DIR)
    os.makedirs(done)
    qdir = os.path.join(ops, 'quarantine')
    os.makedirs(qdir)
    jobs = [('%s_2022-09-%02d' % (lst, d), sb) for lst in ('00h', '01h') for d in (20, 21) for sb in ('18MHz', '23MHz', '27MHz')]
    rows = []
    t_old, t_mark = 1.6e9, 1.7e9                     # 2020 and 2023: both before now
    for cell, sb in jobs:
        for S in 'VI':
            p = FS.cube_path(cell, sb, S)
            write_cube(p, 1.05, n=8, nt=1)
            q = os.path.join(qdir, os.path.basename(p))
            open(q, 'w').close()
            os.utime(q, (t_old, t_old))
            rows.append(result_row(p, 1.05, '2026-11-14T10:00:00Z', quarantine=q))
        with open(os.path.join(os.path.dirname(FS.cube_path(cell, sb, 'V')), 'MANIFEST.txt'), 'w') as f:
            f.write('COMPLETE 2026-11-13\n')
        mk = os.path.join(done, '%s_%s' % (cell, sb))
        open(mk, 'w').close()
        os.utime(mk, (t_mark, t_mark))
    with open(os.path.join(ops, fx.GEN5_JOBS), 'w') as f:
        f.write('# (cell, subband) with both flow cubes\n' + ''.join('%s %s\n' % j for j in jobs))
    with open(os.path.join(ops, 'snapshot_cube_rescale_results', 'RESULTS_hostC.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(RES_COLS)
        w.writerows(rows)
    order = {lst: sorted((hashlib.md5(('%s_%s' % j).encode()).hexdigest(), j) for j in jobs if j[0][:3] == lst)
             for lst in ('00h', '01h')}
    first = [j for lst in ('00h', '01h') for _, j in order[lst][:2]]
    got = FS.gen5_sample(ex=Ex())
    check('gen5 sample: the first 2 per LST hour in md5(<cell>_<subband>) order', got == first, (got, first))
    # make the first of 00h ineligible three ways in turn, and the second of 01h not a generation job
    j0 = order['00h'][0][1]
    os.remove(os.path.join(done, '%s_%s' % j0))
    got = FS.gen5_sample(ex=Ex())
    check('gen5 sample: no generation-5 done marker -> skipped, the next in md5 order taken',
          got[:2] == [order['00h'][1][1], order['00h'][2][1]], got)
    mk0 = os.path.join(done, '%s_%s' % j0)
    open(mk0, 'w').close()
    os.utime(mk0, (t_mark, t_mark))                                         # the marker back ...
    q = os.path.join(qdir, os.path.basename(FS.cube_path(j0[0], j0[1], 'I')))
    os.utime(q, None)                                                       # ... and the I original newer than it
    got = FS.gen5_sample(ex=Ex())
    check('gen5 sample: a quarantine original newer than the generation-5 marker -> skipped', j0 not in got, got)
    os.remove(q)
    got2 = FS.gen5_sample(ex=Ex())
    check('gen5 sample: a quarantine original that does not exist -> skipped', j0 not in got2, got2)
    j1 = order['01h'][1][1]
    got = FS.gen5_sample(ex=Ex({(j1[0][:3], j1[1], j1[0].split('_', 1)[1])}))
    check('gen5 sample: an excluded (cell, subband) is not a generation job -> skipped', j1 not in got
          and got[2:] == [order['01h'][0][1], order['01h'][2][1]], got)
    got = FS.gen5_sample(jobs=[order['01h'][2][1]], ex=Ex())
    check('gen5 sample: with an explicit generation job list only its jobs are candidates', got == [order['01h'][2][1]], got)


if __name__ == '__main__':
    sys.exit(main())
