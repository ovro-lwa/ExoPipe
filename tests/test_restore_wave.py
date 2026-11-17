#!/usr/bin/env python3
"""STANDING TEST - the restore-only wave's per-batch test (lib/catalog/restore_wave.py, port of exopipe_ops
snapshot_restore_wave_diff.py 8ff3fc8, the V3 round 2026-11-05, and its Release 1 changes 9cfcca4..0904351, ported 2026-11-16), the
run a new run REPLACES (catalog --tier snapshot --stages wavediff), the view's G-V11 and the snapshot serve's G27 on such a view.

  pin        the ported production file is the pinned one (tests/production_pin.py)
  parity     synthetic OLD / NEW batch runs holding every class (served, card2, secondary by the removing stage's group, by a
             fallback grouping and by the previous identity's members, secondary-pool, 3d-gate, other-later, other-earlier,
             not-served, excluded - a cell-band the REAL exclusion list excludes now -, reprocessed - re-found served at the same
             stamp, re-found not served at a stamp one frame away, not re-found - from a table a recorded re-run replaced between
             the two records, its MANIFEST.csv beside one stamped before the old record) and every FAIL (an absent detection, the
             restored count off the summary, a tag off its pair, a page detections.csv without the column, the chain's tagged rows
             off the record, no gate3d_restore_only, a detection served on an excluded cell-band, a re-run directory without a
             UTC stamp), and a FIRST VERSION (OLD none; refused when the batch has another run): PRODUCTION's pinned script (under
             its interpreter, writing into NEW; its REMADE_ROOTS line pointed at the world's re-run root) and the port (writing into
             scratch, the same OLD / NEW paths) give the same csv and json byte for byte (the json's read time aside) and the same
             exit (PASS / FAIL / refusal); the totals line likewise
  content    class restore_wave_diff reads every world's files back against the two records (passes on the PASS worlds, fails on
             the FAIL ones and on a tampered cause, count, status, re-run directory or served-on-excluded count)
  stage      the chain step's wavediff stage through the real CLI: the replaced run from --snap-replaces or the wave schedule
             (catalog_snapshot.REPLACES_SCHEDULES; the re-run roots catalog_snapshot.RESTORE_WAVE_REMADE_ROOTS), a first version
             (--snap-replaces none, and refused with another run of its batch), NO-INPUT for a run without the restore-only gate, P-G22 (no replaced run,
             itself, no record), P-G23 (no pages), P-G24 (FAIL: refused, its files written), the PASS: done, class restore_wave_diff
  view       G-V11 (lib/serve/snapshot_view.restore_wave_gate): a restore-only member without its test, with a FAIL one, one made
             for another run or older than its record refuse; with the stage's PASS test the gate passes and returns the totals
  serve      G27 (lib/serve/snapshot_copy.restore_only_go) accepts a view whose restore-only member carries the stage's PASS test
             (with Gregg's go naming the view) and refuses it without the test
  real       READ ONLY, where the review disk is mounted: the port on the first served diff whose replaced run (its json's 'old')
             still holds its record, or that is a first version, made by the pinned production script = production's own files in
             the new run, byte for byte (the json's read time aside; a changed exclusion list since is named). Skipped when no such
             pair is left
Scratch: /tmp/exopipe3_tests_restore_wave/ (removed at the end). Production trees are only read.
Run standalone:  python tests/test_restore_wave.py [test_name ...]
"""
import collections
import csv
import filecmp
import glob
import gzip
import json
import os
import re
import shutil
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
os.environ.setdefault('EXOPIPE3_SITE', '/lustre/gh/main/exopipe_ops/exopipe3_site.py')
import production_pin as PIN        # noqa: E402
SCRATCH = '/tmp/exopipe3_tests_restore_wave'
PY36 = '/opt/devel/pipeline/envs/development/bin/python'
PY38 = '/opt/devel/pipeline/envs/py38_orca_nkosogor/bin/python3'
RB = 'restore-only 3-D gate 20261104'
csv.field_size_limit(10 ** 9)


class Skip(Exception):
    pass


def _RW():
    from exopipe.phase3.lib.catalog import restore_wave as RW
    return RW


def _pin():
    return PIN.materialize(os.path.join(SCRATCH, 'production_pin'), ['snapshot_restore_wave_diff.py'])


OLD_COLS = ['det_uid', 'cand_id', 'cell', 'subband', 'band', 'stokes', 'tier', 'ts', 'snr', 'identity_jname', 'served',
            'catalog_promoted', 'xm', 'ra_peak', 'dec_peak']
NEW_COLS = ['det_uid', 'cand_id', 'cell', 'subband', 'band', 'stokes', 'tier', 'ts', 'snr', 'ra_peak', 'dec_peak', 'served',
            'first_removed_by', 'removed_detail', 'identity_jname', 'gate_3d', 'gate_3d_installed', 'gate_3d_restore',
            'restored_by', 'catalog_promoted', 'xm', 'src_index', 'identity_index', 'anchor_cluster', 'assoc_cluster',
            'assoc0_cluster']


def _gz(path, cols, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, 'wt', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, '') for c in cols})


def _csv(path, cols, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, '') for c in cols})


def det(i, **kw):
    """A detection row: det_uid u<i>, cand_id / position of its own."""
    d = dict(det_uid=f'u{i:03d}', cand_id=f'07h_2025-10-29_46MHz_I_10s_20251029_07{i:04d}', cell='07h_2025-10-29', subband='46MHz',
             band='Green', stokes='I', tier='10s', ts=f'20251029_07{i:04d}', snr='%.2f' % (8.0 + i / 100.0),
             ra_peak='%.5f' % (100.0 + i / 10.0), dec_peak='%.5f' % (20.0 + i / 10.0), identity_jname=f'J{i:04d}+2000')
    d.update(kw)
    return d


T0 = 1762000000         # the OLD record's mtime (2025-11-01 12:26:40 UTC); NEW's T0 + 7200; the recorded re-run between, at T0 + 3600


def stamp(t):
    return time.strftime('%Y%m%dT%H%M%SZ', time.gmtime(t))


def exc_key():
    """(lst, subband, date) of a cell-band the REAL exclusion list excludes now (the snapshot serve gate's read; a subband row other
    than 18 MHz), after checking that the worlds' own cell-band (07h 2025-10-29 46 MHz) is not excluded"""
    from exopipe.phase3.lib.serve.exclusion_gate import fresh_exclusions
    ex = fresh_exclusions('snapshot')
    assert not ex.is_excluded('07h', '46MHz', '2025-10-29'), "the worlds' cell-band 07h 2025-10-29 46 MHz is excluded now"
    return sorted(r for r in ex.rows if r[1].isdigit() and r[1] != '18' and len(r[0]) == 3 and r[0].endswith('h'))[0]


def vdet(t, x, y, **kw):
    """a Stokes V detection of the re-made table candidates_46MHz_V_median.csv at 07:<t> (HHMMSS), peak pixel (x, y)"""
    cand = f'07h_2025-10-29_46MHz_V_10s_20251029_{t}'
    d = dict(det_uid=f'{cand}_{x}_{y}', cand_id=cand, cell='07h_2025-10-29', subband='46MHz', band='Green', stokes='V', tier='10s',
             ts=f'20251029_{t}', snr='7.50', ra_peak='%.5f' % (150.0 + x / 100.0), dec_peak='%.5f' % (30.0 + y / 100.0),
             identity_jname=f'JV{t}')
    d.update(kw)
    return d


def remade_root(new):
    """the world's re-run root (<world>/remade), given its NEW run"""
    return os.path.join(os.path.dirname(os.path.dirname(new)), 'remade')


def wdiff(old, new, out=None):
    """the port on a world (its own re-run root; no supersede records)"""
    return _RW().diff(old, new, out=out, remade_roots=[remade_root(new)], records_roots=[])


# the classes: (old row fields, new row fields or None = absent from the new record)
CASES = [
    ('served', {}, dict(served='1')),
    ('card2', {}, dict(served='0', first_removed_by='card2', removed_detail='I under it > 20 x |V|')),
    ('3d-gate', {}, dict(served='0', first_removed_by='bright-parent-3d', gate_3d='remove', gate_3d_installed='remove',
                         gate_3d_restore='remove')),
    ('secondary', {}, dict(served='0', first_removed_by='asymmetry', assoc_cluster='77', src_index='5')),          # own: src_index
    ('secondary', {}, dict(served='0', first_removed_by='assoc', assoc_cluster='88')),                             # own column
    ('secondary', dict(identity_jname='J0999+2000'), dict(served='0', first_removed_by='no-route', src_index='9')),  # prev identity
    ('secondary-pool', dict(catalog_promoted='gaia', xm='Gaia DR3 1(sep=1.0 E=0.485)'),
     dict(served='0', first_removed_by='no-route', src_index='11', xm='Gaia DR3 1(sep=1.0 E=0.521)')),
    ('other-later', {}, dict(served='0', first_removed_by='seed-veto', src_index='12')),
    ('other-earlier', {}, dict(served='0', first_removed_by='domain')),
    ('not-served', {}, dict(served='0')),
]


def world(name, fail=None):
    """OLD and NEW runs under SCRATCH/<name>/review (one CASES row each, plus restored and gained detections); ``fail`` makes one
    of the FAIL worlds. Returns (OLD, NEW)."""
    root = os.path.join(SCRATCH, name)
    shutil.rmtree(root, ignore_errors=True)
    rv = os.path.join(root, 'review')
    OLD, NEW = os.path.join(rv, 'Run_SNAPB9X_20261025'), os.path.join(rv, 'Run_SNAPB9X3_20261104')
    old_rows, new_rows = [], []
    for i, (cause, o, n) in enumerate(CASES):
        old_rows.append(det(i, served='1', **o))
        if n is not None:
            new_rows.append(det(i, **dict(n, identity_jname=o.get('identity_jname', det(i)['identity_jname']))))
    # a previous identity's other member: served before and again, its new group (src_index 31) holds a restored detection
    old_rows.append(det(50, served='1', identity_jname='J0999+2000'))
    new_rows.append(det(50, served='1', src_index='31'))
    # restored detections (installed pass removes, restore pass keeps): r1 shares assoc_cluster 77, r2 assoc_cluster 88, r3 src 31
    for j, grp in ((60, dict(assoc_cluster='77')), (61, dict(assoc_cluster='88')), (62, dict(src_index='31'))):
        new_rows.append(det(j, served='1', gate_3d='pass', gate_3d_installed='remove', gate_3d_restore='pass', restored_by=RB,
                            **grp))
    new_rows.append(det(63, served='0', gate_3d_installed='remove', gate_3d_restore='pass', restored_by=RB,
                        first_removed_by='neg-lobe'))                                    # restored, removed later: not served
    new_rows.append(det(70, served='1'))                                                 # gained, not restored
    # excluded: served before on a cell-band excluded now (the real list), not in the new record
    el, es, ed = exc_key()
    ecand = f'{el}_{ed}_{es}MHz_I_10s_{ed.replace("-", "")}_120000'
    erow = dict(det_uid=ecand + '_10_10', cand_id=ecand, cell=f'{el}_{ed}', subband=f'{es}MHz', band='Green', stokes='I', tier='10s',
                ts=ecand[-15:], snr='9.00', ra_peak='200.00000', dec_peak='40.00000', identity_jname='JE0001')
    old_rows.append(dict(erow, served='1'))
    if fail == 'served_excl':
        new_rows.append(dict(erow, det_uid=ecand + '_11_10', served='1'))              # served now on the excluded cell-band
    # reprocessed: served before from the table a recorded re-run replaced between the two records (46 MHz Stokes V)
    old_rows += [vdet('071000', 100, 200, served='1'), vdet('071100', 300, 300, served='1'), vdet('071200', 400, 400, served='1')]
    new_rows += [vdet('071000', 101, 200, served='1'),                                   # re-found served: the same stamp, 1 px
                 vdet('071110', 302, 301, served='0', first_removed_by='seed-veto'),     # re-found not served: 10 s later, 2 px
                 vdet('071200', 404, 400, served='1')]                                   # 4 px away: not re-found
    rr = os.path.join(root, 'remade', '07h_2025-10-29')
    for t, table in ((T0 + 3600, 'candidates_46MHz_V_median.csv'), (T0 - 3600, 'candidates_46MHz_I_median.csv')):
        _csv(os.path.join(rr, stamp(t), 'MANIFEST.csv'), ['original_path', 'quarantine_path', 'bytes', 'sha256'],
             [dict(original_path=f'/x/snapshot_fleet_out/07h_2025-10-29/{table}', quarantine_path=f'{rr}/{stamp(t)}/fleet_out/{table}',
                   bytes='1', sha256='0'),
              dict(original_path='/x/snapshot_fleet_out/07h_2025-10-29/frame_times_46MHz.csv', quarantine_path='q', bytes='1',
                   sha256='0')])
    if fail == 'nostamp':
        _csv(os.path.join(rr, 'rerun_a', 'MANIFEST.csv'), ['original_path'], [dict(original_path='/x/07h_2025-10-29/candidates_x.csv')])
    restored = 4
    if fail == 'absent':
        new_rows = [r for r in new_rows if r['det_uid'] != 'u001']
    if fail == 'summary':
        restored = 5
    if fail == 'pair':
        next(r for r in new_rows if r['det_uid'] == 'u061')['gate_3d_restore'] = 'remove'
    _gz(os.path.join(OLD, 'detection_record.csv.gz'), OLD_COLS, old_rows)
    _gz(os.path.join(NEW, 'detection_record.csv.gz'), NEW_COLS, new_rows)
    os.utime(os.path.join(OLD, 'detection_record.csv.gz'), (T0, T0))
    os.utime(os.path.join(NEW, 'detection_record.csv.gz'), (T0 + 7200, T0 + 7200))
    summ = dict(cells=['07h_2025-10-29'])
    if fail != 'not_restore':
        summ['gate3d_restore_only'] = dict(table='t.json', removed_installed_pass=10, removed_restore_pass=6, removed=6,
                                           restored=restored, restore_pass_only=0)
    json.dump(summ, open(os.path.join(NEW, 'chain_summary.json'), 'w'))
    json.dump(dict(cells=['07h_2025-10-29']), open(os.path.join(OLD, 'chain_summary.json'), 'w'))
    # served detections.csv: the chain's (candidates/Green/) with the column and the served tagged rows; the pages' (I)
    dcols = ['cand_id', 'ra', 'dec', 'ra_peak', 'dec_peak', 'snr', 'pos_err_stat_arcsec', 'pos_err_arcsec', 'restored_by']
    served_new = [r for r in new_rows if r['served'] == '1']
    chain = [dict(r, ra=r['ra_peak'], dec=r['dec_peak'], pos_err_stat_arcsec='30', pos_err_arcsec='60') for r in served_new]
    if fail == 'tagged':
        chain = [dict(r, restored_by='') if r['det_uid'] == 'u060' else r for r in chain]
    _csv(os.path.join(NEW, 'candidates', 'Green', 'rank0001_Green_J0000+2000', 'detections.csv'), dcols, chain)
    pcols = dcols if fail != 'nocol' else dcols[:-1]
    _csv(os.path.join(NEW, 'candidates', 'I', 'J0000+2000_46MHz', 'detections.csv'), pcols, chain)
    return OLD, NEW


WORLDS = (('pass', None), ('absent', 'absent'), ('summary', 'summary'), ('pair', 'pair'), ('nocol', 'nocol'),
          ('tagged', 'tagged'), ('not_restore', 'not_restore'), ('served_excl', 'served_excl'), ('nostamp', 'nostamp'))
READ_TIME = re.compile(rb'"exclusions_read": "[0-9TZ:-]+"')


def _prod_script(new):
    """the pinned production script with its REMADE_ROOTS line pointed at the world's re-run root (the one line changed)"""
    pin = _pin()
    src = open(os.path.join(pin, 'snapshot_restore_wave_diff.py')).read()
    line = 'REMADE_ROOTS = ["/lustre/pipeline/exopipe/phase3/Run_SNAP_FRAMETIME_QUARANTINE_20261007"]'
    assert src.count(line) == 1, 'the pinned script has no single REMADE_ROOTS line'
    p = os.path.join(pin, 'snapshot_restore_wave_diff_world.py')
    open(p, 'w').write(src.replace(line, 'REMADE_ROOTS = [%r]' % remade_root(new)))
    return p


def prod_diff(old, new):
    """production's pinned script (its interpreter) on OLD NEW: (exit, its csv bytes, its json bytes, its output); its files
    moved aside (None when it wrote none)."""
    r = subprocess.run([PY36, _prod_script(new), old, new], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       universal_newlines=True, timeout=600)
    RW = _RW()
    out = []
    for fn in (RW.CSV_NAME, RW.JSON_NAME):
        p = os.path.join(new, fn)
        out.append(open(p, 'rb').read() if os.path.exists(p) else None)
        if os.path.exists(p):
            os.remove(p)
    return r.returncode, out[0], out[1], r.stdout


def same_json(a, b):
    """production's json bytes = the port's, the time of the exclusion read aside (each has one)"""
    return len(READ_TIME.findall(a)) == len(READ_TIME.findall(b)) == 1 and READ_TIME.sub(b'', a) == READ_TIME.sub(b'', b)


def test_production_pin():
    """the ported file is the pinned one (snapshot_restore_wave_diff.py at tests/production_pin.PIN_COMMIT)"""
    d = PIN.drift(['snapshot_restore_wave_diff.py', 'snapshot_batch/comb_build_wave_20261104.sh'])
    assert not d, PIN.drift_message(d)


def test_parity():
    """production's pinned script and the port on every world: the same csv and json bytes, the same PASS / FAIL"""
    RW = _RW()
    seen = set()
    for name, fail in WORLDS:
        old, new = world(name, fail)
        rc, pcsv, pjson, pout = prod_diff(old, new)
        out = os.path.join(SCRATCH, name, 'port')
        j = wdiff(old, new, out=out)
        assert open(os.path.join(out, RW.CSV_NAME), 'rb').read() == pcsv, (name, pout[-1500:])
        assert same_json(open(os.path.join(out, RW.JSON_NAME), 'rb').read(), pjson), name
        assert (rc == 0) == (j['status'] == 'PASS') == (fail is None), (name, rc, j['status'])
        assert j['line'] in pout, (name, j['line'][:200], pout[-400:])
        jp = json.loads(pjson)
        if fail is None:
            causes = collections.Counter(r['cause'] for r in csv.DictReader(open(os.path.join(out, RW.CSV_NAME))))
            assert dict(causes) == {'card2': 1, '3d-gate': 1, 'secondary': 3, 'secondary-pool': 1, 'other-later': 1,
                                    'other-earlier': 1, 'not-served': 1, 'excluded': 1, 'reprocessed': 3}, causes
            c = jp['counts']
            assert c['served'] == 2 and c['restored'] == 4 and c['restored_served'] == 3 and c['gained'] == 6 and \
                c['gained_restored'] == 3 and c['served_on_excluded'] == 0, c
            assert (c['reprocessed_refound_served'], c['reprocessed_refound_not_served'], c['reprocessed_refound_shifted'],
                    c['reprocessed_not_refound']) == (1, 1, 1, 1), c
            assert jp['remade_manifests'] == 1 and jp['remade_tables'] == 1 and jp['exclusion_rows'] > 0, jp
            det_ = {r['det_uid']: r['removed_detail'] for r in csv.DictReader(open(os.path.join(out, RW.CSV_NAME)))}
            assert 'at a time stamp 10 s away' in det_['07h_2025-10-29_46MHz_V_10s_20251029_071100_300_300'] and \
                '(removed by seed-veto)' in det_['07h_2025-10-29_46MHz_V_10s_20251029_071100_300_300'], det_
            groups = {r['det_uid']: r['group_column'] for r in csv.DictReader(open(os.path.join(out, RW.CSV_NAME)))}
            assert groups['u003'] == 'assoc_cluster' and groups['u004'] == 'assoc_cluster' and \
                groups['u005'].startswith("previous identity members' "), groups
        else:
            seen.add(fail)
            assert jp['status'].startswith('FAIL'), (name, jp['status'])
    assert seen == {f for _n, f in WORLDS if f}
    # a FIRST VERSION (OLD none): nothing served before, every new served detection gained; PASS
    old, new = world('first')
    rc, pcsv, pjson, pout = prod_diff('none', new)
    out = os.path.join(SCRATCH, 'first', 'port')
    j = wdiff('none', new, out=out)
    assert rc == 0 and j['status'] == 'PASS' and open(os.path.join(out, RW.CSV_NAME), 'rb').read() == pcsv, (rc, j['status'],
                                                                                                            pout[-800:])
    assert same_json(open(os.path.join(out, RW.JSON_NAME), 'rb').read(), pjson) and j['line'] in pout, pout[-800:]
    assert j['counts']['prev_served'] == 0 and j['counts']['gained'] == 8 and j['tag_old'] == 'none', j['counts']
    # ... refused when its batch has another run in the review directory: production exits with REFUSING, writes nothing
    os.makedirs(os.path.join(os.path.dirname(new), 'Run_SNAPB9X2_20261030'))
    rc, pcsv, pjson, pout = prod_diff('none', new)
    assert rc != 0 and 'REFUSING' in pout and pcsv is None and pjson is None, (rc, pout[-400:])
    from exopipe.phase3.core.record import Refusal
    out2 = os.path.join(SCRATCH, 'first', 'port2')
    try:
        wdiff('none', new, out=out2)
        raise AssertionError('the port accepted a first version with another run of its batch')
    except Refusal as e:
        assert 'REFUSING' in str(e) and 'Run_SNAPB9X2_20261030' in str(e), e
    assert not os.path.exists(out2)
    # the totals over the worlds' production json files = the port's
    runs = []
    for name, fail in WORLDS[:2]:
        old, new = world(name + '_tot', fail)
        wdiff(old, new)
        runs.append(new)
    st, line = RW.totals(runs)
    pin = _pin()
    r = subprocess.run([PY36, os.path.join(pin, 'snapshot_restore_wave_diff.py'), '--totals'] + runs, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, universal_newlines=True, timeout=600)
    assert r.stdout.strip().splitlines()[-1] == line and (r.returncode == 0) == (st == 0) and st == 1, (r.stdout[-600:], line)


def test_content():
    """class restore_wave_diff: passes on the PASS world and a first version, fails on every FAIL world and on a tampered cause,
    count, status, re-run directory or served-on-excluded count"""
    from exopipe.phase3.lib.catalog import tests_snapshot as TS
    RW = _RW()
    for name, fail in WORLDS:
        old, new = world('ct_' + name, fail)
        wdiff(old, new)
        ok_, _v, detail = TS.restore_wave_rows(os.path.join(new, RW.JSON_NAME))
        assert ok_ == (fail is None), (name, detail[:400])
    old, new = world('ct_first')
    wdiff('none', new)
    ok_, _v, detail = TS.restore_wave_rows(os.path.join(new, RW.JSON_NAME))
    assert ok_, detail[:400]
    # a first version outside the review root: an earlier run of its batch IN the review root refuses it too (v2)
    from exopipe.phase3.config import CFG
    from exopipe.phase3.core.record import Refusal
    rr, saved = os.path.join(SCRATCH, 'review_root'), CFG.paths.SNAPSHOT_REVIEW_ROOT
    os.makedirs(os.path.join(rr, 'Run_SNAP%s9_20261101' % RW.tag_of(new)[:3]), exist_ok=True)
    CFG.paths.SNAPSHOT_REVIEW_ROOT = rr
    try:
        RW.diff('none', new, out=os.path.join(SCRATCH, 'ct_first_rr'))
        raise AssertionError('a first version with an earlier run of its batch in the review root was not refused')
    except Refusal as e:
        assert 'Run_SNAP%s9_20261101' % RW.tag_of(new)[:3] in str(e), str(e)
    finally:
        CFG.paths.SNAPSHOT_REVIEW_ROOT = saved
    old, new = world('ct_tamper')
    wdiff(old, new)
    jp, cp = os.path.join(new, RW.JSON_NAME), os.path.join(new, RW.CSV_NAME)
    good_j, good_c = open(jp).read(), open(cp).read()

    def rewrite(p, f):
        t = open(p).read()
        open(p, 'w').write(f(t))

    def fail_as_pass():
        world('ct_tamper', 'absent')
        wdiff(old, new)
        rewrite(jp, lambda t: t.replace(json.loads(t)['status'], 'PASS'))
    assert ',other-earlier,' in good_c and '"card2": 1' in good_j and ',excluded,' in good_c and stamp(T0 + 3600) in good_c
    assert '"served_on_excluded": 0' in good_j
    for label, edit in (('a cause', lambda: rewrite(cp, lambda t: t.replace(',other-earlier,', ',not-served,', 1))),
                        ('a count', lambda: rewrite(jp, lambda t: t.replace('"card2": 1', '"card2": 2'))),
                        ('a status', fail_as_pass),
                        ('an excluded detection called absent', lambda: rewrite(cp, lambda t: t.replace(',excluded,', ',absent,', 1))),
                        ('a re-run directory before the old record',
                         lambda: rewrite(cp, lambda t: t.replace(stamp(T0 + 3600), stamp(T0 - 3600), 1))),
                        ('a served-on-excluded count',
                         lambda: rewrite(jp, lambda t: t.replace('"served_on_excluded": 0', '"served_on_excluded": 1'))),
                        ('a re-found pixel distance', lambda: rewrite(cp, lambda t: t.replace('within 1 px', 'within 2 px', 1))),
                        ('a re-found time offset', lambda: rewrite(cp, lambda t: t.replace(' at a time stamp 10 s away', '', 1))),
                        ('an exclusion list of another size',
                         lambda: rewrite(jp, lambda t: t.replace('"exclusion_rows": %d' % json.loads(t)['exclusion_rows'],
                                                                 '"exclusion_rows": %d' % (json.loads(t)['exclusion_rows'] + 1))))):
        edit()
        ok_, _v, detail = TS.restore_wave_rows(jp)
        assert not ok_, (label, detail[:300])
        world('ct_tamper')
        wdiff(old, new)


def _cli(*args, timeout=900):
    e = dict(os.environ, PYTHONPATH=REPO)
    return subprocess.run([PY38, '-m', 'exopipe.phase3'] + list(args), cwd=REPO, env=e, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, universal_newlines=True, timeout=timeout)


def _rec(out, name, stage):
    p = os.path.join(out, '_records', 'catalog', f'{name}.{stage}.json')
    return json.load(open(p)) if os.path.exists(p) else None


def test_stage():
    """catalog --tier snapshot --stages wavediff through the real CLI (module docstring)"""
    RW = _RW()
    old, new = world('stage')
    name = os.path.basename(new)
    cells = os.path.join(SCRATCH, 'stage', 'cells.txt')
    open(cells, 'w').write('07h 2025-10-29\n')
    base = ['catalog', '--tier', 'snapshot', '--run', name, '--cells', '@' + cells, '--stages', 'wavediff', '--out', new,
            '--cores', 'none', '--set', f'catalog_snapshot.RESTORE_WAVE_REMADE_ROOTS=("{remade_root(new)}",)']
    # P-G22: no --snap-replaces, no schedule row
    r = _cli(*base, '--set', 'catalog_snapshot.REPLACES_SCHEDULES=()')
    j = _rec(new, name, 'wavediff')
    assert r.returncode == 2 and j and j['status'] == 'refused' and 'P-G22' in j['reason'], (r.returncode, j and j['reason'],
                                                                                             r.stderr[-800:])
    # the schedule names the replaced run (production's CSV layout); P-G23 first without pages
    sch = os.path.join(SCRATCH, 'stage', 'schedule.csv')
    _csv(sch, ['host', 'position', 'new', 'run', 'tag'], [dict(host='h', position='1', new=name, run=old, tag='B9X')])   # a path:
    # production's rows name runs under paths.SNAPSHOT_REVIEW_ROOT, an absolute path is taken as it is (os.path.join)
    pages = os.path.join(new, 'candidates', 'I')
    shutil.move(pages, pages + '.aside')
    r = _cli(*base, '--set', f'catalog_snapshot.REPLACES_SCHEDULES=("{sch}",)')
    j = _rec(new, name, 'wavediff')
    assert r.returncode == 2 and 'P-G23' in j['reason'] and not os.path.exists(os.path.join(new, RW.JSON_NAME)), j['reason']
    shutil.move(pages + '.aside', pages)
    # PASS through the schedule
    r = _cli(*base, '--set', f'catalog_snapshot.REPLACES_SCHEDULES=("{sch}",)')
    j = _rec(new, name, 'wavediff')
    assert r.returncode == 0 and j['status'] == 'done', (r.returncode, j['status'], j['reason'], r.stderr[-800:])
    assert j['meta']['wave']['status'] == 'PASS' and 'schedule' in j['meta']['wave']['how'], j['meta']['wave']
    t = {x['name']: x for x in j['tests']}
    assert t.get('restore_wave_diff.restore_wave_rows', {}).get('passed'), j['tests']
    assert json.load(open(os.path.join(new, RW.JSON_NAME)))['status'] == 'PASS'
    assert json.load(open(os.path.join(new, RW.JSON_NAME)))['counts']['reprocessed'] == 3   # through the config key's root
    # --snap-replaces overrides the schedule; itself refuses
    r = _cli(*base, '--snap-replaces', new)
    assert r.returncode == 2 and 'P-G22' in _rec(new, name, 'wavediff')['reason']
    # a FIRST VERSION: --snap-replaces none, and a schedule row whose run is none; refused with another run of its batch
    r = _cli(*base, '--snap-replaces', 'none')
    j = _rec(new, name, 'wavediff')
    assert r.returncode == 0 and j['status'] == 'done' and 'first version' in j['meta']['wave']['how'], \
        (r.returncode, j['status'], j['reason'], r.stderr[-800:])
    assert json.load(open(os.path.join(new, RW.JSON_NAME)))['tag_old'] == 'none'
    sch0 = os.path.join(SCRATCH, 'stage', 'schedule_none.csv')
    _csv(sch0, ['host', 'position', 'new', 'run', 'tag'], [dict(host='h', position='1', new=name, run='none', tag='B9X')])
    r = _cli(*base, '--set', f'catalog_snapshot.REPLACES_SCHEDULES=("{sch0}",)')
    j = _rec(new, name, 'wavediff')
    assert r.returncode == 0 and j['meta']['wave']['old'] == 'none' and 'schedule' in j['meta']['wave']['how'], \
        (r.returncode, j['reason'], r.stderr[-800:])
    other = os.path.join(os.path.dirname(new), 'Run_SNAPB9X2_20261030')
    os.makedirs(other)
    r = _cli(*base, '--snap-replaces', 'none')
    j = _rec(new, name, 'wavediff')
    assert r.returncode == 2 and j['status'] == 'refused' and 'REFUSING' in j['reason'] and os.path.basename(other) in j['reason'], \
        (r.returncode, j['status'], j['reason'])
    os.rmdir(other)
    # P-G24: a FAIL world - refused, its files written
    old_f, new_f = world('stage_fail', 'absent')
    name_f = os.path.basename(new_f)
    open(os.path.join(SCRATCH, 'stage_fail', 'cells.txt'), 'w').write('07h 2025-10-29\n')
    r = _cli('catalog', '--tier', 'snapshot', '--run', name_f, '--cells', '@' + os.path.join(SCRATCH, 'stage_fail', 'cells.txt'),
             '--stages', 'wavediff', '--out', new_f, '--cores', 'none', '--snap-replaces', old_f,
             '--set', f'catalog_snapshot.RESTORE_WAVE_REMADE_ROOTS=("{remade_root(new_f)}",)')
    jf = _rec(new_f, name_f, 'wavediff')
    assert r.returncode == 2 and 'P-G24' in jf['reason'] and os.path.exists(os.path.join(new_f, RW.JSON_NAME)), jf['reason']
    # a run without the restore-only gate: no input
    old_n, new_n = world('stage_norestore', 'not_restore')
    name_n = os.path.basename(new_n)
    open(os.path.join(SCRATCH, 'stage_norestore', 'cells.txt'), 'w').write('07h 2025-10-29\n')
    r = _cli('catalog', '--tier', 'snapshot', '--run', name_n, '--cells',
             '@' + os.path.join(SCRATCH, 'stage_norestore', 'cells.txt'), '--stages', 'wavediff', '--out', new_n, '--cores', 'none',
             '--snap-replaces', old_n)
    jn = _rec(new_n, name_n, 'wavediff')
    assert jn['status'] == 'no-input' and 'no restore-only 3-D gate' in jn['reason'], (jn['status'], jn['reason'])
    # the dry run names the replaced run and writes nothing
    r = _cli(*base, '--set', f'catalog_snapshot.REPLACES_SCHEDULES=("{sch}",)', '--dry-run')
    assert r.returncode == 0 and 'wavediff: the restore-only wave' in r.stdout and os.path.basename(old) in r.stdout, \
        (r.returncode, r.stdout[-800:], r.stderr[-400:])


class _Unit:
    def __init__(self):
        self.meta, self.notes, self.inputs = {}, [], []

    def refuse(self, why):
        raise RuntimeError('REFUSED ' + why)

    def add_input(self, p):
        self.inputs.append(p)

    def note(self, t):
        self.notes.append(t)


def test_view_gate():
    """G-V11 (snapshot_view.restore_wave_gate) and G27 (snapshot_copy.restore_only_go) on a view of a restore-only member"""
    import types
    from exopipe.phase3.config import CFG
    from exopipe.phase3.lib.serve import snapshot_copy, snapshot_view as SV
    RW = _RW()
    old, new = world('view')
    plain = os.path.join(os.path.dirname(old), 'Run_SNAPB9Y_20261025')
    os.makedirs(plain)
    json.dump(dict(cells=['07h_2025-10-30']), open(os.path.join(plain, 'chain_summary.json'), 'w'))
    assert SV.restore_wave_gate(_Unit(), [plain]) is None                   # no restore-only member: no gate
    for label, prep in (('no test', lambda: None),
                        ('a FAIL test', lambda: json.dump(dict(json.load(open(os.path.join(new, RW.JSON_NAME))),
                                                               status='FAIL: x'), open(os.path.join(new, RW.JSON_NAME), 'w'))),
                        ('another run\'s', lambda: json.dump(dict(json.load(open(os.path.join(new, RW.JSON_NAME))),
                                                                  tag_new='B9Q3'), open(os.path.join(new, RW.JSON_NAME), 'w'))),
                        ('older than the record', lambda: os.utime(os.path.join(new, RW.JSON_NAME), (1, 1)))):
        if label != 'no test':
            wdiff(old, new)
        elif os.path.exists(os.path.join(new, RW.JSON_NAME)):
            os.remove(os.path.join(new, RW.JSON_NAME))
        prep()
        try:
            SV.restore_wave_gate(_Unit(), [plain, new])
            raise AssertionError(f'G-V11 accepted {label}')
        except RuntimeError as e:
            assert 'G-V11' in str(e), (label, e)
    import contextlib
    import io

    def dry(runs):
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            SV.describe(types.SimpleNamespace(U=os.path.join(SCRATCH, 'view', 'dry_view'), parity=True, runs=runs,
                                              omit_dir=os.path.join(SCRATCH, 'view', 'dry_omit'), omit_given=False), ['omit'])
        return f.getvalue()
    assert 'G-V11 restore-only wave: WOULD REFUSE' in dry([plain, new])          # the last state: a test older than its record
    wdiff(old, new)
    u = _Unit()
    line = SV.restore_wave_gate(u, [plain, new])
    assert line and line.startswith('RESTORE-WAVE-TOTALS over 1 runs') and 'every run PASS' in line, line
    assert u.meta['restore_wave']['runs'] == [os.path.basename(new)], u.meta
    out = dry([plain, new])
    assert 'G-V11 restore-only wave: RESTORE-WAVE-TOTALS over 1 runs' in out and 'no member ran' in dry([plain]), out[-800:]
    # G27 on a view beside its members (production's layout: one run per tag of its batches)
    view = os.path.join(os.path.dirname(old), 'Run_SNAPCOMB_WAVE_20261105')
    os.makedirs(view)
    json.dump(dict(view=os.path.basename(view), batches=['B9X3', 'B9Y']), open(os.path.join(view, 'combined_view_summary.json'),
                                                                               'w'))
    os.rename(old, old + '.old')              # the replaced run is not beside the view any more (a tag glob would find two)
    go = os.path.join(SCRATCH, 'view', 'SNAPCOMB_RESTORE_ONLY_SERVE_APPROVED')
    plan = types.SimpleNamespace(source=view, U=None, run_name=os.path.basename(view))
    S = CFG.serve
    saved = S.SNAPSHOT_RESTORE_ONLY_APPROVED
    S.SNAPSHOT_RESTORE_ONLY_APPROVED = go
    try:
        open(go, 'w').write(f'{os.path.basename(view)} - the wave totals shown; APPROVED\n')
        why, detail, used, n = snapshot_copy.restore_only_go(plan)
        assert why == '' and n == 1 and 'every test PASS' in detail, (why, detail)
        os.remove(os.path.join(new, RW.JSON_NAME))
        why, detail, used, n = snapshot_copy.restore_only_go(plan)
        assert 'G27' in why and 'no restore_wave_served_diff.json' in why, why
    finally:
        S.SNAPSHOT_RESTORE_ONLY_APPROVED = saved


def test_real():
    """READ ONLY: the port on the first served diff made by the pinned production script (its json carries remade_manifests and the
    re-found split; older versions carry exclusions_read only) whose replaced run still exists, or that is a first version =
    production's own files, byte for byte (the read time aside)"""
    RW = _RW()
    R = '/data07/gh/phase3/snapshot_search/review'
    served = sorted(glob.glob(os.path.join(R, 'Run_SNAP*', RW.JSON_NAME)))
    if not served:
        raise Skip('no review disk with a wave run on this host (lwacalim07 only)')
    js = [(json.load(open(p)), os.path.dirname(p)) for p in served]
    pairs = [(j['old'], n, j) for j, n in js if 'remade_manifests' in j and 'reprocessed_refound_served' in j.get('counts', {}) and
             (j['old'] == 'none' or os.path.exists(os.path.join(j['old'], 'detection_record.csv.gz')))]
    if not pairs:
        raise Skip(f'none of the {len(served)} served diffs was made by the pinned script with its replaced run on disk')
    old, new, jp = pairs[0]
    out = os.path.join(SCRATCH, 'real')
    t0 = time.time()
    j = RW.diff(old, new, out=out)
    if j['exclusion_rows'] != jp['exclusion_rows']:
        raise Skip(f'the exclusion list changed since production made {os.path.basename(new)}\'s test ({jp["exclusion_rows"]} rows '
                   f'then, {j["exclusion_rows"]} now)')
    if j['remade_manifests'] != jp['remade_manifests']:
        raise Skip(f'the re-run manifests changed since production made {os.path.basename(new)}\'s test ({jp["remade_manifests"]} '
                   f'then, {j["remade_manifests"]} now)')
    assert filecmp.cmp(os.path.join(new, RW.CSV_NAME), os.path.join(out, RW.CSV_NAME), shallow=False), RW.CSV_NAME
    assert same_json(open(os.path.join(new, RW.JSON_NAME), 'rb').read(), open(os.path.join(out, RW.JSON_NAME), 'rb').read())
    print(f'    {os.path.basename(new)} against {os.path.basename(old)}: {j["status"]}, the csv and json identical to '
          f'production\'s ({time.time() - t0:.0f} s)')


def test_recorded_reruns():
    """v2's own re-runs: the quarantine an `exopipe3 archive supersede` run record names is read, wherever its --quar-root put it
    (catalog_snapshot.RESTORE_WAVE_RECORDS_ROOTS), not only the configured roots of production's re-runs"""
    RW = _RW()
    old, new = world('recorded')
    root = os.path.dirname(os.path.dirname(new))
    base = wdiff(old, new, out=os.path.join(root, 'o_base'))
    assert base['status'] == 'PASS' and base['remade_manifests'] == 1 and base['counts']['reprocessed'] == 3, base
    qroot, cell = os.path.join(root, 'quarantine_elsewhere'), '07h_2025-10-29'
    shutil.move(remade_root(new), qroot)                      # the same re-run, quarantined under another root
    recs = os.path.join(root, 'records')
    qdir = os.path.join(qroot, cell, stamp(T0 + 3600))
    os.makedirs(os.path.join(recs, '_records', 'archive'))
    with open(os.path.join(recs, '_records', 'archive', f'supersede_{cell}.json'), 'w') as f:
        json.dump(dict(unit=f'supersede_{cell}', status='done',
                       meta=dict(cell=cell, quar_root=qroot, shipped=dict(quarantine=qdir))), f)
    rec = RW.diff(old, new, out=os.path.join(root, 'o_rec'), remade_roots=[remade_root(new)], records_roots=[recs])
    assert rec['status'] == 'PASS' and rec['counts'] == base['counts'] and rec['remade_manifests'] == 1, rec
    assert all(r['cause'] != 'reprocessed' or qdir in r['removed_detail']
               for r in csv.DictReader(open(os.path.join(root, 'o_rec', RW.CSV_NAME)))), 'removed_detail names the quarantine'
    none = RW.diff(old, new, out=os.path.join(root, 'o_none'), remade_roots=[remade_root(new)], records_roots=[])
    assert none['status'].startswith('FAIL') and none['counts'].get('reprocessed', 0) == 0, none['status'][:300]
    # a record whose script stopped before its SHIPPED line (no shipped key): the cell's quarantine directories still count
    with open(os.path.join(recs, '_records', 'archive', f'supersede_{cell}.json'), 'w') as f:
        json.dump(dict(unit=f'supersede_{cell}', status='refused', meta=dict(cell=cell, quar_root=qroot)), f)
    part = RW.diff(old, new, out=os.path.join(root, 'o_part'), remade_roots=[], records_roots=[recs])
    assert part['status'] == 'PASS' and part['counts'] == base['counts'], part['status'][:300]
    # a record naming a quarantine without its MANIFEST.csv; an unreadable record
    with open(os.path.join(recs, '_records', 'archive', f'supersede_{cell}.json'), 'w') as f:
        json.dump(dict(meta=dict(cell=cell, quar_root=qroot, shipped=dict(quarantine=qdir + '_gone'))), f)
    with open(os.path.join(recs, '_records', 'archive', 'supersede_08h_2025-10-30.json'), 'w') as f:
        f.write('{not json')
    bad = RW.diff(old, new, out=os.path.join(root, 'o_bad'), remade_roots=[], records_roots=[recs])
    assert bad['status'].startswith('FAIL') and 'which has no MANIFEST.csv' in bad['status'] \
        and 'could not be read' in bad['status'], bad['status'][:500]
    # FAILs only for what can bear on the window: an unreadable record written before OLD's record, and a named quarantine
    # without its MANIFEST.csv stamped outside the window, are not FAILs
    arch = os.path.join(recs, '_records', 'archive')
    with open(os.path.join(arch, f'supersede_{cell}.json'), 'w') as f:
        json.dump(dict(meta=dict(cell=cell, quar_root=qroot, shipped=dict(quarantine=os.path.join(qroot, cell, stamp(T0 - 3600))))), f)
    os.utime(os.path.join(arch, 'supersede_08h_2025-10-30.json'), (T0 - 10, T0 - 10))
    old_rec = RW.diff(old, new, out=os.path.join(root, 'o_oldrec'), remade_roots=[], records_roots=[recs])
    assert old_rec['status'] == 'PASS' and old_rec['counts'] == base['counts'], old_rec['status'][:500]
    # a record of the wrong shape is a FAIL, not a crash
    for i, meta in enumerate(('x', dict(cell=cell, quar_root=qroot, shipped=['q']), dict(cell=cell, quar_root=[qroot]))):
        with open(os.path.join(arch, 'supersede_09h_2025-10-31.json'), 'w') as f:
            json.dump(dict(meta=meta), f)
        sh = RW.diff(old, new, out=os.path.join(root, f'o_shape{i}'), remade_roots=[], records_roots=[recs])
        assert sh['status'].startswith('FAIL') and 'supersede_09h_2025-10-31.json' in sh['status'] \
            and 'could not be read' in sh['status'], (meta, sh['status'][:500])
    os.remove(os.path.join(arch, 'supersede_09h_2025-10-31.json'))
    # a records root given as text is one root; production's roots list as it is (a root given twice counts its manifest twice)
    # and a record's manifest already among them is not added again
    one = RW.diff(old, new, out=os.path.join(root, 'o_text'), remade_roots=[], records_roots=recs)
    assert one['status'] == 'PASS' and one['remade_manifests'] == 1, one['status'][:300]
    dup = RW.diff(old, new, out=os.path.join(root, 'o_dup'), remade_roots=[qroot, qroot], records_roots=[recs])
    assert dup['status'] == 'PASS' and dup['remade_manifests'] == 2 and dup['counts'] == base['counts'], dup
    # config: () reads no records; None reads the archive step's records root
    from exopipe.phase3.config import CFG
    k = 'catalog_snapshot.RESTORE_WAVE_RECORDS_ROOTS'
    try:
        CFG.set(k, ())
        emp = RW.diff(old, new, out=os.path.join(root, 'o_empty'), remade_roots=[])
        assert emp['status'].startswith('FAIL') and emp['counts'].get('reprocessed', 0) == 0, emp['status'][:300]
        CFG.set(k, None)
        CFG.set('archive.RECORDS_ROOT', recs)
        dflt = RW.diff(old, new, out=os.path.join(root, 'o_default'), remade_roots=[])
        assert dflt['status'] == 'PASS' and dflt['counts'] == base['counts'], dflt['status'][:300]
    finally:
        CFG.reset()


TESTS = [test_production_pin, test_parity, test_content, test_recorded_reruns, test_stage, test_view_gate, test_real]


def main(argv):
    want = set(argv[1:])
    t0 = time.time()
    failed, skipped = [], []
    os.makedirs(SCRATCH, exist_ok=True)
    try:
        for t in TESTS:
            if want and t.__name__ not in want:
                continue
            t1 = time.time()
            try:
                t()
                print(f'PASS {t.__name__} ({time.time() - t1:.1f} s)')
            except Skip as e:
                skipped.append(t.__name__)
                print(f'SKIP {t.__name__}: {e}')
            except Exception:
                import traceback
                failed.append(t.__name__)
                print(f'FAIL {t.__name__}\n{traceback.format_exc()}')
    finally:
        if os.environ.get('EXOPIPE3_TEST_KEEP') != '1':
            shutil.rmtree(SCRATCH, ignore_errors=True)
    n = len([t for t in TESTS if not want or t.__name__ in want])
    print((f'{len(failed)} failed: {failed}; ' if failed else '') + f'{n - len(failed) - len(skipped)} of {n} passed'
          + (f', {len(skipped)} skipped: {skipped}' if skipped else '') + f' ({time.time() - t0:.1f} s)')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
