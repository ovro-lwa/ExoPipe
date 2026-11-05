#!/usr/bin/env python3
"""STANDING TEST - the restore-only wave's per-batch test (lib/catalog/restore_wave.py, port of exopipe_ops
snapshot_restore_wave_diff.py 8ff3fc8; the V3 round 2026-11-05), the run a new run REPLACES (catalog --tier snapshot --stages
wavediff), the view's G-V11 and the snapshot serve's G27 on such a view.

  pin        the ported production file is the pinned one (tests/production_pin.py)
  parity     synthetic OLD / NEW batch runs holding every class (served, card2, secondary by the removing stage's group, by a
             fallback grouping and by the previous identity's members, secondary-pool, 3d-gate, other-later, other-earlier,
             not-served) and every FAIL (an absent detection, the restored count off the summary, a tag off its pair, a page
             detections.csv without the column, the chain's tagged rows off the record, no gate3d_restore_only): PRODUCTION's pinned
             script (under its interpreter, writing into NEW) and the port (writing into scratch, the same OLD / NEW paths) give the
             same csv and json byte for byte and the same exit (PASS / FAIL); the totals line likewise
  content    class restore_wave_diff reads every world's files back against the two records (passes on the PASS worlds, fails on
             the FAIL ones and on a tampered cause, count or status)
  stage      the chain step's wavediff stage through the real CLI: the replaced run from --snap-replaces or the wave schedule
             (catalog_snapshot.REPLACES_SCHEDULES), NO-INPUT for a run without the restore-only gate, P-G22 (no replaced run,
             itself, no record), P-G23 (no pages), P-G24 (FAIL: refused, its files written), the PASS: done, class restore_wave_diff
  view       G-V11 (lib/serve/snapshot_view.restore_wave_gate): a restore-only member without its test, with a FAIL one, one made
             for another run or older than its record refuse; with the stage's PASS test the gate passes and returns the totals
  serve      G27 (lib/serve/snapshot_copy.restore_only_go) accepts a view whose restore-only member carries the stage's PASS test
             (with Gregg's go naming the view) and refuses it without the test
  real       READ ONLY, where the review disk is mounted: the port on Run_SNAPB6L_20261025 -> Run_SNAPB6L3_20261104 = production's
             own files in the new run, byte for byte (skipped elsewhere)
Scratch: /tmp/exopipe3_tests_restore_wave/ (removed at the end). Production trees are only read.
Run standalone:  python tests/test_restore_wave.py [test_name ...]
"""
import collections
import csv
import filecmp
import gzip
import json
import os
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
    restored = 4
    if fail == 'absent':
        new_rows = [r for r in new_rows if r['det_uid'] != 'u001']
    if fail == 'summary':
        restored = 5
    if fail == 'pair':
        next(r for r in new_rows if r['det_uid'] == 'u061')['gate_3d_restore'] = 'remove'
    _gz(os.path.join(OLD, 'detection_record.csv.gz'), OLD_COLS, old_rows)
    _gz(os.path.join(NEW, 'detection_record.csv.gz'), NEW_COLS, new_rows)
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
          ('tagged', 'tagged'), ('not_restore', 'not_restore'))


def prod_diff(old, new):
    """production's pinned script (its interpreter) on OLD NEW: (exit, its csv bytes, its json bytes); its files moved aside."""
    pin = _pin()
    r = subprocess.run([PY36, os.path.join(pin, 'snapshot_restore_wave_diff.py'), old, new], stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, universal_newlines=True, timeout=600)
    RW = _RW()
    out = []
    for fn in (RW.CSV_NAME, RW.JSON_NAME):
        p = os.path.join(new, fn)
        out.append(open(p, 'rb').read())
        os.remove(p)
    return r.returncode, out[0], out[1], r.stdout


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
        j = RW.diff(old, new, out=out)
        assert open(os.path.join(out, RW.CSV_NAME), 'rb').read() == pcsv, (name, pout[-1500:])
        assert open(os.path.join(out, RW.JSON_NAME), 'rb').read() == pjson, name
        assert (rc == 0) == (j['status'] == 'PASS') == (fail is None), (name, rc, j['status'])
        assert j['line'] in pout, (name, j['line'][:200], pout[-400:])
        jp = json.loads(pjson)
        if fail is None:
            causes = collections.Counter(r['cause'] for r in csv.DictReader(open(os.path.join(out, RW.CSV_NAME))))
            assert dict(causes) == {'card2': 1, '3d-gate': 1, 'secondary': 3, 'secondary-pool': 1, 'other-later': 1,
                                    'other-earlier': 1, 'not-served': 1}, causes
            assert jp['counts']['served'] == 2 and jp['counts']['restored'] == 4 and jp['counts']['restored_served'] == 3 and \
                jp['counts']['gained'] == 4 and jp['counts']['gained_restored'] == 3, jp['counts']
            groups = {r['det_uid']: r['group_column'] for r in csv.DictReader(open(os.path.join(out, RW.CSV_NAME)))}
            assert groups['u003'] == 'assoc_cluster' and groups['u004'] == 'assoc_cluster' and \
                groups['u005'].startswith("previous identity members' "), groups
        else:
            seen.add(fail)
            assert jp['status'].startswith('FAIL'), (name, jp['status'])
    assert seen == {f for _n, f in WORLDS if f}
    # the totals over the worlds' production json files = the port's
    runs = []
    for name, fail in WORLDS[:2]:
        old, new = world(name + '_tot', fail)
        RW.diff(old, new)
        runs.append(new)
    st, line = RW.totals(runs)
    pin = _pin()
    r = subprocess.run([PY36, os.path.join(pin, 'snapshot_restore_wave_diff.py'), '--totals'] + runs, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, universal_newlines=True, timeout=600)
    assert r.stdout.strip().splitlines()[-1] == line and (r.returncode == 0) == (st == 0) and st == 1, (r.stdout[-600:], line)


def test_content():
    """class restore_wave_diff: passes on the PASS world, fails on every FAIL world and on a tampered cause, count or status"""
    from exopipe.phase3.lib.catalog import tests_snapshot as TS
    RW = _RW()
    for name, fail in WORLDS:
        old, new = world('ct_' + name, fail)
        RW.diff(old, new)
        ok_, _v, detail = TS.restore_wave_rows(os.path.join(new, RW.JSON_NAME))
        assert ok_ == (fail is None), (name, detail[:400])
    old, new = world('ct_tamper')
    RW.diff(old, new)
    jp, cp = os.path.join(new, RW.JSON_NAME), os.path.join(new, RW.CSV_NAME)
    good_j, good_c = open(jp).read(), open(cp).read()

    def rewrite(p, f):
        t = open(p).read()
        open(p, 'w').write(f(t))

    def fail_as_pass():
        world('ct_tamper', 'absent')
        RW.diff(old, new)
        rewrite(jp, lambda t: t.replace(json.loads(t)['status'], 'PASS'))
    assert ',other-earlier,' in good_c and '"card2": 1' in good_j
    for label, edit in (('a cause', lambda: rewrite(cp, lambda t: t.replace(',other-earlier,', ',not-served,', 1))),
                        ('a count', lambda: rewrite(jp, lambda t: t.replace('"card2": 1', '"card2": 2'))),
                        ('a status', fail_as_pass)):
        edit()
        ok_, _v, detail = TS.restore_wave_rows(jp)
        assert not ok_, (label, detail[:300])
        world('ct_tamper')
        RW.diff(old, new)


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
            '--cores', 'none']
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
    # --snap-replaces overrides the schedule; itself refuses
    r = _cli(*base, '--snap-replaces', new)
    assert r.returncode == 2 and 'P-G22' in _rec(new, name, 'wavediff')['reason']
    # P-G24: a FAIL world - refused, its files written
    old_f, new_f = world('stage_fail', 'absent')
    name_f = os.path.basename(new_f)
    open(os.path.join(SCRATCH, 'stage_fail', 'cells.txt'), 'w').write('07h 2025-10-29\n')
    r = _cli('catalog', '--tier', 'snapshot', '--run', name_f, '--cells', '@' + os.path.join(SCRATCH, 'stage_fail', 'cells.txt'),
             '--stages', 'wavediff', '--out', new_f, '--cores', 'none', '--snap-replaces', old_f)
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
            RW.diff(old, new)
        elif os.path.exists(os.path.join(new, RW.JSON_NAME)):
            os.remove(os.path.join(new, RW.JSON_NAME))
        prep()
        try:
            SV.restore_wave_gate(_Unit(), [plain, new])
            raise AssertionError(f'G-V11 accepted {label}')
        except RuntimeError as e:
            assert 'G-V11' in str(e), (label, e)
    RW.diff(old, new)
    u = _Unit()
    line = SV.restore_wave_gate(u, [plain, new])
    assert line and line.startswith('RESTORE-WAVE-TOTALS over 1 runs') and 'every run PASS' in line, line
    assert u.meta['restore_wave']['runs'] == [os.path.basename(new)], u.meta
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
    """READ ONLY: the port on the first wave run = production's own files in it (B6L3 against B6L), byte for byte"""
    RW = _RW()
    R = '/data07/gh/phase3/snapshot_search/review'
    old, new = os.path.join(R, 'Run_SNAPB6L_20261025'), os.path.join(R, 'Run_SNAPB6L3_20261104')
    if not os.path.exists(os.path.join(new, RW.JSON_NAME)):
        raise Skip('no review disk with the wave run on this host (lwacalim07 only)')
    out = os.path.join(SCRATCH, 'real')
    t0 = time.time()
    j = RW.diff(old, new, out=out)
    for fn in (RW.CSV_NAME, RW.JSON_NAME):
        assert filecmp.cmp(os.path.join(new, fn), os.path.join(out, fn), shallow=False), fn
    print(f'    B6L3 against B6L: {j["status"]}, the csv and json identical to production\'s ({time.time() - t0:.0f} s)')


TESTS = [test_production_pin, test_parity, test_content, test_stage, test_view_gate, test_real]


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
