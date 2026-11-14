#!/usr/bin/env python3
"""STANDING TEST - the window-tie tables the 10-minute conformance read, recorded and checked (lib/catalog/conform_inputs.py;
port of production exopipe_ops 074f96c + 5d98279, refactor-chat ruling 2026-10-18 option (a) "record and check"; 2026-11-13).

Asserts:
  (i)    record: pertie/conform_inputs holds byte copies of pertie_detections.csv and pertie_frames.csv and a SHA256SUMS in
         sha256sum's format; a second record replaces the first (no stale file kept);
  (ii)   verify: '' on the record; a changed copy, a missing table, an empty SHA256SUMS each give the reason;
  (iii)  reads: a stage header ('<time> CATALOGS <arm> <stage> ...') drops that stage's earlier read lines, another stage's
         lines are kept, lines before any header (a CONFORM_LOG.txt) all count;
  (iv)   check: passes on a read of the copy with its sha256; refuses a read of pertie/ itself, a logged sha256 that is not
         the copy's, a log without a read line, and a copy that does not verify;
  (v)    on the cluster: production's bash ci_check (sublst_cumulative.sh, extracted verbatim) gives the same verdict and the
         same message as the port on every case of (iv);
  (vi)   the catalog chain: 'tablefilter' and 'record' run between 'buildv' and 'test', both heavy, tablefilter optional
         (no-input without the no-veto build); the content tests of the new classes conform_inputs, table_filter and
         detection_record pass on good content and fail on bad.

Scratch: a fresh directory under $EXOPIPE3_TEST_SCRATCH, else the system temp directory, removed at exit.

Run standalone:  python tests/test_conform_inputs.py
Under pytest:    pytest tests/test_conform_inputs.py
"""
import atexit
import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')

PROD_CUM = '/lustre/gh/main/exopipe_ops/sublst_cumulative.sh'
_D = {}


def _scratch():
    if 'dir' not in _D:
        base = os.environ.get('EXOPIPE3_TEST_SCRATCH') or tempfile.gettempdir()
        _D['dir'] = tempfile.mkdtemp(prefix='test_conform_inputs_', dir=base)
        atexit.register(shutil.rmtree, _D['dir'], True)
    return _D['dir']


def _pertie(name):
    """A fresh run dir <scratch>/<name>/ with pertie/pertie_detections.csv and pertie/pertie_frames.csv."""
    u = os.path.join(_scratch(), name)
    if os.path.lexists(u):
        shutil.rmtree(u)
    os.makedirs(os.path.join(u, 'pertie'))
    with open(os.path.join(u, 'pertie', 'pertie_detections.csv'), 'w') as f:
        f.write('cand_id,tie_status,sys_arcmin,sys_source\nV1,tied,0.5,T1\nV2,untied-few,0.9,fallback\n')
    with open(os.path.join(u, 'pertie', 'pertie_frames.csv'), 'w') as f:
        f.write('frame,scale\nw1,1.01\n')
    return u


def _log(path, lines):
    with open(path, 'w') as f:
        f.write(''.join(ln + '\n' for ln in lines))
    return path


def _cases():
    """{name: (pertie_dir, log, expect_ok)} for (iv)/(v)."""
    from exopipe.phase3.lib.catalog import conform_inputs as CI
    out = {}
    u = _pertie('good')
    ci, _ = CI.record(os.path.join(u, 'pertie'))
    h = CI.sums(ci)['pertie_detections.csv']
    copy = os.path.join(ci, 'pertie_detections.csv')
    hdr = '2026-11-13T12:00:00Z CATALOGS B conform 17h'
    out['good'] = (os.path.join(u, 'pertie'), _log(f'{u}/catalogs.log', [hdr, f'{CI.READ}{copy} sha256 {h}']), True)
    out['superseded-bad-read'] = (os.path.join(u, 'pertie'), _log(f'{u}/catalogs2.log', [
        hdr, f'{CI.READ}{u}/pertie/pertie_detections.csv sha256 {h}', hdr, f'{CI.READ}{copy} sha256 {h}']), True)
    out['read-of-pertie'] = (os.path.join(u, 'pertie'), _log(f'{u}/catalogs3.log', [
        hdr, f'{CI.READ}{u}/pertie/pertie_detections.csv sha256 {h}']), False)
    out['other-sha'] = (os.path.join(u, 'pertie'), _log(f'{u}/catalogs4.log', [hdr, f'{CI.READ}{copy} sha256 {"0" * 64}']), False)
    out['no-read'] = (os.path.join(u, 'pertie'), _log(f'{u}/catalogs5.log', [hdr, 'conformance check with the sub-LST tables']),
                      False)
    out['second-stage-bad'] = (os.path.join(u, 'pertie'), _log(f'{u}/catalogs6.log', [
        hdr, f'{CI.READ}{copy} sha256 {h}', '2026-11-13T12:30:00Z CATALOGS B tablefilter 17h',
        f'{CI.READ}{copy} sha256 {"1" * 64}']), False)
    v = _pertie('changed')
    ci2, _ = CI.record(os.path.join(v, 'pertie'))
    h2 = CI.sums(ci2)['pertie_detections.csv']
    with open(os.path.join(ci2, 'pertie_detections.csv'), 'a') as f:
        f.write('V3,tied,0.4,T1\n')
    out['copy-changed'] = (os.path.join(v, 'pertie'), _log(f'{v}/catalogs.log', [
        hdr, f'{CI.READ}{ci2}/pertie_detections.csv sha256 {h2}']), False)
    w = _pertie('norecord')
    out['no-copy'] = (os.path.join(w, 'pertie'), _log(f'{w}/catalogs.log', [hdr, f'{CI.READ}{copy} sha256 {h}']), False)
    return out


def test_record():
    from exopipe.phase3.lib.catalog import conform_inputs as CI
    u = _pertie('rec')
    pt = os.path.join(u, 'pertie')
    ci, text = CI.record(pt)
    for n in CI.FILES:
        assert open(os.path.join(ci, n), 'rb').read() == open(os.path.join(pt, n), 'rb').read(), n
    lines = open(os.path.join(ci, CI.SUMS)).read().splitlines()
    assert lines == [f'{CI.sha256_file(os.path.join(pt, n))}  {n}' for n in CI.FILES], lines
    assert text == open(os.path.join(ci, CI.SUMS)).read()
    open(os.path.join(ci, 'stale.txt'), 'w').write('x')
    CI.record(pt)
    assert sorted(os.listdir(ci)) == sorted(CI.FILES + (CI.SUMS,)), os.listdir(ci)
    if shutil.which('sha256sum'):
        r = subprocess.run(['sha256sum', '-c', '--quiet', '--status', CI.SUMS], cwd=ci)
        assert r.returncode == 0, 'sha256sum -c refuses the SHA256SUMS the port wrote'


def test_verify():
    from exopipe.phase3.lib.catalog import conform_inputs as CI
    u = _pertie('ver')
    ci, _ = CI.record(os.path.join(u, 'pertie'))
    assert CI.verify(ci) == ''
    with open(os.path.join(ci, 'pertie_frames.csv'), 'a') as f:
        f.write('w2,0.99\n')
    assert 'does not verify' in CI.verify(ci), CI.verify(ci)
    CI.record(os.path.join(u, 'pertie'))
    os.remove(os.path.join(ci, 'pertie_frames.csv'))
    assert CI.verify(ci).startswith('no '), CI.verify(ci)
    assert 'does not verify' in CI.verify(ci, complete=False)
    CI.record(os.path.join(u, 'pertie'))
    open(os.path.join(ci, CI.SUMS), 'w').close()
    assert CI.verify(ci).startswith('no '), CI.verify(ci)


def test_reads():
    from exopipe.phase3.lib.catalog import conform_inputs as CI
    d = _scratch()
    r = CI.READ
    p = _log(os.path.join(d, 'reads.log'), [
        f'{r}a sha256 0', '2026-11-13T12:00:00Z CATALOGS B conform 17h', f'{r}b sha256 1', 'other line',
        '2026-11-13T12:10:00Z CATALOGS B tablefilter 17h', f'{r}c sha256 2', f'{r}d sha256 3',
        '2026-11-13T12:20:00Z CATALOGS B conform 17h', f'{r}e sha256 4'])
    got = [ln.split()[4] for ln in CI.reads(p)]
    assert got == ['a', 'e', 'c', 'd'], got        # the pre-header line, conform's current run, tablefilter's
    assert CI.reads(os.path.join(d, 'missing.log')) == []


def test_check():
    from exopipe.phase3.lib.catalog import conform_inputs as CI
    for name, (pt, log, want) in _cases().items():
        ok, msg = CI.check(pt, log)
        assert ok == want, f'{name}: {ok} {msg}'


def test_bash_parity():
    if not os.path.exists(PROD_CUM):
        print('  (production sublst_cumulative.sh not here: bash parity UNTESTED)')
        return
    from exopipe.phase3.lib.catalog import conform_inputs as CI
    src = open(PROD_CUM).read().splitlines()
    i = next(k for k, ln in enumerate(src) if ln.startswith('ci_reads(){'))
    j = next(k for k, ln in enumerate(src) if ln.startswith('  echo "conformance inputs verified'))
    fn = '\n'.join(src[i:j + 1])
    for name, (pt, log, want) in _cases().items():
        r = subprocess.run(['bash', '-c', fn + '\nci_check "$1" "$2"', 'x', pt, log], stdout=subprocess.PIPE,
                           universal_newlines=True)
        ok, msg = CI.check(pt, log)
        assert (r.returncode == 0, r.stdout.strip()) == (ok, msg), f'{name}: bash {r.returncode} {r.stdout!r} port {ok} {msg!r}'


def test_chain_stages():
    from exopipe.phase3.lib.catalog import tier_sublst as T
    i = T.STAGES.index
    assert i('buildv') + 1 == i('tablefilter') and i('tablefilter') + 1 == i('record') and i('record') + 1 == i('test'), T.STAGES
    assert 'tablefilter' in T.HEAVY and 'record' in T.HEAVY
    assert 'tablefilter' in T.OPTIONAL and 'record' not in T.OPTIONAL
    assert T.FUNCS['tablefilter'] is T.st_tablefilter and T.FUNCS['record'] is T.st_record


def test_content_classes():
    from exopipe.phase3.core import validate
    from exopipe.phase3.lib.catalog import conform_inputs as CI
    u = _pertie('cls')
    ci, _ = CI.record(os.path.join(u, 'pertie'))
    res = validate.run('conform_inputs', ci)
    assert res and all(r['passed'] for r in res), res
    with open(os.path.join(ci, 'pertie_detections.csv'), 'a') as f:
        f.write('V9,tied,0.1,T1\n')
    assert not all(r['passed'] for r in validate.run('conform_inputs', ci))
    with open(os.path.join(u, 'master_catalog.csv'), 'w') as f:
        f.write('cand_id,band\nV1,41\nV2,55\n')
    with open(os.path.join(u, 'table_filter.csv'), 'w') as f:
        f.write('# comment\ncand_id,complete_table,today_table\nV1,keep,keep\nV2,remove,keep\n')
    assert all(r['passed'] for r in validate.run('table_filter', os.path.join(u, 'table_filter.csv')))
    with open(os.path.join(u, 'table_filter.csv'), 'w') as f:
        f.write('cand_id,complete_table,today_table\nV2,keep,keep\nV1,n/a,keep\n')
    assert not any(r['passed'] for r in validate.run('table_filter', os.path.join(u, 'table_filter.csv')))
    sj = os.path.join(u, 'detection_record_10min_summary.json')
    json.dump({'rows': 2, 'status': 'PASS'}, open(sj, 'w'))
    with gzip.open(os.path.join(u, 'detection_record_10min.csv.gz'), 'wt') as f:
        f.write('cand_id\nV1\nV2\n')
    assert all(r['passed'] for r in validate.run('detection_record', sj))
    json.dump({'rows': 2, 'status': 'FAIL'}, open(sj, 'w'))
    assert not any(r['passed'] for r in validate.run('detection_record', sj))


TESTS = [test_record, test_verify, test_reads, test_check, test_bash_parity, test_chain_stages, test_content_classes]


def main():
    failed = 0
    for t in TESTS:
        try:
            t()
            print(f'PASS  {t.__name__}')
        except AssertionError as e:
            failed += 1
            print(f'FAIL  {t.__name__}: {e}')
    print(f'{len(TESTS) - failed} of {len(TESTS)} tests passed')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
